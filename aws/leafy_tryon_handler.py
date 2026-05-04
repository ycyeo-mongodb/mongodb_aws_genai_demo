"""
Lambda: leafy-tryon-handler — Virtual Try-On with Claude Vision + Nova Canvas image gen.
Persists sessions to MongoDB Atlas with style embeddings for recommendations.
Deployed behind API Gateway route POST /leafy_tryon.

Actions: analyze, generate, save, history, recommend
Required env vars: MONGODB_URI, VOYAGE_API_KEY
Required IAM: bedrock:InvokeModel
"""

import json
import os
from datetime import datetime, timezone

import boto3
import voyageai
from bson import ObjectId
from pymongo import MongoClient

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
CLAUDE_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA_MODEL = "amazon.nova-canvas-v1:0"

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}

_mongo = None
_products = None
_tryon = None
_vo = None


def get_collections():
    global _mongo, _products, _tryon
    if _tryon is None:
        _mongo = MongoClient(os.environ["MONGODB_URI"])
        db = _mongo["workshop"]
        _products = db["products"]
        _tryon = db["tryon_sessions"]
    return _products, _tryon


def get_vo():
    global _vo
    if _vo is None:
        _vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _vo


def get_embedding(text: str) -> list:
    vo = get_vo()
    result = vo.embed([text], model="voyage-4-large", input_type="document")
    return result.embeddings[0]


# ── Action: analyze ──

def do_analyze(image_base64: str, product: dict) -> dict:
    system_prompt = """You are a fashion style analyst. Given a user's photo and a product, provide:
1. compatibility_score: integer 1-10
2. body_type_analysis: brief body type observation
3. color_matching: how the product colors work with the user
4. fit_recommendation: sizing/fit advice
5. styling_tips: 2-3 tips to style this product

Return ONLY valid JSON with these exact keys."""

    product_desc = f"{product.get('name', '')} — {product.get('description', '')} (Category: {product.get('category', '')})"

    resp = bedrock.invoke_model(
        modelId=CLAUDE_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64},
                        },
                        {
                            "type": "text",
                            "text": f"Analyze style compatibility for this product:\n{product_desc}",
                        },
                    ],
                }
            ],
        }),
    )
    result = json.loads(resp["body"].read())
    text = result["content"][0]["text"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"raw_analysis": text, "compatibility_score": 5}


# ── Action: generate ──

def do_generate(prompt: str, style_context: str = "") -> dict:
    full_prompt = prompt
    if style_context:
        full_prompt += f". Style notes: {style_context}"

    try:
        resp = bedrock.invoke_model(
            modelId=NOVA_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": full_prompt},
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 768,
                    "width": 768,
                    "cfgScale": 8.0,
                },
            }),
        )
        result = json.loads(resp["body"].read())
        image_b64 = result.get("images", [None])[0]
        if image_b64:
            return {"image_base64": image_b64}
        return {"error": "No image generated"}
    except Exception as e:
        return {"error": f"Image generation failed: {str(e)}"}


# ── Action: save ──

def do_save(product_id, analysis: dict, generated_image_b64: str, user_preferences: list) -> dict:
    _, tryon_coll = get_collections()

    analysis_text = " ".join(str(v) for v in analysis.values() if isinstance(v, str))
    style_embedding = get_embedding(analysis_text) if analysis_text else []

    doc = {
        "product_id": product_id,
        "analysis": analysis,
        "generated_image_base64": generated_image_b64[:500] if generated_image_b64 else "",
        "style_embedding": style_embedding,
        "user_preferences": user_preferences,
        "created_at": datetime.now(timezone.utc),
    }
    result = tryon_coll.insert_one(doc)
    return {"session_id": str(result.inserted_id)}


# ── Action: history ──

def do_history(limit: int = 10) -> dict:
    _, tryon_coll = get_collections()
    cursor = tryon_coll.find({}, {"style_embedding": 0, "generated_image_base64": 0}).sort("_id", -1).limit(limit)
    sessions = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        sessions.append(doc)
    return {"sessions": sessions}


# ── Action: recommend ──

def do_recommend(session_id: str, limit: int = 5) -> dict:
    products_coll, tryon_coll = get_collections()
    session = tryon_coll.find_one({"_id": ObjectId(session_id)})
    if not session or not session.get("style_embedding"):
        return {"recommendations": [], "error": "Session not found or no embedding"}

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "description_embedding",
                "queryVector": session["style_embedding"],
                "numCandidates": 50,
                "limit": limit,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"description_embedding": 0}},
    ]
    results = []
    for doc in products_coll.aggregate(pipeline):
        doc.pop("_id", None)
        results.append(doc)
    return {"recommendations": results}


def lambda_handler(event, context):
    try:
        if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"message": "OK"})}

        body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event.get("body", event)
        action = body.get("action", "")

        if action == "analyze":
            image_b64 = body.get("image_base64", "")
            product = body.get("product", {})
            if not image_b64:
                return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": "image_base64 required"})}
            result = do_analyze(image_b64, product)
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result)}

        elif action == "generate":
            prompt = body.get("prompt", "")
            style_context = body.get("style_context", "")
            if not prompt:
                return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": "prompt required"})}
            result = do_generate(prompt, style_context)
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result)}

        elif action == "save":
            result = do_save(
                body.get("product_id", ""),
                body.get("analysis", {}),
                body.get("generated_image_base64", ""),
                body.get("user_preferences", []),
            )
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result)}

        elif action == "history":
            result = do_history(body.get("limit", 10))
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result, default=str)}

        elif action == "recommend":
            session_id = body.get("session_id", "")
            if not session_id:
                return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": "session_id required"})}
            result = do_recommend(session_id, body.get("limit", 5))
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result, default=str)}

        elif action == "health":
            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps({"service": "leafy-tryon-handler", "status": "ok"}),
            }

        return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": f"Unknown action: {action}"})}

    except Exception as e:
        return {"statusCode": 500, "headers": CORS_HEADERS, "body": json.dumps({"error": str(e)})}
