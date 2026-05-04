"""
Lambda: leafy-agent-handler — AI Shopping Agent with tool-use loop.
Connects directly to MongoDB Atlas for product queries and uses Voyage AI for embeddings.
Deployed behind API Gateway route POST /leafy_agent.

Required env vars: MONGODB_URI, VOYAGE_API_KEY
Required IAM: bedrock:InvokeModel
"""

import json
import os
import boto3
import voyageai
from pymongo import MongoClient

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}

_mongo = None
_coll = None
_vo = None


def get_collection():
    global _mongo, _coll
    if _coll is None:
        _mongo = MongoClient(os.environ["MONGODB_URI"])
        _coll = _mongo["workshop"]["products"]
    return _coll


def get_vo():
    global _vo
    if _vo is None:
        _vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _vo


def get_query_embedding(text: str) -> list:
    vo = get_vo()
    result = vo.embed([text], model="voyage-4-large", input_type="query")
    return result.embeddings[0]


# ── Tool implementations ──

def _normalize_doc(doc):
    """Ensure every product has a string 'product_id' the agent can reference."""
    if doc is None:
        return None
    if "id" in doc:
        doc["product_id"] = str(doc["id"])
    elif "_id" in doc:
        doc["product_id"] = str(doc["_id"])
    doc.pop("_id", None)
    return doc


def _find_by_product_id(product_id):
    """Look up a product by numeric id or ObjectId string."""
    from bson import ObjectId
    coll = get_collection()
    try:
        doc = coll.find_one({"id": int(product_id)}, {"description_embedding": 0})
    except (ValueError, TypeError):
        doc = None
    if not doc:
        try:
            doc = coll.find_one({"_id": ObjectId(product_id)}, {"description_embedding": 0})
        except Exception:
            doc = None
    return _normalize_doc(doc)


def tool_search_products(query, category=None, max_price=None, limit=10):
    coll = get_collection()
    query_vector = get_query_embedding(query)
    vs_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "description_embedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": limit,
        }
    }
    if category:
        vs_stage["$vectorSearch"]["filter"] = {"category": category}

    pipeline = [
        vs_stage,
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"description_embedding": 0}},
    ]
    results = []
    for doc in coll.aggregate(pipeline):
        doc = _normalize_doc(doc)
        if max_price and doc.get("price", 0) > max_price:
            continue
        results.append(doc)
    return results[:limit]


def tool_get_product_details(product_id):
    doc = _find_by_product_id(product_id)
    if doc:
        return doc
    return {"error": f"Product {product_id} not found"}


def tool_compare_products(product_ids):
    results = []
    for pid in product_ids:
        doc = _find_by_product_id(pid)
        if doc:
            results.append(doc)
    return results


def tool_add_to_cart(product_id):
    doc = _find_by_product_id(product_id)
    if doc:
        return {"action": "add_to_cart", "product": doc}
    return {"error": f"Product {product_id} not found"}


def tool_get_cart_contents(cart):
    return cart or []


TOOLS_SPEC = [
    {
        "name": "search_products",
        "description": "Search the product catalog using semantic vector search. Returns matching products with names, prices, categories, and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "category": {"type": "string", "description": "Optional category filter (Shoes, Clothing, Electronics, Home & Kitchen, Sports & Outdoors, Beauty & Personal Care, Books & Stationery, Bags & Accessories)"},
                "max_price": {"type": "number", "description": "Optional maximum price filter"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_product_details",
        "description": "Get full details of a specific product by its product_id (numeric or string ObjectId).",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product_id from search results"},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "compare_products",
        "description": "Compare multiple products side by side. Pass an array of product_id values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_ids": {"type": "array", "items": {"type": "string"}, "description": "List of product_id values to compare"},
            },
            "required": ["product_ids"],
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add a product to the user's shopping cart by product_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product_id to add to cart"},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "get_cart_contents",
        "description": "View the current contents of the user's shopping cart.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """You are LeafyShop's AI shopping assistant. You help customers find products, compare items, and manage their cart.

You have access to a product catalog with items across 8 categories. Use the tools to search, look up details, compare, and add items to the cart.

Guidelines:
- Always search before recommending. Don't make up products.
- When users ask vague questions, use search_products with a descriptive query.
- If users want the "best", "cheapest", or "most expensive" item, search and then sort the results.
- Every product in search results has a "product_id" field. Use this value when calling add_to_cart, get_product_details, or compare_products.
- When adding to cart, confirm the product name and price.
- Be concise and helpful. Format responses with bullet points when listing products."""


def execute_tool(name, inp, cart):
    if name == "search_products":
        return tool_search_products(inp.get("query", ""), inp.get("category"), inp.get("max_price"))
    elif name == "get_product_details":
        return tool_get_product_details(inp["product_id"])
    elif name == "compare_products":
        return tool_compare_products(inp["product_ids"])
    elif name == "add_to_cart":
        return tool_add_to_cart(inp["product_id"])
    elif name == "get_cart_contents":
        return tool_get_cart_contents(cart)
    return {"error": f"Unknown tool: {name}"}


def run_agent(message, history, cart, max_iterations=5):
    messages = []
    for h in (history or []):
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    tool_calls_log = []
    cart_actions = []

    for _ in range(max_iterations):
        resp = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "temperature": 0.3,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "tools": TOOLS_SPEC,
            }),
        )
        result = json.loads(resp["body"].read())
        content = result.get("content", [])
        stop_reason = result.get("stop_reason", "end_turn")

        messages.append({"role": "assistant", "content": content})

        if stop_reason != "tool_use":
            reply_parts = [b["text"] for b in content if b.get("type") == "text"]
            return {
                "reply": "\n".join(reply_parts),
                "tool_calls": tool_calls_log,
                "cart_actions": cart_actions,
            }

        tool_results = []
        for block in content:
            if block.get("type") != "tool_use":
                continue
            tool_name = block["name"]
            tool_input = block.get("input", {})
            tool_id = block["id"]

            result_data = execute_tool(tool_name, tool_input, cart)

            summary = tool_name
            if tool_name == "search_products":
                summary = f"Searched for '{tool_input.get('query', '')}' → {len(result_data) if isinstance(result_data, list) else 0} results"
            elif tool_name == "get_product_details":
                pname = result_data.get("name", "") if isinstance(result_data, dict) else ""
                summary = f"Looked up product: {pname}"
            elif tool_name == "add_to_cart":
                if isinstance(result_data, dict) and result_data.get("action") == "add_to_cart":
                    product = result_data["product"]
                    cart_actions.append({
                        "name": product.get("name", ""),
                        "price": product.get("price", 0),
                        "category": product.get("category", ""),
                        "brand": product.get("brand", ""),
                        "quantity": 1,
                    })
                    summary = f"Added {product.get('name', '')} to cart"

            tool_calls_log.append({"tool": tool_name, "input": json.dumps(tool_input)[:100], "summary": summary})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result_data, default=str)[:4000],
            })

        messages.append({"role": "user", "content": tool_results})

    return {"reply": "I ran out of steps. Please try a simpler request.", "tool_calls": tool_calls_log, "cart_actions": cart_actions}


def lambda_handler(event, context):
    try:
        if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps({"message": "OK"})}

        body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event.get("body", event)
        action = body.get("action", "chat")

        if action == "chat":
            message = body.get("message", "")
            if not message:
                return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": "message is required"})}
            result = run_agent(message, body.get("history", []), body.get("cart", []))
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result)}

        elif action == "health":
            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps({"service": "leafy-agent-handler", "status": "ok", "model": MODEL_ID}),
            }

        return {"statusCode": 400, "headers": CORS_HEADERS, "body": json.dumps({"error": f"Unknown action: {action}"})}

    except Exception as e:
        return {"statusCode": 500, "headers": CORS_HEADERS, "body": json.dumps({"error": str(e)})}
