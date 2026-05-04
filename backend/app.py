"""
E-Commerce Product Search API.
Supports vector search, text search, hybrid search, and hybrid + rerank.
Includes mock cart & checkout for workshop interactivity.
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
import voyageai

load_dotenv()

logger = logging.getLogger("leafyshop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Repo root for templates / static (`backend/` is the parent of this file)
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BACKEND_DIR.parent / "frontend").resolve()

db_client: MongoClient = None
coll = None
orders_coll = None
vo: voyageai.Client = None
_watcher_stop = threading.Event()
_watcher_thread: Optional[threading.Thread] = None


def _enrich_product(doc_id, name: str, price: float) -> None:
    """Call Bedrock + Voyage to fill in description/category/tags/embedding for a bare product."""
    bedrock_url = os.environ.get("BEDROCK_API_URL", "").rstrip("/")
    if not bedrock_url:
        logger.warning("BEDROCK_API_URL not set — skipping enrichment for %s", doc_id)
        return

    prompt = (
        "You are a product catalog AI. Given a product name and price, "
        "generate detailed e-commerce listing information.\n\n"
        f"Product Name: {name}\nPrice: ${price}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"description":"2-3 sentence description","category":"Top > Sub",'
        '"tags":["t1","t2","t3","t4","t5"],"brand":"BrandName"}'
    )
    resp = requests.post(
        f"{bedrock_url}/genai_workshop",
        json={"action": "answer", "question": prompt, "context": "", "max_tokens": 500, "temperature": 0.3},
        timeout=45,
    )
    resp.raise_for_status()
    answer = (resp.json() or {}).get("answer", "")
    start, end = answer.find("{"), answer.rfind("}") + 1
    details = json.loads(answer[start:end] if start >= 0 and end > start else answer)

    embedding = vo.embed([details["description"]], model="voyage-4-large", input_type="document").embeddings[0]
    coll.update_one(
        {"_id": doc_id},
        {"$set": {
            "description": details["description"],
            "category": details.get("category", "Uncategorized"),
            "tags": details.get("tags", []),
            "brand": details.get("brand", "Generic"),
            "description_embedding": embedding,
            "status": "active",
            "enriched_by": "claude-4.5-haiku",
            "enriched_at": datetime.now(timezone.utc),
        }},
    )
    logger.info("Enrichment complete: %s (%s)", name, doc_id)


def _catalog_watcher_loop() -> None:
    """Watch for new bare-product inserts and trigger enrichment in-process.

    Runs on its own thread because PyMongo's change stream cursor is sync/blocking.
    """
    pipeline = [{"$match": {"operationType": "insert", "fullDocument.status": "pending_enrichment"}}]
    while not _watcher_stop.is_set():
        try:
            with coll.watch(pipeline, max_await_time_ms=5000) as stream:
                logger.info("Catalog watcher attached to change stream")
                while not _watcher_stop.is_set():
                    change = stream.try_next()
                    if change is None:
                        continue
                    doc = change["fullDocument"]
                    logger.info("New product detected: %s ($%s)", doc.get("name"), doc.get("price"))
                    try:
                        _enrich_product(doc["_id"], doc["name"], doc.get("price", 0))
                    except Exception as e:
                        logger.exception("Enrichment failed: %s", e)
        except Exception as e:
            if _watcher_stop.is_set():
                break
            logger.exception("Catalog watcher error, retrying in 5s: %s", e)
            _watcher_stop.wait(5)
    logger.info("Catalog watcher stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, coll, orders_coll, vo, _watcher_thread
    db_client = MongoClient(os.environ["MONGODB_URI"])
    db = db_client["workshop"]
    coll = db["products"]
    orders_coll = db["orders"]
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    if os.environ.get("ENABLE_CATALOG_WATCHER", "1") != "0":
        _watcher_stop.clear()
        _watcher_thread = threading.Thread(target=_catalog_watcher_loop, name="catalog-watcher", daemon=True)
        _watcher_thread.start()
        logger.info("Catalog watcher thread started")

    yield

    _watcher_stop.set()
    if _watcher_thread:
        _watcher_thread.join(timeout=10)
    db_client.close()


app = FastAPI(title="LeafyShop API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost(:\d+)?|.*\.cloudfront\.net|.*\.awsapprunner\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def get_query_embedding(text: str) -> list[float]:
    result = vo.embed([text], model="voyage-4-large", input_type="query")
    return result.embeddings[0]


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(vector|text|hybrid|rerank)$"),
    category: Optional[str] = None,
    limit: int = Query(12, ge=1, le=50),
):
    if mode == "vector":
        results = vector_search(q, category, limit)
    elif mode == "text":
        results = text_search(q, category, limit)
    elif mode == "rerank":
        results = hybrid_rerank_search(q, category, limit)
    else:
        results = hybrid_search(q, category, limit)
    return {"query": q, "mode": mode, "count": len(results), "results": results}


def vector_search(query: str, category: Optional[str], limit: int):
    query_vector = get_query_embedding(query)
    vs_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "description_embedding",
            "queryVector": query_vector,
            "numCandidates": max(100, limit * 10),
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
    return _serialize(coll.aggregate(pipeline))


def text_search(query: str, category: Optional[str], limit: int):
    compound: dict = {
        "must": [
            {"text": {"query": query, "path": ["name", "description"]}}
        ]
    }
    if category:
        compound["filter"] = [{"text": {"query": category, "path": "category"}}]

    pipeline = [
        {"$search": {"index": "text_search_index", "compound": compound}},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        {"$project": {"description_embedding": 0}},
        {"$limit": limit},
    ]
    return _serialize(coll.aggregate(pipeline))


def hybrid_search(query: str, category: Optional[str], limit: int, k: int = 60):
    vector_results = vector_search(query, category, limit=50)
    text_results = text_search(query, category, limit=50)

    rrf: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        did = str(doc.get("id", rank))
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
        doc_map[did] = doc

    for rank, doc in enumerate(text_results):
        did = str(doc.get("id", rank))
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
        doc_map[did] = doc

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:limit]
    results = []
    for did in sorted_ids:
        doc = doc_map[did]
        doc["score"] = round(rrf[did], 6)
        results.append(doc)
    return results


def hybrid_rerank_search(query: str, category: Optional[str], limit: int):
    candidates = hybrid_search(query, category, limit=30)
    if not candidates:
        return candidates
    descriptions = [doc.get("description", doc.get("name", "")) for doc in candidates]
    reranked = vo.rerank(query=query, documents=descriptions, model="rerank-2.5", top_k=limit)
    results = []
    for r in reranked.results:
        doc = candidates[r.index]
        doc["score"] = round(r.relevance_score, 4)
        results.append(doc)
    return results


def _serialize(cursor) -> list[dict]:
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        for k, v in doc.items():
            if isinstance(v, float):
                doc[k] = round(v, 4)
        results.append(doc)
    return results


class CartItem(BaseModel):
    name: str
    price: float
    category: str = ""
    brand: str = ""
    quantity: int = 1


class CheckoutRequest(BaseModel):
    user_name: str
    items: list[CartItem]
    search_mode: str = "hybrid"


@app.post("/api/checkout")
def checkout(req: CheckoutRequest):
    if not req.items:
        return {"error": "Cart is empty"}

    total = round(sum(item.price * item.quantity for item in req.items), 2)
    order_doc = {
        "user_name": req.user_name,
        "items": [item.model_dump() for item in req.items],
        "total": total,
        "item_count": sum(item.quantity for item in req.items),
        "search_mode_used": req.search_mode,
        "created_at": datetime.now(timezone.utc),
    }
    result = orders_coll.insert_one(order_doc)
    order_doc["_id"] = str(result.inserted_id)
    return {"order_id": order_doc["_id"], "total": total, "item_count": order_doc["item_count"]}


@app.get("/api/orders")
def get_orders(user_name: str = Query(...)):
    cursor = orders_coll.find({"user_name": user_name}).sort("created_at", -1).limit(20)
    orders = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        orders.append(doc)
    return {"orders": orders}


class SellerProduct(BaseModel):
    name: str
    price: float


@app.post("/api/seller/add-product")
def seller_add_product(product: SellerProduct):
    doc = {
        "name": product.name,
        "price": round(product.price, 2),
        "status": "pending_enrichment",
        "created_at": datetime.now(timezone.utc),
    }
    result = coll.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "name": doc["name"],
        "price": doc["price"],
        "status": doc["status"],
    }


@app.get("/api/seller/recent")
def seller_recent(limit: int = Query(10, ge=1, le=30)):
    from bson import ObjectId
    cursor = coll.find().sort("_id", -1).limit(limit)
    items = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["has_embedding"] = "description_embedding" in doc
        doc.pop("description_embedding", None)
        if "created_at" in doc and isinstance(doc["created_at"], datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        if "enriched_at" in doc and isinstance(doc["enriched_at"], datetime):
            doc["enriched_at"] = doc["enriched_at"].isoformat()
        items.append(doc)
    return {"products": items}


@app.get("/healthz")
def healthz():
    """App Runner health check."""
    return {"status": "ok"}


def _serve_index() -> str:
    html = (FRONTEND_DIR / "index.html").read_text()
    bedrock_url = os.environ.get("BEDROCK_API_URL", "").rstrip("/")
    inject = f'<script>window.BEDROCK_API_URL = "{bedrock_url}";</script>'
    return html.replace("</head>", f"{inject}</head>", 1)


@app.get("/", response_class=HTMLResponse)
def root():
    return _serve_index()


# CloudFront may rewrite `/` to `/index.html` when DefaultRootObject is set on the
# distribution — serve the same SPA bundle for both paths so prefix-routed deployments work.
@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
def index_html():
    return _serve_index()
