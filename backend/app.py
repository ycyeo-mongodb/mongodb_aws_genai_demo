"""
E-Commerce Product Search API.
Supports vector search, text search, hybrid search, hybrid + rerank, and
multimodal image search (upload a photo and find matching catalog items).
Includes mock cart & checkout for workshop interactivity.
"""

import io
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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


# ─────────────────────────────────────────────────────────────────
# Search configuration constants — surfaced so the UI can show them.
# ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "voyage-4-large"
MULTIMODAL_MODEL = "voyage-multimodal-3.5"
RERANK_MODEL = "rerank-2.5"
RRF_K = 60                  # reciprocal-rank-fusion constant
HYBRID_CANDIDATES = 50      # docs pulled from each retriever before fusion
RERANK_CANDIDATES = 30      # docs sent into the reranker after RRF
FUZZY_MAX_EDITS = 2         # Atlas Search fuzzy: 0–2 character edits tolerated
FUZZY_PREFIX_LEN = 1        # first N chars must match exactly (faster + safer)
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB hard limit on uploads
IMAGE_RESIZE_MAX = 1024              # cap longest side; multimodal models tokenize per pixel


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(vector|text|hybrid|rerank)$"),
    category: Optional[str] = None,
    limit: int = Query(12, ge=1, le=50),
    explain: bool = Query(False, description="Include per-stage diagnostics for the UI modal."),
):
    if mode == "vector":
        results, debug = vector_search(q, category, limit, explain=explain)
    elif mode == "text":
        results, debug = text_search(q, category, limit, explain=explain)
    elif mode == "rerank":
        results, debug = hybrid_rerank_search(q, category, limit, explain=explain)
    else:
        results, debug = hybrid_search(q, category, limit, explain=explain)

    payload = {"query": q, "mode": mode, "count": len(results), "results": results}
    if explain:
        payload["explain"] = debug
    return payload


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _build_vector_pipeline(query_vector: list[float], category: Optional[str], limit: int) -> list[dict]:
    """Build the $vectorSearch pipeline. Returned as a plain list[dict] so we can
    both execute it and surface it in the explain payload."""
    vs_stage: dict = {
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
    return [
        vs_stage,
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"description_embedding": 0}},
    ]


def _build_text_pipeline(query: str, category: Optional[str], limit: int) -> list[dict]:
    """Build the $search compound pipeline with both an exact and a fuzzy clause.

    Why two clauses?
      * `should[0]` (boosted exact text) — clean queries score highest
      * `should[1]` (fuzzy) — `runing` still finds `running` (typo recovery)
    """
    fuzzy_cfg = {"maxEdits": FUZZY_MAX_EDITS, "prefixLength": FUZZY_PREFIX_LEN}
    compound: dict = {
        "should": [
            {
                "text": {
                    "query": query,
                    "path": ["name", "description"],
                    "score": {"boost": {"value": 3}},
                }
            },
            {
                "text": {
                    "query": query,
                    "path": ["name", "description"],
                    "fuzzy": fuzzy_cfg,
                }
            },
        ],
        "minimumShouldMatch": 1,
    }
    if category:
        compound["filter"] = [{"text": {"query": category, "path": "category"}}]

    return [
        {"$search": {"index": "text_search_index", "compound": compound}},
        {"$addFields": {"score": {"$meta": "searchScore"}}},
        {"$project": {"description_embedding": 0}},
        {"$limit": limit},
    ]


def _doc_key(doc: dict, fallback_index: int) -> str:
    """Stable per-document key used by the RRF fusion table."""
    return str(doc.get("id", fallback_index))


# ─────────────────────────────────────────────────────────────────
# Search modes
# ─────────────────────────────────────────────────────────────────

def vector_search(query: str, category: Optional[str], limit: int, explain: bool = False):
    query_vector = get_query_embedding(query)
    pipeline = _build_vector_pipeline(query_vector, category, limit)
    results = _serialize(coll.aggregate(pipeline))
    debug = None
    if explain:
        debug = {
            "stages": [
                {
                    "name": "Embed query",
                    "detail": f"Voyage AI {EMBEDDING_MODEL} → {len(query_vector)}-dim vector",
                    "embedding_preview": [round(v, 4) for v in query_vector[:8]],
                    "embedding_dims": len(query_vector),
                },
                {
                    "name": "$vectorSearch",
                    "detail": "Cosine similarity over description_embedding",
                    "pipeline": pipeline,
                    "results": _summarise(results, "similarity"),
                },
            ],
        }
    return results, debug


def text_search(query: str, category: Optional[str], limit: int, explain: bool = False):
    pipeline = _build_text_pipeline(query, category, limit)
    results = _serialize(coll.aggregate(pipeline))
    debug = None
    if explain:
        debug = {
            "stages": [
                {
                    "name": "$search (compound: exact + fuzzy)",
                    "detail": (
                        f"Atlas Search BM25-like text scoring on name + description, "
                        f"with fuzzy maxEdits={FUZZY_MAX_EDITS}, prefixLength={FUZZY_PREFIX_LEN}"
                    ),
                    "fuzzy": {"maxEdits": FUZZY_MAX_EDITS, "prefixLength": FUZZY_PREFIX_LEN},
                    "pipeline": pipeline,
                    "results": _summarise(results, "text score"),
                },
            ],
        }
    return results, debug


def hybrid_search(query: str, category: Optional[str], limit: int, k: int = RRF_K, explain: bool = False):
    """Reciprocal Rank Fusion of vector + text results.
    score(doc) = sum over retrievers of 1 / (k + rank_in_retriever + 1)
    """
    query_vector = get_query_embedding(query)
    vector_pipeline = _build_vector_pipeline(query_vector, category, HYBRID_CANDIDATES)
    text_pipeline = _build_text_pipeline(query, category, HYBRID_CANDIDATES)

    vector_results = _serialize(coll.aggregate(vector_pipeline))
    text_results = _serialize(coll.aggregate(text_pipeline))

    # Snapshot retriever scores BEFORE fusion mutates `score` to the RRF value.
    vector_summary = _summarise(vector_results, "similarity", top=10) if explain else None
    text_summary = _summarise(text_results, "text score", top=10) if explain else None

    rrf: dict[str, float] = {}
    contrib: dict[str, dict] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        did = _doc_key(doc, rank)
        c = 1 / (k + rank + 1)
        rrf[did] = rrf.get(did, 0) + c
        contrib.setdefault(did, {})["vector"] = {"rank": rank + 1, "score": doc.get("score"), "contrib": round(c, 6)}
        doc_map[did] = doc

    for rank, doc in enumerate(text_results):
        did = _doc_key(doc, rank)
        c = 1 / (k + rank + 1)
        rrf[did] = rrf.get(did, 0) + c
        contrib.setdefault(did, {})["text"] = {"rank": rank + 1, "score": doc.get("score"), "contrib": round(c, 6)}
        doc_map[did] = doc

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:limit]
    results: list[dict] = []
    for did in sorted_ids:
        doc = doc_map[did]
        doc["score"] = round(rrf[did], 6)
        results.append(doc)

    debug = None
    if explain:
        # Build a flat fusion table: top docs with their per-retriever contributions.
        fusion_rows = []
        for did in sorted_ids:
            d = doc_map[did]
            row = {
                "id": d.get("id"),
                "name": d.get("name"),
                "vector": contrib.get(did, {}).get("vector"),
                "text": contrib.get(did, {}).get("text"),
                "rrf_score": round(rrf[did], 6),
            }
            fusion_rows.append(row)

        debug = {
            "stages": [
                {
                    "name": "Embed query",
                    "detail": f"Voyage AI {EMBEDDING_MODEL} → {len(query_vector)}-dim vector",
                    "embedding_preview": [round(v, 4) for v in query_vector[:8]],
                    "embedding_dims": len(query_vector),
                },
                {
                    "name": "Retriever A — $vectorSearch",
                    "detail": f"Top {HYBRID_CANDIDATES} candidates by cosine similarity",
                    "pipeline": vector_pipeline,
                    "results": vector_summary,
                },
                {
                    "name": "Retriever B — $search (exact + fuzzy)",
                    "detail": (
                        f"Top {HYBRID_CANDIDATES} candidates from Atlas Search; "
                        f"fuzzy maxEdits={FUZZY_MAX_EDITS}, prefixLength={FUZZY_PREFIX_LEN}"
                    ),
                    "fuzzy": {"maxEdits": FUZZY_MAX_EDITS, "prefixLength": FUZZY_PREFIX_LEN},
                    "pipeline": text_pipeline,
                    "results": text_summary,
                },
                {
                    "name": "Fusion — Reciprocal Rank Fusion",
                    "detail": f"score(doc) = Σ 1 / (k + rank_in_retriever + 1) with k = {k}",
                    "rrf_k": k,
                    "fusion_rows": fusion_rows,
                },
            ],
        }
    return results, debug


def hybrid_rerank_search(query: str, category: Optional[str], limit: int, explain: bool = False):
    candidates, hybrid_debug = hybrid_search(query, category, RERANK_CANDIDATES, explain=explain)
    if not candidates:
        return candidates, hybrid_debug

    descriptions = [doc.get("description", doc.get("name", "")) for doc in candidates]
    reranked = vo.rerank(query=query, documents=descriptions, model=RERANK_MODEL, top_k=limit)

    # Capture the pre-rerank order so the UI can show the reordering.
    rrf_order = [
        {"id": d.get("id"), "name": d.get("name"), "rrf_score": d.get("score")}
        for d in candidates
    ]

    results: list[dict] = []
    rerank_rows = []
    for new_rank, r in enumerate(reranked.results):
        doc = candidates[r.index]
        prev_rank = r.index + 1
        doc["score"] = round(r.relevance_score, 4)
        results.append(doc)
        rerank_rows.append({
            "id": doc.get("id"),
            "name": doc.get("name"),
            "rrf_rank": prev_rank,
            "rerank_rank": new_rank + 1,
            "rrf_score": rrf_order[r.index]["rrf_score"],
            "rerank_score": round(r.relevance_score, 4),
            "moved": prev_rank - (new_rank + 1),
        })

    debug = None
    if explain:
        debug = {
            "stages": (hybrid_debug.get("stages", []) if hybrid_debug else []) + [
                {
                    "name": f"Rerank — Voyage AI {RERANK_MODEL}",
                    "detail": (
                        f"Top {RERANK_CANDIDATES} RRF candidates → cross-encoder relevance "
                        "scores (query × document jointly), highest-relevance shown first"
                    ),
                    "rerank_model": RERANK_MODEL,
                    "rerank_rows": rerank_rows,
                }
            ],
        }
    return results, debug


def _summarise(results: list[dict], score_label: str, top: int = 10) -> list[dict]:
    """Trim per-stage results for the UI: id, name, category, score only."""
    out = []
    for i, doc in enumerate(results[:top]):
        out.append({
            "rank": i + 1,
            "id": doc.get("id"),
            "name": doc.get("name"),
            "category": doc.get("category"),
            "score": doc.get("score"),
            "score_label": score_label,
        })
    return out


def _serialize(cursor) -> list[dict]:
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        for k, v in doc.items():
            if isinstance(v, float):
                doc[k] = round(v, 4)
        results.append(doc)
    return results


# ─────────────────────────────────────────────────────────────────
# Multimodal image search
#
# Shopper uploads a photo (e.g. a model wearing a denim jacket) and optionally
# adds a text refinement ("similar but in red"). We embed [text?, image] with
# voyage-multimodal-3.5 — a single transformer backbone that puts text and
# images in the same vector space — then $vectorSearch on the catalog's
# `multimodal_embedding` field, which was built with the same model.
# ─────────────────────────────────────────────────────────────────

def _load_and_resize_image(raw: bytes) -> Image.Image:
    """Open + downscale the upload so we don't pay for unnecessary tokens
    (every 560 pixels is one token for voyage-multimodal-3.5)."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")
    # Drop alpha + EXIF orientation handling
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > IMAGE_RESIZE_MAX:
        scale = IMAGE_RESIZE_MAX / longest
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _multimodal_search_pipeline(query_vector: list[float], category: Optional[str], limit: int) -> list[dict]:
    vs_stage: dict = {
        "$vectorSearch": {
            "index": "multimodal_index",
            "path": "multimodal_embedding",
            "queryVector": query_vector,
            "numCandidates": max(150, limit * 12),
            "limit": limit,
        }
    }
    if category:
        vs_stage["$vectorSearch"]["filter"] = {"category": category}
    return [
        vs_stage,
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"description_embedding": 0, "multimodal_embedding": 0}},
    ]


@app.post("/api/search/image")
async def search_by_image(
    image: UploadFile = File(..., description="Product photo, outfit shot, or any reference image"),
    q: Optional[str] = Form(None, description="Optional text refinement, e.g. 'similar but in black'"),
    category: Optional[str] = Form(None),
    limit: int = Form(12),
    explain: bool = Form(False),
):
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Limit is {MAX_IMAGE_BYTES // (1024*1024)}MB.",
        )

    pil_image = _load_and_resize_image(raw)
    refinement = (q or "").strip()

    # Voyage multimodal accepts a list of inputs; each input is an interleaved
    # sequence of strings and PIL Images. With one input, we get one vector.
    if refinement:
        # Phrase the text as an instruction so the model understands the
        # refinement applies to the image.
        composed_prompt = f"Find products matching this photo. Additional preference: {refinement}"
        sequence = [composed_prompt, pil_image]
    else:
        sequence = ["Find products matching this photo.", pil_image]

    try:
        result = vo.multimodal_embed(
            inputs=[sequence],
            model=MULTIMODAL_MODEL,
            input_type="query",
        )
    except Exception as e:
        logger.exception("multimodal embed failed")
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    query_vector = result.embeddings[0]
    pipeline = _multimodal_search_pipeline(query_vector, category, limit)
    results = _serialize(coll.aggregate(pipeline))

    payload: dict = {
        "mode": "multimodal",
        "query_text": refinement,
        "image_filename": image.filename,
        "image_size_bytes": len(raw),
        "image_resized_to": list(pil_image.size),
        "model": MULTIMODAL_MODEL,
        "count": len(results),
        "results": results,
    }
    if explain:
        payload["explain"] = {
            "stages": [
                {
                    "name": f"Embed [text?, image] with {MULTIMODAL_MODEL}",
                    "detail": (
                        "Single transformer backbone produces a 1024-dim vector for the "
                        "interleaved query (instruction text + uploaded image). The same "
                        "model embedded every product description, so they share a vector space."
                    ),
                    "embedding_preview": [round(v, 4) for v in query_vector[:8]],
                    "embedding_dims": len(query_vector),
                    "input_pieces": (
                        ["text", "image"] if refinement else ["image"]
                    ),
                    "image_size": list(pil_image.size),
                    "total_tokens": getattr(result, "total_tokens", None),
                },
                {
                    "name": "$vectorSearch on multimodal_index",
                    "detail": (
                        f"Cosine similarity over multimodal_embedding (1024-dim, {MULTIMODAL_MODEL}); "
                        f"top {limit} of {pipeline[0]['$vectorSearch']['numCandidates']} candidates"
                    ),
                    "pipeline": pipeline,
                    "results": _summarise(results, "similarity"),
                },
            ],
        }
    return payload


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
