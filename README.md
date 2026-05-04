# Advanced Vector Search + Multi-Agent AI Workshop

An advanced hands-on workshop: build AI-powered e-commerce search with **MongoDB Atlas**, **Voyage AI**, and a **multi-agent layer** powered by **Amazon Bedrock (Claude 4.5 Haiku)** and **MongoDB Change Streams**.

## What You'll Build

### Part 1: Search Foundation
- Load a 1,000-product catalog and generate embeddings with Voyage AI
- Build four search modes: **Semantic**, **Text**, **Hybrid**, and **Hybrid + Rerank**
- Run a FastAPI server with a browser-based product search UI

### Part 2: Multi-Agent AI
- **MongoDB Change Streams** — real-time product enrichment pipeline with a Seller Dashboard UI
- **AI Shopping Agent** — conversational chat bubble with tool-use (search, compare, add to cart)
- **Virtual Try-On** — style analysis with Claude Vision and image generation with Nova Canvas

## Repository Structure

```
├── backend/                  # FastAPI application
│   ├── app.py                # Full app (all search modes + seller endpoints)
│   ├── app_starter.py        # Skeleton app (developer track — fill in TODOs)
│   ├── requirements.txt      # Python dependencies
│   ├── data/
│   │   └── products.json     # 1,000-product catalog
│   └── utils/
│       └── generate_catalog.py
├── frontend/                 # Browser-based shop UI
│   └── index.html            # Single-page product search interface
├── scripts/                  # Workshop exercise scripts (run in order)
│   ├── 01_load_and_embed.py  # Load products + generate embeddings
│   ├── 02_create_indexes.py  # Create Atlas search indexes
│   ├── 03_semantic_search.py # Test semantic (vector) search
│   ├── 04_hybrid_search.py   # Test hybrid search with RRF
│   ├── 05_reranking.py       # Test reranking with Voyage AI
│   ├── 06_catalog_watcher.py # Change Streams agent — watches for new products
│   ├── 07_add_bare_product.py# Insert bare product to trigger enrichment
│   └── 08_shopping_assistant.py # Conversational shopping agent with tool use
├── aws/                      # AWS Lambda functions (AI agent backends)
│   ├── leafy_agent_handler.py  # AI Shopping Agent (tool-use loop)
│   ├── leafy_tryon_handler.py  # Virtual Try-On (Claude Vision + Nova Canvas)
│   └── deploy_lambdas.sh       # Deployment script (instructor use only)
├── .env.example
└── .gitignore
```

## Tech Stack

| Technology | Role |
|---|---|
| MongoDB Atlas | Document database, Vector Search, Atlas Search, Change Streams |
| Voyage AI | Embeddings (`voyage-4-large`), Reranking (`rerank-2.5`) |
| Amazon Bedrock | Claude 4.5 Haiku (enrichment, agent), Nova Canvas (image gen) |
| AWS Lambda | Serverless AI agent backends behind API Gateway |
| FastAPI | Python API server |

## Quick Start

```bash
# Clone
git clone https://github.com/ycyeo-mongodb/Advanced_GenAI_MongoDB_Workshop.git
cd Advanced_GenAI_MongoDB_Workshop

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — see "Environment Variables" below
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Where to get it |
|---|---|
| `MONGODB_URI` | Your MongoDB Atlas connection string (free tier works) |
| `VOYAGE_API_KEY` | Atlas → Services → AI Models → Create model API key |
| `BEDROCK_API_URL` | **Provided by the workshop instructor** |

> **Note:** The `BEDROCK_API_URL` is the API Gateway endpoint for the AWS Lambda AI backends. During a live workshop, your instructor will provide this. For self-paced learners, you'll need to deploy the Lambda functions in the `aws/` directory to your own AWS account.

## Running the Workshop

### Part 1: Search

```bash
python scripts/01_load_and_embed.py     # Load products + generate embeddings
python scripts/02_create_indexes.py     # Create search indexes (wait until READY)
python scripts/03_semantic_search.py    # Test semantic search
python scripts/04_hybrid_search.py      # Test hybrid search
python scripts/05_reranking.py          # Test reranking

# Start the search app
cd backend
uvicorn app:app --reload        # Open http://localhost:8000
```

### Part 2: Agents

```bash
# Terminal 1: Start the Catalog Watcher (keep running)
python scripts/06_catalog_watcher.py

# Terminal 2: Insert a bare product (triggers enrichment)
python scripts/07_add_bare_product.py

# Terminal 3: Start the Shopping Assistant
python scripts/08_shopping_assistant.py
```

Then open http://localhost:8000 and explore:
- **Seller Dashboard** — click in the header to add products and watch AI enrichment live
- **AI Shopping Agent** — click the chat bubble in the bottom-right corner
- **Virtual Try-On** — click "Try On" on any clothing/shoes/bags product

## Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- Voyage AI API key (via Atlas → Services → AI Models)
- Bedrock API URL (provided by instructor)
