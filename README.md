# 🔍 RAG Assistant — Enterprise Document Q&A

A production-grade Retrieval-Augmented Generation (RAG) assistant for internal knowledge access.
Built with **Python**, **LangChain**, **Ollama**, and **pgvector (PostgreSQL)**.

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Document Sources                    │
│   PDFs  │  Confluence Pages  │  Markdown  │  Word    │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│                  Ingestion Pipeline                   │
│  Load → Clean → Chunk → Embed → Store in pgvector   │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│             PostgreSQL + pgvector                     │
│   langchain_pg_collection / langchain_pg_embedding   │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│                  Retrieval Layer                      │
│    Query Embed → Cosine Similarity → Top-K Chunks    │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│                  Generation Layer                     │
│   System Prompt + Context + Question → Ollama LLM   │
│              + Source Citations                      │
└──────────────────────────────────────────────────────┘
```

---

## 🗄️ Why pgvector?

| Feature | pgvector |
|---|---|
| Storage | PostgreSQL — enterprise-standard |
| Filtering | Full SQL WHERE + JSONB metadata filtering |
| Transactions | ACID — safe concurrent writes |
| Ops | Single database to back up and monitor |
| Scaling | Horizontal read replicas, connection pooling |
| Indexing | IVFFlat and HNSW index types supported |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed
- PostgreSQL with pgvector **or** Docker

---

### Option A — Docker (recommended, easiest)

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant

cp .env.example .env          # no changes needed for Docker

docker-compose up -d          # starts postgres + ollama + app

# Pull AI models into Ollama
docker-compose exec ollama ollama pull llama3.2
docker-compose exec ollama ollama pull nomic-embed-text

# Ingest sample document
docker-compose exec app python scripts/ingest.py

# Ask a question
docker-compose exec app python scripts/query.py --question "What is the vacation policy?"
```

---

### Option B — Local (manual setup)

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # edit POSTGRES_URL if your DB differs

ollama pull llama3.2
ollama pull nomic-embed-text

python scripts/setup_db.py      # enable pgvector extension
python scripts/ingest.py        # ingest documents
python scripts/query.py         # interactive Q&A
```

---

## ⚙️ Configuration

Edit `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text

# postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
POSTGRES_URL=postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb

CHUNK_SIZE=600
CHUNK_OVERLAP=75
TOP_K=5
MIN_SIMILARITY_SCORE=0.30
```

---

## 📁 Project Structure

```
rag-assistant/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF, MD, TXT, DOCX loading
│   │   ├── cleaner.py         # text normalisation
│   │   ├── chunker.py         # recursive splitting + metadata
│   │   └── confluence.py      # Confluence REST API
│   ├── retrieval/
│   │   ├── embedder.py        # Ollama nomic-embed-text
│   │   ├── vectorstore.py     # pgvector interface (add/delete/stats)
│   │   └── retriever.py       # cosine similarity search
│   ├── generation/
│   │   ├── prompt.py          # ChatPromptTemplate
│   │   └── chain.py           # full RAG chain
│   ├── evaluation/
│   │   └── evaluator.py       # Recall@k, Precision@k, MRR
│   ├── guardrails/
│   │   └── guardrails.py      # score threshold + injection detection
│   └── api.py                 # FastAPI REST interface
├── scripts/
│   ├── ingest.py              # run ingestion pipeline
│   ├── query.py               # interactive CLI
│   ├── setup_db.py            # one-time DB setup
│   ├── stats.py               # show collection stats
│   └── init_db.sql            # CREATE EXTENSION vector
├── tests/
│   ├── unit/                  # chunker, cleaner, guardrails
│   └── integration/           # ingestion pipeline, query pipeline
├── config/settings.py         # Pydantic settings
├── data/raw/                  # drop documents here
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 🛠️ CLI Commands

```bash
# Ingest all documents from data/raw/
python scripts/ingest.py

# Also pull Confluence pages
python scripts/ingest.py --confluence

# Interactive Q&A
python scripts/query.py

# Single question
python scripts/query.py --question "What is the remote work policy?"

# Filter by file type
python scripts/query.py --question "Password rules?" --filter '{"file_type":"pdf"}'

# Check DB setup
python scripts/setup_db.py

# Show how many chunks are stored
python scripts/stats.py
```

---

## 🌐 REST API

```bash
uvicorn src.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'

# Trigger ingestion
curl -X POST http://localhost:8000/ingest

# Collection stats
curl http://localhost:8000/stats

# Health check
curl http://localhost:8000/health

# Delete a document's chunks
curl -X DELETE http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/app/data/raw/hr_handbook.pdf"}'
```

---

## 🧪 Tests

```bash
pytest                          # all tests
pytest tests/unit/ -v           # unit only (no services needed)
pytest --cov=src --cov-report=html
```

---

## 📊 Evaluation

```bash
python -m src.evaluation.evaluator --benchmark data/eval_questions.json
```

---

## 🔒 Guardrails

- Refuses to answer when similarity score < threshold
- Sanitises prompt injection patterns found in documents
- Always returns source citations
- Deterministic fallback: *"I could not find a reliable answer in the indexed documents."*

---
