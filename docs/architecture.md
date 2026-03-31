# Architecture — RAG Assistant

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      OFFLINE (Batch)                        │
│                                                             │
│  Document Sources                                           │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐ │
│  │   PDFs   │  │ Confluence │  │ Markdown │  │   DOCX   │ │
│  └────┬─────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘ │
│       └──────────────┴──────────────┴──────────────┘       │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │   Loader    │  PyPDF, Docx2txt, etc.  │
│                    └──────┬──────┘                          │
│                    ┌──────▼──────┐                          │
│                    │   Cleaner   │  text normalisation      │
│                    └──────┬──────┘                          │
│                    ┌──────▼──────┐                          │
│                    │   Chunker   │  recursive splitter      │
│                    └──────┬──────┘                          │
│                    ┌──────▼──────┐                          │
│                    │  Embedder   │  nomic-embed-text/Ollama │
│                    └──────┬──────┘                          │
│                    ┌──────▼──────┐                          │
│                    │ Vector Store│  ChromaDB / pgvector     │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      ONLINE (Per Query)                     │
│                                                             │
│  User Question                                              │
│       │                                                     │
│  ┌────▼────────┐                                            │
│  │  Retriever  │  embed query → similarity search → top-k  │
│  └────┬────────┘                                            │
│  ┌────▼────────┐                                            │
│  │ Guardrails  │  check score threshold, detect injection   │
│  └────┬────────┘                                            │
│  ┌────▼────────┐                                            │
│  │   Prompt    │  system + context + question               │
│  └────┬────────┘                                            │
│  ┌────▼────────┐                                            │
│  │  Ollama LLM │  llama3.2 local inference                  │
│  └────┬────────┘                                            │
│  ┌────▼────────┐                                            │
│  │   Answer    │  text + source citations                   │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Chunking Strategy
- Structural split first (headings, paragraph breaks)
- Then hard size limit (600 tokens) with 75-token overlap
- Overlap prevents answers from falling across chunk boundaries
- Metadata attached per chunk for source tracing

### Embedding Model
- `nomic-embed-text` via Ollama — runs fully locally
- Consistent embedding space for both ingestion and queries
- Can be swapped for any LangChain-compatible embedder

### Vector Store
- **ChromaDB** (default): zero-config, SQLite-backed, local
- **pgvector** (production): integrates with existing Postgres infrastructure, enterprise-friendly

### Guardrails
- Similarity threshold blocks hallucination on off-topic questions
- Prompt injection patterns are sanitised before passing to LLM
- Fallback message is deterministic and honest

### LangChain Role
- Document loaders (PDF, Markdown, DOCX)
- RecursiveCharacterTextSplitter
- Prompt templates (ChatPromptTemplate)
- LLM interface (ChatOllama)
- Vector store wrappers (Chroma, PGVector)

## Tradeoffs

| Decision | Alternative | Why this choice |
|----------|-------------|-----------------|
| ChromaDB default | pgvector | Zero setup for dev/demo; pgvector available for prod |
| Ollama local LLM | OpenAI API | Privacy-first, works offline, no data leaves the network |
| LangChain | Custom pipeline | Faster dev, modular, well-documented; easy to replace components |
| nomic-embed-text | OpenAI embeddings | Free, local, strong performance on English text |
| Cosine similarity | BM25 | Semantic matching; hybrid search is a planned extension |
