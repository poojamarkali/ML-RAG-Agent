# ML RAG AI Agent

A chatbot that answers machine learning questions **only from the documents and data you give it**
(ML PDFs + a SQL database of model benchmarks). The same setup works for any other niche:
swap the PDFs and the tables.

**Stack:** LangChain · Groq or AWS Bedrock · Qdrant (hybrid dense + BM25) · cross-encoder re-ranking ·
SQLAlchemy (SQLite / Postgres) · FastAPI · Streamlit · MCP · pydantic · Docker

## How it fits together

```
SOURCES            INGESTION                                   STORAGE
PDFs ──┐
       ├─► loaders.py ─► chunking.py ─► pipeline.py ─────────► Qdrant (dense + BM25 vectors)
SQL ───┘   (read)        (split)        (run + store)

QUESTION ─► semantic cache ─► agent ─┬─ search_documents ─► hybrid search ─► re-rank ─► chunks
                                     ├─ get_database_schema / query_database ─► SQL (read-only)
                                     └─ any MCP tools from config/mcp_servers.json
```

## Project layout

| File | What it does |
|---|---|
| `config/settings.py` | All settings, read from `.env` |
| `config/chunking.json` | Which chunking strategy each PDF uses |
| `config/mcp_servers.json` | External MCP servers whose tools the agent gets |
| `src/ingestion/loaders.py` | **Read** a source → Documents (PDF pages, SQL rows) |
| `src/ingestion/chunking.py` | **Split** Documents → chunks (fixed / semantic / agentic) |
| `src/ingestion/pipeline.py` | **Run** loader → chunking → store in Qdrant |
| `src/retrieval/vector_store.py` | Qdrant client, embedding + re-ranker models |
| `src/retrieval/search.py` | `search()`: hybrid / dense / sparse + optional re-ranking |
| `src/retrieval/rag.py` | Plain RAG: retrieve → answer with citations |
| `src/database/schema.py` | Tables + sample data |
| `src/database/executor.py` | Schema description + read-only `run_query()` |
| `src/agent/` | LLM choice, prompt, tools (+ MCP loader), agent, semantic cache |
| `src/evaluation/` | Retrieval metrics, chunking comparison, LLM-as-judge, test questions |
| `src/api/` | FastAPI endpoints + request/response models |
| `src/mcp_server/server.py` | Exposes our tools as an MCP server |
| `ui/app.py` | Streamlit UI (calls the API) |
| `run_server.py` | One CLI for everything |

## Run locally

```bash
uv sync
cp .env.example .env                                   # add GROQ_API_KEY
uv run python run_server.py ingest                     # every PDF in data/documents
uv run python run_server.py eval                       # retrieval quality
uv run python run_server.py api                        # http://localhost:8000/docs
uv run python run_server.py ui                         # http://localhost:8501 (needs the api)
```

Local Qdrant (a folder) allows only one process at a time: stop the API before `ingest`/`eval`,
or run the Qdrant server from docker and set `QDRANT_URL=http://localhost:6333`.

## Run with Docker

```bash
docker compose up --build
docker compose exec api python run_server.py ingest
```

## Adding a new PDF

1. Put it in `data/documents/`.
2. Add 10–20 questions for it to `src/evaluation/golden_dataset.json` (with the PDF page that answers each).
3. Compare chunking strategies on it and pick the best:
   `uv run python run_server.py eval --chunking --file "new_book.pdf"`
   then record the winner in `config/chunking.json`:
   `{"default": "fixed", "files": {"new_book.pdf": "semantic"}}`
4. `uv run python run_server.py ingest --file "data/documents/new_book.pdf"`

## Evaluation

| Command | Measures |
|---|---|
| `eval` | Hit rate + MRR for dense / BM25 / hybrid / hybrid+rerank |
| `eval --chunking` | Same metrics for each chunking strategy (re-indexes the book per strategy) |
| `eval --llm` | LLM-as-judge: faithfulness, relevance, and whether off-topic questions are refused |

`expected_pages` are PDF page numbers as shown in a PDF viewer, not the printed book page.

## Adding MCP tools

```json
{
  "filesystem": {"transport": "stdio", "command": "npx",
                 "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]},
  "my_server": {"transport": "streamable_http", "url": "http://localhost:9000/mcp"}
}
```
