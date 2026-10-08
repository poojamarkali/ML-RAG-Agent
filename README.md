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

CHAT UI (/ask) ─► semantic cache ─► hybrid search (20) ─► re-rank (top 5) ─► LLM answer + [file, page]

AGENT (/chat) ─┬─ search_documents ─► same retrieval as above
               ├─ get_database_schema / query_database ─► SQL (read-only)
               └─ any MCP tools from config/mcp_servers.json
```

The chat UI uses plain RAG (`/ask`): it answers **only** from the retrieved chunks and refuses
off-topic questions. The agent (`/chat`) is ready for when SQL and MCP tools are added to the app.

## Project layout

| File | What it does |
|---|---|
| `config/settings.py` | All settings, read from `.env` |
| `config/chunking.json` | Which chunking strategy each PDF uses |
| `config/mcp_servers.json` | External MCP servers whose tools the agent gets |
| `src/ingestion/loaders.py` | **Read** a source → Documents (PDF pages, SQL rows) |
| `src/ingestion/chunking.py` | **Split** Documents → chunks (fixed / semantic / agentic) |
| `src/ingestion/pipeline.py` | **Run** loader → chunking → store in Qdrant |
| `src/ingestion/suggestions.py` | LLM writes suggested questions per PDF at ingest time |
| `src/retrieval/vector_store.py` | Qdrant client, embedding + re-ranker models |
| `src/retrieval/search.py` | `search()`: hybrid / dense / sparse + optional re-ranking |
| `src/retrieval/rag.py` | Plain RAG: retrieve → answer with citations |
| `src/database/schema.py` | Tables + sample data |
| `src/database/executor.py` | Schema description + read-only `run_query()` |
| `src/agent/` | LLM choice, prompt, tools (+ MCP loader), agent, semantic cache |
| `src/evaluation/` | Retrieval metrics, chunking comparison, LLM-as-judge, inspector, test questions |
| `src/api/main.py` | Endpoints the chat app uses (`/ask`, `/suggestions`, `/documents`) |
| `src/api/admin.py` | Admin-only endpoints under `/admin` (inspect, stats, evaluation, upload) |
| `src/mcp_server/server.py` | Exposes our tools as an MCP server |
| `ui/app.py` | Streamlit chat for users: suggested questions, cited answers, sources |
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

## Admin: check how the RAG works (not visible to chat users)

```bash
uv run python run_server.py inspect "What does StandardScaler do?"   # every step for one question
uv run python run_server.py stats                                     # chunks per PDF, sizes, strategy
uv run python run_server.py chunks --file "Introduction to Machine Learning with Python.pdf" --page 147
```

The same tools are in the API under `/admin/...` (see http://localhost:8000/docs):
`/admin/inspect`, `/admin/stats`, `/admin/chunks`, `/admin/search`, `/admin/eval/*`,
`/admin/documents` (upload a PDF), `/admin/cache`. Set `ADMIN_API_KEY` in `.env` to require
an `X-Admin-Key` header for all of them. Evaluation runs are saved in `data/eval_results/`.

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
   (this also generates its suggested questions for the chat UI)

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
