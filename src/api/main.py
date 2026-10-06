"""FastAPI app. Run: uv run python run_server.py api  ->  docs at http://localhost:8000/docs"""

import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import settings
from src.agent import cache
from src.agent.agent import ask_agent
from src.api.schemas import AskResponse, ChatRequest, ChatResponse, Chunk, SearchRequest, SQLRequest
from src.database.executor import get_schema, run_query
from src.database.schema import init_db
from src.evaluation import llm_eval, retrieval_eval
from src.ingestion.chunking import STRATEGIES
from src.ingestion.pipeline import ingest_pdf, list_documents
from src.retrieval.rag import answer_question
from src.retrieval.search import search
from src.retrieval.vector_store import get_qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ML RAG AI Agent", lifespan=lifespan)


@app.get("/health")
def health():
    get_qdrant_client().get_collections()
    run_query("SELECT 1")
    return {"status": "ok", "llm_provider": settings.llm_provider}


# ---- documents -------------------------------------------------------------
@app.get("/documents")
def documents():
    return list_documents()


@app.get("/chunking/strategies")
def chunking_strategies():
    return list(STRATEGIES)


@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    strategy: str | None = Form(None),
):
    """Upload a PDF and index it. No strategy -> the one set in config/chunking.json."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    if strategy and strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {list(STRATEGIES)}")
    path = settings.documents_dir / Path(file.filename).name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file": path.name, "chunks": ingest_pdf(path, strategy)}


# ---- retrieval / RAG / agent -------------------------------------------------
@app.post("/search", response_model=list[Chunk])
def search_chunks(req: SearchRequest):
    docs = search(req.query, k=req.k, source=req.source, mode=req.mode, rerank=req.rerank)
    return [Chunk(text=d.page_content, metadata=d.metadata) for d in docs]


@app.post("/ask", response_model=AskResponse)
def ask(req: SearchRequest):
    """Plain RAG (no tools): retrieve chunks, LLM answers from them."""
    result = answer_question(req.query, k=req.k, source=req.source)
    return AskResponse(answer=result["answer"],
                       sources=[Chunk(text=d.page_content, metadata=d.metadata) for d in result["sources"]])


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Agent: decides between document search, SQL and MCP tools."""
    return await ask_agent(req.question, [m.model_dump() for m in req.history])


# ---- SQL -------------------------------------------------------------------
@app.get("/sql/schema")
def sql_schema():
    return {"schema": get_schema()}


@app.post("/sql")
def sql(req: SQLRequest):
    try:
        return run_query(req.sql)
    except Exception as e:
        raise HTTPException(400, str(e))


# ---- evaluation / cache ----------------------------------------------------
@app.get("/eval/retrieval")
def eval_retrieval(k: int = 5):
    return retrieval_eval.compare_strategies(k=k)


@app.get("/eval/llm")
def eval_llm():
    return llm_eval.run_llm_eval()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"cleared": True}
