"""Admin endpoints -- for you to check and manage the RAG system, not for chat users.
All under /admin. If ADMIN_API_KEY is set in .env, send it as the header  X-Admin-Key.
Try them from http://localhost:8000/docs (click "Authorize" first if a key is set).
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from config import settings
from src.agent import cache
from src.api.schemas import Chunk, SearchRequest, SQLRequest
from src.database.executor import get_schema, run_query
from src.evaluation import chunking_eval, inspector, llm_eval, retrieval_eval
from src.evaluation.results import load_results, save_results
from src.ingestion.chunking import STRATEGIES
from src.ingestion.pipeline import ingest_pdf
from src.retrieval.search import search

admin_key = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def check_admin(key: str | None = Security(admin_key)):
    if settings.admin_api_key and key != settings.admin_api_key:
        raise HTTPException(401, "Missing or wrong X-Admin-Key header.")


router = APIRouter(prefix="/admin", dependencies=[Depends(check_admin)])


# ---- see how the pipeline works ---------------------------------------------
@router.get("/inspect", tags=["admin: inspect"])
def inspect(question: str, source: str | None = None, with_answer: bool = True):
    """One question through every step: hybrid candidates -> re-ranked -> prompt -> answer."""
    return inspector.inspect_query(question, source=source, with_answer=with_answer)


@router.get("/stats", tags=["admin: inspect"])
def stats():
    """Per PDF: number of chunks, pages, chunking strategy, chunk sizes."""
    return inspector.document_stats()


@router.get("/chunks", tags=["admin: inspect"])
def chunks(source: str, page: int):
    """The chunks that were made from one page of one PDF."""
    return inspector.page_chunks(source, page)


@router.post("/search", response_model=list[Chunk], tags=["admin: inspect"])
def search_chunks(req: SearchRequest):
    """Raw retrieval with any mode, with or without re-ranking."""
    docs = search(req.query, k=req.k, source=req.source, mode=req.mode, rerank_results=req.rerank)
    return [Chunk(text=d.page_content, metadata=d.metadata) for d in docs]


# ---- evaluation ---------------------------------------------------------------
@router.get("/eval/results", tags=["admin: evaluation"])
def eval_results():
    """The latest saved run of each evaluation (also saved by the CLI `eval` commands)."""
    return load_results()


@router.post("/eval/retrieval", tags=["admin: evaluation"])
def eval_retrieval(k: int = 5):
    """Hit rate + MRR for dense / BM25 / hybrid / hybrid+rerank. ~1 minute."""
    rows = retrieval_eval.compare_strategies(k=k)
    save_results("retrieval", rows)
    return rows


@router.post("/eval/llm", tags=["admin: evaluation"])
def eval_llm():
    """LLM-as-judge on every golden question (2 LLM calls each)."""
    rows = llm_eval.run_llm_eval()
    save_results("llm", rows)
    return rows


@router.post("/eval/chunking", tags=["admin: evaluation"])
def eval_chunking(file_name: str | None = None, only: str | None = None):
    """Compare chunking strategies. SLOW (re-indexes the PDF per strategy) -- the CLI is better for this."""
    kwargs = {"candidates": [s.strip() for s in only.split(",")]} if only else {}
    if file_name:
        kwargs["file_name"] = file_name
    rows = chunking_eval.compare_chunking(**kwargs)
    save_results("chunking", rows)
    return rows


# ---- manage documents / cache -------------------------------------------------
@router.post("/documents", tags=["admin: manage"])
def upload_document(file: UploadFile = File(...), strategy: str | None = Form(None)):
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


@router.delete("/cache", tags=["admin: manage"])
def clear_cache():
    cache.clear()
    return {"cleared": True}


# ---- SQL (kept for the agent; not used by the chat app yet) -------------------
@router.get("/sql/schema", tags=["admin: sql"])
def sql_schema():
    return {"schema": get_schema()}


@router.post("/sql", tags=["admin: sql"])
def sql(req: SQLRequest):
    try:
        return run_query(req.sql)
    except Exception as e:
        raise HTTPException(400, str(e))
