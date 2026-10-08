"""FastAPI app. Run: uv run python run_server.py api  ->  docs at http://localhost:8000/docs

This file = what the chat app uses. Developer tools (upload, inspect, evaluation)
live in admin.py under /admin.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from src.agent.agent import ask_agent
from src.api.admin import router as admin_router
from src.api.schemas import AskRequest, AskResponse, ChatRequest, ChatResponse
from src.database.schema import init_db
from src.ingestion.pipeline import list_documents
from src.ingestion.suggestions import load_suggestions
from src.retrieval.rag import answer_question
from src.retrieval.vector_store import get_qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ML RAG AI Agent", lifespan=lifespan)
app.include_router(admin_router)


@app.get("/health", tags=["chat app"])
def health():
    get_qdrant_client().get_collections()
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.get("/documents", tags=["chat app"])
def documents():
    """The PDFs the bot can answer from."""
    return list_documents()


@app.get("/suggestions", tags=["chat app"])
def suggestions():
    """Suggested questions per PDF, generated when each PDF was ingested."""
    return load_suggestions()


@app.post("/ask", response_model=AskResponse, tags=["chat app"])
def ask(req: AskRequest):
    """RAG answer from the documents only, with the source chunks it used."""
    return answer_question(req.question, source=req.source)


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(req: ChatRequest):
    """Agent: decides between document search, SQL and MCP tools (not used by the chat UI yet)."""
    return await ask_agent(req.question, [m.model_dump() for m in req.history])
