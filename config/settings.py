"""All project settings in one place. Values come from .env (see .env.example)."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Same as the notebooks: put .env into os.environ so LangChain / boto3 / Groq /
# LangSmith pick up their own keys (GROQ_API_KEY, AWS_*, LANGSMITH_*) directly.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # LLM -- "groq" or "bedrock"
    llm_provider: Literal["groq", "bedrock"] = "groq"
    groq_model: str = "qwen/qwen3.8-27b"
    bedrock_model_id: str = "global.anthropic.claude-opus-5-5"  # Bedrock needs an inference-profile ID
    aws_region: str = "us-east-1"

    # Embeddings (dense + BM25 sparse for hybrid search) and the re-ranker
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model_name: str = "Qdrant/bm25"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Qdrant -- QDRANT_URL set = Qdrant server (docker/cloud), empty = local folder
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = BASE_DIR / "data" / "qdrant_storage"
    qdrant_collection_name: str = "ml_documents"

    # SQL -- any SQLAlchemy URL (sqlite locally, postgres in docker)
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'database' / 'ml_metrics.db').as_posix()}"

    # Folders / files
    documents_dir: Path = BASE_DIR / "data" / "documents"
    mcp_servers_file: Path = BASE_DIR / "config" / "mcp_servers.json"
    chunking_config_file: Path = BASE_DIR / "config" / "chunking.json"
    suggestions_file: Path = BASE_DIR / "data" / "suggestions.json"
    eval_results_dir: Path = BASE_DIR / "data" / "eval_results"

    # Admin endpoints (/admin/...): if set, requests need the header  X-Admin-Key: <this value>
    admin_api_key: str | None = None

    # Semantic cache -- reuse an answer when a new question is this similar
    enable_semantic_cache: bool = True
    semantic_cache_similarity_threshold: float = 0.88  # paraphrases scored 0.90-0.92, different questions <= 0.78

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
