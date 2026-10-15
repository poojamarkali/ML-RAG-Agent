"""Request / response shapes for the API (pydantic)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---- chat app ---------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(examples=["What does StandardScaler do to each feature?"])
    source: str | None = Field(default=None, description="Only answer from this PDF file name")


class Source(BaseModel):
    text: str
    source: str | None
    page: int | None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(examples=["Explain Random Forest and show its accuracy on the breast cancer dataset."])
    history: list[Message] = []
    research: bool = Field(default=False, description="Research mode: also use Hugging Face (MCP) tools")


class SQLResult(BaseModel):
    sql: str
    rows: list[dict[str, Any]]
    error: str | None = None


class ExternalResult(BaseModel):
    tool: str                        # e.g. hub_repo_search (from the Hugging Face MCP server)
    text: str


class ChatResponse(BaseModel):
    answer: str
    cached: bool
    sources: list[Source] = []       # PDF passages the agent used
    sql: list[SQLResult] = []        # database queries the agent ran, with their rows
    external: list[ExternalResult] = []  # results from external MCP tools (Research mode)
    tools_used: list[dict[str, Any]]


# ---- admin ------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(examples=["What is the difference between Lasso and Ridge regression?"])
    k: int = Field(default=5, ge=1, le=20)
    source: str | None = Field(default=None, description="Only search this PDF file name")
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    rerank: bool = True


class Chunk(BaseModel):
    text: str
    metadata: dict[str, Any]


class SQLRequest(BaseModel):
    sql: str = Field(examples=["SELECT model_name, metric_value FROM model_benchmarks ORDER BY metric_value DESC LIMIT 5"])
