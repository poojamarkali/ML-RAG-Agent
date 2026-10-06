"""Request / response shapes for the API (pydantic)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(examples=["Explain Random Forest and show its accuracy on the breast cancer dataset."])
    history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    cached: bool
    tools_used: list[dict[str, Any]]


class SearchRequest(BaseModel):
    query: str = Field(examples=["What is the difference between Lasso and Ridge regression?"])
    k: int = Field(default=5, ge=1, le=20)
    source: str | None = Field(default=None, description="Only search this PDF file name")
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    rerank: bool = True


class Chunk(BaseModel):
    text: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]


class SQLRequest(BaseModel):
    sql: str = Field(examples=["SELECT model_name, metric_value FROM model_benchmarks ORDER BY metric_value DESC LIMIT 5"])
