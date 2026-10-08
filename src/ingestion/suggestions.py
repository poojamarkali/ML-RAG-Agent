"""Suggested questions shown in the chat UI.

When a PDF is ingested, the LLM reads a sample of its chunks and writes a few
questions those passages answer -- so every suggestion is answerable from the
documents. Saved in data/suggestions.json as {file name: [questions]}.
"""

import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings

SUGGESTIONS_PROMPT = ChatPromptTemplate.from_template("""\
Below are passages from a machine learning document. Write {n} short questions a student might ask
that these passages clearly answer. Cover different topics, keep each question under 15 words,
and don't mention "the passage" or "the document".

{passages}
""")


class Suggestions(BaseModel):
    questions: list[str] = Field(description="The questions")


def sample_chunks(chunks: list[Document], n: int = 12) -> list[Document]:
    """n chunks spread evenly through the document, skipping the first/last 10% of pages
    (cover, table of contents, index) and very short chunks."""
    pages = [c.metadata.get("page", 0) for c in chunks]
    low, high = min(pages) + (max(pages) - min(pages)) * 0.1, max(pages) - (max(pages) - min(pages)) * 0.1
    body = [c for c in chunks if low <= c.metadata.get("page", 0) <= high and len(c.page_content) > 300]
    step = max(len(body) // n, 1)
    return body[::step][:n]


def generate_suggestions(chunks: list[Document], n: int = 6) -> list[str]:
    from src.agent.llm import get_llm

    passages = "\n\n---\n\n".join(c.page_content for c in sample_chunks(chunks))
    chain = SUGGESTIONS_PROMPT | get_llm().with_structured_output(Suggestions)
    return chain.invoke({"n": n, "passages": passages}).questions[:n]


def load_suggestions() -> dict[str, list[str]]:
    path = settings.suggestions_file
    return json.loads(path.read_text()) if path.exists() else {}


def save_suggestions(file_name: str, questions: list[str]):
    data = load_suggestions()
    data[file_name] = questions
    settings.suggestions_file.parent.mkdir(parents=True, exist_ok=True)
    settings.suggestions_file.write_text(json.dumps(data, indent=2))
