"""Retrieval: pull a wide candidate pool, then re-rank with a cross-encoder
(same as `retrieve_and_rerank` in the combined chunking notebook)."""

from langchain_core.documents import Document

from src.retrieval.vector_store import get_reranker, get_vector_store, source_filter


def search(
    query: str,
    k: int = 5,
    source: str | None = None,
    mode: str = "hybrid",
    rerank: bool = True,
    candidates: int = 20,
    collection: str | None = None,
) -> list[Document]:
    """mode: "hybrid" | "dense" | "sparse". `source` limits the search to one PDF."""
    store = get_vector_store(mode, collection)
    docs = store.similarity_search(query, k=candidates if rerank else k, filter=source_filter(source))
    if not rerank or not docs:
        return docs

    scores = get_reranker().predict([[query, doc.page_content] for doc in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
    for doc, score in ranked:
        doc.metadata["score"] = round(float(score), 4)
    return [doc for doc, _ in ranked]


def format_docs(docs: list[Document]) -> str:
    """Chunks -> one context string with [file, page] citations for the LLM."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )
