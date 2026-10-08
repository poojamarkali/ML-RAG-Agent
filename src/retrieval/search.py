"""Retrieval in two stages (same as `retrieve_and_rerank` in the combined chunking notebook):

  1. retrieve_candidates(): hybrid search (meaning + keywords) pulls a wide pool, e.g. 20 chunks
  2. rerank():              a cross-encoder reads question + chunk together and keeps the best k
"""

from langchain_core.documents import Document

from src.retrieval.vector_store import get_reranker, get_vector_store, source_filter


def retrieve_candidates(
    query: str, k: int = 20, source: str | None = None, mode: str = "hybrid", collection: str | None = None
) -> list[Document]:
    """Stage 1. mode: "hybrid" | "dense" | "sparse". `source` limits the search to one PDF."""
    return get_vector_store(mode, collection).similarity_search(query, k=k, filter=source_filter(source))


def rerank(query: str, docs: list[Document], k: int = 5) -> list[Document]:
    """Stage 2. Score every (question, chunk) pair and keep the k best. Score is saved in metadata."""
    if not docs:
        return []
    scores = get_reranker().predict([[query, doc.page_content] for doc in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
    for doc, score in ranked:
        doc.metadata["score"] = round(float(score), 4)
    return [doc for doc, _ in ranked]


def search(
    query: str,
    k: int = 5,
    source: str | None = None,
    mode: str = "hybrid",
    rerank_results: bool = True,
    candidates: int = 20,
    collection: str | None = None,
) -> list[Document]:
    """Both stages. With rerank_results=False you get plain stage-1 results (used to compare in evaluation)."""
    if not rerank_results:
        return retrieve_candidates(query, k, source, mode, collection)
    return rerank(query, retrieve_candidates(query, candidates, source, mode, collection), k)


def format_docs(docs: list[Document]) -> str:
    """Chunks -> one context string with [file, page] citations for the LLM."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )
