"""Semantic cache: if a new question means the same as one we already answered
(cosine similarity >= threshold), return the saved answer and skip the LLM.
Stored in its own small Qdrant collection."""

from config import settings
from src.retrieval.vector_store import create_collection, get_qdrant_client, get_vector_store

CACHE_COLLECTION = "semantic_cache"


def _store():
    create_collection(CACHE_COLLECTION, hybrid=False)
    return get_vector_store("dense", collection=CACHE_COLLECTION)


def lookup(question: str) -> str | None:
    if not settings.enable_semantic_cache:
        return None
    hits = _store().similarity_search_with_score(
        question, k=1, score_threshold=settings.semantic_cache_similarity_threshold
    )
    return hits[0][0].metadata["answer"] if hits else None


def save(question: str, answer: str):
    if settings.enable_semantic_cache:
        _store().add_texts([question], metadatas=[{"answer": answer}])


def clear():
    client = get_qdrant_client()
    if client.collection_exists(CACHE_COLLECTION):
        client.delete_collection(CACHE_COLLECTION)
