"""Qdrant + embedding models. Same idea as `Chroma.from_documents(...)` in the
notebooks, but Qdrant also stores a BM25 sparse vector per chunk so we get
hybrid (meaning + keyword) search without a separate BM25 index file."""

import atexit
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from config import settings

DENSE, SPARSE = "dense", "sparse"  # vector names inside the collection


@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


@lru_cache
def get_sparse_embeddings():
    return FastEmbedSparse(model_name=settings.sparse_model_name)


@lru_cache
def get_reranker():
    return CrossEncoder(settings.reranker_model_name)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.qdrant_path))  # local mode: one process at a time
    atexit.register(client.close)  # close cleanly before Python shuts down
    return client


def create_collection(name: str, hybrid: bool = True):
    """Create the collection if missing (dense vector + optional BM25 sparse vector)."""
    client = get_qdrant_client()
    if client.collection_exists(name):
        return
    dim = len(get_embeddings().embed_query("dimension check"))
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()} if hybrid else None,
    )


def get_vector_store(mode: str = "hybrid", collection: str | None = None) -> QdrantVectorStore:
    """mode: "hybrid" (dense + BM25, fused by Qdrant), "dense" or "sparse"."""
    collection = collection or settings.qdrant_collection_name
    create_collection(collection)
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode(mode),
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )


def source_filter(source: str | None):
    """Qdrant filter to search inside one PDF (by file name)."""
    if not source:
        return None
    return models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))])


def delete_source(source: str, collection: str | None = None):
    """Remove all chunks of one source (used before re-ingesting it)."""
    collection = collection or settings.qdrant_collection_name
    client = get_qdrant_client()
    if client.collection_exists(collection):
        client.delete(collection, points_selector=models.FilterSelector(filter=source_filter(source)))


def reset_collection(collection: str | None = None):
    """Remove every point but keep the collection. (Deleting the whole collection is unreliable
    with local Qdrant on Windows: the files stay on disk and come back when it's recreated.)"""
    collection = collection or settings.qdrant_collection_name
    client = get_qdrant_client()
    if client.collection_exists(collection):
        client.delete(collection, points_selector=models.FilterSelector(filter=models.Filter()))
