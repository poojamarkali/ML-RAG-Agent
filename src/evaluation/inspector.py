"""Admin tools to SEE what the RAG pipeline does (not shown to chat users).

  inspect_query()  -> one question, every step: hybrid candidates -> re-ranked -> prompt -> answer
  document_stats() -> per PDF: how many chunks, how big, which strategy
  page_chunks()    -> the actual chunks made from one page

Used by `run_server.py inspect / stats / chunks` and the /admin API endpoints.
"""

from collections import defaultdict

from qdrant_client import models

from config import settings
from src.agent.llm import get_llm
from src.retrieval.rag import RAG_PROMPT
from src.retrieval.search import format_docs, rerank, retrieve_candidates
from src.retrieval.vector_store import get_qdrant_client


def inspect_query(question: str, source: str | None = None, candidates: int = 20, k: int = 5,
                  with_answer: bool = True) -> dict:
    # Step 1: hybrid search (dense meaning + BM25 keywords) -> wide pool
    pool = retrieve_candidates(question, k=candidates, source=source)
    for i, doc in enumerate(pool):
        doc.metadata["candidate_rank"] = i + 1

    # Step 2: cross-encoder re-ranks the pool, keeps the best k
    top = rerank(question, pool, k)

    # Step 3: the exact prompt the LLM receives
    context = format_docs(top)
    messages = RAG_PROMPT.format_messages(context=context, question=question)

    return {
        "question": question,
        "step1_hybrid_candidates": [
            {"rank": d.metadata["candidate_rank"], "source": d.metadata.get("source"),
             "page": d.metadata.get("page"), "preview": d.page_content[:120]}
            for d in pool
        ],
        "step2_reranked": [
            {"rank": i + 1, "was_rank": d.metadata["candidate_rank"], "score": d.metadata["score"],
             "source": d.metadata.get("source"), "page": d.metadata.get("page"), "text": d.page_content}
            for i, d in enumerate(top)
        ],
        "step3_prompt": "\n\n".join(f"[{m.type}]\n{m.content}" for m in messages),
        "step4_answer": get_llm().invoke(messages).text if with_answer else None,
    }


def _all_chunks(source: str | None = None):
    """Every stored chunk's payload (optionally only one PDF)."""
    client = get_qdrant_client()
    flt = None
    if source:
        flt = models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))])
    offset = None
    while True:
        points, offset = client.scroll(settings.qdrant_collection_name, scroll_filter=flt, limit=500,
                                       offset=offset, with_payload=True, with_vectors=False)
        yield from (p.payload for p in points)
        if offset is None:
            break


def document_stats() -> list[dict]:
    by_source = defaultdict(list)
    for payload in _all_chunks():
        by_source[payload["metadata"]["source"]].append(payload)
    stats = []
    for source, chunks in sorted(by_source.items()):
        sizes = [len(c["page_content"]) for c in chunks]
        stats.append({
            "source": source,
            "chunks": len(chunks),
            "pages": len({c["metadata"].get("page") for c in chunks}),
            "strategy": chunks[0]["metadata"].get("chunk_method"),
            "avg_chars": sum(sizes) // len(sizes),
            "min_chars": min(sizes),
            "max_chars": max(sizes),
        })
    return stats


def page_chunks(source: str, page: int) -> list[str]:
    return [p["page_content"] for p in _all_chunks(source) if p["metadata"].get("page") == page]


def print_inspection(result: dict):
    print(f"\nQUESTION: {result['question']}\n")
    print("STEP 1 - hybrid search candidates (dense + BM25):")
    for c in result["step1_hybrid_candidates"]:
        print(f"  #{c['rank']:>2}  p.{c['page']:<4} {c['preview'][:80]!r}")
    print("\nSTEP 2 - after cross-encoder re-ranking (kept for the LLM):")
    for c in result["step2_reranked"]:
        print(f"  #{c['rank']}  (was #{c['was_rank']:>2})  score {c['score']:>7.3f}  p.{c['page']}  {c['text'][:70]!r}")
    print("\nSTEP 3 - prompt sent to the LLM:\n")
    print(result["step3_prompt"])
    if result["step4_answer"]:
        print("\nSTEP 4 - answer:\n")
        print(result["step4_answer"])
