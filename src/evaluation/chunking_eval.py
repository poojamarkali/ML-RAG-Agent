"""Chunking evaluation: which chunking strategy gives the best retrieval?

Each strategy chunks the same PDF into its own temporary Qdrant collection
("eval_<name>"), then the golden questions are run against each one with the
same search (hybrid + rerank). Higher hit rate / MRR = better chunks.

Use `pages=(start, end)` to test on part of a book -- needed for "agentic",
which makes one LLM call per page. Only questions inside that range are used.
"""

from config import settings
from src.evaluation.retrieval_eval import evaluate, load_dataset
from src.ingestion.chunking import agentic_chunks, fixed_chunks, semantic_chunks
from src.ingestion.loaders import load_pdf
from src.ingestion.pipeline import store
from src.retrieval.vector_store import reset_collection

# (name, function that turns pages into chunks) -- add variations here to try them
CANDIDATES = [
    ("fixed 400", lambda pages: fixed_chunks(pages, chunk_size=400, chunk_overlap=80)),
    ("fixed 600", lambda pages: fixed_chunks(pages, chunk_size=600, chunk_overlap=100)),
    ("fixed 1000", lambda pages: fixed_chunks(pages, chunk_size=1000, chunk_overlap=150)),
    ("semantic", semantic_chunks),
    ("agentic", agentic_chunks),
]


def compare_chunking(
    file_name: str = "Introduction to Machine Learning with Python.pdf",
    pages: tuple[int, int] | None = None,
    candidates: list[str] | None = None,
    k: int = 5,
) -> list[dict]:
    all_pages = load_pdf(settings.documents_dir / file_name)
    if pages:
        all_pages = [p for p in all_pages if pages[0] <= p.metadata["page"] <= pages[1]]
    dataset = [q for q in load_dataset() if q["source"] == file_name
               and (not pages or any(pages[0] <= p <= pages[1] for p in q["expected_pages"]))]

    rows = []
    for name, chunker in CANDIDATES:
        if candidates and name not in candidates:
            continue
        collection = "eval_" + name.replace(" ", "_")
        reset_collection(collection)
        chunks = chunker(all_pages)
        store(chunks, file_name, collection)
        scores = evaluate(dataset, k=k, collection=collection)
        rows.append({"strategy": name, "chunks": len(chunks),
                     "avg_chunk_chars": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
                     "hit_rate": round(scores["hit_rate"], 3), "mrr": round(scores["mrr"], 3),
                     "results": scores["results"]})
        reset_collection(collection)  # temporary -- don't keep eval data around
        print(f"  {name}: done")
    print(f"({len(dataset)} questions, {len(all_pages)} pages)")
    return rows


def print_report(rows: list[dict]):
    print(f"\n{'strategy':<14}{'chunks':>8}{'avg chars':>11}{'hit rate':>10}{'MRR':>8}")
    for r in rows:
        print(f"{r['strategy']:<14}{r['chunks']:>8}{r['avg_chunk_chars']:>11}{r['hit_rate']:>10.0%}{r['mrr']:>8.3f}")
