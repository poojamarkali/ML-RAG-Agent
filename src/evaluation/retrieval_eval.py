"""Retrieval evaluation -- same metrics as the "Evaluation of RAG chunking" notebook.

  Hit Rate @ k : did an expected page show up anywhere in the top-k results?
  MRR          : how high was it? (1/rank, averaged -- 1.0 means always rank 1)

golden_dataset.json is hand-written: question -> the PDF file + the page(s) that answer it.
Questions marked "answerable": false are off-topic; they're only used by llm_eval.py.
"""

import json
from pathlib import Path

from src.retrieval.search import search

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# The retrieval setups we compare -- (name, kwargs passed to search())
STRATEGIES = [
    ("dense", {"mode": "dense", "rerank_results": False}),
    ("bm25 (sparse)", {"mode": "sparse", "rerank_results": False}),
    ("hybrid", {"mode": "hybrid", "rerank_results": False}),
    ("hybrid + rerank", {"mode": "hybrid", "rerank_results": True}),
]


def load_dataset(answerable: bool = True) -> list[dict]:
    items = json.loads(DATASET_PATH.read_text())
    return [i for i in items if i.get("answerable", True) == answerable]


def evaluate(dataset: list[dict], k: int = 5, **search_kwargs) -> dict:
    results = []
    for item in dataset:
        docs = search(item["query"], k=k, **search_kwargs)
        found = [(d.metadata.get("source"), d.metadata.get("page")) for d in docs]
        expected = {(item["source"], page) for page in item["expected_pages"]}
        rank = next((i + 1 for i, hit in enumerate(found) if hit in expected), None)
        results.append({"query": item["query"], "expected_pages": item["expected_pages"],
                        "retrieved_pages": [page for _, page in found], "rank": rank})

    n = len(results) or 1
    return {
        "hit_rate": sum(r["rank"] is not None for r in results) / n,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n,
        "results": results,
    }


def compare_strategies(k: int = 5) -> list[dict]:
    dataset = load_dataset()
    rows = []
    for name, kwargs in STRATEGIES:
        scores = evaluate(dataset, k=k, **kwargs)
        rows.append({"strategy": name, "hit_rate": round(scores["hit_rate"], 3), "mrr": round(scores["mrr"], 3),
                     "results": scores["results"]})
    return rows


def print_report(rows: list[dict]):
    print(f"\n{'strategy':<22}{'hit rate':>10}{'MRR':>8}")
    for row in rows:
        print(f"{row['strategy']:<22}{row['hit_rate']:>10.0%}{row['mrr']:>8.3f}")
    best = max(rows, key=lambda r: r["mrr"])
    print(f"\nPer query ({best['strategy']}):")
    for r in best["results"]:
        status = f"HIT @ rank {r['rank']}" if r["rank"] else "MISS"
        print(f"  [{status}] {r['query']}  (expected p.{r['expected_pages']}, got {r['retrieved_pages']})")
