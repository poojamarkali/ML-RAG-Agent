"""Pipeline: loader -> chunking -> Qdrant.

  loaders.py  = how to READ a source (PDF, SQL rows, ...)
  chunking.py = how to SPLIT the text
  pipeline.py = runs the steps in order and stores the result

Which chunking strategy each PDF uses is set in config/chunking.json.
"""

import json
from pathlib import Path

from langchain_core.documents import Document

from config import settings
from src.ingestion.chunking import chunk_pages
from src.ingestion.loaders import load_pdf, load_sql
from src.retrieval.vector_store import delete_source, get_vector_store


def store(chunks: list[Document], source: str, collection: str | None = None):
    """Replace everything stored for `source` with these chunks."""
    delete_source(source, collection)
    get_vector_store(collection=collection).add_documents(chunks)


def strategy_for(file_name: str) -> str:
    config = json.loads(settings.chunking_config_file.read_text())
    return config.get("files", {}).get(file_name, config["default"])


def ingest_pdf(path: Path, strategy: str | None = None) -> int:
    path = Path(path)
    strategy = strategy or strategy_for(path.name)
    pages = load_pdf(path)
    chunks = chunk_pages(pages, strategy)
    store(chunks, path.name)
    print(f"{path.name}: {len(pages)} pages -> {len(chunks)} chunks ({strategy})")
    return len(chunks)


def ingest_folder(folder: Path | None = None, strategy: str | None = None) -> dict[str, int]:
    """Every PDF in the folder (default: data/documents)."""
    folder = Path(folder or settings.documents_dir)
    return {pdf.name: ingest_pdf(pdf, strategy) for pdf in sorted(folder.glob("*.pdf"))}


def ingest_sql(query: str, name: str) -> int:
    """Rows from the SQL database -> searchable chunks (each row is already a small chunk)."""
    docs = load_sql(query, name)
    store(docs, f"sql:{name}")
    print(f"sql:{name}: {len(docs)} rows")
    return len(docs)


def list_documents() -> list[str]:
    return sorted(p.name for p in settings.documents_dir.glob("*.pdf"))
