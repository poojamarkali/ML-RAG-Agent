"""Loaders: READ a data source and turn it into Documents (text + metadata).

A loader only reads -- it doesn't chunk, embed or store anything (that's pipeline.py).
To support a new kind of source (Word files, web pages, another database...),
add a load_xxx() function here that returns list[Document]; the pipeline stays the same.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sqlalchemy import text

from src.database.schema import engine


def load_pdf(path: Path) -> list[Document]:
    """One Document per PDF page."""
    path = Path(path)
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        # keep metadata small: which file, which page (1-based, as shown in a PDF viewer)
        page.metadata = {"source": path.name, "page": page.metadata.get("page", 0) + 1}
    return [p for p in pages if p.page_content.strip()]


def load_sql(query: str, name: str) -> list[Document]:
    """One Document per database row, written as "column: value" lines.
    Use for tables with descriptive text you want to search by meaning
    (exact numbers are better answered by the agent's SQL tool)."""
    with engine.connect() as conn:
        rows = conn.execute(text(query)).mappings().all()
    return [
        Document(
            page_content="\n".join(f"{col}: {val}" for col, val in row.items()),
            metadata={"source": f"sql:{name}", "row": i + 1},
        )
        for i, row in enumerate(rows)
    ]
