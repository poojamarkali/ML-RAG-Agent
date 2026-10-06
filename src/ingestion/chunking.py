"""Chunking strategies -- one function per notebook experiment.

  fixed     -> RecursiveCharacterTextSplitter (baseline used in the evaluation notebook)
  semantic  -> SemanticChunker, splits where the meaning shifts (semantic_chunking notebook)
  agentic   -> the LLM splits each page into self-contained chunks (agentic_chunking notebook)

Every function takes the PDF pages (list[Document]) and returns chunks (list[Document]).
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# fixed
# ---------------------------------------------------------------------------
def fixed_chunks(pages: list[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# semantic
# ---------------------------------------------------------------------------
def semantic_chunks(pages: list[Document]) -> list[Document]:
    from langchain_experimental.text_splitter import SemanticChunker

    from src.retrieval.vector_store import get_embeddings

    splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,  # higher = fewer, bigger chunks
    )
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# agentic (1 LLM call per page)
# ---------------------------------------------------------------------------
AGENTIC_CHUNKING_PROMPT = """\
You are a document chunking assistant. Split the text below into self-contained chunks. Each chunk should:
- Cover ONE clear idea/topic (don't mix unrelated topics in one chunk)
- Be understandable on its own, without needing the surrounding text
- Preserve the original wording (do not summarize or rephrase)
- Be roughly 50-200 words

Text to chunk:
{text}
"""


class ChunkList(BaseModel):
    chunks: list[str] = Field(description="The chunk texts, in document order")


def agentic_chunks(pages: list[Document]) -> list[Document]:
    from src.agent.llm import get_llm

    # Structured output (pydantic) instead of parsing JSON text by hand
    chain = ChatPromptTemplate.from_template(AGENTIC_CHUNKING_PROMPT) | get_llm().with_structured_output(ChunkList)

    chunks = []
    for page in pages:
        try:
            texts = chain.invoke({"text": page.page_content}).chunks
        except Exception as e:  # bad LLM output -> keep the whole page as one chunk
            print(f"  [warn] agentic chunking failed on page {page.metadata.get('page')}: {e}")
            texts = [page.page_content]
        chunks += [Document(page_content=t, metadata=dict(page.metadata)) for t in texts if t.strip()]
    return chunks


# ---------------------------------------------------------------------------
STRATEGIES = {
    "fixed": fixed_chunks,
    "semantic": semantic_chunks,
    "agentic": agentic_chunks,
}


def chunk_pages(pages: list[Document], strategy: str = "fixed") -> list[Document]:
    chunks = STRATEGIES[strategy](pages)
    for chunk in chunks:
        chunk.metadata["chunk_method"] = strategy
    return chunks
