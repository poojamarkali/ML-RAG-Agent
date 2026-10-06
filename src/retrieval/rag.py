"""Plain RAG: retrieve -> put chunks in the prompt -> LLM answers.
Same as `answer_question` in the combined chunking notebook (no agent, no tools)."""

from langchain_core.prompts import ChatPromptTemplate

from src.agent.llm import get_llm
from src.retrieval.search import format_docs, search

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about machine learning documents. Use ONLY the context below. "
     "Cite sources as [file, page]. If the context does not contain the answer, say so -- do not guess.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def answer_question(question: str, k: int = 5, source: str | None = None) -> dict:
    docs = search(question, k=k, source=source)
    context = format_docs(docs)
    response = get_llm().invoke(RAG_PROMPT.format_messages(context=context, question=question))
    return {"answer": response.text, "context": context, "sources": docs}
