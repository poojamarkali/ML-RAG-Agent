"""Chat app for users -- a thin client that only calls the FastAPI backend.
(Inspection and evaluation are admin-only: see /admin in http://localhost:8000/docs.)
Run the api first, then: uv run python run_server.py ui"""

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
api = httpx.Client(base_url=API_URL, timeout=300)

st.set_page_config(page_title="ML Study Assistant", page_icon="📘", layout="centered")


@st.cache_data(ttl=60)
def get_json(path: str):
    return api.get(path).json()


try:
    documents = get_json("/documents")
    suggestions = [q for questions in get_json("/suggestions").values() for q in questions]
except httpx.HTTPError:
    st.error(f"Can't reach the backend at {API_URL}. Start it with:  uv run python run_server.py api")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources", "sql"}


def ask_suggestion(q: str):
    # runs before the page re-draws, so the clicked question is known from the top of the script
    st.session_state.pending = q


# ---- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Documents")
    for doc in documents:
        st.caption(doc)
    ALL = "All data (documents + database)"
    # All data -> the agent (/chat): PDFs and the SQL database. One PDF -> documents-only RAG (/ask).
    source = st.selectbox("Answer from", [ALL] + documents)
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if suggestions:
        st.subheader("Try asking")
        for i, q in enumerate(suggestions[:8]):
            st.button(q, key=f"side_{i}", use_container_width=True, on_click=ask_suggestion, args=(q,))

# ---- chat ---------------------------------------------------------------------
st.title("📘 ML Study Assistant")
st.caption("Answers only from your ML documents and benchmark database, with sources.")


def show_sources(sources: list[dict], sql: list[dict]):
    # Only passages the re-ranker judged relevant (score > 0) -- an off-topic question shows none
    sources = [s for s in sources if (s.get("score") or 0) > 0]
    if sources:
        with st.expander(f"📄 Document sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['source']} — page {s['page']}**")
                st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
    sql = [q for q in sql if q.get("rows")]  # only queries that actually returned data
    if sql:
        with st.expander(f"🗄️ Database queries ({len(sql)})"):
            for q in sql:
                st.code(q["sql"], language="sql")
                st.dataframe(pd.DataFrame(q["rows"]), hide_index=True)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources", []), m.get("sql", []))

question = st.chat_input("Ask about ML concepts or model benchmarks")
question = question or st.session_state.pop("pending", None)

# Suggested questions as buttons while the chat is empty
if not st.session_state.messages and not question and suggestions:
    st.markdown("##### Suggested questions")
    cols = st.columns(2)
    for i, q in enumerate(suggestions[:6]):
        cols[i % 2].button(q, key=f"main_{i}", use_container_width=True, on_click=ask_suggestion, args=(q,))

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents and database..."):
            try:
                if source == ALL:
                    res = api.post("/chat", json={"question": question})
                else:
                    res = api.post("/ask", json={"question": question, "source": source})
                res.raise_for_status()
                result = res.json()
            except httpx.HTTPError as e:
                result = {"answer": f"Sorry, something went wrong: {e}"}
        sources, sql = result.get("sources", []), result.get("sql", [])
        st.markdown(result["answer"])
        show_sources(sources, sql)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": sources, "sql": sql})
    st.rerun()
