"""Streamlit UI -- a thin client that only calls the FastAPI backend.
Run the api first, then: uv run python run_server.py ui"""

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
api = httpx.Client(base_url=API_URL, timeout=600)

st.set_page_config(page_title="ML RAG Agent", layout="wide")
st.title("ML RAG Agent")

tab_chat, tab_search, tab_docs, tab_sql, tab_eval = st.tabs(["Chat", "Search", "Documents", "SQL", "Evaluation"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    if question := st.chat_input("Ask about your ML documents or benchmark data"):
        st.chat_message("user").markdown(question)
        with st.spinner("Thinking..."):
            res = api.post("/chat", json={"question": question, "history": st.session_state.messages}).json()
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            st.caption("from cache" if res["cached"] else f"tools: {[t['tool'] for t in res['tools_used']]}")
        st.session_state.messages += [{"role": "user", "content": question},
                                      {"role": "assistant", "content": res["answer"]}]

with tab_search:
    query = st.text_input("Query", "What is the difference between Lasso and Ridge regression?")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Mode", ["hybrid", "dense", "sparse"])
    k = c2.number_input("Top k", 1, 20, 5)
    rerank = c3.checkbox("Re-rank", True)
    if st.button("Search"):
        for chunk in api.post("/search", json={"query": query, "k": k, "mode": mode, "rerank": rerank}).json():
            meta = chunk["metadata"]
            st.markdown(f"**{meta.get('source')}** — page {meta.get('page')} — score {meta.get('score', '-')}")
            st.info(chunk["text"])

with tab_docs:
    st.write(api.get("/documents").json())
    pdf = st.file_uploader("Add a PDF", type="pdf")
    strategy = st.selectbox("Chunking strategy", api.get("/chunking/strategies").json())
    if pdf and st.button("Ingest"):
        with st.spinner("Chunking and indexing..."):
            res = api.post("/documents", files={"file": (pdf.name, pdf.getvalue())}, data={"strategy": strategy})
        st.success(res.json())

with tab_sql:
    st.code(api.get("/sql/schema").json()["schema"])
    sql = st.text_area("SQL", "SELECT model_name, metric_value, train_time_sec FROM model_benchmarks ORDER BY metric_value DESC")
    if st.button("Run"):
        res = api.post("/sql", json={"sql": sql})
        st.dataframe(pd.DataFrame(res.json())) if res.is_success else st.error(res.json()["detail"])

with tab_eval:
    if st.button("Run retrieval evaluation"):
        rows = api.get("/eval/retrieval").json()
        st.table(pd.DataFrame(rows)[["strategy", "hit_rate", "mrr"]])
    if st.button("Run LLM evaluation (calls the LLM)"):
        st.dataframe(pd.DataFrame(api.get("/eval/llm").json()))
