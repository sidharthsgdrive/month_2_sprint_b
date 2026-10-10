"""
RAG Notes — Streamlit frontend.

Lets the user upload PDF notes and chat with them. All the real work
(PDF -> markdown, chunking, embedding, vector search, LLM generation)
happens in backend.py; this file only talks to that API over HTTP.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Notes Q&A", page_icon="📚", layout="wide")


def backend_get(path: str, timeout: int = 30, **kwargs):
    return requests.get(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def backend_post(path: str, timeout: int = 120, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def backend_delete(path: str, timeout: int = 30, **kwargs):
    return requests.delete(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['source']}** · chunk {src['chunk_index']} · distance {src['distance']:.3f}")
            snippet = src["text"]
            st.caption(snippet[:500] + ("…" if len(snippet) > 500 else ""))


if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]

# --------------------------------------------------------------------------- #
# Sidebar: backend status, upload, document list
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 Notes Q&A")
    st.caption("Upload PDF notes, then ask questions about them.")

    try:
        health = backend_get("/health", timeout=5).json()
        if health.get("groq_configured"):
            st.success("Backend connected")
        else:
            st.warning("Backend connected, but GROQ_API_KEY isn't set on the server.")
    except requests.exceptions.RequestException:
        st.error(f"Can't reach backend at {BACKEND_URL}")

    st.divider()
    st.subheader("Upload notes")
    uploaded_file = st.file_uploader("PDF file", type=["pdf"])
    if uploaded_file and st.button("Index this PDF", use_container_width=True):
        with st.spinner(f"Converting and indexing {uploaded_file.name}... this can take a minute"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = backend_post("/upload", files=files, timeout=600)
                if resp.ok:
                    data = resp.json()
                    st.success(
                        f"Indexed {data['chunks_indexed']} chunks from {data['filename']}."
                    )
                else:
                    st.error(resp.json().get("detail", "Upload failed."))
            except requests.exceptions.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = backend_get("/documents", timeout=10).json()
        if not docs:
            st.caption("No documents indexed yet.")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{doc['source']}**  \n{doc['chunks']} chunks")
            if col2.button("🗑️", key=f"del-{doc['source']}"):
                backend_delete(f"/documents/{doc['source']}")
                st.rerun()
    except requests.exceptions.RequestException:
        st.caption("Could not load document list.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
st.header("Ask your notes")

if not st.session_state.messages:
    st.info("Upload a PDF in the sidebar, then ask a question about it below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about your uploaded notes...")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = []
            try:
                resp = backend_post("/ask", json={"question": question}, timeout=120)
                if resp.ok:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                else:
                    answer = resp.json().get("detail", "Something went wrong.")
            except requests.exceptions.RequestException as exc:
                answer = f"Could not reach backend: {exc}"

        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
