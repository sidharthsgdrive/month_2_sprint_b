"""
RAG Notes — Streamlit frontend.

The frontend uploads PDFs and calls the FastAPI backend for indexing,
document management, retrieval, and answer generation.
"""

import os
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

st.set_page_config(page_title="Notes Q&A", page_icon="📚", layout="wide")


def backend_get(path: str, timeout: int = 30, **kwargs):
    return requests.get(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def backend_post(path: str, timeout: int = REQUEST_TIMEOUT, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def backend_delete(path: str, timeout: int = 30, **kwargs):
    return requests.delete(f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)


def response_detail(resp: requests.Response) -> str:
    """Extract a useful error message from JSON or plain-text responses."""
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            return str(payload.get("detail", payload.get("message", payload)))
        return str(payload)
    except ValueError:
        return resp.text[:1000] or f"HTTP {resp.status_code}"


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            if not isinstance(src, dict):
                st.write(str(src))
                continue
            source = src.get("source", "Unknown source")
            chunk_index = src.get("chunk_index", "?")
            distance = src.get("distance")
            if isinstance(distance, (int, float)):
                st.markdown(f"**{source}** · chunk {chunk_index} · distance {distance:.3f}")
            else:
                st.markdown(f"**{source}** · chunk {chunk_index}")
            snippet = str(src.get("text", ""))
            st.caption(snippet[:500] + ("…" if len(snippet) > 500 else ""))


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("📚 Notes Q&A")
    st.caption("Upload PDF notes, then ask questions about them.")

    try:
        health_resp = backend_get("/health", timeout=5)
        health_resp.raise_for_status()
        health = health_resp.json()
        if health.get("groq_configured"):
            st.success(f"Backend connected · model: {health.get('model', 'configured')}")
        else:
            st.warning("Backend connected, but GROQ_API_KEY isn't configured on the server.")
    except (requests.exceptions.RequestException, ValueError) as exc:
        st.error(f"Can't reach backend at {BACKEND_URL}: {exc}")

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
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}.")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {response_detail(resp)}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Upload failed: {exc}")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Unexpected upload response: {exc}")

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs_resp = backend_get("/documents", timeout=15)
        docs_resp.raise_for_status()
        docs = docs_resp.json()

        # Expected API response is a list of {"source": ..., "chunks": ...}.
        # Accept a wrapper for compatibility, but show unexpected payloads clearly.
        if isinstance(docs, dict) and isinstance(docs.get("documents"), list):
            docs = docs["documents"]

        if not isinstance(docs, list):
            st.warning("Unexpected response format from /documents.")
            st.code(str(docs)[:1000])
        elif not docs:
            st.caption("No documents indexed yet.")
        else:
            for index, doc in enumerate(docs):
                if not isinstance(doc, dict):
                    st.warning(f"Skipping unexpected document entry: {doc}")
                    continue

                source = doc.get("source")
                chunks = doc.get("chunks")
                if not isinstance(source, str) or chunks is None:
                    st.warning(f"Invalid document record: {doc}")
                    continue

                col1, col2 = st.columns([4, 1])
                col1.write(f"**{source}**  \n{chunks} chunks")
                if col2.button("🗑️", key=f"del-{source}-{index}", help=f"Delete {source}"):
                    try:
                        delete_resp = backend_delete(
                            f"/documents/{quote(source, safe='')}",
                            timeout=30,
                        )
                        if delete_resp.ok:
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {response_detail(delete_resp)}")
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Delete failed: {exc}")

    except requests.exceptions.RequestException as exc:
        st.caption(f"Could not load document list: {exc}")
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid /documents response: {exc}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


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
        with st.spinner("Searching your notes and generating an answer..."):
            sources = []
            try:
                resp = backend_post("/ask", json={"question": question}, timeout=REQUEST_TIMEOUT)
                if resp.ok:
                    data = resp.json()
                    answer = str(data.get("answer", "No answer returned."))
                    sources = data.get("sources", [])
                else:
                    answer = f"Request failed (HTTP {resp.status_code}): {response_detail(resp)}"
            except requests.exceptions.RequestException as exc:
                answer = f"Could not reach backend: {exc}"
            except (ValueError, TypeError) as exc:
                answer = f"Unexpected response from backend: {exc}"

        st.markdown(answer)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
