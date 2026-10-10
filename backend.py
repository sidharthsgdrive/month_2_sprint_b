"""
RAG Notes — FastAPI backend.

Turns uploaded PDF notes into a searchable knowledge base and answers
questions about them with retrieval-augmented generation.

Pipeline
--------
    PDF upload
        -> text extraction (pypdf)
        -> chunking
        -> embedding (sentence-transformers)
        -> vector store (ChromaDB)

    User question
        -> query embedding
        -> semantic search over the vector store
        -> relevant chunks assembled into context
        -> context + question sent to an LLM (Groq)
        -> generated answer (+ cited source chunks) returned to the client
"""

import os
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "notes")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = (
    "You are a study assistant that answers questions using ONLY the provided "
    "context from the user's uploaded notes. If the answer is not present in "
    "the context, reply exactly: \"I don't know based on the provided notes.\" "
    "Be concise, and mention the source file (e.g. [lecture3.pdf]) when it is "
    "useful context for the user."
)

# --------------------------------------------------------------------------- #
# Lazily-initialised globals (embedding model, vector store, LLM client)
# --------------------------------------------------------------------------- #
embedding_model: Optional[SentenceTransformer] = None
chroma_client = None
collection = None

groq_client = None
if GROQ_API_KEY:
    from groq import Groq

    groq_client = Groq(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def init_models() -> None:
    """Load the embedding model and connect to the vector store (idempotent)."""
    global embedding_model, chroma_client, collection
    if embedding_model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()  # warm up on startup so the first request isn't slow
    yield


# --------------------------------------------------------------------------- #
# PDF -> text extraction
# --------------------------------------------------------------------------- #
def pdf_to_text(pdf_path: str) -> str:
    """Extract plain text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, breaking on paragraph boundaries
    where possible so each chunk stays semantically coherent."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) > chunk_size:
            # a single paragraph longer than chunk_size: slide a window over it
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                start += chunk_size - overlap
        else:
            current = para

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n\n{chunks[i]}")
        return overlapped

    return chunks


def retrieve(question: str, k: int):
    """Embed the question and run semantic search against the vector store."""
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []
    return docs, metas, dists


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="RAG Notes API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int


@app.get("/health")
def health():
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    init_models()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        try:
            text = pdf_to_text(tmp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Failed to extract text from PDF: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(400, "No extractable text found in this PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Document produced no chunks after processing.")

    embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()

    # Re-uploading the same filename replaces its previous chunks instead of
    # duplicating them.
    existing = collection.get(where={"source": file.filename})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = [f"{file.filename}-{i}-{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    return UploadResponse(filename=file.filename, chunks_indexed=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    if groq_client is None:
        raise HTTPException(500, "GROQ_API_KEY is not configured on the server.")

    init_models()

    if collection.count() == 0:
        raise HTTPException(400, "No documents have been indexed yet. Upload a PDF first.")

    k = request.top_k or TOP_K
    docs, metas, dists = retrieve(request.question, k)

    if not docs:
        return AskResponse(answer="I don't know based on the provided notes.", sources=[])

    context_blocks = []
    sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        context_blocks.append(f"[{meta['source']} - chunk {meta['chunk']}]\n{doc}")
        sources.append(
            SourceChunk(source=meta["source"], chunk_index=meta["chunk"], text=doc, distance=dist)
        )
    context = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer = completion.choices[0].message.content

    return AskResponse(answer=answer, sources=sources)


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    init_models()
    if collection.count() == 0:
        return []
    all_items = collection.get()
    counts = {}
    for meta in all_items["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [DocumentInfo(source=src, chunks=n) for src, n in counts.items()]


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    init_models()
    existing = collection.get(where={"source": filename})
    if not existing or not existing.get("ids"):
        raise HTTPException(404, f"No document named '{filename}' found.")
    collection.delete(ids=existing["ids"])
    return {"deleted": filename, "chunks_removed": len(existing["ids"])}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
