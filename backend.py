"""
RAG Notes — FastAPI backend.

PDF upload -> text extraction -> chunking -> embeddings -> ChromaDB.
Questions -> semantic retrieval -> Groq chat completions -> answer + sources.
"""

import os
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from urllib.parse import quote

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq SDK appends /chat/completions; do not add that path here.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # Hardcoded Groq API base URL
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "notes")
CHUNK_SIZE = max(100, int(os.getenv("CHUNK_SIZE", "1000")))
CHUNK_OVERLAP = max(0, min(int(os.getenv("CHUNK_OVERLAP", "150")), CHUNK_SIZE - 1))
TOP_K = max(1, int(os.getenv("TOP_K", "4")))

SYSTEM_PROMPT = (
    "You are a study assistant that answers questions using ONLY the provided "
    "context from the user's uploaded notes. If the answer is not present in "
    'the context, reply exactly: "I don\'t know based on the provided notes." '
    "Be concise, and mention the source file (e.g. [lecture3.pdf]) when useful."
)

embedding_model = None
chroma_client = None
collection = None
groq_client = None

if GROQ_API_KEY:
    from groq import Groq

    groq_client = Groq(api_key=GROQ_API_KEY)


def init_models() -> None:
    """Initialize embedding model and persistent Chroma collection once."""
    global embedding_model, chroma_client, collection
    if embedding_model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield


def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks without duplicating overlap twice."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = paragraph
        else:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            step = chunk_size - overlap
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += step

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for index in range(1, len(chunks)):
            # Only add overlap when the chunks do not already overlap.
            previous_tail = chunks[index - 1][-overlap:]
            if chunks[index].startswith(previous_tail):
                overlapped.append(chunks[index])
            else:
                overlapped.append(f"{previous_tail}\n\n{chunks[index]}")
        return overlapped
    return chunks


def retrieve(question: str, k: int):
    query_embedding = embedding_model.encode(question).tolist()
    result_count = min(k, collection.count())
    if result_count <= 0:
        return [], [], []
    results = collection.query(query_embeddings=[query_embedding], n_results=result_count)
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    return docs, metas, dists


app = FastAPI(title="RAG Notes API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


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
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    init_models()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        try:
            text = pdf_to_text(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        await file.close()

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in this PDF. It may be scanned/image-only.",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks after processing.")

    embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
    existing = collection.get(where={"source": filename})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = [f"{uuid.uuid4().hex}-{index}" for index in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": index} for index in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return UploadResponse(filename=filename, chunks_indexed=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")

    init_models()
    total = collection.count()
    if total == 0:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet. Upload a PDF first.")

    docs, metas, distances = retrieve(question, min(request.top_k or TOP_K, total))
    if not docs:
        return AskResponse(answer="I don't know based on the provided notes.", sources=[])

    context_blocks = []
    sources = []
    for doc, meta, distance in zip(docs, metas, distances):
        if not meta or "source" not in meta or "chunk" not in meta:
            continue
        context_blocks.append(f"[{meta['source']} - chunk {meta['chunk']}]\n{doc}")
        sources.append(
            SourceChunk(
                source=str(meta["source"]),
                chunk_index=int(meta["chunk"]),
                text=doc,
                distance=float(distance),
            )
        )

    if not context_blocks:
        return AskResponse(answer="I don't know based on the provided notes.", sources=[])

    user_prompt = f"Context:\n{'\n\n---\n\n'.join(context_blocks)}\n\nQuestion: {question}\n\nAnswer:"
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        # Keep the client-facing error actionable while details remain in server logs.
        print(f"Groq completion failed ({type(exc).__name__}): {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Groq API request failed. Verify GROQ_API_KEY and GROQ_MODEL. ({type(exc).__name__})",
        ) from exc

    answer = completion.choices[0].message.content or "I don't know based on the provided notes."
    return AskResponse(answer=answer, sources=sources)


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    init_models()
    all_items = collection.get(include=["metadatas"])
    counts = {}
    for meta in (all_items.get("metadatas") or []):
        if meta and meta.get("source"):
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [DocumentInfo(source=source, chunks=count) for source, count in sorted(counts.items())]


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    init_models()
    existing = collection.get(where={"source": filename}, include=["metadatas"])
    ids = existing.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail=f"No document named '{filename}' found.")
    collection.delete(ids=ids)
    return {"deleted": filename, "chunks_removed": len(ids)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
