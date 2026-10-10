# RAG Notes — ask questions about your PDF notes

Upload PDF notes, they get converted to markdown, chunked, embedded, and stored
in a vector database. Ask a question and the app retrieves the most relevant
chunks and asks an LLM to answer using only that context.

```
User Query
    ↓
Query Embedding
    ↓
Semantic Search  (ChromaDB)
    ↓
Relevant Documents / Chunks
    ↓
Context + User Query
    ↓
LLM  (Groq)
    ↓
Generated Response
```

## How it works

| Stage | Tool |
|---|---|
| PDF → text | [`pypdf`](https://pypdf.readthedocs.io/) |
| Chunking | Paragraph-aware splitter with overlap (`backend.py`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2` by default) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persisted to disk) |
| LLM | [Groq](https://groq.com/) (`llama-3.3-70b-versatile` by default) |
| Backend | FastAPI |
| Frontend | Streamlit |

### Files

```
├── backend.py                 # FastAPI app: upload/ingest, retrieval, LLM generation
├── requirements.txt           # Backend dependencies (chromadb, pypdf, groq, ...)
├── frontend/
│   ├── frontend.py            # Streamlit chat UI, talks to backend over HTTP
│   └── requirements.txt       # Frontend-only dependencies (streamlit, requests, dotenv)
├── .env.example
├── .gitignore
└── README.md
```

> **Why two `requirements.txt` files?** The backend's dependencies
> (`chromadb`, `sentence-transformers`...) pull in `torch` and other sizeable
> packages. The frontend doesn't need any of that — it only talks to the
> backend over HTTP. Keeping them separate means the frontend install stays
> fast and isn't coupled to changes in the RAG stack's dependencies.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com/keys).

> **Note on PDF extraction:** `pypdf` pulls plain text out of each page. It
> won't reconstruct tables or complex multi-column layouts as cleanly as a
> layout-aware tool would, but it's fast, dependency-light, and good enough
> for most lecture notes and text-heavy PDFs.

## 2. Run locally

In one terminal, start the backend:

```bash
uvicorn backend:app --reload --port 8000
```

In another terminal, install the frontend's (separate, lighter) dependencies and start it:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/frontend.py
```

Open the Streamlit URL it prints (usually `http://localhost:8501`), upload a
PDF from the sidebar, wait for it to be indexed, then ask a question in the
chat box.

## 3. API reference (backend.py)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check, reports whether `GROQ_API_KEY` is set |
| `POST` | `/upload` | `multipart/form-data` with a `file` field (PDF). Converts, chunks, embeds, and indexes it |
| `POST` | `/ask` | JSON `{"question": "...", "top_k": 4}`. Returns `{"answer": "...", "sources": [...]}` |
| `GET` | `/documents` | Lists indexed documents and their chunk counts |
| `DELETE` | `/documents/{filename}` | Removes a document's chunks from the index |

## 4. Deploying

### Backend → Railway

1. Push this project to a GitHub repo.
2. In Railway: **New Project → Deploy from GitHub repo**, select the repo.
3. Set the start command:
   ```
   uvicorn backend:app --host 0.0.0.0 --port $PORT
   ```
   (Railway sets `$PORT` automatically — `backend.py` also reads it if you
   run it directly with `python backend.py`.)
4. Add environment variables from `.env.example` under **Variables**
   (at minimum `GROQ_API_KEY`).
5. **Persistence:** Railway's filesystem is ephemeral by default, so the
   Chroma index (`chroma_db/`) will reset on redeploys/restarts. Attach a
   [Railway Volume](https://docs.railway.app/reference/volumes) mounted at
   the path you set for `CHROMA_DIR` to persist uploaded notes across deploys.
6. Once deployed, note the public URL Railway gives you
   (e.g. `https://your-app.up.railway.app`) — you'll need it for the frontend.

### Frontend → Streamlit Community Cloud (or a second Railway service)

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**, pick
   your repo/branch, and set **Main file path** to `frontend/frontend.py`.
   Streamlit Cloud automatically looks for a `requirements.txt` in the same
   directory as the main file, so it will pick up `frontend/requirements.txt`
   (the lightweight one) instead of the backend's.
2. In the app's **Settings → Secrets**, add:
   ```
   BACKEND_URL = "https://your-app.up.railway.app"
   ```
   (your Railway backend URL, no trailing slash).
3. Deploy. The install should now only pull `streamlit`, `requests`, and
   `python-dotenv` — a few seconds, not several minutes.

If using Railway for the frontend instead, set the service's root directory
to `frontend/` and the start command to:
```
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
```

## 5. Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required for `/ask` to work |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-hosted chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any `sentence-transformers` model |
| `CHROMA_DIR` | `chroma_db` | Where the vector store is persisted |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters per chunk / overlap |
| `TOP_K` | `4` | Chunks retrieved per question |
| `BACKEND_URL` | `http://localhost:8000` | Used by `frontend.py` only |

## Notes & possible extensions

- Re-uploading a file with the same name replaces its previous chunks rather
  than duplicating them.
- Answers are grounded strictly in retrieved context — the system prompt
  instructs the model to say it doesn't know rather than guess.
- Ideas to extend: multi-user auth, chat history persisted server-side,
  swapping Chroma for a hosted vector DB (Pinecone/Qdrant) for durability
  without Railway volumes, streaming responses, re-ranking retrieved chunks.
