# RAG Notes — ask questions about your PDF notes

Upload PDF notes, they get extracted to text, chunked, embedded, and stored
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
├── backend.py        # FastAPI app: upload/ingest, retrieval, LLM generation
├── frontend.py        # Streamlit chat UI, talks to backend over HTTP
├── requirements.txt    # All Python dependencies (backend + frontend)
├── .env.example       # Copy to .env and fill in
├── .gitignore
└── README.md
```

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
> layout-aware OCR tool would, but it's fast, lightweight, and good enough
> for most lecture notes and text-heavy PDFs. If a PDF is scanned/image-only
> (no selectable text), extraction will return nothing — `/upload` will
> report that clearly rather than silently indexing an empty document.

## 2. Run locally

In one terminal, start the backend:

```bash
uvicorn backend:app --reload --port 8000
```

In another terminal, start the frontend:

```bash
streamlit run frontend.py
```

Open the Streamlit URL it prints (usually `http://localhost:8501`), upload a
PDF from the sidebar, wait for it to be indexed, then ask a question in the
chat box.

## 3. API reference (backend.py)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check, reports whether `GROQ_API_KEY` is set |
| `POST` | `/upload` | `multipart/form-data` with a `file` field (PDF). Extracts, chunks, embeds, and indexes it |
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

1. Deploy `frontend.py` the same way (new Railway service, or
   [share.streamlit.io](https://share.streamlit.io)).
2. Set `BACKEND_URL` (in Streamlit Cloud: **Settings → Secrets**) to your
   backend's public URL from the step above, e.g.:
   ```
   BACKEND_URL = "https://your-app.up.railway.app"
   ```
3. If using Railway for the frontend too, set the start command:
   ```
   streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
   ```

> **Heads up:** this project uses a single `requirements.txt` for both
> services, which includes `sentence-transformers` (and therefore `torch`).
> The frontend doesn't actually need it — it only talks to the backend over
> HTTP — so the Streamlit install will take noticeably longer and pull more
> than it strictly needs. That's the tradeoff for keeping one shared
> requirements file; if install time on Streamlit Cloud becomes a problem,
> the fix is to give the frontend its own minimal `requirements.txt`
> (`streamlit`, `requests`, `python-dotenv`) in a separate folder or repo.

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
- Scanned/image-only PDFs won't extract any text with `pypdf`; adding OCR
  (e.g. `pytesseract`) is the natural next step if you need to support those.
- Ideas to extend: multi-user auth, chat history persisted server-side,
  swapping Chroma for a hosted vector DB (Pinecone/Qdrant) for durability
  without Railway volumes, streaming responses, re-ranking retrieved chunks.
