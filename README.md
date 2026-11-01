# Nova AI Virtual Assistant

Flask serves the chat interface on port 5000. FastAPI provides chat, document indexing, and database endpoints on port 8000.

## Setup

Use Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` if you do not already have one. Set `GEMINI_API_KEY` to your Gemini API key and set `FLASK_SECRET_KEY` to a random value. Generate one with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Keep `.env` private. It is ignored by Git.

Set `REDIS_URL` to a Redis-compatible server. Message history remains in SQLite; Redis caches recent conversation transcripts for 120 seconds and invalidates them on writes. Redis also caches repeated document retrieval results for 300 seconds, with index timestamps in each cache key so new uploads do not reuse old results. If Redis is unavailable, both features fall back to the normal database/search path.

## Run locally

Open two PowerShell terminals in the project folder and activate the virtual environment in each.

Terminal 1:

```powershell
uvicorn app:app --reload --port 8000
```

Terminal 2:

```powershell
python flask_app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The FastAPI status appears in the interface; if it is offline, start or restart the backend and refresh the page.

## Documents

Use the **+** button to upload up to 5 text-based PDFs at once (10 MB per file, 30 MB combined). The original upload bytes are processed in memory and are not saved as PDF files. Search combines BM25 keyword matching with FAISS semantic matching using reciprocal-rank fusion, keeps PDF/page metadata and extracted text in `vectorstore/chunks.jsonl` for source citations, and incrementally adds later uploads to the existing index. The FAISS index and extracted chunks still use local disk space, but the original PDFs are not retained by new uploads. Scanned PDFs without selectable text are not indexed because OCR is not configured. The embedding model loads on the first indexing or search request.

For an evidence-backed task workflow, ask Nova AI to **create an action plan**, **extract action items**, or **make a checklist** from the PDFs. It returns a cited checklist, saves it in the tasks table, and displays saved plans in the sidebar. Comparison requests retrieve a wider, source-diverse set of passages. These document actions do not carry out external operations such as sending messages or changing calendars.

The first upload after this version is installed migrates the previous plain-text FAISS index to page-aware metadata and incrementally indexes the new batch. Rebuilding with `python rag.py` indexes PDFs currently present under `documents/`; in-memory uploads that are represented only in the vector index cannot be recovered after the index is deleted. Keep your original PDFs elsewhere if you may need to rebuild or re-upload them. PDFs already in `documents/` are left untouched.

Use **My PDFs** in the sidebar to remove an uploaded PDF from the document list and search index. Use the **×** beside a conversation to delete that conversation and its messages. PDFs already present on disk from before the in-memory upload change are left in place unless their saved document record points to them and you delete that entry.
