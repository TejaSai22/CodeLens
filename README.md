# CodeLens — RAG-based Codebase Q&A

CodeLens indexes a GitHub repository or local codebase and lets you ask natural-language questions about it, returning answers grounded in the actual code with file/line citations.

It uses **local embeddings** for retrieval (your code is embedded on your machine) and **Google Gemini** for answer generation. Retrieval is **hybrid** — dense vector search fused with BM25 keyword search via Reciprocal Rank Fusion — and source files are chunked along **syntax boundaries** (functions, classes, structs, interfaces…) using tree-sitter across all supported languages.

## Architecture

```
                ┌─────────────────┐
                │  React + Vite   │  ← primary UI (codelens-ui/)
                │  Streamlit      │  ← optional UI (src/app.py)
                └────────┬────────┘
                         │ HTTP (JSON)
                ┌────────▼────────┐
                │   FastAPI API   │  src/api/
                │  /index /query  │
                └────────┬────────┘
        ┌────────────────┼────────────────┐
        ▼                ▼                 ▼
   Ingestion         Retrieval         Generation
  repo_loader      vector_store        llm_client
   chunker         (ChromaDB +         (Gemini)
   embedder         BM25 + RRF)
```

| Concern | Implementation |
|---|---|
| Embeddings | `sentence-transformers` (local, default `all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, on disk) |
| Keyword search | `rank_bm25` (BM25Okapi), fused with dense results via RRF |
| Chunking | tree-sitter AST chunking for all supported languages (definition-level chunks + module-level gaps), sliding-window fallback for anything without a grammar |
| LLM | Google Gemini (default `gemini-2.5-flash`) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Tailwind (primary); Streamlit (optional) |

## Prerequisites

- Python 3.10+
- Node.js 18+ (for the React UI)
- A **Google Gemini API key** — https://aistudio.google.com/app/apikey

> Note: answer generation uses the Gemini cloud API, so a key and network access are required. Embeddings and retrieval run locally.

## Setup

### 1. Backend

```bash
# from the repo root
uv sync                 # or: python -m venv .venv && pip install -r requirements.txt

cp .env.example .env    # then edit .env and set GEMINI_API_KEY
```

Start the API:

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

The API serves:
- `GET  /health` — status + repo/chunk counts (unauthenticated)
- `POST /api/v1/rag/index` — start indexing a repo (`{"repo_input": "<github-url-or-local-path>"}`), returns immediately with status `indexing`
- `GET  /api/v1/rag/repos` — list indexed repos and their status
- `DELETE /api/v1/rag/repos/{repo_id}` — delete a repo's index
- `POST /api/v1/rag/query` — ask a question (`{"repo_id": "...", "query": "...", "conversation_history": [...], "language_filter": "python"}`)
- `POST /api/v1/rag/query/stream` — same, but streams the answer as Server-Sent Events

When `API_KEY` is set, all `/api/v1/rag/*` routes require an `X-API-Key` header.

### 2. Frontend (React)

```bash
cd codelens-ui
npm install
npm run dev             # opens http://localhost:5173
```

### Alternative: Streamlit UI

```bash
uv run streamlit run src/app.py
```

Both frontends talk to the same FastAPI backend on port 8000.

## Deployment (Docker)

The repo ships a backend image, an nginx-served frontend image, and a Compose
file that wires them together with a persistent volume for the vector store.

```bash
cp .env.example .env          # set GEMINI_API_KEY (and optionally API_KEY)
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000

Notes:
- The backend's first boot downloads the embedding model, so the container's
  healthcheck has a generous `start_period`.
- The frontend bakes its backend URL at build time via the `VITE_API_URL` build
  arg (default `http://localhost:8000`); set it for non-local deployments.
- Indexed data persists in the `codelens-data` volume.
- If `API_KEY` is set in `.env`, also pass `VITE_API_KEY` as a frontend build arg
  so the UI can authenticate.

## Configuration

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Gemini API key for generation |
| `API_KEY` | _(empty)_ | If set, `/api/v1/rag/*` requires a matching `X-API-Key` header |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB location |
| `MAX_CHUNK_SIZE` | `1500` | Max chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Sliding-window overlap (chars) |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `CORS_ORIGINS` | `localhost:5173,3000,8501` | Allowed frontend origins |

## How it works

**Indexing** (`POST /index`)
1. Clone the GitHub repo (shallow) or read the local path.
2. Filter to supported source files, skipping `node_modules`, `.git`, `venv`, `dist`, `build`, etc.
3. Chunk each file by AST node (functions, classes, structs, interfaces, …) via tree-sitter, plus block chunks for module-level code; files with no grammar use an overlapping line-window fallback.
4. Embed chunks locally and store them in ChromaDB; (re)build the BM25 keyword index.

**Querying** (`POST /query`)
1. Embed the question locally.
2. Retrieve candidates with **hybrid search**: dense cosine similarity + BM25, fused with Reciprocal Rank Fusion.
3. Send the question, retrieved snippets, and recent conversation history to Gemini.
4. Return the answer plus the source chunks (file path, line range, relevance) for citation.

## Supported file types

`.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`

All of the above get AST-aware chunking via tree-sitter (`tree-sitter-language-pack`). `.tsx` uses the dedicated TSX grammar. Any other extension falls back to sliding-window chunking.

## Testing

```bash
uv run pytest tests/test_chunker.py tests/test_vector_store.py tests/test_api.py
```

`tests/test_integration.py` exercises the local pipeline (loading, chunking, embedding, retrieval) and needs no API key. CI (`.github/workflows/ci.yml`) runs the unit, API, integration and eval-metric tests on every push and PR to `main`.

## Retrieval evaluation

Retrieval quality is measured, not assumed. The eval harness indexes a repo
into a throwaway store and scores a golden set of questions (mapping each to the
source file that should be retrieved) under both **dense-only** and **hybrid**
retrieval:

```bash
uv run python -m evals.run_eval            # evaluates this repo's src/
uv run python -m evals.run_eval --no-ast   # A/B: force sliding-window chunking
uv run python -m evals.run_eval --k 5 --repo /path/to/repo --golden evals/golden.json
```

It reports `recall@k`, `MRR` and `hit@k` for both dense-only and hybrid retrieval,
and the `--no-ast` flag lets you A/B the chunking strategy on the same corpus.

The bundled golden set is small (10 Python questions over `src/`), so treat the
margins as directional rather than definitive. On it, hybrid is at least as good
as dense, and AST chunking (with module-level gap-filling) reaches recall@5 / hit@5
parity with sliding-window while producing function-level chunks — so source
citations point to exact functions, and non-Python languages get real structural
chunks instead of arbitrary line windows. The metric math is covered by
`tests/test_eval_metrics.py`.

## Project structure

```
codelens/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app + CORS + /health
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routers/rag.py       # /index and /query endpoints
│   ├── ingestion/
│   │   ├── repo_loader.py       # Clone repos, traverse files
│   │   ├── chunker.py           # tree-sitter + sliding-window chunking
│   │   └── embedder.py          # Local sentence-transformers embeddings
│   ├── retrieval/
│   │   └── vector_store.py      # ChromaDB + BM25 + RRF hybrid search
│   ├── generation/
│   │   └── llm_client.py        # Google Gemini client
│   └── app.py                   # Optional Streamlit frontend
├── codelens-ui/                 # React + Vite + Tailwind frontend
├── config/settings.py           # Environment-driven configuration
├── tests/                       # Unit + integration tests
└── data/                        # ChromaDB storage (gitignored)
```

## Known limitations & roadmap

CodeLens supports **multiple repositories** — each is indexed into its own
collection and selectable from the sidebar. Indexing runs **asynchronously**:
`POST /index` returns immediately with status `indexing`, and clients poll
`GET /repos` to watch progress (cloning → chunking → embedding → ready). The
BM25 keyword index is built lazily and persisted per repo.

Planned improvements:

- Auth, rate limiting, and containerized deployment
- Expand the eval golden set (more languages, larger corpus) for stronger signal

## License

Free for personal and educational use.
