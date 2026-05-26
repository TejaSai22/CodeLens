import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from typing import List

from src.ingestion.repo_loader import RepoLoader
from src.ingestion.chunker import CodeChunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store_manager import VectorStoreManager
from src.retrieval.repo_registry import RepoRegistry, derive_repo_id
from src.generation.llm_client import LLMClient
from src.api.schemas import (
    IndexRequest, IndexStartedResponse, QueryRequest, QueryResponse,
    SourceChunk, RepoInfo, DeleteResponse,
)

router = APIRouter()

# Shared, stateless singletons (expensive to construct).
embedder = Embedder()
chunker = CodeChunker()
llm_client = LLMClient()

# Per-repo state.
registry = RepoRegistry()
store_manager = VectorStoreManager()

# Single worker: indexing jobs run off the request thread but are serialized,
# so concurrent jobs never share the embedding model. Jobs report progress and
# their terminal state through the registry.
_index_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")


def _run_index_job(repo_id: str, repo_input: str):
    """Clone/read, chunk, embed and store a repository. Runs in a worker thread."""
    loader = None
    try:
        registry.set_progress(repo_id, "preparing")
        store_manager.drop(repo_id)          # start from a clean collection
        store = store_manager.get(repo_id)

        if repo_input.startswith("http"):
            registry.set_progress(repo_id, "cloning repository")
            loader = RepoLoader(repo_url=repo_input)
        else:
            loader = RepoLoader(repo_path=repo_input)

        files = loader.get_files()
        if not files:
            registry.mark_error(repo_id, "No supported files found in repository")
            return

        registry.set_progress(repo_id, f"chunking {len(files)} files")
        all_chunks = []
        for file_data in files:
            all_chunks.extend(chunker.chunk_file(file_data))

        if not all_chunks:
            registry.mark_error(repo_id, "No code chunks generated")
            return

        registry.set_progress(repo_id, f"embedding {len(all_chunks)} chunks")
        embeddings = embedder.embed([chunk.content for chunk in all_chunks])

        registry.set_progress(repo_id, "storing")
        store.add_chunks(all_chunks, embeddings)

        registry.mark_ready(repo_id, files_indexed=len(files), chunks_created=len(all_chunks))
        logger.success(f"Indexed {repo_id}: {len(files)} files, {len(all_chunks)} chunks")

    except Exception as e:
        logger.exception(f"Indexing job failed for {repo_id}: {str(e)}")
        registry.mark_error(repo_id, str(e))
    finally:
        if loader is not None:
            loader.cleanup()


@router.post("/index", response_model=IndexStartedResponse, status_code=202)
def index_repository(request: IndexRequest):
    """Start indexing a repository in the background. Returns immediately; poll
    GET /repos to watch the status transition from 'indexing' to 'ready'."""
    logger.info(f"Received index request for: {request.repo_input}")
    repo_id = derive_repo_id(request.repo_input)

    existing = registry.get(repo_id)
    if existing and existing["status"] == "indexing":
        raise HTTPException(status_code=409, detail="This repository is already being indexed.")

    repo = registry.upsert(request.repo_input, status="indexing")
    _index_executor.submit(_run_index_job, repo_id, request.repo_input)

    return IndexStartedResponse(
        success=True,
        message="Indexing started.",
        repo_id=repo_id,
        display_name=repo["display_name"],
        status="indexing",
    )


def _retrieve_for_query(request: QueryRequest):
    """Validate the repo and run hybrid retrieval. Returns (retrieved_chunks,
    sources). Raises HTTPException for unknown/not-ready repos so both the JSON
    and streaming endpoints fail before any response body is sent."""
    repo = registry.get(request.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found. Index it first.")
    if repo["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Repository is not ready (status: {repo['status']}).")

    store = store_manager.get(request.repo_id)
    query_embedding = embedder.embed_single(request.query)
    retrieved_chunks = store.hybrid_query(
        query_text=request.query,
        query_embedding=query_embedding,
        language_filter=request.language_filter,
    )

    sources = [
        SourceChunk(
            file_path=chunk["metadata"].get("file_path", "unknown"),
            start_line=chunk["metadata"].get("start_line", 0),
            end_line=chunk["metadata"].get("end_line", 0),
            similarity=chunk.get("similarity", 0.0),
            chunk_type=chunk["metadata"].get("chunk_type", "code"),
            name=chunk["metadata"].get("name", ""),
            content=chunk["content"],
            language=chunk["metadata"].get("language", ""),
            retrieval=chunk.get("retrieval"),
        )
        for chunk in retrieved_chunks
    ]
    return retrieved_chunks, sources


@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Query a repository and return the full answer as a single JSON response."""
    logger.info(f"Query on repo {request.repo_id}: '{request.query}'")
    try:
        retrieved_chunks, sources = _retrieve_for_query(request)
        if not retrieved_chunks:
            return QueryResponse(answer="No relevant code chunks found.", sources=[])

        answer = llm_client.generate_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=request.conversation_history or [],
        )
        return QueryResponse(answer=answer, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Exception during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
def handle_query_stream(request: QueryRequest):
    """Query a repository and stream the answer as Server-Sent Events.

    Emits newline-delimited `data:` frames, each a JSON object:
      {"type": "sources", "sources": [...]}  once, first
      {"type": "token", "text": "..."}       repeatedly as the answer streams
      {"type": "error", "message": "..."}    if generation fails mid-stream
      {"type": "done"}                        terminal
    """
    logger.info(f"Streaming query on repo {request.repo_id}: '{request.query}'")
    # Validation/retrieval happen up front so we can still return 404/400 before
    # committing to a 200 streaming response.
    retrieved_chunks, sources = _retrieve_for_query(request)

    def event_stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        yield sse({"type": "sources", "sources": [s.model_dump() for s in sources]})

        if not retrieved_chunks:
            yield sse({"type": "token", "text": "No relevant code chunks found."})
            yield sse({"type": "done"})
            return

        try:
            for piece in llm_client.generate_answer_stream(
                query=request.query,
                retrieved_chunks=retrieved_chunks,
                conversation_history=request.conversation_history or [],
            ):
                yield sse({"type": "token", "text": piece})
        except Exception as e:
            logger.exception(f"Streaming generation failed: {str(e)}")
            yield sse({"type": "error", "message": str(e)})
            return

        yield sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/repos", response_model=List[RepoInfo])
def list_repos():
    """List all indexed repositories."""
    return [RepoInfo(**row) for row in registry.list()]


@router.delete("/repos/{repo_id}", response_model=DeleteResponse)
def delete_repo(repo_id: str):
    """Delete a repository's index and registry entry."""
    repo = registry.get(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    store_manager.drop(repo_id)
    registry.delete(repo_id)
    logger.info(f"Deleted repo {repo_id} ({repo['display_name']})")
    return DeleteResponse(success=True, repo_id=repo_id)
