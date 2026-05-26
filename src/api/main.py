from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from typing import Optional
from config.settings import settings
from src.api.routers import rag

# Configure structured logging
logger.add("logs/codelens_{time}.log", rotation="100 MB", retention="10 days")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Gate the API behind a shared key when one is configured.

    If settings.API_KEY is empty (the default), auth is disabled — convenient for
    local development and the test suite.
    """
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI(
    title="CodeLens API",
    description="RAG-based codebase Q&A system backend",
    version="1.0.0"
)

# CORS middleware: restrict to known frontend origins. Credentials are disabled
# because the API is currently stateless (no cookie/session auth); a wildcard
# origin combined with credentials is rejected by browsers anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (gated by the optional API key)
app.include_router(
    rag.router,
    prefix="/api/v1/rag",
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)

@app.get("/health")
def health_check():
    from src.retrieval.repo_registry import RepoRegistry
    repos = RepoRegistry().list()
    total_chunks = sum(r["chunks_created"] for r in repos)
    logger.info(f"Health check: {len(repos)} repos, {total_chunks} chunks")
    return {
        "status": "healthy",
        "repos": len(repos),
        "total_chunks": total_chunks,
        # Retained for backward compatibility with older clients.
        "indexed_chunks": total_chunks,
    }
