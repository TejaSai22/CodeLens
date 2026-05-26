"""
RepoRegistry: source of truth for which repositories are indexed.

Each indexed repository maps to its own ChromaDB collection. This registry
stores the human-facing metadata (display name, status, counts) that the
collection itself cannot hold, backed by SQLite for atomic status updates
(which the planned async indexing path will rely on).
"""

import os
import re
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from config.settings import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source(repo_input: str) -> str:
    """Produce a stable, canonical form of a repo input for hashing."""
    source = repo_input.strip()
    if source.startswith("http"):
        source = source.rstrip("/")
        if source.endswith(".git"):
            source = source[:-4]
        return source.lower()
    # Local path: resolve to an absolute, normalized path.
    return os.path.normpath(os.path.abspath(source))


def derive_repo_id(repo_input: str) -> str:
    """Deterministic short id derived from the normalized source."""
    normalized = normalize_source(repo_input)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def collection_name_for(repo_id: str) -> str:
    """ChromaDB collection name for a repo id (valid: [a-z0-9_], 3-512 chars)."""
    return f"repo_{repo_id}"


def derive_display_name(repo_input: str) -> str:
    """Human-friendly name: 'owner/repo' for URLs, basename for local paths."""
    source = repo_input.strip()
    if source.startswith("http"):
        source = source.rstrip("/")
        if source.endswith(".git"):
            source = source[:-4]
        parts = [p for p in source.split("/") if p]
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1] if parts else source
    return os.path.basename(os.path.normpath(source)) or source


class RepoRegistry:
    """SQLite-backed registry of indexed repositories."""

    def __init__(self, db_path: Optional[str] = None):
        persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(persist_dir, "codelens.db")
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repos (
                    repo_id         TEXT PRIMARY KEY,
                    display_name    TEXT NOT NULL,
                    source          TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    files_indexed   INTEGER NOT NULL DEFAULT 0,
                    chunks_created  INTEGER NOT NULL DEFAULT 0,
                    progress        TEXT,
                    error           TEXT,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
                """
            )
            # Migrate older databases that predate the progress column.
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(repos)").fetchall()}
            if "progress" not in cols:
                conn.execute("ALTER TABLE repos ADD COLUMN progress TEXT")

    def upsert(self, repo_input: str, status: str = "indexing") -> Dict[str, Any]:
        """Create or reset a registry row for a repo. Returns the row."""
        repo_id = derive_repo_id(repo_input)
        display_name = derive_display_name(repo_input)
        collection_name = collection_name_for(repo_id)
        now = _now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM repos WHERE repo_id = ?", (repo_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT INTO repos
                    (repo_id, display_name, source, collection_name, status,
                     files_indexed, chunks_created, progress, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, 0, 'queued', NULL, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    source=excluded.source,
                    collection_name=excluded.collection_name,
                    status=excluded.status,
                    files_indexed=0,
                    chunks_created=0,
                    progress='queued',
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (repo_id, display_name, normalize_source(repo_input),
                 collection_name, status, created_at, now),
            )
        return self.get(repo_id)

    def set_progress(self, repo_id: str, progress: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE repos SET progress=?, updated_at=? WHERE repo_id=?",
                (progress, _now(), repo_id),
            )

    def mark_ready(self, repo_id: str, files_indexed: int, chunks_created: int):
        with self._connect() as conn:
            conn.execute(
                """UPDATE repos SET status='ready', files_indexed=?,
                   chunks_created=?, progress=NULL, error=NULL, updated_at=? WHERE repo_id=?""",
                (files_indexed, chunks_created, _now(), repo_id),
            )

    def mark_error(self, repo_id: str, error: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE repos SET status='error', progress=NULL, error=?, updated_at=? WHERE repo_id=?",
                (error, _now(), repo_id),
            )

    def get(self, repo_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repos WHERE repo_id = ?", (repo_id,)
            ).fetchone()
            return dict(row) if row else None

    def list(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repos ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, repo_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
            return cur.rowcount > 0
