"""
Tests for the SQLite-backed RepoRegistry.
"""

import os
import tempfile
import shutil
import pytest

from src.retrieval.repo_registry import (
    RepoRegistry, derive_repo_id, derive_display_name,
    normalize_source, collection_name_for,
)


@pytest.fixture
def registry():
    temp_dir = tempfile.mkdtemp()
    reg = RepoRegistry(db_path=os.path.join(temp_dir, "test.db"))
    yield reg
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_repo_id_is_stable_and_normalized():
    a = derive_repo_id("https://github.com/psf/requests")
    b = derive_repo_id("https://github.com/psf/requests.git")
    c = derive_repo_id("https://github.com/psf/requests/")
    assert a == b == c
    assert derive_repo_id("https://github.com/psf/requests") != derive_repo_id("https://github.com/psf/black")


def test_display_name_derivation():
    assert derive_display_name("https://github.com/psf/requests") == "psf/requests"
    assert derive_display_name("https://github.com/psf/requests.git") == "psf/requests"


def test_collection_name_is_valid():
    name = collection_name_for(derive_repo_id("https://github.com/psf/requests"))
    assert name.startswith("repo_")
    assert all(ch.isalnum() or ch == "_" for ch in name)
    assert 3 <= len(name) <= 512


def test_upsert_creates_indexing_row(registry):
    repo = registry.upsert("https://github.com/psf/requests")
    assert repo["status"] == "indexing"
    assert repo["display_name"] == "psf/requests"
    assert repo["files_indexed"] == 0


def test_mark_ready_updates_counts(registry):
    repo = registry.upsert("https://github.com/psf/requests")
    registry.mark_ready(repo["repo_id"], files_indexed=10, chunks_created=42)
    updated = registry.get(repo["repo_id"])
    assert updated["status"] == "ready"
    assert updated["files_indexed"] == 10
    assert updated["chunks_created"] == 42


def test_reupsert_resets_counts_keeps_created_at(registry):
    repo = registry.upsert("https://github.com/psf/requests")
    registry.mark_ready(repo["repo_id"], 10, 42)
    created_at = registry.get(repo["repo_id"])["created_at"]

    again = registry.upsert("https://github.com/psf/requests")
    assert again["status"] == "indexing"
    assert again["files_indexed"] == 0
    assert again["created_at"] == created_at


def test_mark_error(registry):
    repo = registry.upsert("https://github.com/psf/requests")
    registry.mark_error(repo["repo_id"], "boom")
    updated = registry.get(repo["repo_id"])
    assert updated["status"] == "error"
    assert updated["error"] == "boom"


def test_list_and_delete(registry):
    r1 = registry.upsert("https://github.com/psf/requests")
    r2 = registry.upsert("https://github.com/psf/black")
    assert len(registry.list()) == 2

    assert registry.delete(r1["repo_id"]) is True
    assert registry.get(r1["repo_id"]) is None
    assert len(registry.list()) == 1
    assert registry.delete("nonexistent") is False
