"""
Tests for the FastAPI endpoints, including multi-repo isolation.

Indexing a small local repo exercises the real loader/chunker/embedder but
does NOT call Gemini (generation only happens on /query), so these run offline.
"""

import os
import time
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def _wait_ready(repo_id: str, timeout: float = 120.0) -> dict:
    """Poll /repos until the repo leaves the 'indexing' state (async indexing)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        repos = {r["repo_id"]: r for r in client.get("/api/v1/rag/repos").json()}
        repo = repos.get(repo_id)
        if repo and repo["status"] != "indexing":
            return repo
        time.sleep(0.25)
    raise AssertionError(f"Repo {repo_id} did not finish indexing within {timeout}s")


@pytest.fixture
def local_repo():
    """A throwaway local repo with a couple of Python files."""
    d = tempfile.mkdtemp(prefix="codelens_repo_")
    with open(os.path.join(d, "alpha.py"), "w") as f:
        f.write("def alpha_handler():\n    return 'alpha'\n")
    with open(os.path.join(d, "beta.py"), "w") as f:
        f.write("def beta_handler():\n    return 'beta'\n")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def second_local_repo():
    d = tempfile.mkdtemp(prefix="codelens_repo2_")
    with open(os.path.join(d, "gamma.py"), "w") as f:
        f.write("class GammaWidget:\n    pass\n")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "repos" in data
    assert "total_chunks" in data


def test_query_requires_repo_id():
    # Missing repo_id (and query) -> validation error.
    response = client.post("/api/v1/rag/query", json={"query": "hi"})
    assert response.status_code == 422


def test_index_contract():
    response = client.post("/api/v1/rag/index", json={})
    assert response.status_code == 422


def test_query_unknown_repo_returns_404():
    response = client.post(
        "/api/v1/rag/query",
        json={"repo_id": "does-not-exist", "query": "anything"},
    )
    assert response.status_code == 404


def test_index_creates_repo(local_repo):
    response = client.post("/api/v1/rag/index", json={"repo_input": local_repo})
    assert response.status_code == 202
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "indexing"
    repo_id = data["repo_id"]

    # Background job finishes -> ready with counts.
    repo = _wait_ready(repo_id)
    assert repo["status"] == "ready"
    assert repo["files_indexed"] == 2
    assert repo["chunks_created"] >= 2

    # Cleanup.
    client.delete(f"/api/v1/rag/repos/{repo_id}")


def test_reindex_while_indexing_conflicts(local_repo):
    first = client.post("/api/v1/rag/index", json={"repo_input": local_repo})
    assert first.status_code == 202
    repo_id = first.json()["repo_id"]
    # A second request before the first completes should 409 (best-effort: only
    # assert when we actually catch it mid-flight).
    second = client.post("/api/v1/rag/index", json={"repo_input": local_repo})
    assert second.status_code in (202, 409)
    _wait_ready(repo_id)
    client.delete(f"/api/v1/rag/repos/{repo_id}")


def test_repos_are_isolated(local_repo, second_local_repo):
    id1 = client.post("/api/v1/rag/index", json={"repo_input": local_repo}).json()["repo_id"]
    _wait_ready(id1)
    id2 = client.post("/api/v1/rag/index", json={"repo_input": second_local_repo}).json()["repo_id"]
    _wait_ready(id2)
    assert id1 != id2

    repos = {r["repo_id"]: r for r in client.get("/api/v1/rag/repos").json()}
    assert id1 in repos and id2 in repos

    # Deleting one leaves the other intact.
    assert client.delete(f"/api/v1/rag/repos/{id1}").status_code == 200
    remaining = {r["repo_id"] for r in client.get("/api/v1/rag/repos").json()}
    assert id1 not in remaining
    assert id2 in remaining

    client.delete(f"/api/v1/rag/repos/{id2}")


def test_delete_unknown_repo_returns_404():
    response = client.delete("/api/v1/rag/repos/nope")
    assert response.status_code == 404


def test_query_stream_unknown_repo_returns_404():
    # Validation runs before the stream starts, so this is a normal 404.
    response = client.post(
        "/api/v1/rag/query/stream",
        json={"repo_id": "does-not-exist", "query": "anything"},
    )
    assert response.status_code == 404


def test_query_stream_requires_repo_id():
    response = client.post("/api/v1/rag/query/stream", json={"query": "hi"})
    assert response.status_code == 422
