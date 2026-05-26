"""
VectorStoreManager: lazily caches one VectorStore per repository.

A repo maps to its own ChromaDB collection. The embedder and LLM client are
shared singletons elsewhere; only the vector store is per-repo, so this manager
keeps the construction cost (and per-repo BM25 index) scoped and reusable.
"""

from typing import Dict
from src.retrieval.vector_store import VectorStore
from src.retrieval.repo_registry import collection_name_for


class VectorStoreManager:
    def __init__(self):
        self._stores: Dict[str, VectorStore] = {}

    def get(self, repo_id: str) -> VectorStore:
        store = self._stores.get(repo_id)
        if store is None:
            store = VectorStore(collection_name=collection_name_for(repo_id))
            self._stores[repo_id] = store
        return store

    def drop(self, repo_id: str):
        """Delete a repo's collection and evict it from the cache."""
        store = self.get(repo_id)
        store.drop()
        self._stores.pop(repo_id, None)

    def evict(self, repo_id: str):
        """Drop the in-memory store without deleting its collection."""
        self._stores.pop(repo_id, None)
