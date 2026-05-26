"""
Vector store module for ChromaDB operations with Hybrid Search.
"""

import os
import pickle
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from src.ingestion.chunker import CodeChunk


class VectorStore:
    """Manages ChromaDB vector database for code chunks."""

    def __init__(self, collection_name: str = "codebase", persist_dir: str = None):
        """Initialize VectorStore."""
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 state. Fully lazy: nothing is loaded from disk or built until the
        # first sparse query. The tokenized corpus is persisted per-collection so
        # restarts don't re-tokenize the whole codebase.
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_corpus: List[List[str]] = []
        self._bm25_loaded = False
        self._bm25_path = os.path.join(self.persist_dir, f"bm25_{self.collection_name}.pkl")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def _persist_bm25(self):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            with open(self._bm25_path, "wb") as f:
                pickle.dump({"ids": self._bm25_ids, "corpus": self._bm25_corpus}, f)
        except Exception as e:
            print(f"Error persisting BM25 index: {e}")

    def _ensure_bm25_loaded(self):
        """Load the tokenized corpus from disk, backfilling from the collection once
        if no persisted index exists yet (e.g. data indexed by an older version)."""
        if self._bm25_loaded:
            return
        if os.path.exists(self._bm25_path):
            try:
                with open(self._bm25_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25_ids = data.get("ids", [])
                self._bm25_corpus = data.get("corpus", [])
                self._bm25_loaded = True
                return
            except Exception as e:
                print(f"Error loading BM25 index: {e}")
        try:
            results = self.collection.get()
            ids = results.get("ids") or []
            docs = results.get("documents") or []
            self._bm25_ids = list(ids)
            self._bm25_corpus = [self._tokenize(d) for d in docs]
            if self._bm25_ids:
                self._persist_bm25()
        except Exception as e:
            print(f"Error backfilling BM25 index: {e}")
        self._bm25_loaded = True

    def _ensure_bm25_model(self):
        """Build the BM25Okapi model from the loaded corpus (rank_bm25 has no
        incremental add, so the model is rebuilt lazily after a corpus change)."""
        self._ensure_bm25_loaded()
        if self._bm25 is None and self._bm25_corpus:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def _reset_bm25(self):
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_corpus = []
        self._bm25_loaded = True  # collection is empty; nothing to load
        try:
            if os.path.exists(self._bm25_path):
                os.remove(self._bm25_path)
        except Exception:
            pass

    def add_chunks(self, chunks: List[CodeChunk], embeddings: List[List[float]]):
        """Store code chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        # Chunk IDs (file_path:start-end) are not guaranteed unique across a
        # repo, and ChromaDB requires unique IDs. Dedupe here so the collection
        # and the BM25 corpus stay consistent (keeping the first occurrence).
        seen_ids = set()
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            ids.append(chunk_id)
            documents.append(chunk.content)

            metadata = {
                'file_path': chunk.file_path,
                'language': chunk.language,
                'chunk_type': chunk.chunk_type,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
            }

            if 'name' in chunk.metadata:
                metadata['name'] = chunk.metadata['name']

            metadatas.append(metadata)
            embedding_list.append(embedding)

        if not ids:
            return

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list
        )

        # Append to the BM25 corpus incrementally (avoids re-reading the whole
        # collection) and invalidate the model so it rebuilds on next query.
        self._ensure_bm25_loaded()
        for cid, document in zip(ids, documents):
            self._bm25_ids.append(cid)
            self._bm25_corpus.append(self._tokenize(document))
        self._bm25 = None
        self._persist_bm25()

    def query(
        self,
        query_embedding: List[float],
        n_results: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar code chunks."""
        n_results = n_results or settings.RETRIEVAL_TOP_K

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )

        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                # Clamp: float error in cosine distance can push this just
                # outside [0, 1], and a similarity > 1 is meaningless.
                similarity = max(0.0, min(1.0, 1 - distance))

                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'distance': distance
                }
                formatted_results.append(result)

        return formatted_results

    def hybrid_query(
        self,
        query_text: str,
        query_embedding: List[float],
        n_results: int = None,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid dense/sparse retrieval."""
        n_results = n_results or settings.RETRIEVAL_TOP_K
        filter_dict = {"language": language_filter} if language_filter else None
        
        # Dense
        dense_results = self.query(query_embedding, n_results=n_results, filter_dict=filter_dict)
        
        # Sparse
        sparse_results = []
        self._ensure_bm25_model()
        if self._bm25 and self._bm25_ids:
            tokenized = self._tokenize(query_text)
            scores = self._bm25.get_scores(tokenized)

            # Rank by score, keep positive hits, then fetch content/metadata for
            # only the top candidates (extra headroom to survive language filtering).
            ranked = sorted(range(len(self._bm25_ids)), key=lambda i: scores[i], reverse=True)
            score_by_id = {}
            candidate_ids = []
            for i in ranked:
                if scores[i] <= 0:
                    continue
                cid = self._bm25_ids[i]
                if cid in score_by_id:
                    continue  # keep the highest-scoring occurrence; ids must be unique for get()
                score_by_id[cid] = float(scores[i])
                candidate_ids.append(cid)
                if len(candidate_ids) >= n_results * 3:
                    break

            if candidate_ids:
                fetched = self.collection.get(ids=candidate_ids)
                fetched_map = {
                    fetched['ids'][j]: {
                        'content': fetched['documents'][j],
                        'metadata': fetched['metadatas'][j],
                    }
                    for j in range(len(fetched['ids']))
                }
                for cid in candidate_ids:
                    if cid not in fetched_map:
                        continue  # stale id (deleted chunk)
                    meta = fetched_map[cid]['metadata']
                    if language_filter and meta.get('language') != language_filter:
                        continue
                    sparse_results.append({
                        'id': cid,
                        'content': fetched_map[cid]['content'],
                        'metadata': meta,
                        'bm25_score': score_by_id[cid],
                    })
                    if len(sparse_results) >= n_results:
                        break
                    
        # RRF Fusion
        rrf_scores = {}
        chunks_map = {}
        k = 60
        
        for rank, res in enumerate(dense_results):
            cid = res['id']
            chunks_map[cid] = res
            rrf_scores[cid] = 1.0 / (k + rank + 1)
            
        for rank, res in enumerate(sparse_results):
            cid = res['id']
            if cid not in chunks_map:
                chunks_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_results = []
        for cid in sorted_ids[:n_results]:
            chunk = dict(chunks_map[cid])
            chunk['similarity'] = rrf_scores[cid]
            final_results.append(chunk)
            
        return final_results

    def clear(self):
        """Delete and recreate collection for new repository."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass 

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._reset_bm25()

    def count(self) -> int:
        return self.collection.count()

    def drop(self):
        """Permanently delete this repo's collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._reset_bm25()

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.collection.get(ids=[chunk_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
        except Exception:
            pass
        return None

    def delete_by_file(self, file_path: str):
        try:
            results = self.collection.get(
                where={"file_path": file_path}
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"Error deleting chunks for file {file_path}: {e}")
