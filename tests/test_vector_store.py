"""
Tests for vector store module.
"""

import pytest
import tempfile
import shutil
from src.retrieval.vector_store import VectorStore
from src.ingestion.chunker import CodeChunk


@pytest.fixture
def temp_vector_store():
    """Create temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(
        collection_name="test_collection",
        persist_dir=temp_dir
    )
    yield store
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_add_and_retrieve(temp_vector_store):
    """Test adding chunks and retrieving them."""
    # Create test chunks
    chunks = [
        CodeChunk(
            content="def test_function():\n    pass",
            file_path="test.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=1,
            metadata={"name": "test_function"}
        ),
        CodeChunk(
            content="class TestClass:\n    pass",
            file_path="test.py",
            language="python",
            chunk_type="class",
            start_line=3,
            end_line=4,
            metadata={"name": "TestClass"}
        )
    ]

    # Create dummy embeddings (1536 dimensions for text-embedding-3-small)
    embeddings = [
        [0.1] * 1536,
        [0.2] * 1536
    ]

    # Add chunks
    temp_vector_store.add_chunks(chunks, embeddings)

    # Check count
    assert temp_vector_store.count() == 2

    # Query with similar embedding
    query_embedding = [0.1] * 1536
    results = temp_vector_store.query(query_embedding, n_results=2)

    assert len(results) == 2
    assert results[0]['content'] in ["def test_function():\n    pass", "class TestClass:\n    pass"]


def test_clear_functionality(temp_vector_store):
    """Test clearing the collection."""
    # Add a chunk
    chunk = CodeChunk(
        content="test content",
        file_path="test.py",
        language="python",
        chunk_type="function",
        start_line=0,
        end_line=0,
        metadata={}
    )

    embedding = [0.5] * 1536

    temp_vector_store.add_chunks([chunk], [embedding])

    assert temp_vector_store.count() == 1

    # Clear collection
    temp_vector_store.clear()

    assert temp_vector_store.count() == 0


def test_metadata_filtering(temp_vector_store):
    """Test querying with metadata filters."""
    # Create chunks with different languages
    chunks = [
        CodeChunk(
            content="python code",
            file_path="test.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=0,
            metadata={}
        ),
        CodeChunk(
            content="javascript code",
            file_path="test.js",
            language="javascript",
            chunk_type="function",
            start_line=0,
            end_line=0,
            metadata={}
        )
    ]

    embeddings = [
        [0.1] * 1536,
        [0.2] * 1536
    ]

    temp_vector_store.add_chunks(chunks, embeddings)

    # Query with filter
    query_embedding = [0.1] * 1536
    results = temp_vector_store.query(
        query_embedding,
        n_results=5,
        filter_dict={"language": "python"}
    )

    # Should only return Python chunks
    assert len(results) == 1
    assert results[0]['metadata']['language'] == 'python'


def test_get_by_id(temp_vector_store):
    """Test retrieving chunk by ID."""
    chunk = CodeChunk(
        content="test content",
        file_path="test.py",
        language="python",
        chunk_type="function",
        start_line=5,
        end_line=10,
        metadata={}
    )

    embedding = [0.5] * 1536

    temp_vector_store.add_chunks([chunk], [embedding])

    # Get by ID
    chunk_id = chunk.chunk_id
    result = temp_vector_store.get_by_id(chunk_id)

    assert result is not None
    assert result['content'] == "test content"
    assert result['metadata']['file_path'] == "test.py"


def test_delete_by_file(temp_vector_store):
    """Test deleting chunks by file path."""
    chunks = [
        CodeChunk(
            content="file1 content",
            file_path="file1.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=0,
            metadata={}
        ),
        CodeChunk(
            content="file2 content",
            file_path="file2.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=0,
            metadata={}
        )
    ]

    embeddings = [
        [0.1] * 1536,
        [0.2] * 1536
    ]

    temp_vector_store.add_chunks(chunks, embeddings)

    assert temp_vector_store.count() == 2

    # Delete by file
    temp_vector_store.delete_by_file("file1.py")

    assert temp_vector_store.count() == 1


def test_similarity_conversion(temp_vector_store):
    """Test that distances are converted to similarity scores."""
    chunk = CodeChunk(
        content="test",
        file_path="test.py",
        language="python",
        chunk_type="function",
        start_line=0,
        end_line=0,
        metadata={}
    )

    embedding = [0.5] * 1536

    temp_vector_store.add_chunks([chunk], [embedding])

    # Query
    results = temp_vector_store.query(embedding, n_results=1)

    assert len(results) == 1
    # Similarity should be between 0 and 1
    assert 0 <= results[0]['similarity'] <= 1
    # Distance should also be present
    assert 'distance' in results[0]


def test_hybrid_query_returns_keyword_match(temp_vector_store):
    """Sparse (BM25) retrieval should surface an exact keyword match even when
    the dense embedding is uninformative."""
    chunks = [
        CodeChunk(
            content="def authenticate_user(token):\n    return verify(token)",
            file_path="auth.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=1,
            metadata={"name": "authenticate_user"},
        ),
        CodeChunk(
            content="def render_template(name):\n    return load(name)",
            file_path="views.py",
            language="python",
            chunk_type="function",
            start_line=0,
            end_line=1,
            metadata={"name": "render_template"},
        ),
    ]
    embeddings = [[0.1] * 1536, [0.1] * 1536]
    temp_vector_store.add_chunks(chunks, embeddings)

    results = temp_vector_store.hybrid_query(
        query_text="authenticate_user token",
        query_embedding=[0.1] * 1536,
        n_results=2,
    )

    assert len(results) >= 1
    top = results[0]
    assert "authenticate_user" in top["content"]
    assert "similarity" in top  # RRF fusion score


def test_hybrid_query_persists_and_reloads_bm25(temp_vector_store):
    """The tokenized BM25 corpus should survive a fresh VectorStore on the same
    collection (lazy load from disk, no full re-read needed to score)."""
    from src.retrieval.vector_store import VectorStore

    chunk = CodeChunk(
        content="def unique_keyword_xyz():\n    pass",
        file_path="x.py",
        language="python",
        chunk_type="function",
        start_line=0,
        end_line=1,
        metadata={"name": "unique_keyword_xyz"},
    )
    temp_vector_store.add_chunks([chunk], [[0.2] * 1536])

    # New instance on the same persist dir + collection.
    reopened = VectorStore(
        collection_name=temp_vector_store.collection_name,
        persist_dir=temp_vector_store.persist_dir,
    )
    results = reopened.hybrid_query(
        query_text="unique_keyword_xyz",
        query_embedding=[0.2] * 1536,
        n_results=1,
    )
    assert len(results) == 1
    assert "unique_keyword_xyz" in results[0]["content"]


def test_duplicate_chunk_ids_do_not_break_hybrid_query(temp_vector_store):
    """Chunk IDs can collide across a repo; add_chunks must dedupe so the
    collection and BM25 corpus stay consistent and hybrid_query doesn't pass
    duplicate IDs to ChromaDB's get()."""
    # Two chunks with the SAME id (same file_path + line range).
    dup = lambda content: CodeChunk(
        content=content, file_path="dup.py", language="python",
        chunk_type="function", start_line=0, end_line=1, metadata={},
    )
    chunks = [dup("def alpha_token():\n    pass"), dup("def beta_token():\n    pass")]
    temp_vector_store.add_chunks(chunks, [[0.1] * 1536, [0.2] * 1536])

    # Only one survives (unique id).
    assert temp_vector_store.count() == 1

    # The sparse path must not raise DuplicateIDError.
    results = temp_vector_store.hybrid_query(
        query_text="alpha_token", query_embedding=[0.1] * 1536, n_results=5,
    )
    assert isinstance(results, list)


def test_clear_removes_bm25_index(temp_vector_store):
    """clear() should drop the persisted BM25 index too."""
    import os

    chunk = CodeChunk(
        content="def something():\n    pass",
        file_path="s.py",
        language="python",
        chunk_type="function",
        start_line=0,
        end_line=1,
        metadata={},
    )
    temp_vector_store.add_chunks([chunk], [[0.3] * 1536])
    assert os.path.exists(temp_vector_store._bm25_path)

    temp_vector_store.clear()
    assert not os.path.exists(temp_vector_store._bm25_path)


def test_hybrid_query_attaches_retrieval_provenance(temp_vector_store):
    """Each fused result must carry a 'retrieval' dict explaining *why* it
    surfaced: which retriever(s) matched, the raw BM25 score, the dense rank,
    the fused RRF score, and the final rank."""
    # BM25 tokenizes on whitespace, so keep query terms as standalone tokens.
    # Use several distinct docs: BM25 IDF collapses to ~0 for a term that
    # appears in half the corpus, so the keyword must be genuinely rare.
    contents = [
        "def authenticate token verify session credential",  # 0 - the target
        "def render template html layout component",         # 1
        "def parse json yaml config loader",                 # 2
        "def cache memory redis store expire",               # 3
        "def schedule cron job worker queue",                # 4
    ]
    chunks = [
        CodeChunk(
            content=c, file_path=f"f{i}.py", language="python",
            chunk_type="function", start_line=0, end_line=1,
            metadata={"name": f"fn{i}"},
        )
        for i, c in enumerate(contents)
    ]
    # Distinct one-hot embedding directions so dense ranking is deterministic.
    embeddings = []
    for i in range(len(contents)):
        v = [0.0] * 1536
        v[i] = 1.0
        embeddings.append(v)
    temp_vector_store.add_chunks(chunks, embeddings)

    q_emb = [0.0] * 1536
    q_emb[0] = 0.9
    q_emb[1] = 0.1
    results = temp_vector_store.hybrid_query(
        query_text="authenticate token",  # keyword-matches only chunk 0
        query_embedding=q_emb,             # vector-nearest chunk 0
        n_results=5,
    )

    assert len(results) >= 1
    # Every result carries a well-formed provenance block.
    for i, r in enumerate(results):
        prov = r.get("retrieval")
        assert prov is not None, "missing retrieval provenance"
        assert prov["rank"] == i
        assert prov["rrf_score"] > 0
        assert prov["matched_by"], "matched_by should never be empty"
        assert set(prov["matched_by"]).issubset({"vector", "keyword"})

    # The auth chunk should be found by BOTH retrievers (vector + keyword).
    top = results[0]
    assert "authenticate" in top["content"]
    assert "keyword" in top["retrieval"]["matched_by"]
    assert "vector" in top["retrieval"]["matched_by"]
    assert top["retrieval"]["bm25_score"] is not None and top["retrieval"]["bm25_score"] > 0
    assert top["retrieval"]["dense_rank"] == 0
