"""
Integration tests for CodeLens.
"""

import pytest
import tempfile
import shutil
import os
from src.ingestion.repo_loader import RepoLoader
from src.ingestion.chunker import CodeChunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore


@pytest.fixture
def test_repo():
    """Create a temporary test repository."""
    temp_dir = tempfile.mkdtemp()

    # Create some test files
    test_py = os.path.join(temp_dir, "test.py")
    with open(test_py, 'w') as f:
        f.write("""
def calculate_sum(a, b):
    '''Calculate the sum of two numbers.'''
    return a + b

def calculate_product(a, b):
    '''Calculate the product of two numbers.'''
    return a * b

class Calculator:
    '''A simple calculator class.'''

    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y
""")

    test_js = os.path.join(temp_dir, "test.js")
    with open(test_js, 'w') as f:
        f.write("""
function greet(name) {
    return `Hello, ${name}!`;
}

const PI = 3.14159;
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_vector_store():
    """Create temporary vector store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(
        collection_name="integration_test",
        persist_dir=temp_dir
    )
    yield store
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_pipeline(test_repo, temp_vector_store):
    """Test the complete local indexing and retrieval pipeline (no LLM call)."""

    # Step 1: Load repository
    loader = RepoLoader(repo_path=test_repo)
    files = loader.get_files()

    assert len(files) == 2  # test.py and test.js

    # Step 2: Chunk files
    chunker = CodeChunker()
    all_chunks = []

    for file_data in files:
        chunks = chunker.chunk_file(file_data)
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0

    # Step 3: Embed chunks (local sentence-transformers, no API key)
    embedder = Embedder()
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed(chunk_texts)

    assert len(embeddings) == len(all_chunks)

    # Step 4: Store in vector database
    temp_vector_store.add_chunks(all_chunks, embeddings)

    assert temp_vector_store.count() == len(all_chunks)

    # Step 5: Query
    query = "How do I calculate the sum of two numbers?"
    query_embedding = embedder.embed_single(query)

    results = temp_vector_store.query(query_embedding, n_results=3)

    assert len(results) > 0

    # Check that results contain relevant information
    found_relevant = False
    for result in results:
        if 'sum' in result['content'].lower() or 'add' in result['content'].lower():
            found_relevant = True
            break

    assert found_relevant, "Should find relevant chunks about sum/addition"

    # Cleanup
    loader.cleanup()


def test_query_accuracy(test_repo, temp_vector_store):
    """Test that queries return accurate results."""

    # Index repository
    loader = RepoLoader(repo_path=test_repo)
    files = loader.get_files()

    chunker = CodeChunker()
    all_chunks = []

    for file_data in files:
        chunks = chunker.chunk_file(file_data)
        all_chunks.extend(chunks)

    embedder = Embedder()
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embedder.embed(chunk_texts)

    temp_vector_store.add_chunks(all_chunks, embeddings)

    # Test query: Find Python functions
    query = "Show me Python functions for arithmetic operations"
    query_embedding = embedder.embed_single(query)

    results = temp_vector_store.query(query_embedding, n_results=5)

    # Should prioritize Python files
    python_count = sum(1 for r in results if r['metadata']['language'] == 'python')
    assert python_count > 0

    # Cleanup
    loader.cleanup()


def test_chunker_preserves_metadata(test_repo):
    """Test that chunker preserves metadata correctly."""

    loader = RepoLoader(repo_path=test_repo)
    files = loader.get_files()

    chunker = CodeChunker()

    for file_data in files:
        chunks = chunker.chunk_file(file_data)

        for chunk in chunks:
            # Check metadata is preserved
            assert chunk.file_path == file_data['path']
            assert chunk.language == file_data['language']
            assert chunk.start_line >= 0
            assert chunk.end_line >= chunk.start_line

            # Check chunk ID format
            assert ':' in chunk.chunk_id
            assert '-' in chunk.chunk_id

    loader.cleanup()


def test_empty_repository():
    """Test handling of empty repository."""
    temp_dir = tempfile.mkdtemp()

    try:
        loader = RepoLoader(repo_path=temp_dir)
        files = loader.get_files()

        assert len(files) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
