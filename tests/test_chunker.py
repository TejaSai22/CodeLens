"""
Tests for code chunker module.
"""

import pytest
from src.ingestion.chunker import CodeChunker, CodeChunk


def test_python_chunking():
    """Test that Python code is split by functions correctly."""
    chunker = CodeChunker()

    # Sample Python code
    python_code = """
def function_one():
    return 1

def function_two():
    return 2

class MyClass:
    def method_one(self):
        pass
"""

    file_data = {
        'path': 'test.py',
        'content': python_code,
        'language': 'python',
        'size': len(python_code)
    }

    chunks = chunker.chunk_file(file_data)

    # Should create chunks for functions and class
    assert len(chunks) > 0

    # Check that chunks have correct attributes
    for chunk in chunks:
        assert chunk.file_path == 'test.py'
        assert chunk.language == 'python'
        assert chunk.chunk_type in ['function', 'class']
        assert chunk.start_line >= 0
        assert chunk.end_line >= chunk.start_line


def test_large_file_handling():
    """Test that large files are split correctly."""
    chunker = CodeChunker(max_chunk_size=100)

    # Create large content
    large_content = "x = 1\n" * 100

    file_data = {
        'path': 'large.py',
        'content': large_content,
        'language': 'python',
        'size': len(large_content)
    }

    chunks = chunker.chunk_file(file_data)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Each chunk should be under max size (except edge cases)
    for chunk in chunks:
        assert len(chunk.content) <= chunker.max_chunk_size * 2  # Allow some buffer


def test_overlap_behavior():
    """Test that sliding window has correct overlap."""
    chunker = CodeChunker(max_chunk_size=200, overlap=50)

    # Create content that will require multiple chunks
    content = "line " + str(1) + "\n" * 50

    file_data = {
        'path': 'test.js',
        'content': content,
        'language': 'javascript',
        'size': len(content)
    }

    chunks = chunker.chunk_file(file_data)

    # Should have overlap
    if len(chunks) > 1:
        # Check that consecutive chunks have some overlap
        # (This is approximate due to line-based chunking)
        assert chunks[0].end_line >= chunks[1].start_line - 20


def test_chunk_id_format():
    """Test that chunk IDs are formatted correctly."""
    chunker = CodeChunker()

    python_code = """
def test_function():
    pass
"""

    file_data = {
        'path': 'test.py',
        'content': python_code,
        'language': 'python',
        'size': len(python_code)
    }

    chunks = chunker.chunk_file(file_data)

    for chunk in chunks:
        # Check ID format: {file_path}:{start_line}-{end_line}
        chunk_id = chunk.chunk_id
        assert ':' in chunk_id
        assert '-' in chunk_id
        assert chunk.file_path in chunk_id


def test_empty_file():
    """Test handling of empty files."""
    chunker = CodeChunker()

    file_data = {
        'path': 'empty.py',
        'content': '',
        'language': 'python',
        'size': 0
    }

    chunks = chunker.chunk_file(file_data)

    # Should handle empty files gracefully
    assert isinstance(chunks, list)


def test_generic_fallback_for_unsupported_language():
    """Files with no grammar fall back to sliding-window 'block' chunks."""
    chunker = CodeChunker()

    content = "some plain text\n" * 5
    file_data = {
        'path': 'notes.txt',
        'content': content,
        'language': 'unknown',
        'size': len(content),
    }

    chunks = chunker.chunk_file(file_data)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_type == 'block'


def test_javascript_ast_chunking():
    """JavaScript is now chunked by definition, not sliding window."""
    chunker = CodeChunker()

    js_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Widget {
    render() { return null; }
}
"""
    chunks = chunker.chunk_file({
        'path': 'app.js', 'content': js_code, 'language': 'javascript', 'size': len(js_code),
    })

    types = {c.chunk_type for c in chunks}
    names = {c.metadata.get('name') for c in chunks}
    assert 'function' in types
    assert 'class' in types
    assert 'greet' in names
    assert 'Widget' in names


def test_typescript_ast_chunking():
    """TypeScript interfaces and functions are captured as chunks."""
    chunker = CodeChunker()

    ts_code = """
interface User {
    id: number;
    name: string;
}

function loadUser(id: number): User {
    return { id, name: "x" };
}
"""
    chunks = chunker.chunk_file({
        'path': 'user.ts', 'content': ts_code, 'language': 'typescript', 'size': len(ts_code),
    })
    names = {c.metadata.get('name') for c in chunks}
    assert 'User' in names
    assert 'loadUser' in names


def test_go_ast_chunking():
    """Go functions are captured by AST chunking."""
    chunker = CodeChunker()

    go_code = """
package main

func Add(a int, b int) int {
    return a + b
}

func Sub(a int, b int) int {
    return a - b
}
"""
    chunks = chunker.chunk_file({
        'path': 'math.go', 'content': go_code, 'language': 'go', 'size': len(go_code),
    })
    names = {c.metadata.get('name') for c in chunks}
    assert 'Add' in names
    assert 'Sub' in names


def test_rust_ast_chunking():
    """Rust structs and functions are captured by AST chunking."""
    chunker = CodeChunker()

    rust_code = """
struct Point {
    x: i32,
    y: i32,
}

fn distance(p: &Point) -> i32 {
    p.x + p.y
}
"""
    chunks = chunker.chunk_file({
        'path': 'geo.rs', 'content': rust_code, 'language': 'rust', 'size': len(rust_code),
    })
    names = {c.metadata.get('name') for c in chunks}
    types = {c.chunk_type for c in chunks}
    assert 'Point' in names
    assert 'distance' in names
    assert 'class' in types  # struct is class-like
    assert 'function' in types
