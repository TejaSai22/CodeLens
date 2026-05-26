"""
Code chunker module for splitting code files into semantic chunks.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from config.settings import settings


# Map a file extension to the tree-sitter grammar name. Extension is more
# reliable than the loader's coarse language label (e.g. .tsx needs the dedicated
# `tsx` grammar, which the plain `typescript` grammar can't parse).
_EXT_TO_GRAMMAR = {
    '.py': 'python',
    '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'tsx',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c', '.h': 'c',
    '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp',
}

# Per-grammar AST node types that represent a chunkable definition.
_DEFINITION_NODES: Dict[str, Set[str]] = {
    'python': {'function_definition', 'class_definition'},
    'javascript': {'function_declaration', 'generator_function_declaration',
                   'class_declaration', 'method_definition'},
    'typescript': {'function_declaration', 'generator_function_declaration',
                   'class_declaration', 'abstract_class_declaration', 'method_definition',
                   'interface_declaration', 'type_alias_declaration', 'enum_declaration'},
    'java': {'class_declaration', 'interface_declaration', 'enum_declaration',
             'method_declaration', 'constructor_declaration'},
    'go': {'function_declaration', 'method_declaration', 'type_declaration'},
    'rust': {'function_item', 'struct_item', 'enum_item', 'trait_item',
             'impl_item', 'mod_item'},
    'c': {'function_definition', 'struct_specifier', 'enum_specifier'},
    'cpp': {'function_definition', 'class_specifier', 'struct_specifier', 'enum_specifier'},
}
_DEFINITION_NODES['tsx'] = _DEFINITION_NODES['typescript']

# Cache one parser per grammar (building a Parser/Language is not free).
_PARSERS: Dict[str, object] = {}


def _get_parser(grammar: str):
    parser = _PARSERS.get(grammar)
    if parser is None:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language
        parser = Parser(get_language(grammar))
        _PARSERS[grammar] = parser
    return parser

# Node types that are "type/container"-like rather than plain functions.
_CLASS_LIKE = {
    'class_definition', 'class_declaration', 'abstract_class_declaration',
    'interface_declaration', 'enum_declaration', 'type_alias_declaration',
    'struct_specifier', 'enum_specifier', 'class_specifier',
    'struct_item', 'enum_item', 'trait_item', 'impl_item', 'mod_item',
    'type_declaration',
}


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    content: str
    file_path: str
    language: str
    chunk_type: str  # 'function', 'class', 'module', 'block'
    start_line: int  # 0-indexed
    end_line: int  # 0-indexed
    metadata: Dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate unique chunk ID."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class CodeChunker:
    """Splits code files into semantic chunks."""

    def __init__(self, max_chunk_size: int = None, overlap: int = None, use_ast: bool = True):
        """
        Initialize CodeChunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap size for sliding window
            use_ast: When False, always use the sliding-window chunker (for A/B eval)
        """
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.use_ast = use_ast

    def chunk_file(self, file_data: Dict) -> List[CodeChunk]:
        """
        Main entry point for chunking a file.

        Args:
            file_data: Dictionary with keys: path, content, language, size

        Returns:
            List of CodeChunk objects
        """
        file_path = file_data['path']
        content = file_data['content']
        language = file_data['language']

        # Prefer AST-aware chunking; fall back to a sliding window when there's
        # no grammar for the language or no definitions were found in the file.
        chunks = self._chunk_with_ast(content, file_path, language) if self.use_ast else None
        if not chunks:
            chunks = self._chunk_generic(content, file_path, language)

        # Split any chunks that exceed max size
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.max_chunk_size:
                final_chunks.extend(self._split_large_chunk(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    @staticmethod
    def _grammar_for(file_path: str) -> Optional[str]:
        return _EXT_TO_GRAMMAR.get(Path(file_path).suffix.lower())

    @staticmethod
    def _node_name(node) -> str:
        """Best-effort name extraction for a definition node, across grammars."""
        named = node.child_by_field_name('name')
        if named is not None and named.text:
            return named.text.decode('utf8', 'ignore')

        for child in node.children:
            if child.type in ('identifier', 'type_identifier', 'field_identifier'):
                return child.text.decode('utf8', 'ignore')

        # C/C++: the name is nested inside (possibly several) declarators.
        decl = node.child_by_field_name('declarator')
        while decl is not None:
            if decl.type in ('identifier', 'field_identifier'):
                return decl.text.decode('utf8', 'ignore')
            decl = decl.child_by_field_name('declarator')
        return ''

    def _chunk_with_ast(self, content: str, file_path: str, language: str) -> Optional[List[CodeChunk]]:
        """Split source into chunks at definition boundaries using tree-sitter.

        Returns None when the language has no grammar (so the caller can fall
        back); returns an empty list only if parsing yielded no definitions.
        """
        grammar = self._grammar_for(file_path)
        targets = _DEFINITION_NODES.get(grammar) if grammar else None
        if not targets:
            return None

        try:
            parser = _get_parser(grammar)
            content_bytes = bytes(content, 'utf8')
            tree = parser.parse(content_bytes)
        except Exception:
            return None

        chunks: List[CodeChunk] = []

        def traverse(node):
            if node.type in targets:
                name = self._node_name(node)
                # Slice on bytes (node offsets are byte-based) to stay correct
                # for non-ASCII source.
                chunk_text = content_bytes[node.start_byte:node.end_byte].decode('utf8', 'ignore')
                chunks.append(CodeChunk(
                    content=chunk_text,
                    file_path=file_path,
                    language=language,
                    chunk_type='class' if node.type in _CLASS_LIKE else 'function',
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    metadata={'name': name} if name else {},
                ))
                return  # don't descend into a captured definition
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        if not chunks:
            return None

        # Definitions only cover their own lines; capture the remaining
        # module-level code (imports, top-level constants, etc.) as block chunks
        # so nothing is dropped from the index.
        covered = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))

        lines = content.split('\n')
        i, total = 0, len(lines)
        while i < total:
            if i in covered:
                i += 1
                continue
            j = i
            while j < total and j not in covered:
                j += 1
            gap_text = '\n'.join(lines[i:j])
            if gap_text.strip():
                chunks.append(CodeChunk(
                    content=gap_text,
                    file_path=file_path,
                    language=language,
                    chunk_type='block',
                    start_line=i,
                    end_line=j - 1,
                    metadata={},
                ))
            i = j

        chunks.sort(key=lambda c: c.start_line)
        return chunks

    def _chunk_generic(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """
        Sliding window fallback for non-Python files.

        Args:
            content: File content
            file_path: Path to file
            language: Programming language

        Returns:
            List of CodeChunk objects
        """
        lines = content.split('\n')
        chunks = []

        # Calculate lines per chunk
        chars_per_line = len(content) / len(lines) if lines else 1
        lines_per_chunk = int(self.max_chunk_size / chars_per_line) if chars_per_line > 0 else 100
        overlap_lines = int(self.overlap / chars_per_line) if chars_per_line > 0 else 10

        # Ensure minimum values
        lines_per_chunk = max(lines_per_chunk, 10)
        overlap_lines = min(overlap_lines, lines_per_chunk // 2)

        i = 0
        while i < len(lines):
            # Calculate chunk boundaries
            start_line = i
            end_line = min(i + lines_per_chunk, len(lines)) - 1

            # Extract chunk content
            chunk_lines = lines[start_line:end_line + 1]
            chunk_content = '\n'.join(chunk_lines)

            # Create chunk
            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                language=language,
                chunk_type='block',
                start_line=start_line,
                end_line=end_line,
                metadata={}
            )

            chunks.append(chunk)

            # Move to next chunk with overlap
            i += lines_per_chunk - overlap_lines

            # Break if we've covered the entire file
            if end_line >= len(lines) - 1:
                break

        return chunks

    def _split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """
        Recursively split chunks that exceed max size.

        Args:
            chunk: CodeChunk to split

        Returns:
            List of smaller CodeChunk objects
        """
        content = chunk.content
        lines = content.split('\n')

        # Calculate how many sub-chunks we need
        chars_per_line = len(content) / len(lines) if lines else 1
        lines_per_chunk = int(self.max_chunk_size / chars_per_line) if chars_per_line > 0 else 50
        lines_per_chunk = max(lines_per_chunk, 10)

        chunks = []
        total_lines = len(lines)

        for i in range(0, total_lines, lines_per_chunk):
            start_idx = i
            end_idx = min(i + lines_per_chunk, total_lines)

            sub_lines = lines[start_idx:end_idx]
            sub_content = '\n'.join(sub_lines)

            sub_chunk = CodeChunk(
                content=sub_content,
                file_path=chunk.file_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                start_line=chunk.start_line + start_idx,
                end_line=chunk.start_line + end_idx - 1,
                metadata=chunk.metadata
            )

            chunks.append(sub_chunk)

        return chunks
