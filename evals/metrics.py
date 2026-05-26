"""
Pure retrieval metrics. Kept free of any model/IO so the math is unit-testable.

`ranked_files` is the list of file paths for the retrieved chunks, in rank order
(it may contain duplicates when several chunks come from the same file).
`expected` is the set of file paths that *should* be retrieved for a query.
"""

from typing import List, Set


def reciprocal_rank(expected: Set[str], ranked_files: List[str]) -> float:
    """1 / (rank of the first relevant file), or 0.0 if none is retrieved."""
    for i, f in enumerate(ranked_files):
        if f in expected:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(expected: Set[str], ranked_files: List[str], k: int) -> float:
    """Fraction of expected files that appear in the top-k results."""
    if not expected:
        return 0.0
    topk = set(ranked_files[:k])
    return len(expected & topk) / len(expected)


def hit_at_k(expected: Set[str], ranked_files: List[str], k: int) -> float:
    """1.0 if any expected file appears in the top-k, else 0.0."""
    return 1.0 if (expected & set(ranked_files[:k])) else 0.0


def first_relevant_rank(expected: Set[str], ranked_files: List[str]) -> int:
    """1-based rank of the first relevant file, or 0 if none is retrieved."""
    for i, f in enumerate(ranked_files):
        if f in expected:
            return i + 1
    return 0
