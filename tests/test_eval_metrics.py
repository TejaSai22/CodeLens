"""
Unit tests for the retrieval eval metrics (pure functions, no model/IO).
"""

from evals.metrics import recall_at_k, reciprocal_rank, hit_at_k, first_relevant_rank


def test_reciprocal_rank_first_position():
    assert reciprocal_rank({"a.py"}, ["a.py", "b.py", "c.py"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank({"b.py"}, ["a.py", "b.py", "c.py"]) == 0.5


def test_reciprocal_rank_miss():
    assert reciprocal_rank({"z.py"}, ["a.py", "b.py"]) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k({"a.py", "b.py"}, ["a.py", "c.py", "d.py"], 5) == 0.5


def test_recall_at_k_full():
    assert recall_at_k({"a.py", "b.py"}, ["a.py", "b.py"], 5) == 1.0


def test_recall_at_k_respects_cutoff():
    # b.py is at rank 2 but k=1, so it is not counted.
    assert recall_at_k({"b.py"}, ["a.py", "b.py"], 1) == 0.0


def test_recall_empty_expected():
    assert recall_at_k(set(), ["a.py"], 5) == 0.0


def test_hit_at_k():
    assert hit_at_k({"b.py"}, ["a.py", "b.py"], 2) == 1.0
    assert hit_at_k({"b.py"}, ["a.py", "b.py"], 1) == 0.0


def test_first_relevant_rank():
    assert first_relevant_rank({"c.py"}, ["a.py", "b.py", "c.py"]) == 3
    assert first_relevant_rank({"z.py"}, ["a.py", "b.py"]) == 0


def test_duplicates_use_first_occurrence():
    # Same file in multiple chunks: rank is the first occurrence.
    assert reciprocal_rank({"a.py"}, ["b.py", "a.py", "a.py"]) == 0.5
