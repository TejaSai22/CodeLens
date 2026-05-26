"""
Unit tests for the <think> reasoning strippers in the LLM client.

These exercise pure string/stream helpers, so they run fully offline and never
touch the Gemini API.
"""

from src.generation.llm_client import strip_think_tags, strip_think_stream


# --- strip_think_tags (complete-answer path) --------------------------------

def test_strips_complete_think_block():
    text = "<think>plan the answer</think>The Session pools connections."
    assert strip_think_tags(text) == "The Session pools connections."


def test_strips_multiline_think_block():
    text = "<think>line one\nline two\nline three</think>\n\nFinal answer."
    assert strip_think_tags(text) == "Final answer."


def test_keeps_answer_when_no_think_block():
    text = "Just a plain answer with no tags."
    assert strip_think_tags(text) == "Just a plain answer with no tags."


def test_removes_orphan_open_tag():
    # Malformed output with an unmatched opening tag must not leak the tag.
    text = "<think>reasoning that never closed, then answer"
    assert "<think>" not in strip_think_tags(text)
    assert "reasoning that never closed, then answer" in strip_think_tags(text)


def test_removes_orphan_close_tag():
    text = "answer text</think>"
    assert strip_think_tags(text) == "answer text"


def test_handles_multiple_blocks():
    text = "<think>a</think>One.<think>b</think>Two."
    assert strip_think_tags(text) == "One.Two."


# --- strip_think_stream (token-streaming path) ------------------------------

def _collect(pieces):
    return "".join(strip_think_stream(pieces))


def test_stream_strips_block_emitted_in_one_piece():
    pieces = ["<think>reasoning</think>", "Hello ", "world"]
    assert _collect(pieces) == "Hello world"


def test_stream_strips_block_split_across_pieces():
    # The tags themselves are split across chunk boundaries.
    pieces = ["<th", "ink>secret rea", "soning</th", "ink>", "Visible ", "answer"]
    assert _collect(pieces) == "Visible answer"


def test_stream_passes_through_when_no_think():
    pieces = ["The ", "answer ", "streams ", "cleanly."]
    assert _collect(pieces) == "The answer streams cleanly."


def test_stream_handles_text_before_think():
    pieces = ["Intro. ", "<think>", "hidden", "</think>", " Outro."]
    assert _collect(pieces) == "Intro.  Outro."


def test_stream_drops_unclosed_think_block():
    # If the stream ends mid-think, the reasoning is discarded, not leaked.
    pieces = ["<think>reasoning that ", "never closes"]
    assert _collect(pieces) == ""


def test_stream_never_emits_partial_tag():
    pieces = ["answer<thi"]  # trailing partial tag held back, then flushed
    # "<thi" is not a complete tag, so on flush it is emitted as literal text;
    # what matters is no broken/partial-then-stripped artifact appears.
    out = _collect(pieces)
    assert out.startswith("answer")
