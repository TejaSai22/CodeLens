"""
LLM client module for generating answers using Google Gemini (google-genai SDK).
"""

import re
from typing import List, Dict, Optional, Any, Iterable, Iterator
from google import genai
from google.genai import types
from loguru import logger
from config.settings import settings

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Defensive safety net: the model is configured to keep its reasoning internal
# (ThinkingConfig include_thoughts=False) and prompted to answer directly, so it
# should not emit <think> blocks. If a model ever does, strip them before the
# text reaches the user rather than relying solely on prompt compliance.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Mops up any orphan tags left behind by malformed output so a stray
# "<think>" / "</think>" never leaks even when the pair is unbalanced.
_THINK_ORPHAN_RE = re.compile(r"</?think>", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Strip the model's <think> reasoning from a complete answer string."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_ORPHAN_RE.sub("", cleaned)
    return cleaned.strip()


def _partial_tag_suffix_len(buffer: str, tag: str) -> int:
    """Length of the longest suffix of `buffer` that is a (partial) prefix of
    `tag`. We hold that many trailing chars back in case a tag is split across
    streaming chunks (e.g. '<thi' now, 'nk>' next)."""
    for k in range(min(len(buffer), len(tag) - 1), 0, -1):
        if tag.startswith(buffer[-k:]):
            return k
    return 0


def strip_think_stream(pieces: Iterable[str]) -> Iterator[str]:
    """Filter <think>...</think> out of a token stream on the fly.

    Buffers across chunk boundaries so a tag split between two tokens is still
    detected, and never emits a partial tag. Text before/after the think block
    streams through unchanged; the block's contents are dropped."""
    buffer = ""
    in_think = False
    for piece in pieces:
        buffer += piece
        out = []
        advanced = True
        while advanced:
            advanced = False
            if not in_think:
                i = buffer.find(_THINK_OPEN)
                if i != -1:
                    out.append(buffer[:i])
                    buffer = buffer[i + len(_THINK_OPEN):]
                    in_think = True
                    advanced = True
                else:
                    hold = _partial_tag_suffix_len(buffer, _THINK_OPEN)
                    if hold < len(buffer):
                        out.append(buffer[:len(buffer) - hold])
                        buffer = buffer[len(buffer) - hold:]
            else:
                j = buffer.find(_THINK_CLOSE)
                if j != -1:
                    buffer = buffer[j + len(_THINK_CLOSE):]
                    in_think = False
                    advanced = True
                else:
                    hold = _partial_tag_suffix_len(buffer, _THINK_CLOSE)
                    buffer = buffer[len(buffer) - hold:] if hold else ""
        text = "".join(out)
        if text:
            yield text
    # Flush whatever is left once the stream ends (an unclosed think block is
    # intentionally discarded rather than leaked).
    if not in_think and buffer:
        yield buffer


class LLMClient:
    """Wrapper for Google Gemini to generate answers."""

    SYSTEM_PROMPT = """You are an expert AI code assistant analyzing a codebase.
You are given user queries and relevant retrieved code snippets.

INSTRUCTIONS:
1. ANSWER DIRECTLY: Reply with the answer only. Do NOT narrate your reasoning, restate the question, or describe your search process. Reason internally and present only conclusions.
2. CITE PROPERLY: Whenever you reference code, you MUST cite the file path. E.g., `As seen in file.py (lines X-Y)...`
3. ACCURACY: Do not hallucinate code. If the context lacks the information, clearly state that you cannot answer based on the context.
4. TONE: Be direct, highly technical, and avoid conversational filler."""

    def __init__(self, model_name: str = None):
        """Configure the client. The underlying genai.Client is created lazily
        (see the `client` property) so importing/booting the API and running
        indexing + retrieval never requires a GEMINI_API_KEY — only answer
        generation does. This also lets the test suite import the app without a
        key in CI."""
        self.model_name = model_name or settings.LLM_MODEL
        self._client = None
        # The model reasons internally (Gemini 2.x native thinking) but
        # include_thoughts=False keeps that reasoning out of the response, so
        # the user only ever sees the final answer.
        self._config = types.GenerateContentConfig(
            system_instruction=self.SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )
        logger.info(f"Configured Google Gen AI client (lazy): {self.model_name}")

    @property
    def client(self):
        """Lazily construct the genai client on first use. Raises a clear error
        if no key is configured at the point generation is actually attempted."""
        if self._client is None:
            if not settings.GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. It is required for answer "
                    "generation (indexing and retrieval work without it)."
                )
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    @staticmethod
    def _to_history(messages: Optional[List[Dict[str, str]]]) -> List[types.Content]:
        """Map chat messages to google-genai Content objects."""
        history: List[types.Content] = []
        for msg in messages or []:
            role = "model" if msg.get("role") == "assistant" else "user"
            history.append(types.Content(role=role, parts=[types.Part(text=msg.get("content", ""))]))
        return history

    def _new_chat(self, history: List[types.Content]):
        return self.client.chats.create(
            model=self.model_name,
            config=self._config,
            history=history,
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a full answer using the Gemini API."""
        context = self._format_context(retrieved_chunks)
        prompt = f"Context snippets:\n{context}\n\nUser Question:\n{query}"
        try:
            chat = self._new_chat(self._to_history(conversation_history))
            response = chat.send_message(prompt)
            return strip_think_tags(response.text)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return f"Error generating answer with Gemini: {str(e)}\n\nPlease ensure your GEMINI_API_KEY is correctly set in the .env file."

    def generate_answer_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Stream the answer from Gemini, yielding text pieces as they arrive."""
        context = self._format_context(retrieved_chunks)
        prompt = f"Context snippets:\n{context}\n\nUser Question:\n{query}"

        chat = self._new_chat(self._to_history(conversation_history))

        def _raw_pieces():
            for chunk in chat.send_message_stream(prompt):
                try:
                    if chunk.text:
                        yield chunk.text
                except Exception:
                    # Some chunks (e.g. safety-blocked) raise on .text; skip them.
                    continue

        # Strip <think> reasoning on the fly so the client only sees the answer.
        yield from strip_think_stream(_raw_pieces())

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks as markdown context."""
        if not retrieved_chunks:
            return "No relevant code snippets found."

        formatted_chunks = []

        for _, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk['metadata']
            content = chunk['content']
            similarity = chunk.get('similarity', 0)

            file_path = metadata.get('file_path', 'unknown')
            start_line = metadata.get('start_line', 0)
            end_line = metadata.get('end_line', 0)
            chunk_type = metadata.get('chunk_type', 'code')
            language = metadata.get('language', '')
            name = metadata.get('name', '')

            header = f"### {file_path} (lines {start_line}-{end_line})"
            relevance = f"{similarity:.2%}" if similarity else "N/A"
            meta_line = f"**Type:** {chunk_type} | **Relevance:** {relevance}"

            if name:
                meta_line += f" | **Name:** {name}"

            code_block = f"```{language}\n{content}\n```"

            formatted_chunk = f"{header}\n{meta_line}\n{code_block}"
            formatted_chunks.append(formatted_chunk)

        return "\n\n".join(formatted_chunks)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Simple chat interface without retrieval."""
        if not messages:
            return ""
        try:
            chat = self._new_chat(self._to_history(messages[:-1]))
            response = chat.send_message(messages[-1].get("content", ""))
            return response.text
        except Exception as e:
            logger.error(f"Error in generic chat context: {str(e)}")
            return str(e)
