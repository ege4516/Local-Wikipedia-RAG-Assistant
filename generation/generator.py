"""
Local LLM generation via Ollama /api/chat.
The system prompt strictly constrains the model to the supplied context.
"""

import logging
from typing import Optional

import requests

from config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, LLM_TIMEOUT_SEC

logger = logging.getLogger(__name__)

_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

_SYSTEM_PROMPT = """You are a helpful assistant. You will be given context passages from Wikipedia and a question. Your job is to answer the question using the context.

IMPORTANT: The context passages below contain the information you need. Read them carefully and answer the question based on what you find. Always prioritize the context in the LAST user message over any previous context.

Guidelines:
1. Answer the question using facts from the context passages.
2. Cite the source when possible (e.g., "According to the passage about Nikola Tesla, ...").
3. Be concise and factual.
4. When comparing subjects, use information from multiple passages.
5. For general questions (e.g., "who is associated with electricity"), scan ALL passages for relevant information.
6. Do NOT make up facts that are not in the context.
7. If the context is only partially relevant, answer only the parts you can confirm from the passages.
8. If comparing two subjects but the context only covers one, answer what you know and state that data for the other subject is not available.
9. Only say "I don't know based on the available data." if the passages contain absolutely NO relevant information about the topic."""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return "No context available."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("metadata", {}).get("source_title", "Unknown")
        parts.append(f"--- Passage {i} (Source: {title}) ---\n{chunk['text']}")
    return "\n\n".join(parts)


_IDK = "I don't know based on the available data."

_REFUSAL_PHRASES = [
    "i don't know",
    "i don\u2019t know",
    "i do not know",
    "no relevant information",
    "couldn't find any information",
    "could not find any information",
    "is not mentioned in the context",
    "is not mentioned in the provided",
    "not available in the provided",
    "i don't have any information",
    "i don't have information",
    "i cannot answer",
    "i can't answer",
    "none of the context passages",
    "none of the provided passages",
    "the context does not contain",
    "the passages do not contain",
    "not present in the context",
    "not in the provided context",
    "i can not provide",
    "i cannot provide",
    "don't have any relevant information",
    "no relevant data",
    "i don't see a question",
]


def _is_refusal(text: str) -> bool:
    """Return True if the LLM response is essentially a refusal.

    Only checks the first 250 characters to avoid false positives where
    a refusal-like phrase appears inside an otherwise valid answer
    (e.g. 'As not mentioned in earlier sources, Einstein was born in Ulm').
    """
    head = text[:250].lower()
    return any(phrase in head for phrase in _REFUSAL_PHRASES)


class Generator:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self._endpoint = f"{base_url}/api/chat"

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        chunks: list[dict],
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate an answer grounded in *chunks*.

        chat_history: list of {"role": "user"|"assistant", "content": "..."}
        Returns the assistant reply string.
        """
        # Early exit — no point calling the LLM with zero context
        if not chunks:
            return _IDK

        context = _build_context_block(chunks)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Include prior turns for multi-turn coherence (last 6 turns max)
        if chat_history:
            messages.extend(chat_history[-6:])

        messages.append({"role": "user", "content": user_message})

        try:
            resp = requests.post(
                self._endpoint,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=LLM_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            logger.info(
                "Generated %d-char response with model '%s'.", len(content), self.model
            )

            # Normalize any refusal variant to the canonical string
            if _is_refusal(content):
                return _IDK

            return content

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s.", OLLAMA_BASE_URL)
            return (
                "Error: Cannot connect to the local LLM. "
                "Please make sure Ollama is running (`ollama serve`)."
            )
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds.", LLM_TIMEOUT_SEC)
            return "Error: The LLM took too long to respond. Try a lighter model."
        except Exception as exc:
            logger.error("Generation error: %s", exc)
            return f"Error during generation: {exc}"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_local_models(self) -> list[str]:
        """Return model names currently pulled in Ollama."""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

