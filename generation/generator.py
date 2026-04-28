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

_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context passages from Wikipedia.

Rules you MUST follow:
1. Answer ONLY using information present in the context passages below.
2. If the context does not contain enough information to answer the question, respond with exactly: "I don't know based on the available data."
3. Never invent, guess, or hallucinate any facts, dates, names, or figures.
4. If the question asks about a person or place not mentioned in the context, say "I don't know based on the available data."
5. Cite the source title when possible (e.g., "According to the passage about Albert Einstein, ...").
6. Be concise and factual.
7. When asked to compare two or more subjects, you MAY synthesize and contrast information from different context passages. Highlight similarities and differences using only facts found in the passages.
8. When asked about a general topic (e.g., "who is associated with electricity"), look through ALL provided passages for relevant information before responding."""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return "No context available."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("metadata", {}).get("source_title", "Unknown")
        parts.append(f"--- Passage {i} (Source: {title}) ---\n{chunk['text']}")
    return "\n\n".join(parts)


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
