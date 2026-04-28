"""
Local embedding client using sentence-transformers.

Uses all-MiniLM-L6-v2 (384-dim) which runs entirely on CPU — no Ollama
dependency for embeddings.  The PDF allows this:
    "nomic embed text via Ollama **or** sentence transformers"

The previous Ollama nomic-embed-text had a critical bug on this system:
short inputs of similar token length (e.g. "Albert Einstein", "Nikola Tesla",
"Eiffel Tower") all produced IDENTICAL embedding vectors, making retrieval
for short entity queries essentially random.

sentence-transformers produces correct, discriminative embeddings even for
two-word inputs.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded once and shared.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model '%s' …", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("Model loaded — dimension = %d", _model.get_embedding_dimension())
    return _model


class OllamaEmbedder:
    """
    Drop-in replacement that uses sentence-transformers instead of Ollama.

    The class name is kept as ``OllamaEmbedder`` so that every existing import
    (retriever.py, app.py, ingest_all.py, debug_test.py) continues to work
    without modification.
    """

    def __init__(self, model: str = EMBED_MODEL, **_kwargs):
        self.model_name = model
        self._model = _get_model()

    # ── Internal ───────────────────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[list[float]]:
        try:
            vec = self._model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as exc:
            logger.error("Embedding error: %s", exc)
            return None

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_document(self, text: str) -> Optional[list[float]]:
        """Embed a passage that will be stored in the vector index."""
        return self._encode(text)

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Embed a user query for retrieval."""
        return self._encode(text)

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Back-compat alias used by the retriever and other callers."""
        return self._encode(text)

    def embed_documents_batch(
        self,
        texts: list[str],
        progress_callback=None,
    ) -> list[Optional[list[float]]]:
        """Embed a batch of passages (vectorised in one call for speed)."""
        try:
            vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            results = [v.tolist() for v in vecs]
        except Exception as exc:
            logger.error("Batch embedding error: %s", exc)
            results = [None] * len(texts)

        # Fire progress callback after completion (batch is atomic)
        if progress_callback:
            progress_callback(len(texts), len(texts))

        return results

    # Back-compat alias — historical callers used this name
    def get_embeddings_batch(
        self,
        texts: list[str],
        progress_callback=None,
    ) -> list[Optional[list[float]]]:
        return self.embed_documents_batch(texts, progress_callback)

    def is_available(self) -> bool:
        """Always True — the model is loaded locally, no server needed."""
        return True
