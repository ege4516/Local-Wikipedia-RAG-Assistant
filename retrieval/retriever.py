"""
Retriever: classify the query → embed it → search relevant collection(s).

For "compare X and Y" queries (category="both" with multiple entity names),
we run separate embeddings per entity to ensure both sides are represented
in the retrieved chunks.
"""

import logging
import re
from typing import Optional

from config import DEFAULT_TOP_K, PEOPLE, PLACES
from db.vector_store import VectorStore
from ingest.embedder import OllamaEmbedder
from retrieval.query_classifier import classify_query

logger = logging.getLogger(__name__)

# Pre-compile patterns for entity extraction
_ALL_ENTITIES = PEOPLE + PLACES
_ENTITY_PATTERN = re.compile(
    "|".join(re.escape(e) for e in sorted(_ALL_ENTITIES, key=len, reverse=True)),
    re.IGNORECASE,
)


class Retriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[OllamaEmbedder] = None,
    ):
        self._vs = vector_store or VectorStore()
        self._embedder = embedder or OllamaEmbedder()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _retrieve_single(
        self, query: str, category: str, k: int
    ) -> list[dict]:
        """Embed *query* once and search the appropriate collection(s)."""
        embedding = self._embedder.get_embedding(query)
        if embedding is None:
            logger.error("Failed to embed query.")
            return []

        if category == "both":
            person_chunks = self._vs.query(embedding, "person", k=k)
            place_chunks = self._vs.query(embedding, "place", k=k)
            all_chunks = person_chunks + place_chunks
            all_chunks.sort(key=lambda c: c["distance"])
            return all_chunks[:k]

        return self._vs.query(embedding, category, k=k)

    @staticmethod
    def _extract_entity_names(query: str) -> list[str]:
        """Return distinct entity names mentioned in *query*."""
        return list(dict.fromkeys(m.group() for m in _ENTITY_PATTERN.finditer(query)))

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        force_category: Optional[str] = None,
    ) -> dict:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Returns:
            {
                "category": "person" | "place" | "both",
                "chunks": [{"text": ..., "metadata": ..., "distance": ...}, ...],
            }

        force_category overrides the automatic classifier (useful for testing).
        """
        category = force_category or classify_query(query)
        logger.info("Query category: %s | query: %r", category, query)

        # ── Multi-entity queries ("Compare Einstein and Tesla") ──────────
        entity_names = self._extract_entity_names(query)

        if len(entity_names) >= 2:
            logger.info(
                "Multi-entity query detected — running per-entity retrieval for: %s",
                entity_names,
            )
            per_entity_k = max(3, k)
            seen_ids: set[str] = set()
            merged: list[dict] = []

            for name in entity_names:
                # Determine which collection this entity belongs to
                name_lower = name.lower()
                if any(p.lower() == name_lower for p in PEOPLE):
                    etype = "person"
                elif any(p.lower() == name_lower for p in PLACES):
                    etype = "place"
                else:
                    etype = category  # fallback

                embedding = self._embedder.get_embedding(f"Tell me about {name}")
                if embedding is None:
                    continue

                chunks = self._vs.query(embedding, etype, k=per_entity_k)
                for c in chunks:
                    cid = (
                        c.get("metadata", {}).get("source_title", "")
                        + str(c.get("metadata", {}).get("chunk_index", ""))
                    )
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(c)

            merged.sort(key=lambda c: c["distance"])
            chunks = merged[:k]
        else:
            # ── Single-entity or generic query ───────────────────────────
            embedding = self._embedder.get_embedding(query)
            if embedding is None:
                logger.error("Failed to embed query — is the model loaded?")
                return {"category": category, "chunks": [], "error": "embedding_failed"}

            if category == "both":
                person_chunks = self._vs.query(embedding, "person", k=k)
                place_chunks = self._vs.query(embedding, "place", k=k)
                all_chunks = person_chunks + place_chunks
                all_chunks.sort(key=lambda c: c["distance"])
                chunks = all_chunks[:k]
            else:
                chunks = self._vs.query(embedding, category, k=k)

        logger.info("Retrieved %d chunks for query.", len(chunks))
        return {"category": category, "chunks": chunks}
