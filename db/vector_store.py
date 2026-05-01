"""
ChromaDB vector store wrapper.
Maintains two persistent collections: people_collection and places_collection.
Embeddings are always supplied externally (from sentence-transformers all-MiniLM-L6-v2, 384-dim).
"""

import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_PATH, PEOPLE_COLLECTION, PLACES_COLLECTION

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str = CHROMA_DB_PATH):
        os.makedirs(path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        # IMPORTANT: embedding_function=None disables Chroma's automatic
        # embedding. We always supply embeddings externally (from
        # sentence-transformers all-MiniLM-L6-v2, 384-dim). Without this flag,
        # Chroma 1.x silently re-embeds documents with its own default model
        # and ignores the vectors we pass in — corrupting the index.
        self._people = self._client.get_or_create_collection(
            PEOPLE_COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self._places = self._client.get_or_create_collection(
            PLACES_COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — people=%d docs, places=%d docs",
            self._people.count(),
            self._places.count(),
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _collection(self, entity_type: str):
        if entity_type == "person":
            return self._people
        if entity_type == "place":
            return self._places
        raise ValueError(f"Unknown entity_type: {entity_type!r}")

    @staticmethod
    def _chunk_id(title: str, chunk_index: int) -> str:
        safe = title.replace(" ", "_").replace("/", "-")
        return f"{safe}__{chunk_index}"

    # ── Public API ─────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        entity_type: str,
    ) -> None:
        """
        Insert chunks into the appropriate collection.

        chunks   : list of {text, metadata}
        embeddings: parallel list of embedding vectors
        """
        if not chunks:
            return

        col = self._collection(entity_type)
        ids = [
            self._chunk_id(c["metadata"]["source_title"], c["metadata"]["chunk_index"])
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # ChromaDB upsert avoids duplicates on re-ingestion
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d chunks into '%s'.", len(chunks), col.name
        )

    def query(
        self,
        query_embedding: list[float],
        entity_type: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Query a single collection; returns up to k results sorted by relevance.
        Each result: {text, metadata, distance}
        """
        col = self._collection(entity_type)
        if col.count() == 0:
            logger.warning("Collection '%s' is empty.", col.name)
            return []

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, col.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def delete_entity(self, title: str, entity_type: str) -> None:
        """Remove all chunks for a given entity title."""
        col = self._collection(entity_type)
        col.delete(where={"source_title": title})
        logger.info("Deleted chunks for '%s' from '%s'.", title, col.name)

    def collection_counts(self) -> dict[str, int]:
        return {
            PEOPLE_COLLECTION: self._people.count(),
            PLACES_COLLECTION: self._places.count(),
        }

    def reset_all(self) -> None:
        """Delete all vectors from both collections — full system reset."""
        self._client.delete_collection(PEOPLE_COLLECTION)
        self._client.delete_collection(PLACES_COLLECTION)
        # Re-create empty collections so the app keeps working
        self._people = self._client.get_or_create_collection(
            PEOPLE_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        self._places = self._client.get_or_create_collection(
            PLACES_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore reset — both collections cleared.")
