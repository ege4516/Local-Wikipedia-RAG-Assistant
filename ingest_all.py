"""
Standalone ingestion script.

Run from the project root:
    python ingest_all.py

Pipeline:
    Wikipedia → SQLite (raw text + chunks) → ChromaDB (vectors)

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — no Ollama
dependency for embedding.  Ollama is only needed for the LLM at query time.
"""

import logging
import sys
import time

from config import PEOPLE, PLACES
from db.sqlite_store import SQLiteStore
from db.vector_store import VectorStore
from ingest.chunker import TextChunker
from ingest.embedder import OllamaEmbedder
from ingest.wikipedia_scraper import WikipediaScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest_all")


def ingest_one(
    title: str,
    entity_type: str,
    sqlite_store: SQLiteStore,
    vector_store: VectorStore,
    chunker: TextChunker,
    embedder: OllamaEmbedder,
    scraper: WikipediaScraper,
) -> bool:
    """Ingest a single Wikipedia entity. Returns True on success."""
    logger.info("→ %s (%s)", title, entity_type)
    data = scraper.scrape_page(title)
    if not data:
        logger.warning("  skipped: page not found")
        return False

    entity_id = sqlite_store.insert_entity(
        title=data["title"],
        entity_type=entity_type,
        url=data["url"],
        raw_text=data["content"],
    )

    # Wipe stale chunks/vectors so re-runs do not duplicate
    sqlite_store.delete_chunks_for_entity(entity_id)
    vector_store.delete_entity(data["title"], entity_type)

    chunks = chunker.chunk_text(
        text=data["content"],
        source_title=data["title"],
        entity_type=entity_type,
        url=data["url"],
    )
    if not chunks:
        logger.warning("  skipped: no chunks produced")
        return False

    for c in chunks:
        sqlite_store.insert_chunk(
            entity_id=entity_id,
            chunk_index=c["metadata"]["chunk_index"],
            chunk_text=c["text"],
            token_count=c["token_count"],
        )

    texts = [c["text"] for c in chunks]
    embeddings = embedder.get_embeddings_batch(texts)

    valid = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    if not valid:
        logger.error("  embedding failed for all chunks")
        return False

    vc, ve = zip(*valid)
    vector_store.add_chunks(list(vc), list(ve), entity_type)
    logger.info("  done — %d chunks embedded", len(valid))
    return True


def main() -> int:
    sqlite_store = SQLiteStore()
    vector_store = VectorStore()
    chunker = TextChunker()
    embedder = OllamaEmbedder()
    scraper = WikipediaScraper()

    started = time.time()
    successes = failures = 0

    for title in PEOPLE:
        if ingest_one(title, "person", sqlite_store, vector_store, chunker, embedder, scraper):
            successes += 1
        else:
            failures += 1

    for title in PLACES:
        if ingest_one(title, "place", sqlite_store, vector_store, chunker, embedder, scraper):
            successes += 1
        else:
            failures += 1

    elapsed = time.time() - started
    counts = vector_store.collection_counts()
    logger.info("=" * 60)
    logger.info(
        "Done in %.1fs — successes=%d failures=%d | people_vectors=%d place_vectors=%d",
        elapsed,
        successes,
        failures,
        counts["people_collection"],
        counts["places_collection"],
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
