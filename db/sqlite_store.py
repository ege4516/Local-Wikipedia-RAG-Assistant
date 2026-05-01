"""
SQLite persistence layer.
Stores raw Wikipedia content and chunk metadata before vectors are built.
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import Optional

from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


class SQLiteStore:
    def __init__(self, db_path: str = SQLITE_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._create_tables()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS entities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    NOT NULL UNIQUE,
            entity_type TEXT    NOT NULL,   -- 'person' or 'place'
            url         TEXT,
            raw_text    TEXT,
            ingested_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   INTEGER NOT NULL REFERENCES entities(id),
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT    NOT NULL,
            token_count INTEGER NOT NULL,
            created_at  TEXT    NOT NULL
        );
        """
        with self._connect() as conn:
            conn.executescript(sql)
        logger.debug("SQLite tables ensured.")

    # ── Public API ─────────────────────────────────────────────────────────

    def entity_exists(self, title: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM entities WHERE title = ?", (title,)
            ).fetchone()
        return row is not None

    def insert_entity(
        self,
        title: str,
        entity_type: str,
        url: str,
        raw_text: str,
    ) -> int:
        """Insert or replace an entity row; returns its row id."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO entities (title, entity_type, url, raw_text, ingested_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(title) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    url         = excluded.url,
                    raw_text    = excluded.raw_text,
                    ingested_at = excluded.ingested_at
                """,
                (title, entity_type, url, raw_text, now),
            )
            entity_id = cur.lastrowid
            # lastrowid is 0 on UPDATE — fetch the real id
            if entity_id == 0:
                entity_id = conn.execute(
                    "SELECT id FROM entities WHERE title = ?", (title,)
                ).fetchone()["id"]
        logger.info("Upserted entity '%s' (id=%d).", title, entity_id)
        return entity_id

    def insert_chunk(
        self,
        entity_id: int,
        chunk_index: int,
        chunk_text: str,
        token_count: int,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chunks (entity_id, chunk_index, chunk_text, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entity_id, chunk_index, chunk_text, token_count, now),
            )

    def delete_chunks_for_entity(self, entity_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE entity_id = ?", (entity_id,))

    def get_all_entities(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, entity_type, url, ingested_at FROM entities"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def get_chunk_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def reset_all(self) -> None:
        """Delete all entities and chunks — full system reset."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM entities")
        logger.info("SQLite reset — all entities and chunks deleted.")
