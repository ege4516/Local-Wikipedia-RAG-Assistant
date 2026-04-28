"""
Fixed-size token chunker with overlap.
Uses tiktoken for accurate token counting so chunk boundaries are
reproducible and independent of whitespace normalisation.
"""

import logging
from typing import Generator

import tiktoken

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TIKTOKEN_ENCODING

logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
        encoding_name: str = TIKTOKEN_ENCODING,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    # ── Core logic ─────────────────────────────────────────────────────────

    def _token_windows(self, tokens: list[int]) -> Generator[list[int], None, None]:
        """Yield sliding windows of token ids."""
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(tokens):
            yield tokens[start : start + self.chunk_size]
            start += step

    def chunk_text(
        self,
        text: str,
        source_title: str,
        entity_type: str,
        url: str,
    ) -> list[dict]:
        """
        Split *text* into overlapping token chunks.

        Returns a list of dicts:
            {
                "text": <decoded chunk text>,
                "metadata": {
                    "source_title": ...,
                    "entity_type": ...,
                    "chunk_index": ...,
                    "url": ...,
                },
                "token_count": <int>,
            }
        """
        if not text.strip():
            return []

        tokens = self._enc.encode(text)
        chunks = []

        for idx, window in enumerate(self._token_windows(tokens)):
            chunk_text = self._enc.decode(window)
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source_title": source_title,
                        "entity_type": entity_type,
                        "chunk_index": idx,
                        "url": url,
                    },
                    "token_count": len(window),
                }
            )

        logger.debug(
            "Chunked '%s' → %d chunks (%d tokens total).",
            source_title,
            len(chunks),
            len(tokens),
        )
        return chunks
