"""Text chunking strategies behind a common Protocol.

Two strategies are provided:

* :class:`FixedSizeChunker` – splits text by a fixed character window with overlap.
* :class:`RecursiveChunker` – splits hierarchically (paragraphs → sentences → words)
  until every chunk fits within the configured size.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Chunker(Protocol):
    """Common interface for all chunking strategies."""

    def split(self, text: str) -> list[str]:
        """Split *text* into a list of chunks.

        Parameters
        ----------
        text:
            The full document text to be chunked.

        Returns
        -------
        list[str]
            Non-empty list of text chunks (empty strings are filtered out).
        """
        ...


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------

class FixedSizeChunker:
    """Splits text into fixed-size character windows with overlap.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters repeated between consecutive chunks.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        settings = get_settings()
        self.chunk_size: int = chunk_size if chunk_size is not None else settings.chunk_size
        self.overlap: int = overlap if overlap is not None else settings.chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    def split(self, text: str) -> list[str]:
        """Split *text* using a sliding window of :attr:`chunk_size` chars.

        Parameters
        ----------
        text:
            Full document text.

        Returns
        -------
        list[str]
            List of character chunks.
        """
        if not text:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        logger.debug(
            "FixedSizeChunker produced %d chunks (size=%d, overlap=%d)",
            len(chunks),
            self.chunk_size,
            self.overlap,
        )
        return chunks


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------

# Ordered list of separators: try paragraph breaks first, then sentence
# boundaries, then individual words.
_SEPARATORS: list[str] = ["\n\n", r"(?<=[.!?])\s+", r"\s+"]


class RecursiveChunker:
    """Hierarchical chunker that respects natural text boundaries.

    Splits by paragraphs first, then sentence boundaries, then whitespace,
    recursing until every piece fits within :attr:`chunk_size`.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters of context carried over to the next chunk.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        settings = get_settings()
        self.chunk_size: int = chunk_size if chunk_size is not None else settings.chunk_size
        self.overlap: int = overlap if overlap is not None else settings.chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> list[str]:
        """Recursively split *text* respecting natural language boundaries.

        Parameters
        ----------
        text:
            Full document text.

        Returns
        -------
        list[str]
            List of text chunks, each at most :attr:`chunk_size` chars.
        """
        if not text:
            return []

        raw_chunks = self._recursive_split(text, separator_index=0)
        merged = self._merge_with_overlap(raw_chunks)
        logger.debug(
            "RecursiveChunker produced %d chunks (size=%d, overlap=%d)",
            len(merged),
            self.chunk_size,
            self.overlap,
        )
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        """Split *text* using the separator at *separator_index*, recursing if needed."""
        if len(text) <= self.chunk_size:
            stripped = text.strip()
            return [stripped] if stripped else []

        if separator_index >= len(_SEPARATORS):
            # No more separators – hard-cut the text.
            pieces: list[str] = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i : i + self.chunk_size].strip()
                if chunk:
                    pieces.append(chunk)
            return pieces

        sep_pattern = _SEPARATORS[separator_index]
        parts = re.split(sep_pattern, text)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Part is still too large – recurse with the next separator.
                result.extend(self._recursive_split(part, separator_index + 1))

        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces greedily and apply overlap between chunks.

        Pieces are merged left-to-right until the next piece would exceed
        :attr:`chunk_size`.  When a chunk is finalised, the last
        :attr:`overlap` characters are prepended to the next chunk to
        provide context continuity.
        """
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""

        for piece in pieces:
            separator = " " if current else ""
            candidate = current + separator + piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Carry overlap from the end of the finalised chunk.
                    tail = current[-self.overlap :] if self.overlap else ""
                    current = (tail + " " + piece).strip() if tail else piece
                else:
                    # The single piece is already too large (shouldn't happen
                    # after recursive splitting, but guard defensively).
                    chunks.append(piece[: self.chunk_size])
                    current = piece[self.chunk_size :]

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_chunker(strategy: str) -> Chunker:
    """Return the appropriate :class:`Chunker` for *strategy*.

    Parameters
    ----------
    strategy:
        One of ``"fixed"`` or ``"recursive"``.

    Returns
    -------
    Chunker
        A concrete chunker instance.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    if strategy == "fixed":
        return FixedSizeChunker()
    if strategy == "recursive":
        return RecursiveChunker()
    raise ValueError(f"Unknown chunk strategy: '{strategy}'. Choose 'fixed' or 'recursive'.")
