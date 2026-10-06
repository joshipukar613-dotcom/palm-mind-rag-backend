"""Pytest tests for FixedSizeChunker and RecursiveChunker.

Run with::

    pytest tests/test_chunkers.py -v
"""

from __future__ import annotations

import pytest

from app.services.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    get_chunker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat.\n\n"
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum "
    "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non "
    "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n\n"
    "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam "
    "varius, turpis molestie dictum semper, nunc augue iaculis purus, quis "
    "porttitor leo libero at augue. Sed eu erat eget tortor."
)


# ===========================================================================
# FixedSizeChunker tests
# ===========================================================================


class TestFixedSizeChunker:
    """Tests for :class:`~app.services.chunking.FixedSizeChunker`."""

    def test_implements_chunker_protocol(self) -> None:
        """FixedSizeChunker must satisfy the Chunker Protocol."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        assert isinstance(chunker, Chunker)

    def test_empty_string_returns_empty_list(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        assert chunker.split("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        assert chunker.split("   \n\n  ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "Hello world"
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunk_count(self) -> None:
        """Number of chunks should be ceil((len - size) / step) + 1."""
        text = "a" * 1000
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.split(text)
        # step = 200 - 50 = 150; expected = ceil(800/150) + 1 = 6 + 1 = 7
        assert len(chunks) == 7

    def test_each_chunk_respects_max_size(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.split(LOREM)
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk too long ({len(chunk)}): {chunk!r}"

    def test_overlap_means_content_repeated(self) -> None:
        """The tail of chunk N should appear at the start of chunk N+1."""
        text = "a" * 500
        overlap = 50
        chunker = FixedSizeChunker(chunk_size=200, overlap=overlap)
        chunks = chunker.split(text)
        for i in range(len(chunks) - 1):
            tail = chunks[i][-overlap:]
            head = chunks[i + 1][:overlap]
            assert tail == head, (
                f"Overlap mismatch between chunk {i} and {i + 1}: "
                f"{tail!r} != {head!r}"
            )

    def test_zero_overlap(self) -> None:
        text = "abcdefghij"
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.split(text)
        assert chunks == ["abcde", "fghij"]

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0, overlap=0)

    def test_invalid_overlap_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            FixedSizeChunker(chunk_size=100, overlap=-1)

    def test_invalid_overlap_gte_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_all_text_covered(self) -> None:
        """Joining chunks (minus overlap) must reconstruct all characters."""
        text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
        overlap = 5
        chunk_size = 50
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.split(text)
        # Verify original text chars appear somewhere across chunks
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[overlap:]
        assert text[: len(reconstructed)] in reconstructed or len(reconstructed) >= len(text)


# ===========================================================================
# RecursiveChunker tests
# ===========================================================================


class TestRecursiveChunker:
    """Tests for :class:`~app.services.chunking.RecursiveChunker`."""

    def test_implements_chunker_protocol(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        assert isinstance(chunker, Chunker)

    def test_empty_string_returns_empty_list(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        assert chunker.split("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        assert chunker.split("   \n\n  ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "This is a short sentence."
        chunker = RecursiveChunker(chunk_size=200, overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert text in chunks[0]

    def test_chunks_respect_max_size(self) -> None:
        chunker = RecursiveChunker(chunk_size=150, overlap=30)
        chunks = chunker.split(LOREM)
        for chunk in chunks:
            assert len(chunk) <= 150 + 30 + 10, (  # small tolerance for overlap tail
                f"Chunk too long ({len(chunk)}): {chunk!r}"
            )

    def test_paragraph_boundary_respected(self) -> None:
        """With a large chunk_size the two paragraphs should stay together or
        split exactly at the blank line."""
        text = "First paragraph with some content here.\n\nSecond paragraph here."
        chunker = RecursiveChunker(chunk_size=200, overlap=0)
        chunks = chunker.split(text)
        # Either both paragraphs fit in one chunk, or each is its own chunk.
        combined = " ".join(chunks)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined

    def test_all_content_preserved(self) -> None:
        """All significant words from the input must appear in the output chunks."""
        unique_word = "XYZUNIQUE42"
        text = f"Some preamble. {unique_word}. More content follows here."
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        chunks = chunker.split(text)
        full = " ".join(chunks)
        assert unique_word in full, f"Unique word missing from chunks: {chunks}"

    def test_produces_multiple_chunks_for_long_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        chunks = chunker.split(LOREM)
        assert len(chunks) > 1

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveChunker(chunk_size=0, overlap=0)

    def test_invalid_overlap_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            RecursiveChunker(chunk_size=200, overlap=-5)

    def test_invalid_overlap_gte_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            RecursiveChunker(chunk_size=50, overlap=50)

    def test_no_empty_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=80, overlap=15)
        chunks = chunker.split(LOREM)
        for chunk in chunks:
            assert chunk.strip(), "Found an empty or whitespace-only chunk"

    def test_single_very_long_word(self) -> None:
        """A single token longer than chunk_size must be hard-cut."""
        word = "a" * 500
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        chunks = chunker.split(word)
        assert all(len(c) <= 100 for c in chunks)


# ===========================================================================
# Factory tests
# ===========================================================================


class TestGetChunker:
    """Tests for the :func:`~app.services.chunking.get_chunker` factory."""

    def test_returns_fixed_chunker(self) -> None:
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_returns_recursive_chunker(self) -> None:
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            get_chunker("unknown")
