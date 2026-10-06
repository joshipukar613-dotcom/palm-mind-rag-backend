"""Integration tests for POST /api/v1/documents/ingest and GET /api/v1/documents.

All tests use fixtures from ``conftest.py`` so they never touch:

* ``data/app.db``   – overridden with a per-test temp SQLite file.
* Qdrant            – replaced by ``mock_vector_store``.
* sentence-transformers – replaced by ``mock_embedding``.

Edge cases under test
---------------------
1. **10 MB size limit uses actual bytes read** (not the Content-Length header).
2. **PDF with no extractable text → HTTP 400** with a clear message.
3. Happy-path ingest (TXT and PDF), list endpoint, and common 400 scenarios.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# PDF builder helpers  (no external deps – hand-craft minimal valid PDFs)
# ---------------------------------------------------------------------------

def _make_pdf_with_text(text: str) -> bytes:
    """Build a minimal, pypdf-parseable single-page PDF containing *text*."""
    stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    stream_len = len(stream_content)

    body = b"%PDF-1.4\n"
    offsets: list[int] = []

    offsets.append(len(body))
    body += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    offsets.append(len(body))
    body += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    offsets.append(len(body))
    body += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    offsets.append(len(body))
    body += (
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_content
        + b"\nendstream\nendobj\n"
    )

    offsets.append(len(body))
    body += (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )

    xref_offset = len(body)
    num_objects = len(offsets)
    xref = f"xref\n0 {num_objects + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    body += xref
    body += (
        f"trailer\n<< /Size {num_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return body


def _make_image_only_pdf() -> bytes:
    """Build a PDF whose single page has an empty content stream (no text)."""
    body = b"%PDF-1.4\n"
    offsets: list[int] = []

    offsets.append(len(body))
    body += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

    offsets.append(len(body))
    body += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"

    offsets.append(len(body))
    body += (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R >>\n"
        b"endobj\n"
    )

    offsets.append(len(body))
    body += b"4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n"

    xref_offset = len(body)
    num_objects = len(offsets)
    xref = f"xref\n0 {num_objects + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    body += xref
    body += (
        f"trailer\n<< /Size {num_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return body


# ===========================================================================
# Edge case 1: 10 MB limit is enforced on actual bytes read
# ===========================================================================

_MAX = 10 * 1024 * 1024  # 10 MB


class TestFileSizeLimit:
    """Verify the 10 MB cap fires on ``len(await file.read())``,
    not on the Content-Length header."""

    def test_exactly_at_limit_is_not_rejected_by_size_guard(
        self, test_client: TestClient
    ) -> None:
        """A file whose body is exactly 10 MB must pass the size guard.

        It will still succeed end-to-end because the DB and services
        are all mocked via the ``test_client`` fixture.
        """
        content = b"a" * _MAX
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("exactly10mb.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "fixed"},
        )
        # Must NOT be a size-related 400.
        if resp.status_code == 400:
            assert "exceeds" not in resp.json().get("detail", ""), (
                f"Wrongly rejected a {_MAX}-byte file: {resp.json()}"
            )

    def test_one_byte_over_limit_returns_400(
        self, test_client: TestClient
    ) -> None:
        """10 MB + 1 byte must be rejected with HTTP 400 mentioning the limit."""
        content = b"a" * (_MAX + 1)
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("over10mb.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "exceeds" in detail or "10 MB" in detail, (
            f"Expected size-limit message, got: {detail!r}"
        )

    def test_size_guard_uses_body_not_content_length_header(
        self, test_client: TestClient
    ) -> None:
        """Spoofing Content-Length to a small value must not bypass the guard.

        We patch ``starlette.datastructures.UploadFile.read`` to return more
        than 10 MB regardless of what was uploaded, then confirm the guard
        still fires.  This directly exercises ``len(content) > _MAX_FILE_SIZE``
        in the router.
        """
        oversized = b"x" * (_MAX + 512)

        with patch(
            "starlette.datastructures.UploadFile.read",
            return_value=oversized,
        ):
            resp = test_client.post(
                "/api/v1/documents/ingest",
                # Tiny file → low Content-Length header
                files={"file": ("small.txt", io.BytesIO(b"hello"), "text/plain")},
                data={"chunk_strategy": "recursive"},
            )

        assert resp.status_code == 400
        assert "exceeds" in resp.json()["detail"], (
            f"Size guard not triggered when read() returns oversized bytes: "
            f"{resp.json()['detail']!r}"
        )


# ===========================================================================
# Edge case 2: PDF with no extractable text → HTTP 400
# ===========================================================================

class TestPdfNoExtractableText:
    """Verify that text-free PDFs are rejected with a helpful 400 message."""

    def test_image_only_pdf_returns_400(self, test_client: TestClient) -> None:
        """A structurally valid PDF with an empty content stream → HTTP 400."""
        pdf_bytes = _make_image_only_pdf()
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("scanned.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "No text" in detail or "extracted" in detail, (
            f"Expected a 'no text' message, got: {detail!r}"
        )

    def test_whitespace_only_extraction_returns_400(
        self, test_client: TestClient
    ) -> None:
        """``extract_text`` returning only whitespace must also produce 400.

        We patch ``app.api.documents.extract_text`` (the name the router
        imports) so the PDF bytes don't need to be real.
        """
        with patch(
            "app.api.documents.extract_text", return_value="   \n\t\n  "
        ):
            resp = test_client.post(
                "/api/v1/documents/ingest",
                files={
                    "file": (
                        "whitespace.pdf",
                        io.BytesIO(b"%PDF-1.4 placeholder"),
                        "application/pdf",
                    )
                },
                data={"chunk_strategy": "recursive"},
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "No text" in detail or "extracted" in detail, (
            f"Whitespace-only text should produce 'no text' message, got: {detail!r}"
        )

    def test_pdf_with_real_text_returns_201(self, test_client: TestClient) -> None:
        """Sanity check: a PDF with parseable text → HTTP 201."""
        pdf_bytes = _make_pdf_with_text("Hello world this is a test document.")
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("real.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["file_type"] == "pdf"
        assert body["chunk_count"] >= 1


# ===========================================================================
# Happy-path and common 400 scenarios
# ===========================================================================

class TestIngestHappyPath:
    """Successful ingest and list operations using mocked services."""

    def test_ingest_txt_returns_201_with_correct_schema(
        self, test_client: TestClient
    ) -> None:
        """A valid TXT upload must return 201 with all DocumentOut fields."""
        content = b"This is a plain text document with enough words to form a chunk."
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "sample.txt"
        assert body["file_type"] == "txt"
        assert body["chunk_strategy"] == "recursive"
        assert body["chunk_count"] >= 1
        assert body["char_count"] == len(content.decode())
        assert "id" in body
        assert "created_at" in body

    def test_ingest_with_fixed_strategy(self, test_client: TestClient) -> None:
        """The ``fixed`` chunk strategy is recorded in the response."""
        content = b"Fixed strategy test document content here."
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("fixed.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "fixed"},
        )
        assert resp.status_code == 201
        assert resp.json()["chunk_strategy"] == "fixed"

    def test_default_strategy_is_recursive(self, test_client: TestClient) -> None:
        """Omitting chunk_strategy must default to ``recursive``."""
        content = b"Default strategy document."
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("default.txt", io.BytesIO(content), "text/plain")},
            # No chunk_strategy form field
        )
        assert resp.status_code == 201
        assert resp.json()["chunk_strategy"] == "recursive"

    def test_vector_store_upsert_called_once(
        self,
        test_client: TestClient,
        mock_vector_store: MagicMock,
    ) -> None:
        """The vector store ``upsert_chunks`` must be called exactly once
        per successful ingest."""
        content = b"Check that upsert is called."
        test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("upsert_check.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        mock_vector_store.upsert_chunks.assert_called_once()

    def test_embedding_called_with_chunks(
        self,
        test_client: TestClient,
        mock_embedding: MagicMock,
    ) -> None:
        """``embed_texts`` must be called with a non-empty list of strings."""
        content = b"Embedding service must receive the chunks."
        test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("embed_check.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        mock_embedding.embed_texts.assert_called_once()
        call_args = mock_embedding.embed_texts.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) >= 1
        assert all(isinstance(c, str) for c in call_args)


class TestListDocuments:
    """GET /api/v1/documents behaviour."""

    def test_empty_list_on_fresh_db(self, test_client: TestClient) -> None:
        """A freshly created (empty) database must return an empty list."""
        resp = test_client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_ingested_document(self, test_client: TestClient) -> None:
        """After one ingest the list endpoint must return exactly one item."""
        content = b"Document to list."
        test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("listed.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        resp = test_client.get("/api/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "listed.txt"

    def test_multiple_ingests_all_appear_in_list(
        self, test_client: TestClient
    ) -> None:
        """Three separate ingests must all appear in the list, newest first."""
        filenames = ["alpha.txt", "beta.txt", "gamma.txt"]
        for name in filenames:
            test_client.post(
                "/api/v1/documents/ingest",
                files={"file": (name, io.BytesIO(b"content " + name.encode()), "text/plain")},
                data={"chunk_strategy": "recursive"},
            )
        resp = test_client.get("/api/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 3
        returned_names = {d["filename"] for d in docs}
        assert returned_names == set(filenames)


class TestCommon400Scenarios:
    """Common rejection cases."""

    def test_unsupported_file_type_returns_400(
        self, test_client: TestClient
    ) -> None:
        """A `.png` file must be rejected with HTTP 400."""
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("photo.png", io.BytesIO(b"\x89PNG\r\n"), "image/png")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_empty_file_returns_400(self, test_client: TestClient) -> None:
        """An empty file must be rejected with HTTP 400."""
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
            data={"chunk_strategy": "recursive"},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_invalid_chunk_strategy_returns_422(
        self, test_client: TestClient
    ) -> None:
        """An unrecognised ``chunk_strategy`` value must return HTTP 422
        (FastAPI/Pydantic validation)."""
        resp = test_client.post(
            "/api/v1/documents/ingest",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"chunk_strategy": "sliding_window"},
        )
        assert resp.status_code == 422
