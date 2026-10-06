"""Text extraction from PDF and plain-text files."""

from __future__ import annotations

import logging
from io import BytesIO

logger = logging.getLogger(__name__)

_SUPPORTED_TYPES: frozenset[str] = frozenset({"application/pdf", "text/plain"})


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    """Extract plain text from *content* bytes.

    Parameters
    ----------
    content:
        Raw file bytes.
    content_type:
        MIME type of the file (e.g. ``"application/pdf"`` or ``"text/plain"``).
    filename:
        Original filename, used only for logging context.

    Returns
    -------
    str
        The extracted plain text.

    Raises
    ------
    ValueError
        If *content_type* is not supported.
    """
    logger.debug("Extracting text from '%s' (type=%s, bytes=%d)", filename, content_type, len(content))

    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return _extract_pdf(content, filename)
    if content_type == "text/plain" or filename.lower().endswith(".txt"):
        return _extract_txt(content, filename)

    raise ValueError(f"Unsupported content type '{content_type}' for file '{filename}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf(content: bytes, filename: str) -> str:
    """Use *pypdf* to extract text from all pages of a PDF."""
    try:
        from pypdf import PdfReader  # imported lazily to keep startup fast
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is not installed; run `pip install pypdf`") from exc

    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    logger.debug("PDF '%s' → %d pages, %d chars", filename, len(reader.pages), len(text))
    return text


def _extract_txt(content: bytes, filename: str) -> str:
    """Decode a plain-text file as UTF-8."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed for '%s'; falling back to latin-1", filename)
        text = content.decode("latin-1")
    logger.debug("TXT '%s' → %d chars", filename, len(text))
    return text
