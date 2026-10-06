"""Document ingestion and listing API.

Endpoints
---------
POST /api/v1/documents/ingest
    Upload a PDF or TXT file for text extraction, chunking, embedding, and
    vector storage.  Returns the saved document metadata.

GET /api/v1/documents
    List all stored document metadata records.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.schemas.ingestion import ChunkStrategy, DocumentOut
from app.services.chunking import get_chunker
from app.services.embedding import get_embedding_service
from app.services.extraction import extract_text
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({"application/pdf", "text/plain"})


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_repo(db: Annotated[Session, Depends(get_db)]) -> DocumentRepository:
    """FastAPI dependency: provide a :class:`~app.repositories.document_repository.DocumentRepository`."""
    return DocumentRepository(db)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document (PDF or TXT)",
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="PDF or TXT file (max 10 MB)")],
    chunk_strategy: Annotated[
        ChunkStrategy,
        Form(description="Chunking strategy: 'fixed' or 'recursive'"),
    ] = ChunkStrategy.RECURSIVE,
    repo: Annotated[DocumentRepository, Depends(_get_repo)] = ...,  # type: ignore[assignment]
) -> DocumentOut:
    """Upload a document, extract text, chunk, embed, and store in Qdrant.

    Parameters
    ----------
    file:
        Multipart file upload.  Must be ``.pdf`` or ``.txt``, non-empty,
        and at most 10 MB.
    chunk_strategy:
        ``"fixed"`` uses a sliding character window; ``"recursive"`` splits
        hierarchically on paragraphs → sentences → words.
    repo:
        Injected :class:`~app.repositories.document_repository.DocumentRepository`.

    Returns
    -------
    DocumentOut
        Metadata of the newly created document record.

    Raises
    ------
    HTTPException
        * **400** – Unsupported file type, empty file, or file exceeds 10 MB.
        * **422** – Invalid *chunk_strategy* value (handled by FastAPI/Pydantic).
    """
    # ------------------------------------------------------------------ #
    # 1. Validate file type                                                #
    # ------------------------------------------------------------------ #
    filename: str = file.filename or ""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    content_type: str = (file.content_type or "").split(";")[0].strip().lower()

    if suffix not in _ALLOWED_EXTENSIONS and content_type not in _ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected file '%s' with content_type='%s'", filename, content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix or content_type}'. "
                "Only .pdf and .txt files are accepted."
            ),
        )

    # ------------------------------------------------------------------ #
    # 2. Read & size-check                                                 #
    # ------------------------------------------------------------------ #
    content: bytes = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File size {len(content) / (1024 * 1024):.1f} MB exceeds the 10 MB limit."
            ),
        )

    logger.info(
        "Ingesting '%s' (strategy=%s, bytes=%d)",
        filename,
        chunk_strategy.value,
        len(content),
    )

    # ------------------------------------------------------------------ #
    # 3. Extract text                                                      #
    # ------------------------------------------------------------------ #
    try:
        text = extract_text(content, content_type or suffix, filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text could be extracted from the uploaded file.",
        )

    # ------------------------------------------------------------------ #
    # 4. Chunk                                                             #
    # ------------------------------------------------------------------ #
    chunker = get_chunker(chunk_strategy.value)
    chunks = chunker.split(text)
    logger.info("'%s' produced %d chunks.", filename, len(chunks))

    # ------------------------------------------------------------------ #
    # 5. Embed                                                             #
    # ------------------------------------------------------------------ #
    embedding_svc = get_embedding_service()
    vectors = embedding_svc.embed_texts(chunks)

    # ------------------------------------------------------------------ #
    # 6. Upsert into Qdrant                                                #
    # ------------------------------------------------------------------ #
    vector_store = get_vector_store()
    vector_store.ensure_collection()

    # Determine file_type from extension or content-type
    file_type = suffix.lstrip(".") if suffix else (
        "pdf" if "pdf" in content_type else "txt"
    )

    # Create the DB record first to get the document_id
    doc = Document(
        filename=filename,
        file_type=file_type,
        chunk_strategy=chunk_strategy.value,
        chunk_count=len(chunks),
        char_count=len(text),
    )
    saved_doc = repo.save(doc)

    vector_store.upsert_chunks(
        document_id=saved_doc.id,
        filename=filename,
        chunks=chunks,
        vectors=vectors,
    )

    logger.info(
        "Document id='%s' ('%s') ingested successfully (%d chunks).",
        saved_doc.id,
        filename,
        len(chunks),
    )
    return DocumentOut.model_validate(saved_doc)


@router.get(
    "",
    response_model=list[DocumentOut],
    summary="List all ingested documents",
)
def list_documents(
    repo: Annotated[DocumentRepository, Depends(_get_repo)] = ...,  # type: ignore[assignment]
) -> list[DocumentOut]:
    """Return metadata for all previously ingested documents.

    Parameters
    ----------
    repo:
        Injected :class:`~app.repositories.document_repository.DocumentRepository`.

    Returns
    -------
    list[DocumentOut]
        All document records, newest first.
    """
    docs = repo.list_all()
    logger.debug("Listing %d documents.", len(docs))
    return [DocumentOut.model_validate(d) for d in docs]
