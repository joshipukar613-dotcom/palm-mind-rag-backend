"""Repository for :class:`~app.models.document.Document` records.

All database access for documents is centralised here so that the router
layer stays thin and free from SQLAlchemy details.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    """CRUD operations for :class:`~app.models.document.Document`.

    Parameters
    ----------
    db:
        An active SQLAlchemy :class:`~sqlalchemy.orm.Session`.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, document: Document) -> Document:
        """Persist *document* to the database and return it with its auto-set fields.

        Parameters
        ----------
        document:
            A :class:`~app.models.document.Document` instance (not yet added
            to the session).

        Returns
        -------
        Document
            The same instance after ``commit`` and ``refresh``.
        """
        self._db.add(document)
        self._db.commit()
        self._db.refresh(document)
        logger.info(
            "Saved document id='%s' filename='%s' chunks=%d",
            document.id,
            document.filename,
            document.chunk_count,
        )
        return document

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> Document | None:
        """Return the :class:`~app.models.document.Document` with *document_id*, or ``None``.

        Parameters
        ----------
        document_id:
            Primary-key UUID string.

        Returns
        -------
        Document | None
            The matched record, or ``None`` if not found.
        """
        doc = self._db.get(Document, document_id)
        logger.debug("get(document_id='%s') → %s", document_id, "found" if doc else "not found")
        return doc

    def list_all(self) -> list[Document]:
        """Return all :class:`~app.models.document.Document` records ordered by creation date.

        Returns
        -------
        list[Document]
            All stored documents, newest first.
        """
        docs = (
            self._db.query(Document)
            .order_by(Document.created_at.desc())
            .all()
        )
        logger.debug("list_all() → %d documents", len(docs))
        return docs
