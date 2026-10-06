"""Qdrant vector store wrapper.

Provides :class:`QdrantVectorStore` with:

* :meth:`~QdrantVectorStore.ensure_collection` – idempotently create the
  collection with cosine distance and the configured embedding dimension.
* :meth:`~QdrantVectorStore.upsert_chunks` – store a batch of text chunks
  with document metadata as payload.
* :meth:`~QdrantVectorStore.search` – retrieve the *top_k* nearest chunks
  to a query vector.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Thin wrapper around the Qdrant Python client.

    Parameters
    ----------
    url:
        URL of the Qdrant instance (e.g. ``"http://localhost:6333"``).
    collection:
        Name of the target collection.
    dim:
        Embedding dimension used when creating the collection.
    """

    def __init__(self, url: str, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qdrant-client is not installed; run `pip install qdrant-client`"
            ) from exc

        self._client = QdrantClient(url=url, timeout=10)
        self._collection = collection
        self._dim = dim
        logger.info(
            "QdrantVectorStore initialised (url=%s, collection=%s, dim=%d)",
            url,
            collection,
            dim,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist.

        Uses cosine distance and the dimension supplied at construction time.
        Safe to call repeatedly (idempotent).
        """
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            logger.debug("Collection '%s' already exists – skipping creation.", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d, cosine).", self._collection, self._dim)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """Upsert text chunks and their embeddings into the collection.

        Parameters
        ----------
        document_id:
            UUID of the parent :class:`~app.models.document.Document`.
        filename:
            Original filename, stored as payload metadata.
        chunks:
            Plain-text content of each chunk.
        vectors:
            Embedding vector for each chunk (parallel to *chunks*).

        Raises
        ------
        ValueError
            If *chunks* and *vectors* have different lengths.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length."
            )

        from qdrant_client.models import PointStruct  # type: ignore[import]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        self._client.upsert(collection_name=self._collection, points=points)
        logger.debug(
            "Upserted %d chunks for document_id='%s' into collection '%s'.",
            len(points),
            document_id,
            self._collection,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve the *top_k* most similar chunks to *query_vector*.

        Parameters
        ----------
        query_vector:
            Dense embedding of the query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict[str, Any]]
            Each element is the *payload* dict of a matched chunk,
            augmented with a ``"score"`` key.
        """
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for hit in results:
            payload: dict[str, Any] = dict(hit.payload or {})
            payload["score"] = hit.score
            hits.append(payload)

        logger.debug("Search returned %d hits (top_k=%d).", len(hits), top_k)
        return hits


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return the application-wide :class:`QdrantVectorStore` (singleton).

    Returns
    -------
    QdrantVectorStore
        Shared vector store instance.
    """
    settings = get_settings()
    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )
