"""Embedding service wrapping *sentence-transformers*.

The underlying model is loaded once per process via :func:`get_embedding_service`
(backed by :func:`functools.lru_cache`) and shared across all requests.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model '%s' …", model_name)
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import]
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed; run `pip install sentence-transformers`"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model '%s' loaded successfully.", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* into dense embedding vectors.

        Parameters
        ----------
        texts:
            List of strings to embed.  Empty list returns an empty list.

        Returns
        -------
        list[list[float]]
            One embedding vector per input string.
        """
        if not texts:
            return []

        logger.debug("Embedding %d texts with model '%s'", len(texts), self._model_name)
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    @property
    def model_name(self) -> str:
        """The HuggingFace model identifier in use."""
        return self._model_name


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the application-wide :class:`EmbeddingService` (singleton).

    The instance is created on first call and cached for the lifetime of the
    process, so the heavy model loading only happens once.

    Returns
    -------
    EmbeddingService
        Shared embedding service instance.
    """
    settings = get_settings()
    return EmbeddingService(settings.embedding_model)
