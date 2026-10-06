"""Shared pytest fixtures for the Palm Mind RAG test suite.

Isolation guarantees
--------------------
* **SQLite** – ``tmp_db_session`` creates a per-test temp file and overrides
  ``get_db`` so ``data/app.db`` is never touched.
* **Qdrant** – ``mock_vector_store`` patches both the documents and chat
  routers so no network call is made.
* **sentence-transformers** – ``mock_embedding`` patches both routers so the
  model is never loaded.
* **Redis** – ``mock_redis`` patches ``app.services.memory.redis_lib.from_url``
  so no real Redis connection is made during chat tests.
* **LLM** – ``mock_llm`` patches ``LLMClient.complete`` so no HTTP call to
  the LLM API is made.

Composite fixtures
------------------
``test_client``     – Wraps the documents API with all services mocked.
``chat_client``     – Wraps the chat API; patches ``get_rag_service`` with a
                      controllable ``MagicMock``.
``mock_rag_service``– The MagicMock RagService injected into the chat router.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base, get_db
from app.main import app

# ---------------------------------------------------------------------------
# Embedding mock  (patches both the documents and RAG routers)
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384
_ZERO_VECTOR: list[float] = [0.0] * _EMBEDDING_DIM


@pytest.fixture()
def mock_embedding() -> Generator[MagicMock, None, None]:
    """Replace the embedding service singleton with a MagicMock.

    ``embed_texts(texts)`` returns one zero-vector per input so neither
    the ingest endpoint nor the RAG pipeline loads the model.

    Yields
    ------
    MagicMock
        The shared mock instance.
    """
    mock_svc = MagicMock()
    mock_svc.embed_texts.side_effect = lambda texts: [_ZERO_VECTOR for _ in texts]

    with (
        patch("app.api.documents.get_embedding_service", return_value=mock_svc),
        patch("app.services.embedding.get_embedding_service", return_value=mock_svc),
    ):
        yield mock_svc


# ---------------------------------------------------------------------------
# Vector store mock  (patches both routers)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_vector_store() -> Generator[MagicMock, None, None]:
    """Replace the Qdrant vector store singleton with a MagicMock.

    Yields
    ------
    MagicMock
        The shared mock instance.
    """
    mock_vs = MagicMock()
    mock_vs.ensure_collection.return_value = None
    mock_vs.upsert_chunks.return_value = None
    mock_vs.search.return_value = []

    with (
        patch("app.api.documents.get_vector_store", return_value=mock_vs),
        patch("app.services.vector_store.get_vector_store", return_value=mock_vs),
    ):
        yield mock_vs


# ---------------------------------------------------------------------------
# Redis / ChatMemory mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_redis() -> Generator[MagicMock, None, None]:
    """Replace the Redis client used inside ChatMemory with a MagicMock.

    ``get`` returns ``None`` (empty history) by default; ``set`` and
    ``delete`` are no-ops.  Individual tests can override via
    ``mock_redis.get.return_value = ...``.

    Yields
    ------
    MagicMock
        The mock Redis client object.
    """
    mock_client = MagicMock()
    mock_client.get.return_value = None   # empty history by default
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1

    with patch("app.services.memory.redis_lib.from_url", return_value=mock_client):
        yield mock_client


# ---------------------------------------------------------------------------
# LLM mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> Generator[AsyncMock, None, None]:
    """Replace ``LLMClient.complete`` with an AsyncMock.

    Default return value is ``"Mocked LLM answer."``.

    Yields
    ------
    AsyncMock
        The patched coroutine method.
    """
    with patch(
        "app.services.llm.LLMClient.complete",
        new_callable=AsyncMock,
        return_value="Mocked LLM answer.",
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Temporary SQLite DB + get_db override
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db_session() -> Generator[sessionmaker, None, None]:
    """Create an isolated SQLite database in a temp file for a single test.

    * A fresh temp file is created before each test and deleted after.
    * All ORM tables are created via ``Base.metadata.create_all``.
    * ``get_db`` is overridden so ``data/app.db`` is never touched.

    Yields
    ------
    sessionmaker
        Bound to the temporary engine.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="test_palm_")
    os.close(fd)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    Base.metadata.create_all(bind=engine)

    def _override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    yield TestingSessionLocal

    app.dependency_overrides.pop(get_db, None)
    engine.dispose()
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Composite: documents test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_client(
    tmp_db_session: sessionmaker,
    mock_embedding: MagicMock,
    mock_vector_store: MagicMock,
) -> TestClient:
    """TestClient with all document-API services replaced by safe fakes.

    Returns
    -------
    TestClient
        Ready-to-use HTTPX-backed test client.
    """
    return TestClient(app)


# ---------------------------------------------------------------------------
# Composite: chat test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_rag_service() -> Generator[MagicMock, None, None]:
    """Replace ``get_rag_service`` with a controllable MagicMock.

    The mock's ``answer`` coroutine returns a default
    :class:`~app.services.rag.RagAnswer`.  Override in tests::

        mock_rag_service.answer = AsyncMock(return_value=RagAnswer("hi", []))

    Yields
    ------
    MagicMock
        The mock RagService instance injected into the chat router.
    """
    from app.services.rag import RagAnswer, get_rag_service

    mock_svc = MagicMock()
    mock_svc.answer = AsyncMock(
        return_value=RagAnswer(answer="Mocked RAG answer.", sources=["doc.pdf"])
    )
    mock_svc._memory = MagicMock()
    mock_svc._memory.clear.return_value = True

    app.dependency_overrides[get_rag_service] = lambda: mock_svc
    try:
        yield mock_svc
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


@pytest.fixture()
def chat_client(mock_rag_service: MagicMock) -> TestClient:
    """TestClient for chat endpoints with the entire RAG pipeline mocked.

    No Redis, no LLM, no Qdrant, no sentence-transformers.

    Returns
    -------
    TestClient
        Ready-to-use HTTPX-backed test client for chat tests.
    """
    return TestClient(app)
