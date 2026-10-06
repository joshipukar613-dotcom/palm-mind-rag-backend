"""Tests for RagService and the chat API endpoints.

All tests are fully isolated — no Redis, no Qdrant, no LLM, no
sentence-transformers model is ever contacted.

Test groups
-----------
TestRagServiceUnit
    Direct unit tests for :class:`~app.services.rag.RagService`.
    Dependencies are provided as plain MagicMock / AsyncMock objects.

TestChatEndpoint
    Integration tests against ``POST /api/v1/chat`` and
    ``DELETE /api/v1/chat/{session_id}`` via the ``chat_client`` fixture
    (which patches ``get_rag_service`` at the router level).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.services.memory import MAX_MESSAGES, ChatMemory, Message
from app.services.rag import (
    _NO_CONTEXT_ANSWER,
    _RELEVANCE_THRESHOLD,
    RagAnswer,
    RagService,
)

# ===========================================================================
# Helpers
# ===========================================================================

def _make_rag_service(
    llm_reply: str = "LLM answer.",
    search_hits: list[dict] | None = None,
    history: list[Message] | None = None,
) -> tuple[RagService, MagicMock, MagicMock, MagicMock, AsyncMock]:
    """Build a RagService with all dependencies mocked.

    Returns
    -------
    tuple of (service, mock_llm, mock_memory, mock_embedding, mock_vs)
    """
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=llm_reply)

    mock_memory = MagicMock()
    mock_memory.load_history.return_value = history or []
    mock_memory.append_messages.return_value = None

    mock_embedding = MagicMock()
    mock_embedding.embed_texts.return_value = [[0.0] * 384]

    mock_vs = MagicMock()
    mock_vs.search.return_value = search_hits if search_hits is not None else []

    svc = RagService(
        llm=mock_llm,
        memory=mock_memory,
        embedding=mock_embedding,
        vector_store=mock_vs,
    )
    return svc, mock_llm, mock_memory, mock_embedding, mock_vs


# ===========================================================================
# RagService unit tests
# ===========================================================================

class TestRagServiceUnit:
    """Direct unit tests for RagService.answer()."""

    # ── No-context case ──────────────────────────────────────────────────

    def test_no_hits_returns_fallback_answer(self) -> None:
        """When Qdrant returns zero chunks the answer must be the canned
        fallback message and sources must be empty."""
        svc, _, _, _, _ = _make_rag_service(search_hits=[])
        result = asyncio.run(svc.answer("sess1", "What is X?"))
        assert result.answer == _NO_CONTEXT_ANSWER
        assert result.sources == []

    def test_low_score_hits_return_fallback(self) -> None:
        """Hits whose score is below _RELEVANCE_THRESHOLD must be filtered out,
        producing the same canned fallback."""
        low_score_hit = {
            "text": "some text",
            "filename": "doc.pdf",
            "score": _RELEVANCE_THRESHOLD - 0.01,
        }
        svc, _, _, _, _ = _make_rag_service(search_hits=[low_score_hit])
        result = asyncio.run(svc.answer("sess1", "What is X?"))
        assert result.answer == _NO_CONTEXT_ANSWER
        assert result.sources == []

    def test_no_context_does_not_call_llm_for_answer(self) -> None:
        """When there is no relevant context the LLM must NOT be called for
        the final answer (it may still be called for query rewriting if there
        is history)."""
        svc, mock_llm, _, _, _ = _make_rag_service(search_hits=[], history=[])
        asyncio.run(svc.answer("sess1", "What is X?"))
        mock_llm.complete.assert_not_called()

    # ── Happy-path ───────────────────────────────────────────────────────

    def test_relevant_hits_call_llm_and_return_answer(self) -> None:
        """With relevant hits the LLM is called and the result is returned."""
        hit = {"text": "Relevant text.", "filename": "report.pdf", "score": 0.85}
        svc, mock_llm, _, _, _ = _make_rag_service(
            llm_reply="The answer is 42.", search_hits=[hit]
        )
        result = asyncio.run(svc.answer("sess1", "What is the answer?"))
        assert result.answer == "The answer is 42."
        assert "report.pdf" in result.sources
        mock_llm.complete.assert_called_once()

    def test_sources_are_deduplicated(self) -> None:
        """Multiple hits from the same file must appear only once in sources."""
        hits = [
            {"text": "chunk 1", "filename": "doc.pdf", "score": 0.9},
            {"text": "chunk 2", "filename": "doc.pdf", "score": 0.8},
            {"text": "chunk 3", "filename": "other.pdf", "score": 0.7},
        ]
        svc, _, _, _, _ = _make_rag_service(search_hits=hits)
        result = asyncio.run(svc.answer("sess1", "Tell me something."))
        assert result.sources.count("doc.pdf") == 1
        assert "other.pdf" in result.sources

    # ── Query rewriting ─────────────────────────────────────────────────

    def test_no_history_skips_rewrite(self) -> None:
        """With no chat history the query-rewrite LLM call must be skipped."""
        hit = {"text": "content", "filename": "f.pdf", "score": 0.9}
        svc, mock_llm, _, mock_embedding, _ = _make_rag_service(
            search_hits=[hit], history=[]
        )
        asyncio.run(svc.answer("sess1", "Hello?"))
        # LLM called exactly once: for the final RAG answer, not for rewrite.
        mock_llm.complete.assert_called_once()
        # Embedding called with the original message, not a rewritten query.
        mock_embedding.embed_texts.assert_called_once_with(["Hello?"])

    def test_with_history_rewrites_query(self) -> None:
        """With existing history the LLM is called FIRST for rewriting (with
        max_tokens=128) and THEN for the final answer."""
        history: list[Message] = [
            Message(role="user", content="Who wrote the report?"),
            Message(role="assistant", content="Dr Smith wrote it."),
        ]
        hit = {"text": "content", "filename": "f.pdf", "score": 0.9}
        # First LLM call returns the rewritten query, second returns the answer.
        mock_llm_obj = MagicMock()
        mock_llm_obj.complete = AsyncMock(
            side_effect=["standalone query about Dr Smith", "Final answer."]
        )
        mock_memory = MagicMock()
        mock_memory.load_history.return_value = history
        mock_memory.append_messages.return_value = None

        mock_embedding = MagicMock()
        mock_embedding.embed_texts.return_value = [[0.0] * 384]

        mock_vs = MagicMock()
        mock_vs.search.return_value = [hit]

        svc = RagService(
            llm=mock_llm_obj,
            memory=mock_memory,
            embedding=mock_embedding,
            vector_store=mock_vs,
        )
        result = asyncio.run(svc.answer("sess1", "What did he say?"))

        assert mock_llm_obj.complete.call_count == 2
        # First call must have max_tokens=128 (rewrite call)
        first_call_kwargs = mock_llm_obj.complete.call_args_list[0]
        assert first_call_kwargs.kwargs.get("max_tokens") == 128 or \
               first_call_kwargs.args[1:] == (128,)
        # Embedding must use the REWRITTEN query, not the original.
        mock_embedding.embed_texts.assert_called_once_with(
            ["standalone query about Dr Smith"]
        )
        assert result.answer == "Final answer."

    # ── Memory persistence ───────────────────────────────────────────────

    def test_user_and_assistant_messages_saved_to_memory(self) -> None:
        """After answering, both the user message and the assistant reply
        must be appended to memory in the correct order."""
        hit = {"text": "context", "filename": "src.pdf", "score": 0.9}
        svc, _, mock_memory, _, _ = _make_rag_service(
            llm_reply="Answer text.", search_hits=[hit]
        )
        asyncio.run(svc.answer("sess42", "My question."))
        mock_memory.append_messages.assert_called_once()
        _, appended = mock_memory.append_messages.call_args.args
        assert appended[0] == Message(role="user", content="My question.")
        assert appended[1] == Message(role="assistant", content="Answer text.")

    def test_memory_saved_even_when_no_context(self) -> None:
        """Memory must be updated even when no relevant chunks were found."""
        svc, _, mock_memory, _, _ = _make_rag_service(search_hits=[])
        asyncio.run(svc.answer("sessX", "Unknown question."))
        mock_memory.append_messages.assert_called_once()
        _, appended = mock_memory.append_messages.call_args.args
        assert appended[0]["role"] == "user"
        assert appended[1]["role"] == "assistant"


# ===========================================================================
# ChatMemory unit tests
# ===========================================================================

class TestChatMemory:
    """Unit tests for ChatMemory using a mocked Redis client."""

    def _make_memory(self, mock_redis_client: MagicMock) -> ChatMemory:
        with patch(
            "app.services.memory.redis_lib.from_url",
            return_value=mock_redis_client,
        ):
            return ChatMemory(redis_url="redis://fake", ttl_seconds=3600)

    def test_load_history_empty_when_key_missing(self) -> None:
        """``load_history`` must return [] when Redis returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mem = self._make_memory(mock_client)
        assert mem.load_history("sess1") == []

    def test_load_history_returns_parsed_messages(self) -> None:
        """``load_history`` must deserialise the JSON stored in Redis."""
        msgs = [{"role": "user", "content": "hi"}]
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(msgs)
        mem = self._make_memory(mock_client)
        assert mem.load_history("sess1") == msgs

    def test_append_messages_trims_to_max(self) -> None:
        """After appending, the stored list must never exceed MAX_MESSAGES."""
        # Pre-fill with MAX_MESSAGES - 1 entries
        existing = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(MAX_MESSAGES - 1)
        ]
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps(existing)
        mem = self._make_memory(mock_client)

        # Append 2 more (user + assistant) — total would be MAX_MESSAGES + 1
        new_msgs = [
            Message(role="user", content="new question"),
            Message(role="assistant", content="new answer"),
        ]
        mem.append_messages("sess1", new_msgs)

        # Inspect what was written to Redis
        written_json = mock_client.set.call_args.args[1]
        written: list[dict] = json.loads(written_json)
        assert len(written) == MAX_MESSAGES

    def test_append_messages_sets_ttl(self) -> None:
        """``append_messages`` must call ``redis.set`` with ``ex=ttl``."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mem = self._make_memory(mock_client)
        mem.append_messages("sess1", [Message(role="user", content="hello")])
        _, kwargs = mock_client.set.call_args
        # ex keyword argument should match the configured TTL
        assert kwargs.get("ex") == 3600

    def test_clear_returns_true_when_key_exists(self) -> None:
        mock_client = MagicMock()
        mock_client.delete.return_value = 1   # Redis: 1 key deleted
        mem = self._make_memory(mock_client)
        assert mem.clear("sess1") is True

    def test_clear_returns_false_when_key_missing(self) -> None:
        mock_client = MagicMock()
        mock_client.delete.return_value = 0   # Redis: 0 keys deleted
        mem = self._make_memory(mock_client)
        assert mem.clear("sess1") is False


# ===========================================================================
# Chat endpoint integration tests
# ===========================================================================

class TestChatEndpoint:
    """Tests for POST /api/v1/chat using the ``chat_client`` fixture."""

    def test_post_chat_returns_200_with_correct_schema(
        self, chat_client: TestClient
    ) -> None:
        """A well-formed request must return 200 with session_id, answer,
        sources, and booking_id."""
        resp = chat_client.post(
            "/api/v1/chat",
            json={"session_id": "test-session", "message": "Hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "test-session"
        assert isinstance(body["answer"], str)
        assert isinstance(body["sources"], list)
        assert body["booking_id"] is None

    def test_post_chat_returns_rag_answer(
        self, chat_client: TestClient, mock_rag_service: MagicMock
    ) -> None:
        """The endpoint answer must match what RagService.answer() returns."""

        mock_rag_service.answer = AsyncMock(
            return_value=RagAnswer(answer="Custom answer.", sources=["x.pdf"])
        )
        resp = chat_client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "message": "Query"},
        )
        assert resp.json()["answer"] == "Custom answer."
        assert resp.json()["sources"] == ["x.pdf"]

    def test_post_chat_missing_session_id_returns_422(
        self, chat_client: TestClient
    ) -> None:
        resp = chat_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert resp.status_code == 422

    def test_post_chat_empty_message_returns_422(
        self, chat_client: TestClient
    ) -> None:
        resp = chat_client.post(
            "/api/v1/chat",
            json={"session_id": "s1", "message": ""},
        )
        assert resp.status_code == 422

    def test_rag_service_called_with_correct_args(
        self, chat_client: TestClient, mock_rag_service: MagicMock
    ) -> None:
        """RagService.answer must be called with the session_id and message
        from the request body."""
        chat_client.post(
            "/api/v1/chat",
            json={"session_id": "my-session", "message": "What is AI?"},
        )
        mock_rag_service.answer.assert_called_once_with(
            session_id="my-session", message="What is AI?"
        )


class TestDeleteSessionEndpoint:
    """Tests for DELETE /api/v1/chat/{session_id}."""

    def test_delete_existing_session_returns_204(
        self, chat_client: TestClient, mock_rag_service: MagicMock
    ) -> None:
        mock_rag_service._memory.clear.return_value = True
        resp = chat_client.delete("/api/v1/chat/existing-session")
        assert resp.status_code == 204

    def test_delete_nonexistent_session_returns_404(
        self, chat_client: TestClient, mock_rag_service: MagicMock
    ) -> None:
        mock_rag_service._memory.clear.return_value = False
        resp = chat_client.delete("/api/v1/chat/ghost-session")
        assert resp.status_code == 404
        assert "ghost-session" in resp.json()["detail"]

    def test_delete_calls_memory_clear(
        self, chat_client: TestClient, mock_rag_service: MagicMock
    ) -> None:
        chat_client.delete("/api/v1/chat/sess-abc")
        mock_rag_service._memory.clear.assert_called_once_with("sess-abc")
