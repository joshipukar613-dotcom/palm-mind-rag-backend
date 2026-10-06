"""Tests for the interview booking flow and bookings API.

All tests are fully isolated:
* SQLite uses a temporary database per test (via tmp_db_session fixture).
* Redis is mocked (mock_redis or local fake/mock).
* LLM is mocked (mock_llm).
* Vector store & embeddings are mocked.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import get_db
from app.main import app
from app.models.booking import Booking
from app.repositories.booking_repository import BookingRepository
from app.services.booking import BookingProcessResult, BookingService, _parse_llm_json
from app.services.memory import ChatMemory


class FakeRedis:
    """In-memory Redis fake for fast, isolated tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        return True

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


@pytest.fixture()
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture()
def booking_repo(tmp_db_session: sessionmaker) -> BookingRepository:
    session = tmp_db_session()
    return BookingRepository(session)


# ===========================================================================
# Unit Tests for JSON Parsing & Helper logic
# ===========================================================================

class TestJsonParsing:
    def test_parses_clean_json(self) -> None:
        raw = '{"intent": "book", "extracted": {"name": "Alice"}}'
        parsed = _parse_llm_json(raw)
        assert parsed["intent"] == "book"
        assert parsed["extracted"]["name"] == "Alice"

    def test_parses_markdown_code_fence(self) -> None:
        raw = '```json\n{"intent": "book", "extracted": {"name": "Bob"}}\n```'
        parsed = _parse_llm_json(raw)
        assert parsed["intent"] == "book"
        assert parsed["extracted"]["name"] == "Bob"

    def test_handles_reasoning_before_json(self) -> None:
        raw = 'Thinking: user wants to book.\n{"intent": "book", "extracted": {"email": "a@b.com"}}'
        parsed = _parse_llm_json(raw)
        assert parsed["intent"] == "book"
        assert parsed["extracted"]["email"] == "a@b.com"

    def test_malformed_json_never_crashes(self) -> None:
        raw = "This is not json at all {broken"
        parsed = _parse_llm_json(raw)
        assert parsed["intent"] == "other"
        assert parsed["extracted"] == {}


# ===========================================================================
# BookingService Unit Tests
# ===========================================================================

class TestBookingServiceUnit:
    def _create_service(
        self,
        fake_redis: FakeRedis,
        llm_reply: str = '{"intent": "other", "extracted": {}}',
    ) -> BookingService:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=llm_reply)

        mock_memory = MagicMock()
        mock_memory.load_history.return_value = []
        mock_memory.append_messages.return_value = None

        return BookingService(
            llm=mock_llm,
            memory=mock_memory,
            redis_client=fake_redis,  # type: ignore[arg-type]
            ttl_seconds=3600,
        )

    def test_multi_turn_collection(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        """Turn 1: user provides name + email -> state saved, asks for date/time.
        Turn 2: user provides date + time -> booking saved in DB, state cleared."""
        tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()

        # Turn 1
        turn1_json = (
            '{"intent": "book", "extracted": '
            '{"name": "Alice Smith", "email": "ALICE@EXAMPLE.COM", "date": null, "time": null}}'
        )
        svc1 = self._create_service(fake_redis, turn1_json)
        res1 = asyncio.run(
            svc1.handle_message("sess_multi", "Hi I'm Alice Smith, alice@example.com", booking_repo)
        )

        assert res1.booking_id is None
        assert "date" in res1.answer.lower()
        # State in Redis must have normalized email and name
        state = svc1.get_state("sess_multi")
        assert state["name"] == "Alice Smith"
        assert state["email"] == "alice@example.com"

        # Turn 2
        turn2_json = (
            f'{{"intent": "book", "extracted": '
            f'{{"name": null, "email": null, "date": "{tomorrow}", "time": "14:00"}}}}'
        )
        svc2 = self._create_service(fake_redis, turn2_json)
        res2 = asyncio.run(
            svc2.handle_message("sess_multi", f"Tomorrow at 2pm", booking_repo)
        )

        assert res2.booking_id is not None
        assert "confirmed" in res2.answer.lower()
        assert "Alice Smith" in res2.answer
        assert res2.booking_id in res2.answer

        # Booking saved in DB
        db_booking = booking_repo.get(res2.booking_id)
        assert db_booking is not None
        assert db_booking.name == "Alice Smith"
        assert db_booking.email == "alice@example.com"
        assert db_booking.booking_date.isoformat() == tomorrow
        assert db_booking.booking_time == time(14, 0)

        # Redis state cleared
        assert svc2.get_state("sess_multi") == {}

    def test_bad_email_validation(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        turn_json = (
            '{"intent": "book", "extracted": '
            '{"name": "Bob", "email": "not-an-email", "date": null, "time": null}}'
        )
        svc = self._create_service(fake_redis, turn_json)
        res = asyncio.run(
            svc.handle_message("sess_bad_email", "Bob, not-an-email", booking_repo)
        )

        assert "valid email" in res.answer.lower()
        assert res.booking_id is None
        # Invalid email must NOT be saved in state
        state = svc.get_state("sess_bad_email")
        assert "email" not in state
        assert state.get("name") == "Bob"

    def test_short_name_rejected(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        turn_json = (
            '{"intent": "book", "extracted": '
            '{"name": "X", "email": "x@example.com", "date": null, "time": null}}'
        )
        svc = self._create_service(fake_redis, turn_json)
        res = asyncio.run(
            svc.handle_message("sess_short_name", "I am X, x@example.com", booking_repo)
        )

        assert "at least 2 characters" in res.answer.lower()
        state = svc.get_state("sess_short_name")
        assert "name" not in state

    def test_past_date_rejected(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        turn_json = (
            f'{{"intent": "book", "extracted": '
            f'{{"name": "Charlie", "email": "charlie@example.com", "date": "{yesterday}", "time": "10:00"}}}}'
        )
        svc = self._create_service(fake_redis, turn_json)
        res = asyncio.run(
            svc.handle_message("sess_past", "Book yesterday at 10am", booking_repo)
        )

        assert "cannot be in the past" in res.answer.lower()
        state = svc.get_state("sess_past")
        assert "booking_date" not in state

    def test_cancel_clears_state(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        fake_redis.set("booking:sess_cancel", '{"name": "Dan"}')

        cancel_json = '{"intent": "cancel", "extracted": {}}'
        svc = self._create_service(fake_redis, cancel_json)
        res = asyncio.run(
            svc.handle_message("sess_cancel", "Never mind, cancel booking", booking_repo)
        )

        assert "cancelled" in res.answer.lower()
        assert svc.get_state("sess_cancel") == {}

    def test_interruption_by_doc_question(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        # User already provided name and email
        fake_redis.set(
            "booking:sess_interrupt",
            '{"name": "Eve", "email": "eve@example.com"}',
        )

        inquiry_json = '{"intent": "inquiry", "extracted": {}}'
        svc = self._create_service(fake_redis, inquiry_json)
        res = asyncio.run(
            svc.handle_message("sess_interrupt", "What are your benefits?", booking_repo)
        )

        assert res.delegate_to_rag is True
        assert res.is_interruption is True
        assert res.missing_fields_reminder is not None
        assert "date" in res.missing_fields_reminder.lower()

    def test_inquiry_without_active_booking_delegates_cleanly(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        # No state in Redis
        inquiry_json = '{"intent": "inquiry", "extracted": {}}'
        svc = self._create_service(fake_redis, inquiry_json)
        res = asyncio.run(
            svc.handle_message("sess_no_booking", "Tell me about the company", booking_repo)
        )

        assert res.delegate_to_rag is True
        assert res.is_interruption is False
        assert res.missing_fields_reminder is None

    def test_double_booking_prevented(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        future_date = datetime.now(timezone.utc).date() + timedelta(days=3)
        future_time = time(15, 30)

        # Existing booking in DB for same slot
        existing = Booking(
            session_id="other_sess",
            name="Existing User",
            email="existing@example.com",
            booking_date=future_date,
            booking_time=future_time,
        )
        booking_repo.save(existing)

        # New user attempts same slot
        turn_json = (
            f'{{"intent": "book", "extracted": '
            f'{{"name": "Frank", "email": "frank@example.com", '
            f'"date": "{future_date.isoformat()}", "time": "15:30"}}}}'
        )
        svc = self._create_service(fake_redis, turn_json)
        res = asyncio.run(
            svc.handle_message("sess_double", "Book for 3 days from now at 15:30", booking_repo)
        )

        assert "already booked" in res.answer.lower()
        assert res.booking_id is None
        # The conflicting slot should not create a second booking
        all_bookings = booking_repo.list_all()
        assert len(all_bookings) == 1

    def test_turns_saved_to_chat_memory(
        self, fake_redis: FakeRedis, booking_repo: BookingRepository
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value='{"intent": "cancel", "extracted": {}}'
        )
        mock_mem = MagicMock()
        mock_mem.load_history.return_value = []

        svc = BookingService(
            llm=mock_llm,
            memory=mock_mem,
            redis_client=fake_redis,  # type: ignore[arg-type]
        )
        asyncio.run(svc.handle_message("sess_mem", "cancel please", booking_repo))

        mock_mem.append_messages.assert_called_once()
        call_session_id, call_msgs = mock_mem.append_messages.call_args.args
        assert call_session_id == "sess_mem"
        assert call_msgs[0]["role"] == "user"
        assert call_msgs[0]["content"] == "cancel please"
        assert call_msgs[1]["role"] == "assistant"


# ===========================================================================
# Endpoint Integration Tests
# ===========================================================================

class TestBookingsApiEndpoint:
    def test_list_bookings_empty(self, tmp_db_session: sessionmaker) -> None:
        client = TestClient(app)
        resp = client.get("/api/v1/bookings")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_bookings_returns_saved_records(
        self, tmp_db_session: sessionmaker
    ) -> None:
        session: Session = tmp_db_session()
        repo = BookingRepository(session)
        b = Booking(
            session_id="sess_123",
            name="Grace Hopper",
            email="grace@example.com",
            booking_date=date(2026, 11, 1),
            booking_time=time(11, 0),
        )
        repo.save(b)

        client = TestClient(app)
        resp = client.get("/api/v1/bookings")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["name"] == "Grace Hopper"
        assert items[0]["email"] == "grace@example.com"
        assert items[0]["booking_date"] == "2026-11-01"
        assert items[0]["booking_time"] == "11:00:00"
        assert items[0]["id"] == b.id


class TestChatWithBookingIntegration:
    def test_chat_returns_booking_id_when_completed(
        self,
        tmp_db_session: sessionmaker,
        mock_rag_service: MagicMock,
    ) -> None:
        """When BookingService completes a booking, POST /api/v1/chat returns
        the answer and booking_id."""
        mock_booking_svc = MagicMock()
        mock_booking_svc.handle_message = AsyncMock(
            return_value=BookingProcessResult(
                answer="Confirmed for Alice on 2026-11-02 at 14:00. Booking ID: b-123.",
                booking_id="b-123",
            )
        )

        app.dependency_overrides[
            "app.services.booking.get_booking_service"
        ] = lambda: mock_booking_svc

        from app.services.booking import get_booking_service
        app.dependency_overrides[get_booking_service] = lambda: mock_booking_svc

        try:
            client = TestClient(app)
            resp = client.post(
                "/api/v1/chat",
                json={"session_id": "test_sess", "message": "Book it"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["booking_id"] == "b-123"
            assert "Confirmed for Alice" in data["answer"]
        finally:
            app.dependency_overrides.pop(get_booking_service, None)

    def test_chat_interruption_combines_rag_and_reminder(
        self,
        tmp_db_session: sessionmaker,
        mock_rag_service: MagicMock,
    ) -> None:
        from app.services.booking import get_booking_service
        from app.services.rag import RagAnswer

        mock_booking_svc = MagicMock()
        mock_booking_svc.handle_message = AsyncMock(
            return_value=BookingProcessResult(
                answer="",
                delegate_to_rag=True,
                is_interruption=True,
                missing_fields_reminder="Please provide your preferred date and time.",
            )
        )
        mock_rag_service.answer = AsyncMock(
            return_value=RagAnswer(
                answer="Our policy is 20 days PTO.", sources=["handbook.pdf"]
            )
        )

        app.dependency_overrides[get_booking_service] = lambda: mock_booking_svc
        try:
            client = TestClient(app)
            resp = client.post(
                "/api/v1/chat",
                json={"session_id": "test_sess", "message": "What is the PTO policy?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "Our policy is 20 days PTO." in data["answer"]
            assert "Please provide your preferred date and time." in data["answer"]
            assert data["sources"] == ["handbook.pdf"]
            assert data["booking_id"] is None
        finally:
            app.dependency_overrides.pop(get_booking_service, None)
