"""Interview booking service.

Handles:
* Intent classification (book, cancel, inquiry, other)
* Entity extraction (name, email, date, time) strictly from user input
* Redis-backed conversational booking state per session
* Validation against :class:`~app.schemas.chat.BookingDetails`
* Double-booking prevention
* Integration with :class:`~app.services.memory.ChatMemory`
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, time, timezone
from typing import Any, NamedTuple

from pydantic import EmailStr, TypeAdapter
import redis as redis_lib

from app.core.config import get_settings
from app.models.booking import Booking
from app.repositories.booking_repository import BookingRepository
from app.services.llm import LLMClient
from app.services.memory import ChatMemory, Message

logger = logging.getLogger(__name__)

_EMAIL_ADAPTER = TypeAdapter(EmailStr)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an interview booking assistant and intent classifier.

Today's date is: {today_str} ({weekday}).

Current booking state for this session:
{current_state_json}

Your task:
Analyze the user's latest message in context of the conversation and determine:
1. "intent":
   - "cancel": User explicitly wants to cancel, abort, or stop the booking process (e.g. "cancel", "never mind", "stop booking", "don't want to book anymore").
   - "book": User wants to book an interview or is providing interview details (name, email, date, time).
   - "inquiry": User is asking a document, product, company, policy, or informational question.
   - "other": General greetings, small talk, or unrelated remarks.

2. "extracted":
   STRICT EXTRACTION RULES:
   - Extract ONLY values that the user EXPLICITLY stated in their messages.
   - NEVER guess, assume, or invent a name, email, date, or time.
   - If a field was not explicitly stated by the user, it MUST be null.
   - For dates: resolve relative expressions ("tomorrow", "next Monday", "this Friday") relative to today's date ({today_str}) into YYYY-MM-DD.
   - For times: format in 24-hour time HH:MM (e.g. "14:00" for 2pm, "09:30" for 9:30 AM).

Respond ONLY with a valid JSON object in this format (no markdown, no extra commentary):
{{
  "intent": "book" | "cancel" | "inquiry" | "other",
  "extracted": {{
    "name": string | null,
    "email": string | null,
    "date": "YYYY-MM-DD" | null,
    "time": "HH:MM" | null
  }}
}}
"""


def _booking_key(session_id: str) -> str:
    return f"booking:{session_id}"


def _clean_json_text(raw_text: str) -> str:
    """Strip markdown code blocks or surrounding text from LLM response."""
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        return text[first_brace : last_brace + 1]
    return text


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    """Parse JSON from LLM output safely, never raising an unhandled exception."""
    cleaned = _clean_json_text(raw_text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse LLM booking JSON (%s): %r", exc, raw_text)
    return {"intent": "other", "extracted": {}}


class BookingProcessResult(NamedTuple):
    """Result of processing a message in the booking flow."""

    answer: str
    booking_id: str | None = None
    delegate_to_rag: bool = False
    is_interruption: bool = False
    missing_fields_reminder: str | None = None


class BookingService:
    """Manages interview booking intent detection, extraction, and validation."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        memory: ChatMemory | None = None,
        redis_client: redis_lib.Redis | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self._llm = llm or LLMClient()
        self._memory = memory or ChatMemory()
        self._ttl: int = ttl_seconds if ttl_seconds is not None else settings.chat_ttl_seconds
        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis_lib.from_url(settings.redis_url, decode_responses=True)

    # -----------------------------------------------------------------------
    # Redis Booking State Management
    # -----------------------------------------------------------------------

    def get_state(self, session_id: str) -> dict[str, str]:
        """Return the current partial booking state dictionary for *session_id*."""
        key = _booking_key(session_id)
        raw = self._redis.get(key)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_state(self, session_id: str, state: dict[str, str]) -> None:
        """Save partial booking state to Redis with configured TTL."""
        key = _booking_key(session_id)
        self._redis.set(key, json.dumps(state), ex=self._ttl)

    def clear_state(self, session_id: str) -> bool:
        """Remove booking state for *session_id* from Redis."""
        key = _booking_key(session_id)
        return bool(self._redis.delete(key))

    # -----------------------------------------------------------------------
    # LLM Intent & Entity Extraction
    # -----------------------------------------------------------------------

    async def _extract_intent_and_entities(
        self, session_id: str, message: str, current_state: dict[str, str]
    ) -> dict[str, Any]:
        """Ask LLM to classify intent and extract booking entities."""
        today = datetime.now(timezone.utc).date()
        today_str = today.isoformat()
        weekday = today.strftime("%A")

        system_prompt = _EXTRACTION_SYSTEM.format(
            today_str=today_str,
            weekday=weekday,
            current_state_json=json.dumps(current_state),
        )

        history = self._memory.load_history(session_id)
        # Include last few history turns for context
        history_snippet = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
        )
        user_prompt = (
            f"Conversation context:\n{history_snippet}\n\n"
            f"Latest User Message:\n{message}"
        )

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_reply = await self._llm.complete(llm_messages, max_tokens=1024)
        return _parse_llm_json(raw_reply)

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_and_normalize(
        extracted: dict[str, Any], current_state: dict[str, str]
    ) -> tuple[dict[str, str], str | None]:
        """Validate newly extracted entities and merge them with current_state.

        Returns (updated_state, validation_error_message).
        If a field is invalid, validation_error_message is set and the invalid field
        is omitted from updated_state.
        """
        updated = dict(current_state)

        # 1. Name validation
        if extracted.get("name"):
            name = str(extracted["name"]).strip()
            if len(name) < 2:
                return updated, "Please provide a valid full name (at least 2 characters)."
            updated["name"] = name

        # 2. Email validation
        if extracted.get("email"):
            email_raw = str(extracted["email"]).strip().lower()
            try:
                _EMAIL_ADAPTER.validate_python(email_raw)
                updated["email"] = email_raw
            except Exception:
                return updated, "Please provide a valid email address (e.g. name@example.com)."

        # 3. Date validation
        if extracted.get("date"):
            date_raw = str(extracted["date"]).strip()
            try:
                d = date.fromisoformat(date_raw)
                today = datetime.now(timezone.utc).date()
                if d < today:
                    return updated, "The booking date cannot be in the past. Please choose a future date."
                updated["booking_date"] = d.isoformat()
            except ValueError:
                return updated, "Please provide a valid date (YYYY-MM-DD)."

        # 4. Time validation
        if extracted.get("time"):
            time_raw = str(extracted["time"]).strip()
            try:
                t = time.fromisoformat(time_raw)
                updated["booking_time"] = t.strftime("%H:%M")
            except ValueError:
                return updated, "Please provide a valid time (e.g. 14:00 or 2:00 PM)."

        return updated, None

    @staticmethod
    def _format_missing_fields_prompt(state: dict[str, str]) -> str:
        """Formulate a prompt asking for the remaining missing fields."""
        missing: list[str] = []
        if "name" not in state:
            missing.append("your full name")
        if "email" not in state:
            missing.append("your email address")
        if "booking_date" not in state and "booking_time" not in state:
            missing.append("preferred date and time")
        else:
            if "booking_date" not in state:
                missing.append("preferred date")
            if "booking_time" not in state:
                missing.append("preferred time")

        if not missing:
            return ""

        if len(missing) == 1:
            return f"Please provide {missing[0]} to proceed with your booking."
        return f"To schedule your interview, please provide {', '.join(missing[:-1])} and {missing[-1]}."

    # -----------------------------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------------------------

    async def handle_message(
        self,
        session_id: str,
        message: str,
        booking_repo: BookingRepository,
    ) -> BookingProcessResult:
        """Process *message* within the interview booking lifecycle."""
        current_state = self.get_state(session_id)
        has_active_state = bool(current_state)

        extraction_result = await self._extract_intent_and_entities(
            session_id, message, current_state
        )
        intent = extraction_result.get("intent", "other")
        extracted = extraction_result.get("extracted", {}) or {}

        # 1. Cancel request
        if intent == "cancel":
            self.clear_state(session_id)
            cancel_msg = "Your interview booking has been cancelled."
            self._memory.append_messages(
                session_id,
                [
                    Message(role="user", content=message),
                    Message(role="assistant", content=cancel_msg),
                ],
            )
            return BookingProcessResult(answer=cancel_msg, booking_id=None)

        # 2. Inquiry or Other with NO active booking state -> Delegate to pure RAG
        if intent in {"inquiry", "other"} and not has_active_state and not any(extracted.values()):
            return BookingProcessResult(answer="", delegate_to_rag=True)

        # 3. Inquiry while booking IS active -> Interruption
        # If user asked a question without providing any booking details
        if intent == "inquiry" and not any(extracted.values()):
            missing_prompt = self._format_missing_fields_prompt(current_state)
            return BookingProcessResult(
                answer="",
                delegate_to_rag=True,
                is_interruption=True,
                missing_fields_reminder=missing_prompt,
            )

        # 4. Booking flow (intent == "book" or user provided booking details)
        updated_state, val_error = self._validate_and_normalize(extracted, current_state)

        # Save updated state
        if updated_state:
            self.save_state(session_id, updated_state)

        # Handle validation error if any
        if val_error:
            self._memory.append_messages(
                session_id,
                [
                    Message(role="user", content=message),
                    Message(role="assistant", content=val_error),
                ],
            )
            return BookingProcessResult(answer=val_error, booking_id=None)

        # Check completeness: name, email, booking_date, booking_time
        required_fields = {"name", "email", "booking_date", "booking_time"}
        missing_keys = required_fields - updated_state.keys()

        if missing_keys:
            ask_msg = self._format_missing_fields_prompt(updated_state)
            self._memory.append_messages(
                session_id,
                [
                    Message(role="user", content=message),
                    Message(role="assistant", content=ask_msg),
                ],
            )
            return BookingProcessResult(answer=ask_msg, booking_id=None)

        # All 4 fields present -> Check for double booking
        booking_date = date.fromisoformat(updated_state["booking_date"])
        booking_time = time.fromisoformat(updated_state["booking_time"])

        existing = booking_repo.get_by_date_and_time(booking_date, booking_time)
        if existing:
            # Slot taken: remove booking_time from state so user can pick another slot
            updated_state.pop("booking_time", None)
            self.save_state(session_id, updated_state)
            conflict_msg = (
                f"The slot on {booking_date} at {booking_time.strftime('%H:%M')} "
                "is already booked. Please choose a different time or date."
            )
            self._memory.append_messages(
                session_id,
                [
                    Message(role="user", content=message),
                    Message(role="assistant", content=conflict_msg),
                ],
            )
            return BookingProcessResult(answer=conflict_msg, booking_id=None)

        # Create booking record
        new_booking = Booking(
            session_id=session_id,
            name=updated_state["name"],
            email=updated_state["email"],
            booking_date=booking_date,
            booking_time=booking_time,
        )
        saved = booking_repo.save(new_booking)
        self.clear_state(session_id)

        # Short confirmation message with name, date, time, booking_id
        confirm_msg = (
            f"Your interview is confirmed for {saved.name} on {saved.booking_date} "
            f"at {saved.booking_time.strftime('%H:%M')}. Booking ID: {saved.id}."
        )

        self._memory.append_messages(
            session_id,
            [
                Message(role="user", content=message),
                Message(role="assistant", content=confirm_msg),
            ],
        )

        return BookingProcessResult(
            answer=confirm_msg,
            booking_id=saved.id,
        )


def get_booking_service() -> BookingService:
    """FastAPI dependency provider for :class:`BookingService`."""
    return BookingService()
