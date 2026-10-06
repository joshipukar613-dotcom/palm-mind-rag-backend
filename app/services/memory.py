"""Redis-backed conversational memory.

Each session's history is stored under the key ``chat:{session_id}`` as a
JSON-encoded list of ``{"role": "user"|"assistant", "content": "..."}`` dicts.

* History is capped at the last :attr:`ChatMemory.MAX_MESSAGES` messages.
* Every write resets the TTL to :attr:`~app.core.config.Settings.chat_ttl_seconds`.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

import redis as redis_lib
from fastapi import HTTPException, status

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_MESSAGES: int = 10  # total stored messages (user + assistant combined)


class Message(TypedDict):
    """A single chat turn stored in Redis."""

    role: str   # "user" or "assistant"
    content: str


def _key(session_id: str) -> str:
    """Return the Redis key for *session_id*."""
    return f"chat:{session_id}"


class ChatMemory:
    """Manages per-session chat history in Redis.

    Parameters
    ----------
    redis_url:
        Redis connection URL.  Defaults to ``settings.redis_url``.
    ttl_seconds:
        Inactivity TTL applied on every write.
        Defaults to ``settings.chat_ttl_seconds``.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        url = redis_url or settings.redis_url
        self._ttl: int = ttl_seconds if ttl_seconds is not None else settings.chat_ttl_seconds
        try:
            self._redis = redis_lib.from_url(url, decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not create Redis client: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not connect to chat memory store.",
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_history(self, session_id: str) -> list[Message]:
        """Return the stored message list for *session_id*.

        Parameters
        ----------
        session_id:
            Unique session identifier.

        Returns
        -------
        list[Message]
            The ordered list of messages, oldest first.  Empty list if the
            session does not exist yet.

        Raises
        ------
        fastapi.HTTPException
            **503** if Redis is unreachable.
        """
        try:
            raw = self._redis.get(_key(session_id))
        except redis_lib.RedisError as exc:
            logger.error("Redis error loading history for '%s': %s", session_id, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat memory store is unavailable.",
            ) from exc

        if raw is None:
            return []

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt history for session '%s'; resetting.", session_id
            )
            return []

    def append_messages(self, session_id: str, new_messages: list[Message]) -> None:
        """Append *new_messages* to the session history, trim to the last
        :data:`MAX_MESSAGES`, and reset the TTL.

        Parameters
        ----------
        session_id:
            Unique session identifier.
        new_messages:
            One or more messages to append.  Typically the user turn and the
            assistant turn for one exchange.

        Raises
        ------
        fastapi.HTTPException
            **503** if Redis is unavailable.
        """
        history = self.load_history(session_id)
        history.extend(new_messages)
        history = history[-MAX_MESSAGES:]   # keep only the most recent N

        try:
            self._redis.set(_key(session_id), json.dumps(history), ex=self._ttl)
        except redis_lib.RedisError as exc:
            logger.error(
                "Redis error saving history for '%s': %s", session_id, exc
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat memory store is unavailable.",
            ) from exc

        logger.debug(
            "Session '%s': stored %d messages (TTL=%ds).",
            session_id,
            len(history),
            self._ttl,
        )

    def clear(self, session_id: str) -> bool:
        """Delete all history for *session_id*.

        Parameters
        ----------
        session_id:
            Unique session identifier.

        Returns
        -------
        bool
            ``True`` if a key was deleted, ``False`` if the session did not
            exist.

        Raises
        ------
        fastapi.HTTPException
            **503** if Redis is unavailable.
        """
        try:
            deleted = self._redis.delete(_key(session_id))
        except redis_lib.RedisError as exc:
            logger.error(
                "Redis error clearing session '%s': %s", session_id, exc
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat memory store is unavailable.",
            ) from exc

        logger.info("Cleared session '%s' (deleted=%d).", session_id, deleted)
        return bool(deleted)
