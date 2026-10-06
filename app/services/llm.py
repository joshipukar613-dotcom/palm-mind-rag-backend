"""Async LLM client for OpenAI-compatible chat completion endpoints.

Uses :mod:`httpx` directly so there is no LangChain or OpenAI SDK dependency.
Settings (``llm_base_url``, ``llm_api_key``, ``llm_model``) come from
:func:`~app.core.config.get_settings`.

Typical usage::

    client = LLMClient()
    reply = await client.complete(messages=[{"role": "user", "content": "Hi"}])
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class LLMClient:
    """Thin async wrapper around an OpenAI-compatible ``/chat/completions`` API.

    Parameters
    ----------
    base_url:
        Base URL of the API (e.g. ``"https://api.groq.com/openai/v1"``).
        Defaults to ``settings.llm_base_url``.
    api_key:
        Bearer token.  Defaults to ``settings.llm_api_key``.
    model:
        Model identifier.  Defaults to ``settings.llm_model``.
    temperature:
        Sampling temperature forwarded to the API.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        settings = get_settings()
        self._base_url: str = base_url or settings.llm_base_url
        self._api_key: str = api_key or settings.llm_api_key
        self._model: str = model or settings.llm_model
        self._temperature: float = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        """Call the chat completions endpoint and return the assistant reply.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts in
            OpenAI message format.
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        str
            The text content of the first assistant choice.

        Raises
        ------
        fastapi.HTTPException
            * **503** – Network error or timeout reaching the LLM API.
            * **502** – The LLM API returned a non-200 status code.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "LLM request: model=%s, messages=%d, max_tokens=%d",
            self._model,
            len(messages),
            max_tokens,
        )

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
                response = await http.post(
                    f"{self._base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            logger.error("LLM request timed out: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service timed out.  Please retry later.",
            ) from exc
        except httpx.RequestError as exc:
            logger.error("LLM request failed (network): %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not reach LLM service: {exc}",
            ) from exc

        if response.status_code != 200:
            logger.error(
                "LLM returned %d: %s", response.status_code, response.text[:300]
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"LLM API returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                ),
            )

        data = response.json()
        try:
            reply: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected LLM response shape: %s", data)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM returned an unrecognisable response shape.",
            ) from exc

        logger.debug("LLM replied with %d chars.", len(reply))
        return reply.strip()
