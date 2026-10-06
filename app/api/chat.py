"""Chat API router.

Endpoints
---------
POST /api/v1/chat
    Send a message and receive a RAG-generated answer with source citations.

DELETE /api/v1/chat/{session_id}
    Clear the conversation memory for a given session.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import RagService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message and receive a RAG-generated answer",
)
async def chat(
    request: ChatRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    """Run the RAG pipeline for *request.message* in *request.session_id*.

    Parameters
    ----------
    request:
        :class:`~app.schemas.chat.ChatRequest` with ``session_id`` and ``message``.
    rag:
        Injected :class:`~app.services.rag.RagService`.

    Returns
    -------
    ChatResponse
        ``session_id``, ``answer``, ``sources``, and ``booking_id`` (always
        ``None`` for now — reserved for future booking integration).

    Raises
    ------
    fastapi.HTTPException
        Propagated from the RAG pipeline (e.g. 503 if Redis/LLM is down).
    """
    logger.info(
        "Chat request: session_id='%s', message_len=%d",
        request.session_id,
        len(request.message),
    )

    rag_result = await rag.answer(
        session_id=request.session_id,
        message=request.message,
    )

    return ChatResponse(
        session_id=request.session_id,
        answer=rag_result.answer,
        sources=rag_result.sources,
        booking_id=None,
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Clear conversation history for a session",
)
def clear_session(
    session_id: str,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> Response:
    """Delete all stored chat history for *session_id*.

    Parameters
    ----------
    session_id:
        The conversation session to clear.
    rag:
        Injected :class:`~app.services.rag.RagService`.

    Raises
    ------
    fastapi.HTTPException
        * **404** – Session not found in Redis.
        * **503** – Redis is unavailable.
    """
    deleted = rag._memory.clear(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    logger.info("Cleared session '%s'.", session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
