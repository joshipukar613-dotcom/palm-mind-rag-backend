"""Chat API router.

Endpoints
---------
POST /api/v1/chat
    Send a message and receive a RAG-generated answer or booking interaction.

DELETE /api/v1/chat/{session_id}
    Clear conversation memory and active booking state for a given session.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.repositories.booking_repository import (
    BookingRepository,
    get_booking_repository,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.booking import BookingService, get_booking_service
from app.services.rag import RagService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message and receive a RAG answer or booking response",
)
async def chat(
    request: ChatRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
    booking_service: Annotated[BookingService, Depends(get_booking_service)],
    booking_repo: Annotated[BookingRepository, Depends(get_booking_repository)],
) -> ChatResponse:
    """Run the conversational pipeline (booking detection + RAG) for *request*.

    Parameters
    ----------
    request:
        :class:`~app.schemas.chat.ChatRequest` with ``session_id`` and ``message``.
    rag:
        Injected :class:`~app.services.rag.RagService`.
    booking_service:
        Injected :class:`~app.services.booking.BookingService`.
    booking_repo:
        Injected :class:`~app.repositories.booking_repository.BookingRepository`.

    Returns
    -------
    ChatResponse
        ``session_id``, ``answer``, ``sources``, and optional ``booking_id``.
    """
    logger.info(
        "Chat request: session_id='%s', message_len=%d",
        request.session_id,
        len(request.message),
    )

    # 1. Evaluate booking intent and state
    booking_res = await booking_service.handle_message(
        session_id=request.session_id,
        message=request.message,
        booking_repo=booking_repo,
    )

    # 2. If delegated to RAG
    if booking_res.delegate_to_rag:
        rag_result = await rag.answer(
            session_id=request.session_id,
            message=request.message,
        )

        answer = rag_result.answer
        # Interrupted by document question in middle of active booking
        if booking_res.is_interruption and booking_res.missing_fields_reminder:
            answer = f"{rag_result.answer}\n\n{booking_res.missing_fields_reminder}"

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=rag_result.sources,
            booking_id=None,
        )

    # 3. Handled directly by booking flow
    return ChatResponse(
        session_id=request.session_id,
        answer=booking_res.answer,
        sources=[],
        booking_id=booking_res.booking_id,
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Clear conversation history and booking state for a session",
)
def clear_session(
    session_id: str,
    rag: Annotated[RagService, Depends(get_rag_service)],
    booking_service: Annotated[BookingService, Depends(get_booking_service)],
) -> Response:
    """Delete all stored chat history and partial booking state for *session_id*."""
    mem_deleted = rag._memory.clear(session_id)
    state_deleted = booking_service.clear_state(session_id)

    if not mem_deleted and not state_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    logger.info("Cleared session '%s'.", session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
