"""Bookings API router.

Endpoints
---------
GET /api/v1/bookings
    List all stored interview bookings.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.repositories.booking_repository import (
    BookingRepository,
    get_booking_repository,
)
from app.schemas.chat import BookingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bookings", tags=["bookings"])


@router.get(
    "",
    response_model=list[BookingResponse],
    summary="List all interview bookings",
)
def list_bookings(
    repo: Annotated[BookingRepository, Depends(get_booking_repository)],
) -> list[BookingResponse]:
    """Retrieve all stored interview bookings, newest first."""
    bookings = repo.list_all()
    logger.info("Retrieved %d bookings.", len(bookings))
    return [BookingResponse.model_validate(b) for b in bookings]
