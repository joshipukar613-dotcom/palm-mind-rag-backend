"""Repository for :class:`~app.models.booking.Booking` records.

All database operations for interview bookings are centralised here.
"""

from __future__ import annotations

import logging
from datetime import date, time
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.booking import Booking

logger = logging.getLogger(__name__)


class BookingRepository:
    """CRUD operations for :class:`~app.models.booking.Booking`.

    Parameters
    ----------
    db:
        An active SQLAlchemy :class:`~sqlalchemy.orm.Session`.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, booking: Booking) -> Booking:
        """Persist *booking* to the database and return it with auto-set fields.

        Parameters
        ----------
        booking:
            A :class:`~app.models.booking.Booking` instance.

        Returns
        -------
        Booking
            The persisted instance after commit and refresh.
        """
        self._db.add(booking)
        self._db.commit()
        self._db.refresh(booking)
        logger.info(
            "Saved booking id='%s' session_id='%s' name='%s' date=%s time=%s",
            booking.id,
            booking.session_id,
            booking.name,
            booking.booking_date,
            booking.booking_time,
        )
        return booking

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, booking_id: str) -> Booking | None:
        """Return the booking with *booking_id*, or ``None``."""
        return self._db.get(Booking, booking_id)

    def get_by_date_and_time(
        self, booking_date: date, booking_time: time
    ) -> Booking | None:
        """Return existing booking matching the same date and time slot, or None.

        Used to prevent double booking of identical interview slots.
        """
        return (
            self._db.query(Booking)
            .filter(
                Booking.booking_date == booking_date,
                Booking.booking_time == booking_time,
            )
            .first()
        )

    def list_all(self) -> list[Booking]:
        """Return all bookings ordered by creation time descending."""
        return (
            self._db.query(Booking)
            .order_by(Booking.created_at.desc())
            .all()
        )


def get_booking_repository(
    db: Annotated[Session, Depends(get_db)],
) -> BookingRepository:
    """FastAPI dependency provider for :class:`BookingRepository`."""
    return BookingRepository(db)
