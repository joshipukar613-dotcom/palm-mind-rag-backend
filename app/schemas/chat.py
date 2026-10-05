from datetime import date, time

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str] = []
    booking_id: str | None = None


class BookingDetails(BaseModel):
    name: str
    email: EmailStr
    booking_date: date
    booking_time: time
