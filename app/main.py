import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.bookings import router as bookings_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.database import init_db

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise DB tables and Qdrant collection."""
    logger.info("Starting up Palm Mind RAG Backend …")
    init_db()
    # Ensure the Qdrant collection exists (non-fatal if Qdrant is unreachable
    # at startup – individual requests will surface the error).
    try:
        from app.services.vector_store import get_vector_store

        get_vector_store().ensure_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure Qdrant collection at startup: %s", exc)
    yield
    logger.info("Palm Mind RAG Backend shut down.")


app = FastAPI(title="Palm Mind RAG Backend", version="0.1.0", lifespan=lifespan)

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(bookings_router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}
