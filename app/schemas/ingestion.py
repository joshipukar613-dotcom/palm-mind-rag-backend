from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    chunk_strategy: ChunkStrategy
    chunk_count: int
    char_count: int
    created_at: datetime
