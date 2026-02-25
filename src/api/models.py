"""API request and response models."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    n_results: int = Field(default=8, ge=1, le=20, description="Number of vector results to retrieve")
    stream: bool = Field(default=False, description="Enable streaming response")


class SourceReference(BaseModel):
    source: str
    page: Optional[str] = None
    slide: Optional[int] = None
    score: Optional[float] = None
    source_type: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = []
    graph_used: bool = False
    context_chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    vector_docs: int
    graph_nodes: int
    ollama_model: str


class IngestRequest(BaseModel):
    data_dir: str = Field(default="./data", description="Path to data directory")
    reset: bool = Field(default=False, description="Reset ChromaDB before ingesting")
