from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a single granular piece of text indexed in the vector store."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class QueryRequest(BaseModel):
    """User request payload for the RAG agent."""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Grounded output returned to the user with citations and confidence."""
    question: str
    answer: str
    sources: List[DocumentChunk] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
