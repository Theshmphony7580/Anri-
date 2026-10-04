from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

from schemas import DocumentChunk


class RAGState(TypedDict, total=False):
    """Execution state passed between LangGraph nodes."""
    question: str
    chat_history: List[BaseMessage]
    retrieved_chunks: List[DocumentChunk]
    confidence_score: float
    rewritten_question: Optional[str]
    rerank_scores: Optional[List[float]]
    intent: Optional[str]
    answer: str
    retry_count: int
    stream_answer: bool

