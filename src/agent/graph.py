"""StateGraph compilation and routing logic for the self-correcting RAG agent.

Assembles the 4-node reasoning loop:
[retrieve] -> [grade] -> (confidence >= 0.70 or retries >= 2) -> [generate] -> END
                      -> (confidence < 0.70 and retries < 2)   -> [rewrite]  -> [retrieve]
"""

import logging
from typing import Literal, Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.state import RAGState
from agent.nodes import (
    triage_node,
    direct_generate_node,
    retrieve_node,
    rerank_node,
    grade_node,
    rewrite_node,
    generate_node,
)
from config import get_settings
from schemas import QueryResponse

logger = logging.getLogger(__name__)


def decide_intent_route(state: RAGState) -> Literal["direct_generate", "retrieve"]:
    """Entry router: decides whether to answer directly or route to retrieval."""
    intent = state.get("intent", "retrieval")
    if intent == "direct":
        logger.info("[decide_intent_route] Intent is 'direct' -> Routing to 'direct_generate'.")
        return "direct_generate"
    logger.info("[decide_intent_route] Intent is 'retrieval' -> Routing to 'retrieve'.")
    return "retrieve"


def decide_next_step(state: RAGState) -> Literal["generate", "rewrite"]:
    """Conditional edge router: determines whether to synthesize answer or rewrite query."""
    settings = get_settings()
    confidence = state.get("confidence_score", 0.0)
    retries = state.get("retry_count", 0)
    threshold = settings.CONFIDENCE_THRESHOLD

    # High confidence: proceed directly to answer generation
    if confidence >= threshold:
        logger.info(f"[decide_next_step] Confidence {confidence:.2f} >= {threshold:.2f} -> Routing to 'generate'.")
        return "generate"

    # Low confidence but retries remaining: attempt self-correction query rewrite
    if retries < 2:
        logger.info(f"[decide_next_step] Confidence {confidence:.2f} < {threshold:.2f} (Attempt #{retries}) -> Routing to 'rewrite'.")
        return "rewrite"

    # Max retries exhausted: route to generate (which invokes graceful refusal)
    logger.info(f"[decide_next_step] Max retries reached ({retries}) -> Routing to 'generate' for fallback refusal.")
    return "generate"


def build_rag_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph self-correcting Document RAG workflow with triage & 2-stage retrieval."""
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("direct_generate", direct_generate_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    # Entry point & intent routing
    workflow.set_entry_point("triage")
    workflow.add_conditional_edges(
        "triage",
        decide_intent_route,
        {
            "direct_generate": "direct_generate",
            "retrieve": "retrieve",
        },
    )

    # Direct conversation termination
    workflow.add_edge("direct_generate", END)

    # Two-stage retrieval edges: retrieve -> rerank -> grade
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade")

    # Conditional routing out of grading node
    workflow.add_conditional_edges(
        "grade",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # Self-correction loop: rewrite feeds back into retrieve -> rerank -> grade
    workflow.add_edge("rewrite", "retrieve")

    # Grounded answer termination
    workflow.add_edge("generate", END)

    return workflow.compile()


_CACHED_RAG_GRAPH: Optional[CompiledStateGraph] = None


def get_rag_graph() -> CompiledStateGraph:
    """Return a process-wide cached instance of the compiled LangGraph workflow."""
    global _CACHED_RAG_GRAPH
    if _CACHED_RAG_GRAPH is None:
        _CACHED_RAG_GRAPH = build_rag_graph()
    return _CACHED_RAG_GRAPH


def run_rag_query(question: str) -> QueryResponse:
    """Convenience helper to run an end-to-end query through the compiled graph and format as QueryResponse."""
    graph = get_rag_graph()
    initial_state: RAGState = {
        "question": question,
        "chat_history": [],
        "retrieved_chunks": [],
        "confidence_score": 0.0,
        "rewritten_question": None,
        "intent": None,
        "answer": "",
        "retry_count": 0,
    }

    final_state = graph.invoke(initial_state)

    return QueryResponse(
        question=final_state.get("question", question),
        answer=final_state.get("answer", ""),
        sources=final_state.get("retrieved_chunks", []),
        confidence=final_state.get("confidence_score", 0.0),
    )

