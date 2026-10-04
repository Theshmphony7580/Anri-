"""LangGraph Node implementations for the self-correcting Document RAG pipeline.

Nodes:
- retrieve_node: Fetches top-k document chunks from Qdrant vector store.
- grade_node: Evaluates chunk relevance and computes a confidence score (0.0 - 1.0).
- rewrite_node: Re-engineers failing queries into high-density search terms and tracks retries.
- generate_node: Synthesizes grounded answers with citations or triggers graceful refusal.
"""

import json
import logging
import re
from typing import Any, Dict, Optional
from langgraph.config import get_stream_writer

from schemas import DocumentChunk
from config import get_settings
from storage.vector_store import QdrantVectorStore, get_vector_store
from ingestion.embeddings import HuggingFaceEmbedder, GeminiEmbedder, get_embedder
from agent.state import RAGState
from agent.prompts import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_TEMPLATE,
    DIRECT_GENERATE_SYSTEM_PROMPT,
    DIRECT_GENERATE_USER_TEMPLATE,
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_TEMPLATE,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    FALLBACK_REFUSAL_TEMPLATE,
    format_chunk_context,
)
from agent.reranker import get_reranker
from agent.llm import LLMClient

logger = logging.getLogger(__name__)

def triage_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Classify user query intent into 'direct' or 'retrieval' to bypass retrieval for casual questions."""
    question = state.get("question", "")
    if not question:
        return {"intent": "retrieval"}

    client = llm or LLMClient()
    prompt = TRIAGE_USER_TEMPLATE.format(question=question)

    try:
        raw_response = client.generate(
            prompt=prompt,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            json_mode=True,
        )

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else {}

        intent = str(data.get("intent", "retrieval")).strip().lower()
        if intent not in ("direct", "retrieval"):
            intent = "retrieval"

        reason = data.get("reason", "N/A")
        logger.info(f"[triage_node] Query intent classified as '{intent}'. (Reason: {reason})")
        return {"intent": intent}
    except Exception as e:
        logger.error(f"[triage_node] Intent classification failed ({e}). Defaulting to 'retrieval'.")
        return {"intent": "retrieval"}


def retrieve_node(
    state: RAGState,
    vector_store: Optional[QdrantVectorStore] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrieve candidate document chunks from the vector store using dense vector similarity."""
    settings = get_settings()
    active_query = state.get("rewritten_question") or state.get("question", "")

    if not active_query:
        logger.warning("[retrieve_node] No active query found in state.")
        return {"retrieved_chunks": []}

    vs = vector_store or get_vector_store()
    emb = embedder or get_embedder()

    try:
        query_vector = emb.embed_text(active_query, is_query=True)
        # If reranker is enabled, retrieve broad candidate pool; otherwise direct top_k
        fetch_limit = (
            getattr(settings, "RERANK_CANDIDATES_K", 15)
            if getattr(settings, "USE_RERANKER", True)
            else settings.RETRIEVAL_TOP_K
        )
        chunks = vs.search(query_vector=query_vector, top_k=fetch_limit)
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"[retrieve_node] Retrieval failed: {e}")
        return {"retrieved_chunks": []}


def rerank_node(
    state: RAGState,
    reranker: Optional[Any] = None,
) -> Dict[str, Any]:
    """Rescore candidate chunks with BAAI/bge-reranker-base and prune to RETRIEVAL_TOP_K."""
    settings = get_settings()
    if not getattr(settings, "USE_RERANKER", True):
        return {}

    active_query = state.get("rewritten_question") or state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    if not chunks or not active_query:
        return {"retrieved_chunks": chunks, "rerank_scores": []}

    engine = reranker or get_reranker()
    try:
        reranked = engine.rerank(
            query=active_query,
            chunks=chunks,
            top_n=settings.RETRIEVAL_TOP_K,
        )
        scores = [(c.metadata or {}).get("rerank_score", 0.0) for c in reranked]
        logger.info(f"[rerank_node] Reranked {len(chunks)} candidate chunks down to {len(reranked)}.")
        return {"retrieved_chunks": reranked, "rerank_scores": scores}
    except Exception as e:
        logger.error(f"[rerank_node] Reranking failed ({e}). Retaining original chunks.")
        return {"retrieved_chunks": chunks[: settings.RETRIEVAL_TOP_K], "rerank_scores": []}


def grade_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Grade relevance of retrieved chunks against the active question and assign confidence score."""
    question = state.get("rewritten_question") or state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    client = llm or LLMClient()

    if not chunks:
        logger.info("[grade_node] Zero chunks retrieved; assigning baseline zero confidence.")
        return {"confidence_score": 0.0}

    context_str = format_chunk_context(chunks)
    prompt = GRADE_USER_TEMPLATE.format(question=question, context=context_str)

    raw_response = client.generate(
        prompt=prompt,
        system_prompt=GRADE_SYSTEM_PROMPT,
        json_mode=True,
    )

    # Robust JSON parsing with fallback regex extraction
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    confidence = float(data.get("confidence", 0.0))
    # Constrain confidence between 0.0 and 1.0
    confidence = max(0.0, min(1.0, confidence))

    logger.info(f"[grade_node] Graded confidence: {confidence:.2f} (Reason: {data.get('reason', 'N/A')})")
    return {"confidence_score": confidence}


def rewrite_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Reformulate the search query into high-density semantic search keywords and increment retry count."""
    orig_question = state.get("question", "")
    current_retries = state.get("retry_count", 0) + 1
    client = llm or LLMClient()

    prompt = REWRITE_USER_TEMPLATE.format(
        question=orig_question,
        attempt_number=current_retries,
    )

    rewritten_raw = client.generate(
        prompt=prompt,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        json_mode=False,
    )

    # Sanitize reformulated query
    rewritten_clean = rewritten_raw.strip('"` \n\r\t')
    logger.info(f"[rewrite_node] Retry #{current_retries}: '{orig_question}' -> '{rewritten_clean}'")

    return {
        "rewritten_question": rewritten_clean,
        "retry_count": current_retries,
    }


def _generate_answer(
    client: LLMClient,
    prompt: str,
    system_prompt: str,
    stream_answer: bool,
) -> str:
    if not stream_answer:
        return client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

    writer = get_stream_writer()
    tokens = []
    for token in client.stream(
        prompt=prompt,
        system_prompt=system_prompt,
        json_mode=False,
    ):
        tokens.append(token)
        writer({"type": "answer_token", "token": token})
    return "".join(tokens)


def generate_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate a grounded response with page/source citations or output a standard refusal."""
    settings = get_settings()
    confidence = state.get("confidence_score", 0.0)
    retries = state.get("retry_count", 0)
    threshold = settings.CONFIDENCE_THRESHOLD

    # If confidence is below threshold and we have exhausted retries, refuse gracefully
    if confidence < threshold and retries >= 2:
        logger.info(f"[generate_node] Retrieval failed after {retries} retries ({confidence:.2f} < {threshold:.2f}). Triggering refusal.")
        refusal_msg = FALLBACK_REFUSAL_TEMPLATE.format(
            confidence_score=confidence,
            threshold=threshold,
        )
        return {"answer": refusal_msg}

    # Otherwise synthesize grounded answer
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    context_str = format_chunk_context(chunks)
    client = llm or LLMClient()

    prompt = GENERATE_USER_TEMPLATE.format(question=question, context=context_str)
    answer = _generate_answer(
        client=client,
        prompt=prompt,
        system_prompt=GENERATE_SYSTEM_PROMPT,
        stream_answer=state.get("stream_answer", False),
    )

    return {"answer": answer}


def direct_generate_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Synthesize immediate conversational response for direct queries without vector retrieval."""
    question = state.get("question", "")
    client = llm or LLMClient()

    prompt = DIRECT_GENERATE_USER_TEMPLATE.format(question=question)
    answer = _generate_answer(
        client=client,
        prompt=prompt,
        system_prompt=DIRECT_GENERATE_SYSTEM_PROMPT,
        stream_answer=state.get("stream_answer", False),
    )

    logger.info("[direct_generate_node] Generated direct response (skipping vector retrieval).")
    return {
        "answer": answer,
        "confidence_score": 1.0,
        "retrieved_chunks": [],
    }
