"""FastAPI telemetry server for the Precision Observability Console.

Provides high-density REST endpoints for:
- System hardware & model configuration
- Qdrant disk vector store telemetry
- Cryptographic document ledger & one-click ingestion
- Step-by-step LangGraph state machine execution tracing
"""

import os
import json

# Prevent OpenBLAS thread allocation crash on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ROOT_DIR, get_settings
from storage.vector_store import QdrantVectorStore, get_vector_store
from ingestion.embeddings import get_embedder
from ingestion.parser import compute_file_sha256
from ingestion.pipeline import IngestionPipeline
from agent.reranker import get_reranker
from agent.graph import get_rag_graph, decide_next_step
from agent.state import RAGState


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load and warm up vector store, embedding model, reranker, and LangGraph workflow at startup."""
    print("\n[Server Lifespan] Preloading and warming up local ML models and LangGraph state machine into RAM...")
    
    # 1. Thread-safe Qdrant vector store
    get_vector_store()

    # 2. Dense embedding model (BGE-base)
    embedder = get_embedder()
    if hasattr(embedder, "warmup"):
        embedder.warmup()

    # 3. Cross-encoder reranker (BGE-reranker)
    settings = get_settings()
    if getattr(settings, "USE_RERANKER", True):
        reranker = get_reranker()
        if hasattr(reranker, "warmup"):
            reranker.warmup()

    # 4. Compiled LangGraph StateMachine
    get_rag_graph()

    print("[Server Lifespan] All models and LangGraph state machine preloaded into RAM. Zero query-time loading latency!\n")
    yield


app = FastAPI(
    title="ANRI // Precision Observability Console",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Eagerly initialize single shared instance at module load
_SHARED_VECTOR_STORE = get_vector_store()


class QueryPayload(BaseModel):
    question: str


class IngestPayload(BaseModel):
    filename: str
    force_reindex: bool = False


@app.get("/api/stats")
def get_system_stats() -> Dict[str, Any]:
    """Return real-time telemetry from Qdrant vector database and system configuration."""
    settings = get_settings()
    vector_store = get_vector_store()

    try:
        col_info = vector_store.client.get_collection(vector_store.collection_name)
        total_vectors = col_info.points_count or 0
    except Exception:
        total_vectors = 0

    # Count files in data directory
    data_files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]

    return {
        "total_documents": len(data_files),
        "total_vectors": total_vectors,
        "vector_dim": vector_store.vector_dim,
        "distance_metric": "Cosine",
        "collection_name": vector_store.collection_name,
        "qdrant_path": str(settings.QDRANT_PATH),
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.LOCAL_EMBEDDING_MODEL,
        "embedding_device": settings.EMBEDDING_DEVICE,
        "use_reranker": getattr(settings, "USE_RERANKER", True),
        "reranker_model": getattr(get_reranker(), "model_name", getattr(settings, "RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")),
        "reranker_device": getattr(get_reranker(), "device", getattr(settings, "RERANKER_DEVICE", "cpu")),
        "reranker_provider": getattr(get_reranker(), "provider", getattr(settings, "RERANKER_PROVIDER", "flashrank")),
        "rerank_candidates_k": getattr(settings, "RERANK_CANDIDATES_K", 15),
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.GEMINI_MODEL,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "status": "ONLINE",
    }


@app.get("/api/documents")
def list_documents() -> List[Dict[str, Any]]:
    """Scan data/ folder and check indexing status against Qdrant SHA-256 hashes."""
    vector_store = get_vector_store()
    documents = []

    for file_path in sorted(DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        try:
            doc_hash = compute_file_sha256(str(file_path))
            is_indexed = vector_store.has_doc_hash(doc_hash)
            size_kb = round(file_path.stat().st_size / 1024, 1)

            documents.append({
                "filename": file_path.name,
                "size_kb": size_kb,
                "doc_hash": doc_hash,
                "doc_hash_short": f"{doc_hash[:8]}...{doc_hash[-8:]}",
                "is_indexed": is_indexed,
                "path": str(file_path),
            })
        except Exception as e:
            documents.append({
                "filename": file_path.name,
                "size_kb": 0,
                "doc_hash": "error",
                "doc_hash_short": "error",
                "is_indexed": False,
                "error": str(e),
            })

    return documents


@app.post("/api/ingest")
def ingest_document(payload: IngestPayload) -> Dict[str, Any]:
    """Ingest a specific file located in data/ directory."""
    safe_filename = Path(payload.filename).name
    target_file = DATA_DIR / safe_filename
    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_filename}' not found in data/ directory.")

    try:
        pipeline = IngestionPipeline(vector_store=_SHARED_VECTOR_STORE)
        result = pipeline.ingest_file(str(target_file), force_reindex=payload.force_reindex)

        return {
            "status": result.status,
            "source_file": result.source_file,
            "doc_hash": result.doc_hash,
            "chunk_count": result.chunk_count,
            "message": result.message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), auto_ingest: bool = True) -> Dict[str, Any]:
    """Upload a document to data/ directory with optional immediate ingestion."""
    safe_filename = Path(file.filename).name if file.filename else "uploaded_file"
    destination = DATA_DIR / safe_filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_hash = compute_file_sha256(str(destination))


    if auto_ingest:
        try:
            pipeline = IngestionPipeline(vector_store=_SHARED_VECTOR_STORE)
            result = pipeline.ingest_file(str(destination), force_reindex=False)
            return {
                "uploaded": True,
                "filename": file.filename,
                "doc_hash": doc_hash,
                "ingest_status": result.status,
                "chunk_count": result.chunk_count,
                "message": result.message,
            }
        except Exception as e:
            return {
                "uploaded": True,
                "filename": file.filename,
                "doc_hash": doc_hash,
                "ingest_status": "error",
                "message": f"Uploaded, but indexing failed: {str(e)}",
            }

    return {
        "uploaded": True,
        "filename": file.filename,
        "doc_hash": doc_hash,
        "ingest_status": "unindexed",
        "message": "Uploaded to data/ directory successfully.",
    }


def _iter_query_events(question: str, stream_answer: bool):
    """
    Execute the graph and emit answer token events followed by the completed query trace.
    """
    settings = get_settings()
    start_time = time.time()

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
        "stream_answer": stream_answer,
    }

    state = dict(initial_state)
    steps_trace: List[Dict[str, Any]] = []
    step_start = time.time()

    # Stream through real LangGraph state machine execution transitions
    for mode, step_output in graph.stream(
        initial_state,
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            if stream_answer and step_output.get("type") == "answer_token":
                yield {"event": "token", "token": step_output.get("token", "")}
            continue

        for node_name, node_update in step_output.items():
            duration_ms = round((time.time() - step_start) * 1000, 1)
            step_start = time.time()
            state.update(node_update)

            if node_name == "triage":
                intent = state.get("intent", "retrieval")
                steps_trace.append({
                    "node": "triage",
                    "attempt": 0,
                    "status": "pass",
                    "intent": intent,
                    "duration_ms": duration_ms,
                    "details": f"Classified query intent as '{intent}'",
                })
            elif node_name == "direct_generate":
                steps_trace.append({
                    "node": "direct_generate",
                    "status": "pass",
                    "duration_ms": duration_ms,
                    "details": "Synthesized direct conversational response without document retrieval",
                })
            elif node_name == "retrieve":
                chunks = state.get("retrieved_chunks", [])
                retry_c = state.get("retry_count", 0)
                prefix = "Re-retrieved" if retry_c > 0 else "Retrieved"
                steps_trace.append({
                    "node": "retrieve",
                    "attempt": retry_c,
                    "status": "pass" if chunks else "empty",
                    "duration_ms": duration_ms,
                    "details": f"{prefix} {len(chunks)} candidate chunks from Qdrant",
                })
            elif node_name == "rerank":
                chunks = state.get("retrieved_chunks", [])
                retry_c = state.get("retry_count", 0)
                top_score = (chunks[0].metadata or {}).get("rerank_score", None) if chunks else None
                steps_trace.append({
                    "node": "rerank",
                    "attempt": retry_c,
                    "status": "pass" if chunks else "empty",
                    "duration_ms": duration_ms,
                    "details": f"Rescored -> Top {len(chunks)} (Top score: {top_score})",
                })
            elif node_name == "grade":
                confidence = state.get("confidence_score", 0.0)
                decision = decide_next_step(state)
                steps_trace.append({
                    "node": "grade",
                    "attempt": state.get("retry_count", 0),
                    "status": "pass" if confidence >= settings.CONFIDENCE_THRESHOLD else "warn",
                    "score": round(confidence, 2),
                    "threshold": settings.CONFIDENCE_THRESHOLD,
                    "duration_ms": duration_ms,
                    "decision": decision,
                })
            elif node_name == "rewrite":
                steps_trace.append({
                    "node": "rewrite",
                    "attempt": state.get("retry_count", 0),
                    "status": "warn",
                    "rewritten_query": state.get("rewritten_question"),
                    "duration_ms": duration_ms,
                })
            elif node_name == "generate":
                confidence = state.get("confidence_score", 0.0)
                steps_trace.append({
                    "node": "generate",
                    "status": "pass" if confidence >= settings.CONFIDENCE_THRESHOLD else "refuse",
                    "duration_ms": duration_ms,
                })

    total_latency_ms = round((time.time() - start_time) * 1000, 1)

    # Format chunks for inspection drawer
    formatted_chunks = []
    for c in state.get("retrieved_chunks", []):
        meta = c.metadata or {}
        formatted_chunks.append({
            "id": c.id,
            "text": c.text,
            "score": round(c.score, 4) if c.score is not None else None,
            "rerank_score": round(meta["rerank_score"], 4) if meta.get("rerank_score") is not None else None,
            "source_file": meta.get("source_file", "unknown"),
            "page_numbers": meta.get("page_numbers", []),
            "section_path": meta.get("section_path", "General"),
            "content_type": meta.get("content_type", "text"),
        })

    is_direct = state.get("intent") == "direct"
    final_confidence = 1.0 if is_direct else round(state.get("confidence_score", 0.0), 2)
    is_passed = True if is_direct else (state.get("confidence_score", 0.0) >= settings.CONFIDENCE_THRESHOLD)

    response = {
        "question": question,
        "final_answer": state.get("answer", ""),
        "confidence_score": final_confidence,
        "threshold": settings.CONFIDENCE_THRESHOLD,
        "passed": is_passed,
        "retries_count": state.get("retry_count", 0),
        "total_latency_ms": total_latency_ms,
        "steps_trace": steps_trace,
        "retrieved_chunks": formatted_chunks,
    }

    yield {"event": "complete", "trace": response}


def _collect_query_trace(question: str) -> Dict[str, Any]:
    for event in _iter_query_events(question, stream_answer=False):
        if event["event"] == "complete":
            return event["trace"]
    raise RuntimeError("Query completed without a final trace.")


@app.post("/api/query")
def execute_query_trace(payload: QueryPayload) -> Dict[str, Any]:
    return _collect_query_trace(payload.question)


@app.post("/api/query/stream")
def execute_query_trace_stream(payload: QueryPayload) -> StreamingResponse:
    def event_stream():
        try:
            for event in _iter_query_events(payload.question, stream_answer=True):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            yield f"data: {json.dumps({'event': 'error', 'detail': 'Query execution failed.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mount static assets directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
