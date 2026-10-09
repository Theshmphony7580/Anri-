"""Authenticated HTTP service for ANRI's local embedding and reranking models.

Run this on a machine with the model weights available, then route ANRI to it through
an HTTPS tunnel. Keep MODEL_SERVICE_URL unset on this machine to avoid proxy loops.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import asyncio
import hmac
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, StringConstraints

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent.reranker import get_reranker
from config import get_settings
from ingestion.embeddings import get_embedder
from schemas import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the configured local models once when the inference service starts."""
    if settings.MODEL_SERVICE_URL:
        raise RuntimeError(
            "MODEL_SERVICE_URL must be unset on the machine hosting model_service; "
            "otherwise the service would call itself."
        )

    embedder = get_embedder()
    if hasattr(embedder, "warmup"):
        await asyncio.to_thread(embedder.warmup)

    if settings.USE_RERANKER:
        reranker = get_reranker()
        if hasattr(reranker, "warmup"):
            await asyncio.to_thread(reranker.warmup)
    yield


app = FastAPI(
    title="ANRI Local Model Service",
    version="1.0.0",
    lifespan=lifespan,
)

TextInput = Annotated[str, StringConstraints(max_length=50_000)]


class EmbeddingRequest(BaseModel):
    """Validated text batch for embedding inference."""

    texts: List[TextInput] = Field(min_length=1, max_length=32)
    is_query: bool = False


class RerankRequest(BaseModel):
    """Validated query and candidate batch for reranking inference."""

    query: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    documents: List[TextInput] = Field(min_length=1, max_length=64)
    top_n: int = Field(default=4, ge=1, le=64)


def require_service_key(authorization: Optional[str] = Header(default=None)) -> None:
    """Require a machine-to-machine bearer token for inference routes."""
    expected_key = settings.MODEL_SERVICE_API_KEY
    if not expected_key:
        raise HTTPException(status_code=503, detail="Model service authentication is not configured.")

    expected_header = f"Bearer {expected_key}"
    if not authorization or not hmac.compare_digest(
        authorization.encode("utf-8"), expected_header.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing model service token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
def health():
    """Return a minimal liveness response without exposing model or config details."""
    return {"status": "ok"}


@app.post("/v1/embeddings", dependencies=[Depends(require_service_key)])
async def create_embeddings(payload: EmbeddingRequest):
    """Generate vectors for a bounded text batch."""
    try:
        vectors = await asyncio.to_thread(
            get_embedder().embed_batch,
            payload.texts,
            is_query=payload.is_query,
        )
    except Exception as exc:
        logger.exception("Local embedding request failed")
        raise HTTPException(status_code=502, detail="Local embedding inference failed.") from exc

    if len(vectors) != len(payload.texts):
        raise HTTPException(status_code=502, detail="Embedding model returned an invalid vector count.")
    return {"embeddings": vectors}


@app.post("/v1/rerank", dependencies=[Depends(require_service_key)])
async def rerank_documents(payload: RerankRequest):
    """Return ranked candidate indexes and scores for a bounded request."""
    candidates = [DocumentChunk(id=str(index), text=text) for index, text in enumerate(payload.documents)]
    try:
        ranked = await asyncio.to_thread(
            get_reranker().rerank,
            query=payload.query,
            chunks=candidates,
            top_n=min(payload.top_n, len(candidates)),
        )
    except Exception as exc:
        logger.exception("Local reranking request failed")
        raise HTTPException(status_code=502, detail="Local reranking inference failed.") from exc

    return {
        "results": [
            {"index": int(chunk.id), "score": float((chunk.metadata or {}).get("rerank_score", 0.0))}
            for chunk in ranked
        ]
    }
