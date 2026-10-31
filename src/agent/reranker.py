"""Local reranker using FlashRank with a cross-encoder fallback.

Implements stage two of the retrieval pipeline:
1. Receives candidate chunks from vector search (e.g., Top-15 from Qdrant).
2. Deeply computes cross-attention relevance scores between the query and each chunk text.
3. Annotates each DocumentChunk metadata with `rerank_score`.
4. Sorts descending and prunes to Top-N (e.g., Top-4).
"""

import logging
import math
import re
import threading
from typing import List, Optional

import httpx

from schemas import DocumentChunk
from config import get_settings, validate_model_service_url

logger = logging.getLogger(__name__)

_RERANKER_LOCK = threading.RLock()
_GLOBAL_RERANKER: Optional["LocalReranker"] = None


class LocalReranker:
    """Thread-safe local reranker using the configured provider."""

    _instance: Optional["LocalReranker"] = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                return cls._instance
        return super().__new__(cls)

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        with self._lock:
            if getattr(self, "_initialized", False):
                return

            settings = get_settings()
            self.provider = getattr(settings, "RERANKER_PROVIDER", "flashrank").lower()
            self.model_name = model_name or getattr(settings, "RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

            # Provider-specific model normalization
            if self.provider == "flashrank":
                # FlashRank uses ONNX models (default: ms-marco-MiniLM-L-12-v2)
                if "bge" in self.model_name.lower() or not self.model_name:
                    self.model_name = "ms-marco-MiniLM-L-12-v2"
                self.device = "cpu"
            else:
                if "bge-reranker-small" in self.model_name.lower():
                    self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.device = device or getattr(settings, "RERANKER_DEVICE", "cpu")

            self._model = None
            self._flashrank_model = None
            self._initialized = True

    def _load_model(self):
        """Lazy-load the reranker model on first reranking request."""
        if self._model is not None or self._flashrank_model is not None:
            return self._model or self._flashrank_model

        # Provider: FlashRank (ultra-lightweight ONNX runtime, immune to Windows pagefile limit)
        if self.provider == "flashrank":
            try:
                from flashrank import Ranker
                model_name = self.model_name
                if "bge" in model_name.lower():
                    model_name = "ms-marco-MiniLM-L-12-v2"
                    self.model_name = model_name
                print(f"[LocalReranker] Loading FlashRank ONNX model '{model_name}'...")
                self._flashrank_model = Ranker(model_name=model_name)
                print(f"[LocalReranker] FlashRank model '{model_name}' successfully loaded into memory (ONNX).")
                return self._flashrank_model
            except Exception as e:
                print(f"[LocalReranker] FlashRank loading failed ({e}). Falling back to sentence-transformers...")

        from sentence_transformers import CrossEncoder

        target_device = self.device
        if target_device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[LocalReranker] CUDA requested but torch.cuda.is_available() is False. Falling back to 'cpu'.")
                    target_device = "cpu"
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[LocalReranker] GPU detected: {gpu_name} ({target_device})")
            except Exception as torch_err:
                print(f"[LocalReranker] PyTorch CUDA check failed ({torch_err}). Falling back to 'cpu'.")
                target_device = "cpu"
        self.device = target_device

        try:
            print(f"[LocalReranker] Loading cross-encoder model '{self.model_name}' onto {self.device}...")
            try:
                # Fast offline load without checking Hugging Face remote repository
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device)
            print(f"[LocalReranker] Cross-encoder '{self.model_name}' successfully loaded into memory on {self.device}.")
        except (MemoryError, OSError) as mem_err:
            print(f"[LocalReranker] Memory/OS Error ({mem_err}): Insufficient commit memory/RAM for '{self.model_name}'.")
            print("[LocalReranker] Auto-recovering: attempting FlashRank ONNX 'ms-marco-MiniLM-L-12-v2' (zero pagefile impact)...")
            try:
                from flashrank import Ranker
                self._flashrank_model = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
                self.provider = "flashrank"
                self.model_name = "ms-marco-MiniLM-L-12-v2"
                print(f"[LocalReranker] Successfully recovered with FlashRank '{self.model_name}'.")
                return self._flashrank_model
            except Exception as fr_err:
                print(f"[LocalReranker] FlashRank recovery failed ({fr_err}). Attempting lightweight 'cross-encoder/ms-marco-MiniLM-L-6-v2' (80 MB)...")
                try:
                    self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    try:
                        self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True)
                    except Exception:
                        self._model = CrossEncoder(self.model_name, device=self.device)
                    print(f"[LocalReranker] Cross-encoder '{self.model_name}' successfully loaded into memory.")
                    return self._model
                except Exception as fallback_err:
                    print(f"[LocalReranker] Fallback reranker error: {fallback_err}. Falling back to heuristic.")
                    self._model = False
        except ImportError:
            logger.warning("[LocalReranker] 'sentence-transformers' not available. Falling back to heuristic reranking.")
            self._model = False
        except Exception as e:
            import traceback
            print(f"[LocalReranker] ERROR loading model '{self.model_name}': {type(e).__name__} -> {repr(e)}")
            traceback.print_exc()
            logger.warning(f"[LocalReranker] Failed to load model '{self.model_name}' ({type(e).__name__}: {e}). Falling back to heuristic.")
            self._model = False

        return self._model

    def warmup(self):
        """Eagerly load model weights and execute dummy pair to warm up runtime."""
        model = self._load_model()
        if getattr(self, "_flashrank_model", None):
            try:
                from flashrank import RerankRequest
                self._flashrank_model.rerank(RerankRequest(query="warmup", passages=[{"id": 0, "text": "passage"}]))
            except Exception as e:
                logger.warning(f"[LocalReranker] FlashRank warmup error ({e})")
        elif model and not isinstance(model, bool):
            try:
                model.predict([["system warmup query", "system warmup passage"]])
            except Exception as e:
                logger.warning(f"[LocalReranker] Warmup error ({e})")

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_n: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Rescore candidate chunks against query and return top_n sorted descending."""
        if not chunks or not query:
            return chunks

        settings = get_settings()
        limit = top_n if top_n is not None else settings.RETRIEVAL_TOP_K

        # FlashRank inference path
        if getattr(self, "_flashrank_model", None) or self.provider == "flashrank":
            self._load_model()
            if getattr(self, "_flashrank_model", None):
                try:
                    from flashrank import RerankRequest
                    passages = [{"id": idx, "text": c.text} for idx, c in enumerate(chunks)]
                    req = RerankRequest(query=query, passages=passages)
                    ranked = self._flashrank_model.rerank(req)
                    id_to_chunk = {idx: c for idx, c in enumerate(chunks)}
                    sorted_chunks = []
                    for item in ranked:
                        c = id_to_chunk[item["id"]]
                        if c.metadata is None:
                            c.metadata = {}
                        c.metadata["rerank_score"] = round(float(item["score"]), 4)
                        sorted_chunks.append(c)
                    return sorted_chunks[:limit]
                except Exception as e:
                    logger.warning(f"[LocalReranker] FlashRank inference failed ({e}). Falling back.")

        model = self._load_model()
        if model and not isinstance(model, bool):
            try:
                pairs = [[query, c.text] for c in chunks]
                raw_scores = model.predict(pairs)

                # Normalize and attach score to metadata
                for idx, chunk in enumerate(chunks):
                    score = float(raw_scores[idx])
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata["rerank_score"] = round(score, 4)

                sorted_chunks = sorted(
                    chunks,
                    key=lambda c: (c.metadata or {}).get("rerank_score", -999.0),
                    reverse=True,
                )
                return sorted_chunks[:limit]
            except Exception as e:
                logger.warning(f"[LocalReranker] Inference failed ({e}). Falling back to heuristic.")

        # Heuristic fallback if model weights fail to load or offline test mode
        return self._heuristic_rerank(query, chunks, limit)

    def _heuristic_rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        limit: int,
    ) -> List[DocumentChunk]:
        """Deterministic keyword-density reranker for offline / unit test resilience."""
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not query_words:
            return chunks[:limit]

        scored_chunks = []
        for c in chunks:
            text_lower = c.text.lower()
            term_hits = sum(text_lower.count(qw) for qw in query_words)
            matched_unique = sum(1 for qw in query_words if qw in text_lower)
            # Normalization into pseudo-logit scale
            pseudo_score = (matched_unique * 1.5) + (min(term_hits, 10) * 0.2)

            if c.metadata is None:
                c.metadata = {}
            c.metadata["rerank_score"] = round(pseudo_score, 4)
            scored_chunks.append(c)

        scored_chunks.sort(
            key=lambda c: (c.metadata or {}).get("rerank_score", 0.0),
            reverse=True,
        )
        return scored_chunks[:limit]


class RemoteModelServiceReranker:
    """Call the configured ANRI model service for reranking candidate chunks."""

    def __init__(self):
        """Load remote inference settings and validate the service destination."""
        settings = get_settings()
        self.base_url = validate_model_service_url(settings.MODEL_SERVICE_URL)
        self.api_key = settings.MODEL_SERVICE_API_KEY
        self.timeout = settings.MODEL_SERVICE_TIMEOUT_SECONDS
        self.provider = "remote"
        self.model_name = settings.RERANKER_MODEL
        self.device = "remote"
        if not self.base_url:
            raise ValueError("MODEL_SERVICE_URL must be configured for remote reranking.")
        if not self.api_key:
            raise ValueError("MODEL_SERVICE_API_KEY must be configured for remote reranking.")

    def warmup(self):
        """Do not block ANRI startup on the remote computer being online."""

    def rerank(self, query: str, chunks: List[DocumentChunk], top_n: Optional[int] = None) -> List[DocumentChunk]:
        """Rerank all candidates in batches and merge scores before applying the limit."""
        if not chunks or not query:
            return chunks

        settings = get_settings()
        limit = top_n if top_n is not None else settings.RETRIEVAL_TOP_K
        if limit <= 0:
            return []

        ranked_results = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(chunks), 64):
                    candidates = chunks[start:start + 64]
                    batch_limit = min(limit, len(candidates))
                    response = client.post(
                        f"{self.base_url}/v1/rerank",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "query": query,
                            "documents": [chunk.text for chunk in candidates],
                            "top_n": batch_limit,
                        },
                    )
                    response.raise_for_status()
                    results = response.json().get("results")
                    if not isinstance(results, list) or len(results) != batch_limit:
                        raise RuntimeError("Remote reranking service returned an invalid result list.")

                    batch_indices = set()
                    for result in results:
                        try:
                            index = result["index"]
                            score = float(result["score"])
                            if (
                                type(index) is not int
                                or index < 0
                                or index >= len(candidates)
                                or index in batch_indices
                                or not math.isfinite(score)
                            ):
                                raise ValueError("invalid candidate index or non-finite score")
                        except (KeyError, TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "Remote reranking service returned an invalid candidate index or score."
                            ) from exc
                        batch_indices.add(index)
                        ranked_results.append((start + index, score))
        except Exception as exc:
            raise RuntimeError(f"Remote reranking service request failed: {exc}") from exc

        ranked_results.sort(key=lambda item: (item[1], -item[0]), reverse=True)

        ranked_chunks = []
        for index, score in ranked_results[:limit]:
            chunk = chunks[index]
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["rerank_score"] = round(score, 4)
            ranked_chunks.append(chunk)
        return ranked_chunks


def get_reranker():
    """Return the thread-safe singleton configured local or remote reranker."""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        with _RERANKER_LOCK:
            if _GLOBAL_RERANKER is None:
                settings = get_settings()
                if settings.MODEL_SERVICE_URL:
                    _GLOBAL_RERANKER = RemoteModelServiceReranker()
                else:
                    _GLOBAL_RERANKER = LocalReranker()
    return _GLOBAL_RERANKER
