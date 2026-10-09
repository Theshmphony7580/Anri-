"""Dense embedding engines supporting local Hugging Face models and Google Gemini API.

Defaults to local execution using Hugging Face's `BAAI/bge-small-en-v1.5` via `sentence-transformers`
(384 dimensions, normalized for cosine similarity).
"""

import hashlib
import logging
import math
import threading
from typing import List, Optional

import httpx

from config import get_settings, validate_model_service_url

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None
_EMBEDDER_LOCK = threading.RLock()


class HuggingFaceEmbedder:
    """Local dense embedding model using Hugging Face sentence-transformers."""

    _shared_model = None
    _shared_model_name: Optional[str] = None
    _shared_device: Optional[str] = None
    _lock = threading.RLock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.vector_dim = vector_dim or settings.EMBEDDING_DIM
        self._model = None

    def _load_model(self):
        """Load and cache the SentenceTransformer model once across the process."""
        with HuggingFaceEmbedder._lock:
            if (
                HuggingFaceEmbedder._shared_model is None
                or HuggingFaceEmbedder._shared_model_name != self.model_name
                or HuggingFaceEmbedder._shared_device != self.device
            ):
                try:
                    target_device = self.device
                    if target_device.startswith("cuda"):
                        try:
                            import torch
                            if not torch.cuda.is_available():
                                print("[HuggingFaceEmbedder] CUDA requested but torch.cuda.is_available() is False. Falling back to 'cpu'.")
                                target_device = "cpu"
                            else:
                                gpu_name = torch.cuda.get_device_name(0)
                                print(f"[HuggingFaceEmbedder] GPU detected: {gpu_name} ({target_device})")
                        except Exception as torch_err:
                            print(f"[HuggingFaceEmbedder] PyTorch CUDA check failed ({torch_err}). Falling back to 'cpu'.")
                            target_device = "cpu"

                    from sentence_transformers import SentenceTransformer
                    print(f"[HuggingFaceEmbedder] Loading local embedding model '{self.model_name}' onto {target_device}...")
                    try:
                        # Fast offline load without checking Hugging Face remote repository
                        HuggingFaceEmbedder._shared_model = SentenceTransformer(
                            self.model_name, device=target_device, local_files_only=True
                        )
                    except Exception:
                        HuggingFaceEmbedder._shared_model = SentenceTransformer(
                            self.model_name, device=target_device
                        )
                    HuggingFaceEmbedder._shared_model_name = self.model_name
                    HuggingFaceEmbedder._shared_device = target_device
                    self.device = target_device
                    print(f"[HuggingFaceEmbedder] Model '{self.model_name}' successfully loaded into memory on {target_device}.")
                except ImportError as err:
                    settings = get_settings()
                    if not settings.ALLOW_MOCK_FALLBACK:
                        raise RuntimeError(
                            "[HuggingFaceEmbedder] 'sentence-transformers' is not installed and ALLOW_MOCK_FALLBACK=False."
                        ) from err
                    logger.warning(
                        "[HuggingFaceEmbedder] 'sentence-transformers' not installed. "
                        "ALLOW_MOCK_FALLBACK=True: Falling back to deterministic mock embedding."
                    )
                    HuggingFaceEmbedder._shared_model = False
                    HuggingFaceEmbedder._shared_model_name = self.model_name
                    HuggingFaceEmbedder._shared_device = target_device
                except Exception as e:
                    settings = get_settings()
                    if not settings.ALLOW_MOCK_FALLBACK:
                        raise RuntimeError(
                            f"[HuggingFaceEmbedder] Fatal: Failed to load '{self.model_name}' on {target_device}: {e}. "
                            "ALLOW_MOCK_FALLBACK is False."
                        ) from e
                    logger.warning(
                        f"[HuggingFaceEmbedder] Failed to load '{self.model_name}' ({e}). "
                        "ALLOW_MOCK_FALLBACK=True: Falling back to deterministic mock embedding."
                    )
                    HuggingFaceEmbedder._shared_model = False
                    HuggingFaceEmbedder._shared_model_name = self.model_name
                    HuggingFaceEmbedder._shared_device = target_device

            self._model = HuggingFaceEmbedder._shared_model
            return self._model

    def warmup(self):
        """Eagerly load model into memory and perform a dummy encoding to warm up execution paths."""
        model = self._load_model()
        if model:
            try:
                model.encode(["system warmup query"], normalize_embeddings=True, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"[HuggingFaceEmbedder] Warmup error ({e})")

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate 384-dim normalized embedding for a single text."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        settings = get_settings()
        model = self._load_model()
        if model:
            try:
                # BGE recommendation: prepend retrieval instruction to query for optimal ranking
                prepared_texts = texts
                if is_query and "bge" in self.model_name.lower():
                    prefix = "Represent this sentence for searching relevant passages: "
                    prepared_texts = [f"{prefix}{t}" for t in texts]

                embeddings = model.encode(
                    prepared_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [arr.tolist() for arr in embeddings]
            except Exception as e:
                if not settings.ALLOW_MOCK_FALLBACK:
                    raise RuntimeError(f"[HuggingFaceEmbedder] Batch embedding inference error: {e}") from e
                logger.warning(f"[HuggingFaceEmbedder] Local inference error ({e}). ALLOW_MOCK_FALLBACK=True: Falling back to mock.")

        if not settings.ALLOW_MOCK_FALLBACK:
            raise RuntimeError(
                f"[HuggingFaceEmbedder] Model '{self.model_name}' is unavailable and ALLOW_MOCK_FALLBACK=False."
            )

        # Offline / deterministic fallback when mock is explicitly allowed
        return [self._generate_mock_embedding(t) for t in texts]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Deterministic 384-dimensional normalized unit vector generated from SHA-256 hash."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw_values = []
        for i in range(self.vector_dim):
            byte_val = seed[i % len(seed)]
            val = ((byte_val + i * 17) % 100) / 100.0 - 0.5
            raw_values.append(val)

        # Normalize to unit length for Cosine metric
        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        return [v / norm for v in raw_values]


class GeminiEmbedder:
    """Generates dense vector embeddings using Google Gemini API with offline test fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_EMBEDDING_MODEL
        self.vector_dim = vector_dim or settings.GEMINI_EMBEDDING_DIM
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate 384-dim embedding for a single text chunk."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        settings = get_settings()
        if not self.api_key:
            if not settings.ALLOW_MOCK_FALLBACK:
                raise ValueError(
                    "[GeminiEmbedder] GEMINI_API_KEY is not configured and ALLOW_MOCK_FALLBACK=False. "
                    "Cannot generate embeddings."
                )
            logger.warning("[GeminiEmbedder] No API key; using deterministic mock embeddings (ALLOW_MOCK_FALLBACK=True).")
            return [self._generate_mock_embedding(t) for t in texts]

        try:
            return self._call_gemini_batch(texts)
        except Exception as e:
            if not settings.ALLOW_MOCK_FALLBACK:
                raise RuntimeError(f"[GeminiEmbedder] Live API call failed: {e}. ALLOW_MOCK_FALLBACK=False.") from e
            logger.warning(f"[GeminiEmbedder] Live API call failed ({e}). ALLOW_MOCK_FALLBACK=True: falling back to mock.")

        return [self._generate_mock_embedding(t) for t in texts]

    def _call_gemini_batch(self, texts: List[str]) -> List[List[float]]:
        """Call Google Gemini batchEmbedContents endpoint via httpx using secure HTTP headers."""
        url = f"{self.base_url}/models/{self.model}:batchEmbedContents"
        api_key = (self.api_key or "").strip()
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        requests_payload = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.vector_dim,
            }
            for t in texts
        ]
        payload = {"requests": requests_payload}

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings_data = data.get("embeddings", [])
            return [item["values"] for item in embeddings_data]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Deterministic 384-dimensional normalized unit vector generated from SHA-256 hash."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw_values = []
        for i in range(self.vector_dim):
            byte_val = seed[i % len(seed)]
            val = ((byte_val + i * 17) % 100) / 100.0 - 0.5
            raw_values.append(val)

        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        return [v / norm for v in raw_values]


class RemoteModelServiceEmbedder:
    """Call a compatible ANRI model service for embeddings instead of loading weights here."""

    def __init__(self):
        """Load remote inference settings and validate the service destination."""
        settings = get_settings()
        self.base_url = validate_model_service_url(settings.MODEL_SERVICE_URL)
        self.api_key = settings.MODEL_SERVICE_API_KEY
        self.vector_dim = settings.EMBEDDING_DIM
        self.timeout = settings.MODEL_SERVICE_TIMEOUT_SECONDS
        if not self.base_url:
            raise ValueError("MODEL_SERVICE_URL must be configured for remote embeddings.")
        if not self.api_key:
            raise ValueError("MODEL_SERVICE_API_KEY must be configured for remote embeddings.")

    def warmup(self):
        """Do not block ANRI startup on the remote computer being online."""

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Return one vector from the remote service."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed texts in bounded batches while preserving input order."""
        if not texts:
            return []

        vectors = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(texts), 32):
                    batch = texts[start:start + 32]
                    response = client.post(
                        f"{self.base_url}/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"texts": batch, "is_query": is_query},
                    )
                    response.raise_for_status()
                    batch_vectors = response.json().get("embeddings")
                    if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                        raise RuntimeError("Remote embedding service returned an unexpected number of vectors.")
                    vectors.extend(batch_vectors)
        except Exception as exc:
            raise RuntimeError(f"Remote embedding service request failed: {exc}") from exc

        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Remote embedding service returned an unexpected number of vectors.")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.vector_dim:
                raise RuntimeError(
                    f"Remote embedding dimension does not match EMBEDDING_DIM={self.vector_dim}; "
                    "keep the local and ANRI embedding model/configuration identical."
                )
        try:
            converted = [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Remote embedding service returned a non-numeric vector value.") from exc
        if any(not math.isfinite(value) for vector in converted for value in vector):
            raise RuntimeError("Remote embedding service returned a non-finite vector value.")
        return converted


def get_embedder():
    """Return the thread-safe singleton instance of the configured embedding provider."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _GLOBAL_EMBEDDER is None:
                settings = get_settings()
                if settings.MODEL_SERVICE_URL:
                    _GLOBAL_EMBEDDER = RemoteModelServiceEmbedder()
                else:
                    provider = settings.EMBEDDING_PROVIDER.lower()
                    if provider == "gemini":
                        _GLOBAL_EMBEDDER = GeminiEmbedder()
                    else:
                        _GLOBAL_EMBEDDER = HuggingFaceEmbedder()
    return _GLOBAL_EMBEDDER


# Backward-compatible alias
LocalEmbedder = HuggingFaceEmbedder
