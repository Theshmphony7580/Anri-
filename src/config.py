import os

# Prevent OpenBLAS/MKL thread memory exhaustion crashes on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QDRANT_PATH = str(ROOT_DIR / "qdrant_data")
DEFAULT_ENV_FILE = str(ROOT_DIR / ".env")


class Settings(BaseSettings):
    """Central application settings loaded from environment or .env file."""

    # --- LLM Provider Selection ---
    LLM_PROVIDER: str = "groq"

    # --- Groq Configuration ---
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_TEMPERATURE: float = 0.3

    # --- Google Gemini Configuration ---
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_DIM: int = 384

    # --- Local Hugging Face Embedding Configuration ---
    EMBEDDING_PROVIDER: str = "huggingface"
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    EMBEDDING_DEVICE: str = "cuda"

    # --- Optional remote local-model service (e.g. behind an authenticated tunnel) ---
    MODEL_SERVICE_URL: Optional[str] = None
    MODEL_SERVICE_API_KEY: Optional[str] = None
    MODEL_SERVICE_TIMEOUT_SECONDS: float = 120.0

    # --- Qdrant Vector Database (Local Disk Default) ---
    QDRANT_PATH: str = DEFAULT_QDRANT_PATH
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "knowledge_chunks"

    # --- Reranker Configuration (Two-Stage Retrieval) ---
    USE_RERANKER: bool = True
    RERANKER_PROVIDER: str = "flashrank"
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    RERANK_CANDIDATES_K: int = 15
    RERANKER_DEVICE: str = "cuda"

    # --- Ingestion & RAG Tuning ---
    DO_OCR: bool = False
    DO_TABLE_STRUCTURE: bool = False
    FORCE_BACKEND_TEXT: bool = True
    GENERATE_PICTURE_IMAGES: bool = False
    CONFIDENCE_THRESHOLD: float = 0.70
    RETRIEVAL_TOP_K: int = 4
    ALLOW_MOCK_FALLBACK: bool = False

    model_config = SettingsConfigDict(
        env_file=(DEFAULT_ENV_FILE, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
