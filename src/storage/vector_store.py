import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from schemas import DocumentChunk
from config import get_settings

logger = logging.getLogger(__name__)

_STORE_LOCK = threading.RLock()
_GLOBAL_VECTOR_STORE: Optional["QdrantVectorStore"] = None



class QdrantVectorStore:
    """Manages document chunk indexing, deduplication, and vector search in Qdrant."""

    _instance: Optional["QdrantVectorStore"] = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        # Return existing singleton if no custom connection overrides are passed
        if not args and not kwargs:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                return cls._instance
        return super().__new__(cls)

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        with self._lock:
            if getattr(self, "_initialized", False):
                return

            settings = get_settings()
            self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            if settings.EMBEDDING_PROVIDER.lower() == "gemini":
                self.vector_dim = getattr(settings, "GEMINI_EMBEDDING_DIM", 384)
            else:
                self.vector_dim = getattr(settings, "EMBEDDING_DIM", 384)

            # Disk storage preferred over in-memory for persistent document RAG
            self.is_remote = bool(url or settings.QDRANT_URL)
            if self.is_remote:
                target_url = url or settings.QDRANT_URL
                target_key = api_key or settings.QDRANT_API_KEY
                self.client = QdrantClient(url=target_url, api_key=target_key)
            else:
                storage_path = path or settings.QDRANT_PATH
                os.makedirs(storage_path, exist_ok=True)
                self.client = QdrantClient(path=storage_path)

            self.ensure_collection()
            self._initialized = True

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        """Create Qdrant collection and payload indexes if they do not already exist.
        
        Detects an existing collection with a mismatched vector dimension and
        raises an actionable error without deleting indexed data.
        """
        target_collection = collection_name or self.collection_name
        target_dim = vector_dim or self.vector_dim

        if self.client.collection_exists(collection_name=target_collection):
            try:
                col_info = self.client.get_collection(collection_name=target_collection)
                vectors_cfg = col_info.config.params.vectors
                existing_dim = getattr(vectors_cfg, "size", None)
                if existing_dim is None and isinstance(vectors_cfg, dict):
                    existing_dim = getattr(next(iter(vectors_cfg.values())), "size", None)

                if existing_dim is not None and existing_dim != target_dim:
                    raise ValueError(
                        f"Vector dimension mismatch for Qdrant collection '{target_collection}': "
                        f"the existing collection uses {existing_dim} dimensions, but the configured "
                        f"embedding model uses {target_dim}. Existing vectors were preserved. Restore "
                        f"the previous embedding configuration, or re-embed the source documents into "
                        f"a new collection before switching models."
                    )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"[QdrantVectorStore] Could not inspect collection '{target_collection}': {e}")

        if not self.client.collection_exists(collection_name=target_collection):
            self.client.create_collection(
                collection_name=target_collection,
                vectors_config=models.VectorParams(
                    size=target_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload keyword index only in server mode (local disk mode does not need/support it)
            if self.is_remote:
                self.client.create_payload_index(
                    collection_name=target_collection,
                    field_name="doc_hash",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def has_doc_hash(
        self,
        doc_hash: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Check if any chunks associated with doc_hash already exist in the store."""
        target_collection = collection_name or self.collection_name
        result = self.client.count(
            collection_name=target_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_hash",
                        match=models.MatchValue(value=doc_hash),
                    )
                ]
            ),
            exact=True,
        )
        return result.count > 0

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        vectors: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> None:
        """Upsert document chunks and their dense embeddings into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(vectors)} vectors."
            )
        if not chunks:
            return

        target_collection = collection_name or self.collection_name
        if vectors:
            actual_dim = len(vectors[0])
            if actual_dim != self.vector_dim:
                self.vector_dim = actual_dim
            self.ensure_collection(collection_name=target_collection, vector_dim=actual_dim)

        points: List[models.PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            # Deterministic UUID derived from chunk ID for idempotent re-writes
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload: Dict[str, Any] = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                # Promote critical metadata fields to top-level payload for indexing & filtering
                "doc_hash": chunk.metadata.get("doc_hash"),
                "chunk_id": chunk.metadata.get("chunk_id", chunk.id),
                "source_file": chunk.metadata.get("source_file"),
                "page_numbers": chunk.metadata.get("page_numbers", []),
                "section_path": chunk.metadata.get("section_path", ""),
                "content_type": chunk.metadata.get("content_type", "text"),
            }
            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Perform dense vector cosine similarity search."""
        target_collection = collection_name or self.collection_name

        try:
            # Modern query_points API
            response = self.client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            scored_points = response.points
        except (AttributeError, TypeError):
            # Fallback to search API for older client bindings
            scored_points = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

        retrieved: List[DocumentChunk] = []
        for point in scored_points:
            payload = point.payload or {}
            retrieved.append(
                DocumentChunk(
                    id=payload.get("id", str(point.id)),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", payload),
                    score=point.score,
                )
            )
        return retrieved

    def delete_by_doc_hash(
        self,
        doc_hash: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Purge all chunks associated with a specific doc_hash."""
        target_collection = collection_name or self.collection_name
        self.client.delete(
            collection_name=target_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_hash",
                            match=models.MatchValue(value=doc_hash),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def close(self) -> None:
        """Explicitly close the Qdrant local storage connection."""
        try:
            self.client.close()
        except Exception:
            pass


def get_vector_store() -> QdrantVectorStore:
    """Return the shared singleton instance of QdrantVectorStore with thread synchronization."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        with _STORE_LOCK:
            if _GLOBAL_VECTOR_STORE is None:
                _GLOBAL_VECTOR_STORE = QdrantVectorStore()
    return _GLOBAL_VECTOR_STORE
