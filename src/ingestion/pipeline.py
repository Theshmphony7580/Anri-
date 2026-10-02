from dataclasses import dataclass
from typing import Any, List, Optional

from schemas import DocumentChunk
from storage.vector_store import QdrantVectorStore, get_vector_store
from ingestion.parser import DocumentParser, compute_file_sha256
from ingestion.embeddings import HuggingFaceEmbedder, GeminiEmbedder, get_embedder
from ingestion.vision import VLMVisionCaptioner


@dataclass
class IngestionResult:
    """Outcome of a document ingestion run."""
    source_file: str
    doc_hash: str
    chunk_count: int
    status: str  # "indexed" | "skipped_duplicate" | "error"
    message: str


class IngestionPipeline:
    """
    End-to-end ingestion orchestrator:
    1. Deduplication via SHA-256
    2. Structure-aware parsing via Docling
    3. Metadata enrichment
    4. Dense vector embedding via Gemini
    5. Qdrant disk storage
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        parser: Optional[DocumentParser] = None,
        embedder: Optional[Any] = None,
        vision_captioner: Optional[VLMVisionCaptioner] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.vision_captioner = vision_captioner or VLMVisionCaptioner()
        self.parser = parser or DocumentParser(vision_captioner=self.vision_captioner)
        self.embedder = embedder or get_embedder()

    def ingest_file(self, file_path: str, force_reindex: bool = False) -> IngestionResult:
        """Run the ingestion pipeline for a single file."""
        import time
        t0 = time.time()
        doc_hash = compute_file_sha256(file_path)
        print(f"\n[IngestionPipeline] >>> Starting ingestion for: {file_path}")

        # Step 1: Deduplication check
        if not force_reindex and self.vector_store.has_doc_hash(doc_hash):
            print(f"[IngestionPipeline] Document already indexed with hash {doc_hash[:10]}... Skipping.")
            return IngestionResult(
                source_file=file_path,
                doc_hash=doc_hash,
                chunk_count=0,
                status="skipped_duplicate",
                message=f"Document already indexed with hash {doc_hash[:10]}... Skipping.",
            )

        if force_reindex:
            print(f"[IngestionPipeline] Force re-index enabled. Existing vectors will be replaced after the new content is indexed.")

        # Step 2: Parse and chunk with metadata
        print("[IngestionPipeline] [1/3] Parsing document & extracting hierarchical chunks...")
        t_parse = time.time()
        chunks = self.parser.parse_document(file_path)
        print(f"[IngestionPipeline] [1/3] Parsing complete: {len(chunks)} chunks in {time.time() - t_parse:.2f}s.")

        if not chunks:
            return IngestionResult(
                source_file=file_path,
                doc_hash=doc_hash,
                chunk_count=0,
                status="error",
                message="Parser produced 0 chunks. Document may be empty or unreadable.",
            )

        # Step 3: Generate dense embeddings
        print(f"[IngestionPipeline] [2/3] Generating dense embeddings for {len(chunks)} chunks...")
        t_embed = time.time()
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts)
        print(f"[IngestionPipeline] [2/3] Embeddings generated in {time.time() - t_embed:.2f}s.")

        # Step 4: Upsert to Qdrant
        print(f"[IngestionPipeline] [3/3] Upserting vectors into Qdrant collection '{self.vector_store.collection_name}'...")
        t_upsert = time.time()
        self.vector_store.upsert_chunks(chunks=chunks, vectors=vectors)
        if force_reindex:
            self.vector_store.delete_by_doc_hash(
                doc_hash,
                exclude_chunk_ids=[c.metadata.get("chunk_id", c.id) for c in chunks],
            )
        print(f"[IngestionPipeline] [3/3] Qdrant upsert complete in {time.time() - t_upsert:.2f}s.")

        total_elapsed = time.time() - t0
        print(f"[IngestionPipeline] >>> SUCCESS: Indexed {len(chunks)} chunks in {total_elapsed:.2f}s total.\n")

        return IngestionResult(
            source_file=file_path,
            doc_hash=doc_hash,
            chunk_count=len(chunks),
            status="indexed",
            message=f"Successfully indexed {len(chunks)} chunks into Qdrant in {total_elapsed:.1f}s.",
        )
