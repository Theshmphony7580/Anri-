from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from schemas import DocumentChunk
from config import get_settings
from ingestion.vision import VLMVisionCaptioner

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def compute_file_sha256(file_path: str) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_text_into_sliding_chunks(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    """Split text into semantically cohesive chunks strictly bounded by max_chars with sliding overlap.

    1,200 characters is ~250-300 tokens, safely within the 512-token context limit
    of local BGE embeddings (BAAI/bge-small-en-v1.5) and Gemini embeddings.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= max_chars:
        return [cleaned]

    # Split into paragraphs first
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    # Break down any oversized paragraphs into smaller units
    atomic_units: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            atomic_units.append(para)
        else:
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            for line in lines:
                if len(line) <= max_chars:
                    atomic_units.append(line)
                else:
                    sentences = re.split(r"(?<=[.!?])\s+", line)
                    for sent in sentences:
                        sent_clean = sent.strip()
                        if not sent_clean:
                            continue
                        if len(sent_clean) <= max_chars:
                            atomic_units.append(sent_clean)
                        else:
                            step = max(100, max_chars - overlap_chars)
                            for i in range(0, len(sent_clean), step):
                                piece = sent_clean[i : i + max_chars].strip()
                                if piece:
                                    atomic_units.append(piece)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for unit in atomic_units:
        unit_len = len(unit)
        if current_chunk and (current_len + unit_len + 2 > max_chars):
            combined_text = "\n\n".join(current_chunk).strip()
            chunks.append(combined_text)

            if overlap_chars > 0 and len(combined_text) > overlap_chars:
                overlap_text = combined_text[-overlap_chars:].strip()
                space_idx = overlap_text.find(" ")
                if space_idx != -1 and space_idx < len(overlap_text) - 1:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = [overlap_text, unit] if overlap_text else [unit]
                current_len = sum(len(u) for u in current_chunk) + 2
            else:
                current_chunk = [unit]
                current_len = unit_len
        else:
            current_chunk.append(unit)
            current_len += unit_len + (2 if current_len > 0 else 0)

    if current_chunk:
        combined_text = "\n\n".join(current_chunk).strip()
        if combined_text and (not chunks or chunks[-1] != combined_text):
            chunks.append(combined_text)

    # Final guarantee: ensure no chunk exceeds max_chars
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            step = max(100, max_chars - overlap_chars)
            for i in range(0, len(c), step):
                piece = c[i : i + max_chars].strip()
                if piece:
                    final_chunks.append(piece)

    return final_chunks


class DocumentParser:
    """
    Parses digital documents using IBM Docling with `do_ocr=False` permanently enforced.
    Extracts text, preserves markdown tables, and routes figures/diagrams to Gemini Vision.
    """

    def __init__(self, vision_captioner: Optional[VLMVisionCaptioner] = None):
        self.settings = get_settings()
        self.vision_captioner = vision_captioner or VLMVisionCaptioner()

    def parse_document(self, file_path: str) -> List[DocumentChunk]:
        """Parse a document or image file into enriched DocumentChunk instances."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_hash = compute_file_sha256(file_path)
        source_path = Path(file_path)
        source_name = source_path.name
        now_iso = datetime.now(timezone.utc).isoformat()

        # Handle standalone image files directly with VLM Vision
        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return self._parse_image_file(file_path, doc_hash, source_name, now_iso)

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.chunking import HierarchicalChunker

            # Enforce permanent do_ocr=False for high speed, disable memory-heavy TableFormer & rasterization
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = getattr(self.settings, "DO_TABLE_STRUCTURE", False)
            pipeline_options.force_backend_text = getattr(self.settings, "FORCE_BACKEND_TEXT", True)
            pipeline_options.generate_page_images = False
            pipeline_options.generate_picture_images = getattr(self.settings, "GENERATE_PICTURE_IMAGES", False)
            pipeline_options.generate_table_images = False

            print(f"[DocumentParser] Converting '{source_name}' (force_backend_text={pipeline_options.force_backend_text})...")
            start_t = datetime.now()

            converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            result = converter.convert(file_path)
            doc = result.document
            chunker = HierarchicalChunker()
            docling_chunks = list(chunker.chunk(doc))
            elapsed = (datetime.now() - start_t).total_seconds()
            print(f"[DocumentParser] Converted '{source_name}' in {elapsed:.2f}s. Generated {len(docling_chunks)} hierarchical chunks.")

            parsed_chunks: List[DocumentChunk] = []

            # 1. Text and Table Chunks (bounded by embedding model context limit)
            chunk_counter = 0
            for c in docling_chunks:
                # If Docling produces an oversized chunk, split it into bounded sliding chunks
                sub_texts = (
                    split_text_into_sliding_chunks(c.text, max_chars=1200, overlap_chars=150)
                    if len(c.text) > 1200
                    else [c.text]
                )
                for sub_text in sub_texts:
                    chunk_id = f"{doc_hash}_{chunk_counter:04d}"
                    meta: Dict[str, Any] = {
                        "doc_hash": doc_hash,
                        "chunk_id": chunk_id,
                        "source_file": source_name,
                        "page_numbers": getattr(c.meta, "page_numbers", [1]) or [1],
                        "section_path": " > ".join(getattr(c.meta, "headings", [])) or "Main",
                        "content_type": "table" if getattr(c.meta, "is_table", False) else "text",
                        "created_at": now_iso,
                    }
                    parsed_chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            text=sub_text,
                            metadata=meta,
                        )
                    )
                    chunk_counter += 1

            # 2. Figure / Diagram Extraction via Gemini Vision (only if enabled)
            if getattr(self.settings, "GENERATE_PICTURE_IMAGES", False) and hasattr(doc, "pictures") and doc.pictures:
                for p_idx, picture in enumerate(doc.pictures):
                    try:
                        # Crop ONLY the isolated bounding box of the figure (not the whole page)
                        pil_image = picture.get_image(doc)
                        if pil_image:
                            img_byte_arr = io.BytesIO()
                            pil_image.save(img_byte_arr, format="PNG")
                            raw_bytes = img_byte_arr.getvalue()

                            # Extract exact page number from Docling provenance
                            page_no = 1
                            if hasattr(picture, "prov") and picture.prov:
                                page_no = getattr(picture.prov[0], "page_no", 1)
                            elif hasattr(picture, "page_no"):
                                page_no = picture.page_no

                            # Extract any textual caption already identified by Docling (e.g., "Figure 2: Architecture")
                            doc_caption = ""
                            if hasattr(picture, "captions") and picture.captions:
                                doc_caption = " ".join([getattr(c, "text", str(c)) for c in picture.captions if c]).strip()

                            # Call VLM on the cropped image
                            caption = self.vision_captioner.describe_figure(raw_bytes, mime_type="image/png")
                            if caption.strip():
                                fig_chunk_id = f"{doc_hash}_p{page_no}_fig_{p_idx:03d}"
                                fig_meta: Dict[str, Any] = {
                                    "doc_hash": doc_hash,
                                    "chunk_id": fig_chunk_id,
                                    "source_file": source_name,
                                    "page_numbers": [page_no],
                                    "section_path": f"Page {page_no} > Figures & Diagrams",
                                    "content_type": "diagram_vlm",
                                    "caption": doc_caption,
                                    "created_at": now_iso,
                                }

                                figure_text = (
                                    f"### Visual Figure on Page {page_no} ({doc_caption})\n{caption}"
                                    if doc_caption
                                    else f"### Visual Figure on Page {page_no}\n{caption}"
                                )

                                parsed_chunks.append(
                                    DocumentChunk(
                                        id=fig_chunk_id,
                                        text=figure_text,
                                        metadata=fig_meta,
                                    )
                                )
                    except Exception as err:
                        # Continue processing remaining chunks if a single figure fails
                        continue

            return parsed_chunks

        except Exception as e:
            # Fallback parser for text/markdown/pdf files when Docling encounters memory or format constraints
            print(f"[DocumentParser] Docling parsing encountered an issue ({e}). Falling back to stream extraction.")
            return self._fallback_text_parse(file_path, doc_hash, source_name, now_iso)

    def _parse_image_file(
        self,
        file_path: str,
        doc_hash: str,
        source_name: str,
        timestamp: str,
    ) -> List[DocumentChunk]:
        """Directly summarize and index a standalone image file via VLM."""
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        suffix = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        caption = self.vision_captioner.describe_figure(raw_bytes, mime_type=mime_type)
        chunk_id = f"{doc_hash}_0000"
        meta: Dict[str, Any] = {
            "doc_hash": doc_hash,
            "chunk_id": chunk_id,
            "source_file": source_name,
            "page_numbers": [1],
            "section_path": "Standalone Figure",
            "content_type": "diagram_vlm",
            "created_at": timestamp,
        }
        return [
            DocumentChunk(
                id=chunk_id,
                text=f"### Visual Figure Description\n{caption}",
                metadata=meta,
            )
        ]

    def _fallback_text_parse(
        self,
        file_path: str,
        doc_hash: str,
        source_name: str,
        timestamp: str,
    ) -> List[DocumentChunk]:
        """Robust, layout-aware text extraction for PDF, plain text, and markdown documents."""
        suffix = Path(file_path).suffix.lower()
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # --- 1. Page-Preserving PDF Traversal ---
        if suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for page_idx, page in enumerate(reader.pages):
                    page_no = page_idx + 1
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        continue

                    # Chunk page text strictly bounded by embedding model context limit (max_chars=1200)
                    page_chunks = split_text_into_sliding_chunks(
                        page_text, max_chars=1200, overlap_chars=150
                    )
                    for text in page_chunks:
                        chunk_id = f"{doc_hash}_{chunk_idx:04d}"
                        meta: Dict[str, Any] = {
                            "doc_hash": doc_hash,
                            "chunk_id": chunk_id,
                            "source_file": source_name,
                            "page_numbers": [page_no],
                            "section_path": f"Page {page_no}",
                            "content_type": "text",
                            "created_at": timestamp,
                        }
                        chunks.append(
                            DocumentChunk(
                                id=chunk_id,
                                text=text,
                                metadata=meta,
                            )
                        )
                        chunk_idx += 1

                if chunks:
                    print(
                        f"[DocumentParser] Fallback PDF parser extracted {len(chunks)} chunks "
                        f"across {len(reader.pages)} pages (all bounded <= 1,200 chars)."
                    )
                    return chunks

            except Exception as pdf_err:
                print(f"[DocumentParser] PDF fallback extractor error ({pdf_err}). Reading raw stream.")

        # --- 2. Text / Markdown / Raw Stream Traversal ---
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        except Exception as read_err:
            print(f"[DocumentParser] Error reading file stream: {read_err}")
            return []

        text_chunks = split_text_into_sliding_chunks(
            full_text, max_chars=1200, overlap_chars=150
        )
        for text in text_chunks:
            chunk_id = f"{doc_hash}_{chunk_idx:04d}"
            # Extract header if present in snippet for section path
            first_line = text.split("\n", 1)[0].strip()
            section = first_line[:40] if first_line.startswith("#") else "Document"

            meta: Dict[str, Any] = {
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "source_file": source_name,
                "page_numbers": [1],
                "section_path": section,
                "content_type": "text",
                "created_at": timestamp,
            }
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=text,
                    metadata=meta,
                )
            )
            chunk_idx += 1

        print(
            f"[DocumentParser] Fallback text parser extracted {len(chunks)} chunks "
            f"(all strictly bounded <= 1,200 chars within embedding context limit)."
        )
        return chunks
