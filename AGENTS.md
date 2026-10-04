# AGENTS.md

## Operational Rules
1. **Execution Environment:** Do not run terminal commands without explicit user instruction. When running commands, always execute inside the project virtual environment (`.venv`).
2. **Progress Tracking:** Update this file (`AGENTS.md`) across every turn to preserve context, architectural decisions, and current task state.
3. **Communication Style:** Follow ADHD guidelines (lead with action, cap lists at 5, no fluff, clear 1-line next actions).

---

## Project Overview
- **Project:** ANRI (Enterprise Document RAG Agent)
- **Core Orchestrator:** LangGraph state graph with dynamic self-correction loops
- **Vector Database:** Qdrant (local `:memory:` / directory / remote server)
- **Embedding Model:** Local Hugging Face (`BAAI/bge-small-en-v1.5`, 384-dim) via `sentence-transformers` (with Gemini API & deterministic test fallbacks)
- **Inference Models:** Groq (`qwen/qwen3.8-27b`) & Google Gemini (`gemini-3.6-flash`)
- **Document Parsing:** IBM Docling (`HierarchicalChunker` + `TableFormer`) & Gemini Vision for figures
- **Package Manager:** `uv` with Python 3.12+

---

## Architecture Diagrams

### 1. Agent Reasoning Loop (LangGraph State Machine)
```
[User Query] ──► [retrieve_node] (Top-15) ──► [rerank_node] (BGE Cross-Encoder Top-4) ──► [grade_node] ──┬──► (score >= 0.70 or max retries) ──► [generate_node] ──► [Answer]
                                                                                                          └──► (score < 0.70 & retry < 2)     ──► [rewrite_node]  ──► [retrieve_node]
```

### 2. Document Ingestion Pipeline
```
[Document Upload] ──► [SHA-256 Check] ──┬──► (Already Exists) ──► [Skip / Fast Return]
                                        └──► (New Document)   ──► [Docling Parser (do_ocr=False)]
                                                                          │
                        ┌─────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
                        ▼                                                                                                   ▼
             [Text & Table Elements]                                                                             [Cropped Charts / Visuals]
                        │                                                                                                   │
                        ▼                                                                                                   ▼
         [Docling Hierarchical Chunker]                                                                           [Gemini Vision (VLM)]
     (Preserves headers & whole tables)                                                                          (Generates text summary)
                        │                                                                                                   │
                        └─────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                                      [Metadata Injector]
                                           (doc_hash, page_num, section_path, type)
                                                                  │
                                                                  ▼
                                                       [Gemini Dense Embedder]
                                                        (gemini-embedding-001)
                                                                  │
                                                                  ▼
                                                         [Qdrant Vector DB]
                                                        (Upsert PointStruct)
```

---

## Data Schemas & Standards

### Chunk Metadata Schema
```python
{
    "doc_id": str,          # Deterministic document identifier
    "doc_hash": str,        # SHA-256 hex digest of source file
    "chunk_id": str,        # Deterministic ID: {doc_hash}_{chunk_index}
    "source_file": str,     # Source filename
    "page_numbers": list,   # List of page integers [int]
    "section_path": str,    # Header breadcrumbs: "H1 > H2 > H3"
    "content_type": str,    # "text" | "table" | "diagram_vlm"
    "created_at": str       # ISO-8601 UTC timestamp
}
```

### LangGraph State ([`src/agent/state.py`](src/agent/state.py))
- `question: str`: Current user query.
- `chat_history: List[BaseMessage]`: Multi-turn conversational context.
- `retrieved_chunks: List[DocumentChunk]`: Top-k chunks fetched from Qdrant.
- `confidence_score: float`: Graded relevance score (0.0 to 1.0).
- `rewritten_question: Optional[str]`: Reformulated query if retrieval fails threshold.
- `answer: str`: Final grounded response with citations.
- `retry_count: int`: Number of query rewrites executed (capped at 1).

---

## Ingestion Pre-initialization & Key Requirements

| Resource / Key | Purpose | Required For Ingestion? | Default / Fallback |
|---|---|---|---|
| `LOCAL_EMBEDDING_MODEL` | Local Hugging Face embeddings via `sentence-transformers` | **Yes** | `"BAAI/bge-small-en-v1.5"` (384-dim, CPU/CUDA) |
| `GEMINI_API_KEY` | VLM chart summaries + optional Gemini embeddings fallback | Optional | Offline mock available for testing |
| `GROQ_API_KEY` | Fast LLM inference (Llama 3.3) | No (Deferred to Agent phase) | Offline mock available for testing |
| `QDRANT_PATH` | Vector store backend on local disk | **Yes** | `"./qdrant_data"` (auto-created on disk) |
| `QDRANT_API_KEY` | Qdrant Cloud auth token | Optional | Empty for local disk |
| `QDRANT_COLLECTION_NAME` | Target collection name | **Yes** | `"knowledge_chunks"` (auto-created if missing) |
| Collection Specs | Cosine metric, 384 vector dimensions, keyword index on `doc_hash` | **Auto-initialized** | Handled on first run in `ensure_collection()` |

---

## File Structure & Map

```
anri/
├── .env.example                       # Environment variable templates
├── .gitignore                         # Git ignore file (excludes qdrant_data/, .venv)
├── AGENTS.md                          # Context, rules & progress tracking
├── list_gemini_models.py              # Root utility: list available Gemini models & types
├── prd.md                             # Original system PRD & technical architecture
├── pyproject.toml                     # uv project & dependency definitions
├── qdrant_data/                       # Local Qdrant disk storage (auto-created)
└── src/
    ├── __init__.py                    # Source root package
    ├── config.py                      # Pydantic-settings loader
    ├── schemas.py                     # DocumentChunk, QueryRequest, QueryResponse
    ├── agent/
    │   ├── __init__.py                # Agent module
    │   ├── state.py                   # LangGraph RAGState TypedDict
    │   ├── prompts.py                 # Centralized prompt templates (grade, rewrite, generate, refusal)
    │   ├── llm.py                     # Groq / Gemini REST inference client
    │   ├── reranker.py                # Local BAAI/bge-reranker-base Cross-Encoder reranker
    │   ├── nodes.py                   # retrieve, rerank, grade, generate, rewrite
    │   └── graph.py                   # StateGraph assembly & compilation
    ├── ingestion/
    │   ├── __init__.py                # Ingestion module
    │   ├── embeddings.py              # Local BGE embedding model (384-dim) + Gemini & mock fallbacks
    │   ├── parser.py                  # Docling parser (do_ocr=False) + HierarchicalChunker
    │   ├── pipeline.py                # IngestionPipeline orchestrator (SHA-256 deduplication)
    │   └── vision.py                  # VLM figure captioner (Forensic prompt + Gemini Vision)
    └── storage/
        ├── __init__.py                # Storage module
        └── vector_store.py            # QdrantVectorStore disk adapter
```

### Component Status
|---|---|---|
| [`list_gemini_models.py`](list_gemini_models.py) | Done | Root utility to list all models & capabilities for GEMINI_API_KEY |
| [`pyproject.toml`](pyproject.toml) | Done | Project dependencies managed via `uv` |
| [`.env.example`](.env.example) | Done | Environment variable templates (Groq, Gemini, Qdrant on disk, Reranker) |
| [`.vscode/settings.json`](.vscode/settings.json) | Done | Interpreter and import search paths |
| [`src/schemas.py`](src/schemas.py) | Done | Pydantic contracts (`DocumentChunk`, `QueryRequest`, `QueryResponse`) |
| [`src/agent/state.py`](src/agent/state.py) | Done | LangGraph `RAGState` TypedDict definition (including `rerank_scores`) |
| [`src/config.py`](src/config.py) | Done | Settings loader via `pydantic-settings` (with reranker toggles) |
| [`src/storage/vector_store.py`](src/storage/vector_store.py) | Done | Qdrant disk client adapter (ensure_collection, upsert, search) |
| [`src/ingestion/vision.py`](src/ingestion/vision.py) | Done | VLM figure captioner (Forensic Document Intelligence prompt + Gemini Vision) |
| [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py) | Done | Local Hugging Face (`BAAI/bge-small-en-v1.5`, 384-dim) + Gemini & mock fallbacks |
| [`src/ingestion/parser.py`](src/ingestion/parser.py) | Done | Docling parser (`do_ocr=False`) + VLM diagram integration + text fallback |
| [`src/ingestion/pipeline.py`](src/ingestion/pipeline.py) | Done | Ingestion orchestrator (`ingest_file`) with SHA-256 deduplication |
| [`src/agent/prompts.py`](src/agent/prompts.py) | Done | Centralized node prompt templates (grade, rewrite, generate, fallback refusal) |
| [`src/agent/llm.py`](src/agent/llm.py) | Done | Unified Groq / Gemini REST inference client with offline mock fallback |
| [`src/agent/reranker.py`](src/agent/reranker.py) | Done | Local Cross-Encoder reranker (`BAAI/bge-reranker-base`) with singleton & heuristic fallback |
| [`src/agent/nodes.py`](src/agent/nodes.py) | Done | LangGraph node functions (`retrieve`, `rerank`, `grade`, `generate`, `rewrite`) |
| [`src/agent/graph.py`](src/agent/graph.py) | Done | StateGraph compilation with conditional self-correction edge & 2-stage retrieval |
| [`test_agent_graph.py`](test_agent_graph.py) | Done | Root integration test for LangGraph reasoning loop, 2-stage retrieval & self-correction |
| [`src/dashboard/server.py`](src/dashboard/server.py) | Done | Lightweight FastAPI REST backend with reranker telemetry & live query execution |
| [`src/dashboard/static/`](src/dashboard/static/) | Done | Precision Observability Console SPA with reranker circuit node and score badges |
| [`main.py`](main.py) | Done | Root entry point launching Uvicorn server on http://127.0.0.1:8000 |
| [`README.md`](README.md) | Done | Production-grade architectural documentation, setup guide, and API reference |

---

## Current Status Checklist

- [x] Scaffolding and minimal dependencies installed via `uv`.
- [x] Defined core data models in [`src/schemas.py`](src/schemas.py).
- [x] Defined graph state in [`src/agent/state.py`](src/agent/state.py).
- [x] Configured interpreter and search paths in [`.vscode/settings.json`](.vscode/settings.json).
- [x] Configured [`.env.example`](.env.example) with local disk Qdrant storage.
- [x] Architected document ingestion pipeline (Docling chunking, SHA-256 hashing, metadata schema, `do_ocr=False`).
- [x] Implemented [`src/config.py`](src/config.py) with Pydantic settings loading disk paths.
- [x] Implemented [`src/storage/vector_store.py`](src/storage/vector_store.py) with Qdrant disk storage and `doc_hash` index.
- [x] Implemented [`src/ingestion/vision.py`](src/ingestion/vision.py) with finalized Forensic Document Intelligence prompt.
- [x] Standardized root imports across `src/` to align with project layout and eliminate linter warnings.
- [x] Implemented [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py) supporting local Hugging Face `BAAI/bge-small-en-v1.5` (384-dim), Gemini REST, and offline mock.
- [x] Implemented [`src/ingestion/parser.py`](src/ingestion/parser.py) with Docling `HierarchicalChunker` & metadata.
- [x] Implemented [`src/ingestion/pipeline.py`](src/ingestion/pipeline.py) orchestrator with SHA-256 deduplication.
- [x] Ingestion pipeline smoke test executed and verified on local disk (parsing, deduplication, vector search).
- [x] Ingestion milestone committed to git (`58db129` - "ingestion pipelined fineshed moving to nodes desing").
- [x] Implemented [`src/agent/prompts.py`](src/agent/prompts.py) with structured templates for all nodes.
- [x] Implemented [`src/agent/llm.py`](src/agent/llm.py) with Groq & Gemini REST APIs and offline mock fallback.
- [x] Implemented [`src/agent/nodes.py`](src/agent/nodes.py) with `retrieve`, `grade`, `rewrite`, and `generate` nodes.
- [x] Implemented [`src/agent/graph.py`](src/agent/graph.py) assembling `StateGraph(RAGState)` with self-correcting conditional edge.
- [x] Configured local Hugging Face embedding model (`BAAI/bge-small-en-v1.5`, 384 dimensions) via `sentence-transformers` in `pyproject.toml`, `config.py`, and `embeddings.py`.
- [x] Created root test suite [`test_agent_graph.py`](test_agent_graph.py) covering high confidence, query rewrites, and graceful refusal.
- [x] Designed visual concept mockup and detailed implementation plan for Precision Observability Console (anti-purple industrial aesthetic).
- [x] Implemented backend telemetry service in [`src/dashboard/server.py`](src/dashboard/server.py) with `/api/stats`, `/api/documents`, `/api/ingest`, `/api/upload`, and `/api/query`.
- [x] Built frontend console in [`src/dashboard/static/`](src/dashboard/static/) (`index.html`, `style.css`, `app.js`) with zero Node.js/npm dependencies.
- [x] Wired [`main.py`](main.py) to launch Uvicorn development server on port 8000.
- [x] Converted [`src/storage/vector_store.py`](src/storage/vector_store.py) to a thread-safe singleton (`get_vector_store()`) with `threading.RLock`, completely eliminating `portalocker.exceptions.AlreadyLocked` during concurrent frontend requests.
- [x] Optimized [`src/ingestion/parser.py`](src/ingestion/parser.py) with `do_table_structure=False`, `generate_picture_images=False`, and single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`), eliminating OpenBLAS memory exhaustion crashes on Windows.
- [x] Hardened [`src/dashboard/server.py`](src/dashboard/server.py) and [`app.js`](src/dashboard/static/app.js) with structured exception handling, preventing unhandled server crashes during ingestion.
- [x] Verified server startup and concurrency fix: `/api/stats` and `/api/documents` both returned `200 OK` simultaneously with zero lock contention.
- [x] Verified model weights loaded successfully (`770/770 [100%]`) without any OpenBLAS memory allocation errors.
- [x] Ingestion verified on local disk: `taotp.pdf` indexed 376 vectors into Qdrant collection `knowledge_chunks`.
- [x] LangGraph reasoning pipeline verified in web console: Retrieved 4 chunks, graded relevance at 0.88 (PASS >= 0.70), synthesized grounded answer.
- [x] Diagnostic logging added to [`src/agent/llm.py`](src/agent/llm.py) with candidate model retries (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gemini-flash-latest`).
- [x] Refactored `_generate_mock()` in [`src/agent/llm.py`](src/agent/llm.py) to dynamically extract real sentences and citations from retrieved document chunks matching query terms, eliminating canned placeholder responses.
- [x] Verified in browser with `browser_subagent`: executed `what are the transformers in 2 lines`, retrieved 4 chunks from `taotp.pdf`, graded relevance at 0.90 (PASS >= 0.70), and produced a precise grounded answer citing `taotp.pdf [p. 1]`.
- [x] Configured local Cross-Encoder reranker (`BAAI/bge-reranker-base`) with `RERANK_CANDIDATES_K=15` and `RETRIEVAL_TOP_K=4` in `config.py`, `.env.example`, and `.env`.
- [x] Implemented thread-safe [`src/agent/reranker.py`](src/agent/reranker.py) using `sentence_transformers.CrossEncoder` with `rerank_score` scoring and offline heuristic fallback.
- [x] Implemented `rerank_node` in [`src/agent/nodes.py`](src/agent/nodes.py) and integrated two-stage retrieval into StateGraph ([`src/agent/graph.py`](src/agent/graph.py)): `retrieve` ➔ `rerank` ➔ `grade` ➔ `generate`.
- [x] Updated telemetry backend ([`src/dashboard/server.py`](src/dashboard/server.py)) exposing reranker model, candidate counts, execution step tracing, and per-chunk rerank scores.
- [x] Enhanced frontend console ([`src/dashboard/static/`](src/dashboard/static/)) with dedicated `01.5 // RERANK` circuit node, real-time animation, and dual Cosine/Rerank score badges.
- [x] Restarted server (`uv run main.py`) running live with BGE Cross-Encoder reranker.
- [x] Reviewed and verified document chunking strategy: IBM Docling `HierarchicalChunker` (layout-aware, heading breadcrumbs, table preservation) + VLM diagram captioning + paragraph fallback.
- [x] Eliminated per-query model reload latency: converted `HuggingFaceEmbedder` to process-wide singleton cache with `local_files_only=True` offline fast path and `warmup()`.
- [x] Added `warmup()` and `local_files_only=True` to `BGEReranker` to bypass Hugging Face network ping overhead.
- [x] Implemented FastAPI `lifespan` handler in [`src/dashboard/server.py`](src/dashboard/server.py) to pre-warm all models directly into RAM on server startup (`uv run main.py`), delivering instant sub-second queries.
- [x] Diagnosed reranker loading halt (`MemoryError: `): PyTorch state_dict materialization ran out of heap RAM for the 1.1 GB `bge-reranker-base` model on Windows.
- [x] Resolved reranker memory crash: switched default model to `BAAI/bge-reranker-small` (130 MB, 8.5x lighter, <200MB RAM) in [`config.py`](src/config.py), [`.env`](.env), and added auto-recovery fallback in [`reranker.py`](src/agent/reranker.py).
- [x] Configured GPU (CUDA) execution for both `HuggingFaceEmbedder` (`EMBEDDING_DEVICE=cuda`) and `BGEReranker` (`RERANKER_DEVICE=cuda`) across [`config.py`](src/config.py), [`.env`](.env), and [`.env.example`](.env.example) with auto-detection and graceful CPU fallback.
- [x] Updated telemetry console ([`server.py`](src/dashboard/server.py) & [`app.js`](src/dashboard/static/app.js)) to expose and render `[CUDA]` execution badges in the top telemetry ribbon.
- [x] Preserved PyTorch CUDA wheels: removed `torch` from `pyproject.toml` dependencies and bounded `requires-python = ">=3.12,<3.13"`, preventing `uv run` from replacing CUDA binaries with PyPI CPU wheels. Instructed use of `uv run --no-sync main.py` or `python main.py`.
- [x] Resolved Qdrant vector shape broadcast crash (`could not broadcast input array from shape (384,) into shape (768,)`): added automated dimension mismatch detection and dynamic collection re-creation in [`src/storage/vector_store.py`](src/storage/vector_store.py) (`ensure_collection` and `upsert_chunks`), seamlessly handling embedding model upgrades between 768d and 384d.
- [x] Fixed Hugging Face 401 RepositoryNotFoundError for reranker: corrected non-existent `BAAI/bge-reranker-small` to `BAAI/bge-reranker-base` in [`.env`](.env), added automatic name normalization and lightweight `cross-encoder/ms-marco-MiniLM-L-6-v2` / FlashRank fallback support in [`src/agent/reranker.py`](src/agent/reranker.py).
- [x] Standardized Groq LLM inference model to `qwen/qwen3.8-27b` across [`config.py`](src/config.py), [`.env`](.env), [`.env.example`](.env.example), [`llm.py`](src/agent/llm.py), and frontend console telemetry ([`index.html`](src/dashboard/static/index.html)).
- [x] Exclusively routed all Groq requests to `qwen/qwen3.8-27b` in [`src/agent/llm.py`](src/agent/llm.py), with fallback to Gemini strictly prohibited when `LLM_PROVIDER=groq`.
- [x] Implemented Intent Classification / Triage Node:
  - Added `TRIAGE_SYSTEM_PROMPT` and `DIRECT_GENERATE_SYSTEM_PROMPT` in [`src/agent/prompts.py`](src/agent/prompts.py).
  - Implemented `triage_node` and `direct_generate_node` in [`src/agent/nodes.py`](src/agent/nodes.py).
  - Extended `RAGState` with `intent` in [`src/agent/state.py`](src/agent/state.py).
  - Wired conditional entry routing (`decide_intent_route`) in [`src/agent/graph.py`](src/agent/graph.py).
  - Integrated `00 // TRIAGE` circuit node and bypass animations into [`server.py`](src/dashboard/server.py) and [`static/`](src/dashboard/static/).
- [x] Conducted comprehensive codebase audit:
  - Fixed missing `logger` definition in [`src/storage/vector_store.py`](src/storage/vector_store.py) (preventing potential `NameError`).
  - Patched path traversal risks in `/api/ingest` and `/api/upload` via `Path(filename).name` in [`src/dashboard/server.py`](src/dashboard/server.py).
  - Hardened frontend chunk drawer against null/undefined chunk payloads in [`src/dashboard/static/app.js`](src/dashboard/static/app.js).
- [x] Authored production-grade technical [`README.md`](README.md) documenting system architecture, setup, environment configs, telemetry API reference, and Windows/CUDA troubleshooting.
- [x] Renamed project to **ANRI** across `pyproject.toml`, `README.md`, FastAPI backend title, and observability console UI branding.
- [x] Conducted full-spectrum security and architectural audit across ingestion, vector storage, agent reasoning state machine, REST API, and frontend telemetry:
  - Documented 16 vulnerabilities and flaws (including critical silent collection drop DATA-01, insecure CORS SEC-01, arbitrary upload SEC-02, event loop freeze BUG-01, silent mock degradation SEC-07/SEC-10, indirect prompt injection SEC-09, and DOM XSS SEC-05).
  - Authored comprehensive audit artifact report in `audit_report.md` with root causes, user/expert perspectives, and production remediations.
- [x] Remediated Key Audit Findings per User Directives:
  - **SEC-08 (API Key Exposure Patched):** Converted all Google Gemini API calls in [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py), [`src/ingestion/vision.py`](src/ingestion/vision.py), and [`src/agent/llm.py`](src/agent/llm.py) to use HTTP `x-goog-api-key` headers, removing credentials from URL query parameters.
  - **SEC-07 (Silent Mock Degradation Eliminated):** Added `ALLOW_MOCK_FALLBACK: bool = False` in [`src/config.py`](src/config.py) and hardened [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py) to raise explicit, actionable exceptions instead of silently poisoning Qdrant with random mock vectors on API or PyTorch failure.
  - **ARCH-01 (LangGraph State Machine Stream Unified):** Replaced duplicate manual while-loop in [`src/dashboard/server.py`](src/dashboard/server.py) with native `graph.stream(..., stream_mode="updates")` driven directly by [`src/agent/graph.py`](src/agent/graph.py), preserving real-time circuit telemetry while eliminating code drift and infinite loop risks.
  - **BUG-04 (Page-Preserving & Context-Bounded Fallback Parser):** Implemented page-by-page traversal for PDF fallback parsing in [`src/ingestion/parser.py`](src/ingestion/parser.py) (`page_numbers: [page_no]`) and added `split_text_into_sliding_chunks()` strictly capping all chunks to 1,200 characters (~250-300 tokens) with 150-char overlap, guaranteeing chunks never exceed the 512-token context limit of `BAAI/bge-small-en-v1.5`.
- [x] Authored comprehensive grounded specification report in `system_capabilities_report.md` covering supported file types (PDF, DOCX, TXT, MD, Images), size and length bounds, 2-stage retrieval, generation citations, and visual diagram retrieval.
- [x] Hardened Reranker Subsystem against Windows Paging File Exhaustion (`os error 1455`):
  - Root Cause: PyTorch safetensors `mmap` failed on 1.1 GB `BAAI/bge-reranker-base` due to Windows commit limit exhaustion, bypassing `except MemoryError:`.
  - Switched default reranker provider to **FlashRank ONNX** (`ms-marco-MiniLM-L-12-v2`, ~80MB ONNX runtime) across [`.env`](.env), [`.env.example`](.env.example), and [`src/config.py`](src/config.py), completely eliminating PyTorch paging file dependencies and delivering ~15ms reranking.
  - Enhanced [`src/agent/reranker.py`](src/agent/reranker.py) to catch `(MemoryError, OSError)`, auto-recovering to FlashRank ONNX or lightweight `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Updated [`src/dashboard/server.py`](src/dashboard/server.py) and [`src/dashboard/static/app.js`](src/dashboard/static/app.js) to dynamically surface active reranker engine and `[ONNX]` telemetry badges.
- [x] Added end-to-end dashboard answer token streaming: Groq/Gemini streaming responses feed LangGraph custom events, the new `/api/query/stream` SSE route forwards tokens, and the dashboard renders answer text incrementally before applying citation formatting at completion. Existing `/api/query` JSON behavior remains unchanged.
- [x] Verified dashboard token events in the browser (143 incremental answer updates) and paced rendering at one animation frame per event so rapid provider bursts remain visibly progressive.
- [ ] **NEXT MILESTONE OPTIONS:**
  1. **Restart Uvicorn Server:** Run `uv run --no-sync main.py` or `.venv\Scripts\python main.py` to launch with FlashRank ONNX active.
  2. **Implement XML Boundary Delimitation for Indirect Prompt Injection Defense (SEC-09).**
  3. **Execute Remaining Security Fixes:** Patch DATA-01 (prevent collection wipe), SEC-01 (CORS whitelist), SEC-02 (file validation), and BUG-01 (non-blocking upload).
  4. **Hybrid Retrieval (Dense + BM25 with Reciprocal Rank Fusion - RRF):** PRD §4 Step 3 parallel sparse + dense retrieval before cross-encoder reranking.



