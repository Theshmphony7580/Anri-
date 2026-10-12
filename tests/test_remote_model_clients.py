"""Offline regression cases for remote inference batching and provider dimensions."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import reranker
from ingestion import embeddings
from schemas import DocumentChunk


class RemoteModelClientTests(unittest.TestCase):
    """Exercise service limits and failures without real models or network calls."""

    def setUp(self):
        """Provide explicit settings without consulting deployment credentials."""
        self.settings = SimpleNamespace(
            MODEL_SERVICE_URL="https://models.example.test",
            MODEL_SERVICE_API_KEY="test-token",
            MODEL_SERVICE_TIMEOUT_SECONDS=1,
            EMBEDDING_PROVIDER="huggingface",
            EMBEDDING_DIM=384,
            GEMINI_EMBEDDING_DIM=768,
            RERANKER_MODEL="test-reranker",
            RETRIEVAL_TOP_K=4,
        )

    def test_rerank_merges_all_batches_and_preserves_large_result_limit(self):
        """The best candidates in later batches must survive a top-65 request."""
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            results = sorted(
                ({"index": i, "score": float(text)} for i, text in enumerate(payload["documents"])),
                key=lambda result: result["score"],
                reverse=True,
            )
            return httpx.Response(200, json={"results": results[:payload["top_n"]]})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        chunks = [DocumentChunk(id=str(i), text=str(i)) for i in range(130)]
        with patch.object(reranker, "get_settings", return_value=self.settings), patch.object(
            reranker.httpx, "Client", return_value=client
        ):
            ranked = reranker.RemoteModelServiceReranker().rerank("query", chunks, top_n=65)

        self.assertEqual([len(request["documents"]) for request in requests], [64, 64, 2])
        self.assertEqual([request["top_n"] for request in requests], [64, 64, 2])
        self.assertEqual([chunk.id for chunk in ranked], [str(i) for i in range(129, 64, -1)])

    def test_rerank_uses_default_limit_across_batches(self):
        """An omitted limit still selects the globally best configured top four."""
        def respond(request):
            payload = json.loads(request.content)
            results = sorted(
                ({"index": i, "score": float(text)} for i, text in enumerate(payload["documents"])),
                key=lambda result: result["score"],
                reverse=True,
            )
            return httpx.Response(200, json={"results": results[:payload["top_n"]]})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        chunks = [DocumentChunk(id=str(i), text=str(i)) for i in range(70)]
        with patch.object(reranker, "get_settings", return_value=self.settings), patch.object(
            reranker.httpx, "Client", return_value=client
        ):
            ranked = reranker.RemoteModelServiceReranker().rerank("query", chunks)
        self.assertEqual([chunk.id for chunk in ranked], ["69", "68", "67", "66"])

    def test_invalid_later_batch_does_not_partially_mutate_chunks(self):
        """A malformed later response leaves every original score untouched."""
        def respond(request):
            payload = json.loads(request.content)
            results = [{"index": i, "score": 1.0} for i in range(payload["top_n"])]
            if payload["documents"][0] == "64":
                results[-1]["index"] = -1
            return httpx.Response(200, json={"results": results})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        chunks = [
            DocumentChunk(id=str(i), text=str(i), metadata={"rerank_score": 0.25})
            for i in range(70)
        ]
        with patch.object(reranker, "get_settings", return_value=self.settings), patch.object(
            reranker.httpx, "Client", return_value=client
        ):
            with self.assertRaises(RuntimeError):
                reranker.RemoteModelServiceReranker().rerank("query", chunks)
        self.assertTrue(all(chunk.metadata == {"rerank_score": 0.25} for chunk in chunks))

    def test_gemini_embeddings_use_provider_dimension_and_preserve_batch_order(self):
        """A 33-text Gemini request accepts 768 dimensions across two batches."""
        self.settings.EMBEDDING_PROVIDER = "Gemini"
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(
                200, json={"embeddings": [[float(text)] * 768 for text in payload["texts"]]}
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch.object(embeddings, "get_settings", return_value=self.settings), patch.object(
            embeddings.httpx, "Client", return_value=client
        ):
            vectors = embeddings.RemoteModelServiceEmbedder().embed_batch(
                [str(i) for i in range(33)], is_query=True
            )
        self.assertEqual([len(request["texts"]) for request in requests], [32, 1])
        self.assertTrue(all(request["is_query"] for request in requests))
        self.assertEqual([vector[0] for vector in vectors], list(range(33)))
        self.assertTrue(all(len(vector) == 768 for vector in vectors))

    def test_gemini_rejects_vectors_with_huggingface_dimension(self):
        """Gemini cannot silently accept a vector from the other provider."""
        self.settings.EMBEDDING_PROVIDER = "gemini"
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"embeddings": [[0.0] * 384]})
            )
        )
        with patch.object(embeddings, "get_settings", return_value=self.settings), patch.object(
            embeddings.httpx, "Client", return_value=client
        ):
            with self.assertRaisesRegex(RuntimeError, "provider dimension.*768"):
                embeddings.RemoteModelServiceEmbedder().embed_text("query", is_query=True)


if __name__ == "__main__":
    unittest.main()
