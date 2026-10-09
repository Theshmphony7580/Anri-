# Run ANRI Models on a Local Computer

ANRI can keep its embedding and reranking models on a separate computer while the
dashboard/API runs elsewhere. The local computer runs a small authenticated HTTP
service; ANRI calls it through an HTTPS tunnel. Qdrant and answer generation remain
on their existing providers.

```text
Browser → ANRI API → Qdrant Cloud
                   → HTTPS tunnel → local embedding/reranking service
```

## Requirements

- The local computer must have the configured embedding and reranker model weights
  and enough CPU/GPU memory to load them.
- Keep the local and deployed ANRI embedding model, provider, and vector dimension
  identical. Changing the embedding model requires re-embedding the corpus.
- Keep the local computer awake and the model service and tunnel running while ANRI
  needs inference. If the computer or tunnel is offline, embedding/reranking calls
  fail; ANRI does not silently switch to a different embedding space.
- The service accepts at most 32 texts per embedding request and 64 candidates per
  reranking request.

## Run locally

In the local computer's `.env`, configure the model used by the indexed Qdrant
collection, set `MODEL_SERVICE_API_KEY` to a long random token, and leave
`MODEL_SERVICE_URL` empty. For a local BGE setup, the relevant settings look like:

```dotenv
EMBEDDING_PROVIDER=huggingface
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
RERANKER_PROVIDER=flashrank
RERANKER_MODEL=ms-marco-MiniLM-L-12-v2
MODEL_SERVICE_URL=
MODEL_SERVICE_API_KEY=<same-long-random-token-used-by-ANRI>
```

Start the service from the repository root:

```powershell
uv run --no-sync uvicorn model_service:app --app-dir src --host 127.0.0.1 --port 8001
```

The unauthenticated `/health` route returns only `{"status":"ok"}`. Inference
routes require `Authorization: Bearer <token>`:

- `POST /v1/embeddings` with `{"texts":["..."],"is_query":false}`
- `POST /v1/rerank` with `{"query":"...","documents":["..."],"top_n":4}`

## Expose the service through a tunnel

Use an HTTPS tunnel that routes a hostname to `http://localhost:8001`. The service
binds to loopback; do not bind it to a public interface or open an inbound router
port. Keep the bearer-token check enabled even when the tunnel provides HTTPS. For
a stable deployment, use a named tunnel/hostname; a temporary tunnel URL changes
when restarted.

## Configure the deployed ANRI service

Set these environment variables on the ANRI host (for example, Render):

```dotenv
MODEL_SERVICE_URL=https://<your-tunnel-hostname>
MODEL_SERVICE_API_KEY=<same-long-random-token>
MODEL_SERVICE_TIMEOUT_SECONDS=120
```

Also set the same embedding provider, model ID, and `EMBEDDING_DIM` as the local
service and the existing Qdrant collection. `MODEL_SERVICE_URL` makes ANRI route
both query/document embeddings and reranking to the local service. Leave the URL
empty on the computer running the inference service to avoid a proxy loop.

Do not commit the token or place it in frontend configuration. Rotate it in both
environments if it is exposed.
