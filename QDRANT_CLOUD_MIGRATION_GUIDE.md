# Move ANRI to Qdrant Cloud

This guide moves this learning project from its local Qdrant database to a new Qdrant Cloud cluster. The recommended simple approach is to re-ingest the original documents into Cloud. ANRI will create its collection and embeddings as part of ingestion.

## What you need

- Access to the Qdrant Cloud account where you can create a cluster.
- The original documents that were ingested into ANRI. Keep them in a known folder; do not delete the local database yet.
- ANRI's existing `.venv` and `.env`.

The local database is in `qdrant_data/` by default. Do not copy that folder to Cloud: it is Qdrant's local storage format, not an import bundle.

## 1. Create a Qdrant Cloud cluster

1. Sign in to the Qdrant Cloud console and create a cluster. A small/free option is enough for learning and small document collections.
2. Wait until the cluster reports that it is ready.
3. In the cluster details, copy its HTTPS cluster URL and create/copy an API key with access to that cluster.
4. Store the URL and key somewhere private temporarily. Treat the key like a password; do not paste it into source files, screenshots, or chat.

## 2. Point ANRI at Cloud

Open the repository's `.env` file (create it from `.env.example` if it does not exist) and set:

```dotenv
QDRANT_URL=https://YOUR-CLUSTER-ENDPOINT
QDRANT_API_KEY=YOUR-CLOUD-API-KEY
QDRANT_COLLECTION_NAME=knowledge_chunks
```

Replace the first two values with the actual cluster URL and key. Keep `QDRANT_PATH` as it is; it is ignored while `QDRANT_URL` is set. Do not put real credentials in `.env.example` or commit `.env`.

ANRI uses its configured embedding model and dimension when it creates the Cloud collection. Keep the embedding provider/model settings the same as the ones used for the local documents. Changing embedding models during this move can make old and new vectors incompatible.

## 3. Start ANRI and confirm it connects

Start ANRI the same way you normally do, from the repository and using its existing environment. For example, if you normally use the project's `uv` setup, use that same startup method. Do not install a second Python environment just for this migration.

Open the dashboard and check the Qdrant collection/vector count. A new Cloud collection will normally be empty at this point. If startup reports an authentication or connection error, recheck the endpoint, API key, and that the cluster is ready. Do not continue until the dashboard/API can reach the Cloud collection.

## 4. Re-ingest the documents

Use the dashboard's document upload/ingestion flow, or the same ingestion flow you used to add the documents originally. Re-ingest the original source files into the Cloud-backed ANRI instance. Keep the app pointed at Cloud for the whole ingestion run.

Wait for each document to finish. Check the dashboard's document/vector totals and confirm that the expected sources appear. If the original files are unavailable, stop here: this project does not currently document a supported local-Qdrant-to-Cloud snapshot restore path. Preserve `qdrant_data/` and ask for a separate export/restore plan rather than deleting it.

## 5. Verify retrieval

Ask a few questions whose answers you know are in the ingested Rust documentation. Confirm that:

- ANRI returns relevant source chunks and citations.
- The Cloud collection's vector count is nonzero and consistent with the completed ingestion.
- Restarting ANRI still shows the same Cloud documents and can answer the same questions.

The restart check matters: it confirms that the data lives in Cloud and was not only available from an old local process. If retrieval is empty or irrelevant, check that the endpoint and collection name are consistent and that the embedding model settings match ingestion. Keep the local copy intact while troubleshooting.

## 6. Keep a simple rollback

To return to local Qdrant, stop ANRI and set `QDRANT_URL=` and `QDRANT_API_KEY=` in `.env`, leaving `QDRANT_PATH` unchanged. Restart ANRI; the local `qdrant_data/` collection should be used again. To switch back to Cloud, restore the Cloud URL and key.

Keep `qdrant_data/` and the original source files until Cloud ingestion, retrieval, and the restart check all pass. After that, you can decide whether to keep the local database as a backup. Never delete it as part of the initial move.

## Completion checklist

- [ ] Cloud cluster is ready; endpoint and key are stored privately.
- [ ] `.env` points ANRI at Cloud and is not committed.
- [ ] ANRI starts and reaches the Cloud collection.
- [ ] Original files have been re-ingested; Cloud has vectors/documents.
- [ ] Known-answer retrieval and a restart check both pass.
- [ ] Local database and original files were retained until verification passed.

## Why this is the efficient learning-project path

ANRI already selects remote Qdrant when `QDRANT_URL` is configured and creates its collection during startup/ingestion. Re-ingesting the source documents uses the existing UI and pipeline, preserves the normal parsing/metadata flow, and avoids adding a migration tool or learning Qdrant snapshot mechanics. Its tradeoff is that ingestion takes time and recomputes embeddings; for this small project, that is the easiest path to understand and recover if anything goes wrong.
