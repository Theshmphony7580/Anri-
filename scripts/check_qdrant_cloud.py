"""Read-only connectivity check for ANRI's configured Qdrant Cloud cluster."""

import sys
from pathlib import Path

from qdrant_client import QdrantClient

# Running a file under scripts/ puts that folder, not src/, on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import get_settings


def main() -> int:
    settings = get_settings()
    url = (settings.QDRANT_URL or "").strip()
    if not url:
        print("QDRANT_URL is empty. Set the Cloud cluster URL in .env first.")
        return 2
    if not settings.QDRANT_API_KEY:
        print("QDRANT_API_KEY is empty. Set the Cloud API key in .env first.")
        return 2

    client = None
    try:
        client = QdrantClient(url=url, api_key=settings.QDRANT_API_KEY, timeout=15)
        collections = client.get_collections().collections
        names = sorted(collection.name for collection in collections)
        print("Connected to Qdrant successfully.")
        print(f"Configured collection: {settings.QDRANT_COLLECTION_NAME}")
        print(f"Collection exists: {settings.QDRANT_COLLECTION_NAME in names}")
        if names:
            print("Collections visible to this key: " + ", ".join(names))
        else:
            print("No collections exist yet; ANRI will create one when it initializes.")
        return 0
    except Exception as exc:
        # Avoid printing client configuration or credential-bearing request details.
        print(f"Could not connect to Qdrant: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
