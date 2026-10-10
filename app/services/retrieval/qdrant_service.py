import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

_client = None


def get_client() -> QdrantClient:
    # Created on first use, not at import time
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY,timeout=120)
    return _client


def setup_collection(dim: int, wipe: bool = False):
    client = get_client()
    name = settings.QDRANT_COLLECTION
    if wipe and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]):
    points = [
        PointStruct(
            # Stable ID: re-running ingestion overwrites instead of duplicating
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['source']}::{chunk['chunk_index']}")),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    get_client().upsert(collection_name=settings.QDRANT_COLLECTION, points=points)


def count_points() -> int:
    return get_client().count(collection_name=settings.QDRANT_COLLECTION, exact=True).count