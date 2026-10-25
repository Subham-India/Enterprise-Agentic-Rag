import logging
import time

from qdrant_client import QdrantClient, models

from app.config import settings

logging.getLogger("httpx").setLevel(logging.WARNING)

BATCH = 64


def bm25_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding("Qdrant/bm25")


def encode_docs(model, texts: list[str]) -> list[models.SparseVector]:
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in model.embed(texts)]


def build(client: QdrantClient, encode, src: str, dst: str) -> int:
    """Copy every point of `src` into a new hybrid collection `dst`, adding a BM25 sparse vector.
    The existing dense vectors are reused, so no embedding API calls are made."""
    first, _ = client.scroll(src, limit=1, with_vectors=True)
    dim = len(first[0].vector)

    if client.collection_exists(dst):
        client.delete_collection(dst)
    client.create_collection(
        collection_name=dst,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    copied, offset = 0, None
    while True:
        points, offset = client.scroll(src, limit=BATCH, offset=offset, with_payload=True, with_vectors=True)
        if not points:
            break
        sparse = encode([p.payload["text"] for p in points])
        client.upsert(dst, points=[
            models.PointStruct(id=p.id, vector={"dense": p.vector, "bm25": sp}, payload=p.payload)
            for p, sp in zip(points, sparse)
        ])
        copied += len(points)
        if offset is None:
            break
    return copied


def main():
    from app.services.retrieval.qdrant_service import get_client
    client = get_client()
    model = bm25_model()
    t = time.time()
    n = build(client, lambda texts: encode_docs(model, texts), settings.QDRANT_COLLECTION, settings.QDRANT_HYBRID_COLLECTION)
    print(f"Copied {n} points into '{settings.QDRANT_HYBRID_COLLECTION}' in {time.time() - t:.1f}s")
    print("Points now in hybrid collection:", client.count(settings.QDRANT_HYBRID_COLLECTION, exact=True).count)
    print("Points in original collection:  ", client.count(settings.QDRANT_COLLECTION, exact=True).count)


if __name__ == "__main__":
    main()