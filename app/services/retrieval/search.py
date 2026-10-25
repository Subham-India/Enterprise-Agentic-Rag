from qdrant_client import models

from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.qdrant_service import get_client

_bm25 = None


def _bm25_model():
    global _bm25
    if _bm25 is None:
        from fastembed import SparseTextEmbedding
        _bm25 = SparseTextEmbedding("Qdrant/bm25")
    return _bm25


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """Dense search only (the current setup, on the original collection)."""
    vector = embed_query(query)
    result = get_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in result.points]


def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    """Dense + BM25 keyword search, fused with reciprocal rank fusion (on the hybrid collection)."""
    dense = embed_query(query)
    sp = next(iter(_bm25_model().query_embed(query)))
    result = get_client().query_points(
        collection_name=settings.QDRANT_HYBRID_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=top_k * 2),
            models.Prefetch(
                query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
                using="bm25", limit=top_k * 2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in result.points]