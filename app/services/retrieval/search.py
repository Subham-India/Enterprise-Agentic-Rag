from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.qdrant_service import get_client


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """Embed the query and return the top_k most similar chunks."""
    vector = embed_query(query)
    result = get_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in result.points]