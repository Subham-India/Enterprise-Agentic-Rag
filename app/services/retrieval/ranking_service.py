from flashrank import Ranker, RerankRequest

_ranker = None


def get_ranker() -> Ranker:
    # Created on first use; the model downloads once into ./flashrank_cache
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="flashrank_cache")
    return _ranker


def rerank(query: str, chunks: list[dict], top_n: int = 5, min_score: float = 0.05) -> list[dict]:
    """Re-score chunks against the query with a cross-encoder.
    Returns up to top_n chunks, dropping ones scoring below min_score
    (the best chunk is always kept)."""
    if not chunks:
        return []
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
    results = get_ranker().rerank(RerankRequest(query=query, passages=passages))
    ranked = []
    for r in results[:top_n]:
        score = float(r["score"])
        if ranked and score < min_score:
            break
        chunk = dict(chunks[r["id"]])
        chunk["vector_score"] = chunk.pop("score", None)
        chunk["rerank_score"] = score
        ranked.append(chunk)
    return ranked