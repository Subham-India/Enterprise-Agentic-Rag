from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import hybrid_search, vector_search


def retriever_node(state: AgentState) -> dict:
    query = state.get("search_query") or state["question"]
    mode = settings.RETRIEVAL_MODE

    if mode == "hybrid":
        # Dense + BM25 fused by RRF; the fused order is used directly (no reranker)
        documents = [{**c, "rerank_score": None} for c in hybrid_search(query, top_k=5)]
    elif mode == "hybrid_rerank":
        documents = rerank(query, hybrid_search(query, top_k=20), top_n=5)
    else:
        documents = rerank(query, vector_search(query, top_k=20), top_n=5)   # previous behaviour

    return {"documents": documents}