from app.agents.state import AgentState
from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import vector_search


def retriever_node(state: AgentState) -> dict:
    query = state.get("search_query") or state["question"]
    candidates = vector_search(query, top_k=20)   # fast, rough
    documents = rerank(query, candidates, top_n=5)  # slower, accurate
    return {"documents": documents}