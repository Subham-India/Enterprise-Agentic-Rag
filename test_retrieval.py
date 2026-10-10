import sys

from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import vector_search

query = " ".join(sys.argv[1:]) or "How do I start Redis for a work queue?"

candidates = vector_search(query, top_k=20)

print(f"QUERY: {query}\n")
print("--- BEFORE: vector search top 5 ---")
for r in candidates[:5]:
    print(f"{r['score']:.3f}  {r['type']:5} | {r['source'][:40]} | chunk {r['chunk_index']}")

print("\n--- AFTER: reranked top 5 ---")
for r in rerank(query, candidates, top_n=5):
    print(f"{r['rerank_score']:.3f}  {r['type']:5} | {r['source'][:40]} | chunk {r['chunk_index']}")
    print(f"        {r['text'][:100]!r}")