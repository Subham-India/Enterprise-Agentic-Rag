import json
import time
from datetime import datetime
from pathlib import Path

from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import vector_search

# The golden dataset labels each question with a "domain": the document that should answer it
DOMAIN_TO_FILE = {
    "parallel_work_queue": "parallel_work_queue.txt",
    "pods_autoscale": "pods_autoscale.html",
    "job_management": "job_management.html",
    "cronjobs": "cronjobs.docx",
    "monitor_job": "monitor_job.docx",
}


def first_rank(sources: list[str], target: str) -> int | None:
    """1-based position of the first chunk from the target file, or None."""
    for i, s in enumerate(sources, start=1):
        if s == target:
            return i
    return None


def main():
    data = json.loads(Path("evals/golden_dataset.json").read_text(encoding="utf-8"))
    rows = []

    for s in data["rag_samples"]:
        question = s["question"]
        target = DOMAIN_TO_FILE[s["domain"]]

        candidates = vector_search(question, top_k=20)   # stage 1: vector search
        reranked = rerank(question, candidates, top_n=5)  # stage 2: rerank

        vec_rank = first_rank([c["source"] for c in candidates], target)
        rr_rank = first_rank([c["source"] for c in reranked], target)

        rows.append({
            "id": s["id"],
            "question": question,
            "expected_file": target,
            "vector_rank": vec_rank,                       # None = not in top 20
            "rerank_rank": rr_rank,                        # None = not in final top 5
            "final_count": len(reranked),
            "noisy_in_final": sum(1 for c in reranked if c["type"] == "noisy"),
            "final_sources": [f"{c['source']}#{c['chunk_index']}" for c in reranked],
        })
        time.sleep(0.5)  # gentle on the embedding quota

    n = len(rows)
    summary = {
        "questions": n,
        "vector_hit@20": sum(r["vector_rank"] is not None for r in rows) / n,
        "rerank_hit@5": sum(r["rerank_rank"] is not None for r in rows) / n,
        "rerank_hit@1": sum(r["rerank_rank"] == 1 for r in rows) / n,
        "rerank_MRR": sum((1 / r["rerank_rank"]) if r["rerank_rank"] else 0 for r in rows) / n,
        "avg_noisy_chunks_in_final": sum(r["noisy_in_final"] for r in rows) / n,
        "avg_chunks_returned": sum(r["final_count"] for r in rows) / n,
    }

    # Per-question table
    print(f"\n{'id':>3}  {'vec':>4}  {'rerank':>6}  {'noisy':>5}  question")
    for r in rows:
        print(f"{r['id']:>3}  {str(r['vector_rank'] or '-'):>4}  {str(r['rerank_rank'] or '-'):>6}  "
              f"{r['noisy_in_final']:>5}  {r['question'][:60]}")

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()