import json
import logging
import time
from datetime import datetime
from pathlib import Path

from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import vector_search

logging.getLogger("httpx").setLevel(logging.WARNING)

RRF_K = 60  # standard constant for reciprocal rank fusion


def cid(c: dict) -> str:
    return f"{c['source']}#{c['chunk_index']}"


def score(ids: list[str], gold: set[str]) -> tuple[bool, float, float]:
    hit = any(i in gold for i in ids)
    recall = len(set(ids) & gold) / len(gold)
    rr = next((1 / r for r, i in enumerate(ids, 1) if i in gold), 0.0)
    return hit, recall, rr


def first_rank(ids: list[str], gold: set[str]):
    return next((r for r, i in enumerate(ids, 1) if i in gold), None)


def main():
    labels = [l for l in json.loads(Path("evals/passage_labels.json").read_text(encoding="utf-8"))
              if l["gold_chunks"]]
    configs = ["dense_top5", "current_rerank_top5", "rerank_top8", "rerank_top10", "rrf_top5"]
    totals = {c: {"hit": 0, "recall": 0.0, "rr": 0.0} for c in configs}
    rows = []

    for l in labels:
        q, gold = l["question"], set(l["gold_chunks"])
        cands = vector_search(q, top_k=20)
        dense_ids = [cid(c) for c in cands]
        full = rerank(q, cands, top_n=20, min_score=-1)          # every candidate, reranker order
        full_ids = [cid(c) for c in full]
        current_ids = [cid(c) for c in rerank(q, cands, top_n=5)]  # exactly what the app does today

        dense_rank = {i: r for r, i in enumerate(dense_ids, 1)}
        rerank_rank = {i: r for r, i in enumerate(full_ids, 1)}
        fused = sorted(dense_rank, key=lambda i: -(1 / (RRF_K + dense_rank[i]) + 1 / (RRF_K + rerank_rank[i])))

        variants = {
            "dense_top5": dense_ids[:5],
            "current_rerank_top5": current_ids,
            "rerank_top8": full_ids[:8],
            "rerank_top10": full_ids[:10],
            "rrf_top5": fused[:5],
        }
        for name, ids in variants.items():
            hit, recall, rr = score(ids, gold)
            totals[name]["hit"] += hit
            totals[name]["recall"] += recall
            totals[name]["rr"] += rr

        rows.append({"id": l["id"], "gold": sorted(gold),
                     "gold_dense_rank": first_rank(dense_ids, gold),
                     "gold_rerank_rank": first_rank(full_ids, gold),
                     "in_current_top5": any(i in gold for i in current_ids)})
        time.sleep(0.5)

    n = len(labels)
    print(f"\nQUESTIONS WITH LABELLED PASSAGES: {n}")
    print(f"\n{'id':>3}  {'dense rank':>10}  {'rerank rank':>11}  in current top5?")
    for r in rows:
        print(f"{r['id']:>3}  {str(r['gold_dense_rank'] or '-'):>10}  {str(r['gold_rerank_rank'] or '-'):>11}  "
              f"{'yes' if r['in_current_top5'] else 'NO'}")

    print(f"\n{'config':<22} {'hit':>5} {'recall':>7} {'MRR':>6}")
    summary = {}
    for name, t in totals.items():
        summary[name] = {"hit": t["hit"] / n, "recall": t["recall"] / n, "MRR": t["rr"] / n}
        s = summary[name]
        print(f"{name:<22} {s['hit']:>5.2f} {s['recall']:>7.2f} {s['MRR']:>6.2f}")

    out = Path("evals/results") / f"ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()