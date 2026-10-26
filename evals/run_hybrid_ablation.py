import json
import logging
import time
from datetime import datetime
from pathlib import Path

from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import hybrid_search, vector_search
from evals.run_ablation import cid, first_rank, score

logging.getLogger("httpx").setLevel(logging.WARNING)

CONFIGS = ["dense_top5", "dense_rerank_top5 (current)", "hybrid_top5", "hybrid_rerank_top5", "hybrid_rerank_top10"]


def main():
    labels = [l for l in json.loads(Path("evals/passage_labels.json").read_text(encoding="utf-8"))
              if l["gold_chunks"]]
    totals = {c: {"hit": 0, "recall": 0.0, "rr": 0.0} for c in CONFIGS}
    off_file = {c: 0 for c in CONFIGS}   # chunks that come from a file other than the expected one
    sizes = {c: 0 for c in CONFIGS}      # how many chunks each setup returns
    pool = {"dense": 0.0, "hybrid": 0.0}
    rows = []

    for l in labels:
        q, gold = l["question"], set(l["gold_chunks"])
        dense = vector_search(q, top_k=20)
        hyb = hybrid_search(q, top_k=20)
        dense_ids, hyb_ids = [cid(c) for c in dense], [cid(c) for c in hyb]

        full_h = rerank(q, hyb, top_n=20, min_score=-1)
        variants = {
            "dense_top5": dense_ids[:5],
            "dense_rerank_top5 (current)": [cid(c) for c in rerank(q, dense, top_n=5)],
            "hybrid_top5": hyb_ids[:5],
            "hybrid_rerank_top5": [cid(c) for c in rerank(q, hyb, top_n=5)],
            "hybrid_rerank_top10": [cid(c) for c in full_h[:10]],
        }
        for name, ids in variants.items():
            hit, recall, rr = score(ids, gold)
            totals[name]["hit"] += hit
            totals[name]["recall"] += recall
            totals[name]["rr"] += rr
            off_file[name] += sum(1 for i in ids if not i.startswith(l["file"] + "#"))
            sizes[name] += len(ids)
        pool["dense"] += len(set(dense_ids) & gold) / len(gold)
        pool["hybrid"] += len(set(hyb_ids) & gold) / len(gold)

        rows.append({"id": l["id"], "gold": sorted(gold),
                     "gold_rank_dense": first_rank(dense_ids, gold),
                     "gold_rank_hybrid": first_rank(hyb_ids, gold),
                     "gold_rank_hybrid_reranked": first_rank([cid(c) for c in full_h], gold)})
        time.sleep(0.5)

    n = len(labels)
    print(f"\nQUESTIONS WITH LABELLED PASSAGES: {n}")
    print(f"\n{'id':>3}  {'dense':>5}  {'hybrid':>6}  {'hybrid+rerank':>13}   (rank of the first gold chunk in the 20 candidates)")
    for r in rows:
        f = lambda x: str(x or "-")
        print(f"{r['id']:>3}  {f(r['gold_rank_dense']):>5}  {f(r['gold_rank_hybrid']):>6}  {f(r['gold_rank_hybrid_reranked']):>13}")

    print(f"\nCandidate pool recall@20:  dense {pool['dense'] / n:.2f}   hybrid {pool['hybrid'] / n:.2f}")
    print(f"\n{'config':<30} {'hit':>5} {'recall':>7} {'MRR':>6} {'off-file':>9} {'chunks':>7}")
    summary = {}
    for name, t in totals.items():
        summary[name] = {"hit": t["hit"] / n, "recall": t["recall"] / n, "MRR": t["rr"] / n,
                         "off_file_chunks": off_file[name] / n, "avg_chunks": sizes[name] / n}
        s = summary[name]
        print(f"{name:<30} {s['hit']:>5.2f} {s['recall']:>7.2f} {s['MRR']:>6.2f} "
              f"{s['off_file_chunks']:>9.2f} {s['avg_chunks']:>7.2f}")
    print("\noff-file = average number of chunks per question that come from a different file than the expected one")

    out = Path("evals/results") / f"hybrid_ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "pool_recall": {k: v / n for k, v in pool.items()}, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()