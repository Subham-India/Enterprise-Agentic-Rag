import json
import logging
import time
from pathlib import Path

from app.agents.nodes.planner import planner_node
from app.services.retrieval.ranking_service import rerank
from app.services.retrieval.search import vector_search
from evals.run_ablation import cid, score

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)


def retrieve(query: str) -> list[str]:
    """Exactly what the app does: vector search, then rerank to the top 5."""
    return [cid(c) for c in rerank(query, vector_search(query, top_k=20), top_n=5)]


def main():
    labels = [l for l in json.loads(Path("evals/passage_labels.json").read_text(encoding="utf-8"))
              if l["gold_chunks"]]
    tot = {"original": [0, 0.0, 0.0], "rewritten": [0, 0.0, 0.0]}

    print(f"{'id':>3}  {'orig recall':>11}  {'rewrite recall':>14}  rewritten query")
    for l in labels:
        q, gold = l["question"], set(l["gold_chunks"])
        plan = planner_node({"question": q, "messages": []})     # first turn: empty history
        rewritten = plan.get("search_query") if plan.get("route") == "search" else q
        rewritten = rewritten or q

        res = {"original": score(retrieve(q), gold), "rewritten": score(retrieve(rewritten), gold)}
        for name, (hit, recall, rr) in res.items():
            tot[name][0] += hit
            tot[name][1] += recall
            tot[name][2] += rr
        flag = "  <-- worse" if res["rewritten"][1] < res["original"][1] else ""
        print(f"{l['id']:>3}  {res['original'][1]:>11.2f}  {res['rewritten'][1]:>14.2f}  {rewritten[:60]}{flag}")
        time.sleep(0.5)

    n = len(labels)
    print(f"\n{'query used':<12} {'hit':>5} {'recall':>7} {'MRR':>6}")
    for name, (h, r, m) in tot.items():
        print(f"{name:<12} {h / n:>5.2f} {r / n:>7.2f} {m / n:>6.2f}")


if __name__ == "__main__":
    main()
    