import glob
import json
import re
from pathlib import Path

DOMAIN_TO_FILE = {
    "parallel_work_queue": "parallel_work_queue.txt",
    "pods_autoscale": "pods_autoscale.html",
    "job_management": "job_management.html",
    "cronjobs": "cronjobs.docx",
    "monitor_job": "monitor_job.docx",
}


def norm(s: str) -> str:
    """Lowercase, unify quotes, drop all whitespace: so layout differences don't matter."""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", "", s.lower())


def main():
    data = json.loads(Path("evals/golden_dataset.json").read_text(encoding="utf-8"))
    chunks = []
    for f in glob.glob("processed_data/*.json"):
        chunks += [c for c in json.loads(Path(f).read_text(encoding="utf-8")) if c["type"] == "true"]
    normed = [(c["source"], c["chunk_index"], norm(c["text"])) for c in chunks]

    labels, unmatched = [], 0
    for s in data["rag_samples"]:
        target = DOMAIN_TO_FILE[s["domain"]]
        gold = set()
        for ctx in s["relevant_contexts"]:
            snippet = norm(ctx)[:80]          # first 80 non-space chars of the reference passage
            hits = [(src, idx) for src, idx, t in normed if src == target and snippet in t]
            if not hits:
                unmatched += 1
                print(f"  UNMATCHED  Q{s['id']}: {ctx[:70]!r}")
            gold.update(hits)
        labels.append({"id": s["id"], "question": s["question"], "file": target,
                       "gold_chunks": sorted(f"{src}#{idx}" for src, idx in gold)})

    for l in labels:
        print(f"Q{l['id']:>2}  {l['gold_chunks']}  {l['question'][:55]}")
    print(f"\nquestions with no labelled chunk: {sum(1 for l in labels if not l['gold_chunks'])}   unmatched passages: {unmatched}")

    Path("evals/passage_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print("Saved: evals/passage_labels.json  (REVIEW IT BY HAND)")


if __name__ == "__main__":
    main()