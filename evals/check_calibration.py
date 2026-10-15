import csv
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("evals/results").glob("judge_calibration_*.csv"))[-1]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"Sheet: {path.name}")
    for metric in ("faithfulness", "correctness"):
        pairs = []
        for r in rows:
            j, h = r[f"judge_{metric}"], r[f"human_{metric}"]
            if j.strip() and h.strip():
                pairs.append((int(float(j)), int(float(h))))
        if not pairs:
            print(f"  {metric}: no human grades filled in yet")
            continue
        n = len(pairs)
        exact = sum(j == h for j, h in pairs) / n
        within1 = sum(abs(j - h) <= 1 for j, h in pairs) / n
        bias = sum(j - h for j, h in pairs) / n
        print(f"  {metric}: n={n}  exact agreement={exact:.2f}  within +-1={within1:.2f}  "
              f"judge minus human (mean)={bias:+.2f}")
    print("\nGuide: within +-1 of 0.85 or more is decent. A positive bias means the judge is too generous.")


if __name__ == "__main__":
    main()