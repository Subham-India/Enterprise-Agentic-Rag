import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from app.agents.graph import ask
from app.gateway.client import get_llm

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

# Different model family from the answerer (gpt-oss), to reduce "judge likes its own style" bias
JUDGE_MODEL = "qwen/qwen3.8-27b"

FAITHFULNESS_PROMPT = """You are a strict grader. Judge ONLY whether the ANSWER is supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Scoring (1-5), based only on the claims the answer makes:
5 = every claim is supported by the context
4 = supported, with one minor detail not in the context
3 = some claims are not in the context
2 = most claims are not in the context
1 = fabricated or contradicts the context
An answer that only says the information could not be found makes no claims: score 5.

Reply with JSON only, no other text:
{{"score": <1-5>, "unsupported_claims": ["..."], "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are a strict grader. Compare the ANSWER with the REFERENCE answer for the QUESTION.

QUESTION: {question}

REFERENCE:
{reference}

ANSWER:
{answer}

Scoring (1-5):
5 = same key facts, commands and values as the reference; nothing important missing
4 = correct, but missing a minor detail
3 = partly correct; an important fact is missing or vague
2 = mostly wrong or off-topic
1 = wrong, or it says the information could not be found
Wording may differ. Judge facts and commands, not style.

Reply with JSON only, no other text:
{{"score": <1-5>, "missing": ["..."], "reason": "<one sentence>"}}"""

SUFFICIENCY_PROMPT = """You are a strict grader. Does the CONTEXT contain the information needed to write the REFERENCE answer?

CONTEXT:
{context}

REFERENCE:
{reference}

Reply "yes" if all key facts are in the context, "partial" if some are, "no" if none are.
Reply with JSON only, no other text:
{{"verdict": "yes|partial|no", "reason": "<one sentence>"}}"""


def build_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({d['source']})\n{d['text']}" for i, d in enumerate(docs, 1))


def clean_answer(answer: str) -> str:
    return re.split(r"\n\s*Sources:", answer)[0].strip()   # the Sources line is added by the app, not the model


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)   # drop any reasoning block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in judge reply")
    return json.loads(m.group(0))


def judge(llm, prompt: str, tries: int = 2) -> dict:
    last = None
    for _ in range(tries):
        try:
            return parse_json(llm.invoke(prompt).content)
        except Exception as e:
            last = e
            time.sleep(1)
    return {"error": str(last)[:120]}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    data = json.loads(Path("evals/golden_dataset.json").read_text(encoding="utf-8"))
    llm = get_llm("judge", model=JUDGE_MODEL)
    rows = []

    for s in data["rag_samples"]:
        q, ref = s["question"], s["reference"]
        try:
            state = ask(q, f"judge-{s['id']}-{datetime.now():%H%M%S}")
            docs = state.get("documents") or []
            answer = clean_answer(state["answer"])
            ctx = build_context(docs)
        except Exception as e:
            rows.append({"id": s["id"], "question": q, "error": f"pipeline: {str(e)[:100]}"})
            print(f"  {s['id']:>2} PIPELINE ERROR")
            continue

        f = judge(llm, FAITHFULNESS_PROMPT.format(context=ctx, answer=answer))
        c = judge(llm, CORRECTNESS_PROMPT.format(question=q, reference=ref, answer=answer))
        r = judge(llm, SUFFICIENCY_PROMPT.format(context=ctx, reference=ref))

        row = {"id": s["id"], "question": q, "answer": answer, "reference": ref,
               "retrieved": [f"{d['source']}#{d['chunk_index']}" for d in docs],
               "faithfulness": f.get("score"), "faithfulness_detail": f,
               "correctness": c.get("score"), "correctness_detail": c,
               "context_sufficient": r.get("verdict"), "sufficiency_detail": r, "error": None}
        # Failure attribution: where did a wrong answer come from?
        if (row["correctness"] or 5) <= 3:
            if row["context_sufficient"] in ("no", "partial"):
                row["failure_type"] = "retrieval (context lacked the answer)"
            elif (row["faithfulness"] or 5) <= 3:
                row["failure_type"] = "generation (unsupported claims)"
            else:
                row["failure_type"] = "generation (had the context, answered wrongly or abstained)"
        rows.append(row)
        print(f"  {s['id']:>2}  faith={row['faithfulness']}  correct={row['correctness']}  "
              f"context={row['context_sufficient']}  {q[:45]}", flush=True)
        time.sleep(1)

    ok = [r for r in rows if not r.get("error")]
    summary = {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "judge_parse_failures": sum(1 for r in ok for k in ("faithfulness", "correctness") if r[k] is None),
        "faithfulness_mean": mean([r["faithfulness"] for r in ok]),
        "correctness_mean": mean([r["correctness"] for r in ok]),
        "faithfulness_ge4": mean([r["faithfulness"] >= 4 for r in ok if r["faithfulness"] is not None]),
        "correctness_ge4": mean([r["correctness"] >= 4 for r in ok if r["correctness"] is not None]),
        "context_sufficient_yes": mean([r["context_sufficient"] == "yes" for r in ok]),
        "judge_model": JUDGE_MODEL,
    }

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    fails = [r for r in ok if r.get("failure_type")]
    if fails:
        print("\nFAILURES (correctness <= 3)")
        for r in fails:
            print(f"  Q{r['id']}: {r['failure_type']}")
            print(f"      judge: {r['correctness_detail'].get('reason', '')}")

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = out_dir / f"judge_{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    # Sheet for hand-grading: fill the two human_* columns, then run evals.check_calibration
    sheet = out_dir / f"judge_calibration_{stamp}.csv"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "reference", "answer", "judge_faithfulness", "judge_correctness",
                    "human_faithfulness", "human_correctness"])
        for r in ok:
            w.writerow([r["id"], r["question"], r["reference"], r["answer"], r["faithfulness"], r["correctness"], "", ""])
    print(f"\nSaved: {out}\nSaved: {sheet}  (fill the human_* columns with 1-5)")


if __name__ == "__main__":
    main()