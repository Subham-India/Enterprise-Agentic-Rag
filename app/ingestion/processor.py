import argparse
import json
import re
import time
from pathlib import Path

from app.ingestion.chunking.splitter import split_text
from app.ingestion.loaders import load_file
from app.services.retrieval.embedding import embed_texts
from app.services.retrieval.qdrant_service import count_points, setup_collection, upsert_chunks

BATCH_SIZE = 25      # chunks per embedding call
PAUSE_SECONDS = 18   # 25 chunks / 18s is about 83 requests/min, under the 100/min free limit


def embed_with_retry(texts: list[str], max_tries: int = 6) -> list[list[float]]:
    """Embed texts; if Gemini says 429, wait as long as it asks and try again."""
    for attempt in range(1, max_tries + 1):
        try:
            return embed_texts(texts)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = float(m.group(1)) + 5 if m else 65
                print(f"   rate limited, waiting {wait:.0f}s (try {attempt}/{max_tries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gave up after repeated rate limits")


def main(data_dir: str, wipe: bool, limit: int | None, max_chunks: int | None):
    root = Path(data_dir)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if limit:
        files = files[:limit]

    dim = len(embed_with_retry(["dimension probe"])[0])
    setup_collection(dim, wipe=wipe)
    print(f"Embedding dimension: {dim}")

    out_dir = Path("processed_data")
    out_dir.mkdir(exist_ok=True)

    ok, skipped, failed = 0, 0, 0
    for path in files:
        try:
            text = load_file(path)
            if not text or not text.strip():
                print(f"SKIP (no text): {path.name}")
                skipped += 1
                continue

            kind = "true" if "true_data" in path.parts else "noisy"
            texts = split_text(text)
            if max_chunks:
                texts = texts[:max_chunks]
            chunks = [
                {"source": path.name, "type": kind, "chunk_index": i, "text": t}
                for i, t in enumerate(texts)
            ]
            if not chunks:
                print(f"SKIP (no chunks): {path.name}")
                skipped += 1
                continue

            (out_dir / f"{path.name}.json").write_text(
                json.dumps(chunks, indent=2), encoding="utf-8"
            )

            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                upsert_chunks(batch, embed_with_retry([c["text"] for c in batch]))
                time.sleep(PAUSE_SECONDS)

            print(f"OK   {path.name}: {len(chunks)} chunks ({kind})")
            ok += 1
        except Exception as e:
            print(f"FAIL {path.name}: {e}")
            failed += 1

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    print(f"Total points in Qdrant: {count_points()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--wipe", action="store_true", help="drop and recreate the collection")
    parser.add_argument("--limit", type=int, help="only process the first N files")
    parser.add_argument("--max-chunks", type=int, help="cap chunks per file (useful for noisy data)")
    args = parser.parse_args()
    main(args.data_dir, args.wipe, args.limit, args.max_chunks)