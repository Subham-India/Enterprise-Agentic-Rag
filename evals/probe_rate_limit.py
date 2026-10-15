import sys
import time

import openai
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings

PROMPT = ("Reply with the single word ok.\n\n" + "kubernetes job parallelism completions " * 250)  # about 2,500 tokens


def probe(model: str, slug: str, n: int = 8):
    client = OpenAI(
        api_key=settings.PORTKEY_API_KEY, base_url=PORTKEY_GATEWAY_URL, max_retries=0,
        default_headers=createHeaders(api_key=settings.PORTKEY_API_KEY, metadata={"feature": "probe"}),
    )
    print(f"\n{model} via @{slug}")
    for i in range(1, n + 1):
        t = time.time()
        try:
            client.chat.completions.create(model=f"@{slug}/{model}", messages=[{"role": "user", "content": PROMPT}], max_tokens=50)
            print(f"  {i}: ok in {time.time() - t:.1f}s", flush=True)
        except openai.APIStatusError as e:
            print(f"  {i}: HTTP {e.status_code} after {time.time() - t:.1f}s | retry-after={e.response.headers.get('retry-after')} | {str(e.message)[:160]}", flush=True)
        except Exception as e:
            print(f"  {i}: {type(e).__name__}: {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    if which == "qwen":
        probe("qwen/qwen3.8-27b", settings.GROQ_SLUG_2)
    else:
        probe(settings.PRIMARY_MODEL, settings.GROQ_SLUG)