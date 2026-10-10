import re

from app.config import settings


def split_text(text: str, max_chars: int = settings.CHUNK_MAX_CHARS) -> list[str]:
    # Collapse runs of blank lines so paragraph splitting is predictable
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks, current = [], ""
    for p in paragraphs:
        # A single paragraph longer than the limit gets hard-split
        while len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(p[:max_chars])
            p = p[max_chars:]

        if len(current) + len(p) + 2 <= max_chars:
            current = f"{current}\n\n{p}" if current else p
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    # Drop tiny fragments (page numbers, stray headings)
    return [c for c in chunks if len(c) > 50]