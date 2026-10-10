from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings


def get_llm(feature: str = "rag") -> ChatOpenAI:
    """An LLM that routes through Portkey.

    The fallback and retry rules live in a saved Portkey config
    (referenced by its pc-... ID), not in this code.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.PRIMARY_MODEL}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_ID,
            metadata={"feature": feature},
        ),
    )