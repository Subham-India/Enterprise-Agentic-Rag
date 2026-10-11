from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings


def get_llm(feature: str = "rag", model: str | None = None, slug: str | None = None) -> ChatOpenAI:
    """An LLM that routes through Portkey.

    Default: the saved config (primary model with fallback).
    If `model` is given: call that one model directly, without the saved config.
    """
    header_args = {"api_key": settings.PORTKEY_API_KEY, "metadata": {"feature": feature}}

    if model:
        model_name = f"@{slug or settings.GROQ_SLUG_2}/{model}"
    else:
        model_name = f"@{settings.GROQ_SLUG}/{settings.PRIMARY_MODEL}"
        header_args["config"] = settings.PORTKEY_CONFIG_ID

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model_name,
        temperature=0,
        default_headers=createHeaders(**header_args),
    )