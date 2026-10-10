from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

_model = None


def get_embedding_model():
    # Created on first use, not at import time
    global _model
    if _model is None:
        _model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBED_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    return get_embedding_model().embed_query(query)