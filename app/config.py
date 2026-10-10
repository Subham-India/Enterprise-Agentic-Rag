import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Embeddings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBED_MODEL = "models/gemini-embedding-001"

    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # LLM gateway
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag1")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "rag2")
    PRIMARY_MODEL = "openai/gpt-oss-20b"
    FALLBACK_MODEL = "qwen/qwen3.8-27b"
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    # Ingestion
    CHUNK_MAX_CHARS = 1500


settings = Settings()