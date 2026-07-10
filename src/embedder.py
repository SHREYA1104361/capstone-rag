"""Returns the configured embedder (local or OpenAI)."""
from __future__ import annotations

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, USE_LOCAL_EMBEDDINGS


def get_embedder():
    if USE_LOCAL_EMBEDDINGS:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
