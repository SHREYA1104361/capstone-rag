from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load from the project root, regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB = os.environ["MONGODB_DB"]
MONGODB_COLLECTION = os.environ["MONGODB_COLLECTION"]          # source: listingsAndReviews
RAG_CHUNKS_COLLECTION = os.getenv("RAG_CHUNKS_COLLECTION", "rag_chunks")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")
TOP_K = int(os.getenv("TOP_K", "5"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "capstone-rag")
