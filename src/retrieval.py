"""MongoDB Atlas $vectorSearch retrieval with LangSmith tracing."""
from __future__ import annotations

from langsmith import traceable
from pymongo import MongoClient

from src.config import MONGODB_URI, MONGODB_DB, RAG_CHUNKS_COLLECTION, VECTOR_INDEX_NAME, TOP_K
from src.embedder import get_embedder

_embedder = get_embedder()


@traceable(name="retrieve", run_type="retriever")
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed *query* and return top-k chunks with listing_id, name, text, score."""
    vector = _embedder.embed_query(query)
    client = MongoClient(MONGODB_URI)
    try:
        col = client[MONGODB_DB][RAG_CHUNKS_COLLECTION]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": k * 10,
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "listing_id": 1,
                    "name": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(col.aggregate(pipeline))
    finally:
        client.close()
