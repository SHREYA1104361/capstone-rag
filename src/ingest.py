"""Chunk documents from listingsAndReviews, embed, and upsert to rag_chunks.

Run:  python -m src.ingest [--limit 200]
Idempotent: existing chunks for a listing_id are deleted before re-inserting.
"""
from __future__ import annotations

import argparse
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

from src.config import MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION, RAG_CHUNKS_COLLECTION
from src.embedder import get_embedder

_TEXT_FIELDS = ("description", "summary", "space", "neighborhood_overview", "transit", "notes")
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
_EMBEDDER = get_embedder()


def _build_text(doc: dict) -> str:
    parts = [str(doc.get(f, "") or "").strip() for f in _TEXT_FIELDS]
    return "\n\n".join(p for p in parts if p)


def ingest(limit: int = 200) -> int:
    client = MongoClient(MONGODB_URI)
    try:
        src = client[MONGODB_DB][MONGODB_COLLECTION]
        dst = client[MONGODB_DB][RAG_CHUNKS_COLLECTION]

        query = {"$or": [{f: {"$nin": [None, ""]}} for f in _TEXT_FIELDS]}
        listings = list(src.find(query, limit=limit))
        print(f"Fetched {len(listings)} listings from '{MONGODB_COLLECTION}'.")

        dst.drop()
        print(f"Dropped existing '{RAG_CHUNKS_COLLECTION}' collection.")

        records: list[dict] = []
        for i, doc in enumerate(listings):
            text = _build_text(doc)
            if not text:
                continue
            chunks = _SPLITTER.split_text(text)
            embeddings = _EMBEDDER.embed_documents(chunks)
            for chunk, vec in zip(chunks, embeddings):
                records.append({
                    "listing_id": str(doc["_id"]),
                    "name": doc.get("name", ""),
                    "text": chunk,
                    "embedding": vec,
                })
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(listings)} listings...")

        if records:
            dst.insert_many(records)
        print(f"Upserted {len(records)} chunks to '{RAG_CHUNKS_COLLECTION}'.")
        return len(records)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    ingest(args.limit)
