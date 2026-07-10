"""Recall@k: fraction of relevant listing IDs present in retrieved context."""
from __future__ import annotations


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids)
