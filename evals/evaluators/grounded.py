"""Groundedness: every citation must come from the retrieved context."""
from __future__ import annotations


def is_grounded(citations: list[str], retrieved: list[dict]) -> bool:
    """Return True if every citation matches a retrieved doc by name or listing_id."""
    if not citations:
        return True  # refusal case — no citations to check
    valid = set()
    for doc in retrieved:
        if doc.get("name"):
            valid.add(doc["name"])
        if doc.get("listing_id"):
            valid.add(doc["listing_id"])
    return all(c in valid for c in citations)
