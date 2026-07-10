"""Refusal-correctness evaluator (deterministic, no LLM)."""
from __future__ import annotations

_REFUSAL_PHRASES = (
    "i don't know",
    "no relevant listings",
    "cannot answer",
    "not enough information",
    "no information",
    "context does not",
    "context doesn't",
    "not mentioned",
    "no listing",
)


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in _REFUSAL_PHRASES)


def refusal_correct(answer: str, should_refuse: bool) -> bool:
    """Return True when the agent's refusal behaviour matches expectation."""
    return is_refusal(answer) == should_refuse
