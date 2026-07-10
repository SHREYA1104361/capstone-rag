"""Eval harness - regression gate.

Usage:
    python evals/run_evals.py [--experiment-name <name>]

Exits non-zero if any metric falls below its threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from src.graph import run as agent_run
from evals.evaluators.recall import recall_at_k
from evals.evaluators.grounded import is_grounded
from evals.evaluators.refusal import refusal_correct

DATASET = Path(__file__).parent / "datasets" / "qa.jsonl"

THRESHOLDS = {
    "recall@k": 0.70,
    "grounded":  0.90,
    "refusal":   0.80,
}


def _load_dataset() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_experiment(rows: list[dict], experiment_name: str) -> list[dict]:
    results = []
    for row in rows:
        result    = agent_run(row["question"])
        context_docs  = result["context"]
        retrieved_ids = [c["listing_id"] for c in context_docs]
        citations     = result.get("citations", [])
        answer        = result["answer"]
        should_refuse = row.get("should_refuse", False)
        relevant      = row.get("relevant_listing_ids", [])

        # recall: skip (score=1.0) when no ground-truth ids provided
        recall   = recall_at_k(relevant, retrieved_ids) if relevant else 1.0
        grounded = is_grounded(citations, context_docs)
        refusal  = refusal_correct(answer, should_refuse)

        results.append({
            "question":     row["question"],
            "answer":       answer,
            "citations":    citations,
            "recall":       recall,
            "grounded":     int(grounded),
            "refusal":      int(refusal),
            "should_refuse": should_refuse,
        })
        print(f"  Q: {row['question'][:55]!r}  recall={recall:.2f}  grounded={int(grounded)}  refusal={int(refusal)}")

    return results


def _log_to_langsmith(results: list[dict], experiment_name: str) -> None:
    try:
        from langsmith import Client
        client = Client()
        dataset_name = f"capstone-rag-qa-{datetime.utcnow().strftime('%Y%m%d')}"
        try:
            ds = client.create_dataset(dataset_name)
        except Exception as e:
            print(f"[LangSmith] Dataset already exists, reading: {e}")
            ds = client.read_dataset(dataset_name=dataset_name)
        for r in results:
            try:
                client.create_example(
                    inputs={"question": r["question"]},
                    outputs={"answer": r["answer"]},
                    dataset_id=ds.id,
                )
            except Exception as e:
                print(f"[LangSmith] Skipped example: {e}")
        print(f"\nLangSmith experiment '{experiment_name}' logged to dataset '{dataset_name}'.")
    except Exception as exc:
        print(f"\n[LangSmith] Skipped - {exc}")


def main(experiment_name: str) -> int:
    rows = _load_dataset()
    print(f"Running {len(rows)} examples  (experiment: {experiment_name})\n")

    results = _run_experiment(rows, experiment_name)

    n = len(results)
    metrics = {
        "recall@k": sum(r["recall"]   for r in results) / n,
        "grounded": sum(r["grounded"] for r in results) / n,
        "refusal":  sum(r["refusal"]  for r in results) / n,
    }

    print("\n-- Metrics --------------------------------------------------")
    print(f"{'Metric':<20} {'Score':>8}  {'Threshold':>10}  {'Pass?':>6}")
    print("-" * 52)
    failed = []
    for name, score in metrics.items():
        threshold = THRESHOLDS[name]
        passed = score >= threshold
        if not passed:
            failed.append(name)
        print(f"{name:<20} {score:>8.2%}  {threshold:>10.2%}  {'PASS' if passed else 'FAIL':>6}")
    print("-" * 52)

    _log_to_langsmith(results, experiment_name)

    if failed:
        print(f"\nREGRESSION GATE FAILED: {', '.join(failed)}")
        return 1
    print("\nAll metrics passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=f"run-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}")
    args = parser.parse_args()
    sys.exit(main(args.experiment_name))
