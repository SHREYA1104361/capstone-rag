from src.graph import run
from evals.evaluators.refusal import is_refusal

cases = [
    ("Is there a listing that is both pet-friendly and near the beach?", False),
    ("Which listings were built before 1900?", True),
    ("Which listings have a Michelin-starred restaurant on site?", True),
]
for q, should_refuse in cases:
    r = run(q)
    print("Q:", q)
    print("A:", r["answer"][:200])
    print("is_refusal:", is_refusal(r["answer"]), "| should_refuse:", should_refuse)
    print()
