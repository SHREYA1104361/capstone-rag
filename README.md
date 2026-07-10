# Capstone RAG

A self-correcting Retrieval-Augmented Generation system built on MongoDB Atlas Vector Search, LangGraph, and LangSmith.

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐     $vectorSearch      ┌──────────────────┐
│   retrieve  │ ──────────────────────▶│  MongoDB Atlas   │
│  (node 1)   │ ◀──────────────────────│  rag_chunks      │
└──────┬──────┘    top-k chunks        └──────────────────┘
       │
       ▼
┌──────────────────┐
│ grade_relevance  │  score < threshold AND rewrites < 2?
│    (node 2)      │ ──── rewrite query ──▶ retrieve again
└──────┬───────────┘
       │ context is good (or max rewrites reached)
       ▼
┌─────────────┐
│  generate   │  answer from context only; "I don't know" if insufficient
│  (node 3)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    cite     │  attach listing names/ids used
│  (node 4)   │
└─────────────┘
       │
       ▼
  Final answer + citations
```

**Ingest pipeline**

```
listingsAndReviews (MongoDB sample dataset)
  └─ concat text fields (description, summary, space, neighborhood_overview, transit, notes)
       └─ RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
            └─ text-embedding-3-small (OpenAI)
                 └─ rag_chunks collection  ←  $vectorSearch index (1536 dims, cosine)
```

---

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in MONGODB_URI, OPENAI_API_KEY, LANGSMITH_API_KEY
```

### 1 — Ingest

```bash
python -m src.ingest --limit 200
```

After ingestion, create the Atlas Vector Search index on `rag_chunks`:

```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 1536,
    "similarity": "cosine"
  }]
}
```

Wait for the index status to show **Active** before querying.

### 2 — Smoke-test retrieval

```python
from src.retrieval import retrieve
print(retrieve("quiet apartment near the beach with wifi", k=5))
```

### 3 — Run the agent

```bash
python -m src.graph
```

### 4 — Run evals (regression gate)

```bash
python evals/run_evals.py
# exits 0 if all thresholds pass, 1 otherwise

# name an experiment for LangSmith comparison
python evals/run_evals.py --experiment-name before-tuning
python evals/run_evals.py --experiment-name after-tuning
```

---

## Metrics

| Metric | Threshold | Meaning |
|---|---|---|
| recall@k | ≥ 0.70 | Fraction of relevant listing IDs present in top-k retrieved chunks |
| grounded | ≥ 0.90 | Every citation appears in the retrieved context (no hallucinated sources) |
| refusal | = 1.00 | Agent refuses on unanswerable questions and answers on answerable ones |

---

## Eval dataset (`evals/datasets/qa.jsonl`)

25 examples across three categories:

| Category | Count | Description |
|---|---|---|
| Normal | 15 | Answerable questions about amenities, location, features |
| No-answer | 5 | Questions with no grounded answer in the corpus (agent must refuse) |
| Multi-doc | 5 | Questions requiring evidence from multiple listings |

---

## LangSmith

Every agent run is a single trace with four child spans (`retrieve`, `grade_relevance`, `generate`, `cite`).

To compare two experiments:
1. Run `python evals/run_evals.py --experiment-name baseline`
2. Tune chunking / k / threshold
3. Run `python evals/run_evals.py --experiment-name tuned`
4. Open LangSmith → Experiments → select both → Compare

---

## What I'd do with more time

- **Hybrid search**: combine `$vectorSearch` with `$search` (BM25) using Reciprocal Rank Fusion for better recall on keyword-heavy queries.
- **HyDE**: generate a hypothetical answer before embedding the query to improve retrieval on abstract questions.
- **Reranking**: add a cross-encoder reranker (e.g. Cohere Rerank) between retrieval and generation.
- **Conversation memory**: store chat history in MongoDB and inject recent turns into the query rewrite step.
- **Streaming UI**: expose the graph via FastAPI with Server-Sent Events so the answer streams token-by-token.
- **Online eval**: attach an evaluator to a Kafka/SQS stream of live questions to catch regressions in production.
