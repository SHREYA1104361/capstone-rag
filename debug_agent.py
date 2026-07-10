from src.retrieval import retrieve
from src.config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

question = "What listings are available near the beach with wifi?"

# Step 1: retrieve
context = retrieve(question)
print("=== Retrieved", len(context), "chunks ===")
for c in context:
    print(f"  {c['name']} | score={round(c['score'],3)}")

# Step 2: check scores
scores = [c.get("score", 0) for c in context]
print("\nBest score:", max(scores, default=0))
print("Threshold: 0.30")
print("Would refuse?", not context or max(scores, default=0) < 0.30)

# Step 3: try generate directly
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)
context_text = "\n\n".join(f"[{c.get('name')}]: {c['text']}" for c in context)
messages = [
    SystemMessage(content="Answer using ONLY the provided context. If insufficient, say I don't know."),
    HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}"),
]
resp = llm.invoke(messages)
print("\n=== LLM Answer ===")
print(resp.content[:400])
