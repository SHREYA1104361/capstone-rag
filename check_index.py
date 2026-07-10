from pymongo import MongoClient
from src.config import MONGODB_URI, MONGODB_DB, RAG_CHUNKS_COLLECTION, VECTOR_INDEX_NAME
from src.embedder import get_embedder

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
col = db[RAG_CHUNKS_COLLECTION]

print("=== Collection stats ===")
print("Chunk count:", col.count_documents({}))
doc = col.find_one({})
if doc:
    print("Fields:", list(doc.keys()))
    print("Embedding dims:", len(doc["embedding"]))

print("\n=== Search indexes ===")
try:
    for idx in col.list_search_indexes():
        print("  name:", idx.get("name"), "| status:", idx.get("status"), "| latestDefinition:", idx.get("latestDefinition", {}).get("fields", "?"))
except Exception as e:
    print("  Error listing indexes:", e)

print("\n=== Raw $vectorSearch test ===")
embedder = get_embedder()
vec = embedder.embed_query("quiet apartment near beach")
pipeline = [
    {"$vectorSearch": {
        "index": VECTOR_INDEX_NAME,
        "path": "embedding",
        "queryVector": vec,
        "numCandidates": 50,
        "limit": 3,
    }},
    {"$project": {"_id": 0, "name": 1, "score": {"$meta": "vectorSearchScore"}}},
]
results = list(col.aggregate(pipeline))
print("Results:", len(results))
for r in results:
    print(" ", r.get("name"), round(r.get("score", 0), 3))
