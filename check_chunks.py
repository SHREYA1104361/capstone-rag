from src.retrieval import retrieve
results = retrieve("listings near beach with wifi")
for r in results:
    print("---", r["name"])
    print(r["text"][:300])
    print()
