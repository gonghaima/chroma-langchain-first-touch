import chromadb

client = chromadb.PersistentClient(path="chroma")
collection = client.get_collection("lc_chroma_demo")

print(f"Total documents: {collection.count()}\n")
print("=" * 80)

results = collection.get(include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
    source = meta.get("source", "unknown")
    page = meta.get("page_label", meta.get("page", "?"))
    total = meta.get("total_pages", "?")

    print(f"[{i}] Page {page}/{total}  |  {source}")
    print(f"{doc[:300].strip()}...")
    print("-" * 80)
