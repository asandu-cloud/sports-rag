# Embedding

import json, os, argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv


# ---- CONFIG (EPL-only) ----
INDEX_DIR  = "/Users/sanduandrei/Desktop/Betting_RAG/Index"
CHROMA_DIR = "/Users/sanduandrei/Desktop/Betting_RAG/Index/chroma"
COLLECTION = "football_top5"   # keep one collection, filter by metadata
EMBED_MODEL = "text-embedding-3-large"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _upsert_batch(collection, batch):
    ids   = [d["id"] for d in batch]
    docs  = [d["text"] for d in batch]
    metas = [d["metadata"] for d in batch]
    embs = client.embeddings.create(model=EMBED_MODEL, input=docs).data
    vecs = [e.embedding for e in embs]
    collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)

def upsert_docs(collection, docs, batch_size=256):
    buf = []
    for d in docs:
        buf.append(d)
        if len(buf) >= batch_size:
            _upsert_batch(collection, buf)
            buf = []
    if buf:
        _upsert_batch(collection, buf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection.")
    args = parser.parse_args()

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # ✅ Use PersistentClient to guarantee on-disk storage
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    if args.reset:
        try:
            db.delete_collection(COLLECTION)
            print(f"Deleted existing collection: {COLLECTION}")
        except Exception:
            pass

    col = db.get_or_create_collection(COLLECTION)

    # only pick EPL normalized files
    norm_paths = sorted(Path(INDEX_DIR).glob("normalized_EPL_*.json"))
    if not norm_paths:
        raise SystemExit(f"No normalized EPL files found in {INDEX_DIR}")

    total = 0
    for path in norm_paths:
        docs = json.loads(path.read_text())
        print(f"Upserting {len(docs)} docs from {path.name}")
        upsert_docs(col, docs)
        total += len(docs)

    # Persist (safe no-op on newer versions)
    try:
        db.persist()
    except Exception:
        pass

    print(f"Done. Upserted {total} EPL docs into {COLLECTION}.")
    try:
        print("Vector count in collection:", col.count())
    except Exception:
        pass
    print("Chroma directory:", CHROMA_DIR)