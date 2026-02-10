# 02_query_smoke.py — EPL-only retrieval smoke test (PersistentClient)
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

try:
    from chroma_backend import backend_description, env_first, get_chroma_client
except ImportError:
    from Scripts.rag_ingest.chroma_backend import backend_description, env_first, get_chroma_client

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = str(ROOT / "Index" / "chroma")
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")
EMBED_MODEL = "text-embedding-3-large"

load_dotenv()
client = OpenAI(api_key=env_first("OPENAI_API_KEY"))

def embed_one(text: str):
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def query(q: str, where: dict, k=10):
    db = get_chroma_client(CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)  # safe if already created
    res = col.query(query_embeddings=[embed_one(q)], n_results=k, where=where)

    hits = []
    for doc, meta, id_ in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
        hits.append({"id": id_, "text": doc, "meta": meta})
    return hits

if __name__ == "__main__":
    where = {"league": "EPL"}  # hard filter
    tests = [
        "aggressive forwards with high cards per 90",
        "Arsenal control and dominance in recent fixtures",
        "Jadon Sancho shots and passes last match",
    ]
    # quick existence check
    db = get_chroma_client(CHROMA_DIR)
    print("Chroma backend:", backend_description(CHROMA_DIR))
    print("Collections present:", [c.name for c in db.list_collections()])
    col = db.get_or_create_collection(COLLECTION)
    print("Vector count in collection:", col.count())

    for t in tests:
        print("\n=== QUERY:", t)
        hits = query(t, where=where, k=5)
        for h in hits:
            print("-", h["meta"].get("doc_type"), "|", h["meta"].get("team"), "|",
                  h["meta"].get("player_name"), "| league:", h["meta"].get("league"))
        assert all(h["meta"].get("league") == "EPL" for h in hits), "Cross-league leak detected!"
    print("\nSmoke tests passed ✅ (EPL-only).")
