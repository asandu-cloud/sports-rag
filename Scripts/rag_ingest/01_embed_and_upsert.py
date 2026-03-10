# Embedding

import argparse
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

try:
    from chroma_backend import backend_description, env_first, get_chroma_client
except ImportError:
    from Scripts.rag_ingest.chroma_backend import backend_description, env_first, get_chroma_client


# ---- CONFIG ----
ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = str(ROOT / "Index")
CHROMA_DIR = str(ROOT / "Index" / "chroma")
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")   # keep one collection, filter by metadata
EMBED_MODEL = "text-embedding-3-large"
ALL_LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL", "UEL", "UECL"]

load_dotenv()
client = OpenAI(api_key=env_first("OPENAI_API_KEY"))

def _upsert_batch(collection, batch):
    ids   = [d["id"] for d in batch]
    docs  = [d["text"] for d in batch]
    metas = [d["metadata"] for d in batch]
    embs = client.embeddings.create(model=EMBED_MODEL, input=docs).data
    vecs = [e.embedding for e in embs]
    collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)

def upsert_docs(collection, docs, batch_size=256, skip_ids=None):
    seen = set()
    buf = []
    skipped = 0
    for d in docs:
        if d["id"] in seen:
            continue
        # In skip mode: skip immutable doc types (fixtures/players) if already embedded.
        # team_profile always re-embeds because averages shift each gameweek.
        if skip_ids and d["id"] in skip_ids:
            doc_type = (d.get("metadata") or {}).get("doc_type", "")
            if doc_type in ("team_fixture", "player_fixture", "player_profile"):
                skipped += 1
                continue
        seen.add(d["id"])
        buf.append(d)
        if len(buf) >= batch_size:
            _upsert_batch(collection, buf)
            buf = []
    if buf:
        _upsert_batch(collection, buf)
    if skipped:
        print(f"  Skipped {skipped} existing docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and upsert normalized docs into Chroma.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection.")
    parser.add_argument(
        "--league", type=str, default="EPL",
        help="League to upsert (e.g. EPL, LaLiga). Default: EPL.",
    )
    parser.add_argument(
        "--all-leagues", action="store_true",
        help="Upsert all discovered normalized league files.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip fixture/player docs already in Chroma (only embed new + updated profiles).",
    )
    args = parser.parse_args()

    db = get_chroma_client(CHROMA_DIR)
    print("Chroma backend:", backend_description(CHROMA_DIR))

    if args.reset:
        try:
            db.delete_collection(COLLECTION)
            print(f"Deleted existing collection: {COLLECTION}")
        except Exception:
            pass

    col = db.get_or_create_collection(COLLECTION)

    # Build set of existing IDs for incremental mode
    existing_ids = set()
    if args.skip_existing:
        try:
            existing_ids = set(col.get(include=[])["ids"])
            print(f"Found {len(existing_ids)} existing docs in collection — will skip unchanged fixtures/players.")
        except Exception:
            pass

    # Discover normalized files for requested league(s)
    if args.all_leagues:
        leagues = ALL_LEAGUES
    else:
        leagues = [args.league.strip()]

    norm_paths = []
    for lg in leagues:
        found = sorted(Path(INDEX_DIR).glob(f"normalized_{lg}_*.json"))
        norm_paths.extend(found)

    if not norm_paths:
        label = "all leagues" if args.all_leagues else args.league
        raise SystemExit(f"No normalized files found for {label} in {INDEX_DIR}")

    total = 0
    for path in norm_paths:
        docs = json.loads(path.read_text())
        print(f"Upserting {len(docs)} docs from {path.name}")
        upsert_docs(col, docs, skip_ids=existing_ids if args.skip_existing else None)
        total += len(docs)

    # Persist (safe no-op on newer versions)
    try:
        db.persist()
    except Exception:
        pass

    league_label = "all leagues" if args.all_leagues else args.league
    print(f"Done. Upserted {total} docs ({league_label}) into {COLLECTION}.")
    try:
        print("Vector count in collection:", col.count())
    except Exception:
        pass
    print("Chroma directory:", CHROMA_DIR)
