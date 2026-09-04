# Embedding & Upserting — Incremental by default

import argparse
import json
import time
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
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")
EMBED_MODEL = "text-embedding-3-large"
ALL_LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL", "UEL", "UECL"]

load_dotenv()
client = OpenAI(api_key=env_first("OPENAI_API_KEY"))

# Fixture documents are immutable after a match is completed, so the normal
# incremental path can safely leave them alone.  Profiles are deliberately
# *not* in this set: their IDs are stable for the season but their rates,
# minutes risk, and form change after every gameweek.
IMMUTABLE_DOC_TYPES = {"team_fixture", "player_fixture"}


def _upsert_batch(collection, batch):
    ids   = [d["id"] for d in batch]
    docs  = [d["text"] for d in batch]
    metas = [d["metadata"] for d in batch]
    embs = client.embeddings.create(model=EMBED_MODEL, input=docs).data
    vecs = [e.embedding for e in embs]
    collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)


def upsert_docs(collection, docs, batch_size=256, skip_ids=None):
    """Upsert docs into Chroma. With skip_ids, skips immutable doc types
    that are already embedded (team_fixture, player_fixture).  Team and
    player profiles are always re-embedded because their season-to-date
    aggregates shift after every gameweek."""
    seen = set()
    buf = []
    skipped = 0
    embedded = 0
    for d in docs:
        if d["id"] in seen:
            continue
        # In skip mode, only completed-fixture documents are immutable.
        # Profile IDs stay stable but their contents are dynamic.
        if skip_ids and d["id"] in skip_ids:
            doc_type = (d.get("metadata") or {}).get("doc_type", "")
            if doc_type in IMMUTABLE_DOC_TYPES:
                skipped += 1
                continue
        seen.add(d["id"])
        buf.append(d)
        if len(buf) >= batch_size:
            _upsert_batch(collection, buf)
            embedded += len(buf)
            buf = []
    if buf:
        _upsert_batch(collection, buf)
        embedded += len(buf)
    return embedded, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and upsert normalized docs into Chroma.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection (FULL rebuild — slow).")
    parser.add_argument("--league", type=str, default="EPL",
                        help="League to upsert (e.g. EPL, LaLiga). Default: EPL.")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Upsert all discovered normalized league files.")
    parser.add_argument(
        "--season", type=int, default=None,
        help="Restrict embedding to one API-Football season year (e.g. 2026). "
             "Keeps a weekly refresh from re-reading historical season files.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip fixture/player docs already in Chroma (DEFAULT — incremental mode).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force re-embed ALL docs even if already in Chroma.")
    args = parser.parse_args()

    # --no-skip overrides the default --skip-existing
    skip_mode = args.skip_existing and not args.no_skip and not args.reset

    t_start = time.time()

    db = get_chroma_client(CHROMA_DIR)
    print("Chroma backend:", backend_description(CHROMA_DIR))

    if args.reset:
        try:
            db.delete_collection(COLLECTION)
            print("Deleted existing collection:", COLLECTION)
        except Exception:
            pass
        skip_mode = False  # no point skipping after a reset

    col = db.get_or_create_collection(COLLECTION)

    # Build set of existing IDs for incremental mode
    existing_ids = set()
    if skip_mode:
        t_ids = time.time()
        try:
            existing_ids = set(col.get(include=[])["ids"])
            print(f"[{time.time() - t_ids:.1f}s] Found {len(existing_ids):,} existing docs — incremental mode active.")
        except Exception:
            pass
    else:
        mode_label = "FULL REBUILD (--reset)" if args.reset else "FULL RE-EMBED (--no-skip)"
        print(f"Mode: {mode_label} — all docs will be embedded.")

    # Discover normalized files for requested league(s)
    if args.all_leagues:
        leagues = ALL_LEAGUES
    else:
        leagues = [args.league.strip()]

    norm_paths = []
    for lg in leagues:
        pattern = (
            f"normalized_{lg}_{args.season}_*.json"
            if args.season is not None
            else f"normalized_{lg}_*.json"
        )
        found = sorted(Path(INDEX_DIR).glob(pattern))
        norm_paths.extend(found)

    if not norm_paths:
        label = "all leagues" if args.all_leagues else args.league
        raise SystemExit(f"No normalized files found for {label} in {INDEX_DIR}")

    total_embedded = 0
    total_skipped = 0
    total_docs = 0

    for path in norm_paths:
        t_file = time.time()
        docs = json.loads(path.read_text())
        total_docs += len(docs)

        embedded, skipped = upsert_docs(
            col, docs, skip_ids=existing_ids if skip_mode else None
        )
        total_embedded += embedded
        total_skipped += skipped

        elapsed = time.time() - t_file
        skip_str = f" (skipped {skipped:,})" if skipped else ""
        print(f"[{elapsed:.1f}s] {path.name}: {embedded:,} embedded{skip_str} of {len(docs):,} total")

    # Persist (safe no-op on newer versions)
    try:
        db.persist()
    except Exception:
        pass

    t_total = time.time() - t_start
    league_label = "all leagues" if args.all_leagues else args.league
    print(f"\nDone in {t_total:.1f}s.")
    print(f"  Processed: {total_docs:,} docs from {len(norm_paths)} files")
    print(f"  Embedded:  {total_embedded:,} (new + updated profiles)")
    print(f"  Skipped:   {total_skipped:,} (unchanged fixture documents)")
    try:
        print(f"  Vectors:   {col.count():,} in collection")
    except Exception:
        pass
    print(f"  Chroma:    {CHROMA_DIR}")
