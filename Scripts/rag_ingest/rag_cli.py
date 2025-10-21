# Small local RAG trial

import os
import sys
import argparse
from typing import List, Dict
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# ---- CONFIG ----
CHROMA_DIR = "/Users/sanduandrei/Desktop/Betting_RAG/Index/chroma"
COLLECTION = "football_top5"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"  # cheap/fast; swap to a bigger model if you like

# ---- INIT ----
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Small helpers ---
def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts).data
    return [r.embedding for r in resp]

def resolve_league_from_flag_or_query(league_flag: str | None, user_q: str) -> str | None:
    if league_flag:
        return league_flag
    q = user_q.lower()
    hints = {
        "EPL": ["epl", "premier league", "england"],
        "LaLiga": ["laliga", "la liga", "spain"],
        "SerieA": ["serie a", "italy"],
        "Bundesliga": ["bundesliga", "germany"],
        "Ligue1": ["ligue 1", "france"],
    }
    for lg, keys in hints.items():
        if any(k in q for k in keys):
            return lg
    return None  # let the query go wide if not provided

def _build_where(league: str | None, season: str | None):
    clauses = []
    if league:
        clauses.append({"league": league})              # equals
    if season:
        clauses.append({"season": season})              # equals

    if not clauses:
        return {}                                       # no filter
    if len(clauses) == 1:
        return clauses[0]                               # single field is OK
    return {"$and": clauses}                            # multiple fields need an operator

def retrieve(q: str, league: str | None, season: str | None, k: int = 12) -> list[dict]:
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    where = _build_where(league, season)
    vec = embed([q])[0]
    res = col.query(query_embeddings=[vec], n_results=k, where=where)

    hits = []
    for doc, meta, id_ in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
        hits.append({"id": id_, "text": doc, "meta": meta})
    return hits


def build_prompt(user_q: str, context_docs: List[Dict], league: str | None):
    lead = (
        f"You are a football analyst. Use ONLY the context. "
        "If a numeric value (e.g., yellow cards) appears, quote the exact number and the row it came from. "
        "If missing, say you don't have it."
        f"{'Stay in ' + league + ' unless the user explicitly asks otherwise. ' if league else ''}"
        "Prefer recent per-fixture rows for props. If data is insufficient, say so clearly. "
        "Show your reasoning succinctly; avoid speculation."
    )
    ctx = "\n\n".join(d["text"] for d in context_docs[:8])
    return [
        {"role": "system", "content": lead},
        {"role": "user", "content": f"Question: {user_q}\n\nContext:\n{ctx}\n\nAnswer with a short rationale and specific player/team stats if relevant."}
    ]

def format_sources(hits: List[Dict], max_sources: int = 5) -> str:
    lines = []
    for h in hits[:max_sources]:
        m = h["meta"]
        tag = f"{m.get('doc_type','?')} · {m.get('league','?')} {m.get('season','?')}"
        who = m.get("player_name") or m.get("team")
        fx  = m.get("fixture")
        lines.append(f"- {who or 'N/A'} | {tag}" + (f" | {fx}" if fx else ""))
    return "Sources:\n" + "\n".join(lines)

def chat_once(user_q: str, league_flag: str | None, season: str | None, k: int, show_sources: bool):
    # Resolve league (flag beats heuristic)
    league = resolve_league_from_flag_or_query(league_flag, user_q)

    # Retrieve
    hits = retrieve(user_q, league=league, season=season, k=k)
    if not hits:
        print("No results found in the index for your filters. Try widening league/season or re-ingest.")
        return

    # Build prompt and call chat
    messages = build_prompt(user_q, hits, league)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    print("\n" + answer.strip())
    if show_sources:
        print("\n" + format_sources(hits))

def main():
    ap = argparse.ArgumentParser(description="Terminal RAG over your football KB")
    ap.add_argument("--league", type=str, default="EPL", help="EPL|LaLiga|SerieA|Bundesliga|Ligue1 (default: EPL)")
    ap.add_argument("--season", type=str, default=None, help='Season filter like "2024/25" (optional)')
    ap.add_argument("--topk", type=int, default=12, help="Top-k docs to retrieve (default: 12)")
    ap.add_argument("--no-sources", action="store_true", help="Hide sources footer")
    ap.add_argument("--once", type=str, default=None, help="Run a single query then exit")
    args = ap.parse_args()

    # Quick DB sanity check
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    names = [c.name for c in db.list_collections()]
    if COLLECTION not in names:
        print(f"Collection '{COLLECTION}' not found at {CHROMA_DIR}. Did you run 01_embed_and_upsert.py?")
        sys.exit(1)

    if args.once:
        chat_once(args.once, args.league, args.season, args.topk, show_sources=not args.no_sources)
        return

    print(f"RAG ready. League={args.league or 'auto'} Season={args.season or 'any'} | type 'exit' to quit.")
    while True:
        try:
            q = input("\nClient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        chat_once(q, args.league, args.season, args.topk, show_sources=not args.no_sources)

if __name__ == "__main__":
    main()