import os, re, json, csv, hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Optional

# --------- CONFIG ---------
# Change output base to /Output
OUTPUT_BASE = "/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering"  # where *feature_engineering folders live
INDEX_DIR   = "/Users/sanduandrei/Desktop/Betting_RAG/Index"   # where normalized_*.json will be written

# Map folder name prefixes to canonical league names
LEAGUE_FROM_DIR = {
    "prem": "EPL",
    "laliga": "LaLiga",
    "ligue1": "Ligue1",
    "seriaa": "SerieA",      # tolerate your folder spelling
    "seriea": "SerieA",
    "bundesliga": "Bundesliga",
}

# --------- UTILS ---------
def make_doc_id(parts: Iterable) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.md5(base.encode()).hexdigest()

def euro_season_from_year(year: int) -> str:
    """2025 -> '2024/25'"""
    return f"{year-1}/{str(year)[-2:]}"

def find_year_token(name: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None

def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def read_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def read_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

def infer_home_away(fixture: str, team: str):
    if not fixture or " vs " not in fixture.lower():
        return None, None
    left, right = fixture.split(" vs ", 1)
    t = (team or "").lower().strip()
    if t == left.lower().strip():
        return "home", right.strip()
    if t == right.lower().strip():
        return "away", left.strip()
    return None, None

def league_from_dir(dirname: str) -> Optional[str]:
    key = dirname.lower()
    for prefix, league in LEAGUE_FROM_DIR.items():
        if key.startswith(prefix):
            return league
    return None

def season_from_any(*candidates: Optional[str]) -> Optional[str]:
    for c in candidates:
        if not c:
            continue
        yr = find_year_token(c)
        if yr:
            return euro_season_from_year(yr)
    return None

# --------- DOC BUILDERS ---------
def doc_from_player_engineered(row: dict, league: str, season: str, source_file: str) -> dict:
    # Ensure numeric-like strings from CSV become usable (best-effort)
    def n(x): 
        try:
            return float(x)
        except (TypeError, ValueError):
            return x
    home_away, opponent = infer_home_away(row.get("fixture",""), row.get("team",""))
    text = (
        f"{league} {season} | Player Fixture\n"
        f"Fixture: {row.get('fixture')} | Team: {row.get('team')} ({home_away or 'n/a'}) vs {opponent or row.get('opponent') or 'n/a'}\n"
        f"Player: {row.get('name')} (id:{row.get('player_id')}, pos:{row.get('position')}) | Min:{row.get('minutes')} | Rating:{row.get('rating')}\n"
        f"G:{row.get('goals')} A:{row.get('assists')} Shots:{row.get('shots_total')} (On:{row.get('shots_on')}) "
        f"Passes:{row.get('passes_total')} (Acc:{row.get('accurate_passes')}, {row.get('pass_accuracy_%')}%) "
        f"Tkl:{row.get('tackles')} Int:{row.get('interceptions')} YC:{row.get('yellow_cards')} RC:{row.get('red_cards')}\n"
        f"Derived: SoT%:{row.get('shots_on_target_ratio')} Conv%:{row.get('goal_conversion_rate')} DuelWin%:{row.get('duel_win_ratio')} "
        f"TackleSucc%:{row.get('tackle_success_rate')} GI:{row.get('goal_involvement')} Cards/90:{row.get('cards_per_90')} Fouls/90:{row.get('fouls_per_90')}\n"
        f"FormIdx:{row.get('form_index')} AggIdx:{row.get('aggression_index_norm')} CtrlIdx:{row.get('control_index')} "
        f"OppAggIdx:{row.get('aggression_index_norm_opp')} OppCtrlIdx:{row.get('control_index_opp')} "
        f"CardRiskExp:{row.get('expected_card_risk')} FoulPressureExp:{row.get('expected_foul_pressure')}"
    )
    meta = {
        "entity_type": "player",
        "doc_type": "player_fixture",
        "league": league, "season": season,
        "fixture": row.get("fixture"),
        "team": row.get("team"),
        "opponent": opponent or row.get("opponent"),
        "home_away": home_away,
        "player_id": _maybe_int(row.get("player_id")),
        "player_name": row.get("name"),
        "position": row.get("position"),
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "player_fixture", meta["player_id"], row.get("fixture")])
    return {"id": uid, "text": text, "metadata": meta}

def doc_from_player_profile(row: dict, season_hint: Optional[str], source_file: str) -> dict:
    league = row.get("league")
    season = row.get("season") or season_hint
    text = (
        f"{league} {season} | Player Profile\n"
        f"{row.get('name')} (id:{row.get('player_id')}, {row.get('team')}) – pos:{row.get('position')} rating:{row.get('rating')}\n"
        f"G/90:{row.get('goals_per_90')} A/90:{row.get('assists_per_90')} Shots/90:{row.get('shots_per_90')} SoT/90:{row.get('sot_per_90')} "
        f"Passes/90:{row.get('passes_per_90')} Acc%:{row.get('pass_accuracy_pct')}\n"
        f"Tkl/90:{row.get('tackles_per_90')} Int/90:{row.get('interceptions_per_90')} Fouls/90:{row.get('fouls_per_90_calc')} "
        f"Cards/90:{row.get('cards_per_90_calc')} DuelWin%:{row.get('duel_win_pct')}\n"
        f"FormIdx:{row.get('form_index')} AggIdx:{row.get('aggression_index_norm')} CtrlIdx:{row.get('control_index')} "
        f"CardRiskExp:{row.get('expected_card_risk')} FoulPressureExp:{row.get('expected_foul_pressure')}\n"
        f"Tags: {', '.join(row.get('style_tags', []))}\n"
        f"Summary: {row.get('summary_nl')}"
    )
    meta = {
        "entity_type": "player", "doc_type": "player_profile",
        "league": league, "season": season,
        "team": row.get("team"),
        "player_id": _maybe_int(row.get("player_id")),
        "player_name": row.get("name"),
        "position": row.get("position"),
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "player_profile", meta["player_id"]])
    return {"id": uid, "text": text, "metadata": meta}

def doc_from_team_engineered(row: dict, league: str, season: str, source_file: str) -> dict:
    home_away, opponent = infer_home_away(row.get("fixture",""), row.get("team",""))
    text = (
        f"{league} {season} | Team Fixture\n"
        f"Fixture: {row.get('fixture')} | Team: {row.get('team')} ({home_away or 'n/a'}) vs {opponent or 'n/a'}\n"
        f"G:{row.get('goals')} A:{row.get('assists')} ShotsFor:{row.get('shots_total_x')} (On:{row.get('shots_on_x')}) "
        f"Passes:{row.get('passes_total')} (Acc:{row.get('accurate_passes')}) Pos:{row.get('possession')}\n"
        f"xG:{row.get('expected_goals')} GoalsPrevented:{row.get('goals_prevented')} Corners:{row.get('corners')} "
        f"Fouls:{row.get('fouls_committed')} YC:{row.get('yellow_cards')} RC:{row.get('red_cards')}\n"
        f"Dominance:{row.get('dominance_index')} Control:{row.get('control_index')} ShotAccFor:{row.get('shots_accuracy_for')} "
        f"Fouls/90:{row.get('fouls_per_90_team')} Cards/90:{row.get('cards_per_90_team')}"
    )
    meta = {
        "entity_type": "team", "doc_type": "team_fixture",
        "league": league, "season": season,
        "fixture": row.get("fixture"),
        "team": row.get("team"),
        "opponent": opponent,
        "home_away": home_away,
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "team_fixture", row.get("team"), row.get("fixture")])
    return {"id": uid, "text": text, "metadata": meta}

def doc_from_team_profile(row: dict, season_hint: Optional[str], source_file: str) -> dict:
    league = row.get("league")
    season = row.get("season") or season_hint
    text = (
        f"{league} {season} | Team Profile\n"
        f"{row.get('team')} | MP:{row.get('matches_played')} G:{row.get('goals')} A:{row.get('assists')} "
        f"G/Match:{row.get('goals_for_pm')} xG/Match:{row.get('expected_goals')} Poss:{row.get('possession')}\n"
        f"Dominance:{row.get('dominance_index')} Control:{row.get('control_index')} "
        f"Fouls/90:{row.get('fouls_per_90_team')} Cards/90:{row.get('cards_per_90_team')} Cards/Foul:{row.get('cards_per_foul_team')}\n"
        f"Tags: {', '.join(row.get('style_tags', []))}\n"
        f"Summary: {row.get('summary_nl')}"
    )
    meta = {
        "entity_type": "team", "doc_type": "team_profile",
        "league": league, "season": season,
        "team": row.get("team"),
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "team_profile", row.get("team")])
    return {"id": uid, "text": text, "metadata": meta}

def _maybe_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return x

# --------- DISCOVERY & NORMALIZATION ---------
def normalize_feature_engineering_dir(dir_path: str) -> List[dict]:
    """
    single folder input
    auto detects files in folder, only ones needed
    """
    p = Path(dir_path)
    league = league_from_dir(p.name) or "UNKNOWN"

    # Helper: best file match with priority to JSON/JSONL
    def pick(patterns: List[str]) -> Optional[Path]:
        # Try JSON/JSONL first
        for pat in patterns:
            for ext in (".json", ".jsonl"):
                hit = list(p.glob(pat + ext))
                if hit: return hit[0]
        # Fallback to CSV
        for pat in patterns:
            hit = list(p.glob(pat + ".csv"))
            if hit: return hit[0]
        return None

    f_player_eng = pick(["player_engineered_features_*"])
    f_player_prof = pick(["player_profiles_*"])
    f_team_eng   = pick(["team_engineered_features_*"])
    f_team_prof  = pick(["team_profiles_*"])

    # Derive a season hint from any filename that has a year token
    season_hint = season_from_any(
        f_player_eng.name if f_player_eng else None,
        f_player_prof.name if f_player_prof else None,
        f_team_eng.name if f_team_eng else None,
        f_team_prof.name if f_team_prof else None,
    )

    docs: List[dict] = []

    # Player engineered
    if f_player_eng:
        rows = read_json(str(f_player_eng)) if f_player_eng.suffix == ".json" else read_csv_rows(str(f_player_eng))
        for r in rows:
            docs.append(doc_from_player_engineered(r, league=league, season=season_hint or "UNKNOWN", source_file=str(f_player_eng)))

    # Player profiles
    if f_player_prof:
        if f_player_prof.suffix == ".jsonl":
            iterator = read_jsonl(str(f_player_prof))
        elif f_player_prof.suffix == ".json":
            iterator = read_json(str(f_player_prof))
        else:
            iterator = read_csv_rows(str(f_player_prof))
        for r in iterator:
            docs.append(doc_from_player_profile(r, season_hint=season_hint, source_file=str(f_player_prof)))

    # Team engineered
    if f_team_eng:
        rows = read_json(str(f_team_eng)) if f_team_eng.suffix == ".json" else read_csv_rows(str(f_team_eng))
        for r in rows:
            docs.append(doc_from_team_engineered(r, league=league, season=season_hint or "UNKNOWN", source_file=str(f_team_eng)))

    # Team profiles
    if f_team_prof:
        if f_team_prof.suffix == ".jsonl":
            iterator = read_jsonl(str(f_team_prof))
        elif f_team_prof.suffix == ".json":
            iterator = read_json(str(f_team_prof))
        else:
            iterator = read_csv_rows(str(f_team_prof))
        for r in iterator:
            docs.append(doc_from_team_profile(r, season_hint=season_hint, source_file=str(f_team_prof)))

    return docs

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    base = Path(OUTPUT_BASE)

    # find all *_feature_engineering folders
    # Change to fe_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.lower().endswith("_feature_engineering")]
    fe_dirs = [Path(OUTPUT_BASE)]
    total = 0

    for d in sorted(fe_dirs):
        league = league_from_dir(d.name) or "UNKNOWN"
        docs = normalize_feature_engineering_dir(str(d))
        # filename token for season (safe for filenames)
        season_token = "unknown"
        if docs:
            # try to fetch first season from metas
            s = next((x["metadata"].get("season") for x in docs if x["metadata"].get("season")), None)
            season_token = s.replace("/", "_") if isinstance(s, str) else "unknown"
        out_path = Path(INDEX_DIR) / f"normalized_{league}_{season_token}.json"
        with open(out_path, "w") as f:
            json.dump(docs, f, ensure_ascii=False)
        print(f"[{d.name}] -> {len(docs)} docs -> {out_path}")
        total += len(docs)

    print(f"Total docs: {total}")

if __name__ == "__main__":
    main()
