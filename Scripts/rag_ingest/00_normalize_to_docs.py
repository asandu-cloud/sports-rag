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

def write_variable_dictionary(out_dir: str) -> str:
    """
    Writes a small glossary the RAG can retrieve when unsure about variable meanings.
    Returns the file path.
    """
    txt = """VARIABLE DICTIONARY (Top-5 Leagues)

corners_for (team fixture): Team's corners won in that fixture.
corners_against (team fixture): Opponent's corners won in the same fixture (opponent row join).
corners_pm (team profile): Average corners won per match (season).
corners_against_pm (team profile): Average corners conceded per match (computed from fixture rows).
corner_edge_pm (team profile): corners_pm - corners_against_pm. Positive = team typically wins more corners than it concedes.

shots_for / sot_for (team fixture): Shots / shots on target by the team in that fixture.
shots_against / sot_against (team fixture): Shots / SOT by the opponent in that fixture.

control_index: Possession + pass control composite index (higher = more control).
dominance_index: Territory + chance quality composite (higher = more dominance).

cards_per_90 (player): Player’s cards per 90 minutes.
fouls_per_90 (player): Player’s fouls committed per 90.
aggression_index_norm: Normalized aggression proxy (fouls, duels, cards).
expected_card_risk / expected_foul_pressure: Model-derived estimates for next match (0–1).

Notes:
- *_pm means per-match season averages.
- Fixture-level 'Against' values are joined from the opponent's row for the same fixture.
"""
    path = Path(out_dir) / "variable_dictionary.txt"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    return str(path)

def dictionary_to_doc(path: str) -> dict:
    text = Path(path).read_text()
    meta = {
        "entity_type": "guide",
        "doc_type": "data_dictionary",
        "league": "GLOBAL",
        "season": "ALL",
        "source_file": path,
    }
    uid = make_doc_id(["GLOBAL", "ALL", "data_dictionary", "v1"])
    return {"id": uid, "text": "GLOBAL | Data Dictionary\n" + text, "metadata": meta}


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

from typing import Optional

def doc_from_team_engineered(row: dict, league: str, season: str, source_file: str, opp_row: Optional[dict] = None) -> dict:
    home_away, opponent = infer_home_away(row.get("fixture", ""), row.get("team", ""))

    # Team's own stats
    shots_for        = row.get("shots_total")
    sot_for          = row.get("shots_on")
    corners_for      = row.get("corners")
    xg_for           = row.get("expected_goals")
    possession_for   = row.get("possession")
    fouls_for        = row.get("fouls_committed")
    yc_for           = row.get("yellow_cards")
    rc_for           = row.get("red_cards")

    # Opponent-derived stats
    # We look up the opponent's row for the same fixture (precomputed in normalize_feature_engineering_dir)
    shots_against    = (opp_row or {}).get("shots_total")
    sot_against      = (opp_row or {}).get("shots_on")
    corners_against  = (opp_row or {}).get("corners")

    # Extra engineered metrics we now trust
    dominance_idx    = row.get("dominance_index")
    control_idx      = row.get("control_index")
    conv_rate        = row.get("goal_conversion_rate")
    sot_ratio        = row.get("shot_on_target_ratio")
    cards_per_90     = row.get("cards_per_90_team")
    fouls_per_90     = row.get("fouls_per_90_team")
    form_idx_team    = row.get("form_index_team")
    aggression_norm  = row.get("aggression_index_norm")

    # New match metadata we want to expose to the RAG so it stops hallucinating scores/timing
    fixture_date     = row.get("fixture_date")
    final_score_str  = row.get("final_score_string")

    text = (
        f"{league} {season} | Team Fixture\n"
        f"Date: {fixture_date} | Fixture: {row.get('fixture')} | Final score: {final_score_str}\n"
        f"Team: {row.get('team')} ({home_away or 'n/a'}) vs {opponent or 'n/a'}\n"
        f"G:{row.get('goals')} A:{row.get('assists')} | xG:{xg_for} | Possession:{possession_for}\n"
        f"Shots For:{shots_for} (On:{sot_for}) | Shots Against:{shots_against} (On:{sot_against})\n"
        f"Corners For:{corners_for} | Corners Against:{corners_against}\n"
        f"Fouls:{fouls_for} YC:{yc_for} RC:{rc_for}\n"
        f"Team Rates → Conv%:{conv_rate} SoT%:{sot_ratio} ControlIdx:{control_idx} "
        f"DomIdx:{dominance_idx} FormIdx:{form_idx_team} AggIdxNorm:{aggression_norm}\n"
        f"Discipline → Fouls/90:{fouls_per_90} Cards/90:{cards_per_90}"
    )

    meta = {
        "entity_type": "team",
        "doc_type": "team_fixture",
        "league": league,
        "season": season,
        "fixture": row.get("fixture"),
        "fixture_date": fixture_date,
        "final_score": final_score_str,
        "team": row.get("team"),
        "opponent": opponent,
        "home_away": home_away,

        # Useful numeric cues for retrieval filters / analytics
        "shots_for": shots_for,
        "shots_against": shots_against,
        "sot_for": sot_for,
        "sot_against": sot_against,
        "corners_for": corners_for,
        "corners_against": corners_against,
        "possession": possession_for,
        "xg_for": xg_for,

        # aggression / discipline for betting props
        "cards_per_90_team": cards_per_90,
        "fouls_per_90_team": fouls_per_90,
        "aggression_index_norm": aggression_norm,

        "control_index": control_idx,
        "form_index_team": form_idx_team,

        "source_file": source_file,
    }

    uid = make_doc_id([league, season, "team_fixture", row.get("team"), row.get("fixture")])
    return {"id": uid, "text": text, "metadata": meta}



def doc_from_team_profile(row: dict, season_hint: Optional[str], source_file: str, c_against_pm_map: Optional[dict] = None) -> dict:
    league = row.get("league")
    season = row.get("season") or season_hint
    team   = row.get("team")
    tkey   = (team or "").lower().strip()
    c_against_pm = (c_against_pm_map or {}).get(tkey)

    corner_edge_pm = None
    try:
        if c_against_pm is not None and row.get("corners_pm") is not None:
            corner_edge_pm = float(row["corners_pm"]) - float(c_against_pm)
    except Exception:
        pass

    text = (
        f"{league} {season} | Team Profile\n"
        f"{team} | MP:{row.get('matches_played')} G:{row.get('goals')} A:{row.get('assists')}\n"
        f"G/Match:{row.get('goals_for_pm')} xG/Match:{row.get('expected_goals')} Poss:{row.get('possession')}\n"
        f"Corners For/Match:{row.get('corners_pm')} | Corners Against/Match:{c_against_pm}\n"
        f"Corner Edge/Match (For - Against):{corner_edge_pm}\n"
        f"Dominance:{row.get('dominance_index')} Control:{row.get('control_index')} Archetype:{row.get('archetype')}\n"
        f"Fouls/90:{row.get('fouls_per_90_team')} Cards/90:{row.get('cards_per_90_team')} Cards/Foul:{row.get('cards_per_foul_team')} Cards/Match:{row.get('cards_pm')}\n"
        f"Tags: {', '.join(row.get('style_tags', []))}\n"
        f"Summary: {row.get('summary_nl')}"
    )
    meta = {
        "entity_type": "team", "doc_type": "team_profile",
        "league": league, "season": season,
        "team": team,
        "corners_against_pm": c_against_pm,
        "corner_edge_pm": corner_edge_pm,
        "control_index": row.get("control_index"),
        "dominance_index": row.get("dominance_index"),
        "aggression_index_norm": row.get("aggression_index_norm"),
        "form_index_team": row.get("form_index_team"),
        "goals_for_pm": row.get("goals_for_pm"),
        "goals_against_pm": row.get("shots_against_pm"),  # proxy if no explicit GA
        "corners_pm": row.get("corners_pm"),
        "cards_per_90_team": row.get("cards_per_90_team"),
        "fouls_per_90_team": row.get("fouls_per_90_team"),
        "possession": row.get("possession"),
        "archetype": row.get("archetype"),
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "team_profile", team])
    return {"id": uid, "text": text, "metadata": meta}


def _maybe_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return x
    
# --- new helper at top-level, near other utils ---
def build_fixture_team_map(team_rows: list[dict]) -> dict:
    """
    Returns: { (fixture_lower): { team_lower: row, ... }, ... }
    Lets us find opponent's row quickly to compute corners_against, etc.
    """
    fx_map = {}
    for r in team_rows:
        fx = (r.get("fixture") or "").lower().strip()
        tm = (r.get("team") or "").lower().strip()
        if fx and tm:
            fx_map.setdefault(fx, {})[tm] = r
    return fx_map

# --- aggregate season-level conceded corners per team from engineered rows ---
def compute_corners_against_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: corners_against_per_match }
    Requires opponent join: we look up opponent's row corners for each fixture.
    """
    fx_map = build_fixture_team_map(team_rows)
    sum_cagainst = {}
    cnt_matches  = {}

    for r in team_rows:
        fx = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team: 
            continue
        # find opponent row, grab their 'corners'
        opp_row = None
        hm, opp_name = infer_home_away(r.get("fixture",""), r.get("team",""))
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())
        c_against = (opp_row or {}).get("corners")
        try:
            c_against = float(c_against) if c_against is not None else None
        except:
            c_against = None

        if c_against is not None:
            sum_cagainst[team] = sum_cagainst.get(team, 0.0) + c_against
            cnt_matches[team]  = cnt_matches.get(team, 0) + 1

    out = {}
    for t, s in sum_cagainst.items():
        n = max(1, cnt_matches.get(t, 1))
        out[t] = s / n
    return out


# --------- DISCOVERY & NORMALIZATION ---------
def normalize_feature_engineering_dir(dir_path: str) -> List[dict]:
    """
    Normalize a single *_feature_engineering folder into doclets.
    - Auto-detects the four input files (JSON/JSONL preferred; CSV fallback).
    - Infers league from folder name, season from filename tokens (e.g., _2025 -> 2024/25).
    - Adds opponent-joined fields for team fixtures (e.g., corners_against).
    - Computes corners_against_pm from engineered rows and injects into team profiles.
    - Appends a GLOBAL data-dictionary doc at the end.
    """
    p = Path(dir_path)
    league = league_from_dir(p.name) or "UNKNOWN"
    docs: List[dict] = []

    # Helper: best file match with priority to JSON/JSONL
    def pick(patterns: List[str]) -> Optional[Path]:
        # Try JSON/JSONL first
        for pat in patterns:
            for ext in (".json", ".jsonl"):
                hit = list(p.glob(pat + ext))
                if hit:
                    return hit[0]
        # Fallback to CSV
        for pat in patterns:
            hit = list(p.glob(pat + ".csv"))
            if hit:
                return hit[0]
        return None

    # Discover files
    f_player_eng = pick(["player_engineered_features_*"])
    f_player_prof = pick(["player_profiles_*"])
    f_team_eng   = pick(["team_engineered_features_*"])
    f_team_prof  = pick(["team_profiles_*"])

    # Derive a season hint from any filename that has a year token (MUST be before using season_hint)
    season_hint = season_from_any(
        f_player_eng.name if f_player_eng else None,
        f_player_prof.name if f_player_prof else None,
        f_team_eng.name if f_team_eng else None,
        f_team_prof.name if f_team_prof else None,
    ) or "UNKNOWN"

    # -----------------------------
    # TEAM ENGINEERED (load first)
    # -----------------------------
    team_rows: List[dict] = []
    if f_team_eng:
        team_rows = read_json(str(f_team_eng)) if f_team_eng.suffix == ".json" else read_csv_rows(str(f_team_eng))

    # Opponent-join map and conceded-corners per match (season-level) from engineered rows
    c_against_pm_map = compute_corners_against_pm(team_rows) if team_rows else {}

    # Emit team fixture docs with opponent join (corners_against, shots_against, etc.)
    if team_rows:
        fx_map = build_fixture_team_map(team_rows)
        for r in team_rows:
            fx = (r.get("fixture") or "").lower().strip()
            opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))[1]
            opp_row = fx_map.get(fx, {}).get((opp_name or "").lower().strip()) if opp_name else None
            docs.append(
                doc_from_team_engineered(
                    r,
                    league=league,
                    season=season_hint,
                    source_file=str(f_team_eng),
                    opp_row=opp_row,
                )
            )

    # -----------------------------
    # PLAYER ENGINEERED
    # -----------------------------
    if f_player_eng:
        rows = read_json(str(f_player_eng)) if f_player_eng.suffix == ".json" else read_csv_rows(str(f_player_eng))
        for r in rows:
            docs.append(
                doc_from_player_engineered(
                    r,
                    league=league,
                    season=season_hint,
                    source_file=str(f_player_eng),
                )
            )

    # -----------------------------
    # PLAYER PROFILES
    # -----------------------------
    if f_player_prof:
        if f_player_prof.suffix == ".jsonl":
            iterator = read_jsonl(str(f_player_prof))
        elif f_player_prof.suffix == ".json":
            iterator = read_json(str(f_player_prof))
        else:
            iterator = read_csv_rows(str(f_player_prof))
        for r in iterator:
            docs.append(
                doc_from_player_profile(
                    r,
                    season_hint=season_hint,
                    source_file=str(f_player_prof),
                )
            )

    # -----------------------------
    # TEAM PROFILES (inject corners_against_pm + corner_edge_pm)
    # -----------------------------
    if f_team_prof:
        if f_team_prof.suffix == ".jsonl":
            iterator = read_jsonl(str(f_team_prof))
        elif f_team_prof.suffix == ".json":
            iterator = read_json(str(f_team_prof))
        else:
            iterator = read_csv_rows(str(f_team_prof))
        for r in iterator:
            docs.append(
                doc_from_team_profile(
                    r,
                    season_hint=season_hint,
                    source_file=str(f_team_prof),
                    c_against_pm_map=c_against_pm_map,
                )
            )

    # -----------------------------
    # GLOBAL data dictionary (single doc)
    # -----------------------------
    dict_path = write_variable_dictionary(INDEX_DIR)  # ensure INDEX_DIR is defined globally
    docs.append(dictionary_to_doc(dict_path))

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
