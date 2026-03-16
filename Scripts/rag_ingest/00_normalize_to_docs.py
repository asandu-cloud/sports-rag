import argparse, os, re, json, csv, hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Optional

# --------- CONFIG ---------
ROOT        = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = str(ROOT / "Output")           # parent of all *_feature_engineering folders
INDEX_DIR   = str(ROOT / "Index")            # where normalized_*.json will be written

# Map folder name prefixes to canonical league names
LEAGUE_FROM_DIR = {
    "prem": "EPL",
    "laliga": "LaLiga",
    "ligue1": "Ligue1",
    "seriaa": "SerieA",      # tolerate your folder spelling
    "seriea": "SerieA",
    "bundesliga": "Bundesliga",
    "ucl": "UCL",
    "uel": "UEL",
    "uecl": "UECL",
}

# --------- UTILS ---------
def make_doc_id(parts: Iterable) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.md5(base.encode()).hexdigest()

def euro_season_from_year(year: int) -> str:
    """2025 -> '2025/26' (API-Football season=2025 means the 2025/26 campaign)"""
    return f"{year}/{str(year + 1)[-2:]}"

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

opp_cards_induced_pm (team profile): Average cards opponents receive per match when playing against this team. Higher = team's style provokes more opponent bookings.
corners_home_pm (team profile): Average corners won per match in HOME fixtures only.
corners_away_pm (team profile): Average corners won per match in AWAY fixtures only.
shots_for_pm (team profile): Average total shots per match (season).
sot_for_pm (team profile): Average shots on target per match (season).
cards_per_foul_team (team profile): Ratio of cards to fouls — card propensity per foul committed.
cards_home_pm (team profile): Average cards received per match in HOME fixtures only.
cards_away_pm (team profile): Average cards received per match in AWAY fixtures only.
sot_against_pm (team profile): Average shots on target conceded per match (season). Opponent-join aggregation.
corners_against_home_pm (team profile): Average corners conceded per match in HOME fixtures only. Opponent-join + venue-split.
corners_against_away_pm (team profile): Average corners conceded per match in AWAY fixtures only. Opponent-join + venue-split.
sot_against_home_pm (team profile): Average SoT conceded per match in HOME fixtures only. Opponent-join + venue-split.
sot_against_away_pm (team profile): Average SoT conceded per match in AWAY fixtures only. Opponent-join + venue-split.
cards_induced_home_pm (team profile): Average cards opponents receive per match in team's HOME fixtures. Opponent-join + venue-split.
cards_induced_away_pm (team profile): Average cards opponents receive per match in team's AWAY fixtures. Opponent-join + venue-split.
fouls_home_pm (team profile): Average fouls committed per match in HOME fixtures only.
fouls_away_pm (team profile): Average fouls committed per match in AWAY fixtures only.

Player Role & Minutes Context (player profile):
- appearances_total: Total appearances across all fixtures (any minutes > 0).
- appearances_as_starter: Count of appearances with >= 60 minutes played.
- start_rate: appearances_as_starter / appearances_total (0.0 to 1.0).
- avg_minutes_per_appearance: Total minutes / appearances_total.
- minutes_risk: Categorical risk label — "nailed_on" (SR>=0.80, avg>=75min), "likely_starter" (SR>=0.60, avg>=60min), "rotation" (SR>=0.30), "bench" (SR<0.30).
- recent_starts_last_5: How many of the team's last 5 fixtures the player started (>= 60 min).
- recent_role_trend: "starting" (4-5 of last 5), "benched" (0-1 of last 5), "mixed" (2-3 of last 5).
- last_5_minutes: List of minutes played in each of the team's last 5 fixtures (0 if player did not appear).
- starter_goals_per_90 / starter_sot_per_90 / etc.: Per-90 rates computed ONLY from appearances >= 60 min. Eliminates sub-appearance rate inflation.
- goals_per_90 / sot_per_90 / etc. (without starter_ prefix): Per-90 rates from ALL appearances (backward compatible, may be inflated by short subs).

Notes:
- *_pm means per-match season averages.
- Fixture-level 'Against' values are joined from the opponent's row for the same fixture.
- Starter-only per-90 rates are preferred for player prop analysis; all-appearances rates are kept for backward compatibility.
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
        "minutes": _maybe_int(row.get("minutes")),
        "source_file": source_file,
    }
    uid = make_doc_id([league, season, "player_fixture", meta["player_id"], row.get("fixture")])
    return {"id": uid, "text": text, "metadata": meta}

def doc_from_player_profile(row: dict, season_hint: Optional[str], source_file: str) -> dict:
    league = row.get("league")
    season = row.get("season") or season_hint

    # Style tags may be a list or a string; normalize to list
    style_tags = row.get("style_tags", [])
    if isinstance(style_tags, str):
        style_tags = [t.strip() for t in style_tags.split(",") if t.strip()]

    # Role / minutes context lines (Fix 1, 4, 5 — graceful when fields absent)
    apps_total = row.get("appearances_total")
    apps_starter = row.get("appearances_as_starter")
    start_rate = row.get("start_rate")
    avg_min = row.get("avg_minutes_per_appearance")
    minutes_risk = row.get("minutes_risk")
    recent_starts = row.get("recent_starts_last_5")
    recent_trend = row.get("recent_role_trend")

    role_line = ""
    if apps_total is not None:
        role_line = (
            f"Role: {minutes_risk or 'unknown'} | "
            f"Apps:{apps_total} Starts:{apps_starter} StartRate:{start_rate} AvgMin:{avg_min} "
            f"RecentStarts(5):{recent_starts} Trend:{recent_trend}\n"
        )

    # Starter-only per-90 line (Fix 2 — graceful when fields absent)
    starter_line = ""
    sg90 = row.get("starter_goals_per_90")
    if sg90 is not None:
        starter_line = (
            f"StarterOnly G/90:{sg90} A/90:{row.get('starter_assists_per_90')} "
            f"Shots/90:{row.get('starter_shots_per_90')} SoT/90:{row.get('starter_sot_per_90')} "
            f"Fouls/90:{row.get('starter_fouls_per_90')} Cards/90:{row.get('starter_cards_per_90')}\n"
        )

    text = (
        f"{league} {season} | Player Profile\n"
        f"{row.get('name')} (id:{row.get('player_id')}, {row.get('team')}) – pos:{row.get('position')} rating:{row.get('rating')}\n"
        f"{role_line}"
        f"G/90:{row.get('goals_per_90')} A/90:{row.get('assists_per_90')} Shots/90:{row.get('shots_per_90')} SoT/90:{row.get('sot_per_90')} "
        f"Passes/90:{row.get('passes_per_90')} Acc%:{row.get('pass_accuracy_pct')}\n"
        f"{starter_line}"
        f"Tkl/90:{row.get('tackles_per_90')} Int/90:{row.get('interceptions_per_90')} Fouls/90:{row.get('fouls_per_90_calc')} "
        f"Cards/90:{row.get('cards_per_90_calc')} DuelWin%:{row.get('duel_win_pct')}\n"
        f"FormIdx:{row.get('form_index')} AggIdx:{row.get('aggression_index_norm')} CtrlIdx:{row.get('control_index')} "
        f"CardRiskExp:{row.get('expected_card_risk')} FoulPressureExp:{row.get('expected_foul_pressure')}\n"
        f"Tags: {', '.join(style_tags)}\n"
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
        # New metadata fields for downstream filtering (Fix 1, 4)
        "minutes_risk": minutes_risk,
        "start_rate": float(start_rate) if start_rate is not None else None,
        "appearances_total": _maybe_int(apps_total),
        "appearances_as_starter": _maybe_int(apps_starter),
        "avg_minutes_per_appearance": float(avg_min) if avg_min is not None else None,
        "recent_role_trend": recent_trend,
    }
    uid = make_doc_id([league, season, "player_profile", meta["player_id"]])
    return {"id": uid, "text": text, "metadata": meta}

from typing import Optional

def doc_from_team_engineered(row: dict, league: str, season: str, source_file: str, opp_row: Optional[dict] = None) -> dict:
    home_away, opponent = infer_home_away(row.get("fixture", ""), row.get("team", ""))

    # Team's own stats
    shots_for        = row.get("shots_total") or row.get("shots_total_x")
    sot_for          = row.get("shots_on") or row.get("shots_on_x")
    corners_for      = row.get("corners")
    xg_for           = row.get("expected_goals")
    possession_for   = row.get("possession")
    fouls_for        = row.get("fouls_committed")
    yc_for           = row.get("yellow_cards")
    rc_for           = row.get("red_cards")

    # Opponent-derived stats
    # We look up the opponent's row for the same fixture (precomputed in normalize_feature_engineering_dir)
    shots_against    = (opp_row or {}).get("shots_total") or (opp_row or {}).get("shots_total_x")
    sot_against      = (opp_row or {}).get("shots_on") or (opp_row or {}).get("shots_on_x")
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



def doc_from_team_profile(
    row: dict,
    season_hint: Optional[str],
    source_file: str,
    c_against_pm_map: Optional[dict] = None,
    cards_induced_pm_map: Optional[dict] = None,
    corners_ha_pm_map: Optional[dict] = None,
    sot_against_pm_map: Optional[dict] = None,
    cards_ha_pm_map: Optional[dict] = None,
    stat_var_map: Optional[dict] = None,
    goals_against_pm_map: Optional[dict] = None,
    goals_ha_pm_map: Optional[dict] = None,
    sot_ha_pm_map: Optional[dict] = None,
    xg_ha_pm_map: Optional[dict] = None,
    matches_played_map: Optional[dict] = None,
    corners_against_ha_pm_map: Optional[dict] = None,
    sot_against_ha_pm_map: Optional[dict] = None,
    cards_induced_ha_pm_map: Optional[dict] = None,
    fouls_ha_pm_map: Optional[dict] = None,
) -> dict:
    league = row.get("league")
    season = row.get("season") or season_hint
    team   = row.get("team")
    tkey   = (team or "").lower().strip()
    c_against_pm = (c_against_pm_map or {}).get(tkey)
    cards_induced_pm = (cards_induced_pm_map or {}).get(tkey)
    corners_ha = (corners_ha_pm_map or {}).get(tkey, {})
    corners_home_pm = corners_ha.get("home") if corners_ha else None
    corners_away_pm = corners_ha.get("away") if corners_ha else None
    sot_against_pm = (sot_against_pm_map or {}).get(tkey)
    goals_against_pm = (goals_against_pm_map or {}).get(tkey)
    cards_ha = (cards_ha_pm_map or {}).get(tkey, {})
    cards_home_pm = cards_ha.get("home") if cards_ha else None
    cards_away_pm = cards_ha.get("away") if cards_ha else None
    goals_ha = (goals_ha_pm_map or {}).get(tkey, {})
    goals_home_pm = goals_ha.get("home") if goals_ha else None
    goals_away_pm = goals_ha.get("away") if goals_ha else None
    sot_ha = (sot_ha_pm_map or {}).get(tkey, {})
    sot_home_pm = sot_ha.get("home") if sot_ha else None
    sot_away_pm = sot_ha.get("away") if sot_ha else None
    xg_ha = (xg_ha_pm_map or {}).get(tkey, {})
    xg_home_pm = xg_ha.get("home") if xg_ha else None
    xg_away_pm = xg_ha.get("away") if xg_ha else None
    stat_vars = (stat_var_map or {}).get(tkey, {})
    corners_var = stat_vars.get("corners_var")
    goals_var = stat_vars.get("goals_var")
    cards_var = stat_vars.get("cards_var")
    sot_var = stat_vars.get("sot_var")
    matches_played = (matches_played_map or {}).get(tkey, 0)
    corners_against_ha = (corners_against_ha_pm_map or {}).get(tkey, {})
    corners_against_home_pm = corners_against_ha.get("home") if corners_against_ha else None
    corners_against_away_pm = corners_against_ha.get("away") if corners_against_ha else None
    sot_against_ha = (sot_against_ha_pm_map or {}).get(tkey, {})
    sot_against_home_pm = sot_against_ha.get("home") if sot_against_ha else None
    sot_against_away_pm = sot_against_ha.get("away") if sot_against_ha else None
    cards_induced_ha = (cards_induced_ha_pm_map or {}).get(tkey, {})
    cards_induced_home_pm = cards_induced_ha.get("home") if cards_induced_ha else None
    cards_induced_away_pm = cards_induced_ha.get("away") if cards_induced_ha else None
    fouls_ha = (fouls_ha_pm_map or {}).get(tkey, {})
    fouls_home_pm = fouls_ha.get("home") if fouls_ha else None
    fouls_away_pm = fouls_ha.get("away") if fouls_ha else None

    corner_edge_pm = None
    try:
        if c_against_pm is not None and row.get("corners_pm") is not None:
            corner_edge_pm = float(row["corners_pm"]) - float(c_against_pm)
    except Exception:
        pass

    text = (
        f"{league} {season} | Team Profile\n"
        f"{team} | MP:{matches_played} G:{row.get('goals')} A:{row.get('assists')}\n"
        f"G/Match:{row.get('goals_for_pm')} xG/Match:{row.get('expected_goals')} Poss:{row.get('possession')}\n"
        f"Goals Against/Match:{goals_against_pm}\n"
        f"Shots For/Match:{row.get('shots_for_pm')} | SoT For/Match:{row.get('sot_for_pm')} | SoT Against/Match:{sot_against_pm}\n"
        f"Corners For/Match:{row.get('corners_pm')} | Corners Against/Match:{c_against_pm}\n"
        f"Goals Home/Match:{goals_home_pm} | Goals Away/Match:{goals_away_pm}\n"
        f"xG Home/Match:{xg_home_pm} | xG Away/Match:{xg_away_pm}\n"
        f"SoT Home/Match:{sot_home_pm} | SoT Away/Match:{sot_away_pm}\n"
        f"Corners Home/Match:{corners_home_pm} | Corners Away/Match:{corners_away_pm}\n"
        f"Corner Edge/Match (For - Against):{corner_edge_pm}\n"
        f"Dominance:{row.get('dominance_index')} Control:{row.get('control_index')} Archetype:{row.get('archetype')}\n"
        f"Fouls/90:{row.get('fouls_per_90_team')} Cards/90:{row.get('cards_per_90_team')} Cards/Foul:{row.get('cards_per_foul_team')} Cards/Match:{row.get('cards_pm')}\n"
        f"Cards Home/Match:{cards_home_pm} | Cards Away/Match:{cards_away_pm}\n"
        f"Opp Cards Induced/Match:{cards_induced_pm}\n"
        f"Corners Against Home/Match:{corners_against_home_pm} | Corners Against Away/Match:{corners_against_away_pm}\n"
        f"SoT Against Home/Match:{sot_against_home_pm} | SoT Against Away/Match:{sot_against_away_pm}\n"
        f"Cards Induced Home/Match:{cards_induced_home_pm} | Cards Induced Away/Match:{cards_induced_away_pm}\n"
        f"Fouls Home/Match:{fouls_home_pm} | Fouls Away/Match:{fouls_away_pm}\n"
        f"Tags: {', '.join(row.get('style_tags', []))}\n"
        f"Summary: {row.get('summary_nl')}"
    )
    meta = {
        "entity_type": "team", "doc_type": "team_profile",
        "league": league, "season": season,
        "team": team, "matches_played": matches_played,
        "corners_against_pm": c_against_pm,
        "corner_edge_pm": corner_edge_pm,
        "control_index": row.get("control_index"),
        "dominance_index": row.get("dominance_index"),
        "aggression_index_norm": row.get("aggression_index_norm"),
        "form_index_team": row.get("form_index_team"),
        "goals_for_pm": row.get("goals_for_pm"),
        "goals_against_pm": goals_against_pm,  # actual goals conceded (opponent-join)
        "corners_pm": row.get("corners_pm"),
        "goals_home_pm": goals_home_pm,
        "goals_away_pm": goals_away_pm,
        "xg_home_pm": xg_home_pm,
        "xg_away_pm": xg_away_pm,
        "sot_home_pm": sot_home_pm,
        "sot_away_pm": sot_away_pm,
        "corners_home_pm": corners_home_pm,
        "corners_away_pm": corners_away_pm,
        "cards_per_90_team": row.get("cards_per_90_team"),
        "fouls_per_90_team": row.get("fouls_per_90_team"),
        "cards_per_foul_team": row.get("cards_per_foul_team"),
        "opp_cards_induced_pm": cards_induced_pm,
        "cards_home_pm": cards_home_pm,
        "cards_away_pm": cards_away_pm,
        "shots_for_pm": row.get("shots_for_pm"),
        "sot_for_pm": row.get("sot_for_pm"),
        "sot_against_pm": sot_against_pm,
        "possession": row.get("possession"),
        "archetype": row.get("archetype"),
        "corners_var": corners_var,
        "goals_var": goals_var,
        "cards_var": cards_var,
        "sot_var": sot_var,
        "corners_against_home_pm": corners_against_home_pm,
        "corners_against_away_pm": corners_against_away_pm,
        "sot_against_home_pm": sot_against_home_pm,
        "sot_against_away_pm": sot_against_away_pm,
        "cards_induced_home_pm": cards_induced_home_pm,
        "cards_induced_away_pm": cards_induced_away_pm,
        "fouls_home_pm": fouls_home_pm,
        "fouls_away_pm": fouls_away_pm,
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


# --- aggregate opponent-induced cards per match (season-level) ---
def compute_cards_induced_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: avg_cards_opponents_receive_per_match }.
    For each of team X's fixtures, look up the opponent's row and grab
    their cards_total.  Average across all matches gives X's
    'card-inducing' rate (how many cards opponents pick up against X).
    """
    fx_map = build_fixture_team_map(team_rows)
    sum_opp_cards: dict[str, float] = {}
    cnt_matches:   dict[str, int]   = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())

        opp_cards = None
        if opp_row:
            opp_cards = opp_row.get("cards_total")
            if opp_cards is None:
                yc = opp_row.get("yellow_cards")
                rc = opp_row.get("red_cards")
                if yc is not None or rc is not None:
                    opp_cards = float(yc or 0) + float(rc or 0)
        try:
            opp_cards = float(opp_cards) if opp_cards is not None else None
        except (TypeError, ValueError):
            opp_cards = None

        if opp_cards is not None:
            sum_opp_cards[team] = sum_opp_cards.get(team, 0.0) + opp_cards
            cnt_matches[team]   = cnt_matches.get(team, 0) + 1

    out = {}
    for t, s in sum_opp_cards.items():
        n = max(1, cnt_matches.get(t, 1))
        out[t] = s / n
    return out


# --- aggregate corners per match split by home / away venue ---
def compute_corners_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_corners_home, "away": avg_corners_away} }.
    Uses 'home_team' field when available, falls back to infer_home_away().
    """
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        team = (r.get("team") or "").lower().strip()
        if not team:
            continue
        # Determine venue: prefer explicit home_team, fall back to fixture parsing
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")

        corners = r.get("corners")
        try:
            corners = float(corners) if corners is not None else None
        except (TypeError, ValueError):
            corners = None
        if corners is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + corners
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + corners
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    all_teams = set(home_sums.keys()) | set(away_sums.keys())
    for t in all_teams:
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


# --- aggregate cards per match split by home / away venue ---
def compute_cards_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_cards_home, "away": avg_cards_away} }.
    Uses 'home_team' field when available, falls back to infer_home_away().
    """
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        team = (r.get("team") or "").lower().strip()
        if not team:
            continue
        # Determine venue: prefer explicit home_team, fall back to fixture parsing
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")

        # Card total — same fallback as compute_cards_induced_pm
        cards = r.get("cards_total")
        if cards is None:
            yc = r.get("yellow_cards")
            rc = r.get("red_cards")
            if yc is not None or rc is not None:
                cards = float(yc or 0) + float(rc or 0)
        try:
            cards = float(cards) if cards is not None else None
        except (TypeError, ValueError):
            cards = None
        if cards is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + cards
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + cards
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    all_teams = set(home_sums.keys()) | set(away_sums.keys())
    for t in all_teams:
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


def _compute_venue_split(team_rows: list[dict], stat_field: str) -> dict:
    """Generic venue split: returns { team_lower: {"home": avg_home, "away": avg_away} }."""
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        team = (r.get("team") or "").lower().strip()
        if not team:
            continue
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")

        val = r.get(stat_field)
        try:
            val = float(val) if val is not None else None
        except (TypeError, ValueError):
            val = None
        if val is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + val
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + val
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    for t in set(home_sums.keys()) | set(away_sums.keys()):
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


def compute_goals_home_away_pm(team_rows: list[dict]) -> dict:
    """Returns { team_lower: {"home": avg_goals_home, "away": avg_goals_away} }."""
    return _compute_venue_split(team_rows, "goals")


def compute_sot_home_away_pm(team_rows: list[dict]) -> dict:
    """Returns { team_lower: {"home": avg_sot_home, "away": avg_sot_away} }."""
    # Some leagues store shots_on=0 but shots_on_x has real values (pandas merge artifact).
    # Prefer shots_on_x (always real when present), fall back to shots_on.
    result = _compute_venue_split(team_rows, "shots_on_x")
    if not result:
        result = _compute_venue_split(team_rows, "shots_on")
    return result


def compute_xg_home_away_pm(team_rows: list[dict]) -> dict:
    """Returns { team_lower: {"home": avg_xg_home, "away": avg_xg_away} }."""
    return _compute_venue_split(team_rows, "expected_goals")


# --- per-team stat variance (season-level) for probabilistic modeling ---
def compute_stat_variance(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {corners_var, goals_var, cards_var, sot_var} }.
    Computes sample variance from per-match fixture data.
    """
    from collections import defaultdict
    team_vals: dict[str, dict[str, list]] = defaultdict(
        lambda: {"corners": [], "goals": [], "cards": [], "sot": []}
    )
    for r in team_rows:
        team = (r.get("team") or "").lower().strip()
        if not team:
            continue
        corners = r.get("corners")
        goals = r.get("goals")
        sot = r.get("shots_on") or r.get("shots_on_x")
        cards = r.get("cards_total")
        if cards is None:
            yc = r.get("yellow_cards")
            rc = r.get("red_cards")
            if yc is not None or rc is not None:
                cards = float(yc or 0) + float(rc or 0)
        try:
            if corners is not None:
                team_vals[team]["corners"].append(float(corners))
            if goals is not None:
                team_vals[team]["goals"].append(float(goals))
            if sot is not None:
                team_vals[team]["sot"].append(float(sot))
            if cards is not None:
                team_vals[team]["cards"].append(float(cards))
        except (TypeError, ValueError):
            pass

    out: dict[str, dict] = {}
    for team, vals in team_vals.items():
        entry: dict[str, float | None] = {}
        for stat, data in vals.items():
            if len(data) >= 5:
                mean = sum(data) / len(data)
                entry[f"{stat}_var"] = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            else:
                entry[f"{stat}_var"] = None
        out[team] = entry
    return out


# --- aggregate goals conceded per match (season-level, opponent-join) ---
def compute_goals_against_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: avg_goals_conceded_per_match }.
    For each of team X's fixtures, look up the opponent's goals scored
    and average across all matches.
    """
    fx_map = build_fixture_team_map(team_rows)
    sum_ga: dict[str, float] = {}
    cnt: dict[str, int] = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())

        opp_goals = None
        if opp_row:
            v = opp_row.get("goals")
            if v is not None:
                try:
                    opp_goals = float(v)
                except (TypeError, ValueError):
                    opp_goals = None

        if opp_goals is not None:
            sum_ga[team] = sum_ga.get(team, 0.0) + opp_goals
            cnt[team] = cnt.get(team, 0) + 1

    return {t: sum_ga[t] / cnt[t] for t in sum_ga if cnt.get(t, 0) > 0}


# --- aggregate opponent SoT conceded per match (season-level) ---
def compute_sot_against_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: avg_sot_conceded_per_match }.
    For each of team X's fixtures, look up the opponent's SoT (shots on target)
    and average across all matches.
    """
    fx_map = build_fixture_team_map(team_rows)
    sum_sot: dict[str, float] = {}
    cnt_matches: dict[str, int] = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())

        opp_sot = None
        if opp_row:
            for _k in ("sot_for", "shots_on_x", "shots_on"):
                v = opp_row.get(_k)
                if v is not None:
                    opp_sot = v
                    break
        try:
            opp_sot = float(opp_sot) if opp_sot is not None else None
        except (TypeError, ValueError):
            opp_sot = None

        if opp_sot is not None:
            sum_sot[team] = sum_sot.get(team, 0.0) + opp_sot
            cnt_matches[team] = cnt_matches.get(team, 0) + 1

    out = {}
    for t, s in sum_sot.items():
        n = max(1, cnt_matches.get(t, 1))
        out[t] = s / n
    return out


# --- venue-split corners conceded per match (opponent-join + home/away) ---
def compute_corners_against_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_corners_conceded_home, "away": avg_corners_conceded_away} }.
    For each of team X's fixtures, look up the opponent's corners won,
    then split by whether X was home or away.
    """
    fx_map = build_fixture_team_map(team_rows)
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        # Determine venue
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")
        # Find opponent's corners
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())
        c_against = (opp_row or {}).get("corners")
        try:
            c_against = float(c_against) if c_against is not None else None
        except (TypeError, ValueError):
            c_against = None
        if c_against is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + c_against
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + c_against
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    for t in set(home_sums.keys()) | set(away_sums.keys()):
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


# --- venue-split SoT conceded per match (opponent-join + home/away) ---
def compute_sot_against_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_sot_conceded_home, "away": avg_sot_conceded_away} }.
    For each of team X's fixtures, look up the opponent's SoT,
    then split by whether X was home or away.
    """
    fx_map = build_fixture_team_map(team_rows)
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        # Determine venue
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")
        # Find opponent's SoT
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())
        opp_sot = None
        if opp_row:
            for _k in ("sot_for", "shots_on_x", "shots_on"):
                v = opp_row.get(_k)
                if v is not None:
                    opp_sot = v
                    break
        try:
            opp_sot = float(opp_sot) if opp_sot is not None else None
        except (TypeError, ValueError):
            opp_sot = None
        if opp_sot is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + opp_sot
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + opp_sot
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    for t in set(home_sums.keys()) | set(away_sums.keys()):
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


# --- venue-split opponent cards induced per match (opponent-join + home/away) ---
def compute_cards_induced_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_cards_induced_home, "away": avg_cards_induced_away} }.
    For each of team X's fixtures, look up the opponent's cards_total,
    then split by whether X was home or away.
    """
    fx_map = build_fixture_team_map(team_rows)
    home_sums: dict[str, float] = {}
    home_cnts: dict[str, int]   = {}
    away_sums: dict[str, float] = {}
    away_cnts: dict[str, int]   = {}

    for r in team_rows:
        fx   = (r.get("fixture") or "").lower().strip()
        team = (r.get("team") or "").lower().strip()
        if not fx or not team:
            continue
        # Determine venue
        home_team_raw = (r.get("home_team") or "").lower().strip()
        if home_team_raw:
            is_home = (team == home_team_raw)
        else:
            venue, _ = infer_home_away(r.get("fixture", ""), r.get("team", ""))
            is_home = (venue == "home")
        # Find opponent's cards
        _hm, opp_name = infer_home_away(r.get("fixture", ""), r.get("team", ""))
        opp_row = None
        if opp_name:
            opp_row = fx_map.get(fx, {}).get(opp_name.lower().strip())
        opp_cards = None
        if opp_row:
            opp_cards = opp_row.get("cards_total")
            if opp_cards is None:
                yc = opp_row.get("yellow_cards")
                rc = opp_row.get("red_cards")
                if yc is not None or rc is not None:
                    opp_cards = float(yc or 0) + float(rc or 0)
        try:
            opp_cards = float(opp_cards) if opp_cards is not None else None
        except (TypeError, ValueError):
            opp_cards = None
        if opp_cards is None:
            continue

        if is_home:
            home_sums[team] = home_sums.get(team, 0.0) + opp_cards
            home_cnts[team] = home_cnts.get(team, 0) + 1
        else:
            away_sums[team] = away_sums.get(team, 0.0) + opp_cards
            away_cnts[team] = away_cnts.get(team, 0) + 1

    out = {}
    for t in set(home_sums.keys()) | set(away_sums.keys()):
        h_avg = home_sums[t] / max(1, home_cnts[t]) if t in home_sums else None
        a_avg = away_sums[t] / max(1, away_cnts[t]) if t in away_sums else None
        out[t] = {"home": h_avg, "away": a_avg}
    return out


# --- venue-split own fouls committed per match ---
def compute_fouls_home_away_pm(team_rows: list[dict]) -> dict:
    """
    Returns { team_lower: {"home": avg_fouls_home, "away": avg_fouls_away} }.
    Uses fouls_committed field from team fixture rows.
    """
    return _compute_venue_split(team_rows, "fouls_committed")


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
        # numeric cleanup for key stats
        num_keys = [
            "yellow_cards", "red_cards", "cards_per_90_team", "fouls_per_90_team",
            "corners", "expected_goals", "goals_prevented",
            "shots_on", "shots_total", "control_index", "form_index_team",
            "aggression_index_norm", "cards_total",
        ]
        for r in team_rows:
            for k in num_keys:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass

    # Opponent-join map and conceded-corners per match (season-level) from engineered rows
    c_against_pm_map = compute_corners_against_pm(team_rows) if team_rows else {}
    cards_induced_pm_map = compute_cards_induced_pm(team_rows) if team_rows else {}
    corners_ha_pm_map = compute_corners_home_away_pm(team_rows) if team_rows else {}
    sot_against_pm_map = compute_sot_against_pm(team_rows) if team_rows else {}
    cards_ha_pm_map = compute_cards_home_away_pm(team_rows) if team_rows else {}
    stat_var_map = compute_stat_variance(team_rows) if team_rows else {}
    goals_against_pm_map = compute_goals_against_pm(team_rows) if team_rows else {}
    goals_ha_pm_map = compute_goals_home_away_pm(team_rows) if team_rows else {}
    sot_ha_pm_map = compute_sot_home_away_pm(team_rows) if team_rows else {}
    xg_ha_pm_map = compute_xg_home_away_pm(team_rows) if team_rows else {}
    corners_against_ha_pm_map = compute_corners_against_home_away_pm(team_rows) if team_rows else {}
    sot_against_ha_pm_map = compute_sot_against_home_away_pm(team_rows) if team_rows else {}
    cards_induced_ha_pm_map = compute_cards_induced_home_away_pm(team_rows) if team_rows else {}
    fouls_ha_pm_map = compute_fouls_home_away_pm(team_rows) if team_rows else {}

    # Count fixtures per team (for matches_played metadata)
    matches_played_map: dict = {}
    if team_rows:
        for r in team_rows:
            tkey = (r.get("team") or "").lower().strip()
            if tkey:
                matches_played_map[tkey] = matches_played_map.get(tkey, 0) + 1

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
    MIN_PLAYER_MINUTES = 15  # skip sub appearances < 15 min (statistically meaningless)
    if f_player_eng:
        rows = read_json(str(f_player_eng)) if f_player_eng.suffix == ".json" else read_csv_rows(str(f_player_eng))
        skipped_low_min = 0
        for r in rows:
            for k in ["yellow_cards", "red_cards", "cards_per_90", "fouls_per_90",
                      "shots_on_target_ratio", "goal_conversion_rate", "shots_on", "shots_total"]:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass
            try:
                mins = int(r.get("minutes") or 0)
            except (TypeError, ValueError):
                mins = 0
            if mins < MIN_PLAYER_MINUTES:
                skipped_low_min += 1
                continue
            docs.append(
                doc_from_player_engineered(
                    r,
                    league=league,
                    season=season_hint,
                    source_file=str(f_player_eng),
                )
            )
        if skipped_low_min:
            print(f"  Skipped {skipped_low_min} player fixtures with <{MIN_PLAYER_MINUTES} min.")

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
                    cards_induced_pm_map=cards_induced_pm_map,
                    corners_ha_pm_map=corners_ha_pm_map,
                    sot_against_pm_map=sot_against_pm_map,
                    cards_ha_pm_map=cards_ha_pm_map,
                    stat_var_map=stat_var_map,
                    goals_against_pm_map=goals_against_pm_map,
                    goals_ha_pm_map=goals_ha_pm_map,
                    sot_ha_pm_map=sot_ha_pm_map,
                    xg_ha_pm_map=xg_ha_pm_map,
                    matches_played_map=matches_played_map,
                    corners_against_ha_pm_map=corners_against_ha_pm_map,
                    sot_against_ha_pm_map=sot_against_ha_pm_map,
                    cards_induced_ha_pm_map=cards_induced_ha_pm_map,
                    fouls_ha_pm_map=fouls_ha_pm_map,
                )
            )

    # -----------------------------
    # GLOBAL data dictionary (single doc)
    # -----------------------------
    dict_path = write_variable_dictionary(INDEX_DIR)  # ensure INDEX_DIR is defined globally
    docs.append(dictionary_to_doc(dict_path))

    return docs


def discover_fe_dirs(output_root: str) -> List[Path]:
    """Find all *_feature_engineering directories under output_root."""
    root = Path(output_root)
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.lower().endswith("_feature_engineering")
    )


def main():
    import time as _time

    parser = argparse.ArgumentParser(description="Normalize feature-engineering output into RAG docs.")
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help="Canonical league name to normalize (e.g. EPL, LaLiga). Default: all discovered leagues.",
    )
    parser.add_argument(
        "--all", dest="all_leagues", action="store_true",
        help="Explicitly normalize all discovered leagues (same as omitting --league).",
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Restrict to a specific season year (e.g. 2025). Normalizes only files matching that year.",
    )
    args = parser.parse_args()

    t_start = _time.time()
    os.makedirs(INDEX_DIR, exist_ok=True)

    fe_dirs = discover_fe_dirs(OUTPUT_ROOT)
    if not fe_dirs:
        raise SystemExit(f"No *_feature_engineering directories found in {OUTPUT_ROOT}")

    # Filter to requested league if specified
    if args.league and not args.all_leagues:
        target = args.league.strip()
        fe_dirs = [d for d in fe_dirs if (league_from_dir(d.name) or "").lower() == target.lower()]
        if not fe_dirs:
            raise SystemExit(
                f"No feature_engineering directory found for league '{target}'. "
                f"Available: {[league_from_dir(d.name) for d in discover_fe_dirs(OUTPUT_ROOT)]}"
            )

    total = 0
    normalized_count = 0
    skipped_season = 0
    for d in fe_dirs:
        league = league_from_dir(d.name) or "UNKNOWN"

        # Season filter: skip directories whose data files don't match the requested season
        if args.season is not None:
            season_files = list(d.glob(f"*_{args.season}.json")) + list(d.glob(f"*_{args.season}.jsonl"))
            if not season_files:
                skipped_season += 1
                continue

        t_dir = _time.time()
        docs = normalize_feature_engineering_dir(str(d))
        season_token = "unknown"
        if docs:
            s = next((x["metadata"].get("season") for x in docs if x["metadata"].get("season")), None)
            season_token = s.replace("/", "_") if isinstance(s, str) else "unknown"

        # Season filter at doc level: if --season specified, only keep docs matching that season
        if args.season is not None and docs:
            target_season = euro_season_from_year(args.season)
            docs = [doc for doc in docs if doc.get("metadata", {}).get("season") == target_season
                    or doc.get("metadata", {}).get("doc_type") == "data_dictionary"]
            if docs:
                season_token = target_season.replace("/", "_")

        out_path = Path(INDEX_DIR) / f"normalized_{league}_{season_token}.json"
        with open(out_path, "w") as f:
            json.dump(docs, f, ensure_ascii=False)
        elapsed = _time.time() - t_dir
        skipped_players = sum(1 for doc in docs if doc.get("metadata", {}).get("doc_type") == "player_fixture"
                              and doc.get("metadata", {}).get("minutes", 99) < 15)
        print(f"[{elapsed:.1f}s] [{d.name}] -> {len(docs)} docs -> {out_path}")
        total += len(docs)
        normalized_count += 1

    t_total = _time.time() - t_start
    print(f"\nDone in {t_total:.1f}s. Normalized {normalized_count} directories, {total:,} total docs.")
    if skipped_season:
        print(f"  Skipped {skipped_season} directories (no files matching season {args.season}).")

if __name__ == "__main__":
    main()
