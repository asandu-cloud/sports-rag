"""
Quick-and-dirty backtest to see if heuristic scores correlate with outcomes.
This is NOT a betting model; it’s a sanity check to tune thresholds.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from heuristics import top_markets


def load_team_fixtures(path: str) -> pd.DataFrame:
    return pd.read_json(path)


def to_meta(row: pd.Series, opp_row: pd.Series | None) -> dict:
    corners_against = opp_row["corners"] if opp_row is not None and "corners" in opp_row else None
    cards_total = row.get("cards_total")
    if cards_total is None:
        cards_total = row.get("yellow_cards", 0) + row.get("red_cards", 0)
    return {
        "team": row.get("team"),
        "control_index": row.get("control_index"),
        "dominance_index": row.get("dominance_index"),
        "aggression_index_norm": row.get("aggression_index_norm"),
        "form_index_team": row.get("form_index_team"),
        "goals_for_pm": row.get("goals"),
        "corners_pm": row.get("corners"),
        "corners_against_pm": corners_against,
        "cards_per_90_team": row.get("cards_per_90_team"),
        "cards_pm": cards_total,  # proxy if per-90 missing
        "fouls_per_90_team": row.get("fouls_per_90_team"),
        "possession": row.get("possession"),
        "shots_for_pm": row.get("shots_total"),
        "shots_against_pm": opp_row.get("shots_total") if opp_row is not None else None,
        "archetype": row.get("archetype") if "archetype" in row else None,
    }


def evaluate(path: str):
    df = load_team_fixtures(path)
    df["fixture_norm"] = df["fixture"].str.lower().str.strip()

    results = {"over_hits": 0, "over_total": 0, "cards_hits": 0, "cards_total": 0}

    for fx, group in df.groupby("fixture_norm"):
        if len(group) < 2:
            continue
        # map team -> row
        rows = {r["team"]: r for _, r in group.iterrows()}
        # identify home/away using home_team field if present
        home_name = group["home_team"].iloc[0] if "home_team" in group else list(rows.keys())[0]
        away_name = group["away_team"].iloc[0] if "away_team" in group else list(rows.keys())[1]
        home_row = rows.get(home_name)
        if home_row is None:
            home_row = group.iloc[0]
        away_row = rows.get(away_name)
        if away_row is None:
            away_row = group.iloc[1]

        home_meta = to_meta(home_row, away_row)
        away_meta = to_meta(away_row, home_row)

        markets = top_markets(home_meta, away_meta, n=3)
        total_goals = float(home_row.get("goals", 0) + away_row.get("goals", 0))
        # Cards: fall back to yellows+reds if cards_total missing
        def card_sum(r):
            if r.get("cards_total") is not None:
                return float(r.get("cards_total"))
            return float(r.get("yellow_cards", 0) + r.get("red_cards", 0))

        total_cards = card_sum(home_row) + card_sum(away_row)

        # crude checks: if top market is over goals, count hit
        over_pick = next((m for m in markets if "Over 2.5" in m["market"]), None)
        if over_pick:
            results["over_total"] += 1
            if total_goals > 2.5:
                results["over_hits"] += 1

        cards_pick = next((m for m in markets if "cards" in m["market"].lower()), None)
        if cards_pick:
            results["cards_total"] += 1
            if total_cards >= 4:  # simple threshold
                results["cards_hits"] += 1

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/Users/sanduandrei/Desktop/Betting_RAG/Output/Prem_feature_engineering/team_engineered_features_2025.json",
                    help="Path to team_engineered_features_YYYY.json")
    args = ap.parse_args()

    res = evaluate(args.path)
    print("Heuristic backtest (rough):")
    print(json.dumps(res, indent=2))
    if res["over_total"]:
        print(f"Over hit rate: {res['over_hits']}/{res['over_total']} ({res['over_hits']/res['over_total']:.2%})")
    if res["cards_total"]:
        print(f"Cards hit rate: {res['cards_hits']}/{res['cards_total']} ({res['cards_hits']/res['cards_total']:.2%})")


if __name__ == "__main__":
    main()
