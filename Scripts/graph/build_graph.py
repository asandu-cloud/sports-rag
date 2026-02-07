"""
Build a lightweight in-process graph for EPL data using NetworkX.
Nodes:
  - team (type=team) with season/profile attrs (control_index_avg, form_index_avg, corners_pm, cards_pm, archetype)
  - fixture (type=fixture) with fixture_date, season
Edges: team -- fixture with per-fixture stats (goals, corners, cards, fouls, control, form, aggression, role).
Saved to: Index/graph_epl.pkl
"""

from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_FE_PATH = BASE_DIR / "Output" / "Prem_feature_engineering" / "team_engineered_features_2025.json"
DEFAULT_PROF_PATH = BASE_DIR / "Output" / "Prem_feature_engineering" / "team_profiles_2025.jsonl"
DEFAULT_OUT = BASE_DIR / "Index" / "graph_epl.pkl"


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: Path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def canonical_team(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().title()


def load_profiles(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    profs = {}
    for r in rows:
        t = canonical_team(r.get("team"))
        if not t:
            continue
        profs[t] = {
            "control_index_avg": r.get("control_index"),
            "form_index_avg": r.get("form_index_team"),
            "corners_pm": r.get("corners_pm"),
            "cards_pm": r.get("cards_pm"),
            "aggression_index_norm_avg": r.get("aggression_index_norm"),
            "archetype": r.get("archetype"),
        }
    return profs


def build_graph(fe_path: Path, prof_path: Path, out_path: Path):
    data = read_json(fe_path)
    profiles = load_profiles(prof_path)

    G = nx.Graph()

    # Group rows by fixture id or fixture string
    fixture_groups: Dict[str, list] = {}
    for r in data:
        fid = r.get("fixture_id") or r.get("fixture")
        if fid is None:
            continue
        fkey = f"fx_{fid}"
        fixture_groups.setdefault(fkey, []).append(r)

    for fkey, rows in fixture_groups.items():
        sample = rows[0]
        fdate = sample.get("fixture_date")
        season = str(sample.get("season") or "2025/26")
        G.add_node(fkey, type="fixture", fixture_date=fdate, season=season, fixture=sample.get("fixture"))

        for r in rows:
            team = canonical_team(r.get("team"))
            if not team:
                continue
            # Add team node with profile attrs if present
            if not G.has_node(team):
                attrs = {"type": "team"}
                attrs.update(profiles.get(team, {}))
                G.add_node(team, **attrs)

            role = None
            ht = canonical_team(r.get("home_team"))
            at = canonical_team(r.get("away_team"))
            if ht and ht == team:
                role = "home"
            elif at and at == team:
                role = "away"

            G.add_edge(
                team,
                fkey,
                role=role,
                goals=r.get("goals"),
                corners=r.get("corners"),
                cards=r.get("cards_total"),
                fouls=r.get("fouls_committed"),
                control=r.get("control_index"),
                form=r.get("form_index_team"),
                aggression=r.get("aggression_index_norm"),
                fixture_date=fdate,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph saved to {out_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", type=str, default=str(DEFAULT_FE_PATH), help="Path to team_engineered_features_YYYY.json")
    ap.add_argument("--profiles", type=str, default=str(DEFAULT_PROF_PATH), help="Path to team_profiles_YYYY.jsonl")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output pickle path for graph")
    args = ap.parse_args()

    fe_path = Path(args.fe)
    prof_path = Path(args.profiles)
    out_path = Path(args.out)

    build_graph(fe_path, prof_path, out_path)


if __name__ == "__main__":
    main()
