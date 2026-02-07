"""
Query the local NetworkX graph for matchup aggregates.
Outputs a dict with recent-form and archetype-cohort aggregates for both teams.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import networkx as nx
import statistics
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = BASE_DIR / "Index" / "graph_epl.pkl"

_graph_cache = None


def load_graph(path: Path = GRAPH_PATH):
    global _graph_cache
    if _graph_cache is None:
        with open(path, "rb") as f:
            _graph_cache = pickle.load(f)
    return _graph_cache


def _safe_avg(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _canonical(team: str) -> str:
    return team.strip().title()


def _get_fixtures(G: nx.Graph, team: str) -> List[Tuple[str, dict]]:
    out = []
    for nbr in G.neighbors(team):
        ed = G[team][nbr]
        fdate = ed.get("fixture_date") or ""
        out.append((nbr, ed, fdate))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def _agg_team(G: nx.Graph, team: str, last_n: int = 6, opponent_archetype: Optional[str] = None) -> Dict:
    fixtures = _get_fixtures(G, team)
    stats = {
        "n": 0,
        "goals_for": [],
        "goals_against": [],
        "corners_for": [],
        "corners_against": [],
        "cards_for": [],
        "cards_against": [],
        "control": [],
        "form": [],
    }
    for fx, ed, _ in fixtures:
        if stats["n"] >= last_n:
            break
        opp = next(t for t in G.neighbors(fx) if t != team)
        # archetype filter if requested
        if opponent_archetype:
            opp_arch = G.nodes[opp].get("archetype")
            if opp_arch != opponent_archetype:
                continue
        opp_ed = G[opp][fx]
        stats["n"] += 1
        stats["goals_for"].append(ed.get("goals"))
        stats["goals_against"].append(opp_ed.get("goals"))
        stats["corners_for"].append(ed.get("corners"))
        stats["corners_against"].append(opp_ed.get("corners"))
        stats["cards_for"].append(ed.get("cards"))
        stats["cards_against"].append(opp_ed.get("cards"))
        stats["control"].append(ed.get("control"))
        stats["form"].append(ed.get("form"))

    def avg(key):
        return _safe_avg(stats[key])

    return {
        "samples": stats["n"],
        "goals_for_pm": avg("goals_for"),
        "goals_against_pm": avg("goals_against"),
        "corners_for_pm": avg("corners_for"),
        "corners_against_pm": avg("corners_against"),
        "cards_for_pm": avg("cards_for"),
        "cards_against_pm": avg("cards_against"),
        "control_index_avg": avg("control"),
        "form_index_avg": avg("form"),
    }


def get_matchup_snapshot(home_team: str, away_team: str, last_n: int = 6) -> Optional[Dict]:
    G = load_graph()
    h = _canonical(home_team)
    a = _canonical(away_team)
    if h not in G or a not in G:
        return None

    a_arch = G.nodes[a].get("archetype")
    h_arch = G.nodes[h].get("archetype")

    home_recent = _agg_team(G, h, last_n=last_n, opponent_archetype=None)
    away_recent = _agg_team(G, a, last_n=last_n, opponent_archetype=None)

    home_vs_arch = _agg_team(G, h, last_n=last_n, opponent_archetype=a_arch) if a_arch else None
    away_vs_arch = _agg_team(G, a, last_n=last_n, opponent_archetype=h_arch) if h_arch else None

    return {
        "home_team": h,
        "away_team": a,
        "home_archetype": h_arch,
        "away_archetype": a_arch,
        "home_recent": home_recent,
        "away_recent": away_recent,
        "home_vs_away_arch": home_vs_arch,
        "away_vs_home_arch": away_vs_arch,
    }


def snapshot_to_text(snap: Dict) -> str:
    if not snap:
        return "Graph snapshot unavailable."

    def fmt(block: Dict, label: str) -> str:
        if not block:
            return f"{label}: no cohort data."
        return (
            f"{label}: n={block.get('samples',0)}, "
            f"GF {block.get('goals_for_pm'):.2f}, GA {block.get('goals_against_pm'):.2f}, "
            f"Corners For {block.get('corners_for_pm'):.2f}, Corners Against {block.get('corners_against_pm'):.2f}, "
            f"Cards For {block.get('cards_for_pm'):.2f}, Cards Against {block.get('cards_against_pm'):.2f}, "
            f"Control {block.get('control_index_avg'):.2f}, Form {block.get('form_index_avg'):.2f}"
        )

    return "\n".join([
        f"HOME ({snap['home_team']}) archetype={snap.get('home_archetype')}",
        fmt(snap.get("home_recent"), "Home recent"),
        fmt(snap.get("home_vs_away_arch"), "Home vs away archetype cohort"),
        f"AWAY ({snap['away_team']}) archetype={snap.get('away_archetype')}",
        fmt(snap.get("away_recent"), "Away recent"),
        fmt(snap.get("away_vs_home_arch"), "Away vs home archetype cohort"),
    ])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    snap = get_matchup_snapshot(args.home, args.away, last_n=args.n)
    print(snapshot_to_text(snap))
