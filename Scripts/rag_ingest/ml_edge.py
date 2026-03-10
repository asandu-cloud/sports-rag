#!/usr/bin/env python3
"""
ML edge layer for the betting RAG.

Two components:
1. Stat-specific Elo ratings per team (goals, corners, cards, SoT)
2. Ridge regression models predicting match totals per stat type

Both are trained offline from team_engineered_features data and loaded
lazily at query time. Graceful degradation: returns None when unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

LEAGUE_TO_FE_DIR = {
    "EPL": "Prem_feature_engineering",
    "LaLiga": "LaLiga_feature_engineering",
    "SerieA": "SeriaA_feature_engineering",
    "Bundesliga": "Bundesliga_feature_engineering",
    "Ligue1": "Ligue1_feature_engineering",
    "UCL": "UCL_feature_engineering",
    "UEL": "UEL_feature_engineering",
    "UECL": "UECL_feature_engineering",
}
ALL_LEAGUES = list(LEAGUE_TO_FE_DIR.keys())

STAT_TYPES = ["goals", "corners", "cards", "sot"]
STAT_FIELD_MAP = {
    "goals": "goals",
    "corners": "corners",
    "cards": "cards_total",
    "sot": "shots_on",
}

MODELS_DIR = ROOT / "Index" / "ml_models"
ELO_PATH = ROOT / "Index" / "elo_ratings.json"

ELO_START = 1500.0
ELO_K = 32.0

# Features extracted from each team's fixture row for the Ridge model.
# Only composites, rolling averages, and activity metrics — no raw per-match
# counts of the target stat itself (corners, shots_on, cards_total, goals).
TEAM_FEATURES = [
    "possession", "control_index", "dominance_index", "form_index_team",
    "aggression_index_norm", "avg_goals_last_5", "avg_cards_last_5",
    "tackles", "interceptions", "fouls_committed", "passes_total",
]

# Minimum cross-validated R2 for a model to be used at query time.
MIN_MODEL_R2 = 0.05

# Prediction clamp ranges per stat (min, max) — prevents extreme outputs
# when query-time features differ from training distribution.
PRED_CLAMP = {
    "goals": (0.5, 6.0),
    "corners": (4.0, 16.0),
    "cards": (1.0, 10.0),
    "sot": (3.0, 14.0),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_fixture_rows(league: str) -> List[dict]:
    fe_dir = LEAGUE_TO_FE_DIR.get(league)
    if not fe_dir:
        return []
    base = ROOT / "Output" / fe_dir
    rows: List[dict] = []
    for path in sorted(base.glob("team_engineered_features_*.json")):
        with open(path) as f:
            rows.extend(json.load(f))
    return rows


def _pair_fixtures(rows: List[dict]) -> List[Tuple[dict, dict]]:
    """Group rows into (home_row, away_row) pairs by fixture."""
    by_fixture: Dict[str, List[dict]] = {}
    for r in rows:
        fx = (r.get("fixture") or "").strip()
        if not fx:
            continue
        by_fixture.setdefault(fx, []).append(r)

    pairs = []
    for fx, group in by_fixture.items():
        if len(group) != 2:
            continue
        # Identify home and away
        r0, r1 = group
        home_team = (r0.get("home_team") or "").strip()
        if (r0.get("team") or "").strip() == home_team:
            pairs.append((r0, r1))
        else:
            pairs.append((r1, r0))
    return pairs


def _sort_pairs_by_date(pairs: List[Tuple[dict, dict]]) -> List[Tuple[dict, dict]]:
    def date_key(pair):
        d = pair[0].get("fixture_date") or pair[1].get("fixture_date") or ""
        return d
    return sorted(pairs, key=date_key)


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Elo system
# ---------------------------------------------------------------------------

def build_elo_ratings(leagues: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Build stat-specific Elo ratings from historical fixtures."""
    leagues = leagues or ALL_LEAGUES
    elo: Dict[str, Dict[str, float]] = {}

    for league in leagues:
        rows = _load_fixture_rows(league)
        if not rows:
            print(f"  [{league}] No fixture data found, skipping.")
            continue

        pairs = _pair_fixtures(rows)
        pairs = _sort_pairs_by_date(pairs)
        print(f"  [{league}] {len(pairs)} fixtures for Elo computation.")

        for home_row, away_row in pairs:
            h_team = (home_row.get("team") or "").lower().strip()
            a_team = (away_row.get("team") or "").lower().strip()
            if not h_team or not a_team:
                continue

            # Initialize if needed
            if h_team not in elo:
                elo[h_team] = {s: ELO_START for s in STAT_TYPES}
            if a_team not in elo:
                elo[a_team] = {s: ELO_START for s in STAT_TYPES}

            for stat in STAT_TYPES:
                field = STAT_FIELD_MAP[stat]
                h_val = _safe_float(home_row.get(field))
                a_val = _safe_float(away_row.get(field))
                total = h_val + a_val
                if total <= 0:
                    continue

                # Game result = proportion of stat won
                s_h = h_val / total
                s_a = a_val / total

                # Expected from Elo
                r_h = elo[h_team][stat]
                r_a = elo[a_team][stat]
                e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
                e_a = 1.0 - e_h

                # Update
                elo[h_team][stat] = r_h + ELO_K * (s_h - e_h)
                elo[a_team][stat] = r_a + ELO_K * (s_a - e_a)

    return elo


def save_elo(elo: Dict[str, Dict[str, float]], path: Path = ELO_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Round for readability
    out = {}
    for team, stats in elo.items():
        out[team] = {s: round(v, 1) for s, v in stats.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved Elo ratings for {len(out)} teams to {path}")


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def _extract_features(home_row: dict, away_row: dict,
                      elo: Optional[Dict] = None,
                      stat: str = "goals") -> Optional[np.ndarray]:
    """Build feature vector for one fixture."""
    feats = []
    for row in [home_row, away_row]:
        for f in TEAM_FEATURES:
            feats.append(_safe_float(row.get(f)))

    # Derived features
    h_poss = _safe_float(home_row.get("possession"))
    a_poss = _safe_float(away_row.get("possession"))
    feats.append(h_poss - a_poss)  # possession_gap

    # Elo diff
    if elo:
        h_team = (home_row.get("team") or "").lower().strip()
        a_team = (away_row.get("team") or "").lower().strip()
        h_elo = elo.get(h_team, {}).get(stat, ELO_START)
        a_elo = elo.get(a_team, {}).get(stat, ELO_START)
        feats.append(h_elo - a_elo)
    else:
        feats.append(0.0)

    arr = np.array(feats, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_regression_models(leagues: Optional[List[str]] = None,
                            elo: Optional[Dict] = None) -> Dict[str, Pipeline]:
    """Train Ridge regression per stat type. Returns dict of pipelines."""
    if not _HAS_SKLEARN:
        print("scikit-learn not available — skipping regression models.")
        return {}

    leagues = leagues or ALL_LEAGUES
    all_pairs: List[Tuple[dict, dict]] = []
    for league in leagues:
        rows = _load_fixture_rows(league)
        if rows:
            pairs = _pair_fixtures(rows)
            all_pairs.extend(pairs)

    if len(all_pairs) < 20:
        print(f"Only {len(all_pairs)} fixtures — too few to train. Need at least 20.")
        return {}

    print(f"Training on {len(all_pairs)} fixtures across {len(leagues)} league(s).")

    models: Dict[str, Pipeline] = {}
    r2_meta: Dict[str, float] = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for stat in STAT_TYPES:
        field = STAT_FIELD_MAP[stat]
        X_list, y_list = [], []

        for home_row, away_row in all_pairs:
            feats = _extract_features(home_row, away_row, elo=elo, stat=stat)
            if feats is None:
                continue
            h_val = _safe_float(home_row.get(field))
            a_val = _safe_float(away_row.get(field))
            target = h_val + a_val
            X_list.append(feats)
            y_list.append(target)

        X = np.array(X_list)
        y = np.array(y_list)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=50.0)),
        ])

        # Cross-validation metrics
        r2_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        mae_scores = -cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")

        print(f"  [{stat}] R2={r2_scores.mean():.3f} (+/-{r2_scores.std():.3f})  "
              f"MAE={mae_scores.mean():.2f} (+/-{mae_scores.std():.2f})  "
              f"n={len(y)}")

        # Train on full data
        pipe.fit(X, y)
        model_path = MODELS_DIR / f"{stat}_ridge.joblib"
        joblib.dump(pipe, model_path)
        models[stat] = pipe

        # Save R2 metadata for query-time gating
        r2_meta[stat] = round(float(r2_scores.mean()), 4)

    # Save R2 metadata
    r2_path = MODELS_DIR / "model_r2.json"
    with open(r2_path, "w") as f:
        json.dump(r2_meta, f, indent=2)
    print(f"Saved {len(models)} models to {MODELS_DIR} (R2 metadata: {r2_meta})")
    return models


# ---------------------------------------------------------------------------
# Query-time prediction API
# ---------------------------------------------------------------------------

_models_cache: Dict[str, Pipeline] = {}
_elo_cache: Optional[Dict[str, Dict[str, float]]] = None
_r2_cache: Optional[Dict[str, float]] = None


def _load_r2_meta() -> Dict[str, float]:
    global _r2_cache
    if _r2_cache is not None:
        return _r2_cache
    r2_path = MODELS_DIR / "model_r2.json"
    if not r2_path.exists():
        _r2_cache = {}
        return _r2_cache
    try:
        with open(r2_path) as f:
            _r2_cache = json.load(f)
    except Exception:
        _r2_cache = {}
    return _r2_cache


def _load_model(stat: str) -> Optional[Pipeline]:
    if stat in _models_cache:
        return _models_cache[stat]
    if not _HAS_SKLEARN:
        return None
    # Check R2 threshold — skip models that predict worse than the mean
    r2_meta = _load_r2_meta()
    if r2_meta.get(stat, 0.0) < MIN_MODEL_R2:
        return None
    path = MODELS_DIR / f"{stat}_ridge.joblib"
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
        _models_cache[stat] = model
        return model
    except Exception:
        return None


def _load_elo() -> Optional[Dict[str, Dict[str, float]]]:
    global _elo_cache
    if _elo_cache is not None:
        return _elo_cache
    if not ELO_PATH.exists():
        return None
    try:
        with open(ELO_PATH) as f:
            _elo_cache = json.load(f)
        return _elo_cache
    except Exception:
        return None


def get_elo_edge(home: str, away: str, stat: str) -> Optional[float]:
    """Elo difference (home - away) for a stat. Positive = home stronger."""
    elo = _load_elo()
    if elo is None:
        return None
    h_elo = elo.get(home.lower().strip(), {}).get(stat, ELO_START)
    a_elo = elo.get(away.lower().strip(), {}).get(stat, ELO_START)
    return h_elo - a_elo


def ml_predict_total(home: str, away: str, league: str, stat: str,
                     home_meta: Optional[dict] = None,
                     away_meta: Optional[dict] = None) -> Optional[float]:
    """Predict match total for a stat type.

    If home_meta/away_meta (Chroma team_profile metadata dicts) are provided,
    uses them to build features. Otherwise returns None.
    """
    model = _load_model(stat)
    if model is None:
        return None
    if home_meta is None or away_meta is None:
        return None

    elo = _load_elo()
    # Build a synthetic "fixture row" from profile metadata for feature extraction
    home_row = _profile_meta_to_fixture_row(home_meta, home)
    away_row = _profile_meta_to_fixture_row(away_meta, away)

    feats = _extract_features(home_row, away_row, elo=elo, stat=stat)
    if feats is None:
        return None
    try:
        # Clip features to ±2σ of training distribution to prevent train/serve
        # skew (profile metadata = season averages, training = per-match values).
        scaler = model.named_steps["scaler"]
        feats_clipped = np.clip(feats,
                                scaler.mean_ - 2.0 * scaler.scale_,
                                scaler.mean_ + 2.0 * scaler.scale_)
        pred = float(model.predict([feats_clipped])[0])
        lo, hi = PRED_CLAMP.get(stat, (0.0, 20.0))
        return max(lo, min(hi, pred))
    except Exception:
        return None


# Training-data means for features unavailable in team_profile metadata.
# Using means (rather than 0) prevents StandardScaler from producing extreme z-scores.
_TRAINING_MEANS = {
    "tackles": 16.9,
    "interceptions": 8.25,
    "passes_total": 436.9,
    "fouls_committed": 10.7,
}


def _profile_meta_to_fixture_row(meta: dict, team_name: str) -> dict:
    """Convert Chroma team_profile metadata to a fixture-like row for feature extraction."""
    return {
        "team": team_name,
        "possession": meta.get("possession"),
        "control_index": meta.get("control_index"),
        "dominance_index": meta.get("dominance_index"),
        "form_index_team": meta.get("form_index_team"),
        "aggression_index_norm": meta.get("aggression_index_norm"),
        "avg_goals_last_5": meta.get("goals_for_pm"),  # best proxy from profiles
        "avg_cards_last_5": meta.get("cards_per_90_team"),  # best proxy from profiles
        "tackles": _TRAINING_MEANS["tackles"],
        "interceptions": _TRAINING_MEANS["interceptions"],
        "fouls_committed": meta.get("fouls_per_90_team") or _TRAINING_MEANS["fouls_committed"],
        "passes_total": _TRAINING_MEANS["passes_total"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_elo_table(elo: Dict[str, Dict[str, float]], stat: Optional[str] = None) -> None:
    stats_to_show = [stat] if stat else STAT_TYPES
    for s in stats_to_show:
        rankings = sorted(elo.items(), key=lambda x: x[1].get(s, ELO_START), reverse=True)
        print(f"\n{'='*50}")
        print(f"  {s.upper()} Elo Rankings")
        print(f"{'='*50}")
        for i, (team, ratings) in enumerate(rankings[:25], 1):
            rating = ratings.get(s, ELO_START)
            diff = rating - ELO_START
            sign = "+" if diff >= 0 else ""
            print(f"  {i:>3}. {team:<30s} {rating:>7.1f}  ({sign}{diff:.1f})")


def main():
    parser = argparse.ArgumentParser(description="ML edge layer: Elo + Ridge regression.")
    parser.add_argument("--train", action="store_true", help="Train all models + build Elo.")
    parser.add_argument("--league", type=str, default=None,
                        help="Restrict to one league (e.g. EPL).")
    parser.add_argument("--elo-table", action="store_true", help="Print Elo rankings.")
    parser.add_argument("--elo-stat", type=str, default=None,
                        help="Filter Elo table to one stat (goals, corners, cards, sot).")
    parser.add_argument("--predict", nargs=3, metavar=("HOME", "AWAY", "LEAGUE"),
                        help="Test prediction for a fixture.")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation and print metrics (no model save).")
    args = parser.parse_args()

    leagues = [args.league] if args.league else None

    if args.train:
        print("Building Elo ratings...")
        elo = build_elo_ratings(leagues)
        save_elo(elo)
        print()
        print("Training Ridge regression models...")
        build_regression_models(leagues, elo=elo)
        print("\nDone.")

    elif args.elo_table:
        elo = _load_elo()
        if elo is None:
            print("No Elo ratings found. Run --train first.")
            return
        _print_elo_table(elo, stat=args.elo_stat)

    elif args.predict:
        home, away, league = args.predict
        print(f"Predictions for {home} vs {away} ({league}):")
        elo = _load_elo()
        for stat in STAT_TYPES:
            elo_diff = get_elo_edge(home, away, stat)
            elo_str = f"(Elo diff: {elo_diff:+.1f})" if elo_diff is not None else "(no Elo)"

            model = _load_model(stat)
            if model is None:
                print(f"  {stat}: model not found")
                continue

            # For CLI prediction, load fixture data to build features
            rows = _load_fixture_rows(league)
            if not rows:
                print(f"  {stat}: no fixture data for {league}")
                continue

            # Find latest rows for each team to use as feature source
            h_row = next((r for r in reversed(rows)
                          if (r.get("team") or "").lower() == home.lower()), None)
            a_row = next((r for r in reversed(rows)
                          if (r.get("team") or "").lower() == away.lower()), None)
            if h_row is None or a_row is None:
                print(f"  {stat}: team not found in data")
                continue

            feats = _extract_features(h_row, a_row, elo=elo if elo else _load_elo(), stat=stat)
            if feats is not None:
                pred = max(0.0, float(model.predict([feats])[0]))
                print(f"  {stat}: {pred:.2f} predicted total {elo_str}")
            else:
                print(f"  {stat}: feature extraction failed")

    elif args.cv:
        if not _HAS_SKLEARN:
            print("scikit-learn not available.")
            return
        print("Running cross-validation (no model save)...")
        elo = _load_elo() or build_elo_ratings(leagues)
        all_pairs: List[Tuple[dict, dict]] = []
        for league in (leagues or ALL_LEAGUES):
            rows = _load_fixture_rows(league)
            if rows:
                all_pairs.extend(_pair_fixtures(rows))

        for stat in STAT_TYPES:
            field = STAT_FIELD_MAP[stat]
            X_list, y_list = [], []
            for home_row, away_row in all_pairs:
                feats = _extract_features(home_row, away_row, elo=elo, stat=stat)
                if feats is None:
                    continue
                h_val = _safe_float(home_row.get(field))
                a_val = _safe_float(away_row.get(field))
                y_list.append(h_val + a_val)
                X_list.append(feats)

            X = np.array(X_list)
            y = np.array(y_list)
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
            mae = -cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
            print(f"  [{stat}] R2={r2.mean():.3f} (+/-{r2.std():.3f})  "
                  f"MAE={mae.mean():.2f} (+/-{mae.std():.2f})  n={len(y)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
