# Scraping footballdata.co.uk

from __future__ import annotations
import pandas as pd
import argparse
import datetime as dt
import hashlib
import io
import os
from pathlib import Path
from typing import Dict, List
import requests
import yaml

path = '/Users/sanduandrei/Desktop/Betting_RAG'

RAW_DIR = Path('/CSVs/Input_csv')
OUT_DIR = Path('/CSVs/Output_csv')
CONFIG = Path('/config/football_columns.yaml')

DIV_TO_LEAGUE = {
    "E0": "EPL",
    "SP1": "LaLiga",
    "I1": "SerieA",
    "D1": "Bundesliga",
    "F1": "Ligue1",
    # keep others unmapped -> use DIV code
}

STRING_LIKE = {'DIV', 'HOME', 'AWAY', 'REF', 'Time'}

def read_yaml_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    cmap = y.get("column_map", {})
    if not isinstance(cmap, dict) or not cmap:
        raise ValueError(
            f"'column_map' missing or empty in {path}. "
            "Make sure it has: column_map: { RawHeader: CanonicalName, ... }"
        )
    return cmap


def ensure_all_target_columns(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def parse_dates_times(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        # dd/mm/yy
        df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce', dayfirst=True)
    if 'Time' in df.columns:
        # standardize to HH:MM (string): if numeric/float -> coerce top NaN
        df['Time'] = df['Time'].astype(str).str.strip()
        # common oddities like '20:0' -> '20:00'
        df['Time'] = df['Time'].str.replace(r"^(\d{1,2}):(\d{1})$", r"\1:0\2", regex=True)
    return df

def infer_season_code_from_dates(series: pd.Series) -> str | None:
    """
    Return season code like '2526' for 2025/26 using the *max* valid date.
    """
    s = pd.to_datetime(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    latest = s.max().date()
    yy = latest.year % 100
    # football seasons generally start Jul/Aug and cross the year boundary
    if latest.month >= 7:  # season start year
        return f"{yy:02d}{(yy+1)%100:02d}"
    else:
        return f"{(yy-1)%100:02d}{yy:02d}"


def map_div_to_league(div_value: str) -> str:
    return DIV_TO_LEAGUE.get(div_value, str(div_value or "")).strip()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert everything that is not string-like to numeric where possible.
    Keeps Date as datetime64.
    """
    for col in df.columns:
        if col == "Date" or col in STRING_LIKE:
            continue
        if df[col].dtype == "O":
            df[col] = pd.to_numeric(df[col], errors="ignore")  # don't force; preserve strings if truly non-numeric
    return df


def tidy_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("HOME", "AWAY", "REF", "DIV"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def canonical_order(target_cols: List[str]) -> List[str]:
    """
    Put core match columns first, then odds/totals/ah*, then closing*.
    """
    core = [
        "DIV", "Date", "Time", "HOME", "AWAY",
        "FT_HOME_GOALS", "FT_AWAY_GOALS", "FT_RESULT",
        "HT_GOME_GOALS", "HT_AWAY_GOALS", "HT_RESULT",
        "REF",
        "HOME_SHOTS", "AWAY_SHOTS", "HOME_SOT", "AWAY_SOT",
        "HOME_FOUL", "AWAY_FOUL", "HOME_CORNER", "AWAY_CORNER",
        "HOME_YELLOW", "AWAY_YELLOW", "HOME_RED", "AWAY_RED",
    ]
    rest = [c for c in target_cols if c not in core]
    # sort rest by family to keep stable layout
    rest_sorted = sorted(rest, key=lambda x: (
        0 if x.startswith("ODDS_") else
        1 if x.startswith("TOTALS_") else
        2 if x.startswith("AH_") else
        3 if x.startswith("CLOSING_") else
        9, x
    ))
    return core + rest_sorted + ["league", "season"]


# --------- main cleaning routine ---------

def clean_one_csv(csv_path: Path, colmap: Dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    # rename to canonical names
    existing_map = {k: v for k, v in colmap.items() if k in raw.columns}
    df = raw.rename(columns=existing_map).copy()

    # make sure all mapped targets exist
    target_cols = list(colmap.values())
    df = ensure_all_target_columns(df, target_cols)

    # basic cleaning
    df = parse_dates_times(df)
    df = tidy_strings(df)
    df = coerce_numeric(df)

    # derive league/season
    if "DIV" in df.columns:
        df["league"] = df["DIV"].map(map_div_to_league)
    else:
        df["league"] = pd.NA

    season_code = infer_season_code_from_dates(df.get("Date"))
    df["season"] = season_code

    # select/ordering
    ordered_cols = [c for c in canonical_order(target_cols) if c in df.columns]
    df = df[ordered_cols]

    # sort by date/time if available
    if "Date" in df.columns:
        df = df.sort_values(["Date", "Time"] if "Time" in df.columns else ["Date"], kind="mergesort").reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, src_name: str, out_dir: Path, overwrite: bool) -> Path:
    league = (df["league"].iloc[0] if "league" in df.columns and not df["league"].isna().all() else "UNKNOWN")
    season = (df["season"].iloc[0] if "season" in df.columns and not df["season"].isna().all() else "UNKN")
    out_leaf = out_dir / str(league) / str(season)
    out_leaf.mkdir(parents=True, exist_ok=True)
    out_path = out_leaf / f"{Path(src_name).stem}_clean.csv"
    if out_path.exists() and not overwrite:
        print(f"   .. exists (skip): {out_path}")
        return out_path
    df.to_csv(out_path, index=False)
    print(f"   .. wrote: {out_path}  (rows={len(df)})")
    return out_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="CSVs/Input_csv", help="Folder with raw CSVs you downloaded.")
    ap.add_argument("--out_dir", type=str, default="CSVs/Output_csv", help="Folder to write cleaned CSVs.")
    ap.add_argument("--config", type=str, default="config/football_columns.yaml", help="YAML mapping file.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cleaned files.")
    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap = read_yaml_map(Path(args.config))

    cleaned_paths: List[Path] = []
    frames: List[pd.DataFrame] = []

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {in_dir.resolve()}. Put your downloads there.")
        return

    for p in csvs:
        print(f"-> Cleaning {p.name} ...")
        try:
            df = clean_one_csv(p, colmap)
        except Exception as e:
            print(f"   !! failed: {e}")
            continue
        cleaned_paths.append(write_outputs(df, p.name, out_dir, args.overwrite))
        frames.append(df)

    if frames:
        union = pd.concat(frames, ignore_index=True, sort=False)
        union_out = out_dir / "top5_union_latest.csv"
        union.to_csv(union_out, index=False)
        print(f"\n✓ Union written: {union_out} (rows={len(union)})")
    else:
        print("\nNo cleaned data produced.")


