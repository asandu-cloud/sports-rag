"""Read-only counts for the Phase 2 tracking handoff; no API calls or grading."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[2]


def audit(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {t: db.execute("SELECT count(*) FROM " + t).fetchone()[0] for t in (
            "predictions", "published_recommendations", "recommendation_deliveries",
            "match_reads", "match_read_deliveries", "match_read_selections", "unit_bet_slips") if t in tables}
        sources = [dict(r) for r in db.execute("""SELECT
            CASE WHEN source LIKE 'canonical_published%' THEN 'canonical_published' ELSE source END AS source,
            count(*) AS count, sum(outcome IS NULL) AS unresolved,
            sum(closing_odds IS NOT NULL) AS with_closing_odds FROM predictions GROUP BY 1""")]
        result = {"counts": counts, "sources": sources}
        if "published_recommendations" in tables:
            rows = list(db.execute("""SELECT r.decision_json, r.model_version, p.outcome, p.closing_odds,
                p.prediction_date, p.kickoff FROM published_recommendations r
                JOIN predictions p ON p.id=r.prediction_id"""))
            today = datetime.now(timezone.utc).date().isoformat()
            result["published"] = {
                "unresolved": sum(r["outcome"] is None for r in rows),
                "past_kickoff_day_unresolved": sum(r["outcome"] is None and bool(r["kickoff"]) and r["kickoff"][:10] < today for r in rows),
                "missing_model_version": sum(r["model_version"] is None for r in rows),
                "missing_system_version": sum(not (json.loads(r["decision_json"]).get("provenance") or {}).get("system_version") for r in rows),
                "publication_vs_kickoff_day_differences": sum(bool(r["kickoff"]) and r["kickoff"][:10] != r["prediction_date"] for r in rows),
                "with_closing_odds": sum(r["closing_odds"] is not None for r in rows),
            }
            result["deliveries_by_surface"] = dict(db.execute("SELECT surface,count(*) FROM recommendation_deliveries GROUP BY surface"))
            result["visible_selections_missing_publication_link"] = db.execute("""SELECT count(*) FROM match_read_selections s
                WHERE s.published_recommendation_id IS NULL AND EXISTS (
                SELECT 1 FROM match_read_deliveries d WHERE d.match_read_id=s.match_read_id)""").fetchone()[0]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--legacy-database", type=Path, default=ROOT / "Index/predictions.db")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"captured_at": datetime.now(timezone.utc).isoformat(), "canonical": audit(args.database),
              "legacy": audit(args.legacy_database) if args.legacy_database.exists() else None,
              "production_writes": False, "api_calls": 0}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
