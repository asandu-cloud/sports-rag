# Data Platform V2

> Converges the bots, website, and live system onto a shared backend and
> service layer. Assumes V1 is in place (see
> [`PLATFORM_V1.md`](PLATFORM_V1.md)); V2 is additive — nothing in V1 changed.

## Goals recap

| Goal | Delivered |
|------|-----------|
| Canonical tables for predictions, parlays, live runtime, KB docs | Yes — 3 new Alembic revisions |
| Service + repository layer for each data domain | Yes — `data_platform/repositories` + `data_platform/services` |
| Drop-in compat shims so bots + website keep working | Yes — `parlay_service.ParlaySessionStore`, `live.live_store.LiveStore`, `prediction_tracker.*` route through Postgres when `PLATFORM_ENABLED=1` |
| Incremental, entity-based KB refresh | Yes — `kb_documents` + `kb_refresh_queue`, `spixctl kb-refresh` |
| Unified API: web + live in one process at `/api/live/*` | Yes — `web_app/routers/live.py` mounted by `web_app/api.py`, plus standalone `live/live_api.py` re-uses the same router |
| Prediction-critical reads don't require Chroma | Yes — fixtures + features are served from Postgres by `FixtureService` / `FixtureRepository` |
| Tests + docs | Yes — `Scripts/tests/data_platform/` covers V1 + V2 |

Chroma stays online; it's clearly downstream now.

## New Postgres tables (V2 revisions)

| Revision | Tables | Purpose |
|----------|--------|---------|
| `0002_predictions_parlays` | `predictions`, `trial_users`, `trial_parlays`, `parlay_sessions` | Replace the corresponding tables in `Index/predictions.db`. Unique dedup key on predictions matches the legacy `log_prediction` behaviour. |
| `0003_live` | `live_fixtures`, `live_state_snapshots`, `live_model_snapshots`, `live_odds_snapshots`, `live_alerts` | Replace `Index/live.db`. Column shapes + indexes mirror the legacy SQLite schema so the live poller runs unchanged. |
| `0004_kb_odds` | `kb_documents`, `kb_refresh_queue`, `odds_snapshots` | Content-hashed doc table for Chroma, refresh queue for entity-based deltas, odds history for CLV + audit. |

## Service + repository layer

```
data_platform/
├── repositories/
│   ├── predictions.py   PredictionRepository   (log / query / grade / CLV)
│   ├── parlays.py       ParlayRepository        (sessions)
│   ├── live.py          LiveRepository          (fixtures / snapshots / alerts)
│   ├── kb.py            KBDocumentRepository    (docs + refresh queue)
│   ├── odds.py          OddsSnapshotRepository  (odds history)
│   ├── fixtures.py      FixtureRepository       (DB-backed fixture reads)
│   └── users.py         UserRepository          (V1)
└── services/
    ├── predictions.py   PredictionService
    ├── parlays.py       ParlayService
    ├── live.py          LiveService
    ├── kb.py            KBService               (orchestrates refresh + Chroma upsert)
    ├── odds.py          OddsService
    └── fixtures.py      FixtureService
```

Every repo accepts a `session_factory` (defaults to `session_scope` from
`data_platform.db`) so tests run in isolated SQLite databases without
touching the module-level engine.

## Compat shims

`PLATFORM_ENABLED=1` routes legacy calls through the repos — no caller
imports changed. Behaviour when the flag is off stays identical to V1.

| Legacy entry point | Platform route | Activation rule |
|--------------------|----------------|-----------------|
| `web_app/users.py` (14 functions) | `UserRepository` | `PLATFORM_ENABLED=1` and no explicit `db_path=` |
| `rag_ingest/core/parlay_service.ParlaySessionStore` | `PlatformParlayStore` | `PLATFORM_ENABLED=1` and default db path |
| `rag_ingest/prediction_tracker.log_prediction` / `get_track_record` / `get_recent_predictions` / `get_daily_breakdown` / `get_calibration_data` | `PredictionService` | `PLATFORM_ENABLED=1` and default db path |
| `live/live_store.LiveStore` | `PlatformLiveStore` | `PLATFORM_ENABLED=1` and db path equals the conventional `Index/live.db` |

Tests that pass explicit `db_path` arguments (e.g. `tmpdir / "live.db"`)
continue to use the legacy SQLite path — this keeps every pre-existing
test passing unchanged.

## Unified API

### One process, both surfaces

```python
# Scripts/web_app/api.py  (runs on :8000)
from web_app.routers.live import router as live_router
app.include_router(live_router)          # /api/live/* now served here
```

### Standalone live process still supported

```bash
# Runs on :8001, same endpoints, uses the same router
cd Scripts/live
python live_api.py
```

Both processes share `web_app/routers/live.py` so only one file defines
the endpoints. The router reads the `LivePoller` singleton for hot data
and falls back to `LiveService` (Postgres) when the poller can't answer.

New endpoint introduced in V2:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/live/fixtures/{id}/alerts` | Historical alert feed, sourced from `live_alerts`. |

Every V1 path stays the same.

## Incremental KB refresh

Entity-based — no more whole-season rebuilds.

1. A change trigger (a fixture finishes, a team profile recomputes,
   etc.) calls
   `KBService.enqueue(entity_type="team" | "player" | "fixture", entity_key=...)`.
   Duplicates inside the queue are coalesced.
2. `spixctl kb-refresh` claims pending items, runs `build_docs_for_entity`
   on each to regenerate the canonical doc text + metadata, then upserts
   into `kb_documents`. Only docs whose `content_hash` actually changes
   are marked for re-embed.
3. Changed docs are shipped to Chroma via the adapter in
   `data_platform/kb/chroma_adapter.py`. Doc ids match the MD5 scheme
   used by `00_normalize_to_docs.py` so existing Chroma rows align with
   the platform rows without re-embedding unchanged content.

```bash
# Populate kb_documents from existing canonical data (one-off after bootstrap)
python -m Scripts.data_platform kb-enqueue-all

# Run incrementally — safe to schedule every N minutes or tack onto refresh
python -m Scripts.data_platform kb-refresh --batch-size 100

# Inspect queue + doc totals
python -m Scripts.data_platform kb-status
```

A `dry_run` flag lets you run the refresher without calling the Chroma
embed + upsert path — useful for verifying content before incurring
embedding costs.

### Triggering refresh from the sync pipeline

`data_platform/kb/refresher.enqueue_after_fixture_change(api_football_id)`
enqueues the fixture + both its teams. Wire this into the existing sync
pipeline (for example, after `upsert_fixture_from_api_row` reports a
`changed=True`) and every fixture-change event becomes a handful of
rows in the refresh queue.

## Migration commands

All idempotent; safe to re-run.

```bash
# Predictions + trial tables
python -m Scripts.data_platform migrate-predictions [--dry-run]

# Parlay sessions + trial_users + trial_parlays
python -m Scripts.data_platform migrate-parlays   [--dry-run]

# Live runtime tables. Cap per-fixture snapshot rows if live.db is huge.
python -m Scripts.data_platform migrate-live --keep-rows-per-fixture 200 [--dry-run]

# Users (V1, kept for completeness)
python -m Scripts.data_platform migrate-users     [--dry-run]
```

`--dry-run` counts rows without writing; re-run without the flag once
you're happy. Running twice is safe: dedup keys on each table resolve
to in-place updates.

## Cloud deployment

Two supported topologies:

### A. Single-process serving

```
PaaS container 1
  uvicorn Scripts.web_app.api:app
  → serves /api/*, /api/live/*, /auth/*, Stripe webhooks
  → reads from PostgreSQL (DATABASE_URL)
  → Chroma for retrieval
```

Good for smaller deployments. The live poller runs inside this process
(set `lifespan` to include `live_lifespan`).

### B. Split processes

```
PaaS container 1 — main API
  uvicorn Scripts.web_app.api:app --port 8000

PaaS container 2 — live API + poller
  uvicorn live_api:app --port 8001

PaaS container 3 — Discord bot
  python Scripts/discord_bot/bot.py

PaaS container 4 — Live Discord bot
  python Scripts/live_discord_bot/bot.py

PaaS container 5 — Scheduled jobs
  spixctl refresh                (existing V1 sync)
  spixctl kb-refresh --batch-size 200
```

Each process connects to the same Postgres + Chroma. Use `PLATFORM_ENABLED=1`
on every container.

### Required env vars

Extends the V1 set — see [`PLATFORM_V1.md`](PLATFORM_V1.md). V2 doesn't
add any new mandatory vars.

```
# Required
DATABASE_URL=postgresql+psycopg://spix:spix@host:5432/spix
PLATFORM_ENABLED=1
API-FOOTBALL-KEY=...
OPENAI_API_KEY=...

# Optional
RAW_ARCHIVE_BACKEND=s3             # from V1 — use S3 for payload archive in cloud
S3_BUCKET=...
DISCORD_TOKEN=...
DISCORD_LIVE_BOT_TOKEN=...
STRIPE_API_KEY=...
```

### Recommended cron schedule

| Job | Cadence | Command |
|-----|---------|---------|
| Incremental data sync | every 15 min during match days | `spixctl refresh --lookback-days 2` |
| KB refresh | every 15 min | `spixctl kb-refresh --batch-size 100` |
| Outcome grading + CLV | every 30 min for the past 2 days | `python Scripts/rag_ingest/prediction_tracker.py --resolve` |
| Full season bootstrap | ad hoc when adding a league/season | `spixctl bootstrap --competition X --season Y` |

## Local development

```bash
# One-time setup
pip install -r requirements.txt
python -m Scripts.data_platform init-db         # builds every table
python -m Scripts.data_platform migrate-users   # pulls users from predictions.db
python -m Scripts.data_platform migrate-predictions
python -m Scripts.data_platform migrate-parlays
python -m Scripts.data_platform migrate-live --keep-rows-per-fixture 100
python -m Scripts.data_platform kb-enqueue-all  # seeds kb_documents from DB
python -m Scripts.data_platform kb-refresh      # upserts deltas to Chroma
export PLATFORM_ENABLED=1

# Run the unified API (web + live)
cd Scripts/web_app
uvicorn api:app --reload

# or run live standalone on port 8001
cd Scripts/live
python live_api.py
```

## Testing

```bash
python -m pytest Scripts/tests/data_platform/
```

V2 additions:

| File | Covers |
|------|--------|
| `test_v2_repositories.py` | Prediction dedup, track record, calibration buckets, parlay round-trip, live snapshots + alerts, KB content-hash dedup + queue coalescing |
| `test_v2_services.py` | Service wiring + KB delta-refresh (two runs, only changed docs re-embed) + shim routing when `PLATFORM_ENABLED=1` |
| `test_v2_api_live.py` | `/api/live/*` router end-to-end via FastAPI TestClient |
| `test_v2_migrations.py` | SQLite → Postgres migrations for predictions, parlays, trial users, live |

All tests stay on in-memory SQLite — no Postgres required locally.

## What's not in V2 (deliberately)

- **Engine rewrite.** `rag_cli_v2.py`, `core/projections.py`, `ml_edge.py`
  still drive projections. They read fixtures + features through the
  existing modules; migrating them to `FixtureService` /
  `FeatureService` is a V3 cleanup.
- **Chroma retirement.** V2 keeps Chroma as the retrieval layer; the
  KB refresh only makes Chroma downstream rather than canonical.
- **Redis cache.** Scaffolded as `data_platform/cache/` was not included —
  the existing TTL caches in `web_app/api.py` and `odds_provider.py` are
  enough for current load. Swap in Redis later without touching any
  repo/service code.
- **Feature formulas.** Preserved exactly — same variable names
  (`control_index`, `cards_per_90_team`, `goals_for_pm`, etc.).

## V3 roadmap (for reference)

- Move the engine modules to read from `FixtureService` /
  `FeatureService` directly; drop the `Output/*` JSON reads.
- Replace Chroma with pgvector if latency profiles support it.
- Wire `enqueue_after_fixture_change` into the incremental sync
  pipeline so KB refresh is fully automatic.
- Introduce Redis for short-lived caching + fan-out of live alerts to
  multiple consumers.
