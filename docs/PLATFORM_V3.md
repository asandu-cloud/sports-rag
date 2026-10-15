# Data Platform V3

> Scale, hardening, observability. Assumes V1 + V2 (see
> [`PLATFORM_V1.md`](PLATFORM_V1.md) and
> [`PLATFORM_V2.md`](PLATFORM_V2.md)) — V3 layers on top without
> touching what already works.

## What V3 adds

| Theme | Shipped in V3 |
|-------|---------------|
| Config-driven league onboarding | `data_platform/registry/competitions.yaml` + loader + validation; `sync/apifootball.COMPETITIONS` derives from it |
| Feature versioning + lineage | `feature_builds` table; `feature_version` / `feature_build_id` / `as_of_fixture_api_id` / `built_at` columns on every feature-snapshot table |
| Replay + repair | `data_platform/replay/` — `ReplayPlan`, `ReplayExecutor`, `compare_values`; CLI `spixctl replay` + `spixctl kb-refresh` already incremental |
| Observability + data quality | `HealthReport`, 8 data-quality checks, sync + queue + watermark metrics; `/api/health/platform` endpoint; `spixctl health` |
| Vector backend abstraction | `VectorBackend` protocol, `ChromaVectorBackend`, `InMemoryVectorBackend`, `build_backend()` factory; KB refresher no longer imports Chroma directly |
| Shim completeness | `resolve_outcomes` now routes through `PredictionService` when `PLATFORM_ENABLED=1` — outcome grading reads from + writes to Postgres |
| CLI surface | `registry`, `health`, `replay` subcommands added to `python -m Scripts.data_platform` |
| Tests | 16 new tests across registry, lineage, replay, observability, vector backends |
| Docs | this file |

V1 migrations: `0001_initial`. V2: `0002_predictions_parlays`, `0003_live`, `0004_kb_odds`. **V3: `0005_feature_lineage`**.

## Onboarding a new league

Previously this meant a new `Scripts/{League}/` folder and three bespoke
scripts. Now:

1. Edit `Scripts/data_platform/registry/competitions.yaml`:

   ```yaml
   Eredivisie:
     name: "Eredivisie"
     country: "Netherlands"
     competition_type: "domestic_league"
     tier: 1
     labels: [domestic]
     providers:
       api_football:
         provider_id: 88
         team_aliases: {}
         endpoint_flags:
           fixtures: true
           fixture_statistics: true
           fixture_players: true
   ```
2. Run `python -m Scripts.data_platform registry` to validate.
3. Bootstrap: `python -m Scripts.data_platform bootstrap --competition Eredivisie --season 2025`.
4. Refresh runs pick it up automatically because the registry feeds both
   `COMPETITIONS` and the sync-cadence scheduler.

The legacy `Scripts/{League}/` folders remain so the old file pipeline
keeps working for archival refresh jobs; new leagues don't need one.

## Feature versioning

Every feature-snapshot row now carries:

- `feature_version` — string from `data_platform/features/versions.py`
- `feature_build_id` → `feature_builds.id`
- `built_at` — UTC stamp on last write
- `as_of_fixture_api_id` — snapshot anchor when relevant

Pipeline writers use the `feature_build` context manager:

```python
from data_platform.features import feature_build
from data_platform.sync.upserts import upsert_team_feature_snapshot

with feature_build(scope="EPL:2025", as_of_date=today,
                   pipeline="default",
                   upstream_payload_ids=[123, 124]) as ctx:
    for team in teams:
        upsert_team_feature_snapshot(
            session,
            team=team, season=season, as_of_date=today,
            fields={...},
            feature_build_id=ctx.build_id,
            feature_version=ctx.feature_version,
        )
```

Bumping a pipeline invalidates past snapshots for the same (scope,
as_of_date, version) trio but keeps lineage traceable.

## Replay + repair

```bash
# Plan only — doesn't mutate anything
python -m Scripts.data_platform replay --date-from 2026-04-10 --date-to 2026-04-12 --plan-only

# Specific fixtures
python -m Scripts.data_platform replay --fixture 12345 67890

# Whole competition/season
python -m Scripts.data_platform replay --competition EPL --season 2025

# KB entities
python -m Scripts.data_platform replay --kb-entity team:EPL:42 fixture:12345
```

Programmatic replay (the real API-Football driver for production):

```python
from data_platform.replay import ReplayScope, build_plan, ReplayExecutor
from data_platform.sync.apifootball import ApiFootballClient
from data_platform.kb.refresher import _make_default_upserter  # or custom

client = ApiFootballClient()
scope = ReplayScope(fixture_api_ids=[12345])
plan = build_plan(scope)

def fetch_row(api_id):
    payload = client._get("/fixtures", params={"id": api_id}).get("response") or []
    return payload[0] if payload else None

executor = ReplayExecutor(
    fetch_fixture_row=fetch_row,
    fetch_team_stats=lambda fid: client.fixture_statistics(fid),
    fetch_player_stats=lambda fid: client.fixture_players(fid),
    kb_enqueuer=lambda etype, ekey, reason: 0,  # plug in KBService
)
executor.execute(plan)
```

Drift investigation:

```python
from data_platform.replay.compare import compare_values
diffs = compare_values(stored_row, regenerated_row)
for diff in diffs:
    print(diff.path, diff.expected, diff.actual)
```

## Observability + data quality

### Health endpoint

`GET /api/health/platform` — full report including data-quality scan.
`GET /api/health/platform?quick=1` — skips the expensive checks.

```json
{
  "status": "warn",
  "counts": {"fixtures": 7432, "kb_documents": 61204, "predictions": 3121, ...},
  "sync": {"window_hours": 24, "counts_by_kind": {"incremental": {"completed": 48}}, "latest_run": {...}},
  "queue": {"kb_queue": {"pending": 12, "completed": 9401}, "kb_oldest_pending_at": "..."},
  "watermarks": {"total": 8, "stale": [...], "stale_after_minutes": 120},
  "data_quality": [
    {"check": "missing_odds_coverage", "severity": "warn", "summary": "3 upcoming fixtures ...", "count": 3}
  ],
  "signals": [
    {"name": "stale_watermarks", "status": "warn", "message": "..."},
    {"name": "missing_odds_coverage", "status": "warn", "message": "..."}
  ],
  "env": {"database_driver": "postgresql+psycopg", "platform_enabled": true}
}
```

### Data-quality checks shipped in V3

| Check | Severity | What it flags |
|-------|----------|---------------|
| `duplicate_teams` / `duplicate_players` / `duplicate_fixtures` | error | Two rows with the same `api_football_id` |
| `orphan_team_stats` / `orphan_player_stats` | error | Stats pointing at a missing fixture |
| `finished_fixtures_without_stats` | warn | Finished fixtures in the last 14 days with no team stats |
| `missing_odds_coverage` | warn | Upcoming fixtures in the next 3 days with no `odds_snapshots` row |
| `fixtures_without_competition` | error | Fixtures pointing at missing competition rows |

Add new checks in `data_platform/observability/checks.py` and include
them in `DEFAULT_CHECKS` — they're picked up automatically.

### Metrics

`sync_job_metrics()` — counts of recent `sync_runs` by kind + status.
`queue_metrics()` — KB queue backlog.
`watermark_metrics(stale_after_minutes=120)` — staleness view.

The health endpoint and CLI both read these.

## Vector backend abstraction

```python
from data_platform.vectors import build_backend

backend = build_backend()        # honors VECTOR_BACKEND env (chroma|inmemory)
backend.upsert([{"id": "...", "text": "...", "metadata": {...}}])
backend.query("Arsenal control index", where={"league": "EPL"}, limit=5)
backend.delete(["doc_id_1", "doc_id_2"])
```

`KBService.refresh` / `refresh_kb` use the backend via a thin upserter
closure — no direct `chromadb`/`openai` imports in the KB layer. Tests
set `VECTOR_BACKEND=inmemory` to exercise the full refresh pipeline
with zero external dependencies.

To introduce pgvector:

1. Add `data_platform/vectors/pgvector.py` implementing `VectorBackend`.
2. Extend `build_backend()` in `factory.py`.
3. Set `VECTOR_BACKEND=pgvector` in the environment.
4. Run `spixctl kb-enqueue-all && spixctl kb-refresh` to populate.

Nothing else in the codebase needs to change.

## Completed shim coverage (V2 → V3)

Every production-critical caller of `Scripts/rag_ingest/prediction_tracker`
now routes through Postgres when `PLATFORM_ENABLED=1`:

- `log_prediction`, `log_direct`, `log_from_render_output` — write
- `get_track_record`, `get_recent_predictions`, `get_daily_breakdown`,
  `get_calibration_data` — read
- `resolve_outcomes` — **new in V3** — reads unresolved picks via
  `PredictionService`, runs the existing grading logic against the
  API-Football fetch helpers, and writes outcomes back via the service

Capture of closing odds still uses the legacy SQLite helper directly —
not in the primary hot path — and is scheduled for V4 because it's
intertwined with odds-history refactoring.

## Cloud topology

Three-process topology recommended for production:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PostgreSQL (canonical, multi-AZ)                                        │
│  S3 (raw payload archive, RAW_ARCHIVE_BACKEND=s3)                         │
│  Chroma / pgvector (vector backend)                                      │
└──────────────────────────────────────────────────────────────────────────┘
        ▲               ▲                        ▲                     ▲
        │               │                        │                     │
┌───────────────┐ ┌─────────────┐ ┌────────────────────────┐ ┌───────────────┐
│ API + Web     │ │ Live engine │ │ Discord bots           │ │ Scheduled     │
│ uvicorn       │ │ uvicorn     │ │ main + live            │ │ worker(s)     │
│ api:app :8000 │ │ live_api    │ │ python bot.py          │ │ spixctl       │
│ /api/*        │ │ :8001       │ │                        │ │ refresh,      │
│ /api/live/*   │ │             │ │                        │ │ kb-refresh,   │
│ /api/health/* │ │ LivePoller  │ │                        │ │ resolve       │
└───────────────┘ └─────────────┘ └────────────────────────┘ └───────────────┘
```

### Startup order (safe)

1. `DATABASE_URL`, `PLATFORM_ENABLED=1` set on every process.
2. **Migrations first** — container init runs
   `python -m Scripts.data_platform init-db` before any app starts.
3. `spixctl health --quick` as a readiness probe; fail the rollout if
   status is `error`.
4. Start scheduled workers (they read from a stable DB).
5. Start API + live processes.
6. Start Discord bots last; they call into the API and require the live
   poller to be up for `/api/live/*` fallbacks.

### Env matrix

| Process | Required | Optional |
|---------|----------|----------|
| API | `DATABASE_URL`, `PLATFORM_ENABLED=1`, `OPENAI_API_KEY`, `API-FOOTBALL-KEY`, `STRIPE_API_KEY` | `VECTOR_BACKEND`, `CORS_ORIGINS` |
| Live API | same as API plus `DISCORD_LIVE_BOT_TOKEN` if co-located | `LIVE_API_PORT` |
| Discord bots | `DATABASE_URL`, `PLATFORM_ENABLED=1`, `DISCORD_TOKEN`, `API-FOOTBALL-KEY` | bot channel ids |
| Scheduled workers | `DATABASE_URL`, `PLATFORM_ENABLED=1`, `API-FOOTBALL-KEY`, `OPENAI_API_KEY` (for kb-refresh) | `RAW_ARCHIVE_BACKEND`, `S3_*` |

## New CLI surface (V3)

```bash
# Registry
python -m Scripts.data_platform registry                     # full dump
python -m Scripts.data_platform registry --label top5
python -m Scripts.data_platform registry --type continental_cup

# Health
python -m Scripts.data_platform health                       # full report
python -m Scripts.data_platform health --quick               # skip DQ scan
# Exit codes: 0 = ok, 1 = warn, 2 = error

# Replay
python -m Scripts.data_platform replay --plan-only \
  --date-from 2026-04-10 --date-to 2026-04-12 --competition EPL
python -m Scripts.data_platform replay --fixture 123 456
python -m Scripts.data_platform replay --kb-entity team:EPL:42 fixture:12345
```

Combined with V1 (`init-db | bootstrap | refresh | build-features |
migrate-users | status`) and V2 (`migrate-predictions | migrate-parlays |
migrate-live | kb-enqueue-all | kb-refresh | kb-status`), the platform
CLI is now the operator's primary interface.

## Testing

```bash
python -m pytest Scripts/tests/data_platform/
```

V3 additions:

| File | Covers |
|------|--------|
| `test_v3_registry.py` | Default registry validity, label filtering, team alias lookups, rejection of invalid YAML, registry→COMPETITIONS parity |
| `test_v3_features_lineage.py` | Feature-build context manager, lineage on snapshot writes, idempotent `start_feature_build` |
| `test_v3_replay.py` | Planner dedup, executor against a stub fetcher, `compare_values` tolerances |
| `test_v3_observability.py` | Clean DB health report, finished-fixture-without-stats detection |
| `test_v3_vectors.py` | Round-trip on InMemoryVectorBackend, factory env switch, KB refresher via the abstraction |

All 47 data_platform tests pass alongside the 36 pre-existing tests = 83 total.

## What V3 deliberately does not change

- **Engine code** (`rag_cli_v2.py`, `core/projections.py`, `ml_edge.py`)
  still drives projections. Migrating it to `FixtureService` /
  `FeatureService` is a V4 cleanup.
- **Legacy `Scripts/{League}/` scripts** remain available. The
  registry-driven sync in `data_platform/sync/` supersedes them for new
  leagues; historical scripts stay because some operators still run them
  for full-season rebuilds.
- **`capture_closing_odds`** hasn't been shimmed through the platform
  yet — scheduled for V4 alongside multi-bookmaker odds history.

## V4 roadmap

- Rewrite the engine to read from `FixtureService` / `FeatureService`
  directly; retire all `Output/*` JSON reads for serving paths.
- pgvector backend + benchmark vs Chroma.
- Multi-bookmaker odds history with per-bookmaker movement analysis.
- Redis layer for cross-process caches, fan-out of live alerts, and
  distributed locks on scheduled workers.
- Signed-URL + presigned-upload workflow for the S3 archive.
