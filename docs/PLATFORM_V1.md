# Data Platform V1

> PostgreSQL-backed canonical store for Betting RAG. Introduced without breaking
> the existing season-snapshot pipeline, Chroma retrieval, or the Discord / live
> bots.

## Why

The historical setup is a season-snapshot, file-heavy pipeline: every refresh
rebuilds whole-season JSON files, embeds everything into Chroma, and writes
user / subscription / prediction state into `Index/predictions.db` (SQLite).
That doesn't survive the next phase of the product:

- weekly refreshes rebuild whole seasons because nothing tracks what has
  already been ingested
- Chroma ends up being used as a quasi-source-of-truth
- app user state lives in a single SQLite file shared across the bot, the
  website, and the scoring engine
- scaling to more leagues + cloud hosting would amplify every one of the
  above issues

V1 introduces the shared canonical layer without disturbing any of it.

## What ships in V1

```
Scripts/data_platform/
├── config.py          # DATABASE_URL, archive backend, PLATFORM_ENABLED flag
├── db.py              # SQLAlchemy engine + session factory (per-URL cached)
├── models/            # ORM models: competitions, seasons, teams, players,
│                      #   fixtures, per-fixture team/player stats,
│                      #   sync_runs, sync_watermarks, raw_payload_archive,
│                      #   team_feature_snapshots, player_feature_snapshots,
│                      #   app_users, app_subscription_events, app_referrals
├── alembic/           # migrations (0001_initial creates the whole schema)
├── storage/archive.py # ObjectStorage interface + LocalDiskStorage + S3 stub,
│                      #   PayloadArchiver writes metadata rows into Postgres
├── sync/              # API-Football client + incremental/bootstrap pipelines
├── repositories/      # Canonical read/write APIs (users, ...)
├── compat/            # Shims so legacy callers keep working
└── cli.py             # `spixctl` CLI
```

Table summary (all created by `alembic upgrade head`):

| Table | Purpose |
|-------|---------|
| `competitions` | One row per league/cup. Keyed by API-Football id. |
| `seasons` | `(competition, year)` with label `"YYYY/NN"`. |
| `teams` | Global team catalogue, keyed by API-Football id. |
| `players` | Global player catalogue. |
| `fixtures` | One row per fixture. Holds scores, venue, referee, and a `payload_digest` so we can detect changes cheaply. |
| `fixture_team_stats` | Per-fixture per-team stats. Column names match the existing variable vocabulary (`corners`, `shots_on`, `possession`, `yellow_cards`, `expected_goals`, …). `stats_json` carries the full raw-parsed payload for future fields. |
| `fixture_player_stats` | Per-fixture per-player stats. Same pattern. |
| `team_feature_snapshots` | Season-to-date team aggregates at an `as_of_date`. Preserves every in-use variable name (`goals_for_pm`, `corners_pm`, `cards_per_90_team`, `control_index`, `dominance_index`, `aggression_index_norm`, `form_index_team`, venue splits, opponent-joined rates, variance fields, …). |
| `player_feature_snapshots` | Per-player, per-season snapshot (`goals_per_90`, `starter_sot_per_90`, `start_rate`, `minutes_risk`, `form_index`, `aggression_index_norm`, …). |
| `sync_runs` | Lineage row for each sync/build. |
| `sync_watermarks` | Per-scope cursor that drives incremental refresh. |
| `raw_payload_archive` | Metadata for archived API-Football payloads. Bytes live in object storage (local disk today, S3-ready). |
| `app_users` / `app_subscription_events` / `app_referrals` | Canonical app identity. Mirrors the shape of the legacy SQLite tables so the shim can cut over without schema drift. |

## What does _not_ change in V1

- **Chroma remains the KB/retrieval layer** and stays populated by the existing
  `00_normalize_to_docs.py` + `01_embed_and_upsert.py` pipeline. Chroma is now
  clearly downstream of structured data, but we don't force a rewrite in V1.
- **Scoring engine** (`rag_cli_v2.py`, `core/projections.py`, `ml_edge.py`) is
  untouched.
- **Discord + live bots** keep working unchanged.
- **Predictions, parlay sessions, and live tables** still live in
  `Index/predictions.db` and `Index/live.db`. They'll migrate in V2+ using the
  same pattern as users (canonical model + feature-flagged shim).

## Environment

Add to `.env`:

```
# Canonical DB — PostgreSQL in cloud, SQLite works fine locally.
DATABASE_URL=postgresql+psycopg://spix:spix@localhost:5432/spix
# DATABASE_URL=sqlite:///Index/platform.db    # local-only fallback

# Turn the shim on after the DB is bootstrapped and migrated.
PLATFORM_ENABLED=1

# Raw payload archival — local disk by default.
RAW_ARCHIVE_ROOT=Index/raw_archive
RAW_ARCHIVE_BACKEND=local
# RAW_ARCHIVE_BACKEND=s3
# S3_BUCKET=spix-raw
# S3_REGION=us-east-1
# S3_PREFIX=api_football
# S3_ENDPOINT_URL=  # optional, e.g. for R2/minio

# Already set by existing scripts; V1 reads it as API_FOOTBALL_KEY too.
API-FOOTBALL-KEY=...
```

`requirements.txt` gains `SQLAlchemy>=2.0`, `alembic>=1.13`, `psycopg[binary]>=3.1`.
`boto3` is optional and only needed for the S3 archive backend.

## Local setup

```bash
pip install -r requirements.txt

# 1. Create the schema (SQLite locally — set DATABASE_URL to Postgres in prod)
python -m Scripts.data_platform init-db

# 2. Copy users/subscriptions/referrals out of Index/predictions.db (idempotent)
python -m Scripts.data_platform migrate-users --dry-run   # preview
python -m Scripts.data_platform migrate-users

# 3. Optional: bring historical feature snapshots in without hitting the API
python -m Scripts.data_platform build-features --competition EPL LaLiga SerieA Bundesliga Ligue1 --season 2023 2024 2025

# 4. Turn the shim on so web_app/users.py routes through PostgreSQL
export PLATFORM_ENABLED=1
```

## Bootstrap a competition/season

```bash
python -m Scripts.data_platform bootstrap \
    --competition EPL LaLiga SerieA Bundesliga Ligue1 \
    --season 2023 2024 2025
```

What this does:
1. Ensures the competition + season rows exist.
2. Fetches `/fixtures?league=X&season=Y&status=FT` once.
3. Upserts every fixture (keyed on `api_football_id`).
4. Fetches `/fixtures/statistics` and `/fixtures/players` for each fixture and
   upserts them. Payload digests skip any fixture whose row hasn't changed since
   the last fetch, so re-running is cheap.
5. Writes a `sync_runs` row and a watermark at
   `fixtures:<code>:<year>` so incremental refresh can take over from here.

Pass `--no-players` to skip per-player calls on first bootstrap; rerun with
players later. Pass `--no-archive` to disable raw payload archival.

## Incremental refresh

```bash
# All competitions, current season, default 7-day lookback:
python -m Scripts.data_platform refresh

# Specific slice:
python -m Scripts.data_platform refresh --competition EPL --season 2025 --lookback-days 10
```

The pipeline:
1. Reads `sync_watermarks` and computes the `[watermark - 1d, today + 7d]`
   window per `(competition, season)`.
2. Pulls `/fixtures` for that window.
3. Upserts fixtures. Fixtures whose payload digest is unchanged are skipped
   without any downstream calls.
4. For fixtures that changed (new score, new ref, status flipped to FT) we
   refetch `/fixtures/statistics` (+ optional `/fixtures/players`) and upsert.
5. Advances the watermark.

You can schedule this hourly/daily from GitHub Actions, a cron job, or the
discord bot's existing background task surface — the run is idempotent and
cheap once caught up.

## Full active-path season refresh

While Discord and the scoring engine still read the established `Output/` and
Chroma artefacts, use the combined workflow after a matchweek rather than
refreshing only one storage path:

```bash
# Prints the plan only.
python -m Scripts.data_platform refresh-season --season 2026 --dry-run

# Top-five domestic leagues.
python -m Scripts.data_platform refresh-season --season 2026

# Add UCL, UEL, and UECL during active European matchweeks.
python -m Scripts.data_platform refresh-season --season 2026 --include-europe
```

It runs, in order: the canonical incremental sync; the legacy raw-data,
feature-engineering, profile, referee, normalization, Chroma, and cumulative
ML refresh; feature-snapshot import back into the canonical database; then a
health check. A failed stage stops later stages and the same command can be
rerun safely after the issue is fixed. It does not recreate the database or
vector collection.

The final health probe accepts a warning exit code so an unrelated old
watermark is visible in the report without masking a successful selected-slice
refresh. It does stop on a health error. Restart long-lived Discord/web workers
after a successful run so cached model and Chroma data are reloaded.

## Inspect status

```bash
python -m Scripts.data_platform status
```

Prints the configured database URL, archive backend, all watermarks, and the
last 20 sync runs (kind, scope, status, stats).

## Raw payload archive

Every API-Football call made by the platform can be archived (default: on).
Payloads are gzipped JSON:

```
Index/raw_archive/api_football/fixtures/YYYY/MM/DD/HHMMSS_<params12>_<payload12>.json.gz
```

`raw_payload_archive` stores the metadata (provider, endpoint, digests,
URI, fetched_at, owning `sync_run_id`). `PayloadArchiver.load_json(id)`
rehydrates any archived payload — useful for replay/debugging:

```python
from data_platform.db import session_scope
from data_platform.storage import PayloadArchiver

with session_scope() as s:
    payload = PayloadArchiver(s).load_json(42)
```

To switch to S3-compatible storage:

```
RAW_ARCHIVE_BACKEND=s3
S3_BUCKET=spix-raw
S3_PREFIX=api_football
```

`boto3` must be installed (`pip install boto3`). Everything else stays the same.

## Feature snapshots

Two paths are supported in V1:

1. **`build-features` CLI** reads existing `Output/*_feature_engineering/*.json`
   files and upserts team / player snapshots. No API calls, no
   re-feature-engineering. This is the fastest way to bring historical data
   into the DB. Column names preserve the existing vocabulary (`control_index`,
   `dominance_index`, `cards_per_90_team`, `form_index_team`,
   `aggression_index_norm`, all the `*_pm` + venue-split fields).

2. **Direct ORM writes** from new feature builders once the engine is migrated
   to read from the DB. The `upsert_team_feature_snapshot` /
   `upsert_player_feature_snapshot` helpers in `sync/upserts.py` are how code
   should land snapshot rows with lineage (`build_run_id`, `extras.source_file`).

## How the shim works

`Scripts/web_app/users.py` still contains the legacy SQLite implementation.
At the top it checks `PLATFORM_ENABLED`:

```python
def _platform_repo():
    try:
        from data_platform.compat import get_user_repository, is_platform_enabled
    except Exception:
        return None
    if not is_platform_enabled():
        return None
    try:
        return get_user_repository()
    except Exception:
        return None
```

Every public function routes through the repo when the flag is on and
there's no explicit `db_path` override (which tests use to pin a temp SQLite
file):

```python
def upsert_user(discord_id, ..., *, db_path=None, **kwargs):
    repo = _platform_repo()
    if repo is not None and db_path is None:
        return repo.upsert_user(discord_id, ...)
    # legacy SQLite path unchanged below
```

This means:

- Web app, Discord bot, whale chat, referrals — all keep their existing imports
  (`from web_app.users import upsert_user`). No caller code changes.
- When `PLATFORM_ENABLED=0` or SQLAlchemy is unavailable, the legacy SQLite path
  runs exactly as before.
- Explicit `db_path` overrides (test harnesses, scripts targeting an alternate
  DB file) bypass the shim.

## What still lives in SQLite

- `Index/predictions.db` → `predictions`, `parlay_sessions`, `trial_users`,
  `trial_parlays`. These will migrate in V2+ following the same
  (canonical model + shim) pattern used for users.
- `Index/live.db` → `live_fixtures`, `live_state_snapshots`, …. These stay as
  a live-engine-local store in V1; moving them to the platform DB is cheap
  once V1 has bedded in.

## Testing

```bash
python -m pytest Scripts/tests/data_platform/
```

Covers:
- Schema boot / CRUD (`test_schema_boot.py`)
- Upsert idempotency + watermarks (`test_sync_upserts.py`)
- Storage backend + archive metadata + replay (`test_archive.py`)
- User repository + web_app shim (`test_user_repo_and_shim.py`)
- SQLite → Postgres user migration idempotency (`test_sqlite_migration.py`)
- Feature snapshot importer from existing JSON (`test_feature_snapshots_from_files.py`)

Tests run against in-memory SQLite — no PostgreSQL needed locally.

## V1 definition of done

- [x] Canonical schema in Postgres (via Alembic; SQLite compatible for dev).
- [x] Incremental refresh driven by watermarks; no more whole-season rebuilds.
- [x] Raw payload archival (local disk now, S3-ready).
- [x] Feature snapshot tables preserving current variable names.
- [x] Website/app identity can cut over via a feature flag; SQLite remains
      the fallback.
- [x] Chroma untouched; still the retrieval layer.
- [x] Existing Discord + live bots keep working.
- [x] CLI + docs + tests.

## V2+ roadmap (not in this scope)

- Migrate `predictions`, `parlay_sessions`, trial tables off `predictions.db`
  using the same canonical-model + shim pattern.
- Move live runtime state into the platform DB so the website can reuse it.
- Author a DB-native feature builder that reads `fixture_team_stats` /
  `fixture_player_stats` directly instead of re-reading JSON files.
- Stand up a Chroma re-embed pipeline that reads from the platform DB
  instead of `Index/normalized_*.json`, closing the loop.
- Swap the archive backend to real S3 + add payload replay tooling.
