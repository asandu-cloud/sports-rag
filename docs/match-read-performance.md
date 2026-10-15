# Match Read reliability — steps 1–4

Steps 1–2 implemented 19 September 2026; steps 3–4 implemented 21 September.
This records the measurement, retrieval, scheduling and refresh-coordination
parts of the six-step reliability plan. Alerts, website availability states
and real busy-matchday acceptance remain outstanding.

## What changed

Generation now reports elapsed time for base odds retrieval, enrichment,
lineups, shared input quality, team profile context, each market, compilation,
briefing cache/rewrite and persistence. Resolver collection reads, profile
cache hits/misses, opponent cache hits/misses and batch fallbacks are counted.
The worker records website-release time separately.

Measurements are emitted as `Match Read performance` JSON in the worker log,
included in CLI output and saved in `sync_runs.stats.fixture_performance`.
An interrupted fixture retains its partial measurements, including which
stages failed. These measurements do not enter immutable Match Read identity
or numerical result payloads. Timings are **inclusive**: Chroma time is part of
profile time, which is part of market evaluation time. Do not sum nested times.
Stage invocation counts are not API-Football billing/request counts. Chroma
counts cover the team resolver, not every possible player/lineup lookup.

The market evaluator now computes its two-team quality audit once per fixture.
Each market receives an independent copy so variance and guardrail fields
cannot leak between markets. Standalone single-market evaluation still builds
its own audit, and callers cannot override it using display context.

The resolver already cached completed profiles. The costly cold-cache path
looked up opponents for historical fixtures individually. It now fetches those
rows in batches scoped to league, season and requested fixture labels, with
pagination. In-memory matching preserves the existing alias, fixture date,
season, first-match and missing-value semantics. A failed batch falls back to
the established individual lookup; it does not cache unread rows as missing.

Local derived caches are invalidated at the next evaluation boundary when
Chroma's SQLite file or WAL revision changes (inode, modification time, size).
Nested calls share that boundary. Concurrent synchronous callers in one process
are serialized around the shared cache, preventing an older request from
repopulating a newer cache. For remote Chroma, where this local revision cannot
be checked, reuse is limited to one evaluation boundary. This does not provide
an atomic snapshot across ingestion, Chroma and platform.db; coordinating those
writers is now covered for the supported local CLI/worker workflow by the
step-4 gate below; direct legacy script invocations do not acquire that gate.

## Baseline and verification

Before changing retrieval, six real profile cases and their metadata inputs
were captured in:

`Index/prediction_experiments/match-read-performance-2026-09-19/`

The capture contains 5,155 team/profile documents from EPL, Bundesliga,
Eredivisie and UCL, the original resolver source, expected results, and checksums.
Cases cover Brentford, Chelsea, Mainz, Frankfurt and Ajax, including Ajax's
domestic context for a UCL prediction. Capture reads Chroma metadata through a
read-only SQLite transaction and never fetches providers or embeds documents.

Replay after the change:

- All six effective profiles, profile audits and recent-stat outputs match exactly.
- Collection reads fell from **421 to 45** on identical frozen inputs: **89.3% fewer**.
- A separate check against the actual local Chroma database matched all six
  cases and completed in **6.467 seconds**, making 48 resolver collection reads.
  This was one local observation, not a before/after latency ratio or a complete
  Match Read cycle benchmark. It excludes odds, briefing and publication.
- Focused regression verification: **79 tests and 10 subtests passed**, covering
  profile/market parity, seasonal filtering, batching/pagination, refresh
  invalidation, timeout telemetry, dispatch, worker, compiler and public release.

The saved profile replay is narrower than a complete live prediction replay.
Existing market regression fixtures separately verify selection behaviour.
The numerical projection formulas, probability models and selection rules were
not changed by this implementation.

Repeat the frozen profile comparison:

```bash
./.venv/bin/python Scripts/ops/profile_performance.py verify \
  --directory Index/prediction_experiments/match-read-performance-2026-09-19
```

Inspect performance from actual scheduled runs:

```bash
rg 'Match Read performance' Index/logs/match-read-worker.stderr.log | tail -n 20
```

The scheduled command starts a new Python process for each invocation and picks
up the changes on its next start. A process already running retains its loaded
code. No scheduler configuration was changed or service manually restarted as
part of these two steps.

## Remaining acceptance and implementation

- Measure the new scheduling and statistical cache on complete live cycles;
  passing offline tests is not a claim of continuous public availability.
- Step 5: failure alerts and accurate delayed/expired website states.
- Step 6: busy-day replay and two real matchdays of operational acceptance.
- Upstream ingestion now has separate local schedules (see the follow-up
  below). The Match Read worker itself remains database-only when planning.

The laptop must still be awake and online to run its local scheduler. These
changes do not establish continuous service while it is unavailable.

## Step 3 — scheduling, retries and statistical reuse

- Work is executed fixture by fixture: urgent missing pre-match fallbacks
  first, other final-window work next, then missing early cards and routine
  early refreshes. Within a class, never-attempted/oldest-attempted work comes
  first, with kickoff and fixture identity as deterministic tie-breakers.
- Failed checks record a durable consecutive-failure count and next retry
  timestamp. With default settings, retries back off for 10, 20, then at most
  30 minutes. A successful check or pending-lineup observation resets that
  failure sequence. Pending lineups retain their ordinary polling cadence.
- The cycle does not begin another fixture when it lacks a full dispatch
  allowance (normally 120 seconds). It yields remaining work without logging
  a fake fixture failure caused by a one-second leftover cycle budget.
- Scheduled confirmed-XI work checks the canonical fixture ID and team names
  before loading odds. Pending XIs perform no odds fetch, enrichment, market
  evaluation or publication. Available contexts are passed into the existing
  dispatcher without a duplicate lineup request. When XIs exist, the current
  context provider still loads player contributions; that work is necessary
  for the existing amendment, not a new prediction formula.
- Persisted Match Read generation can reuse statistical intermediates from
  `Index/cache/match_read_projections.sqlite3`: profile-quality evidence,
  projections and variances. All odds extraction, probability/edge/EV selection,
  confidence/release guardrails, compilation and provenance run again.
- Reuse is scoped to fixture identity, kickoff/date, requested markets and
  model/lineup contexts. Local Chroma metadata/WAL, model files, feature inputs,
  code/config files and environment fingerprints invalidate it. A six-hour
  expiry and UTC-date change also invalidate reuse; hits do not extend expiry.
  Environment values are hashed and never stored as plaintext in this cache.
- Unknown/remote Chroma revisions, unsupported contexts, incomplete evaluations,
  interruptions and cache read/write failures fall back to normal computation.
  Statistical cache hits/misses appear in the existing performance counters.

The cache contains neither selected bets nor odds. No model weights, seasonal
blending rules, probability formulas or bet-selection thresholds were changed.

## Step 4 — local database and refresh safety

The scheduled worker and supported ingestion CLI commands now share a
file-backed cross-process gate next to the configured SQLite database. A
worker holds a shared gate through one fixture's generation and publication;
`refresh-season` holds an exclusive gate across its entire child-command chain.
Child commands inherit the actual lock descriptor. No database transaction
is used as this long-lived gate.

Standalone `refresh`, `sync-schedule`, `bootstrap`, feature-build and KB refresh/enqueue commands
also take the exclusive gate. Manual `match-reads` and `publish-match-reads`
commands take the shared gate. Refresh commands wait up to 180 seconds to obtain
the gate, then exit with a retry message; workers defer instead of overlapping.
A waiting writer closes a separate entrance gate to new fixture readers so a
busy generation cycle cannot repeatedly reacquire the lock and starve refresh.
If a refresh happens between fixtures, the cycle yields to a new process so
all module-level inputs can reload. Existing published cards are not deleted;
their normal freshness/expiry policy continues to apply during a refresh.

A full refresh writes an incomplete marker before it starts. Only a successful
full refresh clears it. If a stage fails or the process is killed, generation
remains blocked until `refresh-season` succeeds. An isolated raw-data refresh
does not pretend to repair a failed multi-stage refresh. Do not manually remove
the marker as a substitute for completing the refresh.

Canonical incremental sync now commits short fixture checkpoints, with
API-Football fixture/statistics/player requests outside write transactions.
A finished fixture awaiting details retains a NULL digest until its details
are successfully stored, allowing the next run to retry without deleting old
rows. Failed detail imports mark the run partial, return a nonzero CLI exit
and do not advance that competition's watermark. Existing all-or-nothing
library callers retain their previous transaction mode unless they opt in;
the supported `refresh` CLI enables short transactions.

SQLite connections wait at most two seconds per lock attempt. Short worker
ledger, lease and Match Read writes retry lock/busy failures at most three
times using fresh transactions. Provider calls are not repeated by this retry
wrapper. Lease takeover/release use conditional updates; competing workers
cannot both acquire an expired lease. Session cleanup rolls back on timeouts
as well as ordinary exceptions. Failures to save the final run report or release
a lease are returned/logged rather than causing a second uncaught traceback.

This implementation targets the current single-host, file-backed SQLite
deployment. It does not enable WAL or change existing database files' journal
mode. Direct legacy ingestion scripts/interactive bot evaluations outside
these entry points are not globally coordinated. Multiple cloud hosts or
PostgreSQL need a distributed gate before this local coordinator is replaced.

### Verification on 21 September

- Focused regression suite: **109 tests and 4 subtests passed**. The command
  below runs the new reliability tests; the larger run also included existing
  worker, dispatcher, profile, market, lineup, persistence and release tests.
- New tests cover queue fairness, budget admission, persisted retry backoff,
  pending-lineup short-circuiting, cache-vs-cold equality after repricing across
  all seven markets, cache invalidation/expiry, cross-process gate inheritance,
  failed-refresh recovery, atomic lease races, cleanup failures and incremental
  player-detail retry/idempotency without holding API-time write locks.
- Local data fingerprint check: available and stable; approximately 0.43 seconds
  on first import and 0.004 seconds when warm. This is not a full-cycle benchmark.
- Read-only live planning succeeded but found **zero upcoming fixture stages**.
  The September 21 incremental refresh is recorded as completed (96 fixtures
  seen, 88 changed, no reported errors), yet the stored schedule still ends on
  September 20. Investigate upcoming fixture coverage before live acceptance;
  these changes cannot create a schedule absent from the canonical database.
- The installed scheduler remains loaded in `website` mode. No live generation,
  manual publication, service restart or provider request was performed for
  this verification.

```bash
./.venv/bin/python -m pytest -q \
  Scripts/tests/data_platform/test_worker_reliability.py \
  Scripts/tests/data_platform/test_incremental_transactions.py \
  Scripts/tests/test_projection_cache.py
```

## Upstream automation follow-up — 21 September 2026

The earlier interpretation of the empty schedule needed correction. Archived
September 21 fixture responses did not contain later matches; SQL was not
dropping them. Two live EPL checks confirmed a valid empty September 21–28
window and the next match on October 10. Checking all 13 competitions through
October 26 returned **392 fixtures**, earliest October 9. Thus the current
four-day public board legitimately has no eligible fixtures.

Implemented and applied:

- `sync-schedule` imports the next 35 days without fetching match/player stats,
  rebuilding profiles, embedding or generating predictions. Separate
  `schedule:<league>:<season>` watermarks record checked coverage, including
  valid empty responses. A schedule check never consumes the finished-fixture
  digest that the statistical importer needs to detect and import new results.
- Routine incremental refresh also looks ahead 35 days, still using one
  fixtures request per competition/season. Historical rows remain intact.
- HTTP 200 API-Football error envelopes fail explicitly rather than silently
  becoming successful empty calendars. Failed checks do not advance coverage.
- The worker excludes cancelled/postponed fixtures from generation and retains
  its existing 96-hour prediction window; a longer stored schedule is not
  permission to publish bets 35 days ahead.
- Installed `com.bettingrag.fixture-schedule`: hourly invocation, making
  requests only when six-hour coverage expires. All 13 competitions are
  included by default; no hardcoded season year is installed.
- Installed `com.bettingrag.daily-data-refresh`: the joined additive refresh
  daily at 07:15 local Mac time. It does not run immediately on install.
  This includes results/player stats, features, profiles and delta embeddings,
  not active-model retraining. Its first scheduled execution remains to be
  observed; the user's preceding manual full refresh had already succeeded.
- Existing `com.bettingrag.match-read-worker` stays loaded in `website` mode
  at ten-minute intervals. No release-mode or prediction logic change.
- Writer admission now has a separate entrance lock: a queued refresh takes
  priority at the next fixture boundary. CLI writers wait at most 180 seconds,
  enough for an existing bounded dispatch to finish. Supported ingestion jobs
  also take a temporary macOS idle-sleep assertion while running.

Live verification:

- Final targeted regression suite: **128 tests and 4 subtests passed**;
  `git diff --check` clean.
- Schedule run **171**: 13 logical provider calls, 392 future fixtures stored,
  zero reported errors. Schedule run **173**, started by launchd, skipped all
  13 fresh coverages and made **zero** provider calls.
- Both upstream jobs report installed/loaded; existing website worker remains
  installed/loaded. A real website-mode cycle (**174**) completed with no errors
  and no due work, correctly reflecting the current calendar gap.
- A **read-only** plan evaluated at October 5 18:00 UTC selects three real
  stored October 9 fixtures within its four-day window. This diagnostic did
  not fetch odds, generate predictions, write reads or publish future cards.
- An isolated end-to-end test feeds synthetic provider/prediction inputs into
  the real schedule loader, dispatcher, compiler, persistence/release bridge and
  website HTTP router. It verifies an early card and replacement by a verified
  lineup amendment with one visible card and retained history. No test fixture
  or recommendation was written to the live website/database.

**Still pending:** real provider-backed card/lineup acceptance when fixtures
enter the outlook (earliest currently October 5). Calendar absence prevents
that live test now; the isolated test is not a claim of live-match acceptance.
Failure alerts and clearer website calendar/delay states remain steps 5–6.
The local jobs still depend on the Mac and network; cloud deployment is separate.

Operations and log paths are in [terminal commands](terminal-commands.md).
