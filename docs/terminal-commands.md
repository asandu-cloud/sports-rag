# Betting RAG — Terminal Command Reference

Run these commands from the project directory:

```bash
cd ~/Desktop/Betting_RAG
```

Use the project's virtual-environment interpreter (`./.venv/bin/python`) in every command. It avoids accidentally using a different Python installation.

## First-time checks

Activate the environment only if you want shorter commands in an interactive terminal:

```bash
source .venv/bin/activate
```

Install or update the Python dependencies when setting up a new machine:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Create or safely upgrade the canonical database schema. This is additive; it does not delete `platform.db`:

```bash
./.venv/bin/python -m Scripts.data_platform init-db
```

Confirm the configured competitions and perform a quick local health check:

```bash
./.venv/bin/python -m Scripts.data_platform registry
./.venv/bin/python -m Scripts.data_platform health --quick
```

Before running live refreshes or bots, ensure `.env` contains the required secrets. Do not commit this file or paste secret values into chat:

- `API_FOOTBALL_KEY`
- `OPENAI_API_KEY`
- `DISCORD_BOT_TOKEN` (for the main bot)
- `DISCORD_LIVE_BOT_TOKEN` (only for the optional live bot)

## Weekly data refresh

This is the normal command to run after a gameweek. It incrementally refreshes
all ten public domestic leagues for 2026/27 without deleting or recreating the
canonical database, legacy Output data, or Chroma indexes:

```bash
./.venv/bin/python -m Scripts.data_platform refresh-season \
  --season 2026 --include-europe
```

For a domestic-league-only refresh, omit `--include-europe`; this still
includes the Championship, Süper Lig, Eredivisie, Primeira Liga, and Jupiler
Pro League alongside the established top five:

```bash
./.venv/bin/python -m Scripts.data_platform refresh-season --season 2026
```

To inspect what the full refresh would do without writing anything:

```bash
./.venv/bin/python -m Scripts.data_platform refresh-season \
  --season 2026 --include-europe --dry-run
```

`--season 2026` means the 2026/27 campaign, because API-Football identifies a season by its starting year. Use `--season 2027` for 2027/28.

As of the Phase 1 model migration, this command refreshes data, profiles,
features and embeddings **without retraining or replacing saved ML models**.
Candidate fitting is now separate and cannot publish or promote a model. The
old `--skip-training` option is retained for compatibility but is no longer
necessary. Direct `pull_season.py --retrain-ml` and legacy `ml_edge.py`
training/CV/raw-row prediction commands are retired.

For the new export, replay, and candidate-only commands, see
[Phase 1 prediction features](prediction-features.md). No model training is
required just to update Match Reads with fresh statistics.

Phase 3 now has a separate isolated benchmark runner. The
[Batch B report](phase3-benchmark-runner-2026-09-26.md) records its research
environment and bounded smoke commands. Verify the already completed run
without training or provider requests:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py verify-run \
  --experiment Index/prediction_experiments/phase3-runner-smoke-2026-09-26
```

That smoke is development software validation. Batch C now has a separate
forward-development command governed by its
[frozen protocol](phase3-forward-development-protocol-2026-09-26.md).
Run from the repository root, using the isolated environment. Preparation
reads frozen local data and produces a new immutable variant sidecar:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py prepare-variants \
  --dataset Index/prediction_experiments/phase3-dataset-2026-09-26 \
  --experiment NEW_VARIANT_EXPERIMENT
```

The declared grid, earlier-only OOF blends and bounded paired ablations run
together. Use a new experiment name for a new specification:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py develop \
  --dataset Index/prediction_experiments/phase3-dataset-2026-09-26 \
  --baselines Index/prediction_experiments/phase3-baselines-2026-09-26 \
  --variants Index/prediction_experiments/phase3-feature-variants-2026-09-26 \
  --experiment NEW_FORWARD_EXPERIMENT
```

To continue an interrupted run, repeat its exact command and experiment name
with `--resume`. Completed task checksums, source/data contracts and the
dependency lock must still match. Changed code requires a new experiment.
These commands do not retrieve provider data, create embeddings, open reserved
test outcomes or publish/promote candidates.

The [completed Batch C report](phase3-forward-development-2026-09-26.md)
identifies the accepted corrected run. Verify it without fitting:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py verify-run \
  --experiment Index/prediction_experiments/phase3-forward-development-v2-2026-09-26
```

The earlier `phase3-forward-development-2026-09-26` attempt was invalidated
for a numerical preprocessing defect and preserved incomplete. Do not resume
that old specification with the corrected code.

The bounded Batch C extension follows its separate
[frozen protocol](phase3-extension-protocol-2026-09-27.md). It reuses the same
dataset and baselines, compares against saved Batch C predictions, and caps
actual estimator attempts at 78. From the repository root:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py extend \
  --dataset Index/prediction_experiments/phase3-dataset-2026-09-26 \
  --baselines Index/prediction_experiments/phase3-baselines-2026-09-26 \
  --batch-c Index/prediction_experiments/phase3-forward-development-v2-2026-09-26 \
  --experiment NEW_EXTENSION_EXPERIMENT
```

Append `--resume` only to continue the same extension specification. Failed or
interrupted estimator attempts count toward its persistent 78-fit cap. No
parameter expansion, reserved test evaluation or public activation is automatic.
Verify the recorded extension without fitting:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py verify-run \
  --experiment Index/prediction_experiments/phase3-extension-2026-09-27
```

Completed older experiments retain their source fingerprints. Adding new
benchmark modules changes the current source specification; use integrity
verification for those completed runs, or their archived source for exact
execution reproduction. Do not update their saved hashes to bypass a mismatch.

Batch D's [one-time confirmation is complete](phase3-confirmation-2026-09-27.md).
All three candidates failed at least one agreed gate; no model was promoted.
Verify its saved artifacts without fitting or scoring:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py verify-run \
  --experiment Index/prediction_experiments/phase3-confirmation-2026-09-27
```

The exact completed operation can also be resumed for integrity verification:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py confirm \
  --dataset Index/prediction_experiments/phase3-dataset-2026-09-26 \
  --baselines Index/prediction_experiments/phase3-baselines-2026-09-26 \
  --batch-c Index/prediction_experiments/phase3-forward-development-v2-2026-09-26 \
  --extension Index/prediction_experiments/phase3-extension-2026-09-27 \
  --experiment phase3-confirmation-2026-09-27 --resume
```

It performs no new inference/scoring once complete. This command consumed
2024, recorded in the permanent calendar-period inspection registry. Do not
rename the experiment or re-export data to treat 2024 as an untouched test.
The original freeze and documented source-only round-label repair both remain
required for exact resumption. Later calibration/final-system reserves were
not opened. The `verify-run` command remains usable after later source changes.

The one-off 2025 canonical history recovery was applied on September 22; do not
repeat it as part of weekly refreshes. Its audit/import/replay commands and
remaining card-stat limitations are in the
[Phase 1 recovery report](phase1-history-recovery-2026-09-22.md).

The additional existing-history import on September 25 added 5,555 verified
fixtures while preserving pre-existing rows. See the
[import and exclusion report](local-history-import-2026-09-25.md).

The eight-season expansion was collected and imported on September 26,
adding 17,233 accepted fixtures. See the
[coverage and exclusion report](eight-season-history-import-2026-09-26.md).
No collection command needs to be rerun for those completed cohorts.
The original launcher commands below are retained for reference; the dry run
describes the 50 planned competition-season pairs without downloads or writes:

```bash
./.venv/bin/python -m Scripts.ops.prediction_history_download --dry-run
```

To collect those fixtures and team statistics using the existing bootstrap
pipeline, with raw archives and a separate staging database:

```bash
./.venv/bin/python -m Scripts.ops.prediction_history_download --execute
```

This targets provider years 2019–2022 for the twelve competitions other than
UECL, and 2021–2022 for UECL. It skips players and never imports into the live
database. Results are stored under
`Index/history_staging/eight-seasons-2026-09-25/` for validation before import.
Rerunning verifies and skips completed seasons; interrupted/failed seasons are
retried in full. Per-fixture errors stop collection even if the underlying CLI
exits zero. Successful collection does not certify complete statistics or
regulation-time eligibility. This is not the deferred statistics backfill for
already represented seasons.

As of reliability steps 3–4, the supported refresh commands coordinate with
the local Match Read worker. An active refresh makes generation defer; an
interrupted full refresh blocks generation until a successful rerun of the
same `refresh-season` command. Do not delete the database or its
`.refresh-incomplete` marker to bypass that protection. A refresh can exit
with a retry message if it cannot acquire the gate within 180 seconds. Waiting
refreshes take priority at the next fixture boundary.

### Refresh timeout and blockage protection (25 September 2026)

API-Football refresh downloads now use a 5-second connection timeout,
30-second socket-read timeout and at most three attempts for transport errors,
HTTP 429 and server errors. Authentication/API-envelope errors fail immediately.
Legacy collectors no longer swallow a failed fixture download and then save a
partial season. A rebuild that loses previously saved fixture/team coverage
also fails before replacing those files. Empty provider coverage is logged;
missing statistics are not invented. Sparse but structurally valid player
coverage (seen in historical UEFA qualifiers) is explicitly warned about,
not silently reclassified as a network failure or a new model-eligibility rule.
Provider corrections to player IDs and minutes remain permitted.

Each child of `pull_season.py` has a 30-minute wall-clock deadline; each outer
`refresh-season` stage has a two-hour deadline. These are ceilings, not polling
intervals. Timed-out stages clean up their process group and exit nonzero.
The full-refresh marker remains until a complete rerun succeeds.

Optional `.env` settings (seconds; positive finite values):

```dotenv
REFRESH_CHILD_TIMEOUT_SECONDS=1800
REFRESH_STAGE_TIMEOUT_SECONDS=7200
REFRESH_ALERT_AFTER_SECONDS=3600
# Set locally to a PRIVATE operations Discord webhook; never commit the URL:
# REFRESH_ALERT_WEBHOOK_URL=...
```

The existing worker ticks check for prolonged blockage even while generation
is deferred. They persist an alert beside the database at
`Index/platform.db.refresh-incomplete.alert.json`, emit an error in worker logs,
and surface a warning in `health --quick`. With the private webhook configured,
they also notify Discord (no mentions), repeat at most every six hours, retry
failed delivery no more than once per five minutes, and send a recovery notice.
No public channel or existing live-bet webhook is reused. Without the URL,
only the local alert/log/health warning is active. No extra scheduler is needed,
but these checks still require this Mac and the scheduled workers to be running.

If the full refresh fails, fix the reported cause and rerun the normal command.
Do not remove the marker or kill unrelated bots to force worker activity.
The canonical database uses additive upserts; legacy current-season aggregate
files are rebuilt only after collection succeeds. Older seasons are retained.

The worker picks up code changes on its next fresh scheduled invocation.
The numerical projection cache is automatic and needs no rebuild command.
Upstream schedules were installed in the follow-up below. They do not keep a
sleeping laptop running; see [reliability details](match-read-performance.md).

### Automatic upstream refresh (installed 21 September 2026)

The local Mac has these scheduled jobs (measurement was added on 24 September):

| Job | Cadence | Work |
| --- | --- | --- |
| Match Read worker | Every 10 minutes | Generate/recheck/publish eligible cards within the next 96 hours; check confirmed lineups near kickoff. |
| Prediction measurement | Every minute | Check due prospective closing-price/settlement tasks within its request budget; defer during refresh. |
| Fixture schedule | Hourly check; fetch when coverage is 6 hours old | Store the next 35 days for all 13 competitions. No player calls or embeddings. |
| Joined data refresh | Daily at 07:15 local Mac time | Run `refresh-season --include-europe`: results, player stats, features, profiles and delta embeddings; preserve saved ML models. |

The schedule-only fetch normally costs 13 API-Football calls per six-hour
refresh (about 52/day, excluding retries or season rollover). The daily full
refresh has additional variable API/embedding usage depending on new data.
These jobs use the existing `.env`; no credentials are placed in LaunchAgents.

Inspect the installed upstream jobs:

```bash
./.venv/bin/python Scripts/ops/upstream_refresh_launchd.py status
```

On a new Mac, install them with:

```bash
./.venv/bin/python Scripts/ops/upstream_refresh_launchd.py install
```

Refresh only upcoming fixtures immediately, retaining historical data:

```bash
./.venv/bin/python -m Scripts.data_platform sync-schedule --days-ahead 35
```

To stop/remove only these two upstream jobs (data and logs are retained):

```bash
./.venv/bin/python Scripts/ops/upstream_refresh_launchd.py uninstall
```

Logs live in `Index/logs/fixture-schedule.*.log` and
`Index/logs/daily-data-refresh.*.log`. The daily job does not run immediately
on installation; its first scheduled time is 07:15. It still depends on this
Mac being available and online; this is not cloud hosting. The refresh runs under
`caffeinate -i -s`, so the Mac cannot idle-sleep until it finishes (system sleep is
also held on mains power). A Mac that is already asleep at 07:15 runs the job at its
next wake; a scheduled wake (`sudo pmset repeat wakeorpoweron MTWRFSU 07:05:00`)
avoids that. Failed full refreshes
retain the incomplete marker and need a successful rerun before cards resume.

**Calendar gaps are valid:** the live provider check on September 21 returned
392 future fixtures, with the earliest on October 9. Consequently, no cards
are due within the current four-day window. Extending stored schedules does
not publish predictions 35 days ahead or widen the existing card policy.

### Recovery-only platform refresh

Use this only when repairing a specific recent canonical-data gap. It is not the normal weekly refresh, since it does not perform the whole joined legacy/model refresh:

```bash
./.venv/bin/python -m Scripts.data_platform refresh \
  --competition EPL LaLiga SerieA Bundesliga Ligue1 Championship SuperLig Eredivisie PrimeiraLiga BelgianProLeague \
  --season 2026 --lookback-days 7
```

## Prospective settlement and closing-price measurement

This is independent of the Match Read publisher and Discord. The local job was
installed on 2026-09-24 with a prospective boundary after recommendation ID 467.
It does not backfill the old recommendation history or change model predictions.
See [implementation and limitations](phase2-measurement-automation-2026-09-24.md).

Check the local scheduler and its work/health:

```bash
./.venv/bin/python -m Scripts.ops.measurement_launchd status
./.venv/bin/python -m Scripts.data_platform measurement-status
```

Inspect eligible work **offline**, without requests or database writes:

```bash
./.venv/bin/python -m Scripts.data_platform measurement-cycle --once --dry-run
```

Run one bounded real cycle manually (normally the scheduler handles this):

```bash
./.venv/bin/python -m Scripts.data_platform measurement-cycle --once
```

Optional narrower passes use `--settlement-only` or `--closing-only`. Unlike the
older resolver dry-run, the measurement-cycle dry-run is fully offline. The real
cycle can use API-Football and write prospective grading/price evidence.

Install/reinstall on this Mac, or pause without deleting evidence:

```bash
./.venv/bin/python -m Scripts.ops.measurement_launchd install
./.venv/bin/python -m Scripts.ops.measurement_launchd stop
```

**These are alternatives, not commands to run consecutively.** Install preserves
the original prospective cutoff. Stop pauses both the schedule and refresh hook.
The Mac must be awake, logged in and online; this is not cloud hosting.

The job checks each minute, but only calls the provider for due work, capped at
20 attempts per cycle / 500 per UTC day. These caps are for measurement only,
not the other data jobs. Logs are `Index/logs/prediction-measurement.stdout.log`
and `Index/logs/prediction-measurement.stderr.log`.

New publications now record the approved [participation-based settlement policy](participation-settlement-policy.md).
The same job settles eligible weighted-card bets using recorded positive-minute
players and voids abandoned fixtures. Player requests share the above caps;
there is no extra scheduler or bulk download command. Old publications are not
retroactively opted in, and yellow-only markets retain their separate rule gate.
Restart long-running website/bot/publishing processes after updating code; the
one-shot measurement job loads it on its next invocation. Reporting labels this
as product-rule performance, not guaranteed bookmaker-equivalent ROI.

`refresh_deferred` means the shared refresh gate is closed. If a refresh is active,
let it finish; the job retries automatically. If the refresh failed, use the
normal additive refresh command after inspecting that error. Never delete the
incomplete marker to force operation.

### Phase 2 acceptance audit and explicit historical recovery

Produce a read-only report of publication links, pending reasons/ages, reporting
contracts and the remaining live sign-off gates. This makes no API requests:

```bash
./.venv/bin/python -m Scripts.ops.phase2_acceptance_audit \
  --output "Index/prediction_experiments/phase2-acceptance-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Historical recovery is **not routine weekly maintenance**. Review a plan before
applying it. The following uses the already saved/checksummed eight-fixture
diagnostic, not fresh downloads. Choose a new plan filename for a new review:

```bash
./.venv/bin/python -m Scripts.data_platform settlement-recovery \
  --plan-out Index/prediction_experiments/historical-recovery-plan.json \
  --evidence-directory Index/prediction_experiments/phase2-settlement-provider-2026-09-24
```

The default historical cutoff is the scheduler activation boundary (467). It
does not add missing market periods or model versions. In the September 24 audit,
no old selection was eligible for an outcome; the reviewed plan only records
461 pending reasons and excludes six not-yet-due selections.

To resume that **specific reviewed plan**, after the active refresh completes:

```bash
./.venv/bin/python -m Scripts.data_platform settlement-recovery \
  --apply-plan Index/prediction_experiments/phase2-historical-recovery-2026-09-24/reviewed-plan.json
```

Application checks the plan, row identities and evidence again; changed records
require a new review. Re-running the same successfully applied plan is a no-op.
If it reports `deferred`, let the existing refresh finish; do not remove its lock
or incomplete marker. A deferred manual recovery is **not automatically queued**.

The website tracking API supports pending visibility and separate scopes:
`/api/track-record?publication_scope=initial`,
`/api/track-record?publication_scope=amendments`, and
`/api/track-record/recent?graded_only=false`. Its daily dates are publication
dates, not match dates. Restart website/bot processes after reporting-code updates.

## Match Reads and website publication

Replace the sample date below with a real ISO date, for example `2026-09-12`. Do not type the literal text `YYYY-MM-DD`.

Preview the fixture-level Match Reads without saving them:

```bash
./.venv/bin/python -m Scripts.data_platform match-reads \
  --league EPL LaLiga --date 2026-09-12
```

Generate and save the pre-match reads for later delivery:

```bash
./.venv/bin/python -m Scripts.data_platform match-reads \
  --league EPL LaLiga --date 2026-09-12 --persist
```

Around an hour before kickoff, generate saved amendments using verified starting lineups. A read is only labelled as lineup-confirmed when both teams have a full verified XI:

```bash
./.venv/bin/python -m Scripts.data_platform match-reads \
  --league EPL LaLiga --date 2026-09-12 \
  --stage confirmed_lineups --persist
```

Preview the website release decision without changing delivery records:

```bash
./.venv/bin/python -m Scripts.data_platform publish-match-reads \
  --league EPL LaLiga --date 2026-09-12 --dry-run
```

Publish eligible saved reads to the website:

```bash
./.venv/bin/python -m Scripts.data_platform publish-match-reads \
  --league EPL LaLiga --date 2026-09-12
```

Saving a Match Read and publishing it are deliberately separate. The website only displays released website deliveries; it does not calculate fresh bets when a visitor opens the page. The release guard rejects stale cards and fixtures that have already started.

### Scheduled Match Read worker

The worker is a one-shot command designed for a scheduler to invoke every ten minutes. It reads upcoming fixtures from `platform.db`, so run the normal season refresh before relying on it. By default it creates pre-match Match Reads for the next four days, refreshes those early cards every six hours, then switches to the normal rapid refresh and lineup amendment windows near kickoff. It uses an additive database migration for its run history, lease, and freshness observations:

```bash
./.venv/bin/python -m Scripts.data_platform init-db
```

Inspect the exact work due without writing records or calling API-Football:

```bash
./.venv/bin/python -m Scripts.data_platform match-read-cycle --once --dry-run
```

Run a safe shadow cycle. It can create/update immutable Match Read research records and observations, but it never creates website deliveries, tracked recommendations, or Discord posts:

```bash
./.venv/bin/python -m Scripts.data_platform match-read-cycle --once --mode shadow
```

After shadow acceptance is complete, website mode releases preliminary pre-match cards throughout the four-day outlook, refreshes them on the slower early cadence, and then updates them rapidly in the final window (including the confirmed-lineup amendment). It still does not post to Discord:

```bash
./.venv/bin/python -m Scripts.data_platform match-read-cycle --once --mode website
```

For a temporary local scheduler, or later for a cloud scheduler, invoke the same one-shot command every ten minutes. The worker obtains a database lease, so overlapping invocations safely skip rather than duplicate a published selection. The public configuration covers all ten domestic leagues plus UCL, UEL, and UECL; use `--league UEL UECL` for a scoped manual check.

The worker processes the nearest kickoffs first and releases each completed
fixture immediately; it does not wait for the entire four-day slate. Production
release checks use the current time, not the time the cycle started. Shared
provider prices retain their original check time, so this does not relax the
stale-odds guardrail.

Runtime limits (optional `.env` overrides):

- `MATCH_READ_MAX_CYCLE_SECONDS=480`: yield after approximately eight minutes
  and replan unfinished work on the next scheduled tick. The effective budget
  is also capped below the worker lease lifetime.
- `MATCH_READ_FIXTURE_TIMEOUT_SECONDS=120`: interrupt a stuck fixture call,
  record its failure, and continue with other fixtures within the cycle budget.

These limits require a POSIX main-thread worker (the normal macOS/Linux CLI).
A yielded run is recorded as `deferred`; any generation/publication failure is
`partial` and exits nonzero. A rejected publication is retried on the short
cadence instead of silently counting as a completed public refresh. `--now` is
for deterministic diagnostics; do not set it on the actual scheduled command.

Inspect per-stage performance and timeout diagnostics from scheduled runs:

```bash
rg 'Match Read performance' Index/logs/match-read-worker.stderr.log | tail -n 20
```

Measurements are also saved under `sync_runs.stats.fixture_performance`.
Nested timings overlap and must not be summed. See
[Match Read performance](match-read-performance.md) for the saved profile
parity check and remaining reliability work.

### Install the local macOS scheduler

On a development Mac, install the tracked `launchd` service once instead of
keeping a terminal open. It runs the existing one-shot worker immediately at
installation and then every ten minutes. The default is deliberately safe
`shadow` mode: it stores/reviews Match Reads but cannot release website cards.

```bash
./.venv/bin/python Scripts/ops/match_read_launchd.py install
```

Check whether the exact service is installed and loaded:

```bash
./.venv/bin/python Scripts/ops/match_read_launchd.py status
```

Its output and errors remain locally inspectable at:

```bash
tail -n 100 Index/logs/match-read-worker.stdout.log
tail -n 100 Index/logs/match-read-worker.stderr.log
```

After a clean shadow acceptance run, switch the same scheduled service to
website release mode. Reinstalling safely unloads/reloads only this service
and starts a new cycle; it does not create a second timer:

```bash
./.venv/bin/python Scripts/ops/match_read_launchd.py install --mode website
```

`website` mode automatically creates/refreshes eligible website Match Read
cards and tracking records. It does **not** post Discord messages; Discord hub
delivery remains the main bot's separate `MATCH_READ_MODE=hub` responsibility.

To preview the generated plist without installing it:

```bash
./.venv/bin/python Scripts/ops/match_read_launchd.py render --mode shadow
```

To stop automatic local cycles and remove only this LaunchAgent plist (the
diagnostic logs are retained):

```bash
./.venv/bin/python Scripts/ops/match_read_launchd.py uninstall
```

The helper never sources `.env` or writes API, Discord, or OpenAI credentials
into the plist. It sets the checkout as the worker's working directory, where
the existing application configuration loads the normal project `.env` at
runtime. This is local-Mac automation, not cloud deployment; a cloud scheduler
will invoke the same one-shot command later. It runs while this Mac is awake
and the logged-in user session is available; sleep, logout, or shutdown pauses
the local schedule until the machine is available again. While an actual cycle
is running, macOS idle sleep is inhibited with `caffeinate`; closing the lid,
explicit sleep, shutdown, and loss of internet still interrupt local hosting.
An expired worker lease prevents a resumed old cycle from publishing.

## Start the local website

Run this in a terminal and leave it running:

```bash
./.venv/bin/python Scripts/web_app/api.py
```

Open the local site at:

```text
http://localhost:8000
```

Stop it with `Ctrl+C`.

## Start the Discord bots

Start the main Discord bot in a separate terminal:

```bash
./.venv/bin/python Scripts/discord_bot/bot.py
```

The main bot's Match Read behavior is controlled in `.env` by `MATCH_READ_MODE`:

- `off` — default; no Match Read hub publishing.
- `shadow` — generate/compare internally without public Match Read delivery.
- `hub` — publish through the league Match Read hubs; use only after the planned acceptance checks.

The bot's lineup-refresh task is active only when `MATCH_READ_MODE=hub`. It looks for fixtures 10–100 minutes before kickoff, persists a verified lineup amendment when possible, then edits the existing hub post and labels it as updated.

The optional separate live bot uses its own token and is not required to run the main bot:

```bash
./.venv/bin/python Scripts/live_discord_bot/bot.py
```

Stop either bot with `Ctrl+C`.

## Health checks and tests

Quick health check:

```bash
./.venv/bin/python -m Scripts.data_platform health --quick
```

Full health and data-quality check:

```bash
./.venv/bin/python -m Scripts.data_platform health
```

Show platform status:

```bash
./.venv/bin/python -m Scripts.data_platform status
```

Run the automated test suite before a meaningful code change or deployment:

```bash
./.venv/bin/python -m pytest
```

For help with a command or its available options:

```bash
./.venv/bin/python -m Scripts.data_platform --help
./.venv/bin/python -m Scripts.data_platform match-reads --help
```

## Normal matchweek operating order

1. Run the full `refresh-season` command after results are available.
2. Run `health --quick`; use full `health` if anything looks wrong.
3. Restart the website and Discord bot so they read the refreshed local data.
4. Generate and persist pre-match Match Reads for the target date.
5. Review the reads, run `publish-match-reads --dry-run`, then perform the actual website release only for cards that should be visible.
6. With hub mode enabled, let the main bot check for verified lineups near kickoff. It will amend the existing Discord hub card rather than create a new market-channel post.

Do not manually delete, reset, or recreate `Index/platform.db`, Chroma indexes, or `Output/` as part of normal maintenance. `refresh-season` is the supported additive refresh path.
