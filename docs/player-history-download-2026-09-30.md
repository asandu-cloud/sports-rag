# Player, lineup and event collection: 2021–2026

Updated 30 September 2026 with the owner's approval to add lineups, events and
the current season. The v2 collector preserves v1 progress when an existing
collection is explicitly extended. No paid provider requests have been started
by the coding agent; all implementation validation is offline.

## Scope and command

Scope is **five completed provider seasons 2021–2025 plus current season 2026**,
meaning 2021/22 through 2026/27. Only already completed current-season matches
are included. A read-only plan found **23,263 completed fixtures** stored across
all 13 public competitions:

| Provider season | Stored completed fixtures |
|---|---:|
| 2021 | 4,503 |
| 2022 | 4,405 |
| 2023 | 4,392 |
| 2024 | 4,496 |
| 2025 | 4,460 |
| 2026 (current, partial) | 1,007 |
| **Total** | **23,263** |

The inventory is frozen on first execution. It does not discover additional
missing fixtures or certify that every season is complete. All recorded completed
rounds are collected, including European qualifiers/knockouts and domestic
playoffs. Storage does not make these rounds eligible for the current card
research, whose ordinary-league/group/league-phase scope remains in force.

Run from a terminal on the Mac:

```bash
cd /Users/sanduandrei/Desktop/Betting_RAG
PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -B -m Scripts.ops.player_history_download \
  --execute --seasons 2021 2022 2023 2024 2025 2026 \
  --endpoints players lineups events \
  --daily-budget 75000 --reserve 0 --rpm 420
```

The script reads `API_FOOTBALL_KEY` or `API-FOOTBALL-KEY` from the environment or
project `.env`; the key is not printed. `caffeinate -i` prevents idle sleep while
the command runs; it does not make the job survive terminal closure or shutdown.
Rerun the same command after interruption. It resumes **each fixture and endpoint**:
if players and lineups succeeded before events failed, only events are requested
again. An optional `--max-fixtures 20` stops after a pilot (up to three base
requests per fixture); remove that option to continue the same collection.

After more current-season matches finish and enter the canonical fixture store,
add `--extend-scope` to the same command to append them. Ordinary resume preserves
the frozen inventory. Extension checks old metadata for contradictions and
retains earlier manifests and identity snapshots.

If the previous five-season player-only command was already run, reuse that
directory to retain its progress:

```bash
PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -B -m Scripts.ops.player_history_download \
  --execute --seasons 2021 2022 2023 2024 2025 2026 \
  --endpoints players lineups events \
  --directory Index/history_staging/player-stats-2021-2025 --extend-scope \
  --daily-budget 75000 --reserve 0 --rpm 420
```

This upgrades staging bookkeeping only, retaining prior player rows, source
archives and request counters. No existing download directory was found during
the current preflight; the main command above is appropriate here.

To see the scope without creating files or making requests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.player_history_download --dry-run
```

After collection, verify saved sources locally (no API key or requests):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.player_history_download --verify-saved
```

If a parser needs correction or derived staging rows need rebuilding, replay
the original responses locally:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.player_history_download --reprocess-saved
```

Both commands use the same default 2021–2026 directory and three endpoints.
For a collection using custom seasons, endpoints or directory, supply its same
arguments. Replay retains raw archives and request counts, rebuilds only derived
staging rows, and records previous source/dependency fingerprints if they changed.
If replay is interrupted, `REPLAY_REQUIRED.json` blocks ordinary downloads until
local replay finishes, preventing mixed parser versions from appearing complete.

## Quota and failure behavior

- Three requests per fixture: `/fixtures/players`, `/fixtures/lineups` and
  `/fixtures/events`, each with the exact saved fixture ID. **69,789 base
  requests**, plus bounded retries, less responses already collected in this
  staging job. No team-statistics, fixture-discovery, odds, profile or OpenAI
  calls. All returned fields are retained; fields do not cost extra requests.
- Maximum 75,000 HTTP attempts per UTC day **for this collection**, including
  failed/retried attempts. The counter persists across process restarts.
- Default and maximum configured rate: **420 requests/minute**, updated after
  the owner confirmed Ultra access. Request starts are evenly spaced by at least
  1/7 second, matching Ultra's seven-per-second limit. Lower provider limits are
  retained across responses that omit headers; per-minute exhaustion and 429
  responses delay the next call. Response latency counts toward the interval,
  rather than having an entire pacing delay added afterward. Requests remain
  sequential, so network latency and saving responses can reduce actual throughput.
  See the provider's [rate-limit explanation](https://www.api-football.com/news/post/how-ratelimit-works).
- The owner approved using the full daily allowance: **zero reserved requests**.
  Stop once a response reports zero daily requests remaining. Other
  jobs share the provider allowance: this is not an account-wide atomic quota
  reservation. Before the first response in an invocation, the provider's current
  remaining quota is unknown. Lower the job budget if necessary.
- A quota pause exits; it does not automatically wait until tomorrow. Rerun the
  same command when quota is available.
- The shared HTTP client has connection/read timeouts and at most three attempts.
  Authentication, provider-envelope errors or exhausted retries stop the run.
  Completed responses remain saved.
- Empty player/lineup responses are `unavailable`; empty event responses are
  `empty_unverified`, because absence of a timeline cannot certify zero cards or
  substitutions. Malformed/mismatched responses and incomplete starting XIs or
  player rosters are `invalid`. Events with unknown team/time/primary-participant
  identities or exact duplicate entries are retained as `partial`. None of these
  states certifies complete evidence. Ten consecutive invalid responses from the
  same endpoint stop for source inspection; empty coverage does not stop the run.
- Previously empty responses can be retried explicitly with `--retry-unavailable`.
  Invalid/partial responses remain held for review; ordinary reruns do not erase
  or repeatedly download them.
- A process lock prevents concurrent copies against the same collection.
- Resume verifies saved archive hashes, source references and frozen fixture/
  team/competition/season identities before issuing more requests. A missing
  collection database, lost checkpoint or corrupt source archive stops the job;
  it does not silently start a new download.
- A durable `pending-responses/` file is written after a successful HTTP response
  and before database archival. Interrupted archive writes resume from that file
  without another API call, retaining the original capture timestamp. Raw archive
  writes use atomic replacement and filesystem synchronization.

These recovery paths protect recorded responses. A process/machine failure
before the first durable write can still require retrying an individual request.

At 420 requests/minute, the theoretical pacing floor is about **2 hours 46
minutes** for the full inventory. This is not a measured completion estimate:
requests and saving responses are sequential, so network latency, validation,
database writes, retries and provider limits add time. The 69,789 base requests
fit within a fresh 75,000-request daily allowance, leaving 5,211 for retries or
other account usage. Earlier/parallel account usage may still cause a quota
pause; a second quota day is no longer required by the job budget itself.
Exit code 2 means paused/pending; code 3 means
collection finished with evidence gaps requiring review. Code 0 does not certify
card/model qualification, only absence of recorded collection gaps.

## Files and exact relationships

Collection directory:

`Index/history_staging/player-stats-2021-2026/`

- `platform.db`: the existing platform schema plus staging-only lineup/event and
  endpoint-checkpoint tables, isolated from the publishing DB.
- `raw_archive/`: compressed, hash-verified complete provider envelopes, including
  response metadata and unfiltered player blocks.
- `fixture-identities.json`: frozen fixture, competition, provider season, stage,
  team identities and kickoff metadata copied read-only from canonical storage.
- `manifest.json`: scope, fixture hash, collector/dependency fingerprints.
- `summary.json`: checkpoint states by endpoint, player/lineup/event row counts,
  shared daily HTTP attempts and foreign-key checks. No outcome distributions or
  prediction metrics.

The existing tables connect each row explicitly:

`fixture_player_stats.fixture_id -> fixtures.id`

`fixture_player_stats.player_id -> players.id`

`fixture_player_stats.team_id -> teams.id`

The fixture and team catalogue IDs are preserved from the canonical inventory;
every catalogue also retains its API-Football ID. Player catalogue IDs are local
to staging and must be mapped by **provider player ID** on later import, never by
name. Each player row records its source archive ID and provider fixture ID in
`stats_json`. All provider statistic sections are retained under
`stats_json.provider_statistics`, including fields without dedicated SQL columns.
Raw player identity/photo fields remain in the original archived envelope.

`fixture_lineup_teams` preserves formation and complete coach/team blocks.
`fixture_lineup_entries` links starter/bench entries to fixture, team and player
catalogue IDs, retaining original shirt number, position and grid fields.
`fixture_events` links each event to its fixture and archived response, with
team IDs mapped to the same catalogue. Player/assist provider IDs, elapsed/extra
time, event type/detail and full original JSON are retained. Missing identities
stay null; ordering and duplicates remain available for review. Names are never
used to guess missing IDs. Substitution actor roles and second-yellow semantics
must be reconciled later, not inferred during download. A listed substitute is
not automatically an appearance or DNP label.

`fixture_endpoint_collection` holds progress per fixture and endpoint.
`player_collection_scope` commits the manifest and frozen metadata in SQLite,
so interruption while writing JSON mirrors does not lose the scope. An
interrupted initial setup resumes from `preparing-scope.json`.

This gives each player performance an unambiguous match/team identity, allowing
team and player evidence to be joined after import. Fixtures are seeded with
identity metadata only: a null score in staging is not a newly observed missing
score, and fixture metadata must not overwrite richer canonical fixture records.

Provider nulls remain null. Zero- and one-minute players are retained as source
evidence. The approved minimum **two recorded minutes** is a later derived
card/profile eligibility rule, not a destructive filter on collection. Ambiguous
pass-accuracy values are retained raw; percentages must not become zero accurate
passes through integer coercion.

## What happens after download

**The download does not replace `Index/platform.db`, publish new predictions or
rebuild team/player/referee profiles.** The existing records and source archives
are preserved while fresh evidence is collected. `ready` means structurally
validated and linked, not statistically complete, card-qualified or imported.

Next: review completion/coverage and reconcile against existing rows by provider
fixture/team/player IDs. Preserve old values, review contradictions, and prevent
new nulls/partial responses from replacing better evidence. A reviewed import
must map staging player and archive IDs to canonical IDs and retain source
collection timestamps. The canonical archive reader must recognize that this
collector saves the complete envelope, whereas some older archives stored only
its `response` list. That import/adaptation is a separate next step, not an
implicit effect of this download command.

Historical responses downloaded today are final-statistic reconstructions, not
proof of what was available at the original forecast time. Mechanical acquisition
and source validation across these six seasons do not open protected final-test
outcomes for model selection. Existing dated research readers and evaluation
boundaries remain in force. Production card-policy/profile changes stay with the
agreed future engine release; weight optimization is not part of collection.

## Player data available

API-Football's fixture-player endpoint supplies player identity plus match
participation (minutes, shirt number, position, captain/substitute flags, rating),
shots, goals/assists, passing, tackles/blocks/interceptions, duels, dribbles,
fouls, yellow/red cards, goalkeeper saves/goals conceded and penalty statistics.
Fields and coverage can be absent in particular league-seasons or fixtures.
Full responses preserve additional fields without requiring another download.

See the provider's [fixture-player explanation](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide)
and [v3 reference](https://www.api-football.com/documentation-v3#tag/Fixtures/operation/get-fixtures-players).
The command also collects full lineup and event responses, including starters,
bench players, formations, coaches, cards, goals and substitutions. It does not
collect historical injuries, transfers or bookmaker player-prop prices.

## Local validation

The initial v2 implementation passed **89 tests** in 6.62 seconds across the collector, existing synchronization
upserts and refresh-safety suites. The v2 extension covers three endpoints,
current-season inclusion, shared quotas, partial event identities, retained
event order, lineup joins, endpoint-specific resume, v1 migration and additive
scope extension. CLI help and the read-only full-scope plan also succeeded.
The prior player-only implementation had passed 69 tests.

Offline tests cover exact joins, zero/null preservation, unfiltered minutes,
full raw/statistic retention, malformed/mismatched responses, request retries and
budgets, provider rate headers, interrupted normalization and restart, concurrent
process rejection, staging path isolation and unchanged source DB bytes. Live
provider coverage and successful historical acquisition remain unverified until
the owner runs the command.

### Second review, before the owner's download

**105 tests passed** in 9.16 seconds across the collector, synchronization upserts
and refresh-safety suites after the review fixes. Tests used synthetic provider
responses with real outbound HTTP blocked; the real provider download is pending.

The second review found and fixed missing archive verification on resume and
the lack of an offline normalization-rebuild command. It also added recovery of
responses interrupted before archival, atomic archive writes, rejection of
staging paths escaping through parent-directory components, and detection of
lost collection databases/checkpoints before they could trigger another fetch.
Additional provider statistic fields are retained without rejecting an otherwise
usable response simply because a future field is new.

A full metadata-only initialization copied all **23,263 fixture identities**,
**13 competitions**, **78 competition/season combinations** and **69,789 endpoint
checkpoints** into a temporary database. All identity checks passed and there
were **zero foreign-key errors**. This test downloaded no provider responses and
read no player outcome values. Synthetic tests separately exercise archive
serialization, exact joins, archive-write failure, quota pauses, local replay,
corrupted/missing archives, lost progress and interrupted initialization/replay.

The current workflow and the required next steps are also appended at the bottom
of [the ML improvement plan](ml-prediction-improvement-plan.md).

### Ultra-plan pacing update

After the owner confirmed Ultra access, the 420-request/minute change passed
**111 tests** across the same collector, synchronization and refresh-safety
suites in 9.61 seconds. Simulated fast and slow responses verify evenly spaced
starts without a redundant post-response delay; lower provider limits persist
when subsequent headers are absent. Existing retry, quota and archive-recovery
checks also passed. No paid provider requests were made, and real throughput
has not been measured. The daily job budget at that validation was 40,000
attempts; the owner subsequently approved 75,000 attempts with zero reserve.
The quota update passed all **72 collector tests**, including stopping at zero
provider allowance, retry accounting and restart behavior. A metadata-only dry
run confirmed 75,000 attempts/day, zero reserve and 420 requests/minute for
23,263 fixtures. No provider requests or collection writes were started.

### Completed owner-run collection — 30 September 2026

The owner-run collection finished at 15:34:09 UTC. All **69,789 fixture/endpoint
checkpoints** have terminal outcomes across 23,263 fixtures; 69,790 HTTP attempts
were recorded. No pending/downloaded checkpoints remain and `stopped` is null.
Subsequent read-only verification checked all 69,789 original archive checksums,
source references and frozen fixture identities in 70.76 seconds. SQLite
integrity and foreign-key checks passed. This verification made no API requests
and wrote neither the staging nor canonical database.

| Endpoint | Ready | Invalid | Other outcomes |
|---|---:|---:|---|
| Players | 21,381 | 75 | 1,807 unavailable |
| Lineups | 23,231 | 30 | 2 unavailable |
| Events | 22,146 | 777 | 335 partial; 5 empty/unverified |

There are 886,553 linked player-statistic rows, 966,138 lineup entries and 358,841
normalized events. All three endpoints are structurally ready for 20,331 fixtures;
this is source-validation status, **not card-target or model eligibility**.

Every invalid event response contains a negative integer `time.elapsed` rejected
by the current validator. Early-season examples include card events with elapsed
time -5. Their meaning remains unresolved; do not replace these with zero or
silently discard the incidents. All original responses are retained for offline
reader repair/reconciliation. Invalid player responses comprise 72 missing or
duplicate-identity cases and three incomplete rosters; invalid lineups comprise
26 identity cases and four incomplete-XI/bench cases.

Empty player responses are concentrated in UECL (1,183), UCL (310), UEL (307)
and BelgianProLeague (7). They occur across all requested seasons. Empty responses
are observed acquisition gaps, not proof that another request cannot supply data.
No automatic bulk retry was started. Canonical import, card reconstruction and
the fixed backtest remain pending; no production profiles or models changed.

### 2 October 2026 — Reconciliation, canonical import and card reconstruction complete

This update supersedes the pending import/reconstruction status above. The owner
authorized the first four post-download steps, stopping before profile builds
and backtesting. No new provider requests were made.

**Source review and repair.** All 69,789 responses were checksum-verified and
copied into a separate sealed preparation. The original collection database's
SHA-256 remained unchanged. All 777 rejected event responses were retained as
`partial` timelines: their 868 negative elapsed values remain in original JSON,
while normalized times become NULL. A negative time has not been interpreted as
zero, a pre-match booking or a post-match booking. The other 105 invalid
player/lineup responses remain quarantined; missing IDs and conflicting duplicate
identities were not guessed. The 1,809 unavailable responses, five empty timelines
and other partial-source issues remain visible. Neither retries nor null-to-zero
repairs were used to conceal coverage gaps.

**Controlled import.** The same importer first ran on a disposable canonical
copy, then on `Index/platform.db` under the normal exclusive refresh writer gate
with a fresh SQLite backup. The pre-existing incomplete refresh finished and its
gate cleared normally before import; no locks or markers were removed to bypass it.

| Import result | Count |
|---|---:|
| Players added | 18,278 |
| New canonical fixture-player statistic rows | 636,553 |
| Incoming player observations retained with provenance | 886,553 |
| Original source archives registered | 69,789 |
| Existing differing player rows preserved | 7,515 |
| Pre-existing rows overwritten | 0 |
| Original tables checked for preservation | 38 |

The database now has 28,692 players and 888,027 fixture-player statistic rows.
It also contains 966,138 imported lineup entries and 373,641 events. Every old
row across the 38 original tables was checked against the pre-import backup;
fixture scores, publications, settlements and existing player statistics were
preserved. Incoming differences (including null/non-null differences) are stored
as separate observations, not silently selected as replacements. Additional
columns are still in their complete original payloads.

`player_history_batches`, `player_history_responses`,
`player_history_observations`, `player_history_lineups`,
`player_history_lineup_teams` and `player_history_events` retain batch-specific
evidence in the canonical database. Player, fixture, team and archive IDs were
mapped to canonical IDs, including `stats_json.source_archive_id`. Archives were
copied under `Index/raw_archive/player-history-32e372498cb025a2/`, so canonical
references do not depend on a temporary directory. Original and prepared
collections remain intact. SQL integrity, foreign-key and observation identity
checks passed; all 868 original negative event times remain traceable.

**Card reconstruction.** Research uses the frozen two-minute v2 participation
rule, the weighted 1/2/3 convention and the existing ordinary-league/group/league
phase scope. Complete player responses are checked against starting lineups,
participation and weighted card events. Null card counts remain unknown; known
older card/minutes conflicts, missing actors/participation, uncertain eligible
booking times and conflicting card events leave targets pending. Exact lineup
coach identities can establish staff exclusion; absence from a roster alone
cannot. Negative event times for eligible carded players remain unresolved.

Only the permitted pre-2024 development cohort was reconstructed:

| Cohort | Fixtures |
|---|---:|
| Development fixtures examined | 11,323 |
| Qualified card targets | 9,325 |
| Deferred special rounds | 1,367 |
| Other unresolved development fixtures | 631 |
| Later fixtures excluded from target decoding | 11,940 |

Qualified targets include 7,773 matches without a recorded red among eligible
players, 1,552 with a recorded red, and 274 with zero weighted cards. These source
availability checks address the earlier red-only-label problem; they do not
establish unbiased sampling, prediction quality or market qualification. All
13 competitions have qualified development targets, with coverage and overlapping
exclusion reasons recorded by league/season. Historical availability remains
`assumed_final`, not an original forecast-time snapshot. Production settlement
continues to use v1; this v2 reconstruction is research only.

**Validation:** 193 collector/reconciliation/card-policy/refresh-safety tests and
119 production-settlement/Asian-arithmetic tests passed (312 total). Import tests
cover rollback, ID remapping, preservation, idempotency, archive tampering and
fixture identity conflicts. Card tests cover weighting, the two-minute boundary,
nulls, event conflicts, missing actors/timing and reserved-period refusal. The
real-data rehearsal and production import both verified all 38 original tables.

Artifacts and code:

- Preparation: `Index/prediction_experiments/player-history-prepared-2026-10-02/`
  (`PREPARED.json` seals source hashes, rules, database and archive index).
- Rehearsal: `Index/prediction_experiments/player-history-import-rehearsal-2026-10-02/`.
- Production import and rollback backup:
  `Index/prediction_experiments/player-history-import-2026-10-02/`
  (`platform-before.db`, `before.json`, `result.json`, `post-import-checks.json`).
- Card artifact: `Research/phase4-player-card-reconstruction-2026-10-02/`.
  Source-reference archive IDs here belong to the sealed preparation; payload
  hashes and the batch ID link them to canonical response/archive mappings.
  Its frozen canonical comparison input is the rehearsal's `platform-before.db`.
- CLI: `Scripts/ops/player_history_reconcile.py`; evidence rules:
  `Scripts/data_platform/features/player_history_reconciliation.py`.

No team/referee/player profiles were rebuilt by this task. No backtest, weight
optimization, model promotion, embedding, publication or production card-policy
activation was run. The next agreed step is historical team/referee inputs and
the fixed card backtest, followed by owner review before weight optimization.
