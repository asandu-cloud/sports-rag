# Local prospective odds archive — 6 October 2026

The owner requested retention of future API-Football odds for later model and
selection-policy research. This authorizes quote collection, not candidate
promotion, policy deployment, outcome inspection or retrospective retuning of
reserved periods. Existing sealed Phase 4/5 archives remain unchanged.

## Scope and separation

- New standalone collector: `Scripts/ops/odds_archive.py`.
- New SQLite database: `Index/odds_archive/odds.sqlite3`, excluded from Git by
  the existing `Index/` rule. No migration or write to `Index/platform.db`.
- Separate job: `com.bettingrag.odds-archive`, every three hours while this Mac
  is awake and the user's launchd session is available. It runs at load/login;
  downtime produces gaps, not fabricated observations.
- All 13 existing competitions; UTC today through 14 days ahead, inclusive.
  Provider season follows the existing European-season date rule.
- All returned pre-match bookmakers, markets, selections and lines are retained,
  irrespective of whether Spix recommends them. Unknown market labels remain
  unchanged. No market IDs are assumed to establish settlement equivalence.
- Only `/odds` requests. No fixture results, live scores, player data, predictions,
  model training, settlement, publication, or protected outcome queries.

## Evidence retained

Each request records its query, start time, receipt time, HTTP status, limited
nonsecret response headers, full response entity bytes (gzip-compressed), and
SHA-256. Request headers and API credentials are never written. Repeated identical
bodies share storage, but every observation retains its own request receipt.

The compact queryable index records fixture/league/season, scheduled kickoff,
provider update time, source age at receipt, quote counts and flagged counts.
Full bookmaker and market IDs/names, unaltered selection/line labels, original
odds values and exact decimal prices remain in the compressed response. The
`quote_rows(db, request_id)` reader reconstructs them on demand using the
original receipt time and verifies the checksum. Unknown or invalid prices
remain null in the parsed representation, with their original values preserved.
Missing provider update time never becomes collection time.
Post-kickoff observations and future/missing source times receive explicit flags.
They are not automatically eligible for any betting evaluation.

`runs` and `scopes` record all attempted league/date combinations, empty successful
responses, incomplete pagination, failures and budget stops. A complete scope
means its declared pages were received, not proof of complete bookmaker coverage
or an atomic snapshot across pages. Raw bodies and indexed quotes are append-only;
request receipts are sealed when finished. Interrupted requests are marked on the
next run. SHA-256 protects integrity; it is not independent provider attestation.

Default limits: 60 requests/minute, 600/cycle, 3,600/UTC day for this collector,
and stop when reported remaining provider daily allowance reaches 1,000. The
first response establishes the current provider allowance; other jobs share the
account, so this reserve is best effort. HTTP/auth/rate-limit/provider errors stop
the cycle with preserved evidence. Later scheduled cycles retry fresh requests.
An exclusive process lock prevents overlapping collectors.
Collection stops before another request if the archive filesystem has less than
2 GiB free; no automatic deletion is used to free space.
The 15-date window needs at least 195 requests per cycle (1,560/day with eight
complete cycles), plus any additional pages, within the configured caps.

The first live cycle used an expanded quote index: 314,843 quote rows and 96
fixtures occupied about 92 MiB, while its compressed raw responses used about
1.4 MiB. With only about 16 GiB free on the host, subsequent cycles use the
compact index. The initial `quotes` table and all initial receipts are preserved
verbatim. Metadata status includes both storage layouts without double counting.

## What this enables and what remains separate

The archive can later support bookmaker-probability comparisons, quote movement,
selection/ROI evaluation, and experiments that use odds as model features.
None of those benefits is established merely by storing data. Forecast/input
snapshots, predeclared development/test assignments, outcomes, exact settlement
rules, price availability and supported probability evidence still need their
own reviewed contracts. Any later features must use quotes actually observed
before the forecast cutoff, not the latest price available when training.

The existing July 2026 onward reserve remains protected. Collection does not
relabel any period as development or permit its outcomes to be used for tuning.
Archive metadata explicitly records `training_authorized=false` and an
unassigned evaluation role. Phase 5 qualification remains incomplete.

API-Football documents pre-match odds updates every three hours and only seven
days of retained odds history. Fresh HTTP retrieval does not refresh source
prices. This cadence is not a precise closing-price or execution-validation
service; CLV and fresh executable-quote requirements remain separate.
[Official API-Football guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).

## Operation

```bash
# Read-only plan and local metadata status
.venv/bin/python -B -m Scripts.ops.odds_archive
.venv/bin/python -B -m Scripts.ops.odds_archive --status

# One bounded collection, or independent recurring collection
.venv/bin/python -B -m Scripts.ops.odds_archive --execute
.venv/bin/python -B -m Scripts.ops.odds_archive_launchd install
.venv/bin/python -B -m Scripts.ops.odds_archive_launchd status

# Stop only this job and remove its login schedule; preserve all evidence
.venv/bin/python -B -m Scripts.ops.odds_archive_launchd stop

# Local synthetic tests; no network or production DB
.venv/bin/python -B -m pytest Scripts/tests/test_odds_archive.py Scripts/tests/test_upstream_refresh_launchd.py Scripts/tests/test_match_read_launchd.py -q -p no:cacheprovider
```

Back up the archive with SQLite's backup API, or while the job is stopped; do
not copy an actively written database without a consistent snapshot. There is
no automatic pruning, cloud upload or backup configured by this change.

## Verified operational status

On 6 October 2026 the independent launchd job was installed and loaded. Its
initial scheduled cycle exited successfully. A subsequent complete cycle of
the final 14-day configuration finished at 08:07:15 UTC: all 195 league/date
scopes completed, retaining 405,030 quote observations across 138 fixtures.
The archive has 719,873 observations across the two nonempty cycles; repeated
observations are not independent matches, bets or distinct price changes.

All 40 focused collector/scheduler tests passed. The compact reader reproduced
all 314,843 original expanded rows exactly. Every successful receipt from the
final compact cycle was checksum-verified and reconstructed; quote and flagged
counts matched the index. SQLite's integrity check passed. The database was
about 93 MiB after both cycles, including the preserved initial expanded index.

The first sandboxed connection failure is retained as a failed request. The
subsequent provider calls succeeded using the existing configured credential.
Some observed source prices were already over four hours old; this archive
does not certify fresh executable odds. All prior Phase 5 bundle seals passed
verification, and no existing tracked `Scripts/` file was changed.

Detailed local evidence: `Index/odds_archive/verification-2026-10-06.json`.
