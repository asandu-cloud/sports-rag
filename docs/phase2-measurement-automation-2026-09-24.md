# Phase 2 — prospective measurement automation

Date: 24 September 2026.

Scope: the first three remaining build points: bounded provider/rule validation,
independent settlement scheduling, and exact pre-kickoff closing-price capture.
Historical outcome recovery and final live acceptance remain separate work.
Prediction formulas, model training, selection thresholds, website publication
mode and the existing Match Read schedule were not changed.

## Status

| Point | Implemented and verified | Still open |
| --- | --- | --- |
| Rule/provider validation | 8 completed fixtures, 16 provider requests; read-only historical audit; 120 explicit contract probes on real scores/statistics | Applicable bookmaker/country card and cancellation policies; real unusual-status acceptance |
| Independent settlement | Local LaunchAgent installed/loaded; prospective cutoff; refresh hook; persistent budgets, retries, lease and health | Live future-publication settlement after the active refresh and on the next matchday |
| Closing-price capture | Exact matching, raw evidence, timestamps, provisional/final states and price-based CLV; scheduled prospectively | Live freshness/coverage acceptance; the provider may not have a sufficiently fresh quote |

This is **not full Phase 2 sign-off**.

## Historical diagnostic: what it actually proves

The SQLite connection enforced `mode=ro` and `PRAGMA query_only=ON`. No outcomes,
publication records or price fields could be written by the diagnostic. Before/
after fingerprints of all 1,799 candidate prediction rows matched.

- Sample: fixtures `1575503`, `1570401`, `1552179`, `1552773`, `1552767`,
  `1550127`, `1550136`, `1575168`, selected round-robin across market groups,
  most recent first. This is a bounded diagnostic, not a representative ROI study.
- Eight fixtures and their team-statistics responses: **16 attempts**, no player
  downloads, no hidden HTTP retries, no historical odds requests.
- **35 actual published selections** in these fixtures were evaluated read-only.
  All lack an explicit saved regulation-time market period. All remain pending;
  **zero historical outcomes were written or approved**. A final score alone
  cannot prove the original bet's period.
- **120 synthetic contract probes** against the eight real fixture results pass.
  These check goals, corners, SoT, BTTS, moneyline, exact score, Asian splits and
  withholding unapproved cards. They are **not 120 real recommended bets**, and
  their success says nothing about the prediction model's hit rate or ROI.
- Corrections, cancellation, extra time, missing values and invalid identities
  remain covered synthetically; this sample does not provide real examples of
  every exceptional status.

Private evidence:

- `Index/prediction_experiments/phase2-settlement-provider-2026-09-24/`:
  original response bodies, per-request ledger with SHA-256 hashes, first report.
- `Index/prediction_experiments/phase2-settlement-provider-2026-09-24-replay/`:
  final report from checksum-verified saved responses; **zero additional calls**.

The first harness run omitted the `pick` field from its eight synthetic correct-
score selections. Those eight probes correctly stayed ungraded. The harness
was corrected, regression-tested and replayed offline. Both reports are retained;
the grader was not loosened to make these probes pass.

### Bookmaker policy boundary

The local published cohort contained no card recommendations when inspected.
That does not justify approving a blanket future card policy.

The [Bet365 Canadian card rules](https://help.bet365.com/s/en-ca/card-markets-queries)
use yellow=1 and red=2, ignore the second yellow, cap a player's contribution at
three, and exclude specified non-playing/post-match cards. Abandonment rules also
depend on the market already being determined. These are **Canadian rules**, not
evidence that every Bet365 country site or every tracked bookmaker is identical.
Aggregate team yellow-plus-red counts cannot establish these eligibility details.

No production card/cancellation policy is activated. The applicable bookmaker
and country/site have been requested from the user. Unknown rules remain a
visible pending state, not a loss, zero or assumed refund. The existing injectable
rule contract still requires exact bookmaker, market, period, version, evidence
and confirmed provider-statistics compatibility before grading cards.

## Independent local scheduler

Installed job: `com.bettingrag.prediction-measurement`.

- LaunchAgent runs `measurement-cycle --once` every **60 seconds**, independently
  of Discord and the browser. This is a due-work check, not a full database/model
  refresh every minute. The Match Read job remains on its existing cadence.
- Activation was recorded at `2026-09-24T08:33:04Z`, after recommendation ID
  **467**. Only subsequently published canonical recommendations are eligible.
  The existing 467 publications and unpublished legacy/shadow rows are excluded.
  Reinstalling/re-enabling preserves that cutoff; it does not rebaseline history.
- The fixture grouping shares each provider response across eligible selections.
  Settlement starts at kickoff +3h; success is final, incomplete statistics retry
  with increasing delays from five minutes up to six hours. Static unverified
  identities/rules are held without repeated provider requests.
- Successful `refresh` / `refresh-season` completion also invokes prospective
  settlement. Scheduled retries do not depend on a new refresh or a running bot.
- Caps: **20 transport attempts per cycle, 500 per UTC day**, across this worker,
  its refresh hook and canonical bot closing capture. Failed attempts count;
  restarting the process does not reset the daily ledger. These are additional
  to other project jobs, not a global account quota. No player calls or LLM calls.
- Up to ten fixtures per work kind per cycle; closing work first, settlement
  retries oldest-checked first. Each cycle has a 50-second write/request deadline,
  8-second HTTP timeout and a 120-second ownership lease. Checks fence writes
  after expiry. Redirects, automatic HTTP retries and multi-page responses are
  not silently followed.
- Reuses existing `sync_runs`, `sync_watermarks`, `worker_leases`, prediction
  extras and `odds_snapshots`; **no schema migration** is required.

During installation, an existing `refresh-season --include-europe` process was
active and the refresh-incomplete marker existed. The first cycles correctly
reported `refresh_deferred` with **zero provider calls**, rather than bypassing
the shared data-access lock. The worker retries automatically. If that refresh
fails, its marker must remain until a successful refresh; never delete it merely
to turn the health signal green. No full refresh was started or stopped here.

`measurement-status` and `health --quick` expose stale/partial/held work. Run
history includes deferral reasons; pending tasks include first/last checks and
next retry time. There are dedicated stdout/stderr logs. This is local
observability, **not a new external alerting service**.

## Closing-price definition and limitations

1. Work is prospective, for the exact published fixture/selection identity.
   Verify the fixture ID, competition, schedule and not-started state first.
2. Attempt capture near T-20, T-5 and T-1 minutes. Exact bookmaker, provider market
   key, period, side and line must match. No different-bookmaker substitution,
   nearest-line fallback, or yellow/total-card interchange is allowed.
3. Record both the provider odds row's `update` and the local **response receipt**
   time. The former is labelled `api_football_odds_row.update`; it is not claimed
   to be a bookmaker-native execution/quote timestamp.
4. Both must precede verified kickoff. The capture window is 20 minutes and
   maximum provider-update age at kickoff is **15 minutes**. Missing, stale,
   future-dated or post-kickoff responses are unavailable, not closing prices.
5. Store eligible quotes provisionally with raw response evidence and snapshot
   linkage. Preserve newer evidence against older responses. Once the provider
   verifies that the match has started, finalize the latest qualifying capture.
   A changed kickoff invalidates an old provisional quote. Never fetch current
   odds after kickoff to fill in a missing close.
6. Price-based CLV is `published_decimal_odds / closing_decimal_odds - 1`.
   Positive means the published price was better than the captured close.
   This is **not** a de-vigged fair-probability measure or proof of profitability.

API-Football documents pre-match odds updates every three hours in its
[official guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).
Therefore a one-minute local check **cannot guarantee fresh closing odds**.
The strict 15-minute policy may produce low coverage. Measure that on the next
matchdays; do not loosen it silently or present a three-hour-old quote as a true
close. The separate in-play feed is not used as a pre-match replacement.

When platform tracking is enabled, the bot's closing-capture entry point now
uses this canonical worker. A canonical failure cannot fall through to a legacy
database. Explicit custom legacy-database callers retain their previous path;
they are not the official published measurement cohort.

## Verification and operation

The focused regression suite covers existing publication/settlement/reporting,
Match Reads, scheduler/refresh integration and the new safety boundaries. It
uses isolated SQLite and mocked networking: **218 tests passed**. A final check
after scheduler installation confirmed the historical prediction fingerprint
was still unchanged. Recurring launchd runs were observed in `sync_runs`, deferring
with zero requests while the existing refresh retained the shared gate. The
offline plan reported zero prospective candidates. See the terminal guide for install,
status, offline planning, one-shot operation and pausing commands.

Live acceptance still needs future recommendations crossing capture, kickoff,
result ingestion, delayed-stat retries and outcome reporting. In particular,
do not claim that a verified empty/deferred scheduled run proves live CLV works.

Remaining Phase 2 work:

1. Confirm bookmaker/site policies and evidence compatibility; keep unsupported
   markets/statuses pending until then.
2. Review safe historical recovery separately. Do not invent old periods,
   versions or closing quotes; no bulk outcome writes are authorized by this pass.
3. Complete prospective matchday acceptance and compare website/Discord outcome
   and unit-return reports. Measure closing coverage and missed-window reasons.

No ML training, calibration or numerical bet-selection change occurred here.
