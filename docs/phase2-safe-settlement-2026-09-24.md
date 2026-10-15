# Phase 2 — exact-fixture settlement and safe grading

Date: 24 September 2026.

This report describes the earlier settlement-engine batch. The later
[measurement automation report](phase2-measurement-automation-2026-09-24.md)
records subsequent scheduling, closing capture and provider validation; the
no-deployment/no-provider-call statements below apply to this earlier batch only.

Scope: the next two approved build points: correct result resolution and safe
outcome grading. This is **not completion of Phase 2**. Independent scheduling,
closing-price capture, bookmaker-policy approval, and live historical recovery
remain separate work.

No production database records, active models, prediction/selection formulas,
release modes, or schedules were changed. No paid provider calls or historical
settlement were run. Verification uses mocked responses and isolated databases.
No new dependency or database migration is required.

## Implemented

### 1. Resolve the actual fixture, not its publication day

The canonical tracker now delegates to `PredictionService.resolve()` and the
shared `SettlementService`. Both real and dry-run calls take this route when
`PLATFORM_ENABLED` is on and no custom legacy database is supplied.

- Group candidates by the exact API-Football fixture ID and fetch `/fixtures?id=...`.
  Do not search for a similarly named team on the card's publication date.
- A date cutoff refers to **fixture kickoff in UTC**, not publication date.
  Publication dates and original selections/prices are not rewritten.
- Use the latest stored canonical fixture schedule when available to avoid
  fetching future/rescheduled matches; re-check the provider's actual kickoff
  before grading. This still depends on the upstream schedule staying current.
- Verify fixture and competition identity. Missing IDs cannot use fuzzy matching.
- Reuse the fixture response across all its selections. Download team statistics
  only when needed; no player endpoint calls. No network occurs just by importing
  the service. Runtime fetches use the existing client/pacing/retry handling.
- A fixture/persistence failure is reported and does not stop other fixtures.
  Canonical failures never silently fall back to the unrelated legacy database.
- `--dry-run` uses the same canonical candidates but writes neither outcomes nor
  pending metadata. It **can make provider requests**; it is not an offline mode.

The provider documents fixture IDs and separate full-time/extra-time/penalty
score fields in its [official API-Football guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).

### 2. Grade only when the evidence is sufficient

- Validate the saved exact market key, bookmaker, participant, side, line and
  `regulation_time` period against the prediction. Old missing periods remain
  unknown rather than being inferred from a convenient final score.
- Preserve full wins/losses, half wins/losses, pushes and voids as distinct stored
  outcomes. Quarter-line handicaps now split the stake correctly, including
  negative lines. Unit-return reporting continues to use the existing shared
  outcome-accounting functions; no new ROI formula was introduced.
- Goals/BTTS/moneyline/handicaps/correct score use regulation scores. For AET/PEN
  matches, missing full-time scores cannot fall back to aggregate/shootout goals.
  Unsegmented corner/card/SoT statistics from these matches remain pending.
- Pair team statistics by provider team ID, never fuzzy names. Unknown,
  negative, fractional, boolean, NaN or infinite count inputs cannot become
  valid results. Explicit zero remains a valid observation.
- Missing red cards no longer silently become zero in the legacy API helper.
- Keep yellow-only and yellow-plus-red card markets distinct. **No production
  bookmaker card policy is approved by this batch.** The injectable reviewed
  rule must specify the exact bookmaker/key/period, a version, evidence reference,
  count definition and confirmed compatibility with the provider statistics.
  Synthetic tests demonstrate this boundary; they are not bookmaker validation.
- Postponements/abandonments are not automatically void. Cancellation/award/
  walkover outcomes also require an explicit applicable bookmaker policy; none
  is installed by default. Unknown cases stay pending, never invented losses.
- Generic legacy `cards` grades are withheld because those rows do not establish
  their precise card market/bookmaker rules.

### Evidence, pending states, and corrections

`Prediction.extras.settlement` records the engine version, result evidence,
policy, evidence hash, pending reason, first/last check times and attempt count.
Repository reads expose this metadata, and resolver/CLI reports list pending
reasons. This is **not a new website pending-results dashboard**.

Retries cannot add stakes or overwrite a newer grade. Writes compare the
candidate's expected grade/evidence and use a conditional database update.
Explicit corrections require `regrade=True` plus specific `prediction_ids`.
They preserve prior outcomes and evidence in a history. An identical correction
is a no-op. If rechecking finds insufficient evidence, the prior stored grade is
retained with `review_pending` and its evidence preserved for manual review;
the service does not silently manufacture a new result.

The CLI exits nonzero on reported errors. Partial/non-final/provider-null results
are exposed as pending rather than automatically treated as process errors.
Operational retry/backoff/alert policy belongs to the scheduling step.

## Verification

The focused suite passes **165 tests** covering settlement, publication lineage,
canonical repositories, Match Reads, website API/release, Discord hubs/dispatch,
worker cycles, baseline preservation and shared Asian/ROI accounting.

Settlement cases include early publication, changed team names, UTC boundaries,
rescheduled future fixtures, duplicate/concurrent retries, targeted corrections,
malformed inputs, missing statistics, legitimate zeros, wrong IDs/competitions,
full/half outcomes, card-rule mismatches, AET/PEN, cancellation policies and dry runs.

Tests establish the implementation contract, not the correctness of every
historical provider result, approval of real bookmaker rules, or a live ROI.
Postgres production concurrency has not been exercised; tests use SQLite.

## Activation and next steps

Long-running bot processes need a restart to load the changed resolver; no
process was restarted here. Existing bot-triggered/manual canonical resolution
will then use this path. **A new independent settlement schedule is not installed.**

1. Review a bounded provider-backed dry run and verify applicable market rules.
   Older unknown-period/card records can correctly remain pending; this is not
   permission to fill all 467 historical outcomes using guessed semantics.
2. Schedule settlement independently of Discord, with bounded batches, transient
   failure retries/backoff and visible status. Do not repeatedly spend API quota
   trying to repair missing market identities or unapproved bookmaker policies.
3. Build exact-market/bookmaker closing-price capture independently.
4. Recover older outcomes only where their identity and final evidence permit it.

No live acceptance gate or historical backfill has been ticked off by this batch.
