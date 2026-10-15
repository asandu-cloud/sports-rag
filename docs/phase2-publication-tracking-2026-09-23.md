# Phase 2 — tracking audit and publication identity

Date: 23 September 2026.

Scope: points **1 and 2 of the implementation order** discussed with the user:
audit existing connections, then make published selections identifiable and
preserve their publication history. This is not completion of all Phase 2.

No numerical prediction formulas, bet-selection thresholds, database records,
release settings or schedules were changed by this implementation. No API calls,
grading, historical backfill, migrations or model training were run. Tests use
isolated databases. New metadata takes effect when the updated processes run.

## 1. What the audit established

The normal path already exists:

`MarketResult → persisted Match Read/selection → website release or Discord
delivery → PublicationService → PublishedRecommendation + Prediction → outcome
resolver → shared reporting`.

Website release uses `MatchReadReleaseService`; Discord delivery is recorded by
`auto_push.py` after sending/editing the post. Both reach
`MatchReadDeliveryService`. Merely generating a shadow read, visiting a page or
refreshing the browser is not publication.

Read-only evidence:
`Index/prediction_experiments/phase2-tracking-audit-2026-09-23.json`.

| Observed item | Count |
| --- | ---: |
| Canonical predictions | 1,799 |
| Canonical published recommendations | 467 |
| Unresolved published recommendations, all on past kickoff days | 467 |
| Published recommendations with closing odds | 0 |
| Published recommendations missing explicit system/model version | 467 |
| Publications dated differently from the fixture kickoff day | 192 |
| Delivered Match Read selections missing publication links | 0 |
| Legacy predictions in the separate `predictions.db` | 3,870 |
| Legacy unit-bet slips | 26 |

All 467 recorded recommendation deliveries in this snapshot are website
deliveries. This does not prove Discord sent nothing historically; it means
there are no canonical Discord recommendation-delivery records here.

### Confirmed gaps, rather than assumed explanations

- **Date mismatch in settlement:** canonical `prediction_date` is the
  publication date. `_maybe_platform_resolve` groups by that date and league,
  then `_fetch_results_from_api` requests fixtures for that day. An early card
  can therefore be looked up on the wrong matchday. The 192 mismatches are
  direct evidence of exposure to this defect, not proof this explains all 467
  unresolved rows. The next settlement step should resolve by fixture ID and
  actual kickoff, retaining publication date as a separate audit field.
- **Bot-dependent scheduling:** result resolution is invoked from Discord
  tasks/manual commands. The daily/recap tasks can return before resolving if
  their configured Discord channel is absent. The standalone Match Read worker
  does not independently settle outcomes. We did not run the bot, invoke a
  resolver, or establish the runtime history of every missed job.
- **Closing-price backend mismatch:** `capture_closing_odds()` opens the legacy
  SQLite tracker directly; it does not use the canonical closing-price shim.
  Its query also uses prediction/publication date. The Discord ten-minute loop
  therefore is not a working canonical near-kickoff quote-capture service.
- **Unsafe closing-price identity:** matching accepts either yellow-only or
  total-card keys for generic `cards`, and does not require the originally
  selected bookmaker. The old capture retains a first captured price rather
  than proving it is the latest eligible pre-kickoff price.
- **Unsafe missing-card grading:** the API result helper still uses `or 0` for
  unavailable yellow/red values. This must be repaired and verified in the
  settlement step before enabling independent automatic canonical grading.
- **Inconsistent reporting policy:** aggregate reporting previously counted
  every released price version, while the daily recap kept the best logged
  price for a repeated leg. That was not a consistent original-publication
  performance record.

The new leagues are already in the API result league-ID registry. Their absence
from that registry is not the explanation for these unresolved records.

A subsequent read-only run of the new lineage query classifies the 467 existing
releases as **276 initial-card selections and 191 amendments**, with no
unclassified releases in this particular database. All remain unresolved and
all historical system versions remain unknown. This is a classification of
saved publication evidence, not a backfill or a measured performance result.

## 2. Implemented identity and publication-history changes

### Prospective system and selection identity

- Canonical evaluations now carry a `system_version` fingerprint and a saved
  manifest of deployed numerical source files and available ML artifacts. It is
  cached per process; restart processes after changing those files.
- This identifies the deployed code/artifact set. It **does not assert a model
  contributed**, invent an unknown `model_version`, or replace the complete
  numerical-input/profile/quote snapshots still required for Phase 0 replay.
- New canonical full-game quotes carry `period: regulation_time`. Old quotes
  without a recorded period stay unknown; no historical rewrite was performed.
- Publication metadata retains fixture ID, precise priced market key,
  participant, side, line, bookmaker, period, snapshot reference, pipeline/model/
  system versions, and separate fixture/publication dates. Exact price,
  probability and original decision remain in the immutable publication.
- Yellow-only and total-card market keys stay distinct even under the same
  broad `cards` model group. Card definitions are labelled as needing bookmaker
  rule verification; this label does not itself change the old resolver.
- Quote/capture times stay unknown when not supplied. Publication/evaluation
  timestamps are not relabelled as bookmaker quote timestamps.

### Idempotency and transaction safety

- Release identity now includes the canonical decision, including model/system
  changes. Two different probabilities or models cannot collide just because
  their fixture, quote and old event-only snapshot hash match.
- Evaluation wall-clock time alone does not create another release.
- Existing old-key releases are reused when their immutable decision matches;
  retrying an old stored Match Read does not manufacture a new recommendation.
- Both delivery surfaces share one tracked prediction for the same release.
- The Match Read delivery, recommendation/prediction creation, recommendation
  delivery, and selection links commit in **one transaction**. An exception on
  the second selection rolls back the entire publication. A retry can then
  complete it. Competing surface transactions are retried safely.
- External Discord posting is necessarily outside the database transaction.
  If Discord succeeds and persistence fails, its external reference still has
  to be retried by the delivery path; this is not a distributed exactly-once
  guarantee over Discord itself.

### Original publications and amendments

The implemented headline policy is **the selections in the first delivered
Match Read for each fixture, across either surface**. All compatible selections
on that card count; there is no one-bet-per-fixture restriction.

- Select the first card using delivery time, then delivery ID as a deterministic
  tie-breaker. Never use outcome, best odds, or latest settled version.
- Later changed selections and later additions are amendments. An unchanged
  release reused by a later card stays the same recommendation, not another stake.
- If the first visible card is no-bet/unavailable, subsequently introduced bets
  belong to amendments, not retroactively to the original card.
- Unlinked old/direct publications are `unclassified`, not silently assigned a
  first-card history. They remain accessible in the all-releases audit.
- Selecting the original card considers unresolved and no-bet records too.
  A winning amendment cannot replace an unresolved original in headline results.
- Published, shadow, legacy standalone and unit-bet-leg records have separate
  cohort labels. Existing unit-bet slips keep their own accounting; individual
  legs are not silently represented as slip ROI.

Official platform queries default to `publication_scope="initial"`; explicit
`"amendments"` and `"all"` views remain available. `all` is a release-version
audit, **not a claim that users should stake on every revision**. Daily reporting
no longer substitutes the best later price for an official recommendation.

The website track-record API exposes the same scopes and identity fields.
Discord's official calls use the same canonical repository and default scope.
Official reads cannot silently fall back to the unrelated legacy database when
canonical access fails or the general legacy-backend flag is disabled. Legacy
non-official calls and explicit custom SQLite audit paths retain their old route.

Examples (existing local web API, no new visual dashboard in this batch):

```text
/api/track-record?publication_scope=initial
/api/track-record?publication_scope=amendments
/api/track-record/recent?publication_scope=all
```

## Verification

**110 tests and four additional regression subtests passed.** The focused runs
cover publication lineage, shared tracking, Match Read services/release/API,
canonical market contracts, settlement-accounting regressions, unit slips,
baseline preservation, guardrails, dispatch, Discord hubs and worker cycles.

Regression tests cover:

- Multiple initial picks, cross-surface deduplication and repeated deliveries.
- Concurrent website/Discord publication and a fresh service instance retry.
- Transaction rollback after a partially processed multi-selection card.
- Lineup amendments, first no-bet cards and unresolved-original/settled-amendment cases.
- Model/system/probability/period/provider-market changes with unchanged input IDs.
- Old-key compatibility and preserved unknown historical provenance.
- Distinct card keys; no invented quote timestamps.
- Shared website/Discord reporting, cohort filtering, invalid scope rejection,
  and visible canonical-access failure instead of legacy fallback.
- Read-only audit access and unchanged existing prediction/guardrail behaviour.

## Next: settlement, not another historical download

1. Resolve using canonical fixture IDs and verified final data, not publication day.
2. Preserve missing outcomes and unsupported card definitions as explicitly
   pending; test full/half wins/losses, pushes, voids, periods and corrections.
3. Schedule that settlement independently of Discord, with visible retries and failures.
4. Implement canonical, exact-market/bookmaker, pre-kickoff closing-price capture.
5. Repair old results only when their identity and final evidence support grading.

The 467 existing outcomes and all missing historical versions/closing prices
were **not** filled in by this batch. These remain later Phase 2 work.

Repeat the read-only counts without API calls:

```bash
./.venv/bin/python Scripts/ops/prediction_tracking_audit.py
```

No new database migration or weekly manual command is required for the identity
changes. Long-running web/bot processes must be restarted to load new code;
new one-shot worker processes load it on their next invocation. This work did
not change or restart any schedules or public-release modes.
