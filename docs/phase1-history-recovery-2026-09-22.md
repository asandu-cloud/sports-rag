# Phase 1 historical recovery — 22 September 2026

Status: the approved local recovery is implemented and applied. **Phase 1 is
not yet fully signed off for unrestricted modelling.** Card-stat ambiguity
remains; the separate Phase 0 full numerical-decision replay is also open.
No prediction formulas, thresholds, release settings, or active models changed.

## What was recovered

The original five leagues and UCL/UEL/UECL had 2025 team exports, but no 2025
fixtures in the canonical database. The exports retained fixture IDs but lost
team IDs and converted missing statistic values to zero.

Eight successful API-Football `/fixtures` season-list requests verified the
provider competition, season, fixture ID, home/away team IDs, kickoff, FT status,
and final score. There were **no per-fixture statistics or player requests**.
The initial sandbox attempt failed DNS resolution; its incomplete preparation
directory has no completion marker and must not be used.

| Competition | Verified FT fixtures added | Paired local team statistics | Fixture/results only |
| --- | ---: | ---: | ---: |
| EPL | 380 | 370 | 10 |
| LaLiga | 380 | 370 | 10 |
| SerieA | 380 | 370 | 10 |
| Bundesliga | 307 | 306 | 1 |
| Ligue1 | 307 | 305 | 2 |
| UCL | 272 | 222 | 50 |
| UEL | 255 | 254 | 1 |
| UECL | 382 | 212 | 170 |
| **Total** | **2,663** | **2,409** | **254** |

Counts describe the provider's FT fixture list, including rounds it assigns to
that competition; they are not assumed round-robin schedule sizes. AET/PEN rows
are not treated as regulation-time totals.

One legacy Ligue1 fixture, `1544684`, was absent from the provider's FT list and
was quarantined. No guess was made about its identity or completion period.
The 254 fixture/results-only entries have known scores and unknown event stats;
they do not become fabricated zero-corner/zero-card games.

The importer added **5,326 team rows and one missing team catalogue entry**.
A before/after comparison confirmed zero pre-existing rows changed or removed
in fixtures, team statistics, player statistics, teams, or seasons. Existing
team names were preserved. No player rows were added: this work repairs the
team-history inputs required by the current candidate feature contract, not a
player-history backfill.

## Safety and provenance

Implemented in:

- `Scripts/data_platform/features/historical_recovery.py`: fixture-scoped exact
  matching, conservative statistics conversion, quarantine, additive import.
- `Scripts/ops/prediction_history.py`: bounded metadata preparation, frozen
  source files/checksums, plan verification, gated import and read-only card audit.
- `Scripts/tests/data_platform/test_historical_recovery.py`: identity conflicts,
  duplicate rows, invalid numbers, unknown versus zero, idempotence, preserved
  names, source tampering, request budget and historical observation timestamps.

Preparation is immutable and repeat verification rebuilds the plan from its
frozen inputs. Applying it takes the existing refresh writer gate, makes a
SQLite online backup, and commits the import in one transaction. Existing
fixtures are skipped, not overwritten, even when a proposed row conflicts.

Imports record their source hash and conservative zero policy in `stats_json`.
Their observation timestamps are current, not backdated. Fixture payload digests
remain unset because player/raw-stat ingestion is incomplete; a future proper
bootstrap must not mistake them for fully fetched fixtures. Sync watermarks,
KB queues, profiles, embeddings and model artifacts were not changed.

Local private artifacts (all under `Index/prediction_experiments/`):

- `phase1-history-2025-2026-09-22-verified/`: frozen exports, provider lists,
  checksums, import plan and quarantine/coverage report.
- `phase1-history-import-2026-09-22/`: `platform-before.db`, import result and
  `additive-verification.json`. The backup may contain application user records;
  do not commit or publish it.
- `phase1-card-audit-2026-09-22/`: pre-import card evidence audit.
- `phase1-card-audit-after-recovery-2026-09-22/`: post-import audit.

## Card-stat investigation

The legacy extractors explicitly changed API `null` to `0` before saving.
Consequently, those exports cannot distinguish a real zero from unavailable
data. Recovery retains nonzero valid statistics but treats ambiguous exported
zeros as unknown. Independently verified fixture score zeros remain genuine
zeros. No player-card sum is substituted for a complete team-card total.

The canonical database's missing card values were compared with its available
checksum-verified raw team-stat archives. There were **zero numeric repair
candidates**, zero conflicting numeric card values and zero invalid archives.
Where archives existed for missing cards, they also contained unknown values;
many older fixtures had no raw archive. This is not evidence that every null
means zero, and no blanket null-to-zero change was made.

Up to **2,636 per-fixture team-statistics requests** could replace ambiguous or
absent legacy event statistics. This excludes player calls and retries. A fresh
response may still contain null cards, so a re-download alone is not a promised
card fix. The user explicitly chose **local recovery and report review first**;
that backfill was not started.

## New candidate dataset

`phase1-features-2025-2026-recovered-2026-09-22` contains **5,083 fixtures** and
the unchanged **148-feature** schema, compared with 2,286 fixtures in the earlier
verified export. Growth includes both the recovered history and newer canonical
2026 results. The original datasets remain untouched.

| Market | Known result labels | Label plus both teams' required production features |
| --- | ---: | ---: |
| Goals | 5,083 | 4,809 |
| Corners | 4,668 | 4,481 |
| Shots on target | 4,652 | 4,464 |
| Cards | 546 | 463 |

These are availability counts, **not proof of model eligibility or accuracy**.
In particular, the small card subset can disproportionately select matches with
recorded dismissals. It must not be represented as an unbiased card-training or
evaluation sample. Zero-card/corner/SoT legacy matches cannot be reliably
recovered from the lossy exports and their labels remain missing, not silently
discarded as bad performances.

The dataset includes all 13 competitions. UECL has 2025 history but still no
completed 2026 rows at this snapshot. Canonical pairing quarantine is empty;
the one unmatched legacy record was excluded before import.

Availability is explicitly `assumed_final`: historical FT statistics are assumed
available three hours after kickoff. This is retrospective reconstruction, not
proof of the original data vintage or absence of later provider corrections.
Only the current and immediately preceding provider season contribute to each
feature vector; the historical recovery did not extend that modelling window.

## Verification and remaining gates

- 65 focused tests passed: recovery plus existing feature leakage, season
  transitions, candidate train/inference parity, isolated fitting and baseline
  preservation tests.
- The original 15 September baseline passed all six artifact checks and all
  187 compiler selection replays again. That check starts from saved canonical
  market results, not from historical profile/quote inputs.
- All **5,083** exported vectors replayed successfully with the archived
  `prematch-features.v1` builder and the `assumed_final` availability policy.
  No model was promoted. This verifies deterministic training/inference parity,
  not historical forecast accuracy.

Remaining before unrestricted modelling sign-off:

1. Decide how to recover/verify missing event statistics, especially card-zero
   semantics, and review market-specific selection bias. The larger API backfill
   remains deferred by choice, not silently running.
2. Capture a genuine future fixture's complete numerical inputs and full quote
   slate and replay projections, probabilities and selection offline. The
   canonical `odds_snapshots` table had zero rows in this audit; the old saved
   selections cannot recover unrecorded candidate quotes or old profile inputs.
   The original Phase 0 archive remains valid and unchanged, with its narrower
   boundary explicitly stated.
3. Review the resulting coverage/exclusions before Phase 3 benchmarking. No
   model has been trained or promoted during this recovery.

## Commands

No repeat import or weekly manual action is required for what was applied today.
Normal scheduled/incremental refreshes retain this historical data.

Verify the new export (local, read-only):

```bash
./.venv/bin/python Scripts/ops/prediction_features.py replay \
  --experiment Index/prediction_experiments/phase1-features-2025-2026-recovered-2026-09-22
```

For a **separately approved future recovery**, use new experiment names:

```bash
./.venv/bin/python Scripts/ops/prediction_history.py prepare \
  --experiment history-review-NEW --season 2025 --max-api-calls 8

# Inspect the frozen report before applying anything.
./.venv/bin/python Scripts/ops/prediction_history.py apply \
  --prepared Index/prediction_experiments/history-review-NEW \
  --experiment history-import-NEW
```

`prepare` makes eight fixture-list calls by default; transport retries may add
requests. It cannot download per-fixture statistics or players. `apply` is
additive and backs up the configured existing SQLite database. Existing
experiment directories cannot be reused. Neither command belongs in the
normal weekly refresh chain.
