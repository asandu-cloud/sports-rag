# Phase 4 local card reconstruction and backtest readiness

Subsequent owner-approved update: [two-minute candidate and league-phase scope](phase4-card-policy-v2-results-2026-09-30.md).
That new immutable run resolves the prior period-certification restriction for
known completed league formats and uses minutes >= 2. The v1 results below and
their artifacts are preserved; the remaining player-source/history gap persists.

30 September 2026. The local reconstruction and fixed comparison implementation
are complete and tested. **A valid real-data card backtest remains blocked:** no
locally reconstructed target meets the agreed source-completeness contract, and
the recovered player evidence provides no earlier 2022 fitting cohort. Zero real
forecasts were scored. No improvement or card-market qualification is claimed.

The owner authorized reconstruction and backtesting, explicitly stopping before
weight optimisation. There were **zero API calls**, no database/history repairs,
no changes to production models, coefficients, referee profiles, workers or public
delivery. API-Football availability was not tested. The final-system and prospective
outcomes remain protected. The existing refresh-incomplete marker was observed
and preserved; this research snapshot does not certify publisher health.

Protocol: [local evidence and fixed-comparison rules](phase4-card-reconstruction-protocol-2026-09-30.md).
Final artifact: `Research/phase4-card-reconstruction-2026-09-30/reconstruction/`.
Separate replay: `Research/phase4-card-reconstruction-2026-09-30/replay/`.
Tests and independent arithmetic review: `Research/phase4-card-reconstruction-2026-09-30/review/`.
The earlier `evidence/` artifact is preserved as the first local run; the final
artifact additionally uses existing explicit registry aliases and saves replay
inputs and the missing-evidence ledger. No numerical scores drove this change.

## What was recovered

The consistent read-only database snapshot contains 19,648 permitted completed
fixtures (kickoff + three hours before 1 January 2024). Its 36,440 normalized player
rows cover 906 fixtures. The reconstruction also processed eight existing 2023
player-export JSON files outside the platform database:

- 68,048 total row identities inspected.
- **36,981 player rows** admitted to the permitted development artifact.
- 30,239 later-period rows skipped before decoding their statistics.
- 484 rows excluded because the same ordered teams met more than once in that
  provider competition/season; the files have no dates or fixture IDs to distinguish
  those matches. This includes European group-stage and knockout rematches across
  the development boundary. They were not disambiguated by inspecting outcomes.
- 340 rows retained as unresolved team identities (`Fenerbahce` / `Besiktas` do not
  exactly match the saved provider names); four rows have invalid player IDs.

Provider fixture identities were checksum-verified against the earlier identity
downloads. Unique competition/season/ordered-name matches were reconciled to exact
fixture and team IDs. Existing registry mappings for Verona and Köln recovered
527 rows; no fuzzy matching or new guessed aliases were used. No conflicting
provider identities or overlapping local player values were found in this run.

There is now some player evidence for **2,135 fixtures** and provisional weighted
arithmetic for **2,101 fixtures**, compared with 895 in the September 28 audit.
All 2,101 provisional totals are in calendar 2023. There are 52 provisional zeros
and 2,049 positive totals. Those zeros are not certified provider zero-card games.

| Competition | Fixtures with player evidence | Provisional arithmetic totals |
|---|---:|---:|
| EPL | 196 | 196 |
| LaLiga | 180 | 180 |
| SerieA | 180 | 180 |
| Bundesliga | 143 | 143 |
| Ligue1 | 153 | 152 |
| Championship | 300 | 299 |
| SuperLig | 170 | 162 |
| Eredivisie | 144 | 144 |
| PrimeiraLiga | 135 | 133 |
| BelgianProLeague | 157 | 157 |
| UCL | 133 | 133 |
| UEL | 90 | 90 |
| UECL | 154 | 132 |
| **Total** | **2,135** | **2,101** |

These are permitted calendar-2023 portions of provider season 2023, not full-season
coverage. None is a qualified training/evaluation target yet. The 34 arithmetic
failures comprise 11 carded players with unknown minutes, one inconsistent
yellow/red combination, and 22 fixtures with incomplete team identity coverage.
17,513 permitted fixtures have no reconstructed local player rows at all.

## Why the additional totals cannot yet support the backtest

1. **Original player evidence is missing.** All 18,172 files in `Index/raw_archive`
   were accounted for by registered archive paths. No extra unregistered files
   were discovered there, and no player/event paths were found in the history
   staging archive. The 367 registered player responses all belong to later
   reserved fixtures and were not opened. No event responses were registered.
   The two top-level platform database backups have no pre-2024 player rows.
   This is a bounded local inventory, not a claim about every external disk or
   duplicate research/baseline backup.
2. **Legacy player exports lose relevant participants and source structure.**
   All eight current collectors use `MIN_MINUTES = 1` and discard `minutes <= 1`.
   Every one of the 36,981 admitted legacy rows records at least two minutes.
   This is consistent with that filter, but the exact historical collector version
   is not certified. Our approved definition includes one-minute participants.
   We cannot establish whether omitted players had cards, or recover complete
   response envelopes/rosters from the surviving rows. Passing the parser's
   minimum roster checks cannot restore those missing records.
3. **Period provenance remains incomplete for part of the database cohort.**
   911 permitted fixtures lack the frozen regulation-source certification; 906
   of these contain the normalized player evidence in the five newer leagues.
   A normalized `FT` label alone does not resolve their historical import provenance.
4. **There is no earlier player-card fitting cohort in the inspected sources.**
   Even accepting all provisional totals would only supply part of 2023. The
   unchanged protocol needs an earlier chronological cohort: at least 500
   qualified fitting fixtures across 26 weeks, then 200 evaluation forecasts
   across 20 weeks with sufficient prior team observations. Thresholds and test
   dates were not reduced or moved to make this run pass.

The old red-card selection issue is still visible: of the 2,101 provisional
fixtures, 160 have known positive two-team red counts and 1,941 have unknown
two-team red evidence; none belongs to the normalized explicit-zero-red cohort.
This describes team-label availability, **not a claim that all these matches
contained dismissals**. Player arithmetic and complete legacy team labels are
different cohorts. Missing team reds were not changed to zero.

`card-targets.jsonl` preserves source alternatives, canonical versus local reasons,
provisional values and the separate null qualified target. The
`missing-evidence-NOT-A-COLLECTION-PLAN.jsonl` ledger records exact fixture IDs,
missing source evidence and endpoint/parameter identities. It is not a selected
API job or approval to fetch all listed fixtures; fixture evidence may be obtained
in consolidated calls. No collection command is automatically queued.

## Referee reconstruction and the fixed comparison

The permitted database cohort has referee assignments for **19,646 / 19,648**
fixtures. Assignment coverage is ample; historical weighted-card observations
are the limiting evidence. Stored full-season referee averages were not reused
as past pre-match estimates.

The new research module builds dated team, opponent-induced, league and referee
inputs exclusively from qualified earlier matches, using the current and previous
provider season and the prior UTC day as the information cutoff. Exact normalized
referee names are separated by competition; initials are not automatically merged
with full names. Sparse/unknown referees produce a recorded neutral fallback.

The comparison implementation uses a fixed participation-aligned statistical
reference with coefficients captured from existing configuration: own/opponent
0.70/0.30, venue/overall 0.50/0.50, season/recent 0.85/0.15, six recent fixtures
with decay 0.85, referee modifier weight 0.35 and anchor 0.15. It preserves the
existing divergence adjustment and referee confidence formula, including the
distinct five-match modifier and eight-match anchor thresholds. No weights,
history windows or distribution parameters were fitted or searched.

Team-only and team-plus-referee use identical eligible fixtures and fixed Poisson
distributions. The runner supports proper count/binary scores, reliability,
intervals, numerical errors, paired week uncertainty and coverage slices. It is
an isolated **simplified statistical reference**, with missing foul/aggression,
lineup, knockout and league-regime context disclosed. It is not the full frozen
production control or a replay of previously published predictions. A complete
card probability-engine comparison and dispersion/calibration qualification still
require qualified data and the full relevant control adapter.

The real run stopped at its evidence gates with zero fitting/evaluation targets.
Its predictions and dated-profile outputs are intentionally empty. Successful
synthetic chronology/engine tests do not establish a working real-data card model.

## Validation and reproduction

**147 tests passed**, covering the new reconstruction/comparison and existing
card qualification, participation settlement, historical-input and qualification
modules. Tests include one-minute participants, yellow/straight-red/second-yellow
arithmetic, null preservation, source conflicts, repeated fixture versions,
ambiguous European rematches, later-outcome opacity, referee chronology, neutral
fallbacks, identical evaluation cohorts and insufficient-support refusal.
Network/secrets/writable database/outside-artifact writes are checked explicitly.

A fresh process reproduced every one of the 19,648 target records and the blocked
backtest result using only the saved evidence. An independent arithmetic audit
recomputed all 2,101 provisional totals, checked the dates of 73,421 admitted
player rows, verified all 43 artifact files, and matched all 21 captured source
files against the working implementation. These are local verification results.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_card_reconstruction \
  --verify Research/phase4-card-reconstruction-2026-09-30/reconstruction

# Optional offline replay; choose a new output directory each time.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_card_reconstruction \
  --replay Research/phase4-card-reconstruction-2026-09-30/reconstruction \
  --output Research/phase4-card-reconstruction-replay-NEW
```

## Resume point

When provider access returns, prepare a separately approved, bounded evidence
diagnostic spanning both earlier history and later evaluation, selected without
conditioning on recorded dismissals. Check complete player responses, one-minute
participants, explicit zero/null semantics and regulation identities before
expanding collection. Qualification may still fail for individual responses.
Use those qualified targets to execute the dated team/referee comparison and
the remaining full-control/probability work. **Weight optimisation has not begun**
and remains a separate review with the owner.
