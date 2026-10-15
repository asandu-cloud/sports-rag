# Phase 4: first card qualification and calibration development results

28 September 2026. Implemented and exercised locally on real stored historical
data. No new provider requests, public recommendations or production activation.
This is a completed first development batch, not completion of Phase 4.

Protocol: [declared scope and criteria](phase4-card-calibration-protocol-2026-09-28.md).
Artifact: `Research/phase4-card-calibration-2026-09-28/evaluation/`.
Entry point: `Scripts/ops/phase4_development.py`. Pure research modules:
`Scripts/data_platform/features/phase4_cards.py` and `count_calibration.py`.

## What was implemented

- `phase4-participation-card-target.v1`, reusing the approved settlement parser.
  Normalized provisional arithmetic, qualified targets and exclusion reasons are
  separate fields. Weighted totals are never substituted with team yellow/red sums.
- A read-only, transactional canonical export restricted in SQL to completed
  fixtures with kickoff + three hours before 1 January 2024. It saves the exact
  selected rows, source digests and archive metadata for replay. Frozen Phase 3
  history supplies verified regulation identities; future objects are skipped
  before JSON decoding. Source conflicts and unavailable evidence fail closed.
- Card coverage by league/season, field missingness, red-card enrichment,
  source/period exclusions, and a non-executable collection proposal.
- Training-period-only selection of negative-binomial dispersion around fixed
  statistical means, Poisson comparison, chronological evaluation, proper scores,
  reliability bins, intervals, league/season/stage/missingness slices and paired
  week-bootstrap uncertainty. No live prediction path imports this code.
- Source copies/hashes, input hashes, dependencies, protocol, fit parameters,
  per-fixture forecasts, metrics and a COMPLETE checksum manifest. The command
  refuses an existing output directory. Failed work retains an incomplete marker.

## Card findings

The permitted canonical cohort contains **19,648 completed fixtures**. There are
**36,440 player-stat rows across 906 fixtures**, all in provider season 2023 in
five domestic competitions. No player response archive belongs to this permitted
cohort. The 367 existing player-response archive records all belong to later
reserved periods; only their metadata was read. No event endpoint archive records
were found in the archive table queried. This is not a claim about every possible
unregistered file elsewhere in the repository.

| Competition | Player-covered fixtures | Provisional arithmetic totals | Unknown minutes for a carded player |
|---|---:|---:|---:|
| Championship | 300 | 299 | 1 |
| BelgianProLeague | 157 | 157 | 0 |
| Eredivisie | 144 | 144 | 0 |
| PrimeiraLiga | 135 | 133 | 2 |
| SuperLig | 170 | 162 | 8 |
| **Total** | **906** | **895** | **11** |

Of the 895 provisional totals, 27 are zero and 868 positive. These are arithmetic
results from normalized data, **not verified provider zero-card fixtures**. All
906 player-covered fixtures lack regulation-source certification in the frozen
history, as well as an original archived player response. Normalized player rows
lose the source response envelope and cannot certify whether original response
statistics/rosters were complete. Therefore **zero targets qualify under this
batch's explicit source/period contract**. This does not prove the arithmetic is
wrong, nor that no alternative source can ever qualify it.

The remaining 18,742 fixtures lack normalized player statistics. Across the full
cohort, 911 lack frozen regulation certification. Missingness reasons overlap;
the report does not add overlapping reasons into an invented fixture count.

The old card selection issue persists in this permitted cohort:

- 3,493 computable legacy team yellow-plus-red totals: 2,953 with positive team
  reds and 540 with explicit normalized team-red zeros.
- For all 19,648 fixtures, the two-team red category is positive for 2,997, zero
  for 540 and unknown for 16,111. Unknown is never counted as no red.
- Within the 906 player-covered fixtures, only 149 have computable legacy team
  card totals; **all 149 have positive recorded team reds**. Player arithmetic and
  complete team-label availability are plainly different cohorts.
- Team-stat rows contain 32,011 null red fields; player card fields in this cohort
  contain no nulls. Without original responses this difference does not establish
  that player zeros are provider-confirmed. Null versus omitted versus lossy old
  export cannot be recovered from these normalized columns alone.

These findings support exclusion reporting, not a calibrated card model. There
are no qualified 2022 card fitting targets here. Even recovering all 906 fixtures
would supply only a partial 2023 evaluation cohort, not the required earlier
fitting cohort. Historical coverage and provider zeros need independent evidence;
training on the currently computable team totals would retain selection bias.

The existing statistical card projection also remains semantically unqualified:
`projected_total_cards_detail` anchors to stored card rates and derives a yellow/red
split around that anchor; it does not reconstruct the approved weighted
participation target. Referee and lineup card-risk inputs must use compatible
rates too. Fitting a calibrator to legacy totals would not fix this mismatch.
No production card mean was silently changed by this batch.

## Evidence repair proposal and its limits

`collection-proposal-NOT-APPROVED.json` enumerates the 906 player-covered fixtures
and their missing evidence. Its simple per-fixture form contains **1,812 requests**:
fixture result/period evidence and fixture-player evidence for each fixture. It is
an inventory-derived proposal, not an approved collection job, a minimum request
count, or a complete training-data acquisition plan. League/season fixture calls
may consolidate identities; inspect provider support and a small chronological,
outcome-independent sample before any wider player collection. Re-downloading
may still return missing minutes or inadequate source evidence.

First seek independently archived source evidence for those exact identities.
If none can qualify them, prepare a separate bounded provider diagnostic spanning
both fitting and evaluation periods, including fixtures selected without reference
to red-card presence. Additional earlier player history would need explicit
collection authorization. Raw target repair alone is not sufficient: reconstruct
past team/referee/card-risk rates from the newly aligned targets, then freeze their
scope, chronology and sample requirements before card probability fitting. Local
arithmetic replay and other markets' calibration work can proceed meanwhile.

## First count-probability experiment

The immutable earlier prior-transition experiment supplied the **reference**
statistical means: the original cutoff-based historical reconstruction, not the
selected gradual-prior candidate. Each fixture/market reference was deduplicated
across that experiment's candidate strengths and checked for disagreement.
Fit on 2022; score on 2023. Both are already development data. The pilot compares
Poisson to negative-binomial dispersion with fixed means. **It does not compare
the fitted candidate against every variance/dependence component of the installed
probability engine**, and it does not yet generate a coherent joint goal model.

Primary score is count negative log likelihood (NLL; lower is better).

| Market | Fit fixtures | Evaluation fixtures | Selected alpha | Poisson NLL | Fitted NLL | 95% interval for fitted minus reference NLL |
|---|---:|---:|---:|---:|---:|---|
| Goals | 3,645 | 3,095 | 0.010 | 1.885857 | 1.886108 | −0.000581 to +0.001041 |
| Corners | 3,612 | 3,028 | 0.025 | 2.638010 | 2.630380 | −0.014055 to −0.000840 |
| Shots on target | 3,587 | 3,037 | 0.010 | 2.537908 | 2.536012 | −0.004157 to +0.000317 |

All markets have 43 fitting weeks and 41 evaluation weeks. Corners passes this
pilot's predeclared development criterion, including no supported league NLL
regression greater than 2%. Goals and SoT do not: their uncertainty intervals
include zero. Unsupported league slices are reported and cannot inherit a
market-wide qualification merely from the pooled result.

Half-line Brier scores also improve slightly: goals 0.246701→0.246598 at 2.5,
corners 0.252420→0.251215 at 9.5, SoT 0.240990→0.240793 at 8.5. These secondary
results do not override the primary criteria. Discrete central-80% interval
coverage rises respectively 87.75%→88.14%, 82.17%→85.77%, and 83.37%→85.48%; wider
intervals and discreteness must be considered, so higher coverage is not itself
proof of improvement. Full reliability tables and slice metrics are preserved.

No ROI, bookmaker equivalence or profitability was inferred. The corner result
is a reason to continue its probability review, not a decision to retain/remove
public corner selections or promote this distribution.

## Validation and operational limits

**408 focused and regression tests passed** across the new research tests and
existing participation, probability-engine and market-contract suites; results
are recorded beside the artifact. All 19 experiment artifact hashes and seven
research/shared-source hashes were also verified. They cover settlement parity, card missingness,
source/period conflicts, archive identity/checksums, reserved-history avoidance,
read-only SQL cutoffs, fixture-version conflicts, training-only fitting, deterministic
replay/bootstrap, normalized distributions and immutable output directories.
Tests run with outbound connections and dotenv loading disabled.

The audit used one consistent read-only transaction. The existing refresh-incomplete
marker was present and preserved: this is a diagnostic snapshot, not certification
that the whole live database is ready for publication. No worker was restarted,
no refresh was launched and no gate was bypassed. Original databases, archives,
active models, publishing behavior and the staged prior-transition candidate were
not modified. The 2024 outcomes and protected 2025+ calibration/final-system
outcomes were not decoded by this batch; 7,067 later frozen history objects were
skipped before decoding. These tests do not establish real-match acceptance.

## Remaining Phase 4 work

- Qualify representative weighted-card history and reconstruct compatible card
  projection inputs; fit only supported scopes. Preserve explicit pending reasons.
- Freeze the complete statistical baseline, including the reviewed prior policy,
  and compare its actual probability distributions with fitted candidates.
- Fit/evaluate coherent goal dependence and dispersion, the agreed logistic
  probability comparator and dedicated calibrators with full Asian outcomes.
- Freeze the exact protocol before accessing the dedicated calibration reserve;
  preserve the final-system test for the completed probability/selection system.
- Continue the corners scope review and prospective tracking separately.
- Execute the newly added [richer lineup-effects plan](phase4-richer-lineup-effects-plan.md)
  after the probability baseline work; it is not a new prerequisite delaying it.
