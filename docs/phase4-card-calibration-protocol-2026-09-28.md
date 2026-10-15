# Phase 4 first card qualification and calibration development batch

Protocol recorded before this batch's outcome audit and fitting, 28 September 2026.
This is an offline development batch, not a promotion or a final Phase 4 baseline.

## Access and scope

Use only completed fixtures whose kickoff plus three hours precedes 1 January
2024. Read the canonical SQLite database in one read-only transaction and export
only that bounded cohort; never decode later player/team outcomes or raw archives.
For period certification reuse checksum-verified, regulation-qualified history
from the frozen Phase 3 export, skipping later history objects before JSON decoding.
Reserved archive inventory uses metadata only. Preserve every existing artifact.
No provider requests, database repairs, worker changes or public publications.

## Card target contract

Reuse the existing `spix-participation-settlement.v1` parser: positive recorded
minutes, yellow 1, straight red 2, two yellows plus dismissal 3; regulation plus
stoppage time. AET/PEN aggregates, ambiguous counts, partial participation and
identity conflicts are excluded. A normalized arithmetic result is provisional
until regulation identity and source completeness are evidenced. Do not assume
raw archive absence proves a count is wrong; record it as an unresolved provenance
restriction. Normalized rows discard the original response envelope, duplicate
statistics and roster completeness, so they alone cannot certify an equivalent
training target. Team yellow/red counts are diagnostics, never replacement targets.

Produce per-fixture reasons, league/season coverage, explicit zero/positive/null
counts, red-card enrichment diagnostics, normalized candidate targets separately
from qualified targets, and a metadata-only collection proposal. No collection is
approved by this protocol. Source conflicts must remain visible. Database snapshot
consistency does not establish historical forecast-time availability: label this
reconstruction `assumed_final`. A later extraction can qualify additional evidence
without overwriting this batch or resetting reserved-period inspection history.

A market fitting cohort needs at least 500 fixtures and 26 ISO weeks in its fitting
period, and 200 fixtures and 20 weeks in evaluation. Per-league conclusions need
200 evaluation fixtures and 20 weeks. Report support and exclusions; a pooled
minimum does not qualify individual leagues. These are minimums, not proof of
representativeness. No-red fixtures must exist; compare coverage of fixtures with
and without recorded team reds, preserving unknown as a separate category.

## First calibration experiment

Proceed independently with goals, corners and SoT while card evidence is assessed.
Reuse the immutable prior-transition experiment's **reference** statistical means
(the installed cutoff policy's reconstructed means), one row per fixture/market.
Do not use its selected candidate predictions or choose a new prior strength here.
Those means omit historical live inputs and are reconstructions, not authentic
published forecasts. This pilot compares count distributions around fixed means,
not the complete installed distribution (which has other variance/dependence
logic). It is not yet a goal-score joint-distribution or market-policy evaluation.

Fit on calendar 2022, evaluate on calendar 2023. Both are already development data;
2023 results are diagnostic, not a new untouched confirmation. Never access 2024,
the Jan–Jun 2025 calibration reserve, Jul 2025–Jun 2026 final-system test or later
reserve. Label availability is assumed kickoff + 3 hours; reject rows that cross
fitting/evaluation cutoffs. Entire fixtures and simultaneous kickoffs stay together.

Compare Poisson against negative binomial variance `mu + alpha * mu^2` with fixed
alpha grid `[0, .01, .025, .05, .1, .2, .4, .8, 1.6]`. Select alpha by 2022 mean
count negative log likelihood only; ties prefer smaller alpha. Alpha zero is valid.
No mean fitting, target correction, league-specific fitting or recency tuning.
Primary metric: 2023 count negative log likelihood. Secondary: Brier score/log loss
at predeclared half-lines (goals 2.5, corners 9.5, SoT 8.5), reliability bins and
central 80% interval coverage/width. Report league, season, season-stage and
missingness slices. Use 1,000 paired ISO-week bootstrap draws, seed 20260928.
An encouraging development result requires primary-score improvement with its
95% interval below zero and no supported league slice worsening by more than 2%.
A failed criterion is reported; do not change thresholds after observing scores.
No ROI or CLV without authentic equivalent quotes. Binary scores at half-lines
are not a substitute for Asian full/half-win/push distributions.

Preserve source/input hashes, protocol, dependencies, fit parameters, predictions,
metrics, uncertainty, exclusions and a COMPLETE checksum manifest. Tests must
cover settlement parity, unknown versus zero, period/identity exclusion, immutable
outputs, date isolation, fit-only selection, deterministic results and probability
normalization. Later work includes card projection semantics, coherent goals,
logistic probability comparison, dedicated calibrators, full Asian probabilities,
calibration-reserve protocol and prospective acceptance. This batch closes none
of those simply by producing a report.
