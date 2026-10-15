# Complete weight-setup comparison — 2 October 2026

The bounded comparison is complete. **Goals and corners show small, consistent
development gains; none of the four preselected setups meets the frozen 0.5%
minimum improvement.** Shots on target and cards also have uncertainty intervals
that include no improvement. This is a research result, not a production decision.
The owner authorized testing and explicitly reserved implementation decisions.

## What was compared

The [protocol](phase4-weight-comparison-protocol-2026-10-02.md) fixed 60
configurations per market, including references: 240 configurations in total.
It includes individual changes and deterministic sampled combinations of
own/opponent weighting, venue weighting, season/recent weighting, recent window,
recent calendar decay and profile pooling. Goals additionally vary attacking xG
share; cards vary pooling, referee treatment and foul contribution.

The distributions stayed fixed: coherent Dixon–Coles goals, retained corner
alpha 0.025 and SoT alpha 0.01, and the fuller card control's per-fixture dispersion.
This isolates changes in the estimated rates. It does not fit a new distribution,
calibrator, ML model or betting-selection policy. The original corner/SoT
variance controls and the earlier card candidate are diagnostic references.

All configurations within a market use identical fixtures. Tuning uses 2022;
one setup per market was locked before generating its 2023 forecasts. The
2023 results for every other setup are descriptive, not an opportunity to replace
the locked choice. Both years were already development data, so this is not an
untouched qualification. No 2025 or later outcomes were read for this comparison.

| Market | 2022 tuning fixtures / weeks | Later development fixtures / weeks |
| --- | ---: | ---: |
| Goals | 1,296 / 28 | 1,533 / 33 |
| Corners | 1,296 / 28 | 1,504 / 33 |
| Shots on target | 1,296 / 28 | 1,521 / 33 |
| Weighted cards | 2,361 / 34 | 1,730 / 23 |

The statistical markets cover EPL, LaLiga, SerieA, Bundesliga and Ligue1, with
both teams having at least eight current-season fixtures. Cards cover the ten
domestic leagues in the qualified card dataset under the same history gate.
Cards use April–December 2023 to preserve the earlier calibrated candidate's
valid comparison period; the other markets use eligible fixtures throughout 2023.
There are no eligible Q3 fixtures in the five-league mature statistical scope;
cards have only three in Q3. These are observed eligible weeks, not artificial
sample splits. Coverage files preserve scope and market-availability exclusions.

## Primary results

NLL measures the probability assigned to actual outcomes; lower is better.
Goals use the joint home/away score, other markets the total count. Percentage
improvement is relative reduction in NLL, **not percentage-point accuracy, hit
rate or ROI**. Difference intervals below zero favour the changed setup.

| Market | Reference NLL | Preselected NLL | Improvement | 95% weekly interval for NLL difference | Holm-adjusted p | Frozen screen |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Goals | 2.917412 | 2.908980 | 0.289% | −0.014129 to −0.002604 | 0.0060 | Below 0.5% minimum |
| Corners | 2.613188 | 2.604277 | 0.341% | −0.013831 to −0.004285 | 0.0040 | Below 0.5% minimum |
| Shots on target | 2.524539 | 2.523918 | 0.025% | −0.007854 to +0.006159 | 0.4498 | Below minimum; uncertainty includes zero |
| Cards | 2.217344 | 2.214711 | 0.119% | −0.006115 to +0.000492 | 0.1399 | Below minimum; uncertainty includes zero |

Goals and corners pass the declared uncertainty/multiplicity checks in this
development comparison. They fail the independently declared minimum size of
improvement; that threshold has not been relaxed. Neither has a supported slice
or related-market regression above 2%. Their average NLL improves in all five
leagues and all three represented quarters. This does not establish separate
league-specific qualification.

The 2022-preselected goals and corners setups also rank first in the descriptive
2023 tables. SoT's preselected setup ranks 18th of 60, and cards' ranks 12th.
The best observed new SoT setup improves NLL by 0.124%, and the best new card
weight setup by 0.211%; both are descriptive and below 0.5%. The earlier card
pool16/NB/calibrated candidate remains the strongest card diagnostic at 0.898%
development improvement. That is an already-known result; its previously failed
later qualification is not reopened or reversed here.

## Which settings produced the primary results

These are the statistical preselections, not approved implementation choices.
Every exact option and setup ID is saved in `registry.json` and `selection.json`.

| Setting | Goals | Corners | Shots on target | Cards |
| --- | --- | --- | --- | --- |
| Own / opponent | 60 / 40 | 50 / 50 | 60 / 40 | 70 / 30 |
| Base season / recent share | 85 / 15 | 90 / 10 | 80 / 20 | 100 / 0 |
| Venue share | 75% | 25% | 50% | 25% |
| Recent window | 6 matches | 8 matches | 6 matches | 4 matches |
| Recent decay | 90-day half-life | 90-day half-life | Original positional 0.85 | 30-day half-life |
| Profile change | Previous-season strength 16 | Previous-season strength 8 | Previous-season strength 16 | Pooling strength 16 |
| Additional setting | Attacking xG share 75% | — | — | Original referee modifier + anchor; foul blend 15% |

All still use current and immediately preceding provider seasons. Strength 16
means a prior coefficient of `16 / (current_match_count + 16)` where prior data
exists; it is not a 16-match history cap. Calendar decay applies only to the
recent component, not the whole season. The original divergence adjustment
remains: when recent and season estimates differ sufficiently, it can transfer
up to 30 percentage points toward recent form. Thus cards' zero **base** recent
share does not remove recent data or make its window/decay irrelevant.

## Numerical accuracy, reliability and scope

| Market | MAE reference → candidate | RMSE reference → candidate | Mean prediction error reference → candidate |
| --- | --- | --- | --- |
| Goals | 1.2936 → 1.2880 | 1.6355 → 1.6274 | +0.0718 → +0.0592 |
| Corners | 2.6889 → 2.6585 | 3.3630 → 3.3314 | −0.0962 → −0.1099 |
| Shots on target | 2.3963 → 2.3995 | 3.0729 → 3.0705 | +0.0486 → −0.0348 |
| Cards | 1.8246 → 1.8127 | 2.2954 → 2.2862 | +0.1839 → +0.1566 |

The artifact includes interval coverage, reliability bins, Brier/log scores,
coherent BTTS/winner/handicap diagnostics and complete Asian category checks.
None of the primary comparisons has a supported >2% related-market regression.
SoT nevertheless slightly worsens MAE and worsens NLL in Bundesliga and SerieA;
an aggregate positive number is not universal improvement.

The 8–15-match and partial/missing-xG slices still lack the declared temporal
support. In cards no individual league or season-stage slice meets both support
requirements, despite sufficient pooled support. Missing/sparse foul context
also lacks support. This experiment cannot authorize those scopes by inference.
Card intervals remain conservative: roughly 91.2% coverage for nominal 80%
intervals. Reliability improvement has not been assumed from a lower mean error.

Saved historical evidence remains `assumed_final`, with reconstructed dated
inputs rather than original published forecasts. Unknown original referee
assignment time, incomplete lineup/context and card bookmaker-equivalence
limitations persist. Weighted-card v2 targets do not silently replace production
settlement v1. No authentic quote-based betting evaluation was performed.

## Validation, artifacts and operational boundary

Successful artifact:
`Research/phase4-weight-comparison-2026-10-02-v4/`.
It preserves the source/protocol registry, dependencies and hashes, dated input
variants, memberships, all predictions and primary losses, frozen 2022 ranking,
2023 scoreboard, detailed primary diagnostics and validation record. See
`VALIDATION.json`, `comparison.json`, `all-setups.json` and `COMPLETE.json`.

The 111 focused tests cover original-control parity, future-result isolation,
missing-versus-zero handling, chronological selection, deterministic registries,
distribution likelihoods and diagnostic array interfaces. Full saved forecast
replay, source/membership/target checks and independent likelihood recomputation
are recorded in `VALIDATION.json`: 2,829 input snapshots rebuilt, 744,038 forecast
calculations replayed with zero difference and 748,129 likelihoods recomputed.
All 458 previously fingerprinted existing
source/test/model/dependency files remain unchanged.

Initial runner attempts are preserved. One stopped before scoring on the card
archive's SQL-style UTC timestamp convention. Subsequent reporting attempts
exposed NumPy scalar incompatibilities with strict archived helpers; these were
fixed at the new reporting interface and regression-tested. The final artifact
imports completed v2 forecasts, losses and locked selections **byte for byte**,
with `FORECAST_IMPORT.json` proving their hashes. No candidate, range, date,
threshold or preselection was changed after scores were observed. The independent
likelihood audit allows 1e-12 floating-point rounding while requiring identical
preselected IDs and support; it does not rewrite the original ranking.

Verification commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -c "from Scripts.data_platform.features.benchmarks.artifacts import verify_complete; verify_complete('Research/phase4-weight-comparison-2026-10-02-v4')"

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
Scripts/tests/test_phase4_weight_setups.py \
Scripts/tests/test_phase4_retained.py \
Scripts/tests/test_phase4_card_candidates.py \
Scripts/tests/test_phase4_candidates.py \
Scripts/tests/test_phase4_backtest.py
```

The original engine, active weights/models/profiles/databases, public delivery
modes and retained corner/SoT research bundle were not changed. No provider calls
or publications occurred; no 2025 qualification or later final-system/prospective
outcomes were opened. Phase 5 and additional optimization have not started.
**Implementation and any further testing decision remain for owner review.**
