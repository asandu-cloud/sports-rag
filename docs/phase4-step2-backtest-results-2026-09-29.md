# Phase 4 Step 2 — individual-candidate backtest results

Completed 29 September 2026. This is a real chronological probability backtest of
individual candidates against the frozen statistical control. Three market-family
claims pass the predeclared development screen. This is not reserve qualification,
proof of profitability, a combined-system evaluation or production promotion.

## Main result

Lower negative log likelihood (NLL) is better. The percentages below are reductions
in probability loss, not increases in betting hit rate or ROI. Goals use the joint
home/away regulation-score likelihood; corners/SoT use total-count likelihood.

| Market / selected candidate | 2023 fixtures | Control NLL | Candidate NLL | Reduction | Paired 95% difference interval |
|---|---:|---:|---:|---:|---|
| strength_goals:10.0 | 3,095 | 2.954369 | 2.933113 | 0.719% | [-0.033133, -0.009622] |
| dispersion_corners:0.025 | 3,028 | 2.657631 | 2.630202 | 1.032% | [-0.034039, -0.020883] |
| dispersion_sot:0.01 | 3,037 | 2.573519 | 2.537215 | 1.411% | [-0.044871, -0.027480] |

All three clear the frozen 0.5% improvement threshold, have paired intervals
entirely below zero and have no supported slice or related-market regression above
2%. The bootstrap uses 2,000 paired calendar-week draws with seed 20260928.
These are development screens across 34 family/market comparisons. The later
qualification stage still requires frozen finalists and its Holm multiplicity
correction; it has not been run.

## Chronology and scope

- The [execution contract](phase4-step2-backtest-execution-2026-09-29.md) was captured
  before scoring, alongside the v2 candidate expansion and unchanged frozen v1 rules.
- The four 2022 calendar quarters supply forward tuning forecasts. Strength fits
  use only qualified earlier results available before each quarter boundary,
  within the current and preceding provider seasons.
- One parameter setting per family/market was selected on 2022 NLL and saved in
  `selection.json`, checksum-locked before 2023 scoring. All 34 families had a
  valid setting; the three lineup settings were explicitly unavailable.
- The 2023 comparison uses those fixed choices. Quarterly team-state refits can
  incorporate already available results; hyperparameters remain fixed.
- 2023 is already-inspected development data, not an untouched confirmation test.
  No 2024, calibration/qualification, final-system or prospective-reserve labels
  were accessed. No historical quotes, ROI or CLV were constructed.
- All 13 competitions contribute to pooled results. In 2023, only EPL, LaLiga,
  SerieA, Bundesliga and Ligue1 meet the 200-fixture/20-week league reporting
  minimum. Championship has enough fixtures but only 19 observed ISO weeks.
  Smaller leagues/European competitions remain insufficient for standalone claims.
- The early-season 0–7-match slice also falls below the 20-week support requirement.
  A pooled pass does not certify all leagues or early-season behavior.
- Targets and profiles remain `assumed_final` historical reconstructions with the
  inherited same-day exclusion and kickoff+3h label delay. Lineups, referee,
  standings/regime and other uncertified contexts remain common omissions.

| Period | Unique eligible fixtures (goals) | Corners | SoT | Scored forecast records |
|---|---:|---:|---:|---:|
| 2022 tuning | 3,645 | 3,612 | 3,587 | 397,947 |
| 2023 comparison | 3,095 | 3,028 | 3,037 | 113,140 |

The 511,087 records include repeated candidate/market forecasts. They are not
511,087 independent matches. Every comparison retains identical eligible fixture
membership; all inherited exclusions are saved in `coverage.json`. In particular,
355 Q3 and 554 Q4 2023 fixture records lack regulation-period evidence. These
gaps reduce coverage, and the resulting cohort must not be called representative
of every fixture across all leagues/seasons.

## What the gains mean

- **Goals:** ridge-10 opponent-adjusted attack/defence estimates improve the pooled
  joint distribution. MAE improves from 1.32158 to 1.31048 and RMSE from 1.66106
  to 1.64845. Winner NLL improves from 1.00610 to 0.99229; BTTS log loss from
  0.69277 to 0.68847. The scored policy includes 339/3,095 explicit control
  fallbacks for unknown/sparse teams. This is evidence for that candidate with
  fallback, not for forcing the fitted model on unsupported teams.
- **Corners and SoT:** selected dispersions are alpha=.025 and .01 respectively.
  Expected counts and numerical MAE/RMSE are unchanged. The gain comes from a
  better outcome distribution around those same means. Nominal 80% interval
  coverage moves from 89.83% to 85.73% for corners and 91.41% to 85.35% for SoT;
  intervals narrow. Nominal 95% coverage moves from 98.32% to 96.96%, and from
  98.78% to 96.11%. Discrete intervals can over-cover; this is not proof of
  complete calibration, which still needs Step 3.
- **Gradual prior:** corners improve by .261%, with a negative paired interval,
  but fail the unchanged .5% materiality requirement. It is not promoted by
  relaxing the threshold. Other markets do not clear the threshold either.
- **xG:** no xG candidate clears the full screen. The attacking fallback changes
  only 81/3,095 forecasts; its tiny pooled change cannot settle usefulness for
  sparse-xG situations. Defensive xG changes 1,982 forecasts but still yields
  only a .014% pooled NLL reduction with an interval crossing zero.
- **Other weights/distributions:** modest or adverse changes remain recorded.
  Learned rho and NB goals do not establish sufficient added value here.

## All 2022-selected settings on the 2023 comparison

Negative change is favorable. The full machine-readable report also includes
reliability bins, calibration slope/intercept diagnostics, numerical scores, RPS,
intervals, categorical class counts, all seven total/nine handicap lines and
league/season/season-stage/forecast-stage/missingness slices. Diagnostic calibration
fits are never applied to forecasts.

| Claim | Selected setting | NLL change | Development screen |
|---|---|---:|---|
| corners:calendar_recency | calendar_recency:30.0 | +0.011% | below improvement threshold; interval includes zero / adverse |
| corners:dispersion_corners | dispersion_corners:0.025 | -1.032% | Pass |
| corners:gradual_prior | gradual_prior:16.0 | -0.261% | below improvement threshold |
| corners:league_pool | league_pool:16.0 | -0.027% | below improvement threshold; interval includes zero / adverse |
| corners:own_weight | own_weight:0.5 | -0.053% | below improvement threshold; interval includes zero / adverse |
| corners:recent_weight | recent_weight:0.1 | -0.007% | below improvement threshold; interval includes zero / adverse |
| corners:recent_window | recent_window:12 | -0.000% | below improvement threshold; interval includes zero / adverse |
| corners:strength_corners | strength_corners:100.0 | +0.076% | below improvement threshold; interval includes zero / adverse |
| corners:venue_pool | venue_pool:16.0 | -0.034% | below improvement threshold; interval includes zero / adverse |
| corners:venue_weight | venue_weight:0.25 | -0.030% | below improvement threshold; interval includes zero / adverse |
| goals:calendar_recency | calendar_recency:30.0 | +0.005% | below improvement threshold; interval includes zero / adverse |
| goals:defensive_xg | defensive_xg:0.25 | -0.014% | below improvement threshold; interval includes zero / adverse |
| goals:goal_nb | goal_nb:0.025 | -0.055% | below improvement threshold; interval includes zero / adverse |
| goals:goal_rho | goal_rho:0.0 | -0.063% | below improvement threshold; interval includes zero / adverse |
| goals:gradual_prior | gradual_prior:16.0 | -0.014% | below improvement threshold; interval includes zero / adverse |
| goals:league_pool | league_pool:4.0 | +0.132% | below improvement threshold; interval includes zero / adverse |
| goals:own_weight | own_weight:0.5 | -0.034% | below improvement threshold; interval includes zero / adverse |
| goals:recent_weight | recent_weight:0.1 | -0.037% | below improvement threshold |
| goals:recent_window | recent_window:4 | +0.035% | below improvement threshold; interval includes zero / adverse |
| goals:strength_goals | strength_goals:10.0 | -0.719% | Pass |
| goals:venue_pool | venue_pool:4.0 | +0.058% | below improvement threshold; interval includes zero / adverse |
| goals:venue_weight | venue_weight:0.25 | +0.099% | below improvement threshold; interval includes zero / adverse |
| goals:xg_fallback | xg_fallback | -0.004% | below improvement threshold; interval includes zero / adverse |
| goals:xg_weight | xg_weight:0.25 | +0.012% | below improvement threshold; interval includes zero / adverse |
| sot:calendar_recency | calendar_recency:30.0 | +0.006% | below improvement threshold; interval includes zero / adverse |
| sot:dispersion_sot | dispersion_sot:0.01 | -1.411% | Pass |
| sot:gradual_prior | gradual_prior:16.0 | -0.063% | below improvement threshold; interval includes zero / adverse |
| sot:league_pool | league_pool:8.0 | -0.021% | below improvement threshold; interval includes zero / adverse |
| sot:own_weight | own_weight:0.5 | -0.010% | below improvement threshold; interval includes zero / adverse |
| sot:recent_weight | recent_weight:0.1 | -0.028% | below improvement threshold; interval includes zero / adverse |
| sot:recent_window | recent_window:8 | -0.011% | below improvement threshold; interval includes zero / adverse |
| sot:strength_sot | strength_sot:100.0 | +0.057% | below improvement threshold; interval includes zero / adverse |
| sot:venue_pool | venue_pool:16.0 | -0.007% | below improvement threshold; interval includes zero / adverse |
| sot:venue_weight | venue_weight:0.25 | -0.007% | below improvement threshold; interval includes zero / adverse |

## Validation and artifacts

- **497 tests passed:** 494 focused/regression checks plus three independent
  acceptance-gate tests. JUnit reports and test sources are preserved.
- All **6,740 eligible controls** match saved Step 1 means/variances exactly.
  The first eligible fixture in each fold additionally checks fast/full candidate
  means and applicable variances. The fast path uses the same frozen functions
  while omitting unpriced publication payloads and repeated diagnostic work.
- **573 fitted bundles** passed an independent audit of exact training IDs,
  target hashes, season/league/market, support, convergence and pre-cutoff label
  availability. 21 earlier fitting attempts were insufficient; none failed
  convergence. Fallbacks remain explicit and are included in the comparisons.
- Every forecast membership, target value and common control count was audited.
  There were zero invalid candidate outputs and zero forbidden-IO events.
- An independent check compared 20,004 scored control likelihoods directly with
  the preserved Step 1 PMFs. The maximum absolute NLL difference was below 1e-10
  (bounded finite-tail normalization); no observed target lay outside those PMFs.
- Inputs, source/dependency hashes, execution contract, quarter memberships,
  fitting bundles, compressed rate evidence/forecasts, selection lock, metrics,
  coverage, audits and this review are preserved in
  `Research/phase4-step2-backtest-2026-09-29/` under a COMPLETE checksum manifest.
- Production source, models, databases and publishing modes were not changed.
  Earlier baseline, Step 1 and candidate implementation artifacts remain intact.

Useful artifact files: `comparison.json`, `comparison-coverage.json`, `selection.json`,
`independent-audit.json`, `coverage.json`, `runs/*/forecasts.jsonl.gz`,
`runs/*/rate-evidence.jsonl.gz` and `validation/`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_backtest verify Research/phase4-step2-backtest-2026-09-29
```

## Remaining work

These three passing individual candidates are supported research finalists for
the next review. No combined configuration was selected or tested in this run.
Any combination must follow the bounded manifest and development-selection rules;
do not tune a new mixture against 2023 and describe that evidence as untouched.
Then proceed with Step 3 logistic comparison/coherent probability calibration,
Step 4 supported-scope review, and Step 5 later-period qualification. Cards and
richer lineup evidence retain their separate outstanding gates. No public
recommendation or deployment change is authorized by these development results.
