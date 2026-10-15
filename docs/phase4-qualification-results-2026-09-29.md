# Phase 4 January–June 2025 qualification — results

The qualification run is complete. Fitted **corner and shots-on-target count
distributions pass** the frozen offline requirements on the predeclared pooled
scope. The opponent-strength goal change and the separate corner Over 9.5 binary
calibrator do not qualify. Production remains unchanged.

Artifact: `Research/phase4-qualification-2026-09-29/`.
Protocol: [prospective qualification amendment](phase4-qualification-amendment-2026-09-29.md),
version `phase4-qualification.v2`. This supersedes the earlier outcome-free Q2
block recorded in the [finalist review](phase4-finalist-review-results-2026-09-29.md).

## What was frozen and evaluated

The owner approved January–June 2025 after metadata showed April–June alone could
not meet the existing sample/week minimum. No 2025 outcomes were decoded before
the new protocol, sources and methods were fixed. Requirements were not lowered.
The five-league goal-strength fit used eligible provider-season 2023/2024 results
available before 1 January 2025. Calendar 2024's reuse was explicitly declared;
it was already consumed Phase 3 evidence, not a fresh test or new tuning period.

| Strength fit | Earlier fixtures | Provider season of fitted bundle |
|---|---:|---:|
| EPL | 539 | 2024 |
| LaLiga | 533 | 2024 |
| SerieA | 528 | 2024 |
| Bundesliga | 421 | 2024 |
| Ligue1 | 418 | 2024 |

All five fits use the existing ridge-10 method and equal fixture weights; pooled
training contains 2,439 fixtures. Effective sample sizes equal fixture counts.
The corner sigmoid remains the original 2022 OOF fit, with no 2025 calibration
refitting. Goal temperature remains identity, rho remains −.10, and count
dispersion is fixed at alpha .025 for corners and .01 for SoT.

The method lock was written before the qualification opening:
`19c416c1fa40bd15ead8af9c67e93f0ebe613bbb03ff18e5578f9adf127e32d3`.
Its files include the exact earlier finalist recipe, source snapshots, data hashes,
dependency versions, fitting history, fitted bundles and frozen calibrator.
Permanent `HISTORY_OPENED.json`, `QUALIFICATION_OPENED.json` and
`SCORING_STARTED.json` record each access stage. Forecasts were locked before
aggregate scoring. There was one scheduled evaluation and no subsequent tuning.

Earlier results within January–June update later deterministic team profiles,
recent statistics and variances, using the same strict availability rules for
both systems. They never refit strengths, calibrators or hyperparameters. A
forecast uses no result from its own fixture, a later fixture, or the same UTC
date. The frozen control's current/prior-season calculations are executed.

## Primary qualification decisions

The fixed active scope is EPL, LaLiga, SerieA, Bundesliga and Ligue1, with both
teams having at least eight current-season matches. All four claims have 20
observed ISO weeks and exceed the 500-fixture/10-week pooled support gate.
Negative loss change is better. The last row compares calibration with its raw
dispersion candidate; the other rows compare with the unchanged control.

| Claim | Fixtures | Reference NLL | Candidate NLL | Relative change | Paired 95% difference interval | Holm-adjusted p | Decision |
|---|---:|---:|---:|---:|---|---:|---|
| Goals, joint score distribution | 934 | 2.905925 | 2.914599 | **+0.299%** | [−.016951, +.032304] | .7641 | Retain control |
| Corners, count distribution | 888 | 2.618127 | 2.589121 | **−1.108%** | [−.036989, −.021080] | .0020 | Qualifies offline |
| SoT, count distribution | 888 | 2.518381 | 2.464856 | **−2.125%** | [−.066625, −.041061] | .0020 | Qualifies offline |
| Corner Over 9.5 sigmoid | 888 | .695377 | .691228 | **−.597%** | [−.012292, +.003063] | .2979 | Inconclusive; retain raw probabilities |

Both count candidates exceed the .5% improvement requirement, have wholly
negative paired intervals, pass Holm correction across all four fixed claims,
and have no supported slice/related-market loss regression above 2%. The goal
candidate is slightly worse on average and has no demonstrated improvement.
The binary calibrator's point estimate improves, but its uncertainty crosses
zero. Its comparison with the frozen control is only −.137%, also inconclusive.

These are probability-distribution results. **Corner and SoT predicted means,
MAE and RMSE do not change.** No betting profitability, win-rate or CLV claim
follows from these results. Authentic historical quote coverage is unavailable.

## Coverage, league contributions and limits

The full reconstruction includes 2,073 fixtures. Existing eligibility rules leave
1,125 goals, 1,065 corners and 1,071 SoT fixtures, across 21 observed weeks.
No originally eligible fixture was silently dropped. Cards remain outside this
qualified target contract. Per-fixture exclusion reasons are preserved in
`reproduction/eligibility.jsonl` and the source requests.

| League | Goals | Corners | SoT | Goal loss change | Corner loss change | SoT loss change |
|---|---:|---:|---:|---:|---:|---:|
| EPL | 192 | 179 | 179 | +.104% | −.693% | −2.325% |
| LaLiga | 199 | 193 | 190 | +.694% | −1.689% | −1.664% |
| SerieA | 202 | 190 | 195 | +.225% | −.839% | −2.354% |
| Bundesliga | 171 | 164 | 162 | −.107% | −1.250% | −1.912% |
| Ligue1 | 170 | 162 | 162 | +.581% | −1.068% | −2.393% |

All league slices meet 100 fixtures/10 weeks; they span 18–20 weeks. These are
slice checks inside the fixed pooled claim, not five separately Holm-qualified
improvement claims. Corner confidence intervals cross zero in EPL and Ligue1.
Do not promise an independently demonstrated improvement in every league.

Outside the active scope, 191 goal, 177 corner and 183 SoT forecasts retain exact
control probabilities. No active-scope goal fixture required an unknown/sparse
team fallback. On the complete served cohorts, changes are +.245% goals,
−.920% corners and −1.760% SoT. Those reports include unchanged fallbacks; they
do not establish new support for other competitions.

All active qualification fixtures have the reported `xg_both_venues` input state.
There is no independent qualification evidence for partial/missing-xG scope.
The 8–15-match stage has just 17 goal, 15 corner and 16 SoT fixtures across two
weeks; its standalone support is insufficient. The 16+ stage has 917/873/872
fixtures across 20 weeks. No scope was searched or redefined after scoring.
The pooled pass must not be presented as separate certification of every sparse
stage, missingness state or league-by-stage intersection. Unsupported conditions
still require the control fallback and later evidence before public activation.

Only reconstructed immediately-before-kickoff forecasts and provider season 2024
are represented in this qualification. Earlier forecast stages and confirmed-XI
amendments remain unverified here.

## Secondary results and unresolved limitations

| Diagnostic | Control | Candidate | Interpretation |
|---|---:|---:|---|
| Corners 80% interval coverage | 89.64% | 86.94% | Less overcoverage; still above nominal |
| Corners 95% interval coverage | 99.10% | 97.52% | Less overcoverage |
| SoT 80% interval coverage | 93.47% | 88.51% | Less overcoverage; still above nominal |
| SoT 95% interval coverage | 99.77% | 97.86% | Less overcoverage |
| Corners RPS | 1.847381 | 1.831556 | Improves |
| SoT RPS | 1.635740 | 1.607674 | Improves |
| Raw corners Over 9.5 log loss | .692175 | .695377 | **.463% worse**, within the fixed 2% limit |
| Raw SoT Over 8.5 log loss | .667844 | .667807 | Essentially unchanged |

Passing the whole count-distribution comparison does not establish better
probabilities at every betting line. The raw corner Over 9.5 Brier score worsens
from .249484 to .251004; SoT Over 8.5 Brier moves slightly from .237766 to .237831.
The saved ten-bin reliability tables and diagnostic intercepts/slopes disclose
these differences. Those diagnostic fits are never applied to predictions.

Goal MAE decreases from 1.291141 to 1.284835, but RMSE increases from 1.590172 to
1.595145 and positive bias increases from .051254 to .093343. This does not
override the failed primary joint-probability result. Goal-derived markets retain
shared-score coherence; no independent binary probabilities replace them.

The historical availability assumption is still `assumed_final`. Lineups, dated
standings/regime/knockout context, qualified referee/card profiles and engineered
profile fields have the same disclosed omissions as the full-control experiment.
This evaluates a documented reconstruction, not exact past user recommendations.

## Verification and artifacts

- **551 tests passed**, including the later-date synthetic archived-engine run,
  chronological boundaries, future/same-day leakage, missing-versus-zero behavior,
  default development guards, fixed-calibrator identity and Holm family handling.
- **2,073 exact serialized control replays**, followed by 1,125 additional
  numerical control parity checks during prediction. No prohibited worker IO.
- All **five independent strength refits** reproduced their frozen bundles.
- **230,737 historical source references** passed availability/date checks.
- **6,522 probability distributions** passed normalization checks; inactive scope
  retained exact control probabilities. Asian vectors and line monotonicity pass.
- A separate post-scoring audit verifies original eligibility membership, saved
  PMFs against primary scores and derived goal/Asian probabilities, and sigmoid
  application: **6,522 primary scores, 2,250 coherent goal distributions, 45,654
  total settlement vectors and 888 sigmoid applications** replayed with maximum
  numerical difference 2.67e-15. All **460** preserved runtime/test/model/config/
  dependency files in the working checkout still match the baseline fingerprints.
  Its source and output are preserved with the experiment.

`comparison.json` contains the complete scores, reliability, slice and uncertainty
reports. `decision.json` records all four corrected decisions. `METHOD_LOCK.json`,
`INPUT_LOCK.json`, `CONTROL_LOCK.json`, `PREDICTION_LOCK.json` and
`EVALUATION_LOCK.json` preserve each stage. `audit.json`, `independent-audit.json`,
`test-results.xml` and `COMPLETE.json` preserve final verification.

## What follows

For the next research stage, retain the existing coherent goal engine and
identity calibration. Carry the qualified corner/SoT dispersions forward only
with their documented scope and fallback requirements. Do not apply the rejected
goal-strength change or the inconclusive standalone sigmoid to production.

The planned later-period qualification for goals/corners/SoT is complete; it is
not completion of card qualification, richer lineup research, final-system
testing, prospective shadow acceptance or production promotion. Phase 5 can use
the retained/qualified components to develop the selection policy on permitted
development evidence. January–June 2025 is now consumed qualification evidence;
it cannot become another tuning set while still being called an untouched test.
July 2025–June 2026 remains closed for the completed probability and selection
system. Databases, active model artifacts and public delivery modes are unchanged.
