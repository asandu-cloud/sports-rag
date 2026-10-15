# Phase 3 Batch C extension — 27 September 2026

The authorized bounded extension is implemented and completed locally in
`Index/prediction_experiments/phase3-extension-2026-09-27/`.
All 72 development component fits and six final component refits succeeded,
using the full 78-fit allocation without failures or expansion.

The selected development procedure improved slightly over Batch C for goals
and corners and was slightly worse for shots on target. **No market reaches
the unchanged 1% point-improvement requirement against both baselines.**
These are exploratory development results, not confirmation qualification.
No candidate is promoted; production remains unchanged.

## Scope and input verification

The [frozen protocol](phase3-extension-protocol-2026-09-27.md) adds four fixed
architectures to the [accepted Batch C work](phase3-forward-development-2026-09-26.md):
signed-residual Ridge, statistical-offset Poisson XGBoost, separate home/away
Poisson regressions and separate home/away Poisson-loss CatBoost models.
All use the reference 126-feature schema, a 1,460-day lookback and 365-day
recency half-life. No extra hyperparameter search was performed.

The existing baseline sidecar covers all 46,231 eligible fixture-market rows
in initial training and development. Each has positive finite baseline means
and recorded home/away integer labels that sum to the total. Therefore no new
baseline reconstruction, API download or embedding was needed. There were
zero additional training or evaluation exclusions in the extension.

The dataset remains `phase3-dataset-2026-09-26`, identity
`4362324537087ea2654d27538614166b305647112fa4749e8222134acdf52de0`.
Historical inputs retain their explicitly labelled
assumed-final availability; they are not original forecast-time snapshots.
The statistical comparator is the same historical core reconstruction with
omitted live/player/context adjustments, not the full prediction shown to users.

## Execution and chronology

Four six-month blocks cover January 2022 through December 2023. The first two
supply earlier development evidence; the two 2023 blocks form the outer
readout. Every fit uses only labels available strictly before its fit cutoff.
Architecture selection and blend fitting also enforce that cutoff.
The extension was designed after inspecting Batch C, so even the outer
readout is explicitly exploratory, not a fresh independent assessment.

- 48 architecture/fold evaluations and 72 development estimator fits.
- Six final estimator refits; three isolated bundles with two team components each.
- 78 committed estimator tasks, zero failed fits, zero retries and no budget expansion.
- 80,016 architecture OOF rows and 20,004 adaptive OOF rows.
- 100,760 outer rows: 11 methods on the same 9,160 market-fixture observations.
- Approximately 65 seconds for fitting and reporting, excluding input loading.
- All final component and combined-strategy serialization replays matched exactly.

First-block adaptive predictions use the fixed residual-Ridge seed. The
second block selects from the first only; later selections use earlier blocks
only. Blend warmup uses zero ML contribution, then fits a convex weight to the
earlier adaptive OOF sequence. Ridge corrections remain signed until added to
the baseline. XGBoost receives the log baseline during both fitting and
inference. Home and away means are summed without assuming independence.
Sample sizes, weighting and uncertainty remain based on unique fixtures.

## Outer development results

Lower RMSE is better. The extension columns describe its earlier-selected
adaptive blend, not the eventual single architecture refitted after development.

| Market | Fixtures | League-average RMSE | Statistical RMSE | Batch C blend RMSE | Extension blend RMSE | Gain vs average | Gain vs statistical | Gain vs Batch C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Goals | 3,095 | 1.65695 | 1.66276 | 1.64511 | 1.64203 | 0.9004% | 1.2471% | 0.1872% |
| Corners | 3,028 | 3.42125 | 3.42911 | 3.39878 | 3.39043 | 0.9010% | 1.1281% | 0.2457% |
| Shots on target | 3,037 | 3.12048 | 3.09903 | 3.07994 | 3.08789 | 1.0443% | 0.3593% | −0.2582% |

Goals and corners remain below 1% against the league average; shots on target
remains below 1% against the statistical reconstruction. No threshold was
rounded down or changed. No betting ROI, CLV or profitability was calculated.

All fixed architectures remain visible:

| Method | Goals RMSE | Corners RMSE | SoT RMSE |
|---|---:|---:|---:|
| Signed-residual Ridge | 1.66322 | 3.40124 | 3.08318 |
| Statistical-offset XGBoost | 1.64966 | 3.40651 | 3.08948 |
| Separate home/away Poisson | 1.64277 | 3.39278 | 3.08110 |
| Separate home/away CatBoost | 1.64868 | 3.39957 | 3.08889 |
| Adaptive selected raw ML | 1.64277 | 3.39278 | 3.08948 |
| Adaptive selected blend | 1.64203 | 3.39043 | 3.08789 |

For both outer blocks the selector chose team Poisson for goals and corners,
and offset XGBoost for SoT. The later all-development selector chose team
Poisson for all three markets. In particular, the SoT final architecture
differs from the one used in its outer adaptive predictions; those predictions
must not be presented as an independent test of the final team-Poisson bundle.

MAE, bias, R2, equal-league summaries, fold/slice metrics and common
uncalibrated Poisson diagnostics are saved in `reports-0001/metrics.json`.
The numerical primary criterion remains RMSE. Distribution fitting and
calibration have not been advanced into this extension.

## Uncertainty and supported scope

Paired weekly bootstrap uses 2,000 draws with the existing 99.1667% intervals
for the six main market/baseline comparisons. Relative RMSE improvement,
expressed in percent:

| Market | Against league average | Against statistical reconstruction | Against Batch C blend, descriptive |
|---|---:|---:|---:|
| Goals | +0.10 to +1.70 | +0.48 to +2.02 | −0.22 to +0.57 |
| Corners | +0.06 to +1.58 | +0.57 to +1.76 | −0.02 to +0.55 |
| SoT | +0.18 to +1.85 | +0.09 to +0.66 | −0.54 to −0.01 |

The small goals/corners gains over Batch C have intervals spanning zero.
The two-week SoT comparison against Batch C also spans zero (−0.57 to +0.05%).
Weekly and two-week main comparisons against both baselines have positive
intervals, but the point-improvement rule still fails for at least one baseline
per market. These exploratory intervals neither correct for all repeated
development decisions nor establish confirmation or production qualification.

Each market has 41 nonempty weeks and 237 match dates. The same coverage and
league limitations from Batch C remain: the original five domestic leagues
have enough support for development slice intervals, while other competitions
have insufficient fixture/date/week support under the existing rules. The
original five leagues' one-sided deterioration bounds stay below 5% in the
weekly comparisons against both baselines. This is not league qualification.
The newer-league period-evidence gaps and historical card-label issues remain.

## Frozen proposals and artifacts

All final proposals use two separate Poisson regressions with alpha 1,
the fixed 1,460-day/365-day settings and a statistical blend:

| Market | Final ML weight | Unique training fixtures | Weighted ESS |
|---|---:|---:|---:|
| Goals | 83.9723% | 14,177 | 9,220.11 |
| Corners | 76.8505% | 14,010 | 9,135.37 |
| SoT | 58.4056% | 13,767 | 9,022.79 |

The final cutoff is 1 January 2024. Counts are unique fixtures, not doubled
home/away rows. The weights were learned from earlier adaptive OOF predictions;
the final fixed refit still needs independent evaluation. Bundles are
`reports-0001/candidate-goals`, `candidate-corners` and `candidate-sot`, all
labelled `UNCONFIRMED_BATCH_C_EXTENSION` with publication and promotion disabled.

Preserved experiment directories under `Index/prediction_experiments/`:

- `phase3-extension-preflight-2026-09-27/`: baseline/target coverage, active-model
  hashes, production dependencies and accepted Batch C integrity.
- `phase3-extension-source-2026-09-27/`: 21 execution source hashes, protocol,
  environment and an archived copy of the fingerprinted source.
- `phase3-extension-2026-09-27/`: immutable component checkpoints, training
  labels/weights/baselines, OOF predictions, reports and final bundles.
- `phase3-extension-validation-2026-09-27/`: post-run integrity, preservation
  and completed-resume checks.
- `phase3-extension-independent-review-2026-09-27/`: separate final audit.

## Verification and remaining work

**609 tests passed:** 480 benchmark tests (including 173 new extension tests)
and 129 relevant existing regressions. Before actual fitting, review identified
and fixed deterministic OOF ordering and the final strategy's label-availability
guard. Their regression tests passed before the source was frozen.

The preserved baseline's six artifacts and all 187 saved canonical-to-Match-Read
compiler replays passed again. This does not close full historical numerical
replay. Actual completed-run resume performed no new fits or invocations.
The active model artifacts, frozen dataset/baselines and accepted Batch C
artifacts are preserved. No production dependency or delivery-mode change was made.

The [independent final audit](../Index/prediction_experiments/phase3-extension-independent-review-2026-09-27/review.json)
passed: all 655 experiment artifacts, 78 component tasks, training target/weight
contracts, 80,016 architecture OOF rows, 20,004 adaptive OOF rows and 100,760
outer rows were checked. It recomputed choices, blends, final strategies and
overall numerical metrics. Detailed slice, bootstrap and Poisson diagnostics
were checksummed rather than independently recomputed; model replay was
verified from the execution evidence, not repeated by the audit.

The experiment budget is exhausted as planned; no further search is queued.
Review the frozen proposals before deciding whether to open Batch D once.
Confirmation, calibration and final-system periods remain unscored. Phase 4
can improve the statistical engine's probability layer even if no ML mean
predictor qualifies. It has not been started. The separate daily-refresh
incomplete marker remains present and was not altered by this work.
