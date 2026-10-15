# Phase 3 Batch C — forward development and review, 26 September 2026

Batch C is implemented and completed locally on the frozen historical export.
The accepted run is **phase3-forward-development-v2-2026-09-26**. It completed
all declared model comparisons, chronological blends, history comparisons and
seven paired feature diagnostics. **No market reaches the agreed 1% point
improvement against both baselines on the outer development cohort.** These
are development results; confirmation qualification has not been assessed.

Production models, delivery modes and the original feature export are
preserved. Confirmation, calibration and final-system targets remain unopened
for model evaluation. No provider retrieval, embeddings or publication was
performed by this work. Cards and player props remain outside this batch.

## Scope and preserved experiments

The [frozen protocol](phase3-forward-development-protocol-2026-09-26.md)
supplements the [technical plan](phase3-technical-plan-2026-09-26.md).

| Artifact | Purpose |
|---|---|
| `Index/prediction_experiments/phase3-dataset-2026-09-26/` | Unchanged Batch A input, eligibility and reserved partitions. |
| `Index/prediction_experiments/phase3-baselines-2026-09-26/` | Unchanged development league-average/statistical reconstructions. |
| `Index/prediction_experiments/phase3-feature-variants-2026-09-26/` | 58,944 rebuilt development rows for three profile policies; 469,883,542 bytes. |
| `Index/prediction_experiments/phase3-forward-development-2026-09-26/` | Interrupted, invalidated first attempt; preserved without top-level completion. |
| `Index/prediction_experiments/phase3-numerical-diagnostic-2026-09-26/` | Exact evidence for the constant-column arithmetic defect. |
| `Index/prediction_experiments/phase3-forward-v2-preflight-2026-09-26/` | Corrected source hashes, protocol and fingerprinted-source archive. |
| `Index/prediction_experiments/phase3-forward-development-v2-2026-09-26/` | Accepted completed Batch C experiment and isolated bundles. |
| `Index/prediction_experiments/phase3-forward-validation-2026-09-26/` | Post-run integrity, preservation and completed-resume checks. |
| `Index/prediction_experiments/phase3-forward-independent-review-2026-09-26/` | Independent checks of scores, chronology, blend weights, cohorts and final contracts. |

The accepted experiment contains `reports-0001/`: recipe scores, chronological
choices, adaptive OOF predictions, outer predictions, point/distribution
metrics, uncertainty, history comparisons, feature predictions/coverage,
frozen recipes and review. Every completed task preserves its contract,
resolved estimator/preprocessing metadata, predictions and checksums.
Final bundles are linked by `reports-0001/frozen-recipes.json`.

## Execution and numerical correction

The first run exposed a preprocessing defect: identically-one training
missingness indicators received a tiny artificial positive variance from
weighted floating-point summation. In the diagnosed corners fold, both xG
missingness flags were constant across 12,435 training fixtures, but their
scale became approximately 5.55e-15. A later observed xG value produced
a spurious transformed magnitude of 1.80e14 and Poisson prediction overflow.

The owned research process was stopped with SIGINT. Its artifacts were not
deleted or reused as valid comparison evidence. Preprocessing v2 detects
exact constant columns among positive-weight training rows, sets their exact
observed mean and unit scale, and leaves nonconstant variances unchanged.
The fix uses no validation-dependent clipping or tuned variance floor.
The grid, metrics, targets, weighting choices and reserved periods did not change.
The diagnosed transform's maximum magnitude became 12.8552 after correction.

The corrected run completed:

- 1,872 main recipe/fold evaluations: 156 recipes × 3 markets × 4 folds.
- All main evaluations complete, with zero failed or unsupported recipes.
- 42 feature comparisons, each paired with a reference fit on identical
  training/evaluation intersections: at most 84 logical fits.
- Three final development refits with exact estimator and strategy replay.
- 1,299 actual unique fits; 660 identical task reuses.
- 20,004 adaptive OOF prediction rows and 91,600 outer prediction rows.
- About 620 seconds for fitting and reporting, excluding feature preparation.

The four blocks span January 2022 through December 2023. Only the last two
are outer development assessment blocks. First-block OOF predictions use
fixed defaults, second-block choices use only the first block, and subsequent
choices use all available earlier blocks. Tuning and blend labels must be
available strictly before the receiving cutoff. The final refit cutoff is
1 January 2024. The future test periods did not enter any selection.

## Outer development results

Each method is scored on the same fixtures within its market. Lower RMSE is
better. The selected procedure uses recipes and blend weights chosen before
each outer block. These numbers are **not** out-of-sample scores of the final
single CatBoost bundle, which is fitted only after development is complete.

| Market | Fixtures | League/venue average RMSE | Statistical reconstruction RMSE | Selected blend RMSE | Improvement vs average | Improvement vs statistical |
|---|---:|---:|---:|---:|---:|---:|
| Goals | 3,095 | 1.65695 | 1.66276 | 1.64511 | 0.71% | 1.06% |
| Corners | 3,028 | 3.42125 | 3.42911 | 3.39878 | 0.66% | 0.88% |
| Shots on target | 3,037 | 3.12048 | 3.09903 | 3.07994 | 1.30% | 0.62% |

None reaches 1% against both baselines. The statistical comparator is the
frozen historical core reconstruction with documented omissions, not an exact
replay of recommendations originally shown to users. No historical betting
ROI, CLV or profitability claim is made.

Paired weekly bootstrap uses 2,000 draws and 99.1667% intervals for the six
primary comparisons. Relative RMSE improvement intervals, in percentage points:

| Market | Against average | Against statistical reconstruction |
|---|---:|---:|
| Goals | −0.20 to +1.60 | +0.35 to +1.82 |
| Corners | −0.15 to +1.30 | +0.33 to +1.53 |
| Shots on target | +0.46 to +2.05 | +0.27 to +1.01 |

These intervals are descriptive after development selection; they do not
replace an untouched confirmation result. Fixed two-week sensitivity outputs
are saved beside weekly results. Raw-ML and blend comparisons are secondary
diagnostics. The six-comparison adjustment is not a simultaneous guarantee
over every reported method, feature diagnostic or slice.

All agreed families remain visible. Each family below uses its own
earlier-selected settings for each outer block:

| Method | Goals RMSE | Corners RMSE | SoT RMSE |
|---|---:|---:|---:|
| Ridge | 1.65175 | 3.39722 | 3.07644 |
| Poisson | 1.64683 | 3.40034 | 3.07765 |
| LightGBM | 1.64731 | 3.39722 | 3.08959 |
| XGBoost | 1.64659 | 3.39614 | 3.08603 |
| CatBoost | 1.64524 | 3.40313 | 3.08769 |
| Adaptive selected ML | 1.64706 | 3.40688 | 3.09143 |
| Adaptive selected blend | 1.64511 | 3.39878 | 3.07994 |

Some individual family procedures outperform the adaptive selector on these
outer blocks. The frozen recipe follows the predeclared all-development
selection rule; it is not replaced by whichever row looks best here.
MAE, bias, R², equal-league summaries and common uncalibrated Poisson log loss,
RPS and interval coverage are preserved in `metrics.json`.

## Frozen proposals, not promoted models

All three proposed strategies are statistical/ML blends. The final recipe
selector chose CatBoost after pooling the four development blocks. This does
not establish universal CatBoost superiority or satisfy the confirmation gate.

| Market | CatBoost depth | Training lookback | Half-life | Final ML weight | Final training fixtures | Weighted ESS |
|---|---:|---:|---:|---:|---:|---:|
| Goals | 4 | 1,460 days | 180 days | 77.6162% | 14,177 | 4,694.26 |
| Corners | 3 | 1,460 days | 730 days | 68.6826% | 14,010 | 12,334.47 |
| Shots on target | 4 | 1,460 days | None | 52.2532% | 13,767 | 13,767.00 |

Training uses the frozen 300 iterations/0.03 learning rate and the other
declared family parameters. Final weights are learned from the earlier
adaptive OOF sequence. The later single-recipe refit differs from that adaptive
sequence, so confirmation is necessary. The bundles have publication and
promotion disabled and no supported league is declared qualified.

`predict_candidate()` is the estimator-only replay interface.
`predict_strategy()` additionally checks equivalent fixture/snapshot/market
identities and the saved statistical baseline contract, then applies the
frozen blend weight. Both interfaces reproduced their expected outputs exactly.
Live ML blending and public recommendations were not changed.

## History and feature comparisons

Four-year and all-history fits were identical at every development cutoff
for the main search: there was not enough earlier eligible history to separate
them. The shorter-window tie rule selected 1,460 days for the final recipe.
At the later final-refit cutoff that window excludes 2019 fixtures, so the
final model is explicitly a four-year proposal. These development results
cannot establish whether more than four years improves predictions.

The paired history table separately compares two/four/all history without
decay and with the earlier-selected decay, keeping estimator parameters fixed.
Four/all no-decay means performed modestly better than two-year no-decay
means for goals and SoT; corners had the opposite direction. No strong general
history-length conclusion or new window policy is adopted.

Feature comparisons use the earlier-selected reference recipe without
retuning under each variant. Positive percentages mean lower RMSE than its
paired reference; negative values mean worse RMSE:

| Variant | Goals | Corners | SoT |
|---|---:|---:|---:|
| Three-season early-season prior | +0.151% | +0.145% | −0.197% |
| Five-season early-season prior | +0.034% | +0.198% | −0.041% |
| No early-season prior | −0.102% | −0.042% | −0.061% |
| Remove Elo | +0.006% | +0.017% | −0.280% |
| Remove referee fields | −0.049% | +0.165% | +0.078% |
| Remove venue-specific fields | −0.111% | +0.164% | −0.151% |
| Remove recent form | +0.067% | +0.094% | −0.037% |

The longer profiles change the pooled earlier-season prior only while the
existing eight-match early-season prior remains active; they are not continuous
three-/five-season averages. Their weekly uncertainty intervals include zero.
These conditional diagnostics do not establish a new feature policy.

No-prior comparisons lose 244 goals, 242 corners and 230 SoT outer fixtures
from the common cohort, mostly at the start of the season. Their matched
comparison sizes are 2,851, 2,786 and 2,807 respectively. Coverage loss is
separate from the small accuracy differences. All other variants preserve
the original outer cohort. The reference two-season feature policy is retained.

## Competition support and limitations

Outer development covers 41 nonempty UTC weeks and 237 match dates overall.
First-outer-block counts are 2,113 for each market; the second has 982 goals,
915 corners and 924 SoT fixtures. Coverage is uneven and these are explicitly
assumed-final historical snapshots, not authentic forecast-time observations.

| Competition | Goals | Corners | SoT |
|---|---:|---:|---:|
| EPL | 397 | 388 | 394 |
| LaLiga | 395 | 386 | 387 |
| SerieA | 395 | 383 | 390 |
| Bundesliga | 304 | 297 | 302 |
| Ligue1 | 363 | 356 | 355 |
| Championship | 254 | 254 | 254 |
| SuperLig | 170 | 170 | 170 |
| Eredivisie | 180 | 180 | 180 |
| PrimeiraLiga | 181 | 181 | 181 |
| BelgianProLeague | 168 | 168 | 168 |
| UCL | 112 | 108 | 108 |
| UEL | 83 | 77 | 78 |
| UECL | 93 | 80 | 70 |

The original five domestic league slices have sufficient development support
and their one-sided deterioration bounds are below 5%. This is not league
qualification. Championship lacks 20 nonempty weeks for the declared interval
precision rule; the other newer domestic and European slices have insufficient
fixture support, and some also lack dates/weeks. They remain descriptive.
The five newer domestic leagues have historical coverage here through earlier
provider seasons; their more recent period-evidence gap is not repaired.

All exports contain one reconstructed pre-kickoff stage. There is no
preliminary/confirmed-lineup/amendment validation from these experiments.
Fitted count dispersion, calibration and coherent derived outcome markets
remain Phase 4; prospective shadow and publication acceptance remain later work.

## Verification and next gate

- **436 tests passed:** 307 benchmark tests and 129 relevant existing regressions.
- Every accepted experiment artifact and the entire frozen dataset passed
  checksum verification. Frozen source hashes match the corrected preflight.
- The preserved baseline's six artifacts and all 187 Match Read compiler
  replay cases passed again. This remains canonical-result-to-compiler replay,
  not complete historical numerical prediction replay.
- All ten active model files and the production dependency inventory are unchanged.
- Actual completed-run `--resume` performed no new fits or invocations and
  preserved completed artifact hashes.
- An [independent audit](../Index/prediction_experiments/phase3-forward-independent-review-2026-09-26/review.json)
  verified all 6,522 artifacts, recomputed the 1,872 recipe scores and 12
  earlier-OOF blend weights, and checked equal comparison cohorts and final
  training cutoffs. It found no new correctness blocker. Bootstrap intervals
  and Poisson diagnostics were checksummed, not independently recomputed.
- The separate daily-refresh incomplete marker remains present; Batch C
  did not retry the refresh or change its safety gate.

Batch C is complete. The decision remains provisional: modest development
gains do not meet the agreed point threshold against both baselines, and no
production improvement or profitability is established. The next planned
operation is a reviewed, one-time Batch D confirmation of the frozen proposals
if the owner elects to proceed. It has not been launched. The later calibration
and final-system periods stay reserved.
