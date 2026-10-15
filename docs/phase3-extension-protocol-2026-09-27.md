# Batch C extension: frozen protocol, 27 September 2026

The owner authorized the bounded extension described in the conversation.
This is new exploratory development informed by Batch C, not a new untouched
test. The [accepted Batch C results](phase3-forward-development-2026-09-26.md)
remain preserved. No threshold, reserved partition or production policy changes.

## Inputs and hypotheses

Use the existing immutable phase3-dataset-2026-09-26 reference features and
phase3-baselines-2026-09-26 historical statistical reconstructions. Reuse saved
initial-training baselines if complete; reconstruction is a separate immutable
preparation step only if coverage is missing. No provider calls or embeddings.
Targets remain regulation-time goals, corners and shots on target. Each home
and away label must be known and sum to the frozen total. All methods use the
same eligible training and evaluation fixtures. Missing/nonfinite/nonpositive
statistical baselines are explicitly excluded from the common correction cohort,
never imputed; report every lost fixture separately. Baselines remain historical
core reconstructions with their existing omissions and assumed-final limitation.

Freeze four architectures, with existing reference features and preprocessing v2:

| Architecture | Fixed configuration | Fits per market/fold |
|---|---|---:|
| residual_ridge | Ridge alpha 50 on signed actual-total minus statistical-total | 1 |
| offset_xgboost | Poisson XGBoost depth 3, lambda 10; log(statistical-total) as training and inference offset | 1 |
| team_poisson | Separate home/away Poisson regressions, alpha 1 | 2 |
| team_catboost | Separate home/away Poisson-loss CatBoost, depth 3, L2 10 | 2 |

Trees use 300 iterations and learning rate 0.03, fixed seed 42, at most two CPU
threads, and existing deterministic settings. Every fit uses a 1,460-day
lookback and 365-day recency half-life. Normalize training weights as before;
fit all preprocessing on those training rows only. No further parameter search.
Ridge corrections stay signed until added to the baseline; then project the
total to nonnegative values, retaining raw values and clipping counts. Offsets
require strictly positive baselines, without a tuned numerical floor. Team
means are added; do not assume independent distributions or double sample size.

## Chronology and selection

Use the existing four six-month development blocks in 2022 and 2023. The two
2023 blocks are the outer development readout, explicitly exploratory because
the extension follows inspection of Batch C. Model fitting, architecture choice
and blending admit labels only if available strictly before their cutoff.

First-block adaptive predictions use residual_ridge as a fixed seed. For the
second block choose using the first block only (warmup). For subsequent blocks
choose the lowest equal-fixture pooled RMSE over all earlier available blocks;
require at least two earlier blocks for an outer assessment. Exact ties use the
table order. Do not back-project Batch C's final selected recipes into early folds.
Fit convex blend weights using earlier predictions from this adaptive choice
sequence, with at least two prior blocks. Insufficient blend history uses zero
ML weight. No current-block labels enter its choice or blend weight.

After development, select one architecture per market on all four earlier
blocks. Select raw adaptive ML, its blend, or statistical-only using identical
outer fixtures; exact ties favor statistical, then blend, then raw ML. Refit
only a nonzero-ML proposal before 1 January 2024. Final blend weights use the
earlier adaptive OOF sequence; they are not a claim of independent validation
of the subsequently refitted fixed architecture. A statistical-only choice
requires no estimator refit. Final bundles remain unconfirmed and disabled.

## Budget, evidence and acceptance

The allocation is 48 architecture/fold evaluations and 72 component fits:
three markets times four folds times six components. At most six component
refits can follow. The hard cap is 78 actual experiment fit attempts, including
failed/interrupted attempts; synthetic software tests are separate. Persist one
checkpoint per estimator, so team models cannot hide two fits in one budget
unit. Resume requires unchanged code, data, protocol and environment. Never
silently expand the cap. Preserve partial work and report unsupported cohorts.

Compare each architecture and the adaptive raw/blended procedure against both
baselines and saved Batch C adaptive predictions on identical outer fixtures.
Report separate coverage exclusions, RMSE/MAE/bias/R2, sample sizes, weighted
ESS and league contributions, plus the existing league/season-stage/missingness
slices. Reuse 2,000-draw paired weekly bootstrap and two-week sensitivity.
The six frozen-proposal versus baseline intervals retain the existing adjustment;
other comparisons are descriptive, not a simultaneous multiple-search guarantee.
All intervals remain exploratory after development inspection.

Preserve the existing 1% improvement against both baselines and slice/precision
rules; this extension cannot qualify a model for production. Before any Batch D
confirmation, review and freeze at most one proposal per market. No automatic
confirmation run, promotion or repeat-until-pass search. Calibration, final-system
test, cards and player-prop work remain outside this operation.

Preserve manifests, source/dependency hashes, training memberships and labels,
OOF predictions, baselines/offset identities, component outputs, selection and
blend evidence, uncertainty, final bundles and exact serialization replay.
Tests cover signed corrections, offsets at fit/inference, preprocessing isolation,
fixture grouping, chronology, unchanged future cutoffs, deterministic membership,
one-estimator checkpoint accounting, replay and prohibited held-out/live access.
