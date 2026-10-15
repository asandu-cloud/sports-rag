# Phase 3 Batch C: frozen development protocol — 26 September 2026

This protocol implements the owner's authorization to run Batch C. It is
written before Batch C comparisons. It supplements the agreed
[technical plan](phase3-technical-plan-2026-09-26.md), without changing
eligibility, reserved periods, production models or delivery.

## Inputs and chronology

Use the immutable phase3-dataset-2026-09-26 export and the completed
phase3-baselines-2026-09-26 development sidecar. Goals, corners and shots on
target remain the only numerical targets. Cards remain excluded.

The four six-month blocks are January–June 2022, July–December 2022,
January–June 2023 and July–December 2023, all UTC. The last two are outer
development assessment blocks. Recipe selection for each outer block uses
only earlier blocks, with at least two valid inner blocks. Labels must have
become available strictly before the receiving fit cutoff. This applies to
training, tuning scores and blend fitting, including matches close to midnight
at a block boundary.

For each family, and for the pooled family-selection procedure, the first
OOF block uses a predeclared seed: default Batch B family parameters,
all earlier training history and a 365-day half-life. The pooled seed is
Ridge alpha 50. The second OOF block selects from the first block only; this
is explicitly a one-block warmup, not an outer assessment. Subsequent OOF
blocks select using all earlier blocks. This sequence creates selection-honest
OOF predictions for the adaptive model-choice procedure.

A blend weight is the constrained least-squares optimum in [0, 1] on earlier
OOF statistical/ML prediction pairs. Equal forecasts select weight zero.
Every outer weight is frozen before that block's labels enter scoring.
The final development weight uses the chronological adaptive OOF sequence;
it is not learned by retrospectively choosing a recipe and relabeling all
its earlier tuning predictions as selection-honest OOF. That adaptive-to-final
refit distinction is reported and remains subject to later confirmation.

## Main search and comparisons

The existing 156 recipes per market are unchanged: 13 estimator settings
crossed with three training lookbacks and four half-lives. The maximum is
1,872 fold fits (156 x 3 markets x 4 folds), before exact duplicate fitting
inputs are reused. Deduplication requires identical training fixture IDs,
labels, weights, feature contract, estimator parameters and evaluation
membership; nominally different windows remain visible in reports.

Compare each family's earlier-selected recipe, the pooled earlier-selected
recipe and its earlier-learned blend with both baselines on identical fixtures.
The statistical comparator remains a reconstruction, not the original
published forecast. History comparisons hold the earlier-selected estimator
parameters fixed and compare all three lookbacks without decay and with that
selection's decay. Unsupported fits remain explicit.

For the final frozen proposal, choose the estimator recipe on all four
development blocks. Choose whether to forward its raw predictions, its
statistical blend, or retain the statistical baseline using the two outer
development blocks. Exact ties favor lower ML contribution, then the
predeclared estimator/window order. At most one recipe is retained per market.
Development results are not confirmation qualification or production approval.

## Bounded feature comparisons

There are seven variants: three-season profile, five-season profile, no
early-season prior, no Elo, no referee, no venue-specific fields and no recent
form. Each uses the main recipe selected from earlier inner blocks, without
retuning under the variant. Each is fitted and evaluated alongside a reference
refit on the same eligible training and evaluation intersection.

The allocation is at most 84 fits: seven variants x three markets x two outer
blocks x two paired methods. Exact duplicates may be reused. These are
conditional diagnostics; they do not enter the main recipe-selection grid
or change the final reference feature policy in this batch. A subsequent
feature-policy selection would require its own declared development procedure
before opening confirmation.

For three-/five-season profiles, replace the previous-season prior aggregate
with a pool of the preceding two/four provider seasons, weighting observed
fixtures equally within each statistic. Null observations remain absent,
never zero. Retain the current-season aggregate and existing eight-match
early-season blend strength. The prior stops contributing after eight
current-season matches, exactly as in the reference. Preserve reference recent
fields, Elo, referee, rest and other unrelated algorithms. Preserve the
reference prior-count predictor when prior weight is zero, so an inactive
older prior cannot alter a mature-season vector merely through a count field.
Expanded counts and contributing IDs remain recorded in audit metadata.
These are tests of a richer early-season prior, not continuous multi-season
averages. Report insufficient older history and unchanged vectors.

The no-prior variant rebuilds current-only profile statistics, including recent
fields affected by prior blending, and sets prior contribution/count to zero.
The other four variants remove their named numeric fields and corresponding
missingness indicators. Required league/rate interactions are retained.
Variant contracts and source snapshot identities are versioned separately;
the original export and pure production feature builder are immutable.

## Measurement and resources

RMSE is primary; MAE, bias, R² and common uncalibrated Poisson diagnostics
remain secondary. Report source, league, season, round, support, missingness
and the single reconstructed forecast stage. Season-stage bands use the smaller
of home/away primary current-season appearance counts: 0–7, 8–19, 20+.
Missingness bands use the fraction of missing numeric reference features:
0, (0, 10%], (10%, 25%], above 25%. Support bands use the smaller of the
two market-specific contributing counts. Report equal-league numerical means
separately from the equal-fixture primary score.

Use 2,000 fixed-seed paired UTC-calendar-week bootstrap samples, and the
predeclared fixed two-week sensitivity analysis anchored Monday 1970-01-05.
Resample a shared calendar block universe across competitions/markets.
Bonferroni-adjusted two-sided intervals cover six primary comparisons of the
frozen proposed strategy (three markets against two baselines); separate
raw-ML and blend intervals are secondary descriptive diagnostics.
League deterioration uses the agreed one-sided 95% bound and support gates.
These development intervals are descriptive after selection. Only Batch D
can apply confirmation acceptance. No historical betting ROI is calculated.

Run sequential fits with at most two native CPU threads. The software supports
safe checkpointing and explicit continuation in the same immutable experiment.
Use an eight-hour active execution budget per invocation, with a hard 2,500
new-fit budget across one experiment. No search expansion is automatic.
Record actual unique fits, duplicate reuse, skipped recipes, timings, dependency
lock and source hashes. Resume requires matching specification, input hashes,
source hashes and environment. Completed task artifacts cannot be overwritten;
partial attempts are retained and retried in a fresh attempt directory.
A competing process must fail an exclusive experiment lock.

Preserve all predictions, recipe-score provenance, exclusions, paired reports,
frozen final recipes, final isolated estimator bundles and completion hashes.
Final refits use only development labels available before 1 January 2024.
The confirmation, calibration and final-system target stores remain unopened.
No provider, embedding, production database or publishing access is permitted
during fitting.

## Numerical correctness amendment before accepting results

The first execution was stopped and preserved after prediction overflows
exposed a weighted-scaling defect. Identically-one training missingness
indicators acquired artificial positive variance through floating-point
summation. A later observed value then produced a spurious standardized
magnitude around 1.8e14. This is a preprocessing implementation error.

Preprocessing v2 detects exact constant columns using positive-weight training
rows only, assigns their exact observed mean and scale one, and leaves every
nonconstant column's variance unchanged. There is no tuned variance floor or
validation-dependent clipping. The corrected execution uses a new experiment
directory and reruns the original grid. No recipes, weighting options, metrics,
eligibility rules or held-out boundaries change. The earlier attempt is not
valid comparative evidence.
