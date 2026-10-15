# Phase 4 frozen evaluation protocol v1

Owner-authorized freeze, 28 September 2026. Identifier: `phase4-evaluation.v1`.
This fixes the control, candidate budget, access rules and decision criteria before
full-engine probability comparisons. No reserve is opened by the freeze itself.
Changes require a new version and an inspection-history entry; never overwrite
this protocol inside the preserved baseline or describe reused evidence as fresh.

## Control and provenance

Control identifier: `phase4-repaired-statistical-control.v1`. Preserve the actual
working-tree source, configuration and installed model artifacts, including dirty
changes; a Git commit alone is insufficient. The repaired canonical engine retains
its eight-match prior cutoff, current variance rules, fixed Dixon–Coles rho −0.10,
quality/price/settlement guards and existing lineup/referee/context behavior.
Assert the effective ML weights are zero for goals/corners/cards/SoT using the
preserved metadata and real gating function. Preserve artifacts without training
or loading serialized estimators for inference. Do not install the gradual prior
candidate, a fitted dispersion or a new calibrator during capture.

Freeze source, dependencies, weights, supported-market contract, source/data hashes
and permitted development inputs. Preserve `baseline-2026-09-15-pre-model-fixes`
unchanged. The new control is an offline research reference, not a rollback of
production, an accuracy endorsement, or certification of worker-loaded code.

Verify deterministic projection → variance → distribution → canonical decision
replay from saved synthetic profile/recent-variance/context/quote inputs. Execute
unmodified numerical code; replace only external input lookups. Preserve the
boundary declaration and golden outputs. Cover seven canonical markets, quarter
lines, unknown variance, sparse profiles, missing inputs and context adjustments.
Synthetic card replay tests preserved behavior only: it does not qualify card
semantics. Existing mathematical/settlement tests remain required independently
of replay equality. Replay neither reconstructs all historical live inputs nor
proves that full historical probability comparison has already been implemented.

Historical evaluation must separately construct immutable pre-match input
snapshots from the permitted evidence. Reconstructed lineup/standings/referee
inputs cannot be invented. Label omitted contexts explicitly, use identical
snapshots for control and candidates, and verify parity with the preserved code.
Any unrepresentable context reduces scope or is a disclosed common omission.
The control's probability distributions, including its variance behavior, must be
used; the earlier Poisson pilot is not a substitute control.

## Data access and chronology (UTC)

| Period | Fixed use |
|---|---|
| Before 2022 | Earlier fitting/history; targets available before each cutoff |
| Calendar 2022 | Forward development fitting/tuning and out-of-fold calibration-method development |
| Calendar 2023 | Later development comparison; already inspected, never called untouched |
| Calendar 2024 | Consumed Phase 3 confirmation; excluded from v1 fitting/selection, diagnostic reuse only with a new explicit entry |
| 1 Jan–31 Mar 2025 | Dedicated calibration fitting, only after methods/features/scope are frozen |
| 1 Apr–30 Jun 2025 | Later Phase 4 qualification; single scheduled evaluation of frozen methods |
| 1 Jul 2025–30 Jun 2026 | Untouched completed probability + selection-system final test, held for Phase 5 |
| From 1 Jul 2026 | Existing prospective reserve; no retrospective tuning against these outcomes |

Split at 00:00 UTC on each boundary. A fixture and all markets, versions and
forecast stages belong to one partition. Simultaneous kickoffs never split.
Historical label availability is explicitly `assumed_final`, kickoff + three
hours, strictly before a fitting cutoff; actual timestamped availability takes
precedence where verified. Later corrections cannot silently rewrite snapshots.
Training-only imputation/scaling/shrinkage/recency estimation; no fitting on a
validation period. Historical eight-season storage does not automatically enlarge
the control's profile window.

Use quarterly forward folds in 2022 (fit earlier, score the next quarter).
Pool genuine out-of-fold forecasts to choose calibrator family/regularization;
then fit a development calibrator on those 2022 forecasts for the 2023 comparison.
Base probability-parameter candidates use 2022 fitting evidence only. Mean-model
learning for the logistic comparator also uses forward folds, with final training
through 2022 for the 2023 comparison. After development selection, methods and
feature definitions are locked; base fits may use data through 2023, excluding
2024, before January–March 2025 calibration. No validation-label refit is allowed
before April–June qualification. Log every reserve access, artifact identity,
cohort and purpose before reading outcomes. An adverse qualification result is a
failure/inconclusive result, not permission to retune on that period.

Do not open 2025 outcomes merely to decide boundaries or sample thresholds.
Outcome-free coverage metadata may establish insufficient support beforehand.
Sparse Jan–Mar/Apr–Jun cohorts must remain insufficient; do not borrow final-test
rows or move dates after scoring. No fitting or reserve access occurs in this freeze.

## Markets and bounded candidate families

Goals, corners and SoT enter only on qualified targets/features. BTTS, home/draw/
away and handicap outcomes require the same certified regulation scores as goals.
Cards remain part of the preserved engine but outside qualified fitting until
`phase4-participation-card-target.v1` source/period/bias requirements pass and
statistical team/referee/player rates estimate the same weighted quantity. A new
card dataset needs an additive qualification record, prospective protocol version
and sufficient earlier fitting data; 895 provisional 2023 totals are not enough.

Candidate budget; keep the unchanged control in every comparison:

1. Existing gradual-prior strength 16 as one separately identified challenger.
   No further strength search in v1. Evaluate it separately before combination;
   do not infer qualification for unsupported leagues/cards from pooled results.
2. Corners/SoT negative-binomial dispersion, `variance = mu + alpha * mu^2`,
   alpha in `[0, .01, .025, .05, .1, .2, .4, .8, 1.6]`; alpha zero is Poisson.
   Compare with the actual control variance, not only with Poisson.
3. Coherent goals: Poisson score marginals with rho in
   `[-.15, -.10, -.05, 0, .05]`; and independent negative-binomial score marginals
   with shared alpha from the same grid and rho zero. Do not combine the Poisson
   Dixon–Coles correction with NB marginals without a separately specified valid
   joint model. Reject invalid parameters on the fixed common evaluation cohort;
   never clip negative probabilities or silently drop difficult fixtures. Bound
   omitted tail mass at 1e-10 and derive all goal markets from the same joint PMF.
4. Regularized logistic regression as an isolated binary comparator: BTTS and
   totals half-lines 2.5 goals, 9.5 corners, 8.5 SoT. Inputs: control probability
   logit, control mean, home-away goal mean difference for BTTS/goals, and league
   indicators; no new broad feature search. L2 `C` in `[.01, .1, 1, 10]`, seed
   20260928, training-only transforms. It is a comparator, not permission to
   replace shared goal probabilities with mutually inconsistent binary outputs.
5. Calibration: identity and monotone regularized logistic scaling of probability
   logits on held-out/OOF forecasts, L2 `C` as above. For the joint goal candidate,
   compare identity with scalar temperature `[.75, .9, 1, 1.1, 1.25, 1.5]` applied
   to joint-score probabilities and renormalized; derive all goal markets again.
   Independent binary calibrators may be reported diagnostically but cannot be
   promoted across related goal markets without a coherent construction. A
   calibrated count PMF/Asian profile requires a distribution-level method;
   calibrating one half-line does not qualify its other lines or quarter lines.

Ties within 1e-8 of mean primary score prefer unchanged control, then fewer fitted
parameters, then smaller alpha/absolute rho adjustment and stronger regularization.
No league-specific parameters in v1. Use pooled estimates with explicit support
and neutral fallback; later hierarchical or league-specific fitting is a new
predeclared experiment. No automatic combination of individually winning changes:
a combined candidate must be selected on development before reserve access.

## Scores, support, uncertainty and acceptance

Primary scores: joint regulation-score NLL for coherent goal systems; count NLL
for corners/SoT; binary log loss for standalone BTTS/half-line comparators and
binary calibrator diagnostics. Primary comparisons use identical fixtures per
market/candidate pair; report absolute coverage and common-cohort attrition.
Failure to produce a valid distribution is a failure/coverage loss, never a
quietly reduced test set. Each family is a separate claim; do not pick the best
looking score or pool correlated derived markets as independent successes.

Secondary: Brier score, ranked probability score where applicable, equal-width
10-bin reliability with bin counts, calibration intercept/slope, 80%/95% discrete
interval coverage and width, numerical MAE/bias/RMSE, and multiclass scores for
winner and five-way Asian outcomes. Predeclared total diagnostic grids:
goals 1.5/2.5/3.5 and 2/2.25/2.75/3; corners 8.5/9.5/10.5 and 9/9.25/9.75/10;
SoT 7.5/8.5/9.5 and 8/8.25/8.75/9; handicap −1/−.75/−.5/−.25/0/.25/.5/.75/1.
These are probability diagnostics, not evidence that a quote was offered.

Development minimums: 500 fitting fixtures/26 ISO weeks; 200 evaluation fixtures/
20 weeks; league-specific conclusions require 200 fixtures/20 weeks. For the
fixed three-month reserve stages, each requires at least 500 fixtures/10 ISO
weeks pooled; league-specific claims require 100 fixtures/10 weeks in evaluation.
Binary fits require at least 50 of each outcome; categorical outcomes require
reported class counts and no unsupported class-specific claims. These minima
are set before reserve inspection and do not themselves establish precision.

Use 2,000 paired calendar-week cluster-bootstrap draws (seed 20260928), retaining
all fixtures/markets/stages together. Report fixture and week counts, effective
sample sizes for any weights, and league contributions. No individual-match
independence assumption for uncertainty. Show league, season, season stage
(0–7/8–15/16+ current matches), forecast stage and missingness slices. Forecast
stages absent historically remain unavailable, not inferred.

A candidate qualifies for the next stage only with at least 0.5% relative primary
score improvement, the 95% paired difference interval entirely below zero, and
no supported league/season-stage/missingness slice worsening by more than 2%.
Related goal markets must not worsen primary proper scores by more than 2% on
supported cohorts. Report secondary regressions even when primary checks pass.
The single April–June evaluation tests only development-frozen finalists; apply
Holm correction at .05 across the primary market-family claims submitted there,
using paired week-resampling inference. Unsupported/inconclusive scope retains
the control. Thresholds are unchanged after disappointing results. The earlier
corners pilot's weaker criterion is historical development evidence only; it does
not satisfy this new full-engine gate automatically.

## Contracts, artifacts and release

Store raw/calibrated probabilities, distribution/calibration versions, support,
uncertainty, forecast stage, target semantics, complete Asian outcome vectors and
all input/source hashes. Validate normalization, nonnegativity, complementarity,
line monotonicity, shared-score coherence and settlement-engine parity. Keep event
probability, expected betting value and confidence separate. Price comparison
requires exact period/market/line/bookmaker/rules and authentic pre-kickoff quotes.
No ROI/CLV from synthetic/current/reconstructed prices; publish no test selections.

Per experiment preserve protocol version, immutable dataset/split definitions,
code/config/dependency/seed identities, fit parameters, OOF/evaluation predictions,
coverage/exclusions, metrics/uncertainty, isolated bundle and reviewed conclusions.
Calibrators never relax quality, freshness or no-bet gates. Card repair, corners
scope review, richer lineup research, full player props and live operational
acceptance retain their documented scopes. Richer lineup work follows this fixed
baseline; any new candidate joins via a predeclared protocol amendment.
Promotion/public-scope changes require separate approval, shadow evidence and a
tested rollback. No live acceptance or final-system qualification is claimed by
freezing source or passing synthetic replay.
