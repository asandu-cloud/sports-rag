**Spix Phase 3 — technical plan for review**

Plan dated 26 September 2026, updated with the owner's decisions. This is the
technical specification; implementation evidence belongs in the separate
dated completion reports. The initial planning task performed only repository
inspection and planning. The owner has selected two separate testing periods:
one for Phase 3 confirmation and a later one for the completed probability and
selection system. The owner has also selected XGBoost and CatBoost for the
Phase 3 numerical comparisons, alongside Ridge, Poisson regression and
LightGBM, and logistic regression for the Phase 4 probability comparisons.
These model additions are approved planning scope. The owner answered the
16 rule questions: A except 6B, then approved refined 9A: pooled league-aware
models, equal fixture weights before recency, and separate league qualification.
Section 11 records the binding selections. The owner subsequently authorized
Batch A (establish the usable dataset) on 26 September. This authorization
does not start model training, dependency installation or production changes.

Subsequent authorization: the owner asked to build the Batch B benchmark
runner on 26 September. [Batch B's completion report](phase3-benchmark-runner-2026-09-26.md)
records the isolated dependency environment, frozen baseline recipes,
52 predeclared league/rate interactions for the linear models, and bounded
development smoke. This authorization does not start the full Batch C search,
open reserved tests or permit production activation.

The owner subsequently authorized Batch C: full forward development, model
comparisons, blends and ablations. The
[frozen Batch C protocol](phase3-forward-development-protocol-2026-09-26.md)
resolves the bounded longer-profile allocation, exact prior arithmetic, OOF
warmup/provenance, primary comparisons and resumable resource budget before
running the development search. Confirmation and final-system evaluation
remain separate; this authorization does not permit production promotion.
[Batch C's completion report](phase3-forward-development-2026-09-26.md)
records the corrected execution, numerical defect and preserved invalidated
attempt, frozen proposals, modest development gains and remaining confirmation gate.

On 27 September the owner authorized the bounded correction/team-target extension.
Its [frozen protocol](phase3-extension-protocol-2026-09-27.md) and
[completion report](phase3-extension-2026-09-27.md) preserve the separate
exploratory experiment: 72 development component fits plus six final refits,
609 passing tests and an independent artifact/chronology audit. The extension
retains the numerical acceptance rules; no market reaches 1% against both
baselines. The reference export and all reserved periods remain preserved.
No Batch D confirmation, Phase 4 fitting or production promotion is authorized
by that completed extension run.

The owner subsequently authorized Batch D on 27 September. Its
[frozen protocol](phase3-confirmation-protocol-2026-09-27.md) and
[completion report](phase3-confirmation-2026-09-27.md) record the one-time 2024
confirmation, a pre-scoring European round-label validation repair, 656 passing
tests and a separate numerical/artifact audit. All three candidates fail at
least one agreed qualification gate. Phase 3's bounded numerical programme is
complete with no promotion. Confirmation is consumed; the later calibration,
final-system-test and prospective reserves remain unopened for evaluation.
Phase 4 has not started.

**1. Objective and scope**

Determine whether corrected machine learning improves expected match totals
over simple historical averages and a reconstruction of Spix's statistical
projection system. Determine whether using older training examples helps.
A valid result is that the statistical baseline remains preferable, or that
the available evidence is inconclusive.

Initial targets: regulation plus stoppage-time total goals, corners and shots
on target. Keep all 13 competitions in coverage reporting, with qualification
by market and competition. Exclude cards as a training target until label bias
and weighted participation semantics are resolved. This phase does not train
BTTS, winner or handicap probabilities from a total alone; their coherent
score-distribution treatment belongs to Phase 4. Regularized logistic
regression will be included there as a direct outcome-probability comparator,
with its own classification targets and probability metrics.

As verified locally for this plan, canonical FT history contains 31,588
fixtures: 31,588 known goals targets, 29,491 corners and 28,957 SoT. Twelve
competitions have represented provider seasons 2019–2026; UECL has 2021–2026.
The 2026/27 season is incomplete. These counts precede Phase 3 eligibility.
The earlier 5,083-fixture export is preserved and will not be extended in place.
See [the import report](eight-season-history-import-2026-09-26.md).

Production remains on its existing models, profile policy and delivery modes.
Training stays separate from publication workers. This plan neither needs a
cloud migration nor requests another bulk download. Historical odds, player
data, manager/formation features, card repairs and live acceptance remain
separate work.

**2. Data source, evidence and immutable export**

Create a consistent local research snapshot of `Index/platform.db` using the
existing backup mechanism. Read relevant raw archives and verified import
plans locally. Write a new immutable experiment directory; do not mutate
canonical history or the preserved baseline. Freeze source/configuration
hashes, registry, dependencies, exclusions and the snapshot identifier.

Extend the canonical adapter rather than replace the feature pipeline. It
currently drops `round`, `elapsed` and `stats_json` source evidence. Preserve
these in an evidence record separate from the numerical vector, including:

- Provider fixture/team/competition/season IDs, kickoff and source revision.
- Original and normalized round; status and regulation-period verification.
- Played-result concerns and explicit review/quarantine status.
- Source class: raw provider archive, verified local reconstruction or
  unresolved legacy evidence; archive IDs, hashes and preparation digest.
- Explicit-zero/null/omitted-statistic policy and response-presence flag.
- Actual observation timestamps and separately labelled assumed availability.
- Feature cutoff, forecast stage, label definition/version and reason codes.

The canonical fixture table does not store the full regulation/extra-time/
penalty score breakdown. Join matching frozen raw payloads or verified import
evidence to establish it. Do not turn an `FT` flag into stronger proof than the
source provides. Unresolved period evidence is excluded from the primary
comparison and counted separately; missing evidence does not trigger API calls.

Retrospective examples use the existing explicit `assumed_final` convention:
stored final statistics are assumed available three hours after kickoff.
Actual fetch timestamps remain unchanged. This cannot establish original
provider vintages or remove later corrections. Prospective `observed` snapshots
remain a separate evidence cohort. Preserve regulation home/away targets as
well as totals for later work.

**3. Versioned eligibility contract**

Add a research-only contract, provisionally `phase3-eligibility.v1`. Its
decisions depend on evidence and predeclared support rules, not model errors.
Maintain separate decisions for an evaluated target and for each historical
input observation. An unsafe fixture must not re-enter through profiles, Elo
or recent form after being excluded as a target.

| Check | Proposed primary rule |
|---|---|
| Identity | Exact provider IDs, consistent competition/season, distinct teams, unambiguous kickoff and statistic pairing. |
| Period and participation | Verified compatible regulation result; exclude AET/PEN aggregates, conflicting scores, confirmed or suspected administrative/unplayed results and unresolved evidence. |
| Target | Both team counts known, finite, nonnegative integers for the particular market; no imputation of labels. |
| Source quality | Preserve source classes and accepted zero policies; ambiguous legacy zero remains unknown. |
| Team support | Both required market production rates available, with at least five distinct valid earlier market observations per team's contributing profile. Count actual observations with nonzero contribution, not nominal season totals. |
| Optional inputs | Missing xG, venue, referee or other optional fields may remain missing; do not reject every partially incomplete row. |
| Availability | Features precede their forecast cutoff; each training label precedes its fitting cutoff under the declared observed/assumed policy. |
| Cards | Explicit excluded status regardless of a non-null legacy card label. |

Five earlier observations is a proposed conservative starting rule, not an
existing rule or proof of adequate precision. Preserve lower-support examples
in a separate descriptive cohort; do not relax the rule after disappointing
results. Report support bands 0, 1–4, 5–9 and 10+.

Proposed round scope: include recognized domestic league phases, including
integral championship/split groups; exclude separate promotion/relegation and
European-qualification playoffs from the primary target cohort. Include
European main-competition group/league and knockout rounds; report qualifying
rounds separately. Build explicit competition/era mappings for the provider's
round labels; unknown mappings remain review-only. Verified qualifiers and
special domestic rounds may supply earlier context when that is explicitly
recorded by the history policy. Do not silently transfer a team's observations
between divisions.

Produce both complete and market-specific reason codes, such as
`missing_away_corners`, `period_conflict`, `unknown_round`,
`insufficient_home_history`, `label_available_after_fit` and
`cards_target_not_qualified`. Report exclusions by competition, season, round,
source and market, including overlapping reasons and one deterministic
primary reason. No database record is deleted.

**4. Feature policy and preprocessing**

Reuse the shared 148-feature builder. Retain its current-plus-immediately-
previous provider-season policy as the reference, and add the owner-selected
three- and five-season team-profile experiments (answer 6B), wherever supported.
Each variant needs a separate versioned feature policy and reproducible
pre-match inputs. Training-example lookback and per-fixture profile history
are separate choices. No variant may use a future match or season.

Before implementing the longer windows, specify how older seasons contribute
to each team profile, their weighting and the early-season prior rule. Merely
widening the snapshot filter is insufficient: the current aggregator reads
only current/prior seasons and drops the prior contribution after eight
current-season matches. Keep unrelated feature choices controlled so the
comparison identifies the profile-history effect; document any necessary
contract changes. Do not import the retired feature/training paths. The
production two-season profile policy remains unchanged.

Proposed primary feature mask removes card-derived predictors, card Elo and
their missingness indicators, including referee-card rate. This prevents the
known card-data defect from becoming an unexplained predictor in other
markets. Other optional fields retain explicit missingness. Save the exact
selected names in the candidate contract; retain the complete source schema.
Referee assignments reconstructed from historical final records carry an
availability caveat and require a no-referee ablation.

Fit every imputer, scaler and any optional feature selection inside the
permitted training slice. Proposed default: weighted median imputation,
weighted mean/variance scaling for linear models, and the same imputation
contract without scaling for trees. Weights are the declared fitting weights.
All-missing training columns have a recorded zero placeholder after
imputation, with their missingness preserved; validation values never choose
the placeholder. Inference reuses the saved transformations.

This extends the current unweighted median/scaler candidate runner. Pipelines
support keeping learned transformations inside the training boundary; see
[scikit-learn's leakage guidance](https://scikit-learn.org/1.8/common_pitfalls.html).
Weighted-median tie handling and zero-variance behavior must be deterministic
and tested.

**5. Chronology and the two testing periods**

Use one global UTC chronology, not separate league dates or random shuffling.
Every market, forecast stage and revision of a provider fixture receives the
same partition. Simultaneous kickoffs stay together. Conflicting revised
kickoff times that could move a fixture across a boundary require resolution
or quarantine.

Proposed ordering:

`initial training → forward development/tuning → Phase 3 confirmation → calibration reserve → final system test`

| Region | Use |
|---|---|
| Initial training | Fit early models; seek at least two elapsed years of supported history. |
| Forward development | Repeated earlier-train/later-validation folds; all model, history-window, recency and blend decisions occur here. |
| Phase 3 confirmation | One later period, opened once after freezing the numerical candidate and its criteria. This is Phase 3's independent test. |
| Calibration reserve | Later predictions for Phase 4 probability calibration; not used to select Phase 3 models. |
| Final system test | Still later, reserved for the frozen probability and selection system in Phases 4–5. Phase 3 does not report its model metrics. |

Starting sizing proposal: twelve months for Phase 3 confirmation, six months for
calibration, and twelve months for the final system test. Prefer whole football
cycles where feasible. The incomplete 2026/27 season alone is not an adequate
representative holdout. Dates will be chosen from the eligibility coverage
report, using counts, availability and calendar structure only, before any
candidate scores are generated. Freeze exact interval endpoints and group
membership in the split manifest. These duration targets may yield an
insufficient-scope report; they do not justify forcing thin slices or moving
boundaries after scores are known.

This necessarily leaves fewer than eight seasons available at early fitting
cutoffs. Report the actual earliest/latest training dates in every fold.
Four-year and all-history variants are identical until more than four years
of eligible fitting history exist; early folds cannot answer that comparison.
The two reserved tests reduce how much older history can be evaluated now.
If no sufficiently supported development blocks distinguish four years from
all available history, mark that question inconclusive rather than use either
test period to choose a longer-history recipe. Require at least two supported
later development blocks with genuinely different fitting memberships before
claiming evidence about that distinction. A six-month calibration reserve
also needs a season-stage coverage review before Phase 4 can qualify it.

Use custom fixture/calendar grouping rather than bare row-count
`TimeSeriesSplit`; its standard fold construction does not encode fixture
revisions or equal-kickoff groups. See the
[documented splitter assumptions](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

At each fit, require label availability strictly before the cutoff. In
retrospective mode this is the declared kickoff-plus-three-hours assumption;
in observed mode also require the actual observation timestamp. Model and
preprocessing parameters stay fixed within each evaluation block. Earlier
completed matches in that block may update later pre-match features under
the frozen temporal policy; this simulates updating match information without
retraining. Log that event order, including same-day cutoffs.

The ordinary training CLI must not access held-out target stores. A separate
confirmation operation requires frozen candidate/specification hashes and
records a permanent inspection event. Replay is allowed; selecting a new
candidate after viewing confirmation results consumes that period as
development evidence. The later system test has a different identifier and
remains unopened. This is an auditable access boundary, not a claim that
public historical football results can be made secret.

Phase 4 may refit a fixed base-model recipe using earlier data and create new
held-out calibration predictions. It cannot fit the base model on calibration
labels and reuse their old predictions as though they were held out. Any
future numerical/probability/selection changes need the later system test.
The calibration reserve must also leave room for honest Phase 5 development:
calibrators fit on an earlier part and policy choices use later predictions,
or a chronological cross-fitting scheme supplies those predictions. Do not
tune policy on calibrator-fitted outcomes and call that independent evidence.
Freeze that internal division before inspecting its calibrated scores; if
support or authentic odds are insufficient, policy development needs later
prospective evidence. The final system test stays reserved in either case.

**6. Baselines and models**

Compare on identical eligible fixtures within each market; preserve wider
coverage and prediction failures as separate results. A method cannot improve
its headline score by silently declining difficult fixtures.

| Method | Specification |
|---|---|
| League/venue average | Earlier home and away scoring/count means for that competition, with a predeclared pooled fallback for sparse history. |
| Statistical reconstruction | Frozen current projection arithmetic applied to reconstructed historical profiles, with ML contribution zero. |
| Ridge | Regularized linear expected-total model using the shared feature contract. |
| Poisson regression | Regularized log-link count-mean model using the same eligible examples. |
| LightGBM | Bounded Poisson-objective boosted-tree candidate. |
| XGBoost | Bounded boosted-tree candidate using `count:poisson` and the same numerical targets/features. |
| CatBoost | Bounded boosted-tree candidate using the Poisson objective and the same numerical targets/features. |
| Statistical/ML blend | Convex combination of reconstructed statistical and ML means; its ML weight can be zero. |

All five ML families are in the agreed benchmark roster. Apply the same
training-support gates and comparable bounded search budgets to the three
boosted-tree families. Initially CatBoost uses the shared numerical feature
contract and competition indicators; native categorical encodings or new
team-ID predictors would be a separately versioned experiment, not a silent
change to its inputs.

The statistical comparator must not call live `projected_total_*` wrappers:
they can read current Chroma profiles, current standings and provider context.
Reuse or extract arithmetic from `projected_goals`, `projected_corners`,
`projected_sot` and their numerical helpers, passing frozen configuration.
Construct production-shaped profiles from snapshot history. Reproduce the
core's six-match exponentially weighted recent calculations and xG/venue
rules; the candidate's five-match averages are not interchangeable with them.

Name the comparator `statistical-core-reconstruction.v1`. Record all omitted
or unavailable lineup, standings/stakes, knockout and other contextual
adjustments. Preserve current/prior candidate season boundaries rather than
fall back to any older Chroma profile. This establishes comparison with a
documented statistical core, not the exact prediction previously published.
Beating it does not prove superiority over the complete live system.

The owner approved one pooled model per market with existing competition
indicators and equal fixture weights before recency weighting. For Ridge and
Poisson, a small predeclared set of regularized league-feature interactions
will allow selected relationships to vary by league; freeze the exact set in
Batch B before development scoring. Tree candidates may learn interactions
from competition identity. Every league still requires separate qualification.
Equal-league weighting and independent league models are not additions to
the agreed search. The last audited FT totals range from
1,632 for UCL to 3,989 for Championship, before exclusions/holdouts: no
individual league currently meets the selected 5,000-fixture tree-training
minimum. Sparse competitions retain explicit qualification limits.
Preserve raw Ridge outputs, but apply
a predeclared nonnegative projection for count predictions; never round
expected values to integers. Record clipping frequency. A tiny positive
floor used for logarithmic distribution scores is separate from that rule.

Local package metadata at the planning check shows scikit-learn 1.8.0 and
LightGBM 4.6.0; XGBoost and CatBoost are not installed. During approved Batch B
implementation, create an isolated research environment, select compatible
versions and freeze its dependency lock. Do not upgrade the runtime used by
the website or bots. Run an isolated native-runtime, fit/predict and
serialization smoke check for all three tree libraries. A failed dependency
check is a visible incomplete comparison, not permission to silently drop an
agreed model; other work may proceed while it is resolved.

Freeze CPU settings, seeds and library versions. Follow the
[LightGBM parameter contract](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[XGBoost objectives](https://xgboost.readthedocs.io/en/stable/parameter.html) and
[CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression).
Each adapter must return an expected count on the original scale. Test the
distinction between a raw log-score/margin and a transformed mean, including
saved-model inference, so no library is scored on the wrong scale or
transformed twice.

For Phase 4, include regularized
[logistic regression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
on predeclared binary outcomes such as BTTS and fixed half-line totals (for
example over 2.5 goals). These are direct probability comparators, separate
from both the numerical-total models and any logistic calibration transform.
Freeze outcome/line definitions and a bounded regularization grid before
fitting. Compare against distribution-derived probabilities on the same
chronological fixtures using log loss, Brier score and reliability analysis.
Apply the same leakage, availability and held-out calibration rules, and
verify consistency across related lines/markets before any later use. Do not
reduce pushes or Asian half outcomes to binary labels. This probability work
does not enter Phase 3's RMSE ranking or consume the final system test early.

**7. History windows, weighting and development search**

Compare trailing 730 days, trailing 1,460 days and all available earlier
eligible training examples. These are elapsed-year windows, not a promise of
two/four complete provider seasons. Pre-match feature history is separately
versioned for the reference and approved three-/five-season experiments.
Retain a no-decay control and the selected elapsed-time half-lives of 180,
365 and 730 days:

`weight = 2 ** (-age_in_days / half_life_in_days)`

Calculate age relative to that fit's cutoff, normalize weights to mean one
within its training slice, and apply them consistently to preprocessing and
estimator fitting in every fold and final fit. Score evaluation fixtures
equally; do not discount disappointing older test outcomes. Report raw counts,
unique fixtures, effective sample size `(sum w)^2 / sum(w^2)`, age distribution
and league contributions before/after weighting. Do not add league balancing
to the agreed equal-fixture-before-recency fitting policy. Report an equal-league
metric summary separately regardless of the training-weight decision.

Proposed bounded grid per market: Ridge alpha `{1, 10, 50, 100}` and Poisson
alpha `{0.1, 1, 10}`. Give each tree family two configurations, all with 300
boosting iterations and learning rate 0.03:

| Tree family | Proposed configuration pair |
|---|---|
| LightGBM | `(num_leaves, min_child_samples)` equal to `(7, 50)` or `(15, 100)`. |
| XGBoost | `max_depth` equal to 3 or 4; `tree_method=hist`, `objective=count:poisson`, `reg_lambda=10`. |
| CatBoost | `depth` equal to 3 or 4; `loss_function=Poisson`, `l2_leaf_reg=10`, `bootstrap_type=No`. |

These are bounded starting proposals, not a claim that the libraries have
identical effective complexity. Hold other settings fixed, use CPU, and
record all resolved defaults against the pinned research versions before
development. Cross each family with the twelve history/half-life combinations:
at most 48 Ridge, 36 Poisson and 24 recipes for each of LightGBM, XGBoost and
CatBoost, totaling 156 recipes per market before removing duplicates. Fold
refits and the declared ablations are additional runs, not additional search
recipes. No unbounded optimizer or post-confirmation grid expansion.

Answers 6B and 11A must both be honored: the owner selected additional
three-/five-season profiles and the documented 156-configuration main budget.
Do not silently cross all three profile policies with that entire grid,
which would create 468 configurations per market. Proposed reconciliation:
retain the 156-recipe reference search and run predeclared, bounded paired
profile-window experiments as additional feature comparisons, using settings
chosen only from earlier development data. Report their extra fits separately
and disclose that this is conditional comparison, not exhaustive retuning
under every profile window. Freeze the exact allocation and longer-window
semantics before training; any full-grid expansion requires a changed budget.

Use nested forward development where coverage permits: inner folds choose
recipes using only earlier data, and later outer blocks assess that complete
choice procedure. Proposed outer blocks are six calendar months, with at
least two earlier inner validation blocks. Never fit early-stopping rules or
preprocessing on an outer evaluation block. Fixed tree count avoids an extra
early-stopping holdout in this first grid. Too few valid folds produces an
insufficient-development-evidence result.

Generate chronological out-of-fold predictions for blending. Each such base
prediction must come from training and hyperparameter selection that used
only its own earlier data. Learn one
nonnegative convex weight per market from earlier OOF prediction pairs,
optimizing squared error; evaluate it on a later block. Base-model selection
must also be contained inside the earlier development data. For outer-fold
scores, do not learn the blend from that fold's outcomes. Freeze the winning
recipe and blend weight before Phase 3 confirmation. On exact development
ties choose the simpler model, with a lower ML blend weight as the blend
tie-breaker and the shorter training window for otherwise tied window
variants; stable recipe IDs break remaining ties. Do not choose a different
model using confirmation results.

A history-length comparison table must show paired scores on the same later
fixtures, both without decay and with the development-selected decay. This
separates the value of additional examples from the effect of their weights.

**8. Metrics, uncertainty and proposed qualification rules**

Primary numerical metric: **RMSE per market**, because these models estimate
expected counts. Also report MAE, signed bias (`prediction - actual`) and R²;
mark R² undefined on a constant-target slice. Do not pool raw errors from
goals, corners and SoT into one number.

Distribution diagnostics: score a declared common Poisson distribution built
from each positive predicted mean using count log loss and ranked probability
score, and report central 50%, 80% and 95% interval coverage/width. Apply the
same numerical floor and tail rule to every method, saving tail mass checks.
These are uncalibrated distribution diagnostics. Estimating negative-binomial
dispersion, score dependence and probability calibrators remains Phase 4.

Break results down by market, league, provider season, season stage, source
class, round, support and missingness. Proposed season-stage bands use the
team's prior current-season appearances: 0–7, 8–19 and 20+, with the aggregation
rule frozen. Historical exports have one reconstructed kickoff-time stage;
do not claim preliminary/confirmed-lineup or amendment validation without
authentic stage-specific snapshots.

Run predeclared development ablations for current/prior priors, Elo, referee,
venue and recent form. A prior-policy ablation must rebuild features and its
source contract, not just drop a summary column. Use a common eligible
intersection and also disclose changed coverage. These are developmental
diagnostics; freeze any chosen variant before confirmation. Cap this batch
at those five original groups, plus the specifically selected three-/five-
season profile comparisons with the budget accounting above. No unrestricted
feature search is implied.

Proposed uncertainty: paired calendar-week block bootstrap, 2,000 development
resamples and 20,000 confirmation resamples,
fixed seed 42. Keep each fixture, its versions/markets and simultaneous
matchday results together; sample common UTC weeks across competitions.
Predeclare a two-week-block sensitivity analysis. Repeated-team dependence
may extend beyond these blocks, so intervals remain an approximation.

| Proposed gate | Requirement |
|---|---|
| Pooled model fit | At least 2,000 unique training fixtures and weighted ESS of at least 1,000; each of LightGBM, XGBoost and CatBoost additionally needs 5,000 fixtures and ESS of at least 2,000. |
| Development validation | At least 500 eligible fixtures per reported pooled validation block; unsupported recipes/folds are visible. |
| Phase 3 confirmation | At least 1,000 unique fixtures and 20 nonempty calendar weeks per market. |
| Improvement | At least 1% relative RMSE reduction against both the statistical reconstruction and simple-average baseline on the fixed common cohort. |
| Uncertainty | The paired improvement interval must exclude zero. Use family-wise 95% coverage across the six predeclared comparisons (three markets times two baselines), conservatively using Bonferroni-adjusted intervals. |
| Supported league/market slice | At least 200 fixtures across 20 match dates; smaller slices remain descriptive. |
| Material slice regression | Relative to the statistical reconstruction on the same slice, the one-sided 95% bound on RMSE deterioration must stay below 5%; insufficient precision leaves that slice unqualified. |
| Reliability | No invalid output, silent prediction drop, leakage, contract mismatch or unexplained integrity failure. |

These are proposed practical standards, not thresholds derived from the
results or guarantees of statistical power. Freeze them before scoring. A
failed recipe or output is reported as failed, not quietly removed to shrink
the common test set. Primary targets/support define the comparison cohort
before inference; planned baseline fallbacks must cover it or the baseline
scope is explicitly insufficient.

A qualified pooled result does not qualify every league. Outcomes can be
qualified for named scope, no measurable improvement, harmful, or inconclusive.
No accuracy improvement implies profitability. Calculate betting returns only
for a separately identified cohort with authentic timestamped, executable and
equivalent historical quotes and settlement evidence; otherwise mark ROI/CLV
unavailable. Do not introduce synthetic odds or a new betting policy here.

**9. Implementation boundaries and artifacts**

Proposed files/modules, subject to existing repository conventions:

| Location | Planned responsibility |
|---|---|
| `Scripts/data_platform/features/model_dataset.py` | Preserve evidence and attach read-only eligibility inputs. |
| `Scripts/data_platform/features/market_eligibility.py` (new) | Versioned history/target checks, source classes, round mappings and reasons. |
| `Scripts/data_platform/features/chronological_splits.py` (new) | Fixture grouping, calendar boundaries, label cutoffs and split manifests. |
| `Scripts/rag_ingest/core/model_features.py` and `candidate_features.py` | Extend the existing contract only where required; retain shared train/inference behavior. |
| `Scripts/data_platform/features/benchmarks/` (new package) | Frozen statistical adapter, estimators, weighting, OOF blends, metrics and grouped uncertainty. |
| `Scripts/ops/prediction_features.py` | New immutable export/evidence/replay support. |
| `Scripts/ops/prediction_benchmark.py` (new) | Separate prepare, develop, freeze, confirm and report operations. Proposed interface, not a command available today. |
| `Scripts/tests/data_platform/` | Contract, temporal isolation, numerical parity and lifecycle tests. |

Do not retrofit Phase 3 into the retired `ml_edge.py` training commands. If
extracting shared arithmetic would touch production code, prefer an isolated
adapter with parity tests for this phase; any necessary behavior-preserving
refactor must be identified and regression-tested separately.

Every experiment preserves evidence and source manifests, feature schema and
mask, eligibility reasons, coverage, split and fold definitions, seed and
dependencies, weighted support, predictions from each development/confirmation
method, model metrics/intervals, ablations, and an isolated candidate bundle.
The bundle contains preprocessing, estimator, feature/eligibility/history/
baseline versions, learned blend weight, supported scope and training cutoff.
Include estimator family, exact package version and output-scale adapter.
It has publication and promotion disabled. Save completion checksums only
after successful verification; interrupted work remains incomplete.

The source exporter currently scans history repeatedly. Index/cache by
competition, season, team and cutoff only after proving parity with the pure
builder on a bounded subset and existing replay fixtures. Keep source history
once rather than duplicating it per model. Training runs sequentially with
bounded CPU threads (proposed maximum two); profile runtime before the full
grid and checkpoint completed immutable tasks. No scheduled retraining.

**10. Tests and delivery sequence**

Batch A — evidence, eligibility and splits. Deliver the new frozen dataset,
coverage/exclusion report, actual boundary dates, membership hashes and tests.
This batch does not fit models. Tests must establish:

- Target/future/simultaneous results cannot affect an earlier feature vector.
- Invalid period/participation evidence is excluded from target and history.
- Null, omitted and explicit zero remain distinguishable with source evidence.
- Late labels are excluded at every fit cutoff; observed and assumed modes
  cannot silently substitute for one another.
- Fixture revisions and same-kickoff groups cannot cross partitions.
- Reordering inputs or repeating execution preserves membership and hashes.
- New contracts reject stale schema/provenance or conflicting fixture evidence.
- No canonical DB, baseline, active model or publication write occurs.

Batch B — comparators and isolated estimator runner. Deliver baseline parity
tests, the isolated locked research environment, all five candidate adapters,
weighted preprocessing/models, inference contracts and a bounded smoke
experiment on development data. Test hand-computed weighted examples,
held-out perturbation invariance, all-missing columns, saved-transform reuse,
library output-scale and serialization parity, nonnegative/count-score
behavior, archive/hash validation, and hard failure
on provider/Chroma/current-model access. Failure leaves active artifacts intact.

Batch C — forward development and review. Run the declared grid, OOF blends,
history comparisons and five ablation groups. Report all coverage/failures,
resource usage and uncertainty; freeze at most one selected recipe per market
and the evaluation specification. Test nested fold boundaries, OOF prediction
provenance, model/weight selection without outer labels, and deterministic
resumption. No confirmation or final-system-test scores enter selection.

Batch D — Phase 3 confirmation and handoff. Run the frozen methods once on
the Phase 3 confirmation period, apply the predeclared gates, preserve an
inspection record and produce the written decision. Save the isolated bundle
and the separate calibration/final-system-test reservations for later phases.
Test that ordinary development cannot open either held-out target store and
that consumed confirmation cannot be presented as untouched for a new recipe.

At the end of each batch provide a concise reviewable report. Implementation
starts only after the owner approves this plan. Material changes to scope,
rules or gates after that approval must be surfaced before use; disappointing
results do not justify lowering standards. Promotion remains a later explicit
decision requiring shadow/live evidence and rollback, regardless of Phase 3.

**11. Recorded rule selections and remaining discussion**

The two-test-period approach and model roster were already selected: Ridge,
Poisson regression, LightGBM, XGBoost and CatBoost for Phase 3, and logistic
regression for Phase 4. The owner's subsequent questionnaire answers are:

| Question | Answer | Selected rule |
|---|---|---|
| 1 | A | Raw archives plus verified local reconstructions; explicitly assumed availability three hours after kickoff. |
| 2 | A | Reconstructed prediction immediately before kickoff; no claim of historical lineup/publication replay. |
| 3 | A | Five valid contributing earlier market observations per team. |
| 4 | A | Score recognized main rounds; verified qualifiers/special playoffs may supply declared historical context. |
| 5 | A | Exclude card-derived predictors; retain other referee information under the availability policy and compare without it. |
| 6 | B | Also compare three- and five-season team-profile windows where supported. |
| 7 | A | Target 12-month confirmation, six-month calibration and 12-month final-system-test periods, subject to eligible coverage. |
| 8 | A | Two/four/all training history, with no decay or 180/365/730-day half-lives. |
| 9 | Refined A | Pooled league-aware models; equal fixture weights before recency; limited predeclared league-feature interactions for linear/count models; separate league qualification. |
| 10 | A | Six-month forward evaluation blocks with earlier inner tuning folds. |
| 11 | A | The documented 156-configuration main search per market; feature comparisons/refits are additional, explicitly accounted runs. |
| 12 | A | RMSE primary; MAE, bias and R² supporting. |
| 13 | A | At least 1% improvement over both baselines, with the declared uncertainty requirements. |
| 14 | A | Adopt the draft training/ESS, validation, confirmation and league-slice minima in Section 8. |
| 15 | A | League-level one-sided 95% RMSE-deterioration bound below 5% against the statistical reconstruction. |
| 16 | A | Report card coverage/exclusions; detailed card repair remains separate. |

These selections supersede earlier wording that described the same choices as
proposals. Question 9 is agreed as refined A above. The longer-profile arithmetic
and bounded allocation are now recorded in the frozen Batch C protocol:
seven paired diagnostics, at most 84 additional logical fits, without
crossing every profile policy with the full grid. The separate
[Batch A report](phase3-dataset-2026-09-26.md)
records the frozen export and actual dates. The later
[Batch B report](phase3-benchmark-runner-2026-09-26.md) records the implemented
comparators and bounded development smoke. Full development is recorded in
[Batch C's report](phase3-forward-development-2026-09-26.md); confirmation
remains outstanding and no model has been promoted.
