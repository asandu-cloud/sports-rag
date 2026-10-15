# Prediction and bet-selection improvement plan

Date: 2026-09-15

Latest planning update, 2026-09-28: the owner approved prioritising card-data
and target qualification at the start of Phase 4, conditional inclusion of
cards in calibration, and a separate corners performance review to inform its
public recommendation scope. These additions are documented in Phase 4 below.
Phase 3 and its bounded extension are complete with no qualifying candidate;
further count-model experiments are on hold. The subsequent probability-engine
audit and owner-authorised correctness repairs passed 541 tests and four
regression subtests, then were installed in the publishing checkout following
explicit approval on 2026-09-28. Another 275 focused tests passed against the
installed source. Workers were restored with unchanged configuration; an existing
provider-access/failed-refresh gate still blocks new publication and measurement
work. Fitted Phase 4 calibration and real-fixture acceptance remain open.
See the [28 September repair report](phase4-probability-engine-repairs-2026-09-28.md).
Provider collection, model activation and changes to public market availability
remain subject to the existing explicit approval boundary.

Status: Phase 0 baseline preservation completed on 2026-09-15. Phase 1's
candidate feature/snapshot pipeline and isolated fitting/inference runner are
implemented and tested as of 2026-09-18. On 2026-09-22, additive historical
recovery supplied 2,663 verified 2025 fixtures for the original five leagues
and European competitions. Ambiguous event/card statistics and full Phase 0
numerical-decision replay remain open. No ML candidates have been activated.
The bounded Phase 1 diagnostic is now complete: 52 provider-sampled fixtures,
117 successful requests, and confirmed card-label selection bias. See the
[diagnostic report and Phase 2 handoff](phase1-history-diagnostic-2026-09-22.md).
Phase 2 may proceed with explicit missing-data and market-definition gates;
the diagnostic did not implement it or approve unrestricted card modelling.
On 2026-09-23, Phase 2's tracking audit and publication-identity/history work
were implemented; settlement automation and closing-price capture remain open.
On 2026-09-24, exact-fixture resolution and conservative grading were implemented
and tested locally. The later batch added independent prospective scheduling and
closing capture, with a bounded provider diagnostic. Real bookmaker-rule approval,
historical recovery and live matchday acceptance remain open; see
[measurement automation and current status](phase2-measurement-automation-2026-09-24.md).

## Objective

Build a trustworthy way to improve match predictions and select useful bets:
correct historical inputs, evaluate against realistic baselines, calibrate
probabilities, and measure the recommendations actually published to users.

Success means a reproducible improvement on unseen matches with appropriate
uncertainty. A larger model, higher hit rate, or positive ROI on a small sample
is not sufficient evidence on its own.

## Starting point: findings from the audit

- Saved validation R²: goals `0.0181`, corners `-0.0099`, cards `0.0059`, and
  shots on target `0.0269`. The latest training-history entry is September 4.
  All four ML blend weights currently resolve to zero; the statistical
  projections remain responsible for live recommendations.
- The old trainer uses some same-match statistics as inputs. The cumulative
  trainer improves this, but has calendar-year season grouping, missing
  production-rate features, differences between training and live inputs,
  and historical Elo leakage.
- The ML training league list omits Championship, SuperLig, Eredivisie,
  PrimeiraLiga, and BelgianProLeague, despite their prediction/UI integration.
- Recency weights decay per fixture across all competitions. Validation fits
  are unweighted, while final fits are weighted. Parameter selection and final
  scoring also reuse the same cross-validation folds.
- Confidence, distribution floors, and selection thresholds are largely
  configured rules rather than fitted probability calibration.
- At audit time, `platform.db` held 94 canonical published predictions with
  no recorded outcomes, closing-price comparisons, or model versions. Of these,
  55 had kickoff dates before September 15. This is an operational snapshot,
  not a claim that every past fixture is ready to settle.

Relevant existing components:

| Area | Existing implementation to extend |
| --- | --- |
| ML training and inference | [ml_edge.py](../Scripts/rag_ingest/ml_edge.py) |
| Numeric projections and profiles | [projections.py](../Scripts/rag_ingest/core/projections.py), [team_resolution.py](../Scripts/rag_ingest/core/team_resolution.py) |
| Market probabilities and decisions | [prob_models.py](../Scripts/rag_ingest/prob_models.py), [market_service.py](../Scripts/rag_ingest/core/market_service.py), [line_selection.py](../Scripts/rag_ingest/core/line_selection.py) |
| Quality limits | [prediction_guardrails.py](../Scripts/rag_ingest/core/prediction_guardrails.py) |
| Fixture selections and website shortlist | [match_read_compiler.py](../Scripts/data_platform/services/match_read_compiler.py), [match_reads router](../Scripts/web_app/routers/match_reads.py) |
| Publication and tracking | [match_read_delivery.py](../Scripts/data_platform/services/match_read_delivery.py), [predictions service](../Scripts/data_platform/services/predictions.py), [prediction_tracker.py](../Scripts/rag_ingest/prediction_tracker.py) |
| Evaluation | [walk_forward.py](../Scripts/rag_ingest/walk_forward.py), [tracking_metrics.py](../Scripts/data_platform/tracking_metrics.py) |
| Refresh and automated delivery | [season_refresh.py](../Scripts/data_platform/season_refresh.py), [match_read_cycle.py](../Scripts/data_platform/services/match_read_cycle.py) |

## Rollout order

| Phase | Deliverable | Depends on |
| --- | --- | --- |
| 0 | Frozen baseline and experiment contracts | Audit |
| 1 | Correct, shared historical/live features | 0 |
| 2 | Reliable settlement and versioned tracking | 0; can start while 1 is underway |
| 3 | Chronological model benchmarks | 1; betting evaluation also needs 2 or archived quotes |
| 4 | Calibrated probabilities and confidence | 3 |
| 5 | Validated bet-selection policy | 2, 4 |
| 6 | Shadow comparison, controlled promotion, monitoring | 3–5 |

The first implementation batch should cover Phase 0 and Phase 1's feature
contract and defects. Start Phase 2 early so evidence accumulates while the
modelling work proceeds. Full bot/database migration and manager-profile work
are not prerequisites.

## Phase 0 — Preserve a reproducible baseline

Preserved baseline: `baseline-2026-09-15-pre-model-fixes`.
See [capture contents, verification commands, and limitations](prediction-baseline.md).
The saved compiler reproduced all 187 archived Match Read selections. This is
selection replay from canonical results; the full historical projection-input
replay gate below remains open.

### Implement

- [x] Record the current source revision plus a digest of relevant uncommitted
  files, configuration, dependency versions, saved model hashes, and data cutoff.
  A Git commit alone cannot describe this currently modified workspace.
- [x] Archive the existing formulas, thresholds, and model metadata as a named
  baseline. Preserve published decisions and existing data.
- [x] Define distinct identities for feature schema, fitted model, probability
  calibration, and selection policy. Derive one prediction-system version from
  these, including a version for heuristic-only predictions.
- [x] Define experiment outputs: dataset manifest, exclusions/coverage report,
  feature schema, split dates, predictions, metrics, and candidate model bundle.
  Keep experiments separate from the active `Index/ml_models` artifacts.
- [x] Specify the forecast stages to evaluate separately: preliminary reads,
  final pre-match reads, and confirmed-lineup amendments.

### Verify / completion gate

- [ ] Replaying a captured input with the baseline produces the same numerical
  decision, selected price, and policy result without network calls.
- [x] A training experiment cannot overwrite active models or publish cards.
  The supported candidate runner is isolated and tested; legacy CLI fitting
  is retired and routine refresh no longer retrains active artifacts.
- [x] Missing historical inputs are explicitly reported; today's profiles or
  odds are never presented as evidence of what was known before an old match.

## Phase 1 — Repair and unify numerical features

Implementation record: [candidate features, commands, policies, and remaining
gates](prediction-features.md). Checked items below describe the **new candidate
contract**, not a production cutover. Existing heuristic recommendations and
the legacy 58-feature model loader have not been changed. Legacy training/CV
and raw-row diagnostic commands are retired. Candidate fitting/inference use
this contract; the honest chronological benchmark suite is Phase 3 work.

### Implement

- [x] Introduce one shared pre-match feature builder used by candidate training and inference
  prediction. Suggested module: `Scripts/rag_ingest/core/model_features.py`.
  It accepts fixture/team IDs, competition, provider season, an `as_of` cutoff,
  and a versioned input snapshot, then returns ordered features plus provenance.
  Candidate builder, verified training exports, fitted candidate inference,
  and prospective capture are implemented. Production activation remains Phase 6.
- [x] Use API-Football fixture/team IDs as primary identities. Use the shared
  competition registry and explicit provider seasons. Quarantine ambiguous or
  undated rows rather than silently placing them at the start of training.
- [x] Build statistics only from completed fixtures available before `as_of`.
  Use current/prior-season blending consistently with the live profile policy.
  Domestic season state must be separate from continental competition state;
  cross-competition rest/Elo history must be an explicit modelling choice.
- [x] Replace calendar-year tracker keys with actual season keys. Specify prior
  strength and transition treatment for promoted/relegated or newly covered teams.
- [x] Preserve the goals, corners, cards, and SoT production-rate fields through
  profile conversion. Remove duplicate/constant features where appropriate;
  version any feature-order change.
- [x] Supply the same recent-form windows, actual rest periods, venue splits,
  referee context, and optional head-to-head features at training and inference.
  Either implement a feature consistently or remove it from both paths.
- [x] Compute Elo sequentially: snapshot before each fixture, then update from
  its result. Define season regression and competition weighting explicitly.
  Historical referee/profile aggregates must also respect their cutoff.
- [x] Distinguish missing statistics from real zeros. Do not automatically
  discard a legitimate zero-card/zero-corner match or treat missing goals as 0–0.
  Fit imputations using training data only and retain missingness indicators.
  Nulls, real zeros, and explicit indicators are implemented in the candidate
  contract. The isolated runner fits median imputation and scaling on the
  explicitly bounded training subset only; dataset export performs no imputation.
- [x] Include all ten domestic leagues and UCL/UEL/UECL via the registry. Emit
  coverage by competition, season, and market. Model eligibility depends on
  verified data coverage, not just inclusion in the registry.
- [x] Retire the same-match training path from normal use. Candidate training
  and prediction diagnostics use the shared contract. Legacy CV is disabled;
  its replacement is the chronological benchmark suite in Phase 3.

### Verify / completion gate

- [x] Changing the target match's result or adding later matches does not change
  its saved pre-match features, including Elo and referee features.
- [x] Training and candidate inference produce identical vectors for the same snapshot.
- [x] December/January stays within one season; summer transitions start the
  correct next season and carry only the configured prior.
- [x] Known nonzero production rates survive conversion; missing and zero values
  have distinct behaviour. New-league fixtures pair correctly by ID.
- [x] Domestic and European profile blending, promoted teams, and sparse samples
  have focused regression coverage.

Deliverable: a versioned feature dataset and coverage report. No model promotion.

Current deliverable: `phase1-features-2025-2026-recovered-2026-09-22` contains
5,083 fixtures; all vectors replayed successfully. The earlier 2,286-fixture
export and prospective capture remain preserved. See the
[recovery report, coverage and verification](phase1-history-recovery-2026-09-22.md).

- [x] Recover verified 2025 fixture identities/results and usable local team
  statistics for the original leagues and European competitions, additively.
- [x] Trace card nulls against local raw archives; retain unknowns instead of
  manufacturing zero-card observations.
- [x] Produce and replay a new versioned candidate export with explicit
  missingness and market/league coverage. No active models or prediction rules changed.
- [x] Run the bounded local/provider diagnostic across all 13 competitions:
  5,083 local FT fixtures audited, 52 provider-sampled, 48 comparable two-team
  statistics responses. Every one of the 305 known 2025 card labels includes a
  recorded positive red count. Fresh-value candidates and revisions are saved
  for review, not applied. See the [report](phase1-history-diagnostic-2026-09-22.md).
- [ ] Resolve ambiguous legacy zeros/provider-null card semantics sufficiently
  for representative market-level training/evaluation. A larger statistics
  backfill is deferred pending review at the user's request.

Phase 1 is not yet fully signed off for unrestricted modelling. The independent
full Phase 0 numerical-decision replay gate also remains open: original profile
inputs and complete quote slates are not recoverable from stored selections.

## Phase 2 — Complete the measurement loop

Implementation update (2026-09-23): points 1–2 of the agreed execution order
(audit connections, preserve publication identity/history) are implemented.
See [audit, behaviour, tests and remaining work](phase2-publication-tracking-2026-09-23.md).
Settlement scheduling, safe outcome recovery and closing-price capture are not
completed by this batch. Existing database records were not rewritten.

Update (2026-09-24): the next two build points—exact-fixture resolution and safe
grading—are implemented. The canonical resolver now uses fixture IDs/UTC kickoff
and persists pending reasons/evidence. It does not grade ambiguous old periods,
invent missing statistics or approve bookmaker card/cancellation rules. No live
settlement, backfill or scheduling ran. See the
[implementation, verification and limitations](phase2-safe-settlement-2026-09-24.md).

Later update (2026-09-24): independent prospective scheduling and exact closing-
price capture are implemented; the local measurement job is installed/loaded.
The provider diagnostic used 16 requests for eight fixtures without changing
historical predictions. All 35 sampled publications lack a saved market period
and remain pending. Applicable bookmaker rules, historical recovery and live
matchday sign-off remain open. See
[measurement automation and validation evidence](phase2-measurement-automation-2026-09-24.md).

Remaining local work update (2026-09-24): offline historical recovery planning/
application, full-cohort audit, explicit pending reporting, website/Discord
contract fixes and an end-to-end isolated lifecycle test are implemented. All
467 old publications lack a saved period (461 currently overdue); no historical
outcomes were invented. Bookmaker/site approval and live prospective acceptance
still prevent full sign-off. See
[local completion, evidence and remaining gates](phase2-local-completion-2026-09-24.md).

Product-policy update (later 2026-09-24): the owner approved a simplified,
participation-based card convention and abandonment-as-void for new publications.
This is now versioned and wired to the measurement worker, with explicit
product-rule estimate labels in reporting. A bookmaker/site choice is no longer
a prerequisite for this product-rule cohort; bookmaker-equivalent settlement is
not claimed. Old rows are not opted in. Live validation and alignment of card
model targets remain open. See [policy and limitations](participation-settlement-policy.md).

Diagnostic handoff: proceed without waiting for all historical gaps to close.
Missing/ambiguous outcomes must remain visibly pending. In particular, preserve
yellow-only versus total-card market identity through grading and closing-price
matching. The new grader enforces this identity and withholds unverified card
definitions. Canonical closing capture now keeps these exact keys separate and
never falls through to the old generic legacy matcher.

### Implement

- [x] Trace the existing resolver and closing-odds capture through their actual
  backend/source filters and schedules. Establish why canonical published rows
  remain unresolved; extend the existing services rather than duplicating them.
- [x] Replace canonical publication-day/name matching with exact provider fixture
  IDs, actual kickoff cutoffs, shared dry-run behaviour and visible failure states.
- [x] Implement/test conservative grading, quarter-handicap half-settlements,
  pending reasons, evidence retention and explicit targeted correction history.
- [x] Schedule settlement independently of whether the Discord bot is running.
  Process unresolved fixture IDs after result ingestion, using shared grading
  and outcome-accounting functions. Retry incomplete provider statistics.
  Installed prospectively after publication ID 467; shared refresh gates still
  apply. Future matchday acceptance is separate from successful installation.
- [ ] Preserve full/half wins, pushes, full/half losses, and voids. Apply the
  correct market period and the recorded settlement policy, especially 90-minute results versus
  extra time. Missing card/SoT data must remain pending, not become a loss or zero.
  Engine and synthetic coverage are implemented; this end-to-end approval remains
  open until the new participation-policy results are verified on a controlled
  live sample. Product rules are approved; exact bookmaker equivalence is not claimed.
- [x] Snapshot the approved participation-card/abandonment policy on new releases,
  grade positive-minute player totals with 1/2/3 weighting, preserve old records,
  and label the measurement basis in API/Discord reports. Yellow-only markets
  remain separate and unapproved; this does not change card prediction formulas.
- [x] Propagate the prediction-system version and snapshot identifiers through
  canonical results, Match Reads, published recommendations, and reports.
  Prospective deployments are fingerprinted; missing historical model/system
  versions remain unknown. The fingerprint is not full numerical-input replay.
- [x] Implement prospective API-Football capture near kickoff. Match the exact
  bookmaker, market, side, line, and period. Record quote time and capture time.
  Define closing odds as the latest valid pre-kickoff quote within a configured
  freshness window; unmatched or stale quotes remain unavailable.
  Scheduled implementation tested; provider odds-row update time is preserved
  separately from capture time. The 15-minute freshness limit may yield low
  coverage with this feed. Live capture/CLV acceptance remains open below.
- [x] Keep published picks, unpublished shadow candidates, legacy standalone
  picks, and unit-bet slips as separate reporting cohorts. Deduplicate website
  and Discord deliveries of the same recommendation.
- [x] Preserve the first-publication headline results and report later amendments
  separately. Version selection must never depend on the eventual outcome.
  Initial scope uses the first delivered fixture card across surfaces, including
  no-bet cards; later additions are amendments. Unlinked publications remain
  unclassified in the explicit all-releases audit.
- [ ] Recover old outcomes where authoritative data exists. Leave historical
  model versions and closing prices unknown unless evidence can recover them.
  The offline plan/apply tool and full historical audit are complete. Currently
  no old published selection has sufficient recorded period identity to grade.
  The 461 pending-note writes were deferred by the active refresh gate; six
  further selections are not yet due on the current canonical schedule.

### Verify / completion gate

- [x] Verify the full publication → closing capture → settlement → restart →
  website/Discord reporting sequence in isolated tests, including Asian halves,
  separate voids, pending reasons and exact metric parity. This is local
  acceptance, not a substitute for the prospective live checks below.
- [ ] Every new published selection links to one trackable prediction and a
  known system version. Retries/restarts produce no duplicate stakes or grading.
- [ ] A controlled completed-fixture sample grades correctly for every supported
  market, including Asian lines, corrections, cancellations, and missing stats.
- [ ] All overdue records with sufficient final data settle; remaining records
  have a visible reason, age, and retry path.
- [ ] Both delivery surfaces report identical outcomes and unit-stake ROI.
  Closing-price comparisons exclude post-kickoff or mismatched quotes.

## Phase 3 — Establish honest model benchmarks

The [detailed Phase 3 technical plan](phase3-technical-plan-2026-09-26.md)
records the four implementation batches, separate Phase 3 and final-system
tests, and the owner-selected model roster. [Batch A is implemented and locally
verified](phase3-dataset-2026-09-26.md): eligibility/evidence contracts, the new
immutable reference dataset, chronological partitions and isolated target
stores. All 31,588 feature records replayed; 186 focused/regression tests passed.
Pooled numerical benchmarks can proceed within the recorded evidence limits;
five newer leagues lack qualifying recent period evidence and European league
slices remain below confirmation sample minima. No benchmark model had been
trained or scored at the Batch A handoff. [Batch B is now implemented and
locally verified](phase3-benchmark-runner-2026-09-26.md): both baselines,
weighted preprocessing, all five estimator adapters and 15 bounded development
smoke fits passed. [Batch C is now complete locally](phase3-forward-development-2026-09-26.md):
the corrected run completed all 1,872 declared recipe/fold evaluations,
chronological OOF blends, history comparisons and seven paired feature diagnostics.
A constant-column preprocessing defect in the first attempt was fixed and
recorded; the accepted run uses preprocessing v2. Development gains are small:
no market meets the agreed 1% point improvement against both baselines.
No candidate was qualified or promoted by that development run.
The [authorized bounded extension is also complete](phase3-extension-2026-09-27.md):
72 development estimator fits plus six final refits compared signed residuals,
statistical offsets and separate home/away models. Its selected procedure improves
slightly over Batch C for goals/corners and is slightly worse for shots on target;
no market reaches 1% against both baselines. All 609 focused/regression tests and
the independent audit passed. The subsequently authorized
[Batch D confirmation is complete](phase3-confirmation-2026-09-27.md): 1,949
goals, 1,776 corners and 1,790 SoT fixtures, with 656 passing tests. Goals and
SoT fall short against the statistical reconstruction; corners improves 1.861%
against it but only 0.883% against league averages, below the agreed 1% minimum
against both. No candidate qualifies. The statistical production engine is
retained. The 2024 confirmation is consumed; calibration and final-system-test
reserves remain unopened. A pre-scoring European round-label validation repair
is preserved with immutable amendment provenance. Phase 4 implementation has
not started; its revised priorities were approved on 28 September below.
XGBoost and CatBoost remain in the completed Phase 3 roster, with logistic
regression for Phase 4 probabilities.

### Implement

- [x] Create chronological train/tuning/calibration/test partitions grouped by
  fixture and kickoff time. Keep all markets and versions of a fixture together;
  reserve the separate confirmation and final-system tests.
- [x] Freeze versioned eligibility, source evidence, exclusions, feature records
  and deterministic membership hashes; restrict development target access.
- [x] Implement weighted preprocessing fitted on training data only in the
  Phase 3 estimator runner. Keep reserved model scores unopened until their
  declared evaluation operation.
- [x] Use forward folds for development; choose hyperparameters within those
  folds rather than selecting and claiming final performance on the same data.
- [x] Use elapsed-time recency decay in candidate benchmarks. Choose its
  half-life on development data and use identical weighting in validation and
  final training. Report effective sample sizes and per-league contributions.
- [x] Compare league/venue averages, reconstructed statistical formulas, regularized
  Ridge and Poisson regression, LightGBM, XGBoost and CatBoost on identical
  fixtures. Use comparable bounded development search budgets for the three
  tree families and isolate their dependencies from production.
- [ ] Compare with vig-adjusted bookmaker probabilities where authentic
  timestamped equivalent quotes exist; no historical quote cohort was available
  for Batch C.
- [x] Benchmark the heuristic/ML blend itself. Learn blend weights using
  out-of-sample predictions, with zero ML contribution as a valid outcome.
  A pooled R² threshold alone must not decide promotion.
- [x] Report MAE, bias, RMSE and R² for totals; common uncalibrated Poisson
  log loss/RPS and prediction-interval coverage/width. Batch D preserves full
  confirmation reports. Fitted probability calibration belongs to Phase 4;
  actual-price betting metrics remain unavailable without authentic quotes.
- [x] Break development reports down by market, league, season stage, forecast stage, and
  missingness. Include ablations for priors, Elo, referee, venue, and recent form.
  Batch C contains one reconstructed forecast stage; real lineup/amendment
  stage comparisons remain unavailable.

### Verify / completion gate

- [x] Every scored numerical example respects its feature cutoff. Backfilled
  history is explicitly `assumed_final`; no historical betting/price comparison
  was scored and this is not an immutable live capture.
- [x] Fixed manifest, split and seed reproduce frozen predictions exactly and
  confirmation metrics/intervals within the recorded numerical tolerance.
  Weighted fitting/validation was verified in Batch C; Batch D performs no fits.
- [x] Assess improvement against the preregistered primary metric with uncertainty
  and supported-slice regressions exposed. Batch D's result is **no qualifying
  candidate**; this marks the assessment complete, not the candidate gate passed.
  Sparse and absent competitions remain unqualified.
- [x] Retain the statistical baseline when gains do not meet the agreed gates.
  Batch D records small positive gains but no qualifying improvement; thresholds
  were preserved and no ML contribution was activated.

## Phase 4 — Calibrate probabilities and confidence

### Frozen research control and evaluation protocol — 28 September 2026

The owner authorized preserving the **current repaired statistical engine** as
`phase4-repaired-statistical-control.v1`, under the
[frozen evaluation protocol](phase4-evaluation-protocol-2026-09-28.md).
The baseline artifact is `Research/phase4-baseline-2026-09-28/control/`; its
COMPLETE manifest and replay-validation record are authoritative for completion.
The [freeze report](phase4-baseline-freeze-2026-09-28.md) explains its boundaries.

The control retains the installed eight-match prior cutoff and existing unfitted
probabilities. Strength-16 gradual weighting remains an isolated challenger;
no model or public delivery change is implied by freezing the reference. Cards
are preserved for behavioral replay but remain outside qualified fitting.
Source/config/model identities and development-only inputs are preserved;
synthetic input replay executes real projection, variance, probability, selector
and guardrail arithmetic, without asserting historical publication replay.
Historical input reconstruction is Step 1 of the agreed remaining work;
candidate comparisons follow separately in Step 2.

Protocol v1 fixes 2022/2023 development use, excludes consumed 2024 outcomes from
new fitting, reserves Jan–Mar 2025 for later calibration fitting and Apr–Jun 2025
for one frozen-method qualification, and protects Jul 2025–Jun 2026 for the full
probability/selection-system final test. No reserved outcomes are opened by this
freeze. Candidate families, thresholds, support minima and failure handling are
predeclared. Amendments create new versions and preserve inspection history.
The later owner-approved [qualification amendment](phase4-qualification-amendment-2026-09-29.md)
explicitly supersedes those 2024/2025 input-period rules: fixed-method fitting can
use earlier 2024 observations, the existing development calibrator is preserved,
and January–June 2025 is one qualification period. July 2025–June 2026 stays closed.

### Full-engine walkthrough and five-step implementation order

The [source-based engine walkthrough](prediction-engine-walkthrough-2026-09-28.md)
records the actual data stores/windows, configured weights, variance/distribution
paths, selector rules and context wiring limitations. The
[specific comparison implementation plan](phase4-full-probability-comparison-plan-2026-09-28.md)
sets out historical snapshots/control parity, separate candidate comparisons,
the probability comparator/calibration, supported-scope review, and later
qualification as five distinct steps. It preserves the frozen
protocol and explicitly lists proposed scope/rolling-history clarifications for
owner agreement. No full-engine comparison fitting or production overhaul was
started while preparing those documents. The owner subsequently authorized
Step 1 on 29 September; its [implementation and validation report](phase4-historical-control-2026-09-29.md)
records the resulting research artifacts. Steps 2–5 are not executed by Step 1.

On 29 September the owner authorized expanded **Step 2 candidate implementation**:
opponent-adjusted strengths, sample-aware pooling, xG handling, bounded history/mean
weights and the original prior/distribution changes. The
[additive protocol v2](phase4-step2-candidate-protocol-v2-2026-09-29.md) fixes the
candidate budget; the [implementation report](phase4-step2-candidate-implementation-2026-09-29.md)
records correctness tests and an unscored historical smoke. Candidate code is
research-only. Those implementation checks alone did not establish improvement.
Richer lineup effects have a gated delta interface, but lack
qualified historical inputs and fitted player contribution/minutes evidence.

The owner subsequently authorized the actual backtest. The
[Step 2 results](phase4-step2-backtest-results-2026-09-29.md) record 2022 forward
tuning, locked settings and 34 later-2023 development comparisons. Opponent-adjusted
goals, fitted corner dispersion and fitted SoT dispersion pass the predeclared
development screen with 0.719%, 1.032% and 1.411% lower primary NLL, respectively.
These are probability-loss changes, not profit/hit-rate claims. 497 tests pass;
6,740 eligible control forecasts replay exactly and 573 fitted bundles pass
the independent training/cutoff audit. No combined configuration, Step 3 calibration,
reserve qualification or production promotion is completed by this result.

The subsequent owner-authorized **Step 3** implemented logistic comparison,
monotone sigmoid calibration and coherent goal-temperature calibration. Its
[development report](phase4-step3-calibration-results-2026-09-29.md) records a
0.583% BTTS comparator and 0.633% corner-9.5 sigmoid log-loss improvement against
their frozen raw controls, each passing the binary development screen. These
isolated outcomes do not replace shared distributions. Control goal temperature
selects identity. That initial run had only 13 challenger validation weeks and
could not evaluate that scope. The subsequent authorized
[earlier-forecast extension](phase4-step3-earlier-forecasts-results-2026-09-29.md)
resolves the shortage with 43 weeks and unchanged thresholds. Challenger corner-9.5
sigmoid calibration passes with .862% lower log loss against its raw challenger;
the remaining challenger sigmoids add no qualifying improvement. Coherent goal
temperature selects identity on sufficient evidence. That extension passed 523 tests.

The subsequent [Step 4 finalist review](phase4-finalist-review-results-2026-09-29.md)
completed the fixed combination and supported-scope comparison. The three primary
components pass in the predeclared five-league, >=8-current-match scope; complete
served-cohort and sparse-slice limitations are disclosed. A research recipe is
checksum-locked; 537 tests and 27,480 exact score replays pass. The initial Step 5
Q2-only metadata block was subsequently resolved by the owner-approved
[January–June 2025 amendment](phase4-qualification-amendment-2026-09-29.md).
The [qualification evaluation](phase4-qualification-results-2026-09-29.md) is
complete: corner/SoT count dispersions pass the unchanged criteria and four-claim
Holm correction, with 1.108%/2.125% lower NLL over 888 active-scope fixtures/20 weeks.
The goal-strength candidate does not qualify (934 fixtures); the standalone
corner sigmoid remains inconclusive. Retain the coherent goal control and identity
calibration. The 2024 input/refit bridge was explicitly declared and fitted bundles
locked before 2025 outcome access; no 2025 calibrator refitting occurred. 551 tests,
2,073 exact control replays and the saved audits pass. January–June is now consumed
qualification evidence. July 2025–June 2026 remains closed for the completed system;
card qualification, richer lineups and production/shadow acceptance remain open.

### Agreed order and scope — 28 September 2026

The owner approved the following priorities after the Phase 3 confirmation and
subsequent read-only review. Cards remain an important product market. Their
exclusion from the frozen research dataset is a target/evidence restriction;
it did not remove cards from production or establish that cards are inherently
unpredictable. Corners has a documented numerical-baseline weakness requiring
review, but permanent removal has not been decided.

1. Prepare the detailed Phase 4 technical protocol for owner review: audit the
   existing probability/settlement paths, define supported scope, fitting and
   evaluation periods, primary scores, sample/precision requirements and
   acceptable regressions before fitting or opening reserved outcomes.
2. Make the bounded card-data and target assessment the first workstream. Start
   with existing archives and player records; separate repairs possible locally
   from evidence that would require additional provider collection.
3. Include cards in probability fitting/calibration only for the market,
   league and period scopes that pass target and sample qualification. Record
   specific blockers and the next evidence needed for other scopes. Eligible
   goals, corners, SoT and goal-derived markets can proceed independently.
4. Conduct the separate corners review below, distinguishing numerical accuracy,
   probability reliability and actual published recommendation performance.
   Use that evidence to propose its public recommendation scope.

Phase 4 starts from the retained statistical engine; a promoted Phase 3 ML model
is not a prerequisite. The agreed logistic regression probability comparison
remains in scope. The owner has also requested a [richer lineup-effects workstream](phase4-richer-lineup-effects-plan.md), extending existing lineup adjustments with timestamped inputs, expected minutes, replacement effects and paired ablations. It follows the probability baseline work and must not delay it. The full player-props engine remains a separate future workstream; reusing player evidence for card targets does not require first building that engine.

The 2024 Phase 3 confirmation has been consumed and can only provide disclosed
post-confirmation diagnostic/development evidence. Preserve January–June 2025
for calibration and July 2025–June 2026 for the completed probability/selection
system's final test; the prospective reserve starts in July 2026. Freeze exact
fitting, calibration and intermediate validation use before accessing those
reserves. A card repair or new export does not reset inspection history. The
initial card audit must use permitted historical outcomes and outcome-free
coverage metadata for reserved periods, rather than decoding reserved outcomes
incidentally through raw archives or player tables. Do not consume the final
system test to choose a Phase 4 method or a new market scope.

### Shared correctness prerequisite — installed after approval

The [probability-engine repair](phase4-probability-engine-repairs-2026-09-28.md)
corrects low-score and Asian-handicap arithmetic, propagates forecast context,
prevents incompatible quote pairing, preserves probability/settlement semantics
and supplies a coherent, unfitted goal distribution. Local validation and
owner-approved source installation are complete; prospective acceptance is not.
The installed engine uses the existing statistical means, fixed rho and heuristic
confidence. These are inputs
to the Phase 4 evaluation protocol, not evidence of calibrated superiority.
The card workstream below remains the first market-specific qualification task.

The follow-up [market-contract and specialist repair](phase4-market-contract-and-specialist-repair-2026-09-28.md)
is also installed. It enforces explicit canonical quote compatibility, shares marginal
probability/Asian settlement calculations with supported parlay and unit-bet selections,
preserves no-bet/release gates and removes invented joint EV. Local validation passed
616 tests plus four regression subtests; 358 focused tests passed after installation.
The configured API-Football key was separately verified as reporting Free-plan access,
so both workers still defer behind the refresh gate. Restore provider entitlement and
complete the ordinary guarded refresh before live acceptance. Review the unchanged
eight-match prior cutoff before freezing the Phase 4 baseline; card qualification and
fitted calibration remain outstanding. See the report for specialist migration limits.


### Prior-transition candidate — development evaluation complete, not installed

The owner-authorized [gradual-prior study](phase4-prior-transition-results-2026-09-28.md)
replaces the abrupt eight-match cutoff in an isolated candidate and compares strengths
4/8/16 under a [predeclared protocol](phase4-prior-transition-protocol-2026-09-28.md).
Strength 16 was selected on 2022 and passed the declared 2023 pooled/supported-slice
checks. RMSE changes are −0.312% goals, −0.683% corners and −0.130% SoT; the SoT
uncertainty interval includes zero. The five major domestic leagues meet slice-support
requirements; other leagues do not, and UECL goals worsen 3.336% on 93 fixtures.
Cards and probability/distribution qualification remain open. 378 tests pass and the
reviewable five-file patch is preserved. Publishing source, active models and reserved
periods remain unchanged. This is a leading baseline candidate, not a frozen universal
baseline or an installed production policy.

### First development batch — card qualification and count dispersion

The [bounded protocol](phase4-card-calibration-protocol-2026-09-28.md) is implemented
and was run on permitted pre-2024 data. See the [results and remaining work](phase4-card-calibration-results-2026-09-28.md).
It adds a versioned research target adapter using the approved settlement parser,
per-fixture exclusions, a read-only card evidence export, and a chronological
fixed-mean count-dispersion experiment. The card audit found 895 provisional
arithmetic totals across 906 player-covered fixtures, but no qualified card targets
under its period/source requirements. This is a measured evidence restriction,
not a decision to abandon the card market. Calibration development proceeded
independently for goals/corners/SoT. No public probability policy was installed,
and the protected calibration/final-system reserves remain unopened by this batch.
408 focused and regression tests passed; artifact and source hashes were verified.

The broader checkboxes below remain open where this first bounded audit does not
complete them. In particular, unarchived source fields cannot yet be classified
as original provider nulls versus lossy exported zeros; the report preserves that
uncertainty. Richer lineup effects are now a separately documented workstream,
with expected/confirmed stages, contribution replacement, uncertainty and
coherent team-market effects; existing lineup support is retained as a comparator.

### First workstream — card evidence, target repair and qualification

**Subsequent 30 September owner decisions:** research now uses the staged
[v2 two-minute rule and league-phase scope](phase4-card-policy-v2-results-2026-09-30.md).
The aligned calculation and profiles will enter production together with the new
engine; existing publications retain v1. Ordinary domestic league phases and
European group/league phases are included; separate playoffs, qualifying and
knockouts are deferred. Exact completed fixture metadata in those known formats
establishes regulation scope. The implementation and scoped local reconstruction
are tested, while the missing earlier player/source evidence still prevents real
card fitting/evaluation. Null reds remain unknown; no download or weight search
has been started.

**30 September local reconstruction update:** the owner authorized card
reconstruction/backtesting and explicitly deferred weight optimisation. The
[local reconstruction results](phase4-card-reconstruction-results-2026-09-30.md)
now preserve 2,101 provisional totals across all 13 competitions, up from 895 in
the earlier audit. Zero targets meet the source contract and the earlier fitting
cohort is absent. The fixed dated team/referee comparison is implemented and
tested, but its real-data scoring is blocked; no performance result or completed
card qualification is implied. No provider calls, production changes or weight
search took place. See the report for exact evidence exclusions and the resume
point. The broader qualification gates below remain open where data is missing.

The [25 September repair](card-stat-missingness-repair-2026-09-25.md) fixed the
legacy team collector's null/zero handling and recovered five evidence-backed
fields; it did not reconstruct every missing historical count. The
[expanded frozen dataset](phase3-dataset-2026-09-26.md) recorded 4,537 computable
legacy yellow-plus-red totals and zero qualified card targets. These are dated
counts, not a current database audit. The earlier red-card-enriched sample and
the mismatch with weighted participation settlement remain unresolved gates;
the larger import alone does not establish representative, equivalent labels.

- [ ] Inventory existing raw team/player responses, event evidence where
  available, normalized records and source identities by league/season. Report
  verified zeros, positive counts, provider nulls, omitted values, lossy legacy
  zeros, participation evidence, period evidence and source conflicts separately.
  Use permitted periods and preserve observation-time/assumed-availability
  provenance; do not manufacture evidence of original forecast-time availability.
- [x] Define a versioned target contract aligned with
  [spix-participation-settlement.v1](participation-settlement-policy.md): regulation
  plus stoppage time, yellow = 1, straight red = 2, two yellows and dismissal =
  3 total, with recorded minutes greater than zero for player eligibility.
  Preserve the policy's limitations around booking time and aggregate evidence.
  Keep yellow-only markets distinct. The product convention is not automatically
  equivalent to a bookmaker's rules; record exact market/settlement compatibility.
- [ ] Identify and implement evidence-supported repairs in new, versioned
  research artifacts, with per-fixture exclusions and provenance. Unknown red
  counts or absent timeline events cannot imply zero. A change to
  `yellow + 2 * red` alone cannot resolve second yellows, participation or period
  scope. Incomplete/inconsistent player records and unsegmented extra-time totals
  remain unavailable. Preserve original exports, frozen experiments, publication
  policies and historical grades; canonical repair, if needed, requires an
  explicit reviewed application step with backup and before/after evidence.
- [ ] Assess selection bias on the resulting usable cohort, including verified
  no-red-card fixtures and verified zero-card outcomes. Compare inclusion and
  exclusions by league, season and source; do not infer that an unlabelled match
  had no dismissal or that a larger usable count proves representativeness.
  Apply predeclared chronological sample/precision requirements by supported
  scope. Report insufficient or unresolved cohorts explicitly.
- [ ] Check that the isolated statistical card projection estimates the same
  quantity as the proposed target before calibrating its probabilities. Record
  any required projection changes as separate candidate work; calibration must
  not silently reinterpret a yellow-plus-red projection as an aligned weighted
  target. Verify target arithmetic against settlement on identical evidence,
  including straight reds, second yellows, zero/unknown minutes, incomplete
  records and regulation/extra-time boundaries.
- [ ] Deliver a card-qualification report and immutable candidate dataset where
  feasible: scope, counts, exclusions, hashes, target version, chronological
  reservations and unresolved bias. List additional evidence needed by fixture
  and endpoint, with a bounded request estimate. No bulk backfill is authorised
  by this plan update; provider collection needs separate approval.

The decision must identify qualified card cohorts or precise outstanding
blockers. Only qualified scopes enter Phase 4 calibration and its later
evaluation. This work does not require a successful card ML count model, and a
local target repair alone does not qualify public recommendations.

### Separate workstream — corners performance and recommendation scope

On the 1,776 eligible 2024 fixtures, the statistical corners reconstruction had
RMSE 3.352327 versus 3.319237 for the league-average baseline. The frozen
candidate reached 3.289928 but missed the agreed 1% improvement against the
average. These [Phase 3 results](phase3-confirmation-2026-09-27.md) concern
reconstructed count forecasts. They do not establish calibrated probability
quality or authentic published-bet returns, and no corner market was disabled.

- [ ] Review numerical accuracy against the same-fixture league-average and
  statistical baselines, with bias, uncertainty and supported league/line-related
  diagnostics. Disclose the consumed status of 2024 evidence. Do not compare raw
  RMSE across goals/corners as if the count scales were interchangeable.
- [ ] Evaluate corner count distributions and probability calibration against
  current defaults on declared chronological cohorts. Report proper scores,
  reliability, interval coverage and supported scope; predeclare selection
  rules and validation use before fitting, with final-system evidence protected.
- [ ] Audit actual published corner recommendations as a separate cohort using
  exact market, line, period, price, system version and settlement evidence.
  Preserve first-delivered Match Read identity, deduplicate website/Discord,
  keep amendments separate, and retain first no-bet publications. Separate
  shadow candidates, legacy picks and unit slips; report pending, push, half
  and void outcomes explicitly. Where evidence permits, report flat-stake ROI,
  turnover, drawdown and verified CLV coverage with fixture/time-grouped
  uncertainty. Missing quotes, equivalent settlements or adequate sample size
  produce unavailable/insufficient-evidence findings, not reconstructed returns.
- [ ] Produce a review distinguishing numerical, probability and published
  performance findings. Propose retaining corners, restricting recommendations
  to supported scopes, or pausing public corner selections as the evidence
  warrants. Sparse or inconclusive evidence is not proof that the whole market
  fails. Retain research/history when proposing a publication pause. A public
  restriction, pause or removal remains an explicit owner decision; this review
  does not automatically change delivery or ranking rules.

### Implement

- [ ] Fit distribution dispersion and the low-score goal-correlation adjustment
  on training data. Evaluate them against the present fixed defaults.
- [ ] Keep goals totals, BTTS, moneyline, and spreads coherent through a shared
  score distribution, or document/test any alternative probability construction.
- [ ] Include regularized logistic regression as a direct probability
  comparator for predeclared binary outcomes such as BTTS and fixed half-line
  totals. Evaluate against distribution-derived probabilities on identical
  chronological fixtures using proper scores and reliability analysis. Keep
  fitting/calibration separate from later testing, check related-market
  consistency, and do not collapse Asian pushes/half outcomes into binary labels.
- [ ] Fit probability calibration on held-out/out-of-fold predictions; score it
  on a later test period. Use pooled estimates with conservative fallback when
  a league/market lacks enough observations for its own calibration.
- [ ] Record raw and calibrated probabilities, calibration version, applicable
  sample size, and uncertainty. Separate likelihood of winning from estimated
  value and confidence in the estimate.
- [ ] Preserve Asian settlement distributions. A non-push equivalent probability
  is not the probability of a full win: use the appropriate settlement outcomes
  for EV and scoring rather than treating every quarter-line result as binary.

### Verify / completion gate

- [ ] Card-target qualification is documented by supported scope, with settlement
  parity, source evidence, sample support and unresolved bias exposed. Every card
  calibration result uses that exact target definition; blocked cohorts remain
  outside fitting/evaluation without being described as failed card predictions.
- [ ] The corners review separates historical count accuracy, probability
  reliability and actual published performance, and records the evidence and
  owner decision required for any public scope change.
- [ ] Probability buckets agree with observed frequencies within reported
  uncertainty; calibration improves a proper score such as Brier/log loss on
  applicable outcomes without fitting to the final test results.
- [ ] Probabilities are finite, normalized, and coherent across lines/outcomes;
  push and half-settlement calculations match the settlement engine.
- [ ] Sparse, missing, or out-of-distribution inputs still trigger the existing
  quality limits. Calibration does not silently bypass those safeguards.

## Phase 5 — Improve selection of the best bets

### Implement

- [ ] Define one versioned selection/ranking policy shared by market decisions,
  the Match Read compiler, and the website shortlist. Evaluate calibrated EV
  with estimation uncertainty, executable odds, and data quality.
- [ ] Carry forward Phase 4's market/league qualification and any explicitly
  approved public scope decisions from the cards/corners reviews. Product
  variety does not override evidence, price or no-bet requirements; research
  eligibility and public market availability remain separately recorded.
- [ ] Tune eligibility and ranking on development periods, then freeze the policy
  before final testing. Compare it directly with today's confidence/edge ranking.
  Retain paired-price checks and a legitimate no-bet outcome.
- [ ] Log the evaluated candidate set, selected quote, and rejection reasons.
  Compare prices only for equivalent settlement rules; reject stale/unavailable
  quotes. This makes selection changes measurable, not just model changes.
- [ ] Preserve up to three compatible selections per fixture and the capped
  five-fixture shortlist. Treat their outcomes as dependent in performance
  analysis. Combined same-game probability/EV remains a separate future project.
- [ ] Measure the user-visible portfolio: first releases, amendments separately,
  market/league concentration, turnover, ROI, drawdown, and closing-line value.
  Report counts of unique fixtures as well as bets.

### Verify / completion gate

- [ ] Website and Discord choose the same selections from the same snapshot.
  Reordering input markets does not change ranking except documented tie-breaks.
- [ ] Candidate thresholds reject negative/insufficient EV and unsupported
  prices. Compatibility, freshness, and no-bet regression tests continue to pass.
- [ ] Compare the complete policy on unseen periods; use fixture/matchday-grouped
  uncertainty estimates so several related bets do not masquerade as independent
  evidence. Report concentration and missing-quote exclusions explicitly.

## Phase 6 — Shadow validation and controlled activation

### Implement

- [ ] Save immutable candidate bundles with data cutoff, feature/model/calibration/
  policy versions, checksums, metrics, and supported market/league scope.
- [ ] Change refresh orchestration so all selected leagues finish feature builds
  before candidate training. Training saves a candidate; only an explicit
  promotion action updates the active version. Failed training keeps the prior
  active bundle. Do not retrain within each ten-minute Match Read cycle.
- [ ] Run current and candidate policies on the same scheduled snapshots. Use
  separate candidate storage/namespaces so shadow reads cannot become public
  cards or supersede the website's effective version.
- [ ] Promote only qualified markets/leagues after reviewing benchmark and shadow
  results. Keep the previous active bundle for rollback and verify that long-lived
  bots reload it; current process caches must not mix old models with new metadata.
- [ ] Monitor input coverage/freshness, settlement backlog, model fallback,
  probability shifts, excessive edges, no-bet rates, and runtime failures.
  Assign actionable failure reasons and schedule periodic performance reviews.

### Verify / completion gate

- [ ] Observe multiple real matchdays, including sparse slates, busy slates,
  postponed fixtures, early-season profiles, and confirmed-lineup amendments.
  Elapsed matchdays alone do not establish statistical readiness.
- [ ] Set market-specific sample/precision requirements and acceptable baseline
  regressions before examining candidate test results. The existing readiness
  defaults of 50 resolved units and 20 per bucket are screening checks, not proof
  of profitability. Record actual counts and uncertainty before promotion.
- [ ] Exercise failure, restart, rollback, and concurrent-run scenarios. Confirm
  no duplicate publications, silent version mixing, or exposure of shadow picks.
- [ ] A released version is traceable from the website/Discord selection through
  its exact input and quote to final settlement and the evaluation report.

## Boundaries and final acceptance

API-Football remains the data/odds provider; this work does not reintroduce The
Odds API. GPT-written briefings describe the numerical assessment and do not
choose bets or determine probabilities. More seasons may be retained for
experiments, but changing the effective history window requires evaluation.
Manager profiles, transfer effects, richer player-fit models, and same-game
joint pricing come after these foundations are verified.

The [player-prop engine technical plan](player-props-engine-technical-plan-2026-09-26.md)
documents the separate future workstream: player-data qualification,
participation/minutes and count models, exact prices, settlement and staged
release. The owner has prioritized continuing team Phase 3 before implementing
that engine. The player roadmap does not change the frozen team dataset,
reserved tests or current production behavior.

This plan is complete when the input defects are corrected, published picks
settle automatically, candidate comparisons are reproducible, probabilities and
selection policies have been evaluated chronologically, and activation/rollback
work without disrupting existing delivery. If the corrected ML models still do
not improve the baseline, keeping them inactive is a valid documented outcome.

## 30 September 2026 — Current player/card evidence collection and next steps

This dated addition records the owner's latest agreed work. It supplements the
earlier phase descriptions and stale checkboxes above; it does not restart Phase
3 or imply that Phase 4, card qualification or the player-props engine is finished.

### What we are doing now

The owner approved collecting complete fixture-level **player statistics,
lineups and match events** for all 13 competitions, for provider seasons
**2021–2026**: five completed seasons, 2021/22–2025/26, plus completed fixtures
from the current 2026/27 season. The purpose is to recover the source evidence
needed to finish cards and retain the foundation for the later player-props
engine in one collection. All fields returned by these endpoints are retained.

The read-only inventory contains **23,263 completed fixtures**, including
**1,007 in provider season 2026**. Three endpoint requests per fixture imply
**69,789 base requests**, before retries or reuse of this job's saved responses.
These counts describe stored fixtures, not certified complete provider seasons.
Unplayed fixtures are excluded. Collection includes recorded European knockout/
qualifying rounds and domestic playoffs for future use; those stages remain
outside the currently approved card-reconstruction scope.

Implementation: [player_history_download.py](../Scripts/ops/player_history_download.py).
Operational details: [collection and recovery instructions](player-history-download-2026-09-30.md).

```bash
cd /Users/sanduandrei/Desktop/Betting_RAG
PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -B -m Scripts.ops.player_history_download \
  --execute --seasons 2021 2022 2023 2024 2025 2026 \
  --endpoints players lineups events \
  --daily-budget 75000 --reserve 0 --rpm 420
```

The command writes only to
`Index/history_staging/player-stats-2021-2026/`. Its `platform.db` uses the
existing platform schema plus staging-only lineup/event/progress tables. The
working database remains **`Index/platform.db`**. No active models, public card
delivery modes, settlement policies, production profiles or baseline archives
are changed by collection. It performs no embeddings or OpenAI requests.

The downloader retains:

- Exact provider fixture/team/player IDs, provider season, saved kickoff,
  competition/round metadata and referee identity. It never joins by display
  team pairing or guesses an ambiguous player from a name.
- Full original provider envelopes, compressed archives, capture timestamps,
  checksums and links from derived rows to their source archive.
- Every returned player, including zero/one-minute appearances and bench
  evidence. The two-minute card convention is not a destructive collection
  filter and is not a universal player-prop eligibility rule.
- Lineup starter/bench membership, formations, coaching and positional fields;
  event order, elapsed/extra time, participant IDs and original event details.
  It does not infer appearance, DNP, card totals or regulation scope from these
  fields during collection.
- Separate checkpoints for every fixture and endpoint. One successful endpoint
  is not re-requested because a different endpoint failed. Request attempts,
  including retries, share a durable 75,000-request UTC daily job budget. The
  owner approved using the full daily allowance with zero reserve; collection
  still stops when the provider reports no remaining daily requests.
- Following the owner's Ultra-plan confirmation, the default and maximum
  configured rate is 420 requests/minute, evenly spaced at no more than seven
  starts per second. Lower provider headers and 429 backoff remain enforced.
  Pacing is measured between starts, avoiding an unnecessary additional delay
  after each response. The 75,000-request daily budget remains independent of
  this speed setting; 69,789 requests have a theoretical pacing floor of about
  2 hours 46 minutes, not a measured completion estimate. Network latency and
  sequential archival/normalization add time. No live requests were made to
  validate speed or the account's current quota.
  The rate update passed 111 offline collector/synchronization/refresh-safety
  tests, including pacing, lower provider limits, retries and saved progress.
- A response recovery file before database archival, atomic archive writes,
  archive verification on resume, and offline replay of normalization. Parser
  repairs can use retained responses rather than requiring a fresh bulk fetch.

Rerun the same download command after interruption or after quota becomes
available. It exits at its daily limit rather than automatically waiting until
tomorrow. The 69,789 base requests fit in a fresh 75,000-request allowance;
retries and prior/parallel account usage can still require a later quota day.
Routine jobs share the account quota; this is not an atomic
reservation of the whole account's remaining requests. When more current-season
fixtures finish and are stored locally, `--extend-scope` can append them without
discarding prior checkpoints. Earlier player-only staging can also be extended;
the operational instructions give the exact command.

**Implementation status:** the script and offline validation are complete; no
paid provider download was started by the coding agent. Full-scope metadata
initialization and synthetic endpoint/recovery tests do not certify real provider
coverage. A fixture marked `ready` has structurally validated linked evidence;
it is not automatically card-qualified, statistically complete or imported.
Empty player/lineup responses remain `unavailable`; empty timelines remain
`empty_unverified`; events with unresolved identity/timing or duplicates remain
`partial`; malformed or conflicting identities remain `invalid`. None is
silently converted to zero or certified complete.

**Second-review evidence:** 105 collector/synchronization/refresh-safety tests
passed in 9.16 seconds. A temporary database initialized all 23,263 fixtures,
13 competitions, 78 league-season combinations and 69,789 endpoint checkpoints;
full identity verification passed with zero foreign-key errors. This check read
fixture metadata only and made zero provider requests. Review fixes include
archive verification before resume, durable response/archival recovery, local
parser replay, and detection of missing databases/checkpoints before another
download could be triggered. Actual provider completeness and card/player target
qualification remain open until collection and reconciliation are performed.

### What must happen after collection, in order

1. **Verify completion and preserve the collected evidence.** Inspect
   `summary.json` and endpoint progress, distinguishing actual failures from
   provider coverage gaps. Verify checksums, exact fixture scope, archive links
   and database foreign keys. Resolve failed requests selectively, not by
   restarting the entire download. Use `--verify-saved` for local verification
   and `--reprocess-saved` for local parser replay; neither makes API requests.
   Preserve a versioned, checksummed collection snapshot before canonical import.

2. **Qualify and reconcile the source data.** Report availability by league,
   season, stage and endpoint; check roster completeness, duplicate IDs,
   minutes, substitution/lineup consistency and disagreements between player
   aggregates and events. Keep explicit zeros, nulls, omitted fields, missing
   responses and contradictions distinguishable. Reconcile card/second-yellow
   semantics using sufficient evidence; an absent event does not establish zero
   when timeline completeness is unproven. Validate each player-statistic field's
   meaning before using it as a target, including ambiguous passing-accuracy
   representations. Source-quality checks may cover the full inventory, but
   numerical outcome inspection and model decisions must obey the frozen
   development/test boundaries. Collection is not permission to inspect reserved
   performance or tune rules against protected outcomes.

3. **Prepare and apply a controlled canonical import.** Take a consistent backup,
   use the normal writer gate, and compare incoming evidence against existing
   records. Map fixture/team/player identities by provider IDs; map staging
   player IDs and archive IDs into canonical IDs and rewrite their provenance
   references accordingly. Preserve prior values and revisions; a new null or
   partial roster must not overwrite better evidence. Adapt and test archive
   readers for complete envelopes as well as older response-list archives.
   Preserve canonical fixture scores and metadata: staging contains identity
   snapshots, not replacements for every fixture column. Verify transaction
   rollback, row counts, source hashes and team/player joins after import. Never
   delete a refresh lock or incomplete marker to force this through.

4. **Reconstruct and qualify historical cards.** Use the agreed research v2
   convention consistently: recorded minutes **>= 2**, yellow = 1, straight
   red = 2, and two yellows with dismissal = 3 total. Reconcile player totals,
   events and participation evidence without manufacturing missing statistics.
   Use ordinary domestic league rounds, approved integral league splits, and
   European group/league phases. European knockouts, qualifiers, separate
   playoffs and ambiguous periods remain deferred. Report qualified and excluded
   cohorts explicitly; a successful download alone cannot close this gate.

5. **Rebuild dated team and referee inputs.** Construct historical team/opponent
   and referee card tendencies from the same qualified target definition. Every
   historical forecast must use only earlier eligible matches and respect label
   availability. Preserve referee-name uncertainty, missingness and sparse-support
   fallbacks. Do not silently replace production profiles or enlarge the active
   history window just because more seasons have been collected.

6. **Run the agreed fixed card backtest, then review.** Use the permitted fitting
   and development cohorts, identical eligible evaluation fixtures for paired
   comparisons, and the documented reference with and without referee context.
   Label any reconstruction/simplification honestly. Report probability scores,
   calibration/reliability, uncertainty, coverage and supported league/stage
   slices. Insufficient evidence is a blocked/inconclusive scope, not a negative
   result for the whole card market. Keep the final-system test and prospective
   reserve protected; do not manufacture historical prices or claim ROI from
   untimestamped/non-equivalent quotes.

7. **Stop before weight optimization and review with the owner.** The current
   authorization covers card reconstruction and its fixed backtest. It does not
   authorize an unbounded weight search or promotion. After the owner reviews the
   results, agree on the bounded development-only optimization, including eligible
   cards and the other agreed markets. Phase 5 selection-policy work follows the
   relevant probability/scope review; data collection is not Phase 5 completion.

8. **Prepare the later player-props workstream from these same sources.** Create
   separate eligibility for participation, playing-time targets and each count
   market. Preserve short appearances and nonappearance evidence; do not apply
   the two-minute card filter globally. Start with the documented participation/
   minutes and player shots/SoT baselines, then qualify goals and additional
   markets individually. Historical final lineups are reconstructed starter
   scenarios, not proof of original pre-match availability. Authentic player
   quotes, their prospective capture, exact settlement rules, calibration, shadow
   evidence and public integration remain separate work in the
   [player-prop engine plan](player-props-engine-technical-plan-2026-09-26.md).

### Production release boundary

Card reconstruction, profile alignment and the new engine must be promoted
together only after qualification and explicit owner approval, with a tested
rollback target. Production retains `spix-participation-settlement.v1` until that
release; existing publications keep their saved policy and grades. New research
uses v2 without retroactively reinterpreting v1 recommendations. Downloading or
importing history does not by itself establish improved prediction accuracy,
bookmaker-equivalent settlement, profitability or live matchday acceptance.

### Collection completion — 30 September 2026

The owner completed all 69,789 planned fixture/endpoint responses using 69,790
HTTP attempts. A subsequent read-only check verified every archive checksum,
frozen identity and source reference; SQLite integrity and foreign-key checks
passed. The staging data contains 886,553 player-statistic rows. All three
endpoints are structurally ready for 20,331 of the 23,263 fixtures, without yet
certifying card eligibility. No canonical import or model evaluation ran.

Remaining source review includes 1,807 empty player responses (1,800 in European
competitions), 75 invalid player responses, 30 invalid lineups, 777 event
responses rejected for negative elapsed time, 335 partial timelines, five empty
timelines and two empty lineup responses. Original responses are retained;
negative event times require interpretation, not automatic conversion to zero.
The next work remains source reconciliation, controlled import, card
reconstruction and the fixed backtest, followed by owner review before weight
optimization. See [the completed collection report](player-history-download-2026-09-30.md#completed-owner-run-collection--30-september-2026).

### 2 October 2026 — First four post-download steps completed

The owner authorized source review, justified parser repair, controlled canonical
import and historical card reconstruction. These steps are now complete:

- Verified and preserved all 69,789 original responses in a separate sealed
  preparation. The downloaded original database is unchanged. Negative event
  times are retained as uncertain evidence; 777 timelines were recovered as
  partial records without inventing timestamps. Other identity/roster gaps remain
  quarantined, and provider nulls remain unknown.
- Rehearsed and applied the additive import under the normal writer gate with
  consistent backups. Added 636,553 canonical player-statistic rows and retained
  all 886,553 incoming observations with canonical source mappings. All old rows
  across the 38 original tables were preserved; 7,515 differing incoming records
  are stored separately. Canonical integrity, foreign keys and source identities
  passed verification. Production `fixture_player_stats` now has 888,027 rows.
- Reconstructed only permitted pre-2024 development card targets under v2:
  9,325 qualified fixtures, 1,367 deferred special rounds and 631 other unresolved
  fixtures. Later target outcomes remain protected. Qualified evidence includes
  7,773 matches without a recorded eligible-player red and 274 zero-card matches;
  this is source qualification, not a backtest or accuracy claim.
- Passed 312 focused/regression tests, including transactional rollback,
  preservation, ID remapping, card arithmetic and protected-period checks.

The refresh which initially held the gate completed normally before the import;
no refresh lock or incomplete marker was bypassed. No new API calls, profile
builds, embeddings, backtests, optimization or production policy/model changes
were performed. Research remains v2; published settlement retains v1 until the
agreed joint release. Historical labels retain the `assumed_final` limitation.

See [the reconciliation/import report](player-history-download-2026-09-30.md#2-october-2026--reconciliation-canonical-import-and-card-reconstruction-complete)
for artifact, source, rollback and exclusion details. The next work is the
previously agreed historical team/referee inputs and fixed card backtest;
weight optimization still requires the owner's subsequent review.

### 2 October 2026 — Historical card inputs and fixed backtest completed

The owner authorized the next two post-download steps: dated team/opponent,
league and referee histories, and the previously agreed fixed card comparison.
These are complete in the isolated
`Research/phase4-card-fixed-backtest-2026-10-02/` experiment.

- Used only the 9,325 source-qualified pre-2024 v2 targets. The 2022 support
  cohort has 3,452 fixtures / 42 ISO weeks; no model parameters were fitted.
- Built 3,846 dated input snapshots for qualified 2023 fixtures. Every contributing
  result is from the same competition and current/previous provider season, with
  assumed label availability strictly before the forecast's UTC day.
- Scored 2,771 paired forecasts / 35 ISO weeks across all ten domestic leagues.
  All ten meet the existing per-league support minimum. The eight-current-season
  history requirement excluded 1,075 targets, including all 264 eligible European
  group-stage fixtures; those competitions remain unassessed by this reference.
- The fixed referee adjustment improved pooled count NLL from 2.229656 to
  2.220334 (0.42%). Its paired weekly-bootstrap difference is -0.009322, with a
  95% interval of [-0.012936, -0.005704]. Secondary scores also improved modestly.
  This is a development diagnostic, not production qualification. League results
  vary: EPL is effectively flat and Ligue1 has a small uncertain regression.
- Retained exact referee identities and neutral sparse-history fallbacks. Saved
  potential alias fragmentation without merging different people. Every target
  exclusion and every forecast-support exclusion remains traceable.
- Replayed all six result files exactly. Independently checked 1,923,646 historical
  fixture links for chronological, team, season, competition and referee accuracy.
  Original source seals remain intact; all 86 focused tests passed.

The reference still overpredicts weighted counts by about 0.23 per fixture and
overestimates some high over-4.5 probabilities. Its fixed Poisson distributions
and incomplete historical context have not been qualified for production. This
is the agreed aligned reference, not the full frozen production engine. The
`assumed_final` evidence and unknown original referee-assignment timestamps remain
explicit limitations. Card v2 counts are not established bookmaker equivalence.

No API requests, database changes, production profile updates, public predictions,
later-period evaluation, optimization or promotion occurred. Production settlement
remains v1. **Next: owner review, then agreement on bounded weight optimization;
do not start it automatically.** The fixed backtest does not finish player props,
European knockout handling, card calibration or a production release.

See [the fixed card report and replay command](phase4-card-fixed-backtest-2026-10-02.md)
for scores, all league intervals, coverage, artifacts and remaining limitations.

### 2 October 2026 — Bounded card improvements and reused later check completed

After reviewing the fixed backtest, the owner approved the proposed bounded
card improvements. This superseded the preceding stop-before-optimization
instruction for this experiment. The owner also explicitly approved reusing
January–June 2025 for a frozen card check, acknowledging prior assessment of
other markets there. It is not an untouched qualification period.

- Implemented the fuller aligned frozen-engine control with dated v2 profiles,
  foul context and referee inputs; retained exact parity with the original
  2,771 fixed-reference forecasts. Compared a bounded registry of team/opponent,
  venue, recent-form, pooling, calendar-decay and referee changes, plus a foul
  ablation, fitted dispersion and coherent count-distribution calibration.
- Earlier training selected pooling strength 16 with NB2 alpha 0.05. Existing
  other weights and referee adjustments were retained. No extra combination
  earned its inclusion. Conditional historical-lineup effects and expanded
  early-season/European coverage remained separate scenarios.
- On 1,730 April–December 2023 fixtures / 23 weeks, the calibrated candidate
  reduced count NLL by 1.079% against the simple fixed reference and 0.898%
  against the fuller observed-variance control. Both pooled development gates
  passed; individual leagues lacked the declared development support.
- Froze methods and calibration before admitting additional 2024 history and
  January–June 2025 targets. On 1,751 later fixtures / 21 weeks, NLL improved
  1.137% against the fixed reference but only 0.295% against the fuller control.
  The latter difference's 95% interval includes zero. **Overall qualification
  failed:** the candidate did not meet the required improvement and uncertainty
  limits against both controls. No post-check retuning or favourable-league
  selection was performed.
- All 116 focused/regression tests passed. Both experiments replayed exactly;
  independent audits checked 4,298,472 historical fixture links and normalized
  count/Asian distributions. A preparation-only checksum/guard failure was
  repaired and regression-tested; its partial artifact and first access record
  remain preserved alongside the successful replacement.

No provider requests, active database/profile/model changes, publication changes
or promotion occurred. Research uses v2; production settlement retains v1.
July 2025–June 2026 final-system outcomes and the July 2026+ prospective reserve
remain protected. The new candidate remains isolated and unqualified for
promotion; these results do not exclude cards as a market or establish bookmaker
equivalence/profitability. Any later release retains the agreed joint
target/profile/engine boundary and requires prospective checks and approval.

See [the complete card improvement results](phase4-card-improvements-results-2026-10-02.md)
for methods, frozen protocols, comparison tables, limitations, artifacts and
replay commands. The bounded card experiment is complete; it does not imply
completion of other-market weight optimization, Phase 5 or production release.

### 2 October 2026 — Qualified corner/SoT components retained for next research

The owner approved carrying the successful corner and shots-on-target
distributions forward before the remaining weight work and Phase 5. The sealed
`Research/phase4-retained-components-2026-10-02/` bundle now contains **corner
alpha 0.025 and SoT alpha 0.01 with unchanged frozen control means**. Goals retain
the coherent frozen control and calibration remains identity. The rejected goal
strengths, inconclusive corner sigmoid and unqualified card candidate are not
activated by this bundle.

The original five-league, >=8-current-match pooled research scope is preserved,
with exact outside-scope control fallbacks and explicit stage/missingness
limitations. It is not blanket qualification for all forecast stages, leagues or
lines. Changed means or weights require a new configuration and validation; the
earlier distribution result cannot automatically qualify those combinations.

All 3,095 saved development reference forecasts replay exactly, and 2,136 saved
qualification count distributions match exactly, including fallbacks. The
eligibility sidecar retains per-market exclusions and fixture/snapshot identities
for later scoring. All 84 focused tests passed; 458 existing source/test/model/
dependency files were preserved. No fitting, new outcome scoring, API calls,
database/profile changes, reserved-outcome access or public activation occurred.

This is the retained research configuration for subsequent comparisons, alongside
the unchanged original frozen control. It does not start weight optimization or
Phase 5 and does not complete production promotion. See the
[carryover report and usage contract](phase4-retained-components-2026-10-02.md).

### 2 October 2026 — Bounded complete weight-setup comparison completed

The owner authorized testing different setups and explicitly required consultation
before any implementation decision. The new research-only runner compared a
frozen registry of 60 configurations per market, including references, across
goals, corners, SoT and cards. Individual changes and sampled complete combinations
covered own/opponent, venue, recent/season, recent windows/decay, profile priors and
pooling, plus goal xG and card referee/foul weights. Distributions remained fixed;
corner/SoT references use the retained alpha 0.025/0.01 components.

- Locked one setup per market using 2022 before 2023 inference. Statistical
  markets retained the five-league >=8-match scope; cards used the qualified ten
  domestic leagues. Later development contained 1,533 goals, 1,504 corners,
  1,521 SoT and 1,730 cards, with identical fixtures across each market's setups.
- Preselected setups reduced development NLL by 0.289% for goals, 0.341% for
  corners, 0.025% for SoT and 0.119% for cards. Goals/corners passed uncertainty
  and multiplicity checks and ranked first in both tuning and later development.
  **None passed the frozen 0.5% minimum-improvement screen.** SoT/card intervals
  also include zero. No thresholds were lowered or later winners substituted.
- No supported primary slice or related-market regression exceeded 2%; sparse
  league/stage/missingness slices remain explicitly unsupported. The earlier
  calibrated card candidate remains a diagnostic with its prior failed later
  qualification unchanged. These probability scores are not ROI or hit-rate gains.
- All 111 focused tests passed. Rebuilt 2,829 input snapshots, exactly replayed
  744,038 forecast calculations and independently recomputed 748,129 likelihoods.
  The successful sealed artifact is
  `Research/phase4-weight-comparison-2026-10-02-v4/`. Earlier timestamp/reporting
  attempts are preserved; final forecasts and locked selections were reused byte
  for byte after reporting-interface repairs, with explicit hash provenance.

No provider requests, database/profile/active-model changes, public activation,
retained-bundle replacement or 2025/later outcome access occurred. The 458 existing
fingerprinted source/test/model/dependency files remain unchanged. No further
optimization or Phase 5 work has started. **Next action is owner review; no new
configuration has been chosen for implementation.** See the
[full setup comparison, results and verification](phase4-weight-comparison-results-2026-10-02.md).

### 2 October 2026 — Focused engine assessment and historical player effects

The owner authorized an assessment against simple forecasts and implementation
of historical player/lineup effects. The completed research experiment is
`Research/phase4-assessment-players-2026-10-02-v2/`; see the
[full assessment, source repair and player results](phase4-assessment-player-results-2026-10-02.md).

- On identical 2023 fixtures and common dispersion, the reference improves NLL
  over dated league means by 1.760% for joint goals, 0.008% for corners, 0.727%
  for SoT and 0.666% for cards. Corners' interval includes zero. Every comparison
  against a simple equal team/opponent mean has an interval including zero;
  corners is worse by point estimate. Complexity is not established superiority.
- BTTS and corner Over 9.5 log loss do not beat dated empirical league event
  frequencies. Reliability diagnostics show overly extreme probabilities in
  parts of the evaluated scope. These diagnostic fits were not applied, and
  earlier calibration/qualification conclusions were not overwritten.
- The new player adapter reconstructs dated role/player rates and 990-minute
  reference/forecast allocations, fits bounded incremental coefficients on 2022
  only, and applies effects before coherent goal/SoT probability calculation.
  Expected-player and conditional actual-XI forecasts remain distinct; original
  lineup-announcement timestamps are absent. No postmatch minutes enter features.
- A source audit found player goals/SoT counts were positive or null, never
  explicit zero. The preserved first attempt's player results are withheld.
  Exact certified team-total reconciliation establishes 118,495 goal zeroes and
  104,903 SoT zeroes in derived research data. Whole unresolved team/field
  observations are excluded from rate fitting; no blanket null conversion,
  invented allocation, raw-data rewrite or card-policy change occurred.
- Expected-player effects change NLL by +0.0342% goals and −0.0037% SoT; neither
  qualifies. Conditional actual-XI changes are +0.0049% goals and −0.0485% SoT;
  the small latter gain is conditional research only and below the threshold.
  This is not evidence that all richer player/lineup approaches are ineffective.
- All 126 tests pass. All 5,658 player snapshots and 6,108 player forecasts replay
  exactly; four coefficient fits, 13,434,053 historical source links and 73,416
  probability vectors passed validation. The 458 existing fingerprinted files,
  prepared player database and previous experiment seals remain unchanged.

Production weights, profiles/models/databases, retained bundles, delivery and
settlement remain unchanged. No provider calls, 2024+ outcome inspection, Phase 5
work or public activation occurred. Richer attribution, defensive/keeper effects,
injuries, scenario-based minutes and prospective lineup acceptance remain separate
work. Any next experiment should target measured deficiencies, preserve inspected
period status and obtain fresh acceptance evidence before a production decision.

### 6 October 2026 — Phase 5 implementation and prospective transition review

The subsequent Phase 5 research bundles implement the shared selection policy,
bet justification, communication, candidate/rejection logging and portfolio
evaluation. The latest sealed implementation is
`Research/phase5-portfolio-readiness-2026-10-05/` (601 Python tests plus four
subtests and JavaScript checks in its saved validation). This is isolated
research code, not an installed production policy. No authentic historical
quote corpus is known, so the historical policy comparison and a
profitability-qualified freeze remain incomplete.

The owner has chosen UK, bet365 and William Hill, API-Football as the only
price source, and one fixed candidate for prospective comparison without tuning
on evaluation results. The separate prospective odds archive is collecting
provider observations; collection is not proof of executable UK prices or
permission to inspect reserved outcomes.

The [transition review](phase5-shadow-transition-2026-10-06.md) records the four
preparation items, current support, official bookmaker-rule differences and a
bounded protocol proposal. A complete archived cycle reproduced exactly;
515 operational source/artifact identities matched the existing input contract.
The offline paired-record contract passed 26 tests. Earlier bundle seals remain
valid. The inspected core quotes did not meet even the broadest three-hour
source-age check, and the real probability/rule/execution registry remains empty.

An existing default recipe is pinned for review, not selected as a historical
winner. The proposed future dates/duration and acceptance bars require review;
runtime capture, full audit/artifact resolution, exact settlement and price
evidence remain explicit dependencies. The historical evaluator's protected
period checks must not be disabled to run a new prospective experiment.

Phase 6 diagnostic engineering can follow that review without requiring every
market to qualify. Betting qualification and public promotion remain separate.
No new prediction weights, models, retained alphas, public recommendations or
settlement rules were activated. This review opened no protected or prospective
outcomes and made no provider requests or archive-scheduler changes. The earlier
dated reports remain evidence of their own scope; this addition supersedes their
statements that Phase 5 work had not started.

### 6 October 2026 — Phase 6 diagnostic shadow runtime started

The owner approved **26 weeks** for the fixed prospective comparison and
authorized starting Phase 6. A separate hourly, outcome-blind collector is now
implemented, tested, installed and loaded. Its first bounded operational check
runs from 6 October 09:37 UTC to 20 October 09:37 UTC while the Mac/user session
is available. Missed observations are not reconstructed or silently added later.
See the [Phase 6 start and remaining dependencies](phase6-shadow-start-2026-10-06.md)
and `Research/phase6-shadow-2026-10-06/`.

- Both policies consume the same immutable saved canonical forecasts and
  archived bet365/William Hill receipts. Complete candidate/rejection audits,
  forecast/receipt bytes and diagnostic shortlists are retained independently.
  Prediction means, distributions, active models and public delivery are unchanged.
- The frozen diagnostic release binds its source, environment, policy recipe
  and 515 existing operational source/artifact identities. Outcome access,
  public publication and performance-evaluation mode are disabled.
- 625 Python tests plus four subtests and JavaScript checks passed. The first
  real capture processed all three forecast-bearing fixtures within 96 hours;
  93 fixtures further out had no saved forecast yet. The candidate selected zero
  bets: price freshness and probability/execution/settlement evidence gates failed.
  Scheduled restart reused the immutable hourly record and exited successfully.
- **The 26-week performance clock has not started.** Its final supported scope,
  evidence, initial/amendment recording and system freeze remain dependencies.
  Hourly observations are diagnostic snapshots, not independent bets or actual
  first publications. The qualification thresholds remain review proposals.

This release replays the selection layer from saved canonical forecasts; full
upstream profile-input replay remains separate. The diagnostic check will assess
coverage, freshness, missing inputs, failures and storage without inspecting
outcome performance. A supported subset may proceed; every market need not qualify.
No protected/reserved outcomes were opened, no new API requests were made, and
the existing odds collector was not changed. Public promotion, model changes,
controlled reload and rollback remain later Phase 6 work requiring their own
evidence and authorization. The prior review's runtime-not-started status is
superseded only for this bounded diagnostic release.

### 6 October 2026 — Communication and decision-audit release activated locally

The owner authorized releasing communication, recommendation identity and logging
independently of the unqualified selection-policy candidate, and chose this Mac
for website startup. The website now serves the installed
`communication-audit.2026-10-06.v1` release at `http://127.0.0.1:8000`.
See the [release evidence and rollback](phase6-communication-release-2026-10-06.md).

The release adds exact quote display, deterministic explanations, explicit
uncertainty/availability limitations, shared identities and persistent audits of
the existing market selectors. All 133 focused Python tests passed, alongside
actual JavaScript rendering checks and source rollback checks. Actual website
responses verified the loaded source identity. A bounded normal refresh appended
three future Match Reads (553–555), with 21 audited markets, two recommended reads
and one no-bet, through the established website publication path.

Prediction means, weights, distributions, active ML artifacts, current selection
thresholds, compatibility and headline ranking remain unchanged. The Phase 5
justification policy remains research-only; this is not performance qualification.
The Discord adapter is installed/tested, but no bot was started or message sent.
The website is local-only, not a remote or internet-facing deployment.

The shadow collector has a separate frozen source epoch matching the release.
Its initial old-forecast mismatch is retained as a failed scheduled observation;
a subsequent separately labelled fresh-forecast replay passed. Its 20 October
diagnostic deadline and the unstarted 26-week performance window are preserved.
No protected research outcomes were inspected or historical API backfills run.
Ordinary provider calls and production profile reads occurred only through the
bounded upcoming-fixture refresh. Earlier archives and publications remain intact.
This entry supersedes earlier production-not-activated statements only for the
specific communication/audit changes above.
