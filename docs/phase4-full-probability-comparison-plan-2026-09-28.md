# Phase 4 full probability comparison: implementation plan for review

Prepared 28 September 2026; updated 29 September after the owner authorized
**Step 2 candidate implementation and backtesting**, including the expanded team-rate
and weighting scope. Step 1 and the individual-candidate development comparison are
complete. Step 3 has now been implemented and evaluated on supported development
cohorts. The authorized earlier-forecast extension has resolved the challenger
calibration support shortage and completed that development comparison. The combined
configuration, Step 4 review and amended January–June 2025 Step 5 qualification
are now complete. Corner/SoT count dispersions qualify offline; the goal-strength
change and separate corner binary calibrator do not. Production is unchanged.
Read together with the
[current-engine walkthrough](prediction-engine-walkthrough-2026-09-28.md).

The archived [frozen evaluation protocol](phase4-evaluation-protocol-2026-09-28.md)
and `phase4-repaired-statistical-control.v1` remain unchanged. The
[earlier-forecast amendment](phase4-step3-earlier-forecasts-protocol-2026-09-29.md)
explicitly extends nested fitting history, preserving thresholds and reserve dates.
The later owner-approved [qualification amendment](phase4-qualification-amendment-2026-09-29.md)
uses January–June 2025 for one qualification, preserves the development calibrator,
and explicitly allows 2024 history for fixed-method pre-2025 fits. July 2025–June
2026 remains protected. See the [results](phase4-qualification-results-2026-09-29.md).
This document translates the protocols into the five agreed steps.
The authorized expansion is recorded separately in the
[additive candidate protocol v2](phase4-step2-candidate-protocol-v2-2026-09-29.md).
The archived v1 protocol and control remain intact.

## Objective and scope

Determine whether better-supported outcome distributions and calibration improve
the current statistical engine on the same fixtures. Preserve canonical
MarketResult/Match Read identity, settlement arithmetic, quality/release gates,
no-bet behavior and website/Discord contracts. Changes are isolated candidates
first; no automatic replacement of the production engine or its feature store.

Three distinct layers require separate evidence:

| Layer | Current control | Phase 4 v1 work |
|---|---|---|
| Expected counts | Fixed attack/defense, venue, xG and season/recent coefficients; cutoff prior | Test the existing gradual k=16 prior challenger; retain other mean coefficients |
| Outcome distributions | Fixed goal rho; heuristic count variance and floors | Fit/compare bounded dispersion and goal-dependence families |
| Probability calibration | Unfitted probabilities; heuristic confidence | Compare identity, held-out logistic calibration/comparator and coherent goal temperature |
| Selection/ranking | Configured odds/edge/EV and quality rules | Preserve gates; log probabilities/EV correctly; portfolio policy tuning is Phase 5 |

The table describes the original v1 budget. On 29 September the owner expanded
Step 2 to test opponent-adjusted strengths, sample-aware league/venue pooling,
defensive and partially available attacking xG, and bounded alternatives to the
attack/defense, venue, xG and recent-rate weights and history windows. V2 specifies
the exact grids and evidence gates before comparison. These values are not proven
optimal. Each change is tested separately; there is no unrestricted Cartesian
search. No new XGBoost/CatBoost/LightGBM feature search is implied; the limited
logistic comparator remains Step 3.

## Step 1 — build historical inputs and reproduce the full control

**Completed locally on 29 September:** see the
[historical-control report](phase4-historical-control-2026-09-29.md). It preserves
8,799 development forecasts plus 10,849 separate earlier fitting references,
with all serialized-input replays matching. Historical context omissions remain
explicit. This completion does not execute Steps 2–5 or change production.

Build a research-only snapshot assembler and frozen-source runner. Start with
2022/2023 fixtures and permitted earlier history. Use exact provider fixture/team/
competition/season identities and the existing regulation-time eligibility contract.
Retain fixed fixture grouping, assumed-final availability and source hashes.
Never open mixed history/SQLite/outcome stores indiscriminately to build this step.

Each snapshot must carry fixture/forecast cutoff, forecast stage, current/prior
profile inputs, actual contributing fixture IDs and per-field counts, venue rates,
recent-six rates, recent-eight variance inputs, availability/missingness, source
and target definitions, plus each optional context's availability and provenance.
The probability comparison does not require invented historical bookmaker quotes;
fixed diagnostic lines are labelled diagnostics, not historical market offers.

Preserve the control's actual source behavior inside the safe research cohort:
newest available earlier prior season, current-season recent window, date-only
exclusion, overlapping season/recent views and current variance/fallback rules.
Do not substitute the Phase 3 reconstructed means: its immediate-prior and recent
history policies differ from live source. Where a production fallback cannot be
justified from saved historical evidence, record an exclusion or common omission;
do not manufacture a profile, map a team by guess, or relabel it an exact replay.

Historical context is an explicit boundary. Some live league-context helpers use
latest profiles/standings, referee lookup omits the cutoff and pins a season, and
player profiles lack historical cutoff filters. Therefore the runner must not
invoke live context retrieval. Inject evidenced pre-cutoff context or a named
neutral/unavailable value identically for all systems. If safe reconstruction
changes a computation (for example missing-as-zero regime logic), that is a named
candidate change rather than a silent control repair. First deliver a context
coverage matrix so the scope of the historical comparison is reviewable.

Execute the preserved projection, variance and distribution arithmetic against
those inputs. Save home/away goal rates, full score/count distribution identity,
control variance and its source, model probabilities, five-way Asian profiles,
quality eligibility and every missing component. Test direct frozen-function parity
on the exact saved inputs. Historical output remains a reconstruction with
`assumed_final` data, not the originally delivered recommendation.

**Acceptance:** future outcomes cannot change earlier snapshots; exact cutoff and
fixture-version tests pass; missing versus zero is preserved; repeated snapshots
and outputs have identical identities; IO guard denies live stores/provider calls;
neutral/omitted context is explicit; no unsupported fixture is silently dropped.
Review counts and common-fixture coverage before candidate fitting.

Deliverables: `snapshot-schema.json`, immutable snapshot/target shards, per-fixture
eligibility/exclusions, `context-coverage.json`, control forecast/distribution
shards, replay/parity report, source/config/data manifest. Use a new experiment
directory; never write inside the frozen control.

## Step 2 — compare individual changes separately

**Implementation status, 29 September:** the isolated candidate registry, fitting
primitives, frozen-engine adapter and unscored smoke runner are implemented and
locally tested. See the [implementation report](phase4-step2-candidate-implementation-2026-09-29.md).
That implementation milestone did not itself complete the forward comparison or
select a winning setup.

**Subsequent backtest completion, 29 September:** the owner authorized the scored
run. See the [development results](phase4-step2-backtest-results-2026-09-29.md).
The 2022 forward folds selected settings before the 2023 comparison; 34
family/market claims were evaluated. Opponent-adjusted goals and fitted
corners/SoT dispersion pass the development screen. The other claims do not.
No combined configuration, calibration, reserve qualification or promotion is
claimed. Frozen thresholds and all later-period boundaries remain unchanged.

Retain unchanged control for every market and compare on identical eligible
fixtures. Each candidate changes a declared component only:

| Candidate | Parameter budget already frozen | What it tests |
|---|---|---|
| Gradual prior | k=16 only; current/prior weights from the saved candidate | Whether smoothing the eight-match transition improves full probability scores, including variance effects |
| Corner/SoT dispersion | alpha 0/.01/.025/.05/.1/.2/.4/.8/1.6 | Whether fitted outcome spread improves over the control's actual variance/floor behavior |
| Poisson goals + low-score correction | rho −.15/−.10/−.05/0/.05 | Whether fitted goal dependence improves the coherent score distribution |
| Independent NB goal scores | Shared alpha grid above, rho=0 | Whether overdispersed goal marginals help without inventing an unsupported dependence correction |

The additional v2 candidates cover opponent-adjusted means for goals/corners/SoT,
league and venue pooling, attacking-xG fallback, defensive xG, elapsed-day recent
weighting, recent-window lengths and individual mean coefficients. A richer
lineup delta interface is implemented but remains unavailable on the sealed
historical cohort until timestamped player inputs and pre-cutoff fitted contribution
coefficients qualify. Its synthetic validation is not historical qualification.

Both the baseline and changes must derive goals, BTTS, winner and handicaps from
one joint score distribution. Reject invalid distributions, bound omitted mass
at 1e-10, preserve nonnegative normalized probabilities and line monotonicity.
A parameter failing on a common fixture cannot win by excluding that fixture.
Document fallback/coverage failure instead of clipping negative probabilities.

Select parameters on development only. Use quarterly forward folds in 2022,
then compare on already-inspected 2023. Independently test each change first;
only development-selected combinations can advance. No automatic multiplication
of apparent gains or unsupported league-wide promotion. Preserve unfitted control
as a valid winner.

**Acceptance:** mathematical/settlement invariants and parity tests pass; proper
scores, support, coverage, adverse slices and uncertainty are complete; failures
remain in the accounting. This step may conclude that a candidate does not help.

Deliverables: predeclared candidate registry, immutable parameters, paired
OOF/development forecast shards, score/coverage tables, slice and uncertainty
reports, and a review recommending a frozen finalist or retaining the control.

## Step 3 — add the agreed probability comparator and calibration

**Supported development comparison completed, 29 September:** see the
[Step 3 results](phase4-step3-calibration-results-2026-09-29.md) and
[pre-scoring execution contract](phase4-step3-calibration-protocol-2026-09-29.md).
The BTTS logistic comparator and control corner-9.5 sigmoid pass their binary
development screens. They remain isolated diagnostics; control joint-temperature
selection retains identity. That first run had only 13 challenger validation weeks.
The [authorized extension](phase4-step3-earlier-forecasts-results-2026-09-29.md)
generates earlier candidate forecasts and supplies 43 weeks without borrowing
2023 validation rows. Challenger corner-9.5 sigmoid calibration passes with .862%
lower log loss against its raw challenger; the other three challenger sigmoids
add no qualifying improvement. Challenger goal temperature now selects identity
on sufficient evidence. Binary diagnostics still do not calibrate complete count
distributions. No later reserve was opened, and no comparator or calibrator was
connected to public prediction paths.

Implement the protocol's regularized logistic regression comparator for BTTS and
fixed half-lines (goals2.5, corners9.5, SoT8.5). Features are the declared control
probability logit, mean, applicable home-away goal-rate difference and league
indicators; L2 C=.01/.1/1/10. Fit transforms on training rows only. League indicators
permit a limited learned league effect, while distribution parameters remain
pooled; do not claim league-specific calibration is fitted in v1.

Fit calibrators on genuine out-of-fold predictions, never in-sample fitted scores.
Compare identity with monotone logistic scaling; explicitly constrain monotonicity
rather than assuming ordinary unconstrained logistic regression guarantees it.
Compare the declared goal joint-distribution temperatures .75/.9/1/1.1/1.25/1.5.
After temperature transformation, recheck tail accuracy, normalization and all
related market probabilities. A binary comparator/calibrator is diagnostic for its
own outcome; it cannot be spliced into goal markets while other probabilities
continue to contradict it. Nor does one calibrated half-line qualify count PMFs,
all lines or Asian settlement vectors.

## Step 4 — review performance and supported scope

**Completed, 29 September:** see the
[finalist review](phase4-finalist-review-results-2026-09-29.md). The restricted
combination passes within the five supported domestic leagues after eight current
matches for both teams: .848% lower goal NLL, 1.087% corners and 1.489% SoT.
All complete-cohort and unsupported-slice limitations remain disclosed; the
served-cohort goal gain is only .415%. A research recipe and comparison lock are
preserved. This is neither later qualification nor production promotion.

Report count/joint/binary proper scores, Brier/reliability, calibration slope and
intercept, 80%/95% interval coverage/width, numerical bias/errors, supported
league/season/stage/missingness slices and 2,000 paired week-bootstrap draws.
Store raw and calibrated distributions and version/support/uncertainty metadata.
Existing quality caps remain; high event probability is not confidence in the
estimate. Calibrator training and application remain separate from publishing.

**Acceptance:** unchanged control/identity remains a choice; eligible finalists
meet the frozen ≥0.5% relative primary-score improvement, negative 95% difference
interval and ≤2% supported slice/related-market regression limits. Multiplicity
handling for later qualification remains as frozen. Secondary regressions are
reported. No ROI/CLV without authentic equivalent timestamped prices.

Deliverable: a review of the complete development comparison, supported scopes,
isolated final candidate bundle and fixed reserve-evaluation specification.
This completes the development comparison; it does not consume final tests.

## Step 5 — qualify frozen finalists on the later periods

**Completed, 29 September:** the owner approved the prospective
[January–June qualification amendment](phase4-qualification-amendment-2026-09-29.md)
after the earlier Q2-only metadata block. The original development-fitted
calibrator was preserved, 2024 history reuse was explicitly declared, and five
fixed-method goal-strength fits were frozen at 1 January 2025 before any 2025
outcomes were decoded. No thresholds were lowered or parameters retuned.

The [qualification report](phase4-qualification-results-2026-09-29.md) records
934 goal and 888 corner/SoT active-scope fixtures across 20 weeks. Corners and SoT
count distributions qualify with 1.108%/2.125% lower NLL and Holm-adjusted p=.0020.
The goal-strength change fails; the corner binary sigmoid remains inconclusive.
Retain the existing coherent goal engine and identity calibration. Sparse-stage,
missingness, individual-league precision and line-specific limitations remain
explicit; count-distribution qualification is not a profitability claim.

The one scheduled evaluation and audits are complete, with 551 passing tests and
2,073 exact full-control replays. January–June 2025 is now consumed qualification
evidence. Preserve July 2025–June 2026 for the complete probability plus Phase 5
selection system's final test. Production promotion, shadow/live acceptance,
cards and richer lineups remain separate work.

## Continuing work — cards and richer lineups

Card work proceeds independently: verify original player/result evidence, build
representative weighted-participation targets and earlier training history, then
reconstruct compatible team/referee/player rates. Explicitly repair/retest the KB
card mapping that currently turns nullable team counts into zeros; do not reuse
those derived zeros as certified targets or silently change the frozen control. The 895 provisional 2023 totals
are insufficient; source evidence and earlier fitting coverage are separate needs.
The 1,812-request inventory is not authorized and is not a complete acquisition
plan. Prefer a small, chronological, outcome-independent provider diagnostic after
local evidence checks and separate collection approval. Card fitting joins only
through a new qualified cohort/protocol record; it must not stall other markets.

[Richer lineup effects](phase4-richer-lineup-effects-plan.md) follow the baseline
probability comparison. The initial task is to make expected versus confirmed XI
inputs and the calculation provenance truthful, then evaluate replacement-player
and expected-minutes effects with timestamped inputs. The current canonical goal/
corner/SoT path does not consume the already-computed missing-contributor
multipliers, and card risk is not actually filtered to confirmed starters. Simply
connecting every heuristic is not validation. Add isolated effects, avoid counting
team production twice, preserve shared goal coherence and test paired ablations.
Full player props remain a separate later engine.

**30 September card reconstruction:** the agreed immediate order is local card
evidence reconstruction and the fixed dated team/referee comparison, followed by
a separate owner review before any weight optimisation. The
[local results](phase4-card-reconstruction-results-2026-09-30.md) document 2,101
provisional totals but no qualified targets or earlier fitting cohort. The
comparison implementation is tested; real card scoring and calibration remain
blocked by source evidence. API access is currently unavailable and no collection
was attempted. This does not reopen consumed qualification periods or the
protected final-system test.

The owner's subsequent [v2 amendment](phase4-card-policy-v2-results-2026-09-30.md)
sets recorded minutes >= 2 and limits current card work to ordinary domestic
league phases and European group/league phases. The shared calculation is staged
and tested; production remains v1 until joint promotion with the new engine and
aligned profiles. Known completed league-format metadata now establishes period
scope. Earlier player-response evidence still needs recovery before real scoring.

## Implementation controls for predictable execution

- Keep immutable baseline and inputs; every experimental stage writes a new
  directory with explicit completion manifests and checksums. Failed runs cannot
  overwrite active artifacts or appear complete.
- Start with a small deterministic smoke cohort covering markets, missingness,
  early/late season and competitions before the full development run. Smoke work
  verifies mechanics; it does not choose candidate parameters.
- Use bounded CPU concurrency, forecast/distribution shards and checkpoints;
  report completed/failed fixture counts and timing. Resume only against matching
  code/config/data hashes. Set resource limits before large score-grid generation;
  exceeding them is an explicit candidate failure, not an unbounded worker.
- Keep provider/Chroma/live DB access disabled during numerical fitting. Permit
  only named research inputs. Tests must catch accidental context fallbacks,
  same-day/future-result leakage, held-out preprocessing, fixture duplication,
  probability/tail failures and complete Asian settlement disagreement.
- Do not wire a candidate into the ten-minute publisher or change source/weight
  settings there. Source/data parity and candidate improvements are separate gates.
- Each step ends with concrete artifacts, passed checks, coverage and unresolved
  limitations. Use the agreed review point before moving from implementation
  mechanics to fitting, reserve inspection or public activation.

## Decisions to agree before implementation or reserve use

1. **Scope now, updated 29 September:** Step 1 is complete and the owner authorized
   the expanded Step 2 candidate implementation. The additive v2 amendment records
   that expansion without rewriting v1. Historical forward comparisons follow
   this implementation review and were subsequently authorized and completed for
   individual candidates; Steps 3–5 retain their separate purposes.
2. **Historical context — recommended:** preserve source arithmetic and use only
   evidenced pre-match contexts, with common neutral omissions/exclusions clearly
   labelled. Alternative: require all live context to be reconstructed first,
   which would materially reduce eligible scope and delay comparisons. Neither
   option permits filling missing historical inputs with today's profiles.
3. **Rolling history in reserves — recommended for a prospective clarification:**
   freeze learned parameters/calibrators but let deterministic team rates update
   chronologically from results available before each subsequent forecast,
   including earlier matches in the evaluation period and 2024 historical inputs.
   Keep evaluation outcomes behind a sequential, logged reader; never expose
   future results to the forecaster or use them to retune parameters. Alternative:
   fixed history through2023 for2025 forecasts is reproducible but unrepresentative
   and is not recommended. The frozen protocol did not explicitly distinguish
   parameter fitting from these rolling input updates. Record agreement in an
   additive clarification/version **before reserve access**; do not edit the
   frozen copy or treat this proposal as approval to inspect reserved labels.

Step 1 uses the evidenced-input/common-omission boundary described in item 2.
Item3 can be settled before the
reserve stage without delaying the permitted development snapshot/runner work.
This document does not claim all future technical choices or live acceptance
checks have already been completed. The five steps replace earlier incidental
"Batch 2A/2B/2C" labels; they do not add a new workstream or change the frozen protocol.
