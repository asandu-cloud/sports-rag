# Phase 4 Step 2: expanded candidate implementation contract v2

Authorized 29 September 2026: implement the proposed team-rate improvements,
history/weighting changes, original distribution candidates and gated lineup
effects. This is an additive amendment to `phase4-evaluation.v1`; neither its
archived text nor the control is replaced. No production promotion is authorized.
Implementation and correctness/smoke testing precede historical model selection.

## Fixed comparison and chronological boundary

Use the sealed Step 1 fitting and development artifacts. Preserve their fixture
IDs, regulation eligibility, one forecast stage, assumed-final availability,
same-day exclusion and common context omissions. Every historical contribution
must have kickoff+3h strictly before its input/fitting cutoff. Preserve zeros and
missingness separately. No 2024+ outcomes, provider requests or live stores.

For subsequent comparisons, 2022 uses quarterly forward folds with earlier
eligible history for initial fitting; 2023 is development comparison, not a new
untouched test. Hyperparameters/combination choices are selected on 2022 only.
Pre-2022 history may initialize the new team-strength fits; this explicitly
clarifies the older protocol's earlier-fitting permission. Parameters learned
from validation outcomes cannot be applied retrospectively. Any deterministic
update of team-strength state during evaluation must use earlier available
results only and be logged, like updates of the control's rolling profiles.
Refit cadence must be fixed in the comparison manifest before scoring.

The frozen proper-score, support and slice-regression gates continue to apply.
V2 explicitly permits competition-specific intercept/home/attack/defence estimates
for the new strength family; the existing probability-parameter grids remain
pooled. The 60-fixture local fit threshold does not replace the 500-fixture/26-week
pooled fitting or subsequent evaluation/qualification support requirements.
The larger family count must be included in multiplicity control at qualification.
Later calibration/qualification/final-system periods remain protected. This
amendment does not activate later-period readers.

## Candidate registry (one component changed at a time)

| Family | Fixed budget | Definition |
|---|---|---|
| Control | one | Saved frozen profiles, means, variances and score distribution |
| Gradual prior | k=16 | Reuse `season-prior-continuous.v1`: k/(n+k), retain profile fields and European gates |
| League pooling | pseudo-count 4/8/16 | Pull each available production/concession rate toward its own league's dated mean; weight from per-field effective support; minimum 20 known league team observations |
| Venue pooling | pseudo-count 4/8/16 | Pull each venue rate toward its team's overall rate according to venue-specific support, then apply the unchanged venue coefficient |
| Attacking xG fallback | one | Retain existing two-venue xG rule when supported; otherwise use the actual observation-weighted overall xG field, without inventing missing values |
| Defensive xG | xGA weight .25/.50/.75 | Blend actual concession rates with dated opponent xG; retain actual-only when xGA unavailable; log known source counts |
| Opponent strength | ridge 1/10/100, each eligible market | Regularized joint Poisson attack/defence model, competition intercept and home effect; mean replaces that market's control mean, not an extra multiplier; current and prior provider seasons only; minimum 60 eligible fixtures and three appearances per predicted team |
| Calendar recency | half-life 30/90/180 days | Reweight current-season recent-six rates by elapsed days; keep season profile and recent-eight variance policy fixed |
| Recent window | 4/8/12 matches | Current-season recent rates with unchanged .85 positional decay; variance window stays eight |
| Mean coefficients | own .50/.70; venue .25/.75; xG .25/.75; recent .10/.25/.40 | Change one coefficient at a time; complementary weights sum to one; own changes apply to season and recent corner/SoT attack-versus-concession calculations |
| Corners dispersion | existing alpha grid | variance=mean+alpha*mean²; compare with actual control variance |
| SoT dispersion | existing alpha grid | Same grid, separate market-specific experiment |
| Goal dependence | rho −.15/−.10/−.05/0/.05 | Poisson marginals and unchanged valid Dixon–Coles arithmetic; no negative-cell clipping |
| Goal alternative | existing alpha grid | Independent negative-binomial marginals (alpha=0 is independent Poisson); rho=0 only |
| Lineup difference | effect strength .25/.50/1 | Dated expected-minutes/contribution differences relative to the reference lineup, with supplied coefficients fitted only before cutoff; no neutral record represented as qualified evidence |

The alpha grid is 0/.01/.025/.05/.10/.20/.40/.80/1.60. No unrestricted Cartesian
search, automatic winning combination, or broad new tree-model search. The
registry records the applicable markets for every candidate. Unchanged control
remains a valid outcome. Subsequent shortlisted combinations need a bounded
manifest declaring constituents before fitting/scoring; incompatible replacements
are rejected rather than silently stacked.

## Numerical and evidence contracts

All mean candidates run through the archived projection/variance functions,
except the explicitly named fitted-strength replacement. Candidate adapters
retain current/prior and domestic/European arithmetic and record component support.
League priors never mix leagues or use later results. Missing fields cannot gain
support from unrelated available fields. Prior/current support uses the actual
mixture weights and the distinct contributing observations; it is not the legacy
displayed support proxy. Unavailable priors remain unavailable.

Fitted strength bundles contain competition/market, cutoffs, exact eligible
fixture IDs and data hash, per-team support, coefficients, ridge/weight policy,
effective fixture sample size and convergence status. Fitting and inference are
separate. A failed/nonconverged fit cannot silently become an executable bundle.
Unknown/sparse teams yield an explicit control fallback and are counted separately.
Season/competition mismatches are rejected; a season change requires an explicit
refit. For the standalone strength variants elapsed-time fitting weights are
uniform; a time-weighted strength bundle cannot masquerade as that variant.
The fit uses `log(mu)=intercept+home_effect+attack+defence`, centred team effects,
and Poisson mean negative log likelihood plus ridge on those effects. The penalty
is ridge divided by twice the sum of observation weights, times squared team
effects. Fixed optimizer bounds are intercept [-10,5], home effect [-2,2] and raw
team effects [-4,4] before centring; L-BFGS-B has 500 iterations, ftol 1e-12 and
gtol 1e-7. Nonconvergence is a failure. These are implementation constraints,
not fitted improvements. Probability-grid ties within 1e-8 retain rho=-.10 when
available, otherwise smaller absolute parameters; final system ties still prefer
the unchanged control under the original protocol.

Mean changes recompute the control's variance/floor and shared goal distribution;
no stale control variance is attached to a changed mean. Distribution-only
candidates retain control means. All goal-related probabilities derive from the
same score PMF; count and joint tails are bounded at 1e-10. Integer and quarter
lines retain full/half wins/losses and pushes. No historical odds or EV are invented.
Numerically invalid candidates are explicit failures, not excluded winning rows.

The lineup contract requires exact fixture/team/player IDs, observed-at and
source-cutoff timestamps before forecast, expected rather than realized minutes,
reference and forecast playing-time allocations, per-player goal/SoT contribution
rates and a pre-cutoff coefficient identity. Full regulation allocations are
990 player-minutes per team in each scenario. Only incremental lineup differences
are applied, avoiding addition of the team's production a second time. Goal
effects rescale the team means coherently; SoT effects adjust the match total.
This is a research interface, not a trained player-contribution/minutes engine.
The sealed Step 1 inputs contain no qualified lineup snapshots, so this family
must report unavailable/control fallback there. No player backfill is implied.

## Implementation acceptance, before model comparison

Require control parity; isolated mutation/no production imports from publishers;
future and held-out result invariance; source ID/cutoff/period validation;
missing-versus-zero and per-field support tests; league separation; deterministic
fitting and JSON bundle replay; synthetic opponent-strength recovery; nonconvergence
failure; finite nonnegative tail-bounded distributions; coherent goal marginals;
Asian complement/quarter arithmetic; truthful fallback and lineup evidence gates.

A small deterministic unscored historical smoke cohort is permitted to check
wiring. It must not select parameters or claim improved scores. Preserve candidate
registry, code/dependency/source hashes, test evidence and readiness limitations
in a new immutable implementation artifact. Actual forward comparisons, selected
combinations and promotion remain subsequent work.
