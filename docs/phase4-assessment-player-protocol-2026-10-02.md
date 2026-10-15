# Focused engine assessment and historical player effects — protocol v1

Owner-authorized research, 2 October 2026. No public activation or replacement
of a retained bundle. Inputs/targets stop strictly before 2024. Later qualification,
final-system and prospective outcomes remain closed. This is additional analysis
on previously inspected development data, never fresh qualification.

## Focused assessment

Use the exact 2023 fixture memberships from the sealed complete-weight comparison.
Compare the retained reference and its already-frozen 2022-selected weight setup
with two simple dated mean baselines: league/venue averages, and an equal blend
of each team's own production and its opponent's conceded production. Use only
the current and preceding provider seasons, earlier UTC dates and kickoff+3h
availability. League/venue averages require 20 observations; team rates require
eight, otherwise use the dated league baseline. No missing-to-zero conversion.

Keep the same distribution parameters across mean comparators: coherent goals
rho=-.10, corners alpha=.025, SoT alpha=.01; cards use the fuller reference's
per-fixture alpha. This is a comparison of mean skill with common dispersion,
not an independent qualification of those simple distribution models.
Also compare fixed half-line and BTTS probabilities with dated empirical league
event frequencies (Beta(1,1) smoothing). No executable quote archive is fabricated.
Report NLL, MAE/RMSE/bias, derived probabilities, reliability, league/stage slices,
weekly paired intervals and benchmark-relative skill. These are diagnostics;
do not search a new winning league or threshold on 2023.

## Historical player implementation

Reuse the immutable prepared player collection. Filter fixture IDs and pre-2024
cutoffs before decoding player or lineup response bodies. Verify archive hashes,
provider IDs and endpoint identities. Limit the new player experiment to the five
major domestic leagues in the existing statistical scope and certified FT history.
Every raw exclusion, missing count and missing response remains recorded. Data is
retrospective assumed_final; original lineup announcement times are unavailable.

Implement two explicit stages:

1. **Expected-player composition:** use only earlier complete team player/lineup
   responses. Expected minutes are a positional-.85 weighted average of the last
   three earlier team minute allocations. Remove a player from this expected
   allocation if a later, still pre-cutoff appearance establishes another team.
   Reference minutes are the average of the current season's earlier qualified
   team allocations. This estimates a change relative to the players already
   represented in the team rate; it does not add player totals to team totals.
2. **Conditional historical actual XI:** use the saved target fixture's starters
   and bench identities only. Never use its minutes, goals, shots, rating, events
   or final result to construct features. Starter minutes use earlier started
   appearances with five role-prior appearances of shrinkage; bench allocation
   uses earlier substitute minutes with a role prior. The record must say
   conditional_actual_xi, announcement_time_unverified. It is excluded from live
   eligibility and cannot be described as a confirmed pre-match backtest.

Each minute allocation sums to 990, each player is bounded to 0–90 and identities
are unique. Prior match allocations normalize observed regulation participation;
this is an explicit full-strength pre-match assumption, not a red-card timing model.
Require three qualified current-season team responses and retain exact control
fallback otherwise. Missing or contradictory current lineups also fall back.

Dated goal and SoT rates use recorded positive minutes and explicit known counts,
with 450-minute shrinkage toward earlier league/position rates (league-wide fallback
for sparse/unknown positions). No player is assigned observed zero for a null count.
Use exact player IDs across earlier teams within the same league/current-prior
seasons; unknown transfers use priors and remain labelled. Reference and forecast
use identical rates, so an unchanged composition has exactly zero effect.

Implement attacking-production effects for goals and SoT first. Defender/keeper
causal attribution, injuries, manager/formation effects, joint expected-lineup
scenario mixtures and new corner/card player modifiers are outside this bounded
implementation; there is no certified attribution supporting them yet. Existing
card context and v2 target boundaries are preserved. This is a reusable player
context component, not completion of the full player-props engine.

## Fitting and evaluation

For each stage and market, fit one nonnegative response coefficient in [0,1]
on eligible 2022 fixtures only, with a fixed penalty 10*coefficient^2. No grid or
hyperparameter search. Require >=500 fixtures/26 observed weeks; fallback is zero
when insufficient. Goals use separate home/away Poisson offset likelihoods; SoT
uses its total Poisson offset likelihood. Features are log ratios of projected
player production to reference production, bounded to +/-log(1.25). Zero remains
an explicit candidate. Goals adjust both team means and retain one coherent score
distribution; SoT adjusts its total using the mean of the two side log ratios.

Freeze all four coefficient bundles before 2023 inference. Fit feature rates and
minutes using only earlier evidence at each forecast. No 2023 coefficient refit,
calibration or new combinations. Full common-cohort scoring includes neutral
fallbacks, with available-only coverage reported separately. Primary player claims
are the two expected-player market comparisons; the two actual-XI comparisons
are conditional diagnostics. Existing 0.5% improvement, negative weekly 95% CI,
>=200 fixtures/20 weeks and supported slice/related-market regression <=2% gates
remain; Holm correction across the two primary claims. Any pass remains development
evidence requiring later prospective verification and explicit release approval.

Save inputs, source fingerprints, per-player support and allocations, fit IDs and
cutoffs, coefficients, forecasts, fallbacks, complete comparison metrics and replay
evidence. Test future/own-match invariance, allocation constraints, null vs zero,
exact IDs, neutral composition, unknown players, guarded confirmed-XI timestamps,
coherent markets, fitting chronology, complete cohort membership and offline I/O.

## Source-contract amendment v2, before inspecting player comparison scores

The first data audit found that all recorded player goals/SoT counts in the
qualified source were positive or null, with no explicit zeroes. Known-only
denominators would therefore be selected on scoring/shooting. Preserve the first
attempt and withhold its player qualification; its simple-baseline assessment is
unaffected. Do not tune the effect model to results from that attempt.

For each qualified team/fixture/field, sum explicitly reported nonnegative player
counts and compare with the certified regulation team total. **Only exact equality
establishes all remaining nonnegative player contributions as zero.** Preserve
reported values, the team total, fixture identity and the derived-zero count.
If the total is missing, inconsistent or has a positive unallocated residual,
exclude the whole team/field observation from player rate estimation, including
its known positive counts. This prevents a positive-only rate sample. It does not
assume what an unresolved residual represents (for example, an own goal), modify
raw/canonical data or generalize to cards. Current fixture statistics still cannot
enter its player features; target actual XI remains conditional identity-only.

Rerun with identical periods, memberships, methods, coefficient bounds, penalty
and acceptance criteria. A market without a dated prior falls back independently;
missing SoT evidence must not exclude otherwise supported goal features.
