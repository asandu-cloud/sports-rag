# Richer lineup effects for team-market predictions

Added to the Phase 4 roadmap at the owner's request, 28 September 2026.
Status: partially implemented in isolated research; the 2 October update below
records delivered work and the broader remaining scope.
This extends the existing engine and is separate from the full
[player-props engine](player-props-engine-technical-plan-2026-09-26.md).

**Update, 2 October:** historical attacking player-composition effects for goals
and SoT are now implemented and backtested in an isolated research path, including
dated player/position rates, expected-minute allocations, incremental reference
differences, frozen 2022 coefficient fits and coherent probability propagation.
Player null counts are resolved only with exact certified team-total evidence;
unresolved team/field observations remain excluded. Expected-player forecasts
are separate from conditional actual-XI scenarios because announcement timestamps
are unavailable. Neither primary expected-player claim qualified for production.
See the [assessment and implementation results](phase4-assessment-player-results-2026-10-02.md).
The broader attribution, defensive/goalkeeper, uncertainty, prospective-lineup
and public-release work below remains outstanding; this update does not complete
the full lineup or player-props roadmap.

**Update, 29 September:** Step 2 now implements a research-only incremental lineup
interface for goals and SoT, with exact identities, dated evidence, expected-minute
allocations and pre-cutoff contribution coefficients. Synthetic tests exercise
nonzero effects and rejection/fallback cases. See the
[candidate implementation report](phase4-step2-candidate-implementation-2026-09-29.md).
The sealed historical inputs have no qualified lineup snapshots, so their lineup
candidates fall back explicitly. Historical evidence qualification, contribution
and minutes fitting, scenario uncertainty and public wiring remain planned; this
interface does not complete the richer lineup engine described below.

## Existing foundation

`Scripts/rag_ingest/lineup_context.py` already verifies starting-XI team identity,
obtains contribution shares, applies dampened missing-contributor adjustments for
goals/corners/cards/SoT and builds card-risk profiles. `core/projections.py` already
uses lineup card-risk context. Canonical predictions support forecast context and
Match Reads support versioned amendments. These capabilities are not equivalent
to validated replacement-player effects, expected-minutes models or a full
formation/manager-history model. Inventory exact call paths and duplicated
adjustments before changing any weights.

## Wiring clarification from the subsequent source walkthrough

The [current-engine audit](prediction-engine-walkthrough-2026-09-28.md) distinguishes
calculation from application: `compute_lineup_adjustments` builds goals/SoT/cards multipliers
with a neutral corners field in the ordinary contribution lookup, but canonical projections do not consume `home_adjustment`
or `away_adjustment`. Only the card-risk context is connected numerically there.
The card-risk function selects likely starters from playing-time profiles instead
of filtering to the verified XI; a `confirmed_xi` context label currently does not
prove that XI supplied the risk total. The implementation must make this provenance
accurate and explicitly wire/test any newly qualified effects. These are planned
repairs/extensions, not improvements delivered by the existing lineage support.
Historical player profiles also require cutoff-safe snapshots rather than today's
team/league lookup. The frozen control and protocol are unchanged by this note.

## Order and deliverables

1. **Freeze the comparison baseline and an input contract.** Finish the current
   probability correctness, card qualification and calibration baseline work.
   Richer lineup research must not hold those tasks open. Define a versioned
   fixture/team/player-ID contract containing forecast cutoff, forecast stage,
   source observation time, expected/confirmed XI, bench, role, availability,
   expected minutes, uncertainty and lineup revision. Keep expected and confirmed
   status explicit. Corrected/retracted lineups create a new immutable snapshot.
   Missing or contradictory lineups use a documented neutral fallback.

2. **Audit historical availability.** Inventory saved lineup responses and player
   histories, roster membership, transfers, starts/substitutions and observation
   timestamps by league/season. A final lineup downloaded after a match is not
   proof it was known at the forecast cutoff. Separate assumed historical
   reconstruction from timestamped prospective evidence. Exclude postmatch
   minutes, future seasonal totals and later transfer/availability information
   from pre-match features. Evaluate early and confirmed-lineup stages separately.
   Propose bounded collection only where needed; do not start bulk player fetches.

3. **Build player contribution and replacement features.** Fit training-only,
   opponent/league/role-adjusted per-minute rates with shrinkage for sparse players.
   Estimate expected starter and substitute minutes, including plausible bench
   replacements and the opportunity for displaced players to appear later. Compare
   the actual/expected lineup to a reference lineup implicit in the team's recent
   statistical rate; model the difference, not the sum of all player season rates.
   This avoids counting the same team production twice. Preserve missingness,
   cold-start/transfer fallbacks and sample support. Learn effect strength on
   development data; no arbitrary guaranteed penalty for every missing scorer.

4. **Propagate effects consistently.** Estimate bounded, testable changes to team
   attacking/defending goal rates, shot production and finishing/on-target rates;
   investigate defensive/goalkeeper effects separately where supported. Derive
   BTTS, winner and handicaps from the same adjusted joint score distribution.
   Evaluate card adjustments only with the approved weighted participation target,
   compatible player rates and referee/opponent context; prevent duplicate use of
   existing card-risk modifiers. Treat corners as a separate ablation because
   individual absences need not affect them like goals. Keep no-effect as a valid
   fitted result. Formation, manager and tactical interactions require certified
   dated inputs and an explicit later experiment, not an assumed initial benefit.

5. **Represent uncertainty.** Expected lineups/minutes should yield an explicit
   scenario mixture or another validated uncertainty treatment. Probabilities are
   a weighted mixture of scenario distributions; do not multiply correlated
   player effects or same-game outcomes as independent events. Record scenario
   weights, missing players, effect size, support and reasons for neutral fallback.
   Confidence in the estimate remains separate from event probability and value.

6. **Run paired chronological ablations.** Compare the fixed engine, current
   absence heuristics, richer minutes/replacement effects and component removals
   on identical eligible fixtures and forecast stages. Use training-only fitting,
   out-of-sample calibration, proper scores, reliability, interval coverage,
   fixture/week-aware uncertainty and league/missingness slices. Report forecast
   coverage changes separately. Predeclare minimum samples, maximum slice
   regressions and complexity limits before scoring. Use permitted development
   periods first. If the final-system test has been consumed by then, reserve a
   new later evaluation period: do not reuse it as an untouched test. Evaluate
   betting performance only with equivalent, timestamped executable quotes.

7. **Prospective shadow and controlled release.** Save both systems against the
   same immutable snapshots. Test delayed/partial lineups, transfers, unavailable
   players, replacements, late corrections, kickoff changes, retries and restart.
   Preserve publication identity; lineup amendments never replace first-publication
   performance. New results must carry lineup version alongside model/calibration
   versions. Require supported-scope improvement, no harmful supported slices,
   traceable inputs, consistent reload and rollback, and explicit approval before
   activating public effects. An inconclusive result retains the simpler engine.

## Tests and completion evidence

Test exact identities, simultaneous fixture/version grouping, cutoff enforcement,
future-result invariance, expected-minutes constraints, neutral fallback, sparse
player shrinkage, no double application of effects, coherent normalized goal
probabilities, full Asian outcomes, deterministic replay, correction/retraction
handling and cross-surface recommendation identity. Preserve manifests, feature
schemas, protocol, hashes, candidate parameters, paired predictions, uncertainty,
slice metrics and an explicit review of limitations. This work can supply reusable
minutes/contribution components to player props later; team-market improvements
alone do not establish a reliable player-props engine.
