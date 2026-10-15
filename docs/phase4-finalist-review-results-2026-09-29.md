# Phase 4 finalist review and qualification readiness

**Subsequent status:** the Q2-only block below was resolved by the owner-approved
[January–June amendment](phase4-qualification-amendment-2026-09-29.md). The
[completed qualification](phase4-qualification-results-2026-09-29.md) qualifies
corner/SoT dispersions, retains the goal control and finds the corner binary
sigmoid inconclusive. The original sealed Step 4 experiment remains unchanged.
January–June 2025 is now consumed; July 2025–June 2026 stays closed.

**At this report's original completion, Step 4 was complete and Step 5 was blocked
by insufficient reserved-period support.** The following records that earlier state.
The combined candidate has been executed, reviewed and frozen as a research recipe.
No dedicated 2025 calibrator was fitted and no 2025 qualification outcomes were
scored. Production, existing archives and the completed-system final test remain
unchanged. This report does not mark Phase 4 or deployment qualification complete.

## Combined configuration and supported development claim

The [execution protocol](phase4-finalist-review-protocol-2026-09-29.md) fixed two
configurations before this run: an unrestricted component composition to verify
parity, and a restricted configuration for the five adequately sampled domestic
leagues, after both teams have at least eight current-season matches.

The restricted research recipe uses:

- Goals: opponent-adjusted attack/defence, ridge 10; Dixon–Coles rho −.10 and
  temperature 1. All goal totals, BTTS, winner and handicaps use the same matrix.
- Corners: existing expected count with negative-binomial alpha .025.
- SoT: existing expected count with negative-binomial alpha .01.
- EPL, LaLiga, SerieA, Bundesliga and Ligue1, with minimum current-season counts
  >=8 for both teams. Other leagues/stages retain the control. Unknown/sparse
  goal teams also retain control. This is development support, not release approval.
- Corner Over 9.5 sigmoid, C=.01, remains a separate binary diagnostic. It is
  excluded from the complete corner PMF and all other corner lines.

No new parameter search or team-state fitting occurred. The runner executes the
frozen numerical engine and uses the exact saved quarterly strength bundles.
All 27,480 market/configuration records reproduce their reference component scores
exactly. The combined components do not interfere numerically in this reconstruction.
No independence assumption or combined cross-market probability is introduced.
There is no claim that combining components creates an additional gain by itself.

On the fixed active scope, each market spans 33 observed development weeks:

| Market | Fixtures | Control NLL | Candidate NLL | Reduction | Paired 95% difference interval | Screen |
|---|---:|---:|---:|---:|---|---|
| Goals, joint score | 1,533 | 2.917412 | 2.892683 | 0.848% | [-.040512, -.009702] | Pass |
| Corners, total count | 1,504 | 2.641917 | 2.613188 | 1.087% | [-.036819, -.020046] | Pass |
| SoT, total count | 1,521 | 2.562689 | 2.524539 | 1.489% | [-.050651, -.024758] | Pass |

The frozen >=.5% reduction, negative paired 95% interval and <=2% supported-slice
and related-market regression limits are unchanged. These are probability-loss
improvements on reused 2023 development data, not betting returns or final proof.

The complete served cohort retains all original eligible fixtures, including
unchanged-control fallbacks. Its reductions are **.415% goals, .537% corners and
.742% SoT**. The full-cohort goal reduction therefore does **not** meet the .5%
materiality threshold. The goal claim is limited to the predeclared active scope;
it is not an all-13-league claim. Active-scope acceptance was declared before this
run, not chosen after seeing this dilution.

Scope fallbacks cover 1,562 goals, 1,524 corner and 1,516 SoT forecasts. There are
120 additional unknown/sparse-team goal fallbacks inside the active scope; these
remain in its evaluation. No difficult fixture was dropped from a comparison.

## Accuracy, reliability and slice review

Within the active scope, goal MAE improves 1.29360→1.27139, RMSE 1.63554→1.61058
and mean error +.07177→+.04752. Goal interval coverage still over-covers: nominal
80% coverage moves 88.58%→89.37%, and 95% moves 98.17%→98.43%. Better primary NLL
does not establish perfect calibration. These secondary regressions are retained.

Corner and SoT expected counts/MAE/RMSE are unchanged. Their improvement is in
distribution shape. Corner 80% coverage moves 90.36%→85.51%; SoT moves
91.98%→86.06%. Corresponding 95% coverage moves 98.67%→97.41% and 98.55%→96.19%.
Over-coverage remains, particularly for discrete intervals.

All five active league slices have >=200 fixtures and >=20 weeks. Pooled success
does not mean every league improves: EPL goal NLL worsens **.354%**, within the
declared 2% tolerance. The other goal league changes range from a .172% reduction
in Ligue1 to 2.072% in LaLiga. No supported slice or related market exceeds the
regression limit. Complete slice tables, categorical class counts, Brier/RPS,
interval widths and ten-bin reliability diagnostics are in `comparison.json`.

The active partial/missing-xG slice has only 211 fixtures across seven weeks.
It remains insufficient for a standalone missingness-specific claim. The full-xG
slice is supported. This research recipe is not certification of every joint
league/stage/missingness intersection; unresolved slices still require evidence
before any public scope decision. Cards, richer lineups, referee and other omitted
historical contexts retain their existing limitations.

The separate corner Over 9.5 sigmoid passes on 1,504 active-scope fixtures:
.880% lower log loss against the raw dispersion challenger and .738% lower
against the frozen control. Its interval against control is [-.010311, -.000101],
close to zero at the upper end. It remains a narrow development diagnostic,
with no calibrated whole-count distribution or qualification claim.

## Frozen research finalist

Artifact: `Research/phase4-finalists-2026-09-29/`.

`finalists.json` freezes method settings, supported-scope policy, fallbacks,
development fit references, diagnostic roles, input/source identities and primary
claims. `FINALIST_LOCK.json` checksums this recipe and the comparison reports.
Recipe identity:
`3778878a8ce078700548925b93e79d6f13bff01fb59de3fe19c5da165a62fdac`.

Future qualification has four distinct primary claims: joint goals, corner count,
SoT count and the separate corner-9.5 diagnostic. Holm correction remains required.
This recipe is not a model refitted for the 2025 season or a deployable publishing
bundle. The source and period rules for such a refit have not been silently changed.

## Step 5: outcome-free preflight blocks qualification

The saved, checksummed split membership file supplies eligible fixture identities
and dates without exposing outcomes. The original reserve rules require **500
fixtures and 10 observed ISO weeks per market**; league claims require 100/10.

| Period/scope | Goals | Corners | SoT | Observed weeks |
|---|---:|---:|---:|---:|
| Jan–Mar 2025, all leagues | 703 | 665 | 668 | 13 |
| Jan–Mar 2025, finalist leagues, upper bound | 546 | 521 | 518 | 13 |
| Apr–Jun 2025, all leagues | 422 | 400 | 403 | 9 |
| Apr–Jun 2025, finalist leagues, upper bound | 388 | 367 | 370 | 8 |

These are metadata upper bounds; current-stage, feature and binary-class checks
could reduce them further. Every finalist league also falls below the 100-fixture
qualification minimum. **Even all 805 Q2 fixture memberships, including ineligible
matches, span only nine weeks**, from 1 April to 1 June. The eligible qualification
fixtures end 31 May. Recovering excluded labels alone cannot produce a tenth
observed week in this saved period.

This is a mismatch between the chosen qualification quarter and the frozen sample
rules. The coverage check should have occurred when the protocol was frozen.
It is distinct from the resolved 13→43-week 2022 calibration-development issue,
and it says nothing about whether the candidate would perform well on Q2 outcomes.

The gate therefore stops before opening either calibration or qualification
outcomes. Dedicated calibrator fitting, later-period comparison and Holm testing
have **not run**. No performance scores are invented and no insufficient period
is treated as a pass. There is no failed model-performance result for this reserve.

There is also a preparation detail to settle before a future authorized reserve
run: the current/prior-provider-season strength fitter rejects bundles from a
different season, while the frozen protocol excludes 2024 fitting labels.
Constructing 2025 inputs/refits needs an explicit dated-input policy; simply
relabelling a 2023 bundle or silently using 2024 training targets is invalid.
The present implementation provides the development composition, finalist freeze
and reserve preflight; it does not claim a completed 2025 input/refit pipeline.

## Validation, preservation and next decision

**537 tests passed**, including 14 composition/scope/qualification-gate tests.
The independent audit verifies 9,285 shared configuration identities and replays
27,480 complete probability-score records exactly. Fixture memberships, inactive
scope fallbacks, market-specific fit provenance, Asian normalization, line
monotonicity, finalist locks and reserve support counts all pass.

Production sources/models/dependencies are checked against the frozen baseline.
No provider calls, database writes, public test recommendations or delivery-mode
changes occurred. The first unsealed `...finalists-2026-09-29-provenance-review/`
run is preserved as superseded before aggregate review; it exposed a metadata
attribution issue, corrected and tested before the final run. Its numerical
predictions matched; it is not the final research artifact.

To continue Step 5, agree a new qualification design **before outcome inspection**,
resolve the dated-input/refit policy, and preserve all previous inspection records.
Keeping the existing sample rules would require a different, sufficiently sized
unseen qualification cohort, potentially collected prospectively. More older
training seasons do not enlarge April–June 2025's observed weeks. This task does
not approve new dates, relaxed thresholds or access to another reserve.

The July 2025–June 2026 completed probability-and-selection final test remains
protected for Phase 5. Shadow evidence and explicit promotion approval still follow
successful offline qualification. Step 5 is pending a valid design, not complete.
