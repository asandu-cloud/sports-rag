# Focused assessment and historical player effects — 2 October 2026

**The engine has useful signal, but this assessment does not show that its
additional complexity consistently beats a simple team/opponent average.**
Corners and BTTS are clear areas for further work. Historical player effects are
now implemented and backtested as isolated research components; this first
attacking-production specification does not qualify for production.

The owner authorized the assessment and player/lineup implementation. The
[protocol and source-contract amendment](phase4-assessment-player-protocol-2026-10-02.md)
were preserved. Production weights, profiles, databases, delivery modes and the
retained research bundle remain unchanged. No provider requests or 2024+ outcome
access occurred. These are additional diagnostics on inspected development data.

## 1. The current research reference versus simple forecasts

All methods use identical 2023 fixtures: 1,533 goals, 1,504 corners, 1,521 SoT and
1,730 weighted cards. The first three cover the five major domestic leagues with
both teams at >=8 current-season matches; cards use the existing ten-league
qualified April–December cohort. Goal distributions are coherent Dixon–Coles;
corners/SoT use the retained alpha .025/.01; cards retain the fuller reference's
per-fixture alpha. Thus the comparison isolates the means under common dispersion.
This is the research reference with documented context omissions, not replay of
every live published recommendation.

The league baseline uses earlier current/prior-season league home/away averages.
The simple team baseline averages a team's earlier own rate with its opponent's
conceded rate, equally, using current/prior-season history. It has no xG, recent
form adjustment, player effect or fitted team-strength model. Sources exclude
the forecast UTC day and require kickoff+3h before the cutoff. Missing statistics
are never observed zeroes; insufficient team support uses the dated league rate.

| Market | League baseline NLL | Simple team/opponent NLL | Engine reference NLL | Engine gain vs league | Engine gain vs simple team |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goals, joint home/away score | 2.969672 | 2.919176 | 2.917412 | 1.760% | 0.060% |
| Corners | 2.613394 | 2.605933 | 2.613188 | 0.008% | −0.278% |
| Shots on target | 2.543024 | 2.529485 | 2.524539 | 0.727% | 0.196% |
| Weighted cards | 2.232200 | 2.218437 | 2.217344 | 0.666% | 0.049% |

Lower NLL is better. Positive gain means a reduction in probability loss, not ROI,
hit rate or percentage-point accuracy. Weekly difference intervals against the
league baseline exclude zero for goals, SoT and cards, but not corners. **Every
engine-versus-simple-team interval includes zero**, including the adverse corner
point estimate. This is evidence against assuming that the complicated reference
is already optimal, not proof that the simpler method is universally superior.

For total goal counts, the reference RMSE is 1.6355 versus 1.6313 for the league
baseline. The joint-score gain therefore must not be called a gain in total-goal
point accuracy. Winner NLL improves from 1.06010 to 1.01592 against the league
distribution: identifying relative team strength is useful even when predicting
the precise total remains difficult.

The frozen weight setups from the preceding experiment remain in the comparison.
Goals/corners NLL is 2.908980/2.604277. They improve the reference, but the corner
advantage over the simple team baseline is only about 0.064% by point estimate.
No new setup or preferred league is selected from these diagnostic results.

## 2. Probability quality at individual outcomes

The empirical comparator is the earlier league frequency of each exact event,
with fixed Beta(1,1) smoothing. It is separate from the league count distribution.

| Outcome | Empirical league log loss | Engine log loss | Engine gain |
| --- | ---: | ---: | ---: |
| BTTS | 0.691862 | 0.692203 | −0.049% |
| Goals Over 2.5 | 0.689919 | 0.686783 | 0.454% |
| Corners Over 9.5 | 0.692010 | 0.696611 | −0.665% |
| SoT Over 8.5 | 0.689696 | 0.673574 | 2.337% |
| Weighted cards Over 4.5 | 0.668584 | 0.652880 | 2.349% |

These point comparisons identify where to investigate; they are not five newly
qualified claims or evidence of beating bookmaker prices. Overall count scores
can improve while a particular betting line remains poorly estimated.

Reliability shows a concrete limitation. In the engine's 60–70% forecast band:

| Outcome | Fixtures | Mean predicted probability | Observed frequency |
| --- | ---: | ---: | ---: |
| BTTS | 388 | 63.6% | 57.7% |
| Corners Over 9.5 | 128 | 63.3% | 53.9% |
| SoT Over 8.5 | 241 | 64.4% | 63.9% |
| Cards Over 4.5 | 129 | 64.4% | 54.3% |

These bins have sampling uncertainty and were inspected diagnostically. The
whole-cohort diagnostic calibration slopes are about .526 for BTTS, .403 for
corners, .924 for SoT and .929 for cards. Slopes well below one are consistent
with probabilities that are too extreme. These fitted diagnostics were **not
applied** to forecasts. Previous calibrator experiments and their failed or
inconclusive later checks remain valid; this finding does not retroactively
qualify a new calibrator or justify fitting one on the evaluation outcomes.

## 3. Player data: a verified source issue and its bounded repair

The immutable prepared collection supplied 4,510 complete historical XIs and
4,421 qualified paired player responses in certified five-league history from
provider seasons 2021–2023. Across all 8,067 certified historical fixture records,
earlier seasons without player responses remain explicitly unavailable. Another
17 responses failed team-minute consistency and 71 failed participation checks;
these were excluded, not clipped into validity.

The qualified raw player fields contained **10,856 positive goal counts and
123,914 nulls; 29,459 positive SoT counts and 105,311 nulls; no explicit zeroes**.
Dividing goals/shots by the minutes of only the non-null entries would estimate
rates on a selected, positive-only sample. This was discovered during source
review before inspecting player-comparison scores. The first attempt is preserved
and its player result withheld. Its independent simple-baseline assessment is
byte-identical to the accepted run.

The v2 adapter establishes a zero only when reported nonnegative player counts
already exhaust the **certified regulation team total**. It retains reported
values, the team-total certificate, exact fixture ID and derived-zero count.
Unallocated or unknown team totals exclude that entire team/field observation
from rate fitting, including reported positives. No residual is assigned to a
particular player or assumed to be an own goal.

| Field | Fully reconciled team/fixture observations | Excluded team/fixture observations | Zeroes established by reconciliation |
| --- | ---: | ---: | ---: |
| Goals | 8,454 | 388 | 118,495 |
| SoT | 8,816 | 26 | 104,903 |

An additional raw-source check found no positive goal/SoT count attached to a
player with zero or missing recorded minutes. These observations support the
accepted source cohort; they do not justify blanket null-to-zero conversion.
Raw archives and canonical records were not rewritten. This rule is specific to
reconciled player goals/SoT; it does not change card nulls or settlement policy.

This problem would bias the new player-rate construction. **It is not established
as the reason earlier team-based ML candidates failed their gates.**

## 4. What the player implementation now does

`phase4_player_context.py` builds exact-ID, cutoff-safe league/position/player
rates with 450-minute shrinkage. It preserves missing fields and source fixture
IDs. Reference minutes describe the players represented in the earlier current
season's team rate. Player effects are changes relative to that reference.

- **Expected-player composition:** derives forecast minutes from the last three
  earlier qualified team allocations. It removes a player when a later, still
  pre-cutoff appearance establishes another team. It never reads the target XI,
  target minutes or target player performance.
- **Conditional actual XI:** takes only starters/bench/positions from the target
  lineup archive. Expected starter and bench minutes come from earlier evidence
  and role priors; actual target-match minutes are forbidden. The archive lacks
  original announcement timestamps, so this is a conditional scenario, never a
  verified historical confirmed-lineup forecast.
- Both allocations total 990 player-minutes with a 90-minute individual cap.
  Missing histories, inadequate known-minute coverage or contradictory identity
  cause market-specific neutral fallback. Other markets remain evaluable.
- One bounded, regularized coefficient per market/stage is fitted on 2022 only.
  All four bundles are frozen before 2023 inference. Goal effects change the two
  underlying team means, then propagate through one coherent goals/BTTS/winner/
  handicap distribution. SoT effects change its total mean with retained
  dispersion. Existing card-risk adjustments are not applied twice.

The code review confirms that the legacy `home_adjustment`/`away_adjustment`
goal/SoT multipliers are constructed in `lineup_context.py` but are not consumed
by the canonical projection calculations. The new research path explicitly
applies its fitted effects before probability calculation and tests that they
reach derived markets. This does not activate those effects in publishing or
validate the older absence heuristic.

## 5. Player backtest results

All common-cohort forecasts are scored, including neutral fallbacks. Eligible
training uses 1,289 expected-player or 1,284 conditional-XI fixtures over 28 weeks.
In 2023, expected effects have input support on 1,532/1,533 goals and 1,520/1,521
SoT fixtures; conditional effects support 1,529 goals and 1,517 SoT. Availability
is not proof that the feature is informative.

| Effect | Fitted coefficient | Development NLL change | 95% weekly NLL-difference interval | Result |
| --- | ---: | ---: | --- | --- |
| Expected players: goals | .09825 | +0.0342% | +.000564 to +.001457 | Small deterioration |
| Expected players: SoT | .04474 | −0.0037% | −.000371 to +.000192 | Inconclusive |
| Conditional actual XI: goals | .25103 | +0.0049% | −.001958 to +.002093 | Inconclusive scenario |
| Conditional actual XI: SoT | .10723 | −0.0485% | −.002093 to −.000332 | Small scenario gain, below threshold |

Negative change favours the player effect. Neither primary expected-player claim
passes the unchanged improvement/uncertainty gates or Holm correction. No
supported slice/related-market regression exceeds 2%. Conditional results cannot
qualify pre-match use regardless of their score. The SoT scenario's clearer but
small gain is an avenue for later timestamped validation, not a release decision.

This implements attacking goal/SoT composition effects. It does **not** complete
opponent-adjusted player attribution, defender/goalkeeper effects, injuries,
formation/manager modelling, a trained multi-scenario expected-lineup/minutes
model, new card/corner player modifiers or the player-props engine. The minute
allocation is an explicit heuristic and the fitted coefficient is intentionally
small and bounded. Failed improvement here is not evidence that all player or
lineup information is useless.

## Validation and interpretation

Successful artifact: `Research/phase4-assessment-players-2026-10-02-v2/`.
It contains source/protocol hashes, source archive references, qualified player
history, every feature/fit/forecast, memberships, full assessment and slice/
reliability results, fallback accounting, validation and completion hashes.

- **126 focused tests passed**, including source-total zero certification,
  missing-versus-zero distinction, current/future result isolation, target-XI
  isolation, allocation limits, cold starts, fit cutoffs and coherent markets.
- Rebuilt all **5,658 player snapshots** and replayed **6,108 player forecasts**
  exactly. Independently checked all four coefficient optima and source cutoffs.
- Audited **9,766,773 player-source links**, **3,667,280 baseline-source links**,
  **73,416 normalized probability vectors** and all **6,288 assessment fixture/
  market pairs**. No method silently drops difficult fixtures.
- Preserved 458 existing fingerprinted source/test/model/dependency files and
  the immutable prepared database; source experiment seals remain intact.

The practical assessment is now specific: the engine distinguishes relative team
strength, while its total-corner estimates and some derived probabilities provide
little advantage over simple forecasts. Mean-estimation quality and probability
extremeness warrant investigation. The confirmed source issue concerns player
count interpretation; there is no proven single remaining bug explaining every
earlier weak result. More seasons, another weight search or adding a player feature
cannot be assumed to fix these limitations.

The next proposal should target those measured weaknesses and use fresh evidence
for acceptance. Historical player-rate/minute work can continue from this reusable
component, with richer attribution and timestamped prospective lineups. Do not
relax previous thresholds or repeatedly tune against these same outcomes while
calling them untouched. Authentic equivalent quote comparisons, full portfolio
evaluation and real publication acceptance remain necessary before claiming
betting value. No further optimization or production release was started here.
