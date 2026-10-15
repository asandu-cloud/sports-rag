# Card improvement experiment v1

Owner authorized the proposed card improvements on 2 October 2026, superseding
the preceding stop-before-optimization instruction for this bounded research.
Production, the original fixed experiment and final-system reserve stay intact.

## Evidence and periods

Use only the sealed qualified participation-v2 targets before 2024. Preserve
the two-minute rule, special-round exclusions, unknown values and source batch
identities. Supplement with exact-ID foul statistics from the immutable import
rehearsal backup, using a read-only SQL cutoff before reading outcome columns.
Load only qualified development fixtures' checksum-verified lineup archives.
No provider calls or active database/profile/model writes.

All feature histories are same-competition, current/previous provider season,
with kickoff + three hours strictly before forecast-day UTC midnight. Team,
opponent and referee histories use qualified weighted targets throughout.
Inputs are `assumed_final`; original referee/lineup announcement times are unknown.

The 2023 fixed results have been inspected. This is an explicitly reused
development comparison, never a fresh test. Mean/dispersion choices use only
forecasts and labels available before January 2023. For nested forecasts, choose
parameters again using only data available before each 2022 quarter. Require
500 fixtures / 26 ISO weeks for fitting; skip insufficient earlier folds.

Outcome-free support inspection found 954 mature 2021 forecasts over 13 weeks;
by April 2022, earlier forecasts span only 24 weeks. July 2022 is the first
quarter with enough training support (2,652 / 32). Therefore use nested July–
December 2022 predictions plus January–March 2023 predictions from the frozen
January recipe for calibration. Fit once at April 2023, requiring 500 / 26;
if insufficient, retain identity and report the block. Do not move the boundary
after scoring. April–December 2023 has 1,730 fixed-reference fixtures / 23 weeks
and is the common comparison period for raw and calibrated finalists. These
dates are a new explicit card-development protocol, not a rewrite of old results.

Later qualification is separate. January–June 2025 has been consumed for other
markets. The owner explicitly approved reusing January–June 2025 on 2 October
2026, with that limitation, after a frozen eligible finalist exists; it cannot
be called untouched. The inherited minimum is 500 fixtures / 10 ISO weeks,
and league conclusions require 100 / 10. No 2025 fitting or method selection.
The necessary 2024 history may supply chronological profiles only after the
method lock, with the same qualification rules and explicit access records.
July 2025–June 2026 and
the later prospective reserve remain closed. No 2024+ outcomes enter this run.

## Reference and bounded candidates

Retain the exact fixed team-only/team-referee forecasts and Poisson scores.
Also execute the archived production card projection functions with explicit
dated v2 profiles, recent counts, referee and foul inputs. Replace external
lookups only. Save the original function source and weights. The inherited trend
weight is zero. Aggression requires unavailable duel inputs and its legacy full-
season normalization is not safe retrospectively; omit it consistently. Original
league-regime/live context is unavailable. Yellow/red display splits from the
legacy function are not certified decompositions of the v2 weighted target.
This fuller aligned control is still a reconstruction, not exact public replay.

Keep the fixed eight-current-season-match common cohort for primary comparisons.
Use a deterministic, bounded registry relative to the fuller control:

- Own/opponent coefficient: 0.5/0.5 or 0.85/0.15, against 0.7/0.3.
- Season/recent coefficient: 1/0 or 0.7/0.3, against 0.85/0.15.
- Venue coefficient: 0 or 0.75, against 0.5.
- Pool team and venue rates using prior strengths 8 or 16. Team prior: previous
  season shrunk toward the league; venue prior: current pooled team rate.
- Calendar-time half-life 90 or 180 days for season/recent averages; no expansion
  beyond current/previous provider seasons and no extra history-window search.
- Referee: none, modifier only, anchor only, or a residual-ratio adjustment.
  The latter compares earlier actual counts with each match's earlier team-only
  forecast, shrinks toward neutrality with strength 20 and coefficient 0.35,
  and does not also add the raw referee anchor. No current/future residuals.
- Foul component off as a separately reported ablation of the fuller control.

Compare these individually on training data. Permit at most two combinations of
the best rate and referee changes, only if each separately improves training NLL
by at least 0.5%; the second may add a separately earned foul ablation. Rebuild
these choices separately inside every forward fold. Ties within 1e-8 prefer the
unchanged fuller control, then fewer changes and deterministic identifier order.
No league-specific weights or switches selected from 2023 results.

For distribution fitting, keep the chosen mean fixed and compare Poisson with
NB2 alpha in [0, .01, .025, .05, .1, .2, .4, .8, 1.6], selected only on earlier
training predictions. Also report the fuller mean under its reconstructed
observed-variance rules where available. Preserve the simpler reference in every
final comparison; mean and dispersion improvements remain distinguishable.

Fit one monotone distribution-level calibrator to genuine nested predictions:
g(F) = sigmoid(a + b*logit(F)), b in [0.25, 4], a in [-2, 2], regularization
C=0.1 toward identity a=0,b=1. Minimize count NLL, not the in-sample reliability
bin gap. Preserve F(-1)=0 and F(infinity)=1. Derive count probabilities, all totals
and five-way Asian outcomes from the same calibrated CDF. Identity is retained
as the comparator; no search over calibrator families/strengths in this run.

## Separate coverage and lineup scenarios

Run the fixed strength-16 pooling candidate on an expanded cohort with at least
50 earlier qualified same-competition matches. Report its additional fixtures,
early-season and European support separately; never mix them into common-cohort
improvement claims or relax per-league minimums.

Actual historical starting XIs may be used only in a labelled scenario excluded
from selection/qualification. Player tendencies use earlier qualified evidence
only, shrink toward the league player rate with 450 minutes, and estimate minutes
from prior appearances with a five-appearance, 90-minute prior. Apply the existing
bounded lineup adjustment to the fuller control. Save starter IDs, prior support,
and source hashes. This is not a player-prop/minutes engine or proof of pre-match
lineup availability. Referee-name fragments remain separate without independent
identity confirmation.

## Scores, artifacts and acceptance

Primary claim: count NLL for the one frozen combined finalist against both the
fixed referee reference and fuller aligned reference. Require >=0.5% improvement,
a paired week-bootstrap 95% interval wholly below zero, and no supported league,
season-stage or foul-missingness slice regression above 2%. Use 2,000 paired
ISO-week draws, seed 20261002. Evaluation support is 200 fixtures / 20 weeks;
the same minimum applies to slice conclusions. Unsupported slices remain
unqualified and cannot be hidden by pooled performance.

Report raw/adjusted means, bias/MAE/RMSE, count NLL, Brier/log loss and reliability
at 3.5/4.5/5.5, 80%/95% interval coverage, five-way Asian vectors at
4/4.25/4.75/5, effective sample sizes and league contributions. Tail error <=1e-10,
normalization, monotonicity, complementarity and settlement parity are required.
Record every candidate and exclusion. No odds, ROI, fabricated timestamps or
profitability claims. Failures do not authorize retuning on 2023 or later periods.

Preserve sealed contextual inputs, protocol/code/dependencies, complete dated
forecast evidence, candidate registry, training memberships/parameters, nested
predictions, method/calibration locks, paired predictions, metrics and isolated
non-public bundle. Test future-result independence, fit isolation, fixed-reference
parity, missing/zero distinction, player chronology, CDF/Asian correctness,
determinism and offline replay. Promotion requires separate approval.
