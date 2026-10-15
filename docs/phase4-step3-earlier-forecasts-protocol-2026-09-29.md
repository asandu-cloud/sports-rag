# Phase 4 Step 3 — earlier challenger forecasts, protocol v2

Owner authorized this extension on 29 September 2026 after the first Step 3
experiment reported only 13 eligible challenger-validation weeks. This is an
additive amendment to evaluation v1 and Step 3 execution v1; their archived
copies remain unchanged. Declare this procedure before generating or scoring
the additional forecasts. Identifier: `phase4-calibration-earlier-forecasts.v2`.

## Inspection history and exact amendment

Step 2 and Step 3 have already inspected 2022 and 2023. The extension responds
to insufficient nested calibration support, not to a new untouched test.
Existing 2019–2021 control forecasts and eligibility metadata have also been
inspected. Newly generated earlier challenger forecasts use those same inputs.
The 2023 rerun is reused development evidence. No 2024 or later labels are read.

For nested base selection at each 2022 quarter, permit all eligible prior
2019–2021 challenger forecasts plus earlier 2022 forecasts, with labels strictly
available before that quarter. This explicitly amends the v1 restriction that
base probability parameter selection use 2022 only, **for initializing and
updating these nested folds only**. Final base selection remains full-2022 only
and must reproduce the existing Step 2 selection lock. Final sigmoid fits remain
2022 out-of-sample base predictions only. Final comparators remain unchanged.

## Earlier forecasts

Generate every available saved 2019–2021 quarter, with identical frozen engine
snapshots, target eligibility, assumed-final availability, omitted context and
fit-eligibility IDs used in the previous research. Do not access the live database
or provider. Copy only pre-2022 history into the earlier forecast experiment.

Run only control, goal-strength ridge 1/10/100, corner dispersion and SoT
dispersion at the existing nine alpha values. No new settings or features.
Fit team strengths before each quarter, using the same current/prior-season
policy and minimum support as Step 2. Preserve explicit control fallback for
insufficient fits and unknown/sparse teams; include these fixtures in comparisons.
Replay every eligible control's means and variances against the saved control.
Preserve fitted bundles, training IDs, cutoffs, forecasts and fallback coverage.

Base-setting choices and each binary calibrator fit at a 2022 quarter use only
preceding forecasts/labels. Calibrator fitting support is still 500 fixtures,
26 observed ISO weeks and 50 examples of each binary outcome. Choose C, identity
and temperature using only pooled 2022 forward predictions; require at least
200 fixtures and 20 observed weeks. Earlier rows initialize fitting; they do not
count as additional 2022 validation observations. Do not suppress insufficient
folds or lower thresholds if the new evidence still fails these requirements.

Each earlier setting's predictions are out of sample for its numerical fitting.
Nested setting selection uses prior forecast losses; it cannot inspect its
validation quarter. Final base setting is selected on 2022 as already declared;
final calibrator fitting on its 2022 OOF forecasts is development fitting, not
an independent performance estimate. Freeze choices and bundles before 2023.

## Unchanged evaluation and required verification

Keep C=.01/.1/1/10, temperatures=.75/.9/1/1.1/1.25/1.5, original tie rules,
2023 common cohorts, proper scores, 2,000 paired week draws, seed 20260928,
0.5% primary improvement, negative 95% difference interval and 2% supported
slice/related-market regression limit. Independent binary calibration remains
diagnostic; coherent goal temperature regenerates the whole score distribution.
Report improvement against each candidate's raw probabilities separately from
improvement against the frozen control. An identity choice is a valid result.

Test that future labels cannot change earlier fits/choices, preprocessing is
training-only, original selection behavior is preserved by default, incomplete
candidate slates fail, deterministic membership/hashes replay, and tuning cannot
read 2023. Audit every fitted bundle and nested selection; replay binary forecasts;
check common membership, earlier strength cutoffs and exact control parity.
Control-only comparator/calibration results must reproduce the first experiment.

Preserve two new immutable artifacts: earlier forecasts, and the extended
calibration comparison. Capture code/data/dependency hashes, this protocol,
coverage/exclusions, predictions, locks, tests and independent review. Verify all
parent archives and unchanged production sources. No promotion, provider calls,
public messages, database updates, reserve access or fitted production changes.
