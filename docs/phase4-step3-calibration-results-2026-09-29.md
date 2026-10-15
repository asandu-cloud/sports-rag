# Phase 4 Step 3 — logistic comparator and calibration development results

**Subsequent status:** the [authorized earlier-forecast extension](phase4-step3-earlier-forecasts-results-2026-09-29.md)
resolved the 13-week shortage with 43 validation weeks and evaluated challenger
calibration. This report preserves the findings and limitations of the first run.

Completed for the supported development scope on 29 September 2026. Implemented
and evaluated the declared four logistic comparators, monotone binary calibration
and coherent joint-goal temperature calibration. Challenger-specific calibration
method selection remains **insufficient**, rather than completed or disproved.

## Findings

Two binary claims pass the frozen development screen against their raw control:
BTTS logistic comparison and sigmoid calibration of corners Over 9.5. These are
outcome-specific diagnostic claims. They do not qualify a combined probability
system, betting policy, profitability or deployment.

| Method | 2023 fixtures | Raw log loss | Adjusted log loss | Reduction | Paired 95% difference interval |
|---|---:|---:|---:|---:|---|
| comparator:control:btts | 3,095 | 0.692773 | 0.688736 | 0.583% | [-0.00787788, -0.00003692] |
| sigmoid:control:corners | 3,028 | 0.694436 | 0.690037 | 0.633% | [-0.00749075, -0.00138483] |

Both satisfy the >=0.5% primary improvement, negative paired 95% interval and
no >2% supported-slice regression requirements. The BTTS interval ends very close
to zero; it is a modest development result, not strong evidence of a large gain.
Later qualification and its multiplicity correction have not been run.

- The BTTS comparator uses C=.01 and improves Brier from .249643 to .247810.
  Its log loss (.688736) is still slightly worse than the existing opponent-adjusted
  goal challenger’s BTTS log loss (.688471). Passing against the frozen control
  does not make this the best coherent research system.
- Corner Over 9.5 sigmoid calibration uses C=.1 and improves Brier from .250509
  to .248445. The diagnostic calibration slope moves from .462 to .969 and the
  intercept from .0566 to -.00165. These are later-period reporting diagnostics,
  not parameters refitted and applied on 2023.
- The other six supported binary tests fail at least one development criterion.
  Corner logistic comparison narrowly misses the uncertainty criterion; goal
  sigmoid calibration has a 2.403% supported Bundesliga log-loss regression.
- Control joint-score temperature selects **1.0 (identity)** on 2022. It adds no
  new 2023 gain. Challenger temperature retains identity because its method-selection
  cohort is insufficient. The earlier .719% opponent-strength joint-score gain
  remains a Step 2 result; it must not be credited to temperature calibration.

## Complete supported binary comparison

Negative log-loss change is favorable. All settings were locked before scoring
2023; its outcomes were not used to pick C or a calibration method.

| Test | Selected C | Log-loss change | Screen |
|---|---:|---:|---|
| comparator:control:btts | 0.01 | -0.583% | Pass |
| comparator:control:corners | 0.1 | -0.567% | interval crosses zero |
| comparator:control:goals | 0.01 | -0.396% | below .5% improvement; interval crosses zero |
| comparator:control:sot | 0.01 | -0.040% | below .5% improvement; interval crosses zero |
| sigmoid:control:btts | 0.01 | -0.538% | interval crosses zero |
| sigmoid:control:corners | 0.1 | -0.633% | Pass |
| sigmoid:control:goals | 0.01 | -0.089% | below .5% improvement; interval crosses zero; supported slice regression |
| sigmoid:control:sot | 0.01 | +0.108% | below .5% improvement; interval crosses zero |

## Challenger-calibration limitation

The frozen support requirements are 500 fitting fixtures across 26 observed ISO
weeks, and 200 validation fixtures across 20 observed weeks. Nested challenger
base selection cannot use the full-2022 setting while pretending that its earlier
validation quarters were unseen. At the start of Q1, Q2 and Q3, preceding **2022**
candidate forecasts do not supply the required base-selection support. Only Q4
qualifies. That leaves 13 validation weeks:

| Challenger target | Available nested validation fixtures | Weeks | Status |
|---|---:|---:|---|
| BTTS / goals Over 2.5 | 1,043 each | 13 | Insufficient |
| Corners Over 9.5 | 1,023 | 13 | Insufficient |
| SoT Over 8.5 | 1,005 | 13 | Insufficient |
| Challenger joint-goal temperature | 1,043 | 13 | Insufficient |

The Q4 fitting/calculation paths executed, but no challenger C/temperature was
selected from that insufficient cohort. In the saved plans, `tuning_available=false`
and identity is an explicit fallback. The 2023 identity rows preserve comparison
membership; they are **not evidence that challenger calibration was tried and
failed to help**. Binary records with no fit ID inherit this status from their plan.
This is a shortage of suitable earlier out-of-sample challenger forecasts under
our locked chronology, not a finding that the eight-season raw database is too small.
No new provider download would automatically resolve it. An extension would need
earlier candidate forecasts and a prospective, explicit base-selection chronology;
it must not silently borrow later-period labels or lower the support limits.

## Implementation and evaluation contract

See the [pre-scoring execution protocol](phase4-step3-calibration-protocol-2026-09-29.md).
It preserves the frozen v1/v2 thresholds, periods and candidate budget.
Inputs retain the inherited `assumed_final` historical availability. These are
cutoff-controlled reconstructions, not exact replays of published recommendations
or immutable records of everything known at the original forecast time.

- Comparator inputs are control logit, control mean, applicable home-away goal-rate
  difference and training-observed league indicators. C is .01/.1/1/10.
- Numeric scaling is fitted only on training rows. Unseen leagues use the pooled
  numeric component, with an explicit flag; there were no such 2023 cases.
- Sigmoid calibration has one logit feature and a constrained nonnegative slope.
  It is pooled, with no league-specific calibration coefficients.
- Earlier qualified reconstructed controls initialize the forward 2022 folds.
  Final comparator fits use earlier history through 2022; final sigmoid fits use
  2022 out-of-sample base forecasts only. No calibrator refits on 2023 labels.
- The challenger’s base setting is selected using preceding 2022 folds only, then
  independently reconciled to the Step 2 full-2022 lock for later comparison.
- Temperature .75/.9/1/1.1/1.25/1.5 transforms the shared score PMF. A geometric
  bound expands the transformed tails to <=1e-10. Related totals/BTTS/winner/Asian
  profiles all derive from that same PMF. Stored goal-rate parameters are the
  pre-temperature inputs; score metrics use the transformed distribution mean.
- Fitted parameters, source/data hashes and selections are checksum-locked before
  the 2023 worker can run. The tuning worker’s IO guard denies the 2023 shard.
- 2023 remains already-inspected development. No 2024+, dedicated calibration,
  qualification, final-system or prospective-reserve labels were read.
- Interval estimates use 2,000 paired calendar-week draws, seed 20260928. Reporting
  includes reliability, class support and league/season/stage/missingness slices.
  Unsupported leagues and the early-season slice retain their Step 2 limitations.

## What can and cannot be connected to the engine

The logistic BTTS comparator is isolated. Replacing only the shared goal engine’s
BTTS number would make its markets inconsistent. Likewise, a calibrated Over 9.5
probability does not provide a full corner PMF, alternative lines or complete Asian
settlement vectors. No such wiring occurred. The Step 2 dispersion models remain
count-distribution candidates; new distribution-level calibration methods would
need their own bounded, declared experiment. Identity remains the supported
joint-temperature choice on the control. Confidence/value are not inferred from
a calibrated outcome probability, and existing quality/release gates remain intact.

## Validation and preserved artifacts

**517 tests passed**, including 20 new calibration tests covering chronology,
held-out preprocessing isolation, nested base selection, monotonicity, bundle
identity/cutoff/period checks, nonconvergence, transformed-tail bounds against a
large independent grid, shared goal coherence and forbidden archive reads.
The independent audit verifies 152 fitted bundles, replays eight supported nested
base choices and 169,133 binary predictions with zero probability differences,
checks evaluation membership, and verifies the selection/bundle lock. All 34,318
joint records satisfy the transformed-tail bound. A further 6,190 raw joint-score
checks reproduce Step 2 NLL within 1e-10. Exact records are saved under `validation/`.

New code is confined to research modules and operations:
`Scripts/data_platform/features/phase4_calibration.py`,
`Scripts/ops/phase4_calibration.py`, `Scripts/ops/phase4_calibration_worker.py`,
`Scripts/ops/phase4_calibration_review.py`, and the corresponding tests.

`Research/phase4-step3-calibration-2026-09-29/` preserves prepared inputs, source and
dependency hashes, execution protocol, OOF predictions, JSON fitting bundles,
nested choices, selection lock, raw/adjusted evaluation predictions, metrics,
audits and this report under a COMPLETE manifest. Production sources, active
models, databases and publishing modes are unchanged. All earlier sealed artifacts
are preserved. No provider calls or public recommendations were made.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_calibration verify Research/phase4-step3-calibration-2026-09-29
```

## Remaining Phase 4 work

Review the supported complete-system configuration and its limitations (Step 4),
keeping these binary diagnostics separate from coherent production candidates.
Challenger-specific calibration remains an open evidence gate. Later calibration
and single qualification (Step 5) require methods/scope locked beforehand; the
final probability-plus-selection test remains protected for Phase 5. Cards and
richer lineup effects retain their separate qualification requirements.
