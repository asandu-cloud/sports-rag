# Phase 3 Batch D — frozen confirmation protocol

Authorized by the owner on 27 September 2026: “Ok, get Batch D done.”
This protocol is written before opening the 2024 confirmation outcomes.
It implements the accepted gates in the Phase 3 technical plan, Section 8.

## Frozen comparison

Use the immutable Batch A dataset `phase3-dataset-2026-09-26`. Confirmation is
2024-01-01 inclusive through 2025-01-01 exclusive, grouped by provider fixture.
Every eligible fixture participates in all three methods for its market.
Do not silently intersect away unavailable predictions. Fail the operation if
an eligible baseline or candidate cannot produce a valid value.

Compare the two completed development proposals using their saved, identical
outer development cohorts and equal-fixture RMSE. Select the lower RMSE per
market; an exact tie retains the original Batch C proposal. This selects:

| Market | Frozen source | Model | ML blend weight |
|---|---|---|---:|
| Goals | Batch C extension | Separate home/away Poisson | 0.8397231117074506 |
| Corners | Batch C extension | Separate home/away Poisson | 0.7685050635680528 |
| Shots on target | Corrected Batch C | CatBoost | 0.5225317775885302 |

Use those existing bundles, fitted strictly before 2024, with no refitting,
preprocessor updates, architecture search, reweighting or calibration. The
extension models use the fixed 1,460-day lookback and 365-day half-life. SoT
uses its saved 1,460-day lookback and no decay. All inference uses the existing
126-feature reference contract. Cards remain excluded.

Features and the two statistical comparators update chronologically from
previously completed matches, including earlier matches within confirmation.
Every input must have assumed availability (kickoff plus three hours) strictly
before the forecast kickoff. This is the existing retrospective `assumed_final`
policy, not an original historical provider-vintage claim. The feature/profile
window remains current plus immediately previous provider season.

Comparators are the existing league/venue-average and statistical-core
reconstructions with exactly the development baseline source/configuration
contract. The latter omits player/lineup, stakes and other live adjustments;
it is not a replay of the full production prediction shown to users.

## Gates and reporting

Primary: relative reduction in equal-fixture RMSE, separately for each market.
Require at least 1,000 confirmation fixtures and 20 nonempty UTC weeks, at
least 1% improvement over **both** comparators, and a strictly positive lower
paired improvement bound. Use 20,000 paired calendar-week resamples, seed 42,
with two-sided Bonferroni intervals at `1 - .05/6` confidence for the six
primary comparisons. Preserve shared fixture/week draws across markets and
leagues. Report the predeclared two-week-block sensitivity alongside the
primary decision; it is a robustness diagnostic, not a new threshold.

Each league/market additionally needs 200 fixtures, 20 match dates, adequate
calendar precision (20 nonempty weeks), and a one-sided 95% upper bound on
relative RMSE deterioration below 5% against the statistical reconstruction.
Pooled failure cannot be rescued by choosing a league after seeing the test.
Absent/sparse leagues remain explicitly unqualified. Other slices receive
the same support, precision and deterioration checks, with their qualification
reported separately. An unsupported or failing slice is not qualified by a
pooled pass. Slice findings cannot create post-hoc model variants or alternative
acceptance tests.

Report MAE, signed bias, RMSE, R², common uncalibrated Poisson log loss/RPS,
interval coverage/width, equal-league descriptive metrics, and all existing
league/season/stage/source/round/support/missingness slices. Unsupported
intervals retain explicit reasons. No ROI/CLV is calculated without authentic
historical executable/equivalent quotes. No probability or betting-policy
qualification follows from count-forecast confirmation.

## Inspection, artifacts and failure handling

Freeze candidate copies, full specification, protocol, code/data checksums,
dependencies, eligibility/split definitions and seed **before** any confirmation
feature/label or outcome-rich history access. Atomically create the external
calendar-period `phase3-confirmation-inspections/phase3_confirmation_20240101_20250101/OPENED.json`
claim, bound to the dataset, specification and experiment identity. It cannot be
overwritten by a new run or bypassed by re-exporting/copying the same test period.
The immutable dataset's original lockbox-policy document remains export-time
evidence; the external record is the current inspection state.

The dedicated reader validates the exact confirmation membership and replays
every eligible snapshot/vector/support set. It structurally skips later audit
history outcomes and the duplicate fixture payload array. It never opens the
calibration, final-system-test or prospective-reserve feature/label stores.
History can contain earlier confirmation results for chronological features;
fixed predictors never fit on them. Save a completed immutable prediction
artifact before opening the confirmation label store for scoring.

Preparation, inference and evaluation use separate immutable stages. Interrupted
attempts remain visible; only the same frozen specification/experiment may
resume. A completed resume verifies hashes and does no new inference/scoring.
An inspection remains consumed even if execution fails. Numerical correctness
repairs after inspection must be disclosed, and cannot reset its status.

Save the result, decision, all predictions, coverage/exclusions, uncertainty,
isolated candidate copies and later-period reservations. Keep active models,
databases, refresh gates, public modes and delivery untouched. No provider calls,
embeddings, model promotion, cloud changes or test publications are authorized
by this batch. Live/shadow acceptance and explicit promotion approval remain
later gates, regardless of the result.
