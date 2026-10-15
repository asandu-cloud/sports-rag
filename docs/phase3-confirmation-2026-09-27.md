# Phase 3 Batch D — confirmation and handoff

Completed locally on 27 September 2026, following the owner's explicit Batch D
authorization. The one-time 2024 confirmation is complete. **No candidate meets
all agreed qualification gates. Keep the existing statistical production engine.**
This completes the bounded Phase 3 numerical benchmark programme; it does not
establish that ML is useless, profitability, or production readiness.

## Results

Each market compares exactly the same eligible fixtures across its frozen
candidate, statistical-core reconstruction and league/venue average. Positive
improvement means lower RMSE. All cohorts span 39 nonempty UTC weeks.

| Market | Fixtures | Candidate RMSE | Statistical RMSE | Average RMSE | Improvement vs statistical | Improvement vs average | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Goals | 1,949 | 1.695467 | 1.700439 | 1.720672 | 0.292% | 1.465% | Not qualified |
| Corners | 1,776 | 3.289928 | 3.352327 | 3.319237 | 1.861% | 0.883% | Not qualified |
| Shots on target | 1,790 | 3.111284 | 3.130370 | 3.218737 | 0.610% | 3.338% | Not qualified |

The accepted gate requires at least **1% improvement over both baselines** and
a positive lower paired improvement bound against both. The primary intervals
use 20,000 UTC-week resamples, seed 42, with two-sided 99.1667% confidence
(Bonferroni family-wise 95% across six comparisons):

| Market | Improvement interval vs statistical | Improvement interval vs average | Failed gate |
|---|---|---|---|
| Goals | −0.736% to +1.393% | +0.674% to +2.269% | Statistical comparison below 1%; positive improvement uncertain |
| Corners | +0.789% to +2.790% | +0.084% to +1.736% | Average comparison below 1% |
| Shots on target | −0.036% to +1.252% | +1.878% to +4.638% | Statistical comparison below 1%; positive improvement uncertain |

Corners shows credible but insufficient improvement under the agreed practical
threshold. The threshold was not lowered. The two-week sensitivity leaves all
three decisions unchanged; SoT's statistical comparison has a positive lower
bound with two-week blocks but still fails the 1% point-improvement gate.
The saved machine-readable decision is `no_qualified_improvement` for every
market. This is more precise than claiming that all observed gains are zero.

Secondary candidate metrics:

| Market | MAE | Bias (prediction − actual) | R² | Poisson log loss | RPS |
|---|---:|---:|---:|---:|---:|
| Goals | 1.362096 | −0.070458 | 0.036894 | 1.913167 | 0.939932 |
| Corners | 2.634675 | +0.045202 | 0.039296 | 2.595296 | 1.845524 |
| Shots on target | 2.447098 | −0.090636 | 0.086681 | 2.529502 | 1.726095 |

The full metrics preserve all baselines, central 50/80/95% interval coverage
and widths, equal-league diagnostics, and league/provider-season/season-stage/
forecast-stage/source/round/support/missingness breakdowns. Poisson scores and
intervals are common **uncalibrated diagnostics**, not Phase 4 acceptance.
Historical ROI and CLV remain unavailable: no authentic equivalent quote cohort
was supplied for this experiment.

## Frozen methods and eligibility

The [protocol](phase3-confirmation-protocol-2026-09-27.md) was frozen before
opening confirmation. Comparing the two completed development proposals on
identical outer cohorts selected the extension's separate home/away Poisson
blends for goals/corners and corrected Batch C's CatBoost blend for SoT. Exact
ML weights are 0.8397231117074506, 0.7685050635680528 and 0.5225317775885302.
Their existing preprocessors, parameters and weights were reused unchanged.
**Batch D performed zero fits.**

The immutable Batch A dataset contains 4,400 confirmation fixture records.
Its frozen eligibility produces 5,515 fixture/market pairs and 16,545 scored
predictions across the three methods. Every eligible feature vector, support
set and snapshot replayed exactly. There were no prediction drops or newly
selected test cohorts. Earlier completed 2024 matches update later pre-match
features and comparator profiles under the existing strict kickoff-plus-three-
hours assumed availability policy; they never refit the models or blends.

| Competition | Goals | Corners | SoT |
|---|---:|---:|---:|
| EPL | 358 | 338 | 347 |
| LaLiga | 368 | 346 | 341 |
| SerieA | 363 | 344 | 344 |
| Bundesliga | 288 | 274 | 276 |
| Ligue1 | 275 | 260 | 262 |
| UCL | 123 | 104 | 106 |
| UEL | 84 | 63 | 65 |
| UECL | 90 | 47 | 49 |
| Championship | 0 | 0 | 0 |
| SuperLig | 0 | 0 | 0 |
| Eredivisie | 0 | 0 | 0 |
| PrimeiraLiga | 0 | 0 | 0 |
| BelgianProLeague | 0 | 0 | 0 |

The original five domestic leagues meet slice count/date/precision minima;
their one-sided deterioration bounds all stay below 5% against the statistical
reconstruction. No other sufficiently supported slice breaches that bound.
They nevertheless inherit the pooled candidate failure, so **no league is
qualified**. European slices are too small. The five newer domestic leagues
lack qualifying saved regulation-period evidence in this confirmation cohort;
their zero eligible counts are not proof that the models work or fail there.
Cards remain excluded for the previously recorded semantics/sample-bias issues.

## Inspection and recorded implementation repair

Permanent inspection was claimed at `2026-09-26T23:07:50.683993+00:00`
(27 September locally), after the candidate, code, input and evaluation freeze.
The calendar-period registry prevents a new experiment or re-export from
presenting 2024 as an untouched test:

`Index/prediction_experiments/phase3-confirmation-inspections/phase3_confirmation_20240101_20250101/OPENED.json`

The first preparation attempt stopped **before history decoding, inference or
label scoring**: the shared reader did not recognise `european_league`, a valid
round classification already implemented in Batch A and present in the 2024
European league phase. Development data predates that format, so the omission
had not surfaced there. It first failed on fixture 1298984.

The repair adds that existing round label to schema validation and records a
source-only implementation amendment. It does not change features, eligibility,
membership, models, blends, metrics or thresholds. `inputs-0001/FAILED.json`,
the original freeze, `correctness-amendment-0001/`, and the permanent registry's
`IMPLEMENTATION-AMENDMENT-0001.json` retain the failure and both source versions.
The original specification/inspection identity remains authoritative. The run
resumed as `inputs-0002`; no test result was available to guide the repair.

The history reader decoded 21,273 eligible pre-2025 context records, structurally
skipped 4,531 later history payloads, and skipped the duplicate fixture-outcome
array. The separate calibration/final-system/reserve feature and label stores
were not opened. Predictions were completed and checksummed before opening
the confirmation label store. The dataset's immutable `lockbox-policy.json`
still describes export-time status; consult the external registry for its
current **consumed** confirmation status.

## Artifacts and verification

Completed experiment:

`Index/prediction_experiments/phase3-confirmation-2026-09-27`

Key artifacts:

- `freeze/`: original specification, protocol/source archive, dataset schema and
  splits, and copied isolated candidate bundles.
- `inputs-0002/`: replayed features, exact comparator reconstructions, complete
  coverage/exclusions and history-access accounting.
- `predictions-0001/`: immutable forecasts without observed targets.
- `evaluation-0001/`: scored predictions, full metrics, uncertainty and decision.
- `RESULT.json`: consumption, implementation-amendment provenance, execution
  code hashes, later reservations and explicit no-training/no-promotion flags.

Run `COMPLETE.json` SHA-256:
`5ae1f37a3a463c8a9cce3e694233d7c2e444e645d501fa7b68d8315e89958a25`.
Original specification ID:
`b53c13863efd77b8cb0dc37188f818cce775a3fbae32acacbc168d9b158940b1`.

Validation: **656 tests passed** (527 benchmark tests plus 129 feature,
eligibility and dataset regressions). The tests cover chronological isolation,
reader boundaries, exact fixture membership, missing versus zero, no silent
prediction drops, immutable consumption, resumed stages, unchanged recipes
under source repair, and prediction-before-scoring ordering.

The separate post-run audit is preserved in
`phase3-confirmation-validation-2026-09-27/`. It verified all 65 completed
artifacts, replayed all frozen predictions exactly, checked 36 numerical metrics
with separate NumPy arithmetic, reproduced all 12 primary/sensitivity intervals
and 30 supported league bounds, and confirmed unchanged dataset/development
completions. It is a correctness audit of the same test, not another independent
confirmation. A completed CLI resume verified integrity without new inference,
scoring or fits.

Active model files, projection configuration/source, production requirements and
the existing refresh-incomplete marker matched their preflight hashes. The
baseline's six archives and 187 saved Match Read compiler replays also passed;
full original historical numerical-decision replay remains an open gate.

## Handoff and remaining work

- Phase 3 numerical confirmation is complete with no qualifying candidate.
  Production remains statistical; candidate copies retain their original
  unconfirmed bundle metadata, with the separate Batch D rejection record.
- Do not tune more candidates on 2024 and continue calling it untouched.
  A later research programme must disclose its consumed status and define fresh
  evaluation evidence before making an improvement claim.
- Phase 4 can improve the statistical engine's distributions, dispersion,
  score dependence, calibration and confidence. Include the planned logistic
  regression comparator with appropriate probability targets. A new Phase 4
  protocol must freeze fitting/validation use before opening later reserves.
- Calibration remains reserved for 2025-01-01 through 2025-07-01; the final
  completed-system test remains 2025-07-01 through 2026-07-01. Neither was
  consumed here. Post-2026-07-01 is the prospective reserve.
- The frozen history is still retrospective `assumed_final`, with evidence-based
  exclusions and one reconstructed kickoff forecast stage. This is local
  evaluation on stored provider-derived history, not new provider verification
  or acceptance of actual published recommendations.
- Real matchday publication → closing capture → settlement acceptance, shadow
  monitoring, Discord hub activation, card targets, production hosting and the
  separate player-props engine remain outside this batch. The previously
  present refresh-incomplete marker remains in place; no refresh/embedding
  recovery or lock bypass was attempted.

No provider requests, embeddings, public recommendations, database writes,
dependency changes, model activation or delivery-mode changes were performed.
