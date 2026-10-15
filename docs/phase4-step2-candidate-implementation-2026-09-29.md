# Phase 4 Step 2 — candidate implementation

Completed locally on 29 September 2026. The owner authorized implementing the
expanded statistical candidates before comparing them against the frozen engine.
This delivers executable research candidates and correctness validation. It does
not select parameters, establish improved accuracy or change public predictions.

## Implemented scope

The [additive protocol v2](phase4-step2-candidate-protocol-v2-2026-09-29.md)
declares 71 settings including the unchanged control, not 71 independent models.
Each setting changes one named component. The archived protocol v1 is preserved.

| Component | Implementation |
|---|---|
| Opponent-adjusted means | Regularized joint Poisson attack/defence estimates, competition intercept and home advantage; separate goals/corners/SoT fits; current and preceding provider seasons; explicit sparse/unknown-team fallback |
| League/venue weighting | Dated within-league priors and venue-to-overall pooling using field-specific effective sample support; fixture IDs retained |
| xG | Opponent xG supplies defensive xGA; partial attacking-xG coverage can use the observed overall average instead of discarding it when one venue is missing |
| Season transition | Gradual strength-16 prior, executed through frozen profile/projection functions; abrupt control transition retained for comparison |
| Recent history and mean weights | Bounded elapsed-day half-lives, recent-window lengths and individual own/opponent, venue, xG and recent/season coefficients |
| Corners/SoT spread | Existing Poisson/NB count tools reused; fixed-mean dispersion grid and preceding-forecast fitting function |
| Goal dependence | Declared Dixon–Coles rho grid with a fitting function; invalid probabilities fail explicitly |
| Goal distribution | Independent negative-binomial home/away marginals; goals, BTTS, winner and handicaps derive from one joint distribution |
| Richer lineup effects | Incremental goal/SoT contribution differences against reference minutes, with exact identities and dated evidence gates; synthetic validation only |

The lineup interface is not a trained player-contribution or minutes engine.
Historical snapshots, pre-cutoff contribution coefficients and expected-minute
evidence still require qualification. The sealed Step 1 cohort cannot yet support
that historical comparison. Cards retain their separate target-qualification work.

## Code and execution contract

- `Scripts/data_platform/features/phase4_candidates.py`: registry, availability
  and support arithmetic, isolated strength fitting/inference, probability-grid
  fitting, bundle validation, PMFs and lineup evidence contract.
- `Scripts/data_platform/features/phase4_candidate_adapter.py`: candidate adapter
  around explicitly supplied frozen functions. Patches are temporary, inputs are
  copied, and control calculations are rerun after candidate execution.
- `Scripts/ops/phase4_candidate_worker.py`: isolated unscored execution with IO
  guards and control parity, mutation and probability checks.
- `Scripts/ops/phase4_candidates.py`: capture, smoke, seal and checksum verification.
- `Scripts/tests/test_phase4_candidates.py`: 37 focused tests including synthetic
  fits and independent-process integration runs.

Fitting is separate from inference. Strength bundles preserve exact training IDs,
target/data hashes, competition/season/market, cutoff, coefficients, support,
effective sample size and convergence. Forecasts reject future or mismatched
bundles. Probability fitting requires one regulation market, at least 500 fixtures
over 26 weeks, saved earlier forecasts and labels available before the fitting
cutoff. All parameter choices in this implementation run were synthetic; there
are no selected historical parameters.

The local 60-fixture strength-fit minimum is not qualification for deployment.
Frozen comparison support, proper-score and slice-regression requirements remain
in force. V2 explicitly records competition-specific strength estimation and
requires multiplicity accounting for the expanded family count.

Candidate records expose final means, count/score distributions, seven diagnostic
total lines and complete Asian settlement probabilities. Their
`projection_path_records` deliberately retain the intermediate canonical profile
path before explicit strength/lineup/distribution replacements; the role is labelled
in each record. Subsequent scoring must use the final candidate outputs. These
research records are not publication payloads and contain no invented prices/EV.

## Validation performed

**482 tests passed:** 37 new candidate tests plus the 445-test historical-control,
baseline, calibration/card, prior-transition, probability/market-contract and
participation-settlement regression suite. The saved JUnit report records zero
failures/errors. The test harness disables dotenv loading and outbound sockets.

Tests cover deterministic fitting and JSON replay, synthetic opponent-strength
recovery, future-label invariance, league isolation, missing versus zero, exact
field support, nonconvergence, stale/future/mismatched bundles, coherent goal
probabilities, bounded tails, Asian arithmetic, lineup identity/time/minutes
validation and forbidden IO. Independent-process synthetic integration exercises
all 71 settings, including fitted strength and nonzero lineup paths, twice.

The bounded **unscored historical smoke** covers seven metadata-selected saved
fixtures across EPL, Championship, BelgianProLeague, UCL and UECL, including early
season and available-xG strata. It generated **497 records**:

- 392 computed records and 105 records with explicit fallback status.
- All seven controls exactly matched their Step 1 canonical records and goal
  distributions. Control checks after candidate execution found no patch leakage.
- No invalid candidates, input mutation or IO violations occurred.
- Nine strength settings per fixture lacked historical fitted bundles (63
  records); three lineup settings lacked qualified evidence (21 records).
- Three defensive-xG settings per fixture reported at least one unavailable
  component (21 records). This status can accompany a partial effect for another
  supported component; per-team evidence is authoritative. It is not proof of
  complete cohort xGA coverage or a reason to invent zeros.

This smoke checks execution and provenance. It computes no accuracy scores,
selects no method, and provides no performance estimate. Fitted paths were tested
synthetically; those tests do not replace historical forward evaluation.

## Preserved artifact and boundaries

`Research/phase4-step2-candidates-2026-09-29/` contains captured implementation,
protocol, registry, source/dependency/input identities, smoke inputs/outputs,
validation evidence and this review, sealed with `COMPLETE.json`.
Earlier local smoke/capture directories remain unsealed, superseded implementation
work; they are not competing performance experiments or frozen references.

The baseline archive (486 manifest entries) and both Step 1 archives (28 and 23
entries) passed checksum verification. All 458 checked frozen runtime/model and
dependency files still match their root working-copy counterparts. Existing dirty
work was preserved. Production engine sources, databases, model artifacts,
publishing modes and workers were not changed by this implementation. No provider
calls, reserve-label inspection or public recommendations were made.

Verify the completed artifact from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_candidates verify Research/phase4-step2-candidates-2026-09-29
```

## What remains in Step 2

Build the chronological comparison execution around these primitives, with the
refit cadence declared before scoring. Fit on permitted earlier data, choose
hyperparameters on 2022 forward development folds, and compare frozen choices on
already-inspected 2023 development data. Preserve identical fixture membership,
fallthrough/failure coverage and paired scores with uncertainty. Combine changes
only after individual evidence supports a bounded combination manifest.

No candidate is yet recommended for production. Step 3's logistic comparator and
calibration, Step 4's supported-scope review and Step 5's later qualification retain
their places in the [five-step plan](phase4-full-probability-comparison-plan-2026-09-28.md).
