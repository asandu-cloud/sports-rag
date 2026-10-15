# Phase 1: candidate pre-match features

Status: the shared candidate feature pipeline and isolated fitting/inference
workflow are implemented, 2026-09-18. Verified local historical recovery was
applied on 2026-09-22; the new 5,083-fixture export replayed in full. Ambiguous
legacy zeros/card statistics and the full baseline numerical-decision replay
gate still need work. See the [recovery report](phase1-history-recovery-2026-09-22.md).
The subsequent [bounded diagnostic](phase1-history-diagnostic-2026-09-22.md)
confirmed 2025 card-label selection bias and saved provider comparisons without
changing production data. Phase 2 can proceed with explicit missing-data and
market-definition gates; unrestricted card modelling is not signed off.
This is a
candidate/research path, not a newly activated prediction model. It does not
change website recommendations or Discord bets.

## Implemented

- `Scripts/rag_ingest/core/model_features.py`: one pure, versioned snapshot and
  vector builder, used by dataset generation and candidate inference.
- `Scripts/data_platform/features/model_dataset.py`: read-only canonical SQLite
  adapter, provider-ID pairing, quarantine report, and prospective fixture capture.
- `Scripts/rag_ingest/core/candidate_features.py`: shared estimator input adapter.
  Inference requires the exact feature names, schema, registry, and policy used
  for fitting. Nulls become NaN, not invented zeros; no imputer is fitted here.
- `Scripts/ops/prediction_features.py`: isolated export, capture, and offline
  replay. Writes only to a new `Index/prediction_experiments/<id>` directory;
  overwrite, path traversal, and symlink escapes are rejected. No training,
  API calls, database updates, active-model writes, or publication takes place.
- `Scripts/ops/prediction_candidate.py`: explicit candidate-only fitting and
  inference. Its median imputer, scaler, and Ridge estimator are fitted only on
  the selected training subset. Bundles are unvalidated and cannot be promoted
  or published by this runner. It is not the Phase 3 benchmark suite.

The dataset contains ordered features, labels kept separate from inputs,
missingness indicators, fixture/season/competition coverage, profile evidence,
source hashes, and an archived pure builder. Replay uses that archived builder,
not whatever implementation happens to be in the working tree later. A
`COMPLETE.json` marker is written only after a successful export. Failed exports
remain inspectable without a completion marker. Only replay trusted local
experiments: verification executes their checksummed archived Python builder.

## Explicit modelling choices

| Area | Candidate policy |
| --- | --- |
| Identities | API-Football fixture/team IDs; no fuzzy names. Provider season is explicit. |
| Historical profile | Current season plus immediately preceding season only. Extra imported seasons are not pooled into each profile. |
| Early-season prior | Existing strength-eight policy: `n/(n+8)` current weight below eight current matches; current-only from eight matches onward. |
| Promotion/relegation | No unadjusted transfer of rates or Elo between domestic leagues. Missing comparable history remains missing. |
| European context | Domestic anchor when uniquely resolved from current-season team-ID history. Prior-season membership alone is insufficient after potential promotion/relegation. Add 20% continental context after three current continental matches. Counts remain separate. |
| Recent form | Five earlier matches in each profile's competition; current/prior and European blending follows the recorded policy. |
| Rest | Time between known kickoffs across the 13 covered competitions, capped at 14 days. Uncovered cup fixtures are not silently assumed present. |
| Venue | Separate home/away observations, with missingness when that venue has no observations. |
| Elo | Sequential market-specific Elo within the target competition, K=32, initial 1500, retaining 75% of the deviation at each provider-season transition. Uses only the current/prior history window. |
| Referee | Exact stored referee label, earlier completed matches in the same competition; no current global referee aggregate. Late-observed assignments are withheld in observed mode. |
| H2H / composite indices | Not included in this new schema. No duplicate/constant home flag or repeated production-rate copies. Legacy composite indices are not substituted with same-match values. |
| Missing statistics | Null plus an indicator. A real zero is retained, including zero-card/corner targets. Both card components must be known to derive total cards. |
| Match period | FT history/labels only. AET/PEN totals are excluded because canonical full-time period details are insufficient. |

These are recorded candidate assumptions, not evidence of better predictive
performance. Changes require a new candidate contract and evaluation.

## Historical availability is not fabricated

`observed` mode requires the latest stored fetch/stat-update timestamp to be
at or before the cutoff. A historical import fetched after a target match
therefore cannot masquerade as a contemporaneous observation. Canonical tables
are mutable, not a complete vintage history; this conservative rule can exclude
records that were originally available but later updated.

`assumed_final` is an explicit retrospective backfill mode. It assumes an FT
match's presently stored statistics were available three hours after kickoff.
That is **not proof of their historical availability**, nor protection against
later corrections. Both modes enforce that conservative completion boundary,
exclude the target fixture itself, and reject post-kickoff forecast cutoffs.
Snapshots label which availability rule was used. Historical referee assignments
in assumed mode have the same retrospective limitation.

## Commands

Export a new research dataset; use a new experiment name every time:

```bash
./.venv/bin/python Scripts/ops/prediction_features.py export \
  --experiment phase1-features-next \
  --season 2025 2026 --availability assumed_final
```

Optional `--league EPL UCL` limits target fixtures; context can still come from
other registered competitions. `--limit 50` creates an explicitly labelled
smoke-test subset. Omitting `--availability` uses strict `observed` mode.

Capture an upcoming fixture using its actual API-Football ID, before kickoff:

```bash
./.venv/bin/python Scripts/ops/prediction_features.py capture \
  --experiment next-prospective-capture --fixture-id ACTUAL_FIXTURE_ID
```

Replace `ACTUAL_FIXTURE_ID` with an integer. This freezes candidate numerical
inputs; it does not calculate odds, selections, or publish a Match Read.

Replay a saved export or prospective capture offline:

```bash
./.venv/bin/python Scripts/ops/prediction_features.py replay \
  --experiment Index/prediction_experiments/phase1-features-2025-2026-v1
```

The final verified export for the current implementation is
`phase1-features-2025-2026-verified`; use that name for new candidate fits.
Earlier exports remain replayable with their own frozen implementation.

### Isolated fitting (not production training)

This command is available for controlled research/software checks. It does not
report a validation score or prove improved predictions. Choose the training
cutoff **before** examining held-out results; formal split design and model
comparisons remain Phase 3 work.

```bash
./.venv/bin/python Scripts/ops/prediction_candidate.py train \
  --dataset Index/prediction_experiments/phase1-features-2025-2026-verified \
  --experiment candidate-goals-software-check --market goals \
  --train-through 2026-08-31T23:59:59+00:00
```

It stores the training fixture/snapshot IDs, cutoff, dependency versions, source
hashes, feature contract, and candidate model under the new experiment directory.
Five eligible rows is only a software minimum, not qualification for deployment.
Known post-cutoff labels are withheld in observed mode. Median imputation is
fitted on the training rows only; all-empty training columns use the imputer's
zero fill with the separate missingness flag retained. No global imputation,
hyperparameter selection, held-out scoring, or automatic model activation occurs.

Inspect that candidate against a saved pre-match snapshot:

```bash
./.venv/bin/python Scripts/ops/prediction_candidate.py predict \
  --candidate Index/prediction_experiments/candidate-goals-software-check \
  --snapshot Index/prediction_experiments/phase1-live-fixture-1558635/input-snapshot.json
```

The response is explicitly unvalidated/unpublished. Only load trusted local
candidate bundles: `joblib` model loading executes serialized Python objects.

### Routine refresh behaviour

The normal `refresh-season` command continues to update results, profiles,
features and embeddings, but **no longer fits or replaces saved ML models**.
Its old `--skip-training` flag remains compatible and is no longer required.
Direct `pull_season.py --retrain-ml` and `ml_edge.py --train`,
`--train-cumulative`, `--cv`, and raw-row `--predict` commands fail early with
migration guidance. Their old source functions remain for historical inspection;
they are not supported candidate APIs. The existing production inference loader
continues serving the existing artifact schema until an approved promotion.

## Local verification and coverage

Verification completed: 167 tests and four subtests passed in the targeted
platform/baseline/refresh/market regression run. All 2,286 final exported vectors
and the current prospective fixture snapshot replayed successfully. The original
Phase 0 archive still verifies, including all 187 stored selection replays.

The final two-season export is
`Index/prediction_experiments/phase1-features-2025-2026-verified` (2,286 fixtures,
148 ordered features including missingness flags).
The current prospective snapshot is `phase1-live-fixture-1558635`, captured before
kickoff on September 18. `phase1-features-2025-2026-v1` and
`phase1-live-fixture-1575167` retain earlier frozen-source verification cases.
The earlier `phase1-features-2026-initial` was a development
export before frozen-source replay was finalised; use the complete v1 export
above for the supported replay command.

At export time:

- All 13 competitions are represented in the coverage report, including empty
  cohorts. There are 489 completed 2026 fixtures and 1,797 completed 2025 fixtures.
- The five newer domestic leagues supply the 2025 canonical history. The older
  leagues' prior-season data is largely still in legacy exports rather than
  canonical fixture/team-ID rows. This is **not a claim that their old data was
  deleted**; it is a source-integration gap for this candidate path.
- UEL/UECL have no completed canonical training rows in this export.
- Many rows have unknown card components. They are excluded from card labels
  where the total cannot be established, not converted to zero.

Coverage's `usable_features` means a known target and production rates for both
teams, not sufficient sample size, acceptable accuracy, or permission to deploy.

## Remaining data/replay gates and later modelling work

1. The verified historical import and bounded diagnostic are complete. Resolve
   remaining ambiguous zeros/provider-null card semantics and review cohort
   selection bias before approving those markets for modelling. Do not equate
   provider null with zero or generic cards with yellow-only bets.
2. Complete full projection/quote/selection replay for newly captured baseline
   inputs. Feature-vector replay is now available, but does **not** close the
   Phase 0 full-numerical-decision replay gate. Historical missing inputs remain
   unrecoverable unless an authoritative archive supplies them.
3. Build Phase 3's chronological benchmark suite on the shared contract. The
   retired legacy CV is not a valid replacement. Add qualified model promotion
   and production-schema routing only at the later controlled-rollout stage.

Only synthetic test models were fitted by this batch; no real-data model was
fitted, promoted, or given a larger production blend weight. Chronological
benchmarks, probability calibration, selection-policy
changes, and live rollout remain later phases.
