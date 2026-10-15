# Phase 3 Batch B — benchmark runner, 26 September 2026

Status: **implemented and verified locally, including a bounded run on real
historical development data**. All five agreed numerical model families and
both baselines completed the same first development fold. This is software
validation, not a claim of improved accuracy, profitability or deployment
eligibility. The full development search, learned blends and confirmation
remain later batches.

The owner authorized building the benchmark runner after completing Batch A.
No new football data was collected. The frozen team dataset, baseline archive,
active model artifacts and production Python dependencies were preserved.
Player-prop implementation remains separate future work.

## Preflight and scope

The preserved `baseline-2026-09-15-pre-model-fixes` manifest and all its
archived artifacts passed checksum verification. The complete Batch A export
also passed integrity verification before and after this work:

`Index/prediction_experiments/phase3-dataset-2026-09-26/`

Dataset ID:
`4362324537087ea2654d27538614166b305647112fa4749e8222134acdf52de0`

The source feature builder, research feature mask and competition registry
remain compatible. A new feature export was not required. Changes to the
benchmark CLI and documentation do not change the saved numerical feature
contract; their new fingerprints are recorded with the runner artifacts.

The strict development reader validates 19,648 initial-training/development
rows, including rows ineligible for some targets. It reads only allowlisted
metadata, feature and label files. It does not open the audit history,
snapshot database, confirmation, calibration or final-system target stores.
It verifies fixture/season/competition identity, chronological memberships,
eligibility, source classes, missingness, support and label availability.

## Implemented runner and comparators

Research code lives under `Scripts/data_platform/features/benchmarks/`.
The CLI extends `Scripts/ops/prediction_benchmark.py` with separate
`prepare-baselines`, `smoke` and `verify-run` operations. Existing Batch A
prepare/verify/replay operations remain available.

- **Historical league/venue average:** earlier complete home/away counts in
  the current and immediately previous provider seasons. Below five complete
  league observations it uses the declared pooled same-season fallback;
  no support returns unavailable. This fallback threshold is part of the
  frozen baseline specification, not a model-qualification threshold.
- **Statistical reconstruction:** production core arithmetic, six-match
  exponential recent form, xG/venue rules, eight-match prior behavior and
  European/domestic profile blending using exact historical identities.
  ML contribution is zero. Saved metadata identifies omitted lineup/player,
  standings/stakes, knockout and other live adjustments. This is not an exact
  replay of predictions historically published to users.
- **Five estimator adapters:** Ridge, Poisson regression, LightGBM, XGBoost and
  CatBoost, all producing expected counts on the original scale. Ridge retains
  raw outputs and records nonnegative clipping. No expected count is rounded.
- **Weighted preprocessing:** lower weighted median on an exact half-weight
  tie; training-only weighted mean/population-variance scaling for linear
  models; imputation without scaling for trees. All-missing training columns
  use a recorded zero placeholder while retaining explicit missing indicators.
- **League effects:** the 126 reference features remain unchanged. Ridge and
  Poisson add 52 predeclared interactions: four market-specific own-production/
  opposing-concession rates crossed with the 13 competition indicators.
  Interactions are built after imputation and before scaling. Trees receive
  the original 126 columns and may learn league interactions themselves.
- **Isolation and persistence:** immutable new experiment directories,
  checksums including nested candidate manifests, exact market/family/schema
  binding, dependency checks and saved-transform reuse. A process-visible
  runtime guard rejects network/database access, protected data reads and
  writes outside the experiment, including from Python worker threads.
  It is a guard for trusted research code, not a security sandbox.

Training eligibility uses the agreed 2,000-fixture/1,000-ESS minima, with
5,000 fixtures/2,000 ESS for tree families and 500 validation fixtures.
Failures or insufficient cohorts stay visible and withhold completion;
methods cannot improve reported performance by silently dropping fixtures.

The baseline preparation operation separately reads the frozen audit input,
filters history before confirmation and enforces each forecast's cutoff. It
replays the saved snapshot identity and writes **46,231 market/fixture rows**
to a development-only sidecar. Ordinary fitting consumes that sidecar without
opening audit history. The original export is never extended in place.

## Frozen software-smoke specification

The bounded run used `development-01`: fitting cutoff **1 January 2022 UTC**,
validation from that date until **1 July 2022 UTC**, exclusive. Training labels
had to be available strictly before fitting. All-history training and a fixed
365-day elapsed-time half-life were selected before scoring to exercise the
weighted path. Weights normalize to training mean one; evaluation fixtures
receive equal weight.

Fixed recipes: Ridge alpha 50, Poisson alpha 1, LightGBM 7 leaves/minimum child
size 50, XGBoost depth 3, CatBoost depth 3. All trees use 300 iterations and
learning rate 0.03 under their declared Poisson objectives. Seed is 42 and
CPU limits are two threads. These are smoke recipes, not development winners.

| Market | Training fixtures | Weighted ESS | Validation fixtures |
|---|---:|---:|---:|
| Goals | 8,909 | 7,391.81 | 1,942 |
| Corners | 8,823 | 7,322.85 | 1,941 |
| Shots on target | 8,495 | 7,064.36 | 1,941 |

All seven methods use exactly the same validation fixtures within each market.
The counts overlap across markets and must not be added as independent games.
All **15 candidate fits** succeeded, producing **40,768 forecast records**
including both baselines. The measured smoke operation took about 17 seconds
on this Mac; baseline preparation and dependency setup were separate work.
This timing does not estimate the cost of the full search and ablations.

Reports contain RMSE, MAE, signed bias, R² where defined, and common uncalibrated
Poisson log loss, ranked probability score and 50/80/95% interval diagnostics.
The positive mean floor and numerical tail limits are explicit and shared
across methods. These are not fitted dispersion or calibrated betting models.
Scores are retained in the experiment; no model is selected from this smoke.

## Environment and artifacts

The isolated environment is:

`Index/research_environments/phase3-benchmark-v1/`

It uses CPython 3.13.3 on macOS arm64, NumPy 2.3.3, SciPy 1.17.0,
scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.4.1 and CatBoost 1.2.10.
All package versions, including transitive dependencies, are pinned in
[`requirements-phase3.lock.txt`](../requirements-phase3.lock.txt).
[`requirements-phase3.in`](../requirements-phase3.in) records direct dependency
intent. Production `requirements.txt` and `.venv` were not changed by this task.
Native fit/predict/serialization checks passed for all five families; another
OS or runtime must repeat those checks rather than inherit this host's result.

Artifacts:

- [Baseline preparation](../Index/prediction_experiments/phase3-baselines-2026-09-26/manifest.json):
  three checksummed artifacts, with scoring/source identities and per-row evidence.
- [Smoke report](../Index/prediction_experiments/phase3-runner-smoke-2026-09-26/report.json):
  49 checksummed artifacts, including the frozen specification, environment,
  predictions and 15 isolated bundles.
- [Preservation and validation](../Index/prediction_experiments/phase3-runner-validation-2026-09-26/validation.json):
  test results, independent acceptance and protected-artifact verification.

Every candidate stores the ordered schema, fitted preprocessing, estimator,
family/package/output-scale metadata, training fixture/snapshot IDs, target
and weight hashes, cutoff, support, source hashes and specification identity.
Both `publication_enabled` and `promotion_allowed` are false. Loading these
trusted local bundles verifies checksums and contracts before deserialization.
They have no integration with the active model loader.

## Validation

**268 tests passed:** 139 new benchmark tests in the isolated research
environment and 129 relevant existing feature/dataset/split regressions in
the existing environment. The latter include ingestion-related imports that
are deliberately not part of the minimal research dependency set.

Checks include weighted hand examples, held-out perturbation invariance,
all-missing features, zero/missing distinction, strict label cutoffs, archive
tampering, fixture grouping, family support gates and exact saved-model parity.
Baseline tests include 240 numerical parity cases against the actual pure
production arithmetic extracted without importing live wrappers. Tree tests
check transformed predictions against native raw margins; CatBoost's Poisson
log output is not accidentally scored as an expected count.

Independent review verified all 40,768 forecast records, common evaluation
membership for all seven methods, all 15 training membership/label/weight
hashes, pre-cutoff availability, source linkage and disabled release flags.
Every saved candidate reproduced its in-memory validation predictions exactly.
All 49 smoke artifacts and all three baseline-sidecar artifacts passed checksums.

Final preservation checks verified all original Batch A artifacts, the
preserved baseline archives, all ten active-model files and the unchanged
production dependency inventory. No canonical database writes, provider-data
requests, embedding requests or publications were performed by this work.

## Commands and next batch

Read-only verification of the completed smoke:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py verify-run \
  --experiment Index/prediction_experiments/phase3-runner-smoke-2026-09-26
```

To repeat the bounded smoke, choose a **new** experiment name. This trains
isolated candidates and writes new research artifacts; it does not publish:

```bash
Index/research_environments/phase3-benchmark-v1/bin/python -B Scripts/ops/prediction_benchmark.py smoke \
  --dataset Index/prediction_experiments/phase3-dataset-2026-09-26 \
  --baselines Index/prediction_experiments/phase3-baselines-2026-09-26 \
  --experiment phase3-runner-smoke-repeat-01
```

The baseline sidecar already exists and need not be rebuilt routinely. A new
baseline recipe or source requires a new sidecar name and recorded fingerprints.
Never overwrite these experiments or run package installation into production.

**Batch C is next:** implement/run nested forward development, the declared
156-recipe main grid, OOF-learned blends, history comparisons, ablations,
grouped uncertainty and model selection. Before that scoring, freeze the
approved three-/five-season profile arithmetic and its bounded paired
experiment allocation; the current export is the two-season reference.
Batch B does not implement the full search/resumption or confirmation-opening
operations. Those are explicitly later work, not hidden smoke functionality.

The four-year versus all-history comparison remains indistinguishable at the
current permitted development cutoffs. Five newer leagues still lack qualifying
recent period evidence, and some European evaluation slices remain sparse.
Cards remain excluded. Neither the confirmation nor the final-system test
has been evaluated, and no deployment qualification or betting ROI is claimed.

Separately, the existing refresh-incomplete marker was still present at the
final check. This offline work did not retry or repair that operational refresh,
and it did not remove the marker. Its recovery remains separate from Batch B.
