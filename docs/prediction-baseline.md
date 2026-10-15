# Preserving the prediction baseline

This implements the preservation portion of Phase 0 in the
[model improvement plan](ml-prediction-improvement-plan.md).
It does not change prediction formulas, retrain models, update release settings,
or alter existing published records.

## Preserved baseline — 2026-09-15

- Label: `baseline-2026-09-15-pre-model-fixes`.
- Capture completed at `2026-09-15T20:14:36Z`.
- System identity: `baseline-6fb58134e5a75cfea9eb6058`.
- Contents: 327 source/config files, 922 data/model files, two consistent
  database backups; approximately 958 MB in total.
- Verification: all six top-level artifacts passed checksum validation;
  all 187 recorded Match Reads replayed without exclusions or differences
  from their historical selected result indexes/roles/status.
- Snapshot tracking counts: 133 canonical published predictions, zero recorded
  settlements and zero closing-price comparisons. These are capture-time counts,
  which can differ from the earlier audit while the worker continues running.
- Five focused capture/verification tests passed, covering archive alteration,
  overwrite refusal, committed WAL data, failure handling, and replay using
  frozen source after the working copy changes.

The archive contains source as it stood at capture time. Documentation updates
recording this result were made afterwards; re-verification uses archived source.
Baseline preservation is complete. Full projection-input replay remains an
outstanding Phase 0 gate. Isolated candidate fitting/IO was implemented and
verified in the Phase 1 work on 2026-09-18; see [candidate features](prediction-features.md).

Re-verified on 2026-09-22: all six artifact checks and all 187 compiler selection
replays still pass. The archive was not modified. The canonical `odds_snapshots`
table had no rows in this audit, and saved selections do not contain the full
historical quote slates/profile inputs. Local historical recovery therefore
does **not** close the full numerical replay gate. See the
[Phase 1 recovery report](phase1-history-recovery-2026-09-22.md).

## Capture and verification

From the repository root, with data refreshes idle:

```bash
./.venv/bin/python Scripts/ops/prediction_baseline.py create \
  --label baseline-2026-09-15-pre-model-fixes
```

The label must be new. Existing baselines cannot be overwritten. Capture checks
for sufficient disk space and fails if an archived source/export changes during
the operation. A failed attempt is left in its directory without a completion
marker; do not use that directory as a valid baseline.

To verify an existing baseline without changing it:

```bash
./.venv/bin/python Scripts/ops/prediction_baseline.py verify \
  --baseline Index/prediction_baselines/baseline-2026-09-15-pre-model-fixes
```

Verification checks all artifact hashes and runs the saved Match Read compiler
against the recorded cases. It loads the archived compiler, so later working-tree
edits cannot silently change the baseline. Network connections are blocked during
this replay. Only verify trusted local baselines: replay executes archived Python.

## What is preserved

- Current source under `Scripts`, supported config files, test recordings, relevant
  documentation, and dependency declarations, including uncommitted file contents.
- Git revision/status, per-file SHA-256 digests, Python and installed dependency
  versions, and explicitly allowlisted non-secret environment settings.
- All JSON/JSONL/CSV exports under `Output`, root `Index` JSON files, saved ML
  models/metadata, raw JSON archives, and shadow JSON recordings.
- Independent, consistent SQLite online backups of `platform.db` and
  `predictions.db`, including committed WAL changes. These preserve fixtures,
  stored features, decisions, versions, publication/delivery records and tracking.
- Database coverage by competition/season, latest completed-fixture dates and
  fetch times, and recorded-publication tracking counts.
- Replay cases from stored canonical market results, with archived probabilities,
  prices, decisions, and historical selections retained unchanged.

The baseline lives under the already Git-ignored `Index/prediction_baselines/`.
Its directory is private to the local user. Database copies may contain application
user records: keep the bundle private. `.env` and API credentials are not copied
as configuration files. Do not commit or upload the bundle as a public artifact.

The 18 GB Chroma vector index and separate live-betting database are excluded.
Preserved exports are not claimed to be identical to the current Chroma collection.
This is a prediction baseline, not a complete disaster-recovery backup. Keep a
separate backup on another disk/storage service if protection against disk loss
is needed; this tool creates a local copy only.

## Replay boundary and honest limitations

The current real-data replay starts at **canonical market results → Match Read
compiler**. It verifies the selected result indexes, core/supporting roles,
selected probabilities/prices, alternatives, and game tags for the captured code.
It does **not** independently recompute those market probabilities or rerun the
market selectors from all original candidate odds.

Original publications may have used older policies. The capture records any
differences between their historical selections and the currently archived
compiler's selections; it never rewrites the original publication. Numerical
baseline expectations describe the code captured now, not a claim about an
unrecorded historical model version.

Full **historical profiles + full quote slate → projections → market selection**
replay remains a Phase 0 verification gap. The old records lack the necessary
input snapshots. Phase 1's shared feature/snapshot builder should supply this
boundary for new captures. Existing synthetic selector tests remain useful but
are not historical accuracy evidence.

SQLite backups are individually consistent. No transaction spans the database,
Output files, and Chroma. Capture times and coverage therefore describe each
source separately, rather than inventing a single universal data cutoff.

## Version and experiment contract

`manifest.json` assigns content-based identities to feature implementation,
saved model artifacts, probability policy, and selection policy. A composite
prediction-system identity also includes all archived source hashes and captured
configuration, so heuristic-only operation has an identifiable baseline.
These are baseline identities; propagation into production record fields is
Phase 2 work. Historical null model versions remain null.

Evaluate forecast cohorts separately:

| Cohort | Input cutoff | Reporting treatment |
| --- | --- | --- |
| Preliminary | Actual early evaluation time, before kickoff | Preserve first-publication decision and odds |
| Final pre-match | Actual later evaluation time, before kickoff | Keep amendments separate from first-publication results |
| Confirmed lineups | Verified lineup availability before evaluation/kickoff | Record lineup source and timestamp; never infer it from a label alone |

Every future experiment should use a new directory under
`Index/prediction_experiments/<experiment-id>/` and produce:

1. `manifest.json`: baseline/system versions, code/data hashes, seed, dependencies,
   hypothesis, forecast cohort, and permitted market/league scope.
2. `dataset-manifest.json` and `coverage.json`: source cutoffs, fixture IDs,
   available fields, missingness, exclusions and historical availability assumptions.
3. `feature-schema.json` and `splits.json`: ordered feature meanings/types and
   chronological train/tuning/calibration/test boundaries.
4. `predictions.jsonl`: fixture, cutoff, stage, versions, raw/calibrated probabilities,
   exact quote/settlement basis, decision and rejection reasons.
5. `metrics.json`: baseline/candidate comparisons, sample sizes, uncertainty,
   league/market/stage breakdowns and pre-match violations.
6. `candidate/`: immutable fitted model, calibrator and policy artifacts.
7. `review.md`: findings, limitations, promotion decision and rollback target.

These filenames specify the full experiment contract, not a claim that all
benchmark/calibration stages are implemented. The preservation tool cannot train
or publish. As of 2026-09-18, `prediction_candidate.py` provides isolated fitting
with training-only preprocessing and schema-checked inference. Legacy
`ml_edge.py` fitting/CV/raw-row prediction commands are retired; routine season
refreshes preserve saved models. Full chronological benchmarks and full
projection/quote/selection replay remain unfinished. A `shadow` flag alone is
still not a substitute for output isolation.

## Next implementation boundary

The shared input snapshot/feature contract and isolated candidate IO are now
implemented. Next resolve historical coverage and verify full baseline numerical
decision replay before drawing accuracy conclusions. Preserve this baseline even when
intentional fixes change its outputs; use it as a comparison, not a requirement
that corrected predictions remain numerically identical.
