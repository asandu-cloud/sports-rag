# Phase 4 research baseline and protocol freeze

Completed 28 September 2026, 16:46 UTC, after owner authorization.

- Control: `phase4-repaired-statistical-control.v1`.
- Protocol: [`phase4-evaluation.v1`](phase4-evaluation-protocol-2026-09-28.md).
- Immutable artifact: `Research/phase4-baseline-2026-09-28/control/`.
- Baseline identity: `5e55d49ac0b5b64ed85c9f43c5959dd3d623d0c753ca04e5522d814d8dd5e1f7`.
- Protocol SHA256: `a34cee8b56db3837ae89cb61efbd15d10372470b67b13a8896b9773bdd25ec4d`.

## What is frozen

The current repaired working-tree engine, including dirty/untracked source and
configuration, active model artifacts, declared dependencies, synthetic input
snapshots and deterministic output recordings. There are 468 preserved source,
configuration, documentation and model files, plus six allowlisted development
dataset files and the original dataset checksum manifest. All **486 artifact
entries** verified against COMPLETE.json; files were made read-only after capture.
A filesystem owner could deliberately change permissions, so hashes remain the
integrity authority. The capture command refuses an existing destination.

The control retains the installed **eight-match prior cutoff**, fixed goal rho
−0.10, current variance rules, selector/quality/settlement guards and heuristic
confidence. Actual gating against preserved model metadata resolves to **zero ML
weight** for goals, corners, cards and SoT. The strength-16 gradual-prior change
remains an isolated challenger. No probability candidate, calibrator or model was
promoted. Cards are preserved for behavioral comparison but are not qualified
for historical probability fitting by this freeze.

The source under `workspace/` is the authoritative frozen copy. Future working-tree
changes do not alter this control. Protocol amendments must receive a new version
and retain the old protocol and access history. The old Phase 0 baseline and all
completed candidate experiments were preserved.

## Replay and validation

**420 focused/regression tests passed.** The new freeze tests cover existing-output
protection, unsafe paths, dirty-byte preservation, manifest tampering/membership,
development access allowlists, deterministic real-engine replay, probability
normalization/coherence, full Asian vectors, variance propagation and sparse-input
guardrails. Existing probability, market-contract, prior-profile and participation
settlement tests also passed. Test logs/XML are beside the control directory.

Nine saved synthetic scenarios produce **63 canonical market results** covering
all seven markets. Scenarios include quarter lines, absent/one-sided variance,
high variance, sparse support, unavailable profiles, context adjustments and a
continental competition. Fresh-process replay against current source and frozen
source produces byte-identical output; a second frozen replay also matches.
Their shared SHA256 is:
`cb8de0e07926e76a9cb26d9a15e8c6ebe04fb19ec7c8518d8295a082c39280f9`.

The replay executes unchanged projection, season/recent variance blending,
combined variance resolution, score/count distributions, canonical selectors,
quality and release guards. It replaces external lookups with saved profile,
recent-statistic/recent-variance, profile-audit and context inputs. The harness
blocks dotenv loading, outbound connections, database access, protected data
reads and writes outside its designated replay output; it records and fails even
if imported code catches a prohibited operation. No IO violations occurred.
Shadow mode is set only inside the replay process; public process settings are
unchanged. The saved scoring weights and system identity make the replay boundary
inspectable.

This is **synthetic input replay of the numerical engine**, not reconstruction of
actual historical published recommendations. Profile lookup, raw-history feature
construction, provider availability, live lineup retrieval, and existing process
bundle reload are not certified by it. Historical full-engine probability
snapshots still need to be built from permitted evidence in the next batch;
missing contexts must remain disclosed. The earlier fixed-mean Poisson pilot is
not substituted for the full control.

## Frozen evaluation decisions

The full protocol fixes candidate families/grids, deterministic tie breaks,
primary proper scores, secondary reliability/interval diagnostics, identical
fixture comparisons, minimum support, grouped uncertainty, slice regressions and
failure handling. It keeps coherent goal-derived markets and full Asian outcomes.
The primary advancement gate requires at least 0.5% relative score improvement,
a paired 95% interval below zero and no supported slice regression over 2%, with
additional coherence/related-market checks. This is predeclared for the new
full-engine comparisons; previous pilot conclusions do not satisfy it by default.

2022 supplies forward development fitting/tuning; 2023 is already-inspected later
development comparison. The consumed 2024 confirmation is excluded from new v1
fitting/selection. Jan–Mar 2025 is reserved for later calibration fitting and
Apr–Jun 2025 for one frozen-method qualification. Jul 2025–Jun 2026 stays untouched
for the completed probability and selection-system final test. From Jul 2026
remains the prospective reserve. The quarter split and distinct support minima
are fixed before reserve inspection; insufficient cohorts stay insufficient.

**No reserved outcomes were opened by this freeze.** Only explicit development
feature/label files and outcome-free dataset manifests were copied; no mixed
history archive, snapshot database or reserved outcome store was copied/read.
The access ledger records this boundary. This command did not run fitting,
provider calls, embeddings, refreshes, publications or database backups. Source,
active-model and development-input fingerprints were checked unchanged before
sealing. Canonical databases, worker modes and refresh markers were not touched.

## Verify and continue

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_baseline verify Research/phase4-baseline-2026-09-28/control
```

The verifier checks both artifact membership and every checksum. Do not write
new runs into the control directory. Replay helpers and their exact source are
preserved under `workspace/Scripts/ops/`; dependencies and protocol are preserved
alongside the inputs. Reproducing results on another runtime requires restoring
compatible dependencies and checking numerical parity, not merely copying Python.

Next: implement historical full-engine probability input reconstruction and the
predeclared development comparisons against this control. Advance card evidence
qualification separately and keep richer lineup effects as the documented later
workstream. Neither the frozen reference nor the passed tests establish improved
accuracy, profitability, live-match acceptance or eligibility for public promotion.
