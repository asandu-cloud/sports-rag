# Qualified corner/SoT components carried into research — 2 October 2026

The owner authorized carrying the qualified corner and shots-on-target
distributions into the research setup before the remaining weight work and
Phase 5. This is complete. It does not activate a public model or repeat the
qualification experiment.

## Retained configuration

| Component | Research configuration |
| --- | --- |
| Corner mean | Unchanged frozen statistical control |
| Corner distribution | Negative binomial, alpha **0.025** |
| Shots-on-target mean | Unchanged frozen statistical control |
| Shots-on-target distribution | Negative binomial, alpha **0.01** |
| Variance | `mu + alpha * mu^2` |
| Goal means and coherent derived markets | Unchanged frozen control |
| Additional probability calibration | Identity |
| Cards | No new card component activated or qualified by this bundle |

The preceding finalist recipe contained the rejected goal-strength change and
the inconclusive corner Over 9.5 sigmoid alongside the successful components.
The new bundle carries only the two qualified distributions. It references the
original qualification decisions and frozen control by checksum, without
rewriting either archive or marking the whole old recipe as qualified.

The qualification remains the [29 September result](phase4-qualification-results-2026-09-29.md):
1.108% lower corner count NLL and 2.125% lower SoT count NLL on 888 fixtures each,
over 20 weeks. No new accuracy, betting-return or production-readiness claim is
created by this carryover.

## Scope and limitations travel with the bundle

The original pooled research comparison applies to EPL, LaLiga, SerieA,
Bundesliga and Ligue1 with both teams having at least eight current-season
matches. Other leagues and fewer than eight matches retain the exact control.
The adapter also retains control for unverified forecast stages and keeps
unavailable projections unavailable. An observed zero is not missing.

For reproducibility, the research reference preserves the original pooled
eight-match boundary. It does **not** silently narrow that historical claim or
certify every slice. Records explicitly retain the insufficient standalone
support for the 8–15-match stage and partial/missing xG. The later evidence covers
reconstructed immediately-before-kickoff forecasts in provider season 2024;
earlier forecasts, confirmed-lineup amendments and broader seasons remain
unverified. Unsupported conditions require control fallbacks and further evidence
before any public activation. Every carried record is non-public and explicitly
not production-qualified.

The new consumer only accepts a frozen-control rate record with matching
fixture/snapshot identity and the pinned baseline identity. A rate record
labelled as a weight/strength candidate is rejected. Changing the means, scope,
distribution parameters or calibration requires a separate configuration and
validation; the saved qualification cannot be inherited by such changes.

## Usable research artifacts

Main bundle:
`Research/phase4-retained-components-2026-10-02/`

- `bundle.json`: versioned components, qualification decisions, scope limits,
  original control/qualification identities and explicit release restrictions.
- `development-reference.jsonl`: **3,095** saved 2023 fixture forecasts with
  unchanged control means and the retained variance substitutions where applicable.
  These contain no target labels, fitted goal-strength output or binary sigmoid.
- `evidence/`: preserved qualification decision, reviewed component decisions,
  method-lock and original artifact checksum manifest. These are provenance
  records, not instructions to activate every old finalist component.
- `source/`, `manifest.json`, `verification.json`, `production-preservation.json`
  and `COMPLETE.json`: implementation, dependencies, exact parity results,
  preservation evidence and artifact seal.

The immutable bundle identity is
`81637379a09cb9e1ec79e4bf6eaa4c142c7196c12e6259c199508ac1fbb6b312`.
Its completion seal SHA256 is
`8c7b16a57692a8099b9204f87a02dcf5946794fbfff4e0757bd2a7d906f2ce40`.
The original frozen control remains
`Research/phase4-baseline-2026-09-28/control/`.

Validation and downstream eligibility sidecar:
`Research/phase4-retained-components-validation-2026-10-02/`

`usage-contract.json` binds the sidecar to the exact research bundle.
`development-eligibility.jsonl` preserves the existing per-market eligibility and
exclusion reasons. **Join on both fixture ID and snapshot ID and require market
eligibility before scoring.** The 3,095 forecast records are not 3,095 qualified
targets in every market: 3,028 corners and 3,037 SoT targets are eligible overall;
1,504 and 1,521 respectively have the retained component within the research
scope. This carryover does not expand training eligibility or reinterpret the
separate card target contract.

## Verification

- **84 tests passed** across the new carryover tests and existing finalist,
  qualification and candidate suites. They cover mean/goal/card preservation,
  exact control fallbacks, support-limit annotations, missing versus zero,
  independent negative-binomial probabilities, rejection of later periods and
  modified candidates, bundle tampering and publication restrictions.
- All **3,095 development records replayed exactly**. Independent comparison
  with the sealed earlier controls confirms identical goal/team total means,
  unchanged fallback variances and the exact retained variance formula.
- All **2,136 saved qualification count distributions matched exactly**:
  888 active corners plus 177 fallbacks, and 888 active SoT plus 183 fallbacks.
  This replays already saved probabilities; no outcome labels were decoded and
  no qualification scores, fits, thresholds or decisions were recomputed.
- Existing source archives verified successfully. **458 existing source, test,
  model and dependency files** remained unchanged during preparation.
- No API calls, database/profile writes, embeddings, active-model changes,
  public deliveries or reserved final-system/prospective outcome access occurred.

Replay from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_retained_components replay \
  --output Research/phase4-retained-components-2026-10-02
```

The reusable numerical adapter is
`Scripts/data_platform/features/phase4_retained.py`; the preparation/replay command
is `Scripts/ops/phase4_retained_components.py`. Both are research-only additions;
existing publishing and archived experiment source was not edited. The adapter's
ordinary input boundary remains pre-2024 development data. The separate parity
check only reads the already saved January–June 2025 forecasts.

## Next use

Use this retained configuration alongside the original frozen control in the
remaining development work and Phase 5. Any altered-mean combination is a new
challenger, with its own identity and evaluation requirements. January–June 2025
remains consumed evidence. July 2025–June 2026 remains reserved for the completed
probability and selection system. Weight optimization, Phase 5 implementation,
shadow acceptance and public promotion were not started by this carryover.
