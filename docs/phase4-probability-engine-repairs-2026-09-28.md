# Phase 4 probability-engine correctness repairs

Date: 2026-09-28.

Status: **installed in the publishing checkout on 2026-09-28 following explicit
owner approval**. All 18 installed files match the reviewed repair hashes. This
is a correctness repair and a coherent, unfitted baseline for Phase 4, not a
completed calibration experiment or a claim of better predictions.

The repair was initially prepared and tested separately because the scheduled
worker executes the main checkout directly. Installation briefly unloaded the
idle Match Read and measurement jobs, applied the verified patch, checked the
installed system fingerprint, and restored both jobs using their unchanged
plists. Both restarted cycles exited with code zero while deferring work behind
the existing incomplete-refresh gate. No continuously running Python web or
Discord server was found to restart.

**Operational blocker:** the latest daily refresh log reports API-Football's
`/fixtures` error: "Free plans do not have access to this season, try from 2022
to 2024." This is the provider response, not a verified diagnosis of the owner's
subscription or configured credential. The incomplete marker was preserved;
no refresh retry or provider collection was initiated during this installation.
Consequently, installation does not establish real-fixture/publication acceptance.

The preserved package is
[Research/probability-engine-repair-2026-09-28](../Research/probability-engine-repair-2026-09-28/).
It includes the original and repaired source, forward and rollback patches,
hashes, dependencies, test evidence and the
[installation record](../Research/probability-engine-repair-2026-09-28/installation.json).
Active ML artifacts, scheduler configuration and refresh-gate files were
verified unchanged. Existing recommendations and databases were not migrated.

## Implemented changes

| Audit finding | Repair | Evidence and limits |
| --- | --- | --- |
| Swapped home/away intensities in the Dixon–Coles low-score correction | `(1,0)` uses `1 + away_mean * rho`; `(0,1)` uses `1 + home_mean * rho`. Invalid rates/rho fail explicitly. | Asymmetric-rate tests verify the formulas, nonnegative probabilities, unit mass, Poisson marginals and home/away symmetry. |
| Score grids discarded appreciable tails and used different cutoffs | Expand the grid until omitted joint mass is at most `1e-10`; old `max_goals` argument now specifies a minimum extent. Invalid or excessive grids fail explicitly. | Tests compare old six/seven-goal starting grids and independently calculate omitted mass. |
| Normal handicap fallback produced negative pushes and missing mass | Integrate integer-margin probabilities, including both infinite tails, and classify each margin directly into full/half wins, pushes or losses. | Independent settlement enumeration across 25 handicap lines and three margin distributions, including EV at different odds. |
| Spread selection lost prediction date and context | Forward cutoff, league context and knockout context; capture one score matrix per selector invocation instead of reloading for every quote. | Spies verify the exact arguments and call count. |
| Goal markets used conflicting distributions | Canonical goal totals, BTTS, winner and spreads derive from the same corrected Dixon–Coles distribution. A JSON-safe score snapshot participates in the existing fixture/context/version cache. | Cross-market identities and cold/warm cache tests. This deliberately changes goal-total pricing from the old independent negative-binomial total to an **unfitted coherent baseline**. Superiority is not established. |
| Knockout context reached totals but not all derived goal markets | Propagate it through the common team-goal split, all four derived helpers and compatibility wrappers. | Synthetic context/cutoff propagation tests. |
| Recent goal variance actually measured xG variance | Use actual scored goals from the same score parser as seasonal goal variance, preserving observed zero and missing scores. The legacy recent-variance wrapper delegates to core. | Synthetic scores with constant, deliberately different xG values; missing scores remain missing. This does not estimate conditional dispersion. |
| Totals paired incompatible market definitions | Pair on bookmaker, fixture, exact market key, period and settlement definition. Exclude yellow-only, red-only and booking-point totals from generic card extraction. | Mismatched market/period/rule/fixture cases cannot qualify through a false pair. Legacy extraction delegates to core. |
| Zero-handicap opposite sides could not pair | Match opposite signed lines including zero; reject unrelated participants when fixture teams are known. | Draw-no-bet pairing and contract-mismatch tests. |
| Spread “model probability” depended on the bookmaker price | Use the same stake-weighted, non-push-equivalent probability as Asian totals; continue calculating EV from all five settlement outcomes. | Change the price while holding the distribution fixed: probability remains fixed, EV changes. |
| Canonical results discarded probability meaning and settlement probabilities | Add optional `probability_basis`, `settlement_profile` and `probability_version` fields to `Decision`, plus optional quote settlement definition. Validate profile support, sum and consistency with the equivalent probability. | New records use `market-probability.v2`; old records round-trip without inferred values. Existing JSON storage needs no database migration. |
| Delivery called an Asian price-comparison probability a generic outcome probability | Carry the fields into persisted-card responses; label them explicitly in website and Discord output. Also preserve missing website probability as unavailable rather than zero. | Persistence-to-card test, existing delivery tests and JavaScript checks. |

The corrected low-score formula follows the Dixon–Coles formulation reproduced
in [§2.1 of this methods paper](https://arxiv.org/pdf/2307.02139). Its use does not
establish that fixed `rho = -0.10` is the best value for Spix.

The outer `market_result.v1` serialization remains compatible: optional new
probability fields have their own version and are omitted when absent in old
records. Historical records and initial publications are not rewritten.
Existing no-bet, quality, release and same-game controls remain in place. The
legacy variance assessment still supplies the previous conservative confidence
cap where applicable; the reported goal variance comes from the actual shared
score distribution. Context explicitly describes calibration as unfitted and
confidence as heuristic.

## Validation

- **541 tests passed, plus four regression subtests**, across the 27 modules
  listed in [test-files.json](../Research/probability-engine-repair-2026-09-28/test-files.json).
- Includes 160 independent count-distribution/Asian-total checks against SciPy
  and direct half-stake settlement, protecting the existing correct arithmetic.
- Covers probability math, selectors, canonical results, temporal profiles,
  cache behaviour, Match Read compilation, persistence, delivery, release gates,
  worker cycles and the existing settlement/tracking contract.
- Existing `pipeline_v1` synthetic expected records are unchanged. The replay
  harness supplies the additional score-distribution input from the explicitly
  frozen synthetic goal mean. These are selector/delivery regressions, not
  original historical prediction replays.
- JavaScript syntax and four probability-label checks passed.
- `git apply --check` passed against the current checkout. Forward and rollback
  patches were applied in a temporary directory; all 18 affected file hashes
  matched in both directions. This round-trip check preceded the subsequently approved installation.
- Before installation, all original `Scripts/` files matched their pre-repair hashes.
  After installation, the 18 repaired files match their reviewed target hashes.
- **275 focused tests passed against the installed checkout** with network access
  blocked and dotenv loading disabled; JavaScript syntax and `git diff --check`
  also passed. The prior 541-test suite remains the full repair-validation run.
  See [installed test results](../Research/probability-engine-repair-2026-09-28/installed-test-results.txt)
  and [worker verification](../Research/probability-engine-repair-2026-09-28/worker-verification.json).

Tests ran with network connections blocked and dotenv loading disabled. Provider
collection, training, embeddings, production publication and reserved-outcome
inspection were not performed. Test databases are disposable and isolated.
The suite is focused regression coverage, not a claim that every repository
integration or a real published matchday has been validated.

Saved results: [text](../Research/probability-engine-repair-2026-09-28/test-results.txt),
[JUnit](../Research/probability-engine-repair-2026-09-28/test-results.xml),
[patch validation](../Research/probability-engine-repair-2026-09-28/patch-validation.txt).

## Remaining work after installation

1. Evaluate the installed corrected baseline under the Phase 4 protocol. The
   coherent goal-total baseline and corrected handicap semantics can change probabilities, EV,
   confidence and selection membership. They have not been evaluated for accuracy
   on real outcomes. Prepare permitted offline/shadow comparisons in the Phase 4
   protocol; do not open the final-system reserve to choose a distribution.
2. Card target qualification remains open: generic provider card names and a
   matched quote pair do not prove equivalence to the weighted participation
   policy. This repair does not change yellow/red historical labels or resolve
   historical selection bias. It prevents the concrete yellow-only/generic
   pricing mismatch; it does not certify all card bookmaker rules.
3. Fit/evaluate dispersion, low-score dependence, probability calibration and
   confidence support on the agreed development/calibration periods. Fixed rho,
   other-market variance floors and heuristic confidence remain unfitted.
4. Specialist/live/unit-bet paths retain separate legacy behaviour. In particular,
   the unit-bet/parlay path still uses the legacy EV-equivalent helper. This batch
   does not claim that every specialist probability consumer is migrated.
5. Resolve the separate provider-access/failed-refresh blocker before expecting
   new published cards or settlement work. Do not bypass refresh locks or markers.
   Validate real-fixture behaviour once the ordinary data gate clears. The
   source/data cache fingerprint changes naturally with the installed code.
6. Use the saved rollback for a coordinated code rollback if needed; it does not
   delete or rewrite recommendations produced after installation. Recheck file
   hashes first so later edits are not overwritten.

The findings concern the probability/selection layer. They do not establish a
cause for Phase 3's count-regression results and do not invalidate its saved
RMSE measurements. No production-impact count or profitability estimate has been
inferred from synthetic tests.


## Subsequent repair — 28 September 2026

The [market-contract and specialist follow-up](phase4-market-contract-and-specialist-repair-2026-09-28.md)
now migrates supported unit-bet/parlay marginal assessments and removes their legacy
joint-EV calculation. It supersedes item 4 above for that supported scope. Other
specialist/live paths remain outside full canonical acceptance. The follow-up also
records the authenticated Free-plan provider diagnosis; the refresh gate remains intact.
