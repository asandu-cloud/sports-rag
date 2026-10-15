# Phase 4 immediate correctness work — 28 September 2026

Status: implemented, tested locally and installed in the publishing checkout under
owner authorization. Real-fixture publication acceptance remains open. This is a
correctness prerequisite, not a fitted calibration or validated accuracy improvement.

## Changes installed

- Added `canonical-market-contract.v1`: an explicit registry for the seven existing
  canonical markets. First/second-half, extra-time, incompatible settlement definitions,
  total shots versus shots on target, non-goal handicaps and unsupported specialty keys
  cannot enter those forecasts through substring matching. Invalid BTTS sides and
  non-finite prices/lines are excluded. Known normalized full-match keys retain the
  established regulation-time interpretation when period metadata is absent.
- API-Football normalization retains source bet ID/name and supplied period/rule
  metadata. The registry rejects explicit conflicting labels. This does not verify
  every bookmaker's rules or replace the provider's existing bet-ID mapping.
- BTTS and winner vig removal now require the same fixture, market key, period and
  settlement definition as well as bookmaker. Winner results preserve their source
  market key. Totals and handicaps retain the previous exact-pair checks.
- Added canonical assessment of a specific existing quote. It uses the same selectors,
  statistical projections, fixture cutoff, supplied model contexts, data-quality gates
  and release policy as the main market service. Missing or ambiguous snapshots,
  unavailable prices and failed gates do not become parlay recommendations.
- The CLI and structured parlay/unit-bet path now use that assessment for supported
  markets, confidence, marginal probability, EV, ranking support and evidence. Locked
  selections cannot bypass the gates. A request-scoped cache avoids repeated assessment
  of a leg; profile caches share the request boundary and invalidate normally afterward.
- New leg results retain probability basis/version, all five Asian settlement
  probabilities, EV, input fingerprint and prediction-system version. Unit-slip JSON
  payloads preserve these fields. Historical records acquire no invented versions or
  settlement distributions. This is not a migration of legacy outcome tracking into
  canonical publication tracking, nor a new persistent archive of full input slates.
- Removed joint-EV calculations that multiplied leg probabilities and filled missing
  probabilities with 50%. Ranking uses eligible marginal EV support; it does not claim
  portfolio EV. Heuristic odds-target/diversity constraints remain. This can change
  specialist selection membership and can produce fewer or no qualifying legs.
- Parlay presentation identifies the individual-price product as an unverified combined
  quote. The legacy numeric `combined_odds` field remains for solver/storage compatibility;
  new structured records label it with
  `combined_odds_basis=individual_price_product_unverified`. It is not a bookmaker quote.
  Automatic unit slips reject same-game and cross-bookmaker combinations without an
  executable combined quote. Compatible selections can still appear separately in a
  Match Read or parlay analysis; no arbitrary one-selection-per-match restriction was
  added to Match Reads. Distinct-fixture, same-book accumulator products remain the
  existing conventional pricing assumption, not newly verified provider combinations.
- Discord/CLI labels distinguish an Asian price-comparison probability from a full-win
  probability and from heuristic confidence.

Unsupported team/player/specialty definitions no longer receive inferred canonical
probabilities through the automatic parlay candidate builder. Their separate specialist
renderers are not certified by this change. The legacy website user-assembled parlay
price-only scoring endpoint, trial-slip tracking and live specialists are not a complete
canonical migration. Do not describe this batch as migrating every numerical consumer.

## Validation and installation

- **616 tests plus four regression subtests passed** across 31 modules in an isolated
  source copy. **358 focused tests passed against the installed checkout.** Tests blocked
  network connections and dotenv loading and used disposable test databases.
- The new 62-case suite covers incompatible periods/definitions, source metadata,
  incomplete or mixed quote sets, canonical parity across all seven markets, integer
  pushes and quarter handicaps, price-independent Asian probability, exact-price
  matching, no-bet/shadow gates, request cache lifetime, unsupported definitions,
  unit-slip combination guards and old/new record serialization.
- The synthetic `pipeline_v1/priced_standard` fixture has one deliberate expectation
  update: moneyline `quote.market_key` is now `h2h` rather than null. Its numerical
  expectations are unchanged. Preserved research/baseline archives were not modified.
- Forward and rollback patches were applied in a disposable directory; all 19 file
  hashes matched in both directions. The rollback also passes against the installed
  checkout. `git diff --check` passes. Three EOF-only formatting corrections after
  installation have a separate amendment record; the original installation patch and
  hashes are retained.
- Publishing and measurement workers were idle before installation. Their existing jobs
  were temporarily unloaded, the reviewed patch applied, and the unchanged plists loaded
  again. Both subsequent cycles exited zero and deferred behind the incomplete-refresh
  gate. That is operational verification of safe deferral, not real-match acceptance.
- Source hashes outside this batch, active model artifacts, worker plists and the
  incomplete marker were verified unchanged. No production database migration, baseline
  replacement, retraining, historical outcome inspection, bulk collection or test
  recommendation publication was performed. Normal scheduled jobs may update their logs,
  operational alert state and run metadata.

Evidence package:
[`Research/market-contract-repair-2026-09-28`](../Research/market-contract-repair-2026-09-28/).
It includes source fingerprints, before-files, dependency snapshot, forward/rollback
patches, test lists/results, installation and formatting records, worker verification,
post-install integrity checks and the redacted provider diagnostic.

## Provider-access blocker — verified separately

One successful authenticated `GET /status` request returned HTTP 200, no errors,
**Free** subscription, active status and a **100-request/day** allowance for the
configured key. Account identity and key values were not logged. The preliminary
sandboxed attempt failed to connect; it did not initiate fixture collection.
No key override was present in the four launchd job plists or launchd global environment.

Existing refresh logs report that this plan cannot access the requested season and
suggest seasons 2022–2024. This verifies the current entitlement mismatch; it does not
establish why the account/key has that entitlement. Restore current-season access for
the configured key (or configure the intended entitled key), then run the ordinary
bounded full refresh and verify it clears its own incomplete marker. No subscription,
secret, schedule, lock or incomplete marker was changed, and no refresh was launched.

## Next work remains

1. Restore provider access, complete an ordinary refresh, and accept genuine fixture
   generation, publication, closing capture and settlement when fixtures are available.
2. Review the eight-match prior cutoff before freezing the Phase 4 baseline; it is
   unchanged by this batch.
3. Qualify card targets, missingness and selection bias against the approved settlement
   semantics. Generic card keys without declared rules remain legacy unqualified scope;
   this registry does not certify their equivalence to weighted participation settlement.
4. Fit and evaluate dispersion, goal dependence, calibration and uncertainty using the
   agreed development/calibration protocol. Fixed rho, variance floors, statistical
   means and heuristic confidence remain. Richer lineup effects remain later work.
