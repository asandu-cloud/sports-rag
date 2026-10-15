# Prediction Pipeline Unification — Phase 3 Canonical Service

**Status:** implemented. Delivery consumers remain unchanged until Phase 4.

## What now has one owner

`Scripts/rag_ingest/core/market_service.py` is the canonical pre-match service
for these decision flows:

- match totals: goals, corners, cards, and shots on target
- BTTS
- moneyline
- score spreads/handicaps

For each normalized event it calls only:

```text
core.projections -> core.odds_extraction -> core.line_selection -> MarketResult
```

The service returns `recommended`, `no_bet`, or `unavailable` as an explicit
decision, with fixture identity, projection components, exact price, evidence,
context, and provenance. It makes no network calls and does no rendering.

## Legacy compatibility changes

The active `rag_cli_v2.py` definitions for these primitive projections now
delegate to `core.projections`:

- `projected_corners`
- `projected_cards`
- `projected_sot`
- `projected_goals`

Its active single-line selectors also delegate to `core.line_selection`:

- totals, BTTS, correct score, moneyline, and spread selection

The old function bodies are intentionally retained above the compatibility
wrappers for one release cycle. They are no longer active runtime definitions;
they exist only for rollback and historical comparison, then become Phase 6
removal candidates.

## Important behavioural consequence

Legacy text paths that call these wrappers now use the canonical core's
selection eligibility. That is intentional: a legacy renderer may now receive
the same `no bet` decision as the core instead of applying its older local
selector behaviour. No Discord, website, live, or parlay consumer was changed
to consume `MarketResult` directly in this phase.

## Still deferred

- team-total and correct-score *structured* service results
- legacy event/odds helper deletion
- candidate/parlay scoring migration to canonical market results
- Discord embed and auto-push migration away from text parsing
- website API migration away from its independently assembled response models
- live poller migration to the canonical service

Those are consumer/data-flow changes and belong to Phase 4 rather than this
compatibility layer.

## Verification

- `test_market_service.py` covers deterministic totals, BTTS, unavailable
  profile/price cases, ordering, and unsupported-market rejection.
- `test_market_result_contract.py` covers the versioned contract and selector
  adapter.
- `test_projection_parity.py` now verifies all legacy primitive and
  single-line selector wrappers delegate to core.
- Full test suite after this change: **167 passed**.
