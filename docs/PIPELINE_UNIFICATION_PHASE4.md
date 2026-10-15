# Pipeline Unification — Phase 4

Phase 4 moves the delivery surfaces onto the `MarketResult` contract created
in Phase 2 and evaluated by the canonical service in Phase 3.

## Implemented delivery paths

- `rendering.market_dispatch.evaluate_market_results_sync()` now fetches and
  enriches provider events, then returns `MarketResult` objects for goals,
  corners, cards, shots on target, BTTS, moneyline, and spreads.
- `render_market_answer_sync()` uses a one-way result-to-text adapter for
  remaining text-only callers; it no longer runs a separate market model.
- Discord `/market` produces bet-card embeds directly from `MarketResult`
  objects for those markets.
- Scheduled Discord market-channel posts and daily top-pick ranking use
  structured decisions, quotes, probabilities, edges, and confidence values.
  In particular, daily-pick ranking no longer scrapes values out of rendered
  strings.
- The website fixture and best-bets endpoints now evaluate the canonical
  service.  Existing response fields remain available for frontend
  compatibility, with additive `status`, `reason`, and `market_results`
  fields so clients can distinguish a recommendation, no-bet, and missing
  data.

## Deliberately retained paths

`correctscore`, `intervals`, and `teamlines` still use their specialist
renderers.  They require their own structured market contracts; representing
them as generic totals would discard important data.  Player props and free
form Whale chat also remain outside this phase.

Parlay creation already has a structured service, but its candidate-generation
path still needs to receive `MarketResult` objects directly.  Live polling has
its own timing/state model and will be migrated after its pre-match snapshot
boundary is defined.

## Behavioural change

Only priced selections with `decision.status == "recommended"` can appear in
the website best-bets list or scheduled Discord top picks.  A model lean without
a qualifying market price is shown as `no_bet` or `unavailable`, rather than
being promoted as a recommendation.

## Verification

`Scripts/tests/test_market_delivery.py` covers dispatch, the text compatibility
adapter, public market support boundaries, and the web-response adapter.
Focused Phase 2–4 tests pass before this document was added.
