# Prediction Pipeline Unification — Phase 2 Contract

**Status:** implemented; no production caller has migrated yet.

## Contract location

`Scripts/rag_ingest/core/market_result.py` defines the versioned,
presentation-free `market_result.v1` contract.

```text
FixtureRef + MarketRef + Projection + Decision + Provenance
    + EvidenceItem[] + context
    = MarketResult
```

The contract is intentionally a standard-library dataclass model. It can be
serialized for the API, Discord jobs, parlay sessions, test fixtures, and
prediction tracking without making any of those consumers own the model.

## Decision states

| State | Meaning | Required fields |
|---|---|---|
| `recommended` | A priced line clears the selector thresholds. | Quote with odds and model probability |
| `no_bet` | Priced options exist, but none meets the recommendation threshold. | Explicit reason; candidate metrics may remain present |
| `unavailable` | The model cannot make a priced decision because input/profile/market data is unavailable. | Explicit reason |

This distinction is intentional. A no-bet is a valuable model decision; an
unavailable result is a data-quality or coverage state.

## Provenance requirements

Every result must record:

- pipeline version
- input snapshot ID
- generation timestamp
- optional model version, as-of timestamp, and named data sources

This is the minimum required to compare legacy and canonical results, replay a
historical recommendation, and diagnose a surface discrepancy.

## Legacy adapter

`decision_from_selector()` translates the existing dictionaries returned by
`core.line_selection` into a `Decision`. It is pure and currently has no
production callers. That is deliberate: Phase 3 will introduce the canonical
market service and use this adapter while the old selector dictionaries still
exist.

## Explicit non-goals

- No projection formula changed.
- No selector threshold changed.
- No Discord, website, live, CLI, or parlay caller has changed.
- No rendered text is part of the contract.

## Verification

`Scripts/tests/test_market_result_contract.py` verifies JSON round-tripping,
contract validation, all three decision states, version rejection, and adapter
behaviour against the current totals selector.

## Phase 3 handoff

Phase 3 implemented the canonical market service and migrated the legacy
projection/selection entry points to thin compatibility wrappers. See
[Phase 3 Canonical Service](PIPELINE_UNIFICATION_PHASE3.md). Discord and
website consumers intentionally remain unchanged until Phase 4.
