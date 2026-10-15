# Prediction Pipeline Unification — Phase 1 Baseline

**Status:** complete audit; no prediction behaviour changed.

## Objective

Establish the current production call paths and the deterministic inputs needed
to prove that a future canonical pipeline produces the same recommendation on
every surface. This document is the baseline for the structured-result work;
it is not a new model specification.

## Canonical destination

The canonical prediction path will be:

```text
fixture + odds snapshot + profile snapshot + context
    -> core projections + core line selection
    -> structured market result
    -> renderer / Discord embed / API response / parlay consumer
```

The arrow is intentionally one-way. Consumer code must never recover model
data by parsing rendered text.

## Current call-path inventory

| Surface | Current entry point | Current behaviour | Target migration | Risk |
|---|---|---|---|---|
| General Discord commands | `discord_bot/cogs/slash_cmds.py:_run_rag()` -> `rag_cli_v2.answer_once()` | Intent parsing and legacy text output | Retain intent parsing; return canonical structured results | High |
| `/market` Discord command | `render_market_answer_sync()` -> `rendering/market_dispatch.py` | Modular renderers use core projections/selectors but return text | Dispatcher returns structured results; renderer becomes final presentation step | Medium |
| `/teamlines` Discord command | Modular dispatcher, then string block extraction | Correct model route, but result is still parsed as text for fixture filtering | Filter structured fixture results before rendering | High |
| Scheduled Discord posts | `auto_push.py` | Mixed: dispatcher text, old RAG prompts, regex/text parsers, and structured parlay service | Consume canonical results directly | High |
| Parlay commands / promotional parlays | `core/parlay_service.py` plus `rag_cli_v2` helpers | Structured session/result objects, but candidate construction, scoring, and evidence remain legacy-owned | Make canonical market results the parlay candidate source | High |
| Whale chat | `whale_chat.py` -> `rag_cli_v2` projection helpers | Mostly information/stat output using compatibility wrappers | Call canonical projections/results while preserving policy restrictions | Medium |
| Website API | `web_app/api.py` | Directly calls core projections/selectors, then independently creates Pydantic response models | Replace hand-built market assembly with canonical result serialization | Medium |
| Live poller | `live/live_poller.py:_compute_priors()` | Uses `rag_cli_v2` prematch wrappers for priors | Use canonical prematch projection adapter; keep live state separate | Medium |
| Live Discord bot | `live_discord_bot/` -> live runtime/API | Presents live snapshots and alerts | Consume the compatible live-result schema after prematch migration | Medium |

## What is already canonical

`rag_cli_v2.py` currently delegates these compatibility wrappers to
`core.projections`:

- total goals, corners, cards, and shots on target
- team goals, corners, and cards
- BTTS, correct score, moneyline, and goal difference

The existing parity tests cover these wrappers. This is a good migration
boundary: callers can move without changing projection formulae first.

The modular market renderers also use `core.projections` and
`core.line_selection` under normal imports.

## Remaining duplicated or unsafe ownership

`rag_cli_v2.py` still owns substantial production behaviour that cannot be
treated as canonical:

- direct per-team `projected_corners`, `projected_cards`, `projected_sot`, and
  `projected_goals` implementations
- event/odds fetching and market extraction
- candidate construction, single-leg quality, parlay combination scoring, and
  evidence construction
- line-selection and market-rendering functions
- natural-language routing mixed with business logic

The main delivery risk is text round-tripping:

- Discord embeds receive renderer text.
- Several auto-push jobs parse fixture blocks and picks from that text to rank,
  regroup, or post them.
- `/teamlines` extracts one fixture from a rendered string.

This makes presentation formatting a hidden API contract. Phase 2 must replace
it with a structured contract before any formatting is redesigned.

## Behavioural parity targets

For a fixed snapshot, all migrated pre-match consumers must agree on:

1. fixture and market identity
2. model projection and any season/recent components exposed by the model
3. selected side and line, or the same `no_bet` outcome
4. offered odds, implied probability, model probability, and value edge
5. confidence classification and evidence identifiers
6. source/as-of timestamps and pipeline version

Text, embed layout, field order, and non-decision wording are deliberately not
parity targets.

## Deterministic snapshot set

Create sanitized, provider-independent recordings under:

```text
Scripts/tests/fixtures/pipeline_v1/
  manifest.json
  <scenario-id>/input.json
  <scenario-id>/expected-legacy.json
```

No recording may contain credentials, Discord IDs, user data, or raw provider
headers. `input.json` should contain only the fields the model observes:

- fixture: stable test ID, league, teams, kickoff, home/away, status
- profile and recent-stat snapshots for both teams
- league, referee, lineup, knockout, and rivalry context used at evaluation
- normalized market options/odds with source and capture timestamp
- expected availability state for each requested market

`expected-legacy.json` is a read-only baseline produced before migration. It
stores decisions and numeric values, not rendered text. It must state the
legacy source function and capture date so intentional model corrections can
be reviewed rather than silently accepted.

### Required initial scenarios

| Scenario | Markets / condition | Purpose |
|---|---|---|
| `priced_standard` | Goals totals, BTTS, moneyline, spread | Normal priced selection across the common markets |
| `team_totals` | Home/away corners and cards | Protect team-total probability and match-total separation |
| `referee_context` | Cards with a known referee modifier | Preserve context and uncertainty handling |
| `contextual_fixture` | Rivalry and/or knockout fixture | Preserve date-dependent contextual modifiers |
| `no_bet` | Fair or unfavourable prices | Ensure a pass is a first-class result, not an error |
| `missing_odds` | Projection exists, no supported price | Keep model-only/unavailable status explicit |
| `insufficient_profile` | One or both team profiles unavailable | Verify safe degradation and user messaging |
| `parlay_candidate_set` | Multiple fixtures and market options | Verify candidate eligibility, conflict rules, and scoring |
| `live_prematch_prior` | Fixture before kickoff | Align live-engine prematch priors with the canonical result |

The first recordings should be created from archived historical fixtures—not
live API requests—so they remain stable after odds or profile refreshes.

## Existing tests to preserve and extend

- `test_projection_parity.py`: legacy wrapper to core projection parity
- `test_market_selection.py`: recommended bet and no-bet selector behaviour
- `test_teamline_projections.py`: team-corner/card model and team-total odds
- `test_parlay_service.py`: session, selection, and follow-up behaviour
- `test_live_poller.py` and `test_live_odds.py`: live prior and alert handling
- `test_moneyline_workflow.py` and `test_spread_workflow.py`: output parsing
  tests to retire once consumers use structured results

## Phase 1 completion criteria

- [x] Every user-facing prediction surface has an identified current path.
- [x] Canonical core ownership and legacy ownership are separated above.
- [x] Cross-surface parity targets are defined.
- [x] A sanitized deterministic-recording format and initial scenario matrix
  are defined.
- [ ] Capture the first archived fixture recordings once the historical source
  to use is selected.

## Next implementation decision

Phase 2 implemented the versioned structured market-result contract in
[`market_result.py`](../Scripts/rag_ingest/core/market_result.py). See
[Phase 2 Contract](PIPELINE_UNIFICATION_PHASE2.md). The contract must support
all scenarios above before any caller is migrated.
