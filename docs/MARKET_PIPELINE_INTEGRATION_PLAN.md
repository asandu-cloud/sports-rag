# Market Pipeline Integration Plan

## Purpose
This document captures the remaining integration work needed so the core market
projections in `Scripts/rag_ingest/core/projections.py` are consistently used
through the full product pipeline:

- selector
- renderer
- Discord command
- scheduled auto-push
- monolith compatibility path
- final embed output

The goal is not new model research. The goal is to remove architectural drift
and make sure the same market logic is used everywhere.

## Current State

### What is already in good shape
- Core projections live in `Scripts/rag_ingest/core/projections.py`.
- Main selectors live in `Scripts/rag_ingest/core/line_selection.py`.
- Main market renderers use the core selectors:
  - `Scripts/rag_ingest/rendering/totals.py`
  - `Scripts/rag_ingest/rendering/btts.py`
  - `Scripts/rag_ingest/rendering/spreads.py`
  - `Scripts/rag_ingest/rendering/moneyline.py`
  - `Scripts/rag_ingest/rendering/correct_score.py`
  - `Scripts/rag_ingest/rendering/team_totals.py`
- `/market` now uses deterministic rendering through
  `Scripts/rag_ingest/rendering/market_dispatch.py`.
- Main market auto-push collection in
  `Scripts/discord_bot/cogs/auto_push.py` now uses deterministic market
  rendering for the daily/high-risk market workflows.
- `/analyze` in `Scripts/discord_bot/cogs/slash_cmds.py` now pulls structured
  projections and selectors directly for most markets.

### What is still not fully integrated
- `/teamlines` is still prompt-driven.
- Scheduled `#team-lines` posting is still prompt-driven.
- `rag_cli_v2.py` still contains selector logic that should delegate fully to
  `core/line_selection.py` for some markets.
- The Discord embed layer still relies on parsing formatted text in
  `Scripts/discord_bot/embeds.py`.
- Some non-Discord surfaces consume projections directly instead of going
  through the same structured market-result contract.

## Main Problems To Solve

### 1. Team lines are still on the old prompt path
Affected files:
- `Scripts/discord_bot/cogs/slash_cmds.py`
- `Scripts/discord_bot/cogs/auto_push.py`

Current issue:
- `/teamlines` and scheduled team-line posts still build prompts, run
  `rag.answer_once()`, and parse the returned prose.

Target state:
- Team lines should use `render_market_answer_sync(..., market="teamlines")`
  via `Scripts/rag_ingest/rendering/market_dispatch.py`, the same way `/market`
  now does for the other markets.

### 2. Monolith selector drift still exists
Affected file:
- `Scripts/rag_ingest/rag_cli_v2.py`

Current issue:
- Some wrappers now delegate cleanly to core, but not every selector does.
- Moneyline and correct score still keep monolith implementations instead of
  fully delegating to `core/line_selection.py`.

Target state:
- `rag_cli_v2.py` should act as a compatibility shell, not a second logic
  engine.

### 3. Embed generation still parses text instead of structured results
Affected file:
- `Scripts/discord_bot/embeds.py`

Current issue:
- Even deterministic market output is still turned back into compact lines via
  text parsing.

Target state:
- Deterministic markets should eventually render from a structured market result
  object instead of regex over formatted text.

### 4. Some downstream consumers bypass the selector layer
Affected areas:
- `Scripts/web_app/api.py`
- `Scripts/discord_bot/cogs/whale_chat.py`
- some evidence/rendering paths in `Scripts/rag_ingest/core/parlay_evidence.py`
  and `Scripts/rag_ingest/rendering/parlay_render.py`

Current issue:
- These paths often consume projection outputs directly.
- That is acceptable for analytics-only surfaces, but not ideal when the
  product is trying to present actual recommended lines.

Target state:
- Analytics surfaces can keep raw projections.
- Recommendation surfaces should use the same selector contract everywhere.

## Recommended Execution Order

### Phase 1. Finish team-line integration
Goal:
- Remove the last prompt-driven market workflow from Discord.

Changes:
- Add team-lines support to the deterministic command path in
  `Scripts/discord_bot/cogs/slash_cmds.py`.
- Replace `_run_rag(...)` usage in `/teamlines` with direct market dispatch.
- Replace `_post_team_lines(...)` in `Scripts/discord_bot/cogs/auto_push.py`
  with deterministic market rendering.
- Keep the existing embed appearance if possible, but feed it deterministic text
  at minimum.

Verification:
- `/teamlines` single-fixture flow still works.
- `/teamlines` league-wide flow still works.
- scheduled `#team-lines` output still renders one embed per league.

### Phase 2. Collapse remaining selector duplication in `rag_cli_v2.py`
Goal:
- Make `rag_cli_v2.py` a compatibility wrapper, not a competing engine.

Changes:
- Delegate moneyline selection to `core/line_selection.py`.
- Delegate correct-score ranking to `core/line_selection.py`.
- Review any remaining in-file selection logic for totals, BTTS, spreads,
  moneyline, and correct score.
- Keep old function signatures intact for compatibility.

Verification:
- Expand `Scripts/tests/test_projection_parity.py`.
- Add selector parity coverage if needed.

### Phase 3. Introduce a structured market result contract
Goal:
- Stop forcing deterministic market output through text parsing before embed
  rendering.

Suggested new module:
- `Scripts/rag_ingest/rendering/market_results.py`

Suggested object shape:
- `fixture`
- `market`
- `projection_summary`
- `bet_recommendation`
- `best_value`
- `no_bet_reason`
- `supporting_lines`
- `notes`

Changes:
- Market renderers should be able to produce:
  - text output for CLI compatibility
  - structured output for bot/embed compatibility

Verification:
- one market first, likely moneyline
- then totals
- then BTTS and spreads

### Phase 4. Move deterministic Discord embeds off regex parsing
Goal:
- Let Discord consume structured market objects directly.

Affected file:
- `Scripts/discord_bot/embeds.py`

Changes:
- Add structured embed builders for:
  - totals
  - BTTS
  - moneyline
  - spreads
  - correct score
  - team lines
- Keep `rag_output_to_embeds(...)` for legacy prompt-based workflows that remain.
- Do not break player props or other text-heavy paths while doing this.

Verification:
- `/market`
- `/teamlines`
- auto-push daily picks
- auto-push high-risk picks

### Phase 5. Align downstream recommendation surfaces
Goal:
- Ensure recommendation-facing endpoints are not bypassing selectors.

Review and decide per surface:
- `Scripts/web_app/api.py`
  - if endpoint is recommendation-facing, route through selector
  - if endpoint is analytics-facing, raw projection is fine
- `Scripts/discord_bot/cogs/whale_chat.py`
  - if it stays analytics-first, raw projection is acceptable
  - if it recommends lines, route through selector
- parlay evidence/rendering
  - should reflect selector/projection logic consistently

## File-by-File Worklist

### High priority
- `Scripts/discord_bot/cogs/slash_cmds.py`
- `Scripts/discord_bot/cogs/auto_push.py`
- `Scripts/rag_ingest/rag_cli_v2.py`
- `Scripts/discord_bot/embeds.py`

### Medium priority
- `Scripts/web_app/api.py`
- `Scripts/discord_bot/cogs/whale_chat.py`
- `Scripts/rag_ingest/core/parlay_evidence.py`
- `Scripts/rag_ingest/rendering/parlay_render.py`

### Existing integration entrypoint
- `Scripts/rag_ingest/rendering/market_dispatch.py`

This should stay the central deterministic market dispatcher unless a better
shared service replaces it.

## Success Criteria
- All market recommendation surfaces use the same projection and selector logic.
- Team lines are no longer prompt-driven.
- `rag_cli_v2.py` stops drifting from `core/*`.
- Discord embeds for deterministic markets no longer depend on text parsing.
- Recommendation outputs can explicitly return:
  - recommended line
  - best value line
  - no bet
  - reason

## Verification Checklist
- `python -m py_compile` on changed files
- targeted unit tests for:
  - selector parity
  - team-lines market dispatch
  - structured embed rendering
- manual smoke checks for:
  - `/market`
  - `/teamlines`
  - auto-push daily picks
  - auto-push high-risk
  - auto-push team lines

## Notes
- This plan is deliberately integration-focused, not model-research-focused.
- The scoring/projection layer has already been improved materially.
- The remaining work is mostly about making sure the rest of the product
  actually consumes that improved logic consistently.
