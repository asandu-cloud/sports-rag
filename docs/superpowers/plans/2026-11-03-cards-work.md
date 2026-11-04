# Cards Work (D54) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cards priced against Pinnacle's fair price (A5.4) ready behind an off switch for Tue 10 Nov, the studies
grading cards the bookmakers' way, and card events in the daily refresh with a backfill from 20 Sep.

**Architecture:** The studies' shared outcome reader (`calibration_data.fixture_outcomes`) counts cards per player
the way published picks already settle (`participation_cards`: yellow 1, red 2, at most 3, players with minutes).
A new engine switch `SPIX_CARDS_REFERENCE=pinnacle` makes the cards selector take Pinnacle's own fair price on each
line as the published chance and pick where another allowed bookmaker beats it by 2%, as stage 4 does for corners
with the median. Because our Pinnacle copy is often stale near kickoff, price confirmation also reads Pinnacle's
current price in OddsPapi and keeps the pick only if it still clears 2% against it. A replay over stored reads and a
golden preview give the owner the evidence before 10 Nov. The refresh adds `/fixtures/events` per finished match into
a new `fixture_match_events` table; a CLI command backfills it.

**Tech Stack:** Python 3.11, SQLAlchemy + Alembic (platform.db), pytest.

**Spec:** `docs/rework-tracker.md`: D54 (the four cards items, owner 3 Nov), D31's A5.4 proposal, D35 (stage 4's
market-value rule), D58 (A5.4 goes for Tue 10 Nov, owner 3 Nov), D61 (the owner's three answers, 3 Nov), the A5.4
row's gate in Open items, and the Findings entries "Card settlement: the bookmakers count a straight red as two
(24 Oct)" and A1.4's Pinnacle freshness (median 130 min stale in the last 3 h).

## What the data says (3 Nov, checked before writing this plan)

- **No cards pick has been published since 17 May.** In the 1,170 reads since 25 Oct, cards results are: no profiled
  referee 484, no prices 472, conviction too thin 103, value too low 83, other guardrails 28. Shots on target: no
  prices in any read. A5.4 therefore *adds* cards picks; "no cards picks" is today's state and ships nothing.
- **Published picks already settle cards the bookmakers' way.** `settlement_policy` stamps
  `yellow_1_red_2_max_3_per_player` on every new publication and `participation_cards` counts it from player
  statistics. Only the studies (`calibration_data.fixture_outcomes`, used by the calibration, disagreement, M1 gate,
  stage 1 and stage 4 replays) count team-stat yellows + reds, a straight red as one. No stored row needs re-grading:
  the studies recompute from the database on each run.
- **Player statistics agree with the events rule** on 18,888 of 19,357 matches with card events (97.6%; 94.3% of
  the 3,427 with a red; where they differ, player statistics are mostly one lower). So the events are not needed to
  settle or to grade; they verify the player statistics and serve later modelling.
- **The backfill is small.** 536 matches finished between 20 Sep and 3 Nov: about 540 requests once (one quiet day
  holds it; normal use is 2,600–4,800 a day), then about one per finished match a day.
- **Pinnacle overstated card overs too** (9–22 Oct, 133 picks: 39.4% stated, 31.8% won) and the softer bookmakers
  lean further to overs, so Pinnacle-reference picks are expected to be mostly unders at the softer bookmakers.

## Global Constraints

- Stage rule (D26, D29): nothing changes published picks until the owner sets `SPIX_CARDS_REFERENCE=pinnacle`
  after the Mon 9 Nov scorecard. Default `off`; `golden_snapshots check` (switches pinned to defaults, the new one
  included) unchanged.
- Scope of 10 Nov: cards only. Corners are already market-priced (stage 4, D35). The bets-to-skip list, labels,
  bookmakers by country and how Spix's own view is shown stay at stage 6 (D33).
- A5.4 cards rule (D31, D35, D54 d): the published chance of a cards line is Pinnacle's own two-sided fair price on
  that line (the live `SPIX_DEVIG_METHOD`); a line Pinnacle does not price on both sides is never a pick, with no
  fallback to the median or to Spix's number; eligible when the value at the quoted price is at least
  `line_selection.MARKET_VALUE_MIN_EV` (2%) and the bookmaker is `bookmaker_sets.quotable`; best line by value.
- Spix's own view (D61, 1): as stage 4. Spix's own probability is kept in `_spix_prob` and the anchor record, not
  used to pick, and nothing new is shown.
- Referees (D61, 2): the guardrail `cards_require_profiled_referee` does not apply to a cards line priced from
  Pinnacle; the profile guardrails still do. The referee stays in Spix's own number (today's referee modifier; D1,
  the fitted referee effect, at A6).
- Recommendation layer (D61, 3, option b): the `recommendations.v1` Value list leaves out picks from markets outside
  the layer (cards, shots on target) until the owner adds them.
- Price confirmation of a Pinnacle-priced pick also needs Pinnacle's current price: Pinnacle seen within `MAX_AGE`
  (60 min), the line listed and active on both sides, and value at the published odds against its current fair
  price at least the selection's threshold. Anything else is `not_confirmed` (fail closed). The published chance
  becomes Pinnacle's current fair price.
- Card count for grading = `min(yellow + 2 × red, 3)` per player with recorded minutes > 0, summed per team, FT only,
  the same as `participation_cards`; anything incomplete is ungraded (None), never a team-stats fallback.
- Database rules: every read inside `data_access()`; writes inside `data_access(writer=True)`, outside refresh hours;
  bulk writes first on a copy (`DATABASE_URL=sqlite:///<copy>` on the command line); a new table is fine, applied to
  the shared `platform.db` only after the code is merged to main. Never read `.env`.
- Paid calls: the backfill's API-Football requests run only after the owner says go, with the count and the day's
  remaining requests shown first. The replay makes no billable OddsPapi request (cached fixture lists only).
- Migration number: `0010_fixture_events` (revises `0009_research_profiles`). The recommendation layer plan's task 6
  takes `0011` instead of `0010` (ledger ruling there).
- Log every finding and plan change in `docs/rework-tracker.md` in the same commit as the work.

## Review Focus

1. **Pinnacle missing or one-sided on a line** (often on alternate cards lines): no anchor, so no pick on that line,
   and never a fallback to the median or to Spix's number. Test in Task 2.
2. **Pinnacle's own quote** is never the pick (its value against its own fair price is below zero); a pick at a book
   outside `quotable` is never published. Test in Task 2.
3. **Pinnacle not seen recently, or moved, at confirmation**: the pick is not confirmed, even when the quoting
   bookmaker's price still holds. Test in Task 3.
4. **Incomplete player statistics** (a team with fewer than 11 rows or 7 players with minutes, a null card count):
   the study leaves cards ungraded, as settlement does. Test in Task 1.
5. **Extra time** (`AET`/`PEN`): player statistics include extra time, so cards are ungraded; goals and corners keep
   today's handling. Test in Task 1.
6. **A refresh or backfill that fails half-way** (an API error on one match): no partial event rows for that match,
   the rest continue, and a rerun fetches it again without duplicates. Test in Tasks 6 and 7.

---

### Task 1: The studies grade cards the bookmakers' way (D54 b)

**Files:**
- Modify: `Scripts/data_platform/calibration_data.py` (`fixture_outcomes`)
- Test: `Scripts/tests/data_platform/test_calibration_data.py`

**Interfaces:**
- Produces: `fixture_outcomes(db_path)[api_id]["cards"]` = the bookmakers' total, or None; the dict gains
  `"cards_rule": "yellow_1_red_2_max_3_per_player"` once per outcome.

- [ ] **Step 1: Write the failing tests** (build a small SQLite file with `fixtures`, `fixture_team_stats`,
  `fixture_player_stats` rows, as the file's existing tests do)

```python
def test_a_straight_red_counts_two_and_a_sending_off_after_two_yellows_three():
    # home: 3 yellows, one straight red (yellow 0, red 1); away: one player yellow 2 red 1, one yellow 1
    assert fixture_outcomes(db)["101"]["cards"] == 3 + 2 + 3 + 1

def test_players_without_minutes_do_not_count():          # a carded substitute with minutes 0 -> not counted
def test_incomplete_player_statistics_leave_cards_ungraded():  # 10 player rows for one team -> cards None
def test_extra_time_leaves_cards_ungraded_but_keeps_goals_and_corners():  # status AET
```

- [ ] **Step 2: Run to verify they fail.** `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_calibration_data.py`
  Expected: the four new tests FAIL (cards still yellow + red from team stats).
- [ ] **Step 3: Implement.** One extra read-only statement over `fixture_player_stats` grouped by fixture and team
  (rows, players with minutes > 0, any null card count among them, `SUM(MIN(yellow + 2*red, 3))`); cards is the
  sum of both teams when each team has ≥ 11 rows, ≥ 7 players with minutes, no null count, and status is FT.
- [ ] **Step 4: Run** the file plus `Scripts/tests/test_phase4_card_calibration.py Scripts/tests/data_platform` → PASS.
- [ ] **Step 5: Rerun the cards part of the calibration study** inside the gate
  (`python -m Scripts.ops.calibration_study --out Research/calibration-<date>-cards-rule`); record cards' Platt
  result and Spix-vs-market line in the tracker's A4.2 row.
- [ ] **Step 6: Commit** (code, tests, tracker).

### Task 2: A5.4 behind `SPIX_CARDS_REFERENCE` (D54 d, D61)

**Files:**
- Modify: `Scripts/rag_ingest/core/engine_switches.py` (new switch `"SPIX_CARDS_REFERENCE": ("off", {"off", "pinnacle"})`)
- Modify: `Scripts/rag_ingest/core/line_selection.py` (`select_best_total_recommendation`: new keyword
  `anchor_books: Optional[Iterable[str]] = None`)
- Modify: `Scripts/rag_ingest/core/market_service.py` (cards branch, ~549; `_apply_quality_guardrails` call)
- Modify: `Scripts/rag_ingest/core/prediction_guardrails.py` (`assess_market_quality(..., market_priced: bool = False)`)
- Modify: `Scripts/data_platform/recommendations.py` (`block`: Value keeps only markets in `MARKETS`)
- Modify: `Scripts/tests/conftest.py`, `Scripts/ops/golden_snapshots.py` if it lists the pinned switches
- Test: `Scripts/tests/test_market_anchor.py`, `Scripts/tests/test_prediction_guardrails.py`,
  `Scripts/tests/test_engine_switches.py`, `Scripts/tests/data_platform/test_recommendations.py`

**Interfaces:**
- Produces: `engine_switches.value("SPIX_CARDS_REFERENCE") -> "off" | "pinnacle"`; with `anchor_books`, the anchor
  consensus uses only those bookmakers (matched by `reference_price.norm` of the quote identity's bookmaker) and the
  anchor record gains `"books_used": [...]`; the cards result's `context.market_anchor` gains
  `"reference": "pinnacle"`. Options priced this way keep `_probability_source == "market_consensus"`, so the
  compiler ranks them with the other market-priced picks.

- [ ] **Step 1: Write the failing tests**

```python
def test_cards_take_pinnacles_fair_price_and_pick_where_another_book_beats_it(monkeypatch):
    # Pinnacle 4.5: over 2.10 / under 1.75; Betano over 2.30 -> value at Betano >= 2% against Pinnacle's power fair
    # -> recommended at Betano, _model_prob == Pinnacle's fair over, _probability_source == "market_consensus"
def test_a_line_pinnacle_prices_on_one_side_only_is_never_a_pick(monkeypatch):
def test_pinnacles_own_quote_is_never_the_pick(monkeypatch):
def test_switch_off_leaves_cards_on_spixs_own_numbers():   # identical result to today's for the same inputs
def test_a_pinnacle_priced_cards_line_does_not_need_a_profiled_referee():  # profile guardrails still apply
def test_the_switch_is_pinned_to_off_in_tests_and_listed_in_the_snapshot():
def test_value_leaves_out_picks_from_markets_outside_the_layer():  # a cards pick at position 1 and a goals pick
    # at position 2 -> value == [the goals pick]
```

- [ ] **Step 2: Run to verify they fail.**
  `.venv/bin/python -m pytest -q Scripts/tests/test_market_anchor.py Scripts/tests/test_prediction_guardrails.py Scripts/tests/test_engine_switches.py Scripts/tests/data_platform/test_recommendations.py`
- [ ] **Step 3: Implement.** In the cards branch, when the switch is `pinnacle`: `anchor_points="all"`,
  `anchor_books=("pinnacle",)`, and pass `market_priced=True` to the guardrail only when the recommended line's point
  is in the anchor record's `lines` (priced from Pinnacle). Corners and every other market unchanged.
- [ ] **Step 4: Run** the four files, `Scripts/tests/test_market_service.py`, `Scripts/tests/test_golden_engine.py` →
  PASS; `python -m Scripts.ops.golden_snapshots check` with every switch pinned to its default, the new one included
  → "no changes across 54 fixtures".
- [ ] **Step 5: Commit** (with the tracker lines).

### Task 3: Price confirmation re-checks Pinnacle for a Pinnacle-priced pick

**Files:**
- Modify: `Scripts/data_platform/services/price_confirmation.py`
- Test: `Scripts/tests/data_platform/test_price_confirmation.py`

**Interfaces:**
- Consumes: Task 2's `context.market_anchor.reference == "pinnacle"` on the cards result.
- Produces: for such a pick, `"pinnacle"` is added to the slugs requested (free histories, as today); the record
  gains `reference: {"site": "pinnacle", "seen_at", "fair_probability"}`; new reasons `reference_not_recent`,
  `reference_not_listed` ("Pinnacle has not been seen recently" / "Pinnacle does not list both sides of the line")
  and `reference_moved` ("against Pinnacle's price now, the pick no longer has value"). A confirmed or repriced pick
  takes Pinnacle's current fair price as `model_probability` (binary profile, as the anchored line) with its value
  recomputed at the published odds.

- [ ] **Step 1: Failing tests:** Betano confirmed at 2.30 and Pinnacle now over 2.10 / under 1.75 → confirmed,
  `model_probability` = Pinnacle's current power fair over; Pinnacle now over 1.95 / under 1.88 (value at 2.30
  under 2%) → `not_confirmed`, `reference_moved`; Pinnacle last seen 90 min ago → `reference_not_recent`; Pinnacle
  lists the over only → `reference_not_listed`; a goals pick (no reference) → requests and verdicts as today; a pick
  quoted at Pinnacle is not possible (Task 2) so needs no case.
- [ ] **Step 2: Run to verify failure. Step 3: Implement. Step 4: Run** the file and
  `Scripts/tests/data_platform` → PASS.
- [ ] **Step 5: Commit** (tracker: the re-check, and why: A1.4's Pinnacle freshness).

### Task 4: A5.4 replay on stored reads (evidence for the owner)

**Files:**
- Create: `Scripts/ops/cards_reference_replay.py`
- Test: `Scripts/tests/test_cards_reference_replay.py`

**Interfaces:**
- Consumes: `calibration_data.latest_read_rows`, `fixture_outcomes` (Task 1), `reference_price.audit_books`,
  `bookmaker_sets.quotable`, `line_selection.MARKET_VALUE_MIN_EV`, the odds archive's final pre-kickoff snapshot (as
  `stage4_replay`), `oddspapi.histories` on cached data.
- Produces: `replay(reads, outcomes, closes, *, pinnacle_at_read=None) -> dict` with `picks`, `fixtures`, `stated`
  (mean chance), `happened` (hit share, half results weighted as settlement), `interval95` (whole fixtures
  resampled), `value_at_close` (Pinnacle's closing fair price × odds taken − 1, with its interval), each also split
  by side (over/under); with `pinnacle_at_read`, `kept_against_pinnacle_then` (picks still ≥ 2% against Pinnacle's
  OddsPapi price at the read) and the unknown count. CLI
  `python -m Scripts.ops.cards_reference_replay --since 2026-10-09 --out Research/cards-reference-<date>`.

- [ ] **Step 1: Failing tests:** one fixture where Betano beats Pinnacle's fair over 4.5 by 3% and the match had 6
  cards → one pick, stated = Pinnacle's fair over, happened = 1; its value at the close from a closing Pinnacle pair;
  a fixture with no Pinnacle price → no pick; no closing Pinnacle pair → the pick counts in results, not in
  value at the close; one pick per fixture, the highest value.
- [ ] **Step 2: Run to verify failure. Step 3: Implement** (pre-match reads, the latest before kickoff per fixture;
  the cards key `totals_cards_over_under`; the reads copied behind the gate in short statements, the replay after
  the gate is released). **Step 4: Run** → PASS.
- [ ] **Step 5: Run it on real data** from 9 Oct; record picks, stated vs happened with the interval, value at the
  close, the over/under split and the Pinnacle-then count in the tracker. Compare with the 9–22 Oct median-reference
  result (stated 45.3%, won 32.3%, 133 picks).
- [ ] **Step 6: Commit** (code, tests, tracker).

### Task 5: The go-live package for Tue 10 Nov

- [ ] **Step 1: Golden preview** on the live stack: with `.env` loaded (no pins),
  `SPIX_CARDS_REFERENCE=pinnacle python -m Scripts.ops.golden_snapshots preview --baseline SPIX_CARDS_REFERENCE=off`.
  Expected: only cards picks change (added); every other market unchanged. If the 17–18 Oct recording holds no
  Pinnacle cards prices, the preview shows no change and Task 4's replay is the evidence; say so in the tracker.
- [ ] **Step 2: Tracker:** D-entry (next free number) with the preview and Task 4's replay; the A5.4 row and Open
  items say A5.4 cards is ready, switch off; the go-live step: owner adds `SPIX_CARDS_REFERENCE=pinnacle` to `.env`
  on Tue 10 Nov after the scorecard, then the website restart; the gate: calibration error ≤ 3 pts and value at the
  close at least neutral over 4+ live matchweeks (first verdict early December). Commit, then tell the owner and the
  orchestrator it is ready to merge (by Sat 7 Nov), with the new switch to pin in the merge checks.

### Task 6: Card events in the daily refresh (D54 a)

**Files:**
- Create: `Scripts/data_platform/models/fixture_events.py` (register in `models/__init__.py`)
- Create: `Scripts/data_platform/alembic/versions/0010_fixture_events.py`
- Modify: `Scripts/data_platform/sync/apifootball.py` (`fixture_events(fixture_id) -> List[dict]`, `/fixtures/events`)
- Modify: `Scripts/data_platform/sync/upserts.py` (`upsert_fixture_events(session, *, fixture, events_response) -> int`)
- Modify: `Scripts/data_platform/sync/pipeline.py` (`sync_fixture_details` and `_buffer_fixture_details`: events
  fetched with the player statistics; `SyncStats.event_rows`)
- Test: `Scripts/tests/data_platform/test_sync_upserts.py`, a pipeline test with a fake client

**Interfaces:**
- Produces: table `fixture_match_events` (not `fixture_events`: the history download's staging DB uses that name): `id`, `fixture_id` (FK `fixtures.id`, cascade), `event_index`, `team_api_id`,
  `player_api_id`, `assist_api_id`, `elapsed`, `extra`, `event_type`, `detail`, `comments`, `raw_json`, `fetched_at`;
  unique `(fixture_id, event_index)`. A new canonical table: the history download's `player_history_events` is a
  batch-import record and stays as it is.

- [ ] **Step 1: Failing tests:** an events payload of 5 rows upserts 5 rows; upserting again with 4 rows leaves 4
  (replace per fixture, no duplicates); an events request that raises records an error with stage `"events"`, keeps
  the fixture's statistics and players, and leaves no event rows for it.
- [ ] **Step 2: Run to verify failure. Step 3: Implement. Step 4: Run** `Scripts/tests/data_platform` → PASS.
- [ ] **Step 5: Commit** (tracker: one more request per finished match; the table is applied to `platform.db` with
  `init-db` only after the merge).

### Task 7: The events backfill from 20 Sep

**Files:**
- Modify: `Scripts/data_platform/cli.py` (`backfill-events --since YYYY-MM-DD [--limit N] [--dry-run]`)
- Test: `Scripts/tests/data_platform/test_events_backfill.py`

**Interfaces:**
- Consumes: Task 6's client method and upsert.
- Produces: `backfill_events(session_factory, client, *, since, limit=None, dry_run=False) -> {"candidates", "fetched", "rows", "errors"}`:
  finished fixtures (FT/AET/PEN) since `since` with no `fixture_match_events` rows, oldest first; `--dry-run` prints the
  count of requests it would make and makes none. Writes inside `data_access(writer=True)`, one fixture per commit.

- [ ] **Step 1: Failing tests:** dry run makes no requests and counts candidates; a run with `limit=2` fetches the two
  oldest; a fixture that already has events is skipped; a failed request is counted and the run continues.
- [ ] **Step 2: Run to verify failure. Step 3: Implement. Step 4: Run** → PASS. **Step 5: Commit.**
- [ ] **Step 6: Rehearsal on a copy** (after the owner's go): copy `Index/platform.db` beside the scratchpad, apply
  `init-db` to the copy, run `DATABASE_URL=sqlite:///<copy> ... backfill-events --since 2026-09-20 --limit 20`; check
  the rows against the player statistics' reds for those matches; delete the copy.
- [ ] **Step 7: After the merge** (on main, outside refresh hours, owner's go with the dry-run count and the day's
  remaining API-Football requests): `init-db`, then the full backfill, in one day if the remaining count allows.
  Record the counts and the events-vs-player-statistics agreement since 20 Sep in the tracker.
