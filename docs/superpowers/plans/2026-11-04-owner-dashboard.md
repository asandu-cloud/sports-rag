# Owner Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, owner-only dashboard on port 8300 with a Report page (said against hit, week by week, per
market) and a History page (played matches, one line per pick with what happened), for Most likely and Value.

**Architecture:** A package `Scripts/ops/owner_dashboard/` with its own SQLite file `Index/owner_dashboard.db`.
`sync.py` reads `Index/platform.db` (read-only, inside `data_access()`, short statements, batches) and writes
only the dashboard's file; `report.py` and `pages.py` render HTML from that file alone; `__main__.py` serves it
with the standard library on `127.0.0.1:8300`.

**Tech Stack:** Python 3 standard library (`sqlite3`, `http.server`), the platform's `recommendations`,
`scorecard` and `calibration_data` helpers (imported, never changed), inline SVG, one small inline script.

**Spec:** `docs/superpowers/specs/2026-11-04-owner-dashboard-design.md`

## Global Constraints

- No change to the engine, to `Scripts/data_platform`, `Scripts/web_app` or any running job; nothing is written
  to `Index/platform.db`; no migration.
- Every `platform.db` read inside `data_access()`, `mode=ro`, short statements; read JSON in batches of 20 per
  gate hold; a `RefreshBusy` skips the rest of the update.
- Server binds `127.0.0.1` only, port 8300 by default; started by `python -m Scripts.ops.owner_dashboard`.
- Most likely: per finished match (`FT`, `AET`, `PEN`), the latest read of either stage with
  `evaluated_at < fixtures.kickoff_utc`, `recommendations.block(...)["most_likely"]`, logged once, with
  `recommendations.VERSION`.
- Grading: `FT` only; `AET`/`PEN` pending "extra time"; corners need both teams' rows, else pending
  "corners not in yet".
- Value: `SPIX_SELECTION_POLICY == "market_value"` in `tracking.engine_switches`, `scorecard.cohort(rows,
  "selections")`, outcome from settlement, weights from `scorecard.outcome_weight`; "said" is `model_prob`.
- Report "said" and "hit" over settled picks only; weeks are ISO weeks by kickoff.
- Colours: the live site's tokens, light and dark; chart pair light `#0A6D37` (hit) / `#4D97DE` (said), dark
  `#47AA62` / `#2773C0`; text never in series colours.
- Copy: plain words, no tipster words, no money figures.
- Tests pin every engine switch to its default (existing `Scripts/tests/conftest.py`); the golden check stays
  unchanged.

## Review Focus

- A match postponed after its first read, played later from a new read: the logged picks are the new read's.
- The refresh takes the gate between update batches: the update stops, keeps the batches it wrote, and the
  page says when it last updated.
- A read with a market missing or no prices: its match is marked seen with no picks and never fetched again.
- A Value pick settled half-won, pushed or void: the report weighs it as the scorecard does.
- A Value handicap whose team is spelled differently ("As Roma" against "AS Roma"): labelled by home or away;
  a team that matches neither keeps the engine's wording.

---

### Task 1: The dashboard's file and the Most likely log

**Files:**
- Create: `Scripts/ops/owner_dashboard/__init__.py`, `Scripts/ops/owner_dashboard/store.py`,
  `Scripts/ops/owner_dashboard/sync.py`
- Test: `Scripts/tests/test_owner_dashboard_sync.py`

**Interfaces:**
- Produces:
  - `store.connect(path) -> sqlite3.Connection` (creates the schema if missing; `row_factory = sqlite3.Row`).
    Tables: `matches(fixture_api_id PK, league, home, away, kickoff, status, home_goals, away_goals,
    corners_home, corners_away, yellow_home, yellow_away, red_home, red_away, sot_home, sot_away, read_id,
    read_at, logged_at)`; `most_likely(fixture_api_id, position, market, side, line, label, chance,
    typical_price, version, outcome, pending_reason, graded_at, UNIQUE(fixture_api_id, market, side, line))`;
    `value_picks(selection_key PK, fixture_api_id, kickoff, league, home, away, market, side, line, label, pick,
    odds, bookmaker, said, outcome, actual_result, published_at)`; `meta(key PK, value)`.
  - `sync.log_most_likely(conn, platform_path, *, now, gate=data_access) -> dict` —
    `{"logged": matches, "picks": rows, "skipped_busy": bool}`.
- Consumes: `calibration_data.READ_COLUMNS`, `recommendations.block`, `recommendations.VERSION`.

- [ ] **Step 1: Write the failing tests** (a temp `platform.db` built with the platform's models
  `Base.metadata.create_all`, rows inserted with SQL; read JSON from `test_fair_prices`' `goals_result`,
  `btts_result`, `spreads_result`; `gate=contextlib.nullcontext`)

```python
def test_a_finished_match_logs_the_most_likely_of_its_last_read_before_kickoff_once(...):
    # reads at 10:00 and 18:00 (pre_match, confirmed_lineups) and 21:00, kickoff 20:00, status FT
    assert log_most_likely(conn, db, now=NOW)["logged"] == 1
    assert {r["read_id"] for r in conn.execute("SELECT read_id FROM matches")} == {READ_1800}
    assert log_most_likely(conn, db, now=NOW)["logged"] == 0          # logged once

def test_a_match_not_over_waits_and_a_postponed_one_logs_its_new_read(...):
    # NS -> nothing; PST with a read before the old kickoff, then FT on a new date with a new read -> new read

def test_a_read_with_no_priced_market_marks_the_match_seen_with_no_picks(...):
    assert counts == {"logged": 1, "picks": 0} and second run fetches no read JSON

def test_a_busy_gate_stops_the_update_and_keeps_what_it_wrote(...):
    # gate raises RefreshBusy on its second entry; 25 finished matches -> 20 logged, skipped_busy True
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest Scripts/tests/test_owner_dashboard_sync.py -q` — FAIL (module missing).
- [ ] **Step 3: Implement** `store.py` and `sync.log_most_likely`: scan `match_reads` metadata in 500-row
  chunks and `fixtures` for finished matches not yet in `matches`, choose each match's latest read with
  `evaluated_at < kickoff_utc`, fetch JSON 20 per gate hold, store the match row and its Most likely rows.
- [ ] **Step 4: Run** the tests — PASS.
- [ ] **Step 5: Commit** (tracker D67 gets the dashboard's design and plan lines in this commit).

### Task 2: Match facts and grading

**Files:** Modify: `Scripts/ops/owner_dashboard/sync.py`. Test: `Scripts/tests/test_owner_dashboard_sync.py`

**Interfaces:**
- Produces: `sync.grade(conn, platform_path, *, now, gate=data_access) -> dict` — `{"graded": n, "pending": n}`;
  refreshes `matches` facts (score, status, corners, cards, shots on target per team) for every match with a
  pending Most likely row or a Value pick, then grades Most likely rows.
- Consumes: `recommendations.wins`.

- [ ] **Step 1: Write the failing tests**

```python
def test_goal_markets_grade_from_the_score_and_corners_from_both_teams(...):
    # 2-1 FT, corners 6 + 5: Over 2.5 goals hit, BTTS no miss, home win hit, home -0.5 hit, Over 9.5 corners hit

def test_extra_time_and_missing_corners_stay_pending_and_grade_when_the_corners_arrive(...):
    # AET -> "extra time"; one team row -> "corners not in yet"; second row added, grade again -> graded
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (chunked `IN (...)` queries of 200 ids). **Step 4: Run** — PASS.
- [ ] **Step 5: Commit.**

### Task 3: Value picks

**Files:** Modify: `Scripts/ops/owner_dashboard/sync.py`. Test: `Scripts/tests/test_owner_dashboard_sync.py`

**Interfaces:**
- Produces: `sync.log_value(conn, *, load_rows, today) -> dict` — `{"picks": n}`; `load_rows(days) -> list[dict]`
  defaults to `PredictionRepository().get_recent(days=days, published_only=True, publication_scope="all",
  limit=10**6)` inside `data_access()`. Window: from the oldest unsettled stored pick's kickoff date, else 1 Nov
  2026. `sync.value_label(market, side, line, home, away, pick) -> str`.
- Consumes: `scorecard.cohort`, `recommendations._label` wording (−/+ signs, "goals"/"corners").

- [ ] **Step 1: Write the failing tests**

```python
def test_only_stage_4_picks_each_selection_once_and_outcomes_update(...):
    # legacy row ignored; initial + amendment of one selection -> one row; outcome None then "hit" on resync

def test_value_labels_use_the_layers_words():
    assert value_label("spreads", "As Roma", -1.5, "AS Roma", "Sassuolo", "As Roma -1.5") == "AS Roma −1.5"
    assert value_label("corners", "under", 10.0, "Everton", "Coventry", "Under 10") == "Under 10 corners"
    assert value_label("moneyline", "away", None, "Lyon", "Lens", "Away") == "Lens to win"
    assert value_label("spreads", "Lens B", 0.5, "Lyon", "Lens", "Lens B 0.5") == "Lens B 0.5"
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS). **Step 5: Commit.**

### Task 4: The Report page

**Files:** Create: `Scripts/ops/owner_dashboard/report.py`, `Scripts/ops/owner_dashboard/pages.py`,
`Scripts/ops/owner_dashboard/dashboard.css`. Test: `Scripts/tests/test_owner_dashboard_pages.py`

**Interfaces:**
- Produces: `report.summary(conn, *, kind, market=None) -> dict` — `{"weeks": [{"week", "said", "hit",
  "settled", "pending"}], "markets": [{"market", "said", "hit", "settled", "pending"}], "overall": {...}}`;
  `pages.report_page(conn, *, kind="most_likely", market=None, updated=None) -> str` (a full HTML page).

- [ ] **Step 1: Write the failing tests**

```python
def test_said_and_hit_are_taken_over_settled_picks_with_half_results_weighted():
    # Value: hit (0.60), half_hit (0.50), push (0.55), pending (0.70) -> settled 1.5 weight, hit 1.0/1.5, said
    # mean of 0.60 and 0.50 weighted 1 and 0.5; pending 1

def test_weeks_follow_kickoff_and_a_market_can_be_opened_alone():
def test_the_report_page_has_the_tiles_the_legend_and_the_weekly_table():
    # tiles link to ?market=, "Hit" and "Said" in a legend, a <table> of weeks; no money words
def test_an_empty_dashboard_says_so():
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS). **Step 5: Commit.**

### Task 5: The History page

**Files:** Modify: `Scripts/ops/owner_dashboard/pages.py`, `dashboard.css`. Test: `Scripts/tests/test_owner_dashboard_pages.py`

**Interfaces:**
- Produces: `pages.history_page(conn, *, day=None, updated=None) -> str`; `pages.what_happened(market, facts,
  pick) -> str` ("3 goals", "10 corners", "2–1", "Lens lost by 1", "5 cards", "9 shots on target").

- [ ] **Step 1: Write the failing tests**

```python
def test_what_happened_names_the_number_the_pick_settles_on():
def test_history_groups_a_day_by_league_with_one_line_per_pick():
    # Most likely and Value lines under one match; Hit/Miss/Pending; said % and price; local-time day strip
def test_history_opens_on_the_latest_day_with_a_settled_pick_and_skips_days_without_picks():
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS). **Step 5: Commit.**

### Task 6: The server, and a first run on real data

**Files:** Create: `Scripts/ops/owner_dashboard/__main__.py`. Test: `Scripts/tests/test_owner_dashboard_server.py`

**Interfaces:**
- Produces: `__main__.route(conn, path, query) -> (status, content_type, body)`; `__main__.update(conn, ...) ->
  dict` runs `log_most_likely`, `grade`, `log_value` under one lock and records `meta.updated_at`;
  `main(argv)` with `--port 8300`, `--db Index/owner_dashboard.db`, `--no-update`.

- [ ] **Step 1: Write the failing tests**

```python
def test_routes_serve_both_pages_the_stylesheet_and_404():
def test_update_now_runs_one_update_and_a_busy_refresh_is_reported_not_raised():
def test_the_server_binds_localhost_only():
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS); then the full suite and the pinned golden check.
- [ ] **Step 5: First real run** (no refresh running): `python -m Scripts.ops.owner_dashboard` — the first
  update rebuilds every finished match since 5 Sep; record counts, time, and that `platform.db` is untouched
  (its alembic version and table list unchanged). Screenshot both pages, light and dark.
- [ ] **Step 6: Commit** with the tracker's D67 results.
