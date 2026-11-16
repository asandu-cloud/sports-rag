# Research endpoints for the Research Area overhaul — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The read-only data the overhauled Research Area needs: a team's or referee's next matches (B1), one
match's research view (B2) and this season's provisional ranks for players under 900 minutes (B5).

**Architecture:** Two new services in `Scripts/data_platform/services/` (`research_upcoming.py`,
`research_match.py`) with their queries added to `repositories/research.py`, served by three new routes in
`Scripts/web_app/routers/research.py`. Spix's figures come only from the match's stored Match Read, read the way the
match page reads it (`match_read_metrics.match_metrics`, `recommendations`, `fair_prices`); nothing is predicted
afresh. B5 extends `ResearchProfileService._ranking`. No engine switch, no published pick changes, no schema change.

**Tech Stack:** Python 3.13.3 (the shared `.venv`), FastAPI, SQLAlchemy, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-research-area-overhaul-design.md` on the `frontend` branch (02f07ae),
section 6.2 (D107, owner approved 16 Nov). The frontend agreed on 16 Nov: no player shots-on-target prices (the
fields are left out), `lineups` may be null, B5 as in Task 5.

**Start condition:** the owner put these endpoints before game-read task 3 on 16 Nov.

## Progress

Each task has a start commit and a done commit (CLAUDE.md, "Large builds"). To undo a task:
`git revert <its done commit>`.

| Task | What it builds | Status | Start commit | Done commit | Tests |
|---|---|---|---|---|---|
| 1 | B1: a team's and a referee's next matches | Done 16 Nov | 6d7bf08 | the "Done research-endpoints task 1" commit | 6 new, all pass; suite 3,105 passed + the 3 known errors |
| 2 | B2: the match, Spix's read and its lines | Not started | | | |
| 3 | B2: each team's last 10, booked, shooters and what it allows | Not started | | | |
| 4 | B2: the referee, head to head and data basis | Not started | | | |
| 5 | B5: provisional ranks under 900 minutes | Not started | | | |

## Global Constraints

- Branch `backend`; files staged by name, no attribution lines; each task's tracker line in the same commit as its
  code, under **D110** in `docs/rework-tracker.md`.
- Never stage `Output/`, `Index/`, `Research/`, `.impeccable/`, `.superpowers/`, `.claude/`, `.taste-skill/`.
- Every new route reads `platform.db` inside `refresh_coordination.read_gate()` (never waits, never alerts, as pick
  evidence does, D82). While the refresh holds the gate the route answers **503** with
  `{"detail": "Research is updating; try again shortly."}`. Unknown subjects answer 404, bad parameters 422.
- Each route's database work is a few short indexed statements in one session; no whole-season scans in B1.
- Public, read-only, like the rest of `/api/research`: no write, no provider call, no model call.
- Figures: chances as stored (0–1), expected figures rounded to 1 decimal, prices as stored (2 decimals).
- `python -m Scripts.ops.golden_snapshots check` stays unchanged; the full suite passes apart from the 3 known
  errors.
- Ad-hoc checks on the real database run only inside `data_access()`, never while `pgrep -f refresh-season` finds
  the refresh, with `PYTHONPATH=.` from the repo root.

## Shared definitions

- **Finished:** status in `FT`, `AET`, `PEN` (`repositories.research.FINISHED`). **Unplayed:** `NS`, `TBD`
  (`repositories.fixture_schedule.UNPLAYED`).
- **A team's numbers in a match:** from `fixture_team_stats` (cards = yellow + red, as `add_known`; null when the
  side's row is missing), goals from the fixture score.
- **Referee key:** the alias table's key for the raw name; a raw name the table lacks gets the key
  `referee_aliases([raw])` computes (the same first-initial-and-surname signature the build uses).
- **Read status** of an unplayed match, from the match read store the way `research_assistant.next_read` loads it:
  `ready` (the board shows it), `preliminary` (shown, `timing.is_preliminary`), `waiting` (no read, or the board
  withholds it). `opens_at` = kickoff − `MATCH_READ_OUTLOOK_HOURS` (96 h) while that is still ahead, else null.

## Review Focus

1. The refresh holds the gate: every new route answers 503 at once and reads nothing.
2. A match with no read, a withheld (stale) read, and the display switch off: `read` is null or partial, never an
   error.
3. A side's statistics row missing: its numbers are null, never 0, and the match still lists.
4. A referee spelled differently this season (B3): B1 and B2 still name him with the key his profile uses.
5. A team with fewer than 10 finished matches, or a fixture with no referee: shorter lists and nulls, never padding.

---

### Task 1: B1 — a team's and a referee's next matches

**Files:**
- Create: `Scripts/data_platform/services/research_upcoming.py`
- Modify: `Scripts/data_platform/repositories/research.py` (add `upcoming`)
- Modify: `Scripts/web_app/routers/research.py` (two routes)
- Test: `Scripts/tests/data_platform/test_research_upcoming.py`

**Interfaces:**
- Produces: `ResearchRepository.upcoming(session, *, team_id=None, referee_names=None, now, limit) -> list[dict]`
  (unplayed fixtures after `now`, every competition, kickoff order, each with ids, kickoff, competition code and name,
  both teams' id, name, short_code, logo_url, raw referee);
  `research_upcoming.read_summary(fixture_event_id, kickoff, *, load_read, now, outlook_hours) -> dict` (the
  `read` object of the schema);
  `ResearchUpcomingService(repo, load_read, now, outlook_hours).team(team_id, limit)` and `.referee(key, limit)`
  returning `research-upcoming.v1`;
  `research_upcoming.referee_ref(raw, alias_rows) -> dict | None` (`{"name", "referee_key"}`).
- Consumes: `match_read_metrics.match_metrics`, `match_read_cards.build_match_read_card`,
  `research_profiles.referee_aliases`.

- [ ] **Step 1: Write the failing tests** — on the `session_factory` fixture with seeded competitions, teams and
  fixtures: `test_team_upcoming_lists_every_competition_in_kickoff_order` (EPL then UCL then EPL, limit 3, a
  finished and a postponed fixture left out); `test_referee_upcoming_uses_the_alias_key_and_null_until_named`;
  `test_read_status_ready_preliminary_waiting_and_opens_at` (fake `load_read`: card ready with metrics
  goals 2.66 → 2.7, corners, cards null; preliminary; None → waiting with `opens_at` = kickoff − 96 h; past that,
  `opens_at` null); `test_upcoming_routes_404_422_and_503_while_the_gate_is_shut` (router with a `read_gate` that
  raises `RefreshBusy` → 503; unknown team → 404; `limit=0` and `limit=11` → 422).
- [ ] **Step 2: Run them** — `.venv/bin/python -m pytest Scripts/tests/data_platform/test_research_upcoming.py -q`
  Expected: FAIL (module not found).
- [ ] **Step 3: Implement.** Routes: `GET /api/research/teams/{team_id}/upcoming?limit=` (1–10, default 3) and
  `GET /api/research/referees/{referee_key}/upcoming?limit=` (1–10, default 1). The referee route matches the raw
  names whose computed key equals the asked key among unplayed fixtures' referees, so a spelling the alias table
  lacks still counts. `venue` is the subject team's side (`home`/`away`); null on the referee route.
- [ ] **Step 4: Run the tests** — Expected: PASS. Then the research test files:
  `.venv/bin/python -m pytest Scripts/tests/data_platform/test_research_profiles.py Scripts/tests/data_platform/test_research_upcoming.py -q`
- [ ] **Step 5: Done commit** with the progress row and the tracker line.

### Task 2: B2 — the match, Spix's read and its lines

**Files:**
- Create: `Scripts/data_platform/services/research_match.py`
- Modify: `Scripts/web_app/routers/research.py` (`GET /api/research/matches/{event_id}`)
- Test: `Scripts/tests/data_platform/test_research_match.py`

**Interfaces:**
- Consumes: Task 1's fixture row shape and `read_summary`; `recommendations.options`, `fair_prices._entry`,
  `fair_prices._lines`, `fair_prices.typical_price`, `devig.audit_books`.
- Produces: `research_match.read_block(card, canonical_results, *, status) -> dict` (the schema's `read`);
  `research_match.total_lines(canonical_results, market) -> list[dict]` for `goals`, `corners`, `cards`;
  `ResearchMatchService.match(event_id) -> dict` (`research-match.v1`, with `teams`, `referee`, `h2h` filled by
  Tasks 3–4; until then `teams` null, `referee` null, `h2h` []).

- [ ] **Step 1: Write the failing tests** — on a saved real read (a copy of one delivered read's card and canonical
  results, trimmed, in `Scripts/tests/fixtures/research_match_read.json`):
  `test_lines_pair_over_and_under_per_half_line_with_chance_and_typical_price` (goals lines equal the
  `recommendations.options` goals options grouped by line); `test_cards_lines_from_the_cards_result_without_a_suggestion`
  (cards lines from `totals_cards_over_under`, `suggestion` null; no cards result → `[]`);
  `test_read_block_figures_round_to_one_decimal_and_keep_chances` (result, team goals, btts_yes, corners and cards
  expected; `markets` is the card's `recommendations.markets`, null with the switch off);
  `test_match_route_404_for_unknown_event_and_null_read_when_waiting`.
- [ ] **Step 2: Run them** — Expected: FAIL (module not found).
- [ ] **Step 3: Implement.** Over/under lines use the goals market's shape (`fair_prices._entry(result, "goals",
  books, line)`) on each market's own books (`totals`, `totals_corners_over_under`, `totals_cards_over_under`), half
  lines only, at least `fair_prices.MIN_BOOKS` books on both sides — the recommendation layer's rule. `lineups`:
  the confirmed line-ups the read stores, else null. `read_at` is the read's `created_at`; `prices_as_of` the
  provenance's `evaluated_at`.
- [ ] **Step 4: Run the tests** — Expected: PASS.
- [ ] **Step 5: Done commit.**

### Task 3: B2 — each team's last 10, booked, shooters and what it allows

**Files:**
- Modify: `Scripts/data_platform/repositories/research.py` (`last_matches`, `booked`, `shooters`, `allowed`)
- Modify: `Scripts/data_platform/services/research_match.py` (`teams`)
- Test: `Scripts/tests/data_platform/test_research_match.py`

**Interfaces:**
- Produces: `teams.home` / `teams.away`, each `{"team": {...}, "last_matches": [...], "booked": [...],
  "shooters": [...], "allowed": {...}}` as section 6.2 lists, without `price_sot_*`.

- [ ] **Step 1: Write the failing tests** — `test_last_matches_are_ten_finished_in_every_competition_newest_last`
  (fixtures before the match's kickoff only; a missing side row gives nulls, not zeros);
  `test_booked_is_the_top_five_by_cards_this_season` (cards = yellow + red, ties by fewer appearances then name);
  `test_shooters_are_up_to_eight_with_most_starts_in_the_last_five` (each with the last 10 appearances
  `{date, opponent, minutes, sot, started}`, newest last); `test_allowed_is_sot_conceded_a_match_and_its_league_rank`
  (rank 1 = fewest conceded, in the team's domestic league this season; `peer_count` given).
- [ ] **Step 2: Run them** — Expected: FAIL.
- [ ] **Step 3: Implement.** "This season" is the season year of the match. The domestic league is the team's
  `domestic_league` competition that season; a team with none (cup-only data) gets `allowed` null.
- [ ] **Step 4: Run the tests** — Expected: PASS.
- [ ] **Step 5: Done commit.**

### Task 4: B2 — the referee, head to head and data basis

**Files:**
- Modify: `Scripts/data_platform/repositories/research.py` (`h2h`)
- Modify: `Scripts/data_platform/services/research_match.py`
- Test: `Scripts/tests/data_platform/test_research_match.py`

- [ ] **Step 1: Write the failing tests** — `test_referee_this_season_cards_league_average_and_last_ten`
  (all competitions, as the referee profile counts; league average over the match's competition this season);
  `test_referee_null_until_named`; `test_h2h_is_up_to_six_meetings_since_2023_newest_first`;
  `test_data_basis_is_the_latest_finished_kickoff_used`.
- [ ] **Step 2: Run them** — Expected: FAIL.
- [ ] **Step 3: Implement**, reusing `referee_summary`/`referee_values` for the referee's counts.
- [ ] **Step 4: Run the tests and the full suite**
  (`.venv/bin/python -m pytest Scripts/tests -q`) and the golden check. Expected: PASS, golden unchanged.
- [ ] **Step 5: Done commit.**

### Task 5: B5 — provisional ranks under 900 minutes

**Files:**
- Modify: `Scripts/data_platform/services/research_profiles.py` (`_ranking`)
- Modify: `Scripts/data_platform/repositories/research.py` (peer values of one scope, metric and peer group)
- Test: `Scripts/tests/data_platform/test_research_profiles.py`

**Interfaces:**
- Produces: for a player whose only reason is `below_minutes_threshold`, each metric with a value gets `rank`,
  `percentile` and `peer_count` placed among the stored ranked peers of the same scope and position
  (`peer_count` = peers + 1), and `ranking.provisional: true`; `ranking.ranked` stays false and `previous_season`
  is kept. Every other profile is unchanged (`provisional: false`).

- [ ] **Step 1: Write the failing tests** — `test_provisional_rank_places_a_low_minutes_player_among_ranked_peers`
  (rank, midrank percentile as `percentiles` gives with the player added, `peer_count` = peers + 1);
  `test_no_provisional_rank_without_a_built_scope_or_for_teams_and_referees`;
  `test_ranked_profiles_are_unchanged` (`provisional: false`).
- [ ] **Step 2: Run them** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite and the golden check.** Expected: PASS, golden unchanged.
- [ ] **Step 5: Done commit**, then tell the owner and the orchestrator: what changed, the tests, no migration,
  no `.env` line, and that the website needs a restart to serve the routes.
