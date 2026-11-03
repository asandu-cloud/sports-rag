# Recommendation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every match on the website shows what Spix would take in each market (Most likely, Value, Explore the
match), the bettor can add any priced option, and goal-market legs from one match combine through the match's score
grid.

**Architecture:** A pure module `Scripts/data_platform/recommendations.py` turns a stored read's canonical results
into a `recommendations.v1` block (options per market, Spix's suggestion per market, Most likely, Value, the score
grid). `build_match_read_card` attaches it when the display switch `SPIX_RECOMMENDATIONS=on`. The front end renders
the block and the slip combines legs. A small ledger logs the Most likely options of the last read before kickoff,
settles them from the canonical `fixtures` tables and feeds a new scorecard section. No engine change.

**Tech Stack:** Python 3.11, SQLAlchemy + Alembic (platform.db), pytest; vanilla JS (`app.js`), node assert tests
run with `/usr/local/bin/node`.

**Spec:** `docs/superpowers/specs/2026-11-03-recommendation-layer-design.md` (approved 3 Nov, answers at its end;
tracker D52, D53).

## Global Constraints

- No engine change and no engine switch. The published Value picks (stage 4) must not change.
- Chance = the market's fair chance, power method, median across books (`fair_prices`). An option needs at least
  `fair_prices.MIN_BOOKS` (2) bookmakers pricing every side of its line.
- Goals, Corners and Handicap: half lines only (x.5). Result and Both teams to score have no line.
- Typical price = median across bookmakers of each book's best price for that outcome, rounded to 2 decimals. No
  bookmaker named anywhere in the block or on screen.
- Most likely: chance ≥ 0.50, typical price ≥ 1.30, at most one per market, at most 3, all able to win together
  (scorelines 0–0 to 10–10 for goal markets; corners never conflict), ranked by chance, ties to the higher price.
- Value: up to 2, the read's released selections, shown as "Worth it at X or better" (`fair_prices.minimum_odds` of
  the outcome's fair chance).
- Cards and shots on target are not in the block.
- Long shot: chance under 0.25 says "a long shot, about 1 in N" once, in the option.
- Same-match legs: goal markets only, combined chance from the score grid. A corners leg with a goals leg from the
  same match is refused: "Corners and goals from one match can't be combined yet."
- Copy: plain words, no "AI", no tipster words ("lock", "banker", "sure thing", "hot"), no push to add legs or stake.
  Under the stake: "Check the price in your bookmaker's app before you place it." Each list: "Prices as of HH:MM".
- Display switch `SPIX_RECOMMENDATIONS`, values `off` (default) / `on`, read from the environment at card-build time.
- Visual changes: checkpoint commit before, one revertible commit after; `?recs=off` in the URL renders today's page
  for side-by-side comparison.
- Safety rules from the handoff hold: never read `.env`; read `Index/platform.db` only inside `data_access()`; never
  stage `Index/`, `Output/`, `Research/`, `.claude/`, `.impeccable/`, `.superpowers/`, `.taste-skill/`.
- Log every plan change or finding in `docs/rework-tracker.md` in the same commit as the work.

## Review Focus

1. **A read with no stored prices** (status `unavailable`, empty `input_options`, as Viking v Sabah on 4 Nov): the
   block must be present with empty lists and `score_grid: null`, and the page must show "Nothing stands out here"
   per market, not crash. Test in Task 1 and Task 4.
2. **A Value pick on a whole or quarter line** (live example: Barcelona v Aston Villa, "Over 5" goals): it is not an
   Explore option (half lines only) but must still get its "Worth it at" from `fair_outcome`. Test in Task 2.
3. **No goal model fit for the match** (`goal_model` null when prices cannot identify both teams): same-match goal
   legs must be refused with a plain line, never combined by multiplying. Test in Task 5.
4. **Extra time** (`AET`/`PEN` in knockout ties): the canonical score includes extra time, bookmakers settle on 90
   minutes; the ledger must leave these rows ungraded with reason `extra_time`. Test in Task 6.
5. **Two legs where one covers the other** (Over 1.5 and Over 2.5, same match): the grid counts the scorelines once;
   the combined chance must equal Over 2.5's alone. Test in Task 5.

---

### Task 1: Options, suggestions and Most likely from a read

**Files:**
- Create: `Scripts/data_platform/recommendations.py`
- Modify: `Scripts/data_platform/fair_prices.py` (add one public helper, below)
- Test: `Scripts/tests/data_platform/test_recommendations.py`

**Interfaces:**
- Consumes: `fair_prices._results`, `_books`, `_lines`, `_entry`, `_teams`, `MARKETS`, `MIN_BOOKS`;
  `reference_price.norm`.
- Produces:
  - `fair_prices.typical_price(books, market: str, side: str, line) -> Optional[float]` — median of each book's best
    odds for that outcome (spreads: `side` is the normalised team, `line` that team's handicap), 2 decimals; None
    when fewer than `MIN_BOOKS` books price it.
  - `recommendations.VERSION = "recommendations.v1"`, `MIN_CHANCE = 0.50`, `MIN_PRICE = 1.30`,
    `MAX_MOST_LIKELY = 3`, `MAX_VALUE = 2`, `LONG_SHOT = 0.25`, `MAX_SCORE = 10`,
    `MARKETS = ("moneyline", "goals", "btts", "spreads", "corners")`, `GOAL_MARKETS = MARKETS[:4]`.
  - `options(canonical_results) -> Dict[str, List[dict]]` — per market, every option, each
    `{"market", "side", "line", "label", "chance", "typical_price", "long_shot": bool}`; lines ascending; sides in
    `fair_prices.SIDES` order (spreads: home team first).
  - `wins(option: Mapping, home: int, away: int) -> bool` — goal markets only.
  - `can_win_together(a: Mapping, b: Mapping) -> bool`
  - `suggestion(market_options: List[dict]) -> Optional[dict]`
  - `most_likely(suggestions: Iterable[dict]) -> List[dict]`

- [ ] **Step 1: Write the failing tests** (reuse the `option`/`result` builders of `test_fair_prices.py`; import
  them or copy the two small helpers)

```python
def test_only_half_lines_priced_by_two_books_on_both_sides_are_options():
    # GOALS fixture of test_fair_prices: 2.5 (books A, B, C) and 3.5 (A, B) qualify; 1.5 (D only) does not
    opts = rec.options([goals_result()])["goals"]
    assert {(o["side"], o["line"]) for o in opts} == {("over", 2.5), ("under", 2.5), ("over", 3.5), ("under", 3.5)}

def test_typical_price_is_the_median_of_the_books_prices_and_chance_is_the_fair_card_chance():
    over = next(o for o in rec.options([goals_result()])["goals"] if o["side"] == "over" and o["line"] == 2.5)
    assert over["typical_price"] == 2.00            # median of 1.90, 2.05, 2.00
    assert over["chance"] == pytest.approx(fp_line(fp.fair_card([goals_result()]), "goals")["outcomes"][0]["fair_probability"])

def test_labels_are_plain_words():
    labels = {o["label"] for m in rec.options([goals_result(), moneyline_result()[0], spreads_result()]).values() for o in m}
    assert {"Over 2.5 goals", "Under 3.5 goals", "Lyon to win", "Draw", "Lens to win", "Lyon −0.5", "Lens +0.5"} <= labels

def test_a_read_with_no_prices_gives_empty_markets():
    assert rec.options([result("goals", [])]) == {"goals": []}

def test_wins_settles_each_goal_market_on_a_scoreline():
    assert rec.wins({"market": "goals", "side": "over", "line": 2.5}, 2, 1)
    assert not rec.wins({"market": "btts", "side": "yes", "line": None}, 3, 0)
    assert rec.wins({"market": "spreads", "side": "home", "line": -1.5}, 2, 0)
    assert not rec.wins({"market": "spreads", "side": "away", "line": 1.5}, 2, 0)

def test_can_win_together_checks_scorelines_and_ignores_corners():
    draw = {"market": "moneyline", "side": "draw", "line": None}
    assert not rec.can_win_together(draw, {"market": "spreads", "side": "home", "line": -0.5})
    assert rec.can_win_together(draw, {"market": "goals", "side": "over", "line": 2.5})   # 2-2
    assert rec.can_win_together(draw, {"market": "corners", "side": "over", "line": 9.5})

def test_suggestion_needs_half_chance_and_a_price_of_1_30_and_prefers_the_higher_chance():
    opts = [mk(chance=0.92, price=1.05), mk(chance=0.71, price=1.33), mk(chance=0.55, price=1.70)]
    assert rec.suggestion(opts)["chance"] == 0.71
    assert rec.suggestion([mk(chance=0.45, price=2.1)]) is None

def test_most_likely_keeps_three_compatible_markets_by_chance_with_price_breaking_ties():
    picks = rec.most_likely([home_win_60, under_1_5_goals_58, btts_yes_57, corners_over_55])
    # under 1.5 and BTTS yes cannot both win (needs 1-1 = 2 goals), so BTTS drops and corners takes its place
    assert [p["market"] for p in picks] == ["moneyline", "goals", "corners"]
```

`mk`, `home_win_60` etc. are small dict builders in the test file (market, side, line, chance, typical_price).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_recommendations.py`
Expected: FAIL (`ModuleNotFoundError: data_platform.recommendations`)

- [ ] **Step 3: Implement `typical_price` in `fair_prices.py` and the functions above in `recommendations.py`**

`options` walks `fair_prices._lines` per market, keeps `line is None or line % 1 == 0.5` (spreads: the home line),
reuses `_entry` for the chances (so the 1X2 rescale and `MIN_BOOKS` rule are the card's), and adds `typical_price`
per outcome; an outcome with no typical price drops its whole line. Labels: goals/corners "Over 2.5 goals" /
"Under 9.5 corners"; moneyline "<Team> to win" / "Draw"; btts "Both teams to score" / "Not both teams to score";
spreads "<Team> −1.5" / "+1.5" (U+2212 minus), team names from the read's fixture, not the normalised key.
`can_win_together`: True if either is corners; else any `(h, a)` in `0..MAX_SCORE` where both `wins`.
`most_likely`: sort by `(-chance, -typical_price)`, add greedily when compatible with every kept pick, stop at
`MAX_MOST_LIKELY`.

- [ ] **Step 4: Run tests to verify they pass, plus the fair-price tests**

Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_recommendations.py Scripts/tests/data_platform/test_fair_prices.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add Scripts/data_platform/recommendations.py Scripts/data_platform/fair_prices.py Scripts/tests/data_platform/test_recommendations.py
git commit -m "Recommendation layer (D52): options per market, Spix's suggestion and Most likely from a read's stored prices"
```

### Task 2: The `recommendations.v1` block on the card, behind the display switch

**Files:**
- Modify: `Scripts/data_platform/recommendations.py`
- Modify: `Scripts/data_platform/services/match_read_cards.py:42-80` (one new key)
- Test: `Scripts/tests/data_platform/test_recommendations.py`, `Scripts/tests/data_platform/test_match_read_api.py`

**Interfaces:**
- Consumes: Task 1; `fair_prices.fair_outcome`, `fair_prices.minimum_odds`; the card's built `selections` and
  `metrics`.
- Produces:
  - `recommendations.enabled() -> bool` — `os.environ.get("SPIX_RECOMMENDATIONS", "").strip().lower() == "on"`.
  - `recommendations.block(canonical_results, *, selections, metrics, prices_as_of) -> dict`:
    `{"version", "prices_as_of", "most_likely": [option + "why"], "value": [...], "markets": [{"market",
    "suggestion": option|None, "options": [...]}], "score_grid": {"home": float, "away": float}|None}`.
    `value` items: `{"position", "market", "pick", "chance", "worth_it_at", "published_recommendation_id"}`.
    `score_grid` is the read's `context.market_anchor.goal_model` home/away means, from any goal-market result.
  - Card key `"recommendations"`: the block when `enabled()`, else `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_value_shows_worth_it_at_the_minimum_odds_even_on_a_whole_line():
    sel = {"position": 1, "market": {"group": "goals"}, "pick": "Over 5", "published_recommendation_id": 9,
           "quote": {"side": "over", "line": 5.0}}
    value = rec.block([whole_line_goals_result()], selections=[sel], metrics={}, prices_as_of="2026-11-03T11:13:27")["value"]
    chance = fp.fair_outcome([whole_line_goals_result()], "goals", "over", 5.0)["fair_probability"]
    assert value == [{"position": 1, "market": "goals", "pick": "Over 5", "chance": pytest.approx(chance),
                      "worth_it_at": fp.minimum_odds(chance), "published_recommendation_id": 9}]

def test_every_market_is_listed_with_its_suggestion_or_none_and_cards_are_absent():
    b = rec.block(all_markets_results(), selections=[], metrics={}, prices_as_of=None)
    assert [m["market"] for m in b["markets"]] == list(rec.MARKETS)
    assert "cards" not in {m["market"] for m in b["markets"]}

def test_score_grid_comes_from_the_stored_goal_model_and_is_null_without_one():
    assert rec.block(results_with_goal_model(1.44, 2.32), selections=[], metrics={}, prices_as_of=None)["score_grid"] == {"home": 1.44, "away": 2.32}
    assert rec.block([goals_result()], selections=[], metrics={}, prices_as_of=None)["score_grid"] is None

def test_card_carries_the_block_only_when_the_switch_is_on(monkeypatch):   # in test_match_read_api.py style
    monkeypatch.delenv("SPIX_RECOMMENDATIONS", raising=False)
    assert build_match_read_card(stored_read())["recommendations"] is None
    monkeypatch.setenv("SPIX_RECOMMENDATIONS", "on")
    assert build_match_read_card(stored_read())["recommendations"]["version"] == "recommendations.v1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform/test_recommendations.py Scripts/tests/data_platform/test_match_read_api.py`
Expected: FAIL (`AttributeError: block` / `KeyError: 'recommendations'`)

- [ ] **Step 3: Implement `block`, `enabled` and the card key**

The selection's side and line come from `selection.data.decision.quote` (`side` or `team`, and `line`); the card
builder passes the raw stored selections alongside its built ones, or `_selection_card` exposes `quote` — pick the
smaller change. `why` per Most likely option is a fixed template filled from `metrics` (owner signs off the copy):
goals "Spix expects about {total:.1f} goals."; btts and result/handicap "Spix expects {home} about {hg:.1f} goals
and {away} about {ag:.1f}."; corners "Spix expects about {corners:.1f} corners." (the corners result's
`projection.value`). A missing figure gives `why: None`. `prices_as_of` is the read's `provenance.evaluated_at`.

- [ ] **Step 4: Run tests to verify they pass, plus the card and golden suites**

Run: `.venv/bin/python -m pytest -q Scripts/tests/data_platform Scripts/tests/test_golden_engine.py`
Expected: all PASS (the switch is off in tests, so existing card snapshots are unchanged)

- [ ] **Step 5: Real-data check.** Inside `data_access()`, build the block for every read of 3–4 Nov and print, per
  fixture, Most likely labels with chance and price, and the count of markets with a suggestion. Expected: every
  priced fixture has at least one Most likely option; Villarreal v PSG shows "Both teams to score" near 71%. Record
  the counts in the tracker.

- [ ] **Step 6: Commit** (code, tests and the tracker line)

### Task 3: Mockups with real matches for the owner (no product code)

**Files:**
- Create: scratchpad HTML only, published as one Artifact with 2–3 versions side by side.

- [ ] **Step 1:** Run `Artifact` `action: "quickstart"`, `intent: "design"`, and follow it (DESIGN.md and
  `docs/brand/` tokens: turf #0E6336 / #89CD9B, night theme, neon #E8FF2A on main buttons only, never as text on
  Chalk; Fraunces wordmark).
- [ ] **Step 2:** Use Task 2's blocks for three real fixtures: one with a Value pick (Barcelona v Aston Villa or
  Bodo/Glimt v Lille), one with none (Villarreal v PSG), one with thin prices. Each version shows the board card,
  the match page top (Most likely, Value) and Explore the match with the Goals stepper open, at phone and desktop
  width. Versions differ in layout only (e.g. list rows / tiles / a single scrolling sheet).
- [ ] **Step 3:** Publish, send the link, and ask the owner to pick one (AskUserQuestion). Record the choice in the
  tracker (D-entry). **Tasks 4–5 wait for this answer.**

### Task 4: Match page and board card (the chosen version)

**Files:**
- Modify: `Scripts/web_app/static/app.js` (`_normaliseMatchReadCard` ~1008, `_createMatchReadFixtureCard` ~1419,
  `_renderMatchReadSummary` ~1557, remove the "Other model angles" block ~1622-1631)
- Modify: `Scripts/web_app/static/style.css`
- Test: `Scripts/tests/test_recommendations_web.js` (new, same `vm` harness as `test_frontend_interactions.js`)

**Interfaces:**
- Consumes: the card's `recommendations` block (Task 2).
- Produces: `chanceWords(p: number) -> string`, `renderRecommendations(f) -> string`, `renderMarketExplorer(f) ->
  string`, view field `f.recommendations` (null when absent or `?recs=off`).

- [ ] **Step 1: Checkpoint commit** of the current front end state (no changes): `git commit --allow-empty -m
  "Checkpoint before the recommendation layer front end"`.
- [ ] **Step 2: Write the failing tests**

```js
assert.equal(chanceWords(0.76), 'about 3 in 4');
assert.equal(chanceWords(0.17), 'about 1 in 6');
assert.equal(chanceWords(0.52), 'about 1 in 2');
// a block with empty markets renders "Nothing stands out here" for each market chip and no Add buttons
// a long shot (chance 0.2) renders "a long shot, about 1 in 5" once
// the Value list with no items renders "No value picks on this match today. The prices are about right."
// Value items render "Worth it at 2.10 or better" and no bookmaker name
// ?recs=off: _normaliseMatchReadCard sets f.recommendations = null and the old summary renders
// the board card shows the first Most likely label and "3 suggestions" (markets with a suggestion + value items)
```

`chanceWords` picks the nearest of 1/10, 1/8, 1/6, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 9/10.

- [ ] **Step 3: Run to verify failure:** `/usr/local/bin/node Scripts/tests/test_recommendations_web.js` → fails
  (`chanceWords is not defined`).
- [ ] **Step 4: Implement** the chosen design: "Spix's picks for this match" (Most likely with label, chance in words
  with % beside it, typical price, why, Add; Value with "Worth it at X or better"), "Prices as of HH:MM", then
  "Explore the match" chips (Result, Goals, Both teams to score, Handicap, Corners), first market open, the others
  closed; Result as three tiles, BTTS yes/no, Goals/Corners/Handicap with a − / + line stepper (swipe on phones)
  over the market's lines; Spix's suggestion marked; Add on every option. Remove "Other model angles".
  In `Scripts/web_app/static/methodology.html`, add one short section: the chance shown is the bookmakers' own view
  without their margin; Spix's work is choosing, explaining and spotting value; Most likely is measured on its own
  (spec section 4). Owner signs off the copy.
- [ ] **Step 5: Run** both JS test files and the Python suite → PASS.
- [ ] **Step 6: Real-data screenshots.** Start the site locally with `SPIX_RECOMMENDATIONS=on` and capture phone and
  desktop screenshots of the three Task 3 fixtures (use the `run` skill). Fix anything that differs from the chosen
  mock. Show the owner before calling it done.
- [ ] **Step 7: Commit** (front end, tests, tracker line) as one revertible commit.

### Task 5: Slip — any option, and goal legs from one match through the score grid

**Files:**
- Modify: `Scripts/web_app/static/app.js` (`slip` ~536-670)
- Test: `Scripts/tests/test_recommendations_web.js`

**Interfaces:**
- Consumes: option objects from Task 4 (market, side, line, chance, typical_price) and `f.recommendations.score_grid`.
- Produces: `jointChance(legs, grid) -> number|null` (Poisson grid 0–12 per team, renormalised, the same as
  `market_goal_model.score_probabilities`), `slip.add(fixture, pick, odds, market, options)` accepting
  `options.leg = {market, side, line, chance}` and `options.grid`.

- [ ] **Step 1: Write the failing tests** (values computed from `data_platform.market_goals._matrix(1.4, 1.1)`)

```js
const grid = { home: 1.4, away: 1.1 };
const over25 = { market: 'goals', side: 'over', line: 2.5 }, over15 = { market: 'goals', side: 'over', line: 1.5 };
assert.ok(Math.abs(jointChance([over25], grid) - 0.4562) < 1e-4);
assert.ok(Math.abs(jointChance([over25, { market: 'btts', side: 'yes', line: null }], grid) - 0.3762) < 1e-4);
assert.ok(Math.abs(jointChance([{ market: 'moneyline', side: 'home', line: null }, { market: 'goals', side: 'under', line: 1.5 }], grid) - 0.1149) < 1e-4);
assert.ok(Math.abs(jointChance([over15, over25], grid) - 0.4562) < 1e-4);               // covered leg counts once
assert.equal(jointChance([over25], null), null);
// slip: goals + corners from one fixture -> refused with "Corners and goals from one match can't be combined yet."
// slip: two goal legs from one fixture with grid null -> refused with "Combining bets from one match needs prices for both teams."
// slip: two goal legs that cannot both win (jointChance 0) -> refused with "These two can't both win."
// slip: legs from different fixtures -> combined chance is the product; shown once as "about 1 in N"
// slip: stake area shows "Check the price in your bookmaker's app before you place it."
```

Spreads in `jointChance` use the home team's view: `side` is "home"/"away" in the leg (Task 4 maps the team to it).

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** Same-fixture group total: "about" the fair price `1 / jointChance`, rounded to 2
  decimals (D51: never the product of the legs' prices); single legs keep their typical price; groups multiply
  across fixtures. Legs added from older flows without `leg` keep today's one-per-fixture rule.
- [ ] **Step 4: Run** JS tests → PASS.
- [ ] **Step 5: Commit** (with a tracker line: "fair price about X" on same-match groups is the builder's call, for
  the owner to confirm).

### Task 6: Most likely ledger — log the last read before kickoff, settle from the canonical tables

**Files:**
- Create: `Scripts/data_platform/models/recommendation_options.py`, register in `models/__init__.py`
- Create: `Scripts/data_platform/alembic/versions/0010_recommendation_options.py` (revises `0009_research_profiles`)
- Create: `Scripts/data_platform/services/recommendation_ledger.py`
- Modify: `Scripts/data_platform/services/measurement_cycle.py:186-200` (call the ledger inside the same
  `data_access()` block, after the cycle, in its own try/except so it can never fail the cycle)
- Test: `Scripts/tests/data_platform/test_recommendation_ledger.py`

**Interfaces:**
- Consumes: `recommendations.block`, `recommendations.wins`, `MatchReadService.list_for_fixture`, `Fixture`
  (`status`, `home_goals`, `away_goals`), `FixtureTeamStats.corners`.
- Produces:
  - Table `recommendation_options`: `id`, `fixture_api_id` (str), `match_read_id` (int), `list` (str, "most_likely"),
    `version` (str), `market`, `side`, `line` (nullable float), `label`, `chance`, `typical_price`, `kickoff`
    (tz datetime), `logged_at`, `outcome` (nullable: "hit"/"miss"), `pending_reason` (nullable), `graded_at`.
    Unique `(fixture_api_id, list, market, side, line)`.
  - `RecommendationLedger(session_factory).sync(now: datetime) -> dict` — `{"logged": n, "graded": n, "pending": n}`.

- [ ] **Step 1: Write the failing tests** (in-memory SQLite via the existing `session_factory` fixture)

```python
def test_logs_the_most_likely_of_the_last_read_before_kickoff_once(session_factory, reads_for_one_fixture):
    # reads at 10:00, 18:00 and 21:00 (after a 20:00 kickoff): the 18:00 read is logged, twice-run syncs log once
    ledger = RecommendationLedger(session_factory)
    assert ledger.sync(at("2026-11-04T22:00Z"))["logged"] == 3
    assert ledger.sync(at("2026-11-04T23:00Z"))["logged"] == 0

def test_nothing_is_logged_while_the_display_switch_is_off(monkeypatch, ...):
    monkeypatch.delenv("SPIX_RECOMMENDATIONS", raising=False)
    assert ledger.sync(...)["logged"] == 0

def test_grades_goal_markets_from_the_score_and_corners_from_both_teams_stats(...):
    # 2-1 FT: Over 2.5 goals hit, BTTS no miss; corners 6 + 5: Over 9.5 corners hit
def test_extra_time_and_missing_corners_stay_pending(...):
    # status AET -> pending_reason "extra_time"; a team stats row missing -> "corners_unavailable"
def test_unfinished_fixtures_stay_ungraded(...):   # status NS / PST -> outcome None, reason "fixture_not_final"
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** model, migration and ledger. Logged only when `recommendations.enabled()` at sync time.
  Fixtures to log: kickoff ≤ now, no rows yet, within the last 3 days. The read: latest `provenance.evaluated_at`
  before kickoff (ties to the higher id). Grade only status `FT`.
- [ ] **Step 4: Run** the new tests and `Scripts/tests/data_platform` → PASS.
- [ ] **Step 5: Commit** (with tracker line: a dedicated table, not the predictions table, so the Most likely
  cohort can never enter the Value track record or the stage measurement — the builder's reading of spec section 5).

### Task 7: Scorecard section "Most likely: said x%, happened y%"

**Files:**
- Modify: `Scripts/data_platform/scorecard.py` (new `most_likely(rows) -> dict` and a markdown section)
- Modify: `Scripts/ops/scorecard.py` (`load_rows` also reads the ledger; `build` gets `most_likely_rows`)
- Test: `Scripts/tests/data_platform/test_scorecard.py`

**Interfaces:**
- Consumes: ledger rows (Task 6).
- Produces: `scorecard.most_likely(rows) -> {"overall": slice, "by_market": {market: slice}}`, slice =
  `{"shown": n, "settled": n, "said": mean chance, "happened": hit share}`; below `MIN_SETTLED` a slice shows counts
  only, as the existing tables do. Card key `"most_likely"`.

- [ ] **Step 1: Failing test:** 4 settled rows (chances 0.6, 0.6, 0.7, 0.7; hits 1, 0, 1, 1) give
  `said == 0.65`, `happened == 0.75`, `settled == 4`; markdown contains "## Most likely" and, under `MIN_SETTLED`,
  only counts.
- [ ] **Step 2: Run to verify failure.** **Step 3: Implement.** **Step 4: Run** `test_scorecard.py` → PASS.
- [ ] **Step 5: Commit.**

### Task 8: Ship

- [ ] **Step 1:** Run the full Python suite and both JS tests. Expected: PASS.
- [ ] **Step 2:** With no refresh running (`pgrep -f refresh-season` empty) and inside the gate, apply the migration:
  `PYTHONPATH=. .venv/bin/python -m Scripts.data_platform init-db`. Check the size of `Index/platform.db` first and
  copy it beside itself if there is room.
- [ ] **Step 3:** Tell the owner where to add `SPIX_RECOMMENDATIONS=on` in `.env` (never read or edit `.env`
  yourself); confirm with `recommendations.enabled()` after `load_dotenv` with the explicit path.
- [ ] **Step 4:** Restart the website (launchd `com.bettingrag.website`), load `/app` and one match page, and
  compare with `?recs=off`. Screenshots to the owner.
- [ ] **Step 5:** Tracker: Status and open-items rows (recommendation layer live, date and time; measurement
  section in the next Monday scorecard), D-entry if anything changed from this plan. Commit.
