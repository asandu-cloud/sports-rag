# Research Area overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Research Area into the step before a bet: answer-first profiles, a team's next matches (rail on
desktop, floating bar on the phone) and a match view with every market's numbers.

**Architecture:** The Research Area stays a web component in a shadow root (`research.js`). New pure render code
goes in three ES modules beside it: `insight.js` (lines, hits, labelled charts, formatting), `upcoming.js` (rail,
bar, sheet) and `match.js` (match view). `research.js` keeps routing, data loading and events. The match view and
the next matches read two new backend endpoints (B1, B2 in the spec) and stay hidden while those answer 404.

**Tech Stack:** Plain ES modules and CSS in `Scripts/web_app/static/research/`; Node tests run with
`/usr/local/bin/node Scripts/tests/<file>.js`.

**Spec:** `docs/superpowers/specs/2026-11-16-research-area-overhaul-design.md`

## Global Constraints

- Frontend files only: `Scripts/web_app/static/**`, `Scripts/tests/*_web.js`, `Scripts/tests/fixtures/research_*.json`,
  the tracker and this plan. Backend code (`Scripts/data_platform`, `Scripts/web_app/routers`) is asked for, never edited.
- Every commit that changes a served file bumps its `?v=` tag: research files use `research-20261116a`, `b`, `c`…
  (one letter per task), set in `index.html` (research.js), `research.js` (content.html, research.css, the new modules).
- Large-build commits: `Start research task N: <what>` (plan row "In progress"), then
  `Done research task N: <what now works> (<tests>, D107)` with code, tests, the plan row "Done" and the tracker line.
  Fixes inside a task: `Fix research task N: <what was wrong>`. Stage files by name; no attribution lines.
- One decimal at most; percentages whole; "N of M" beside any rate; zeros drawn with "0"; a dash only for missing data.
- Colour: cobalt (`--accent`) for the bettor's selection and hits only; the lime `<mark>` on one figure per screen.
- Copy: "statistical models", never "AI"; no tipster words, no profit or win-rate claims, nothing that urges a bet;
  one foot line per page: "Past matches are records, not a forecast. Spix's figures come from statistical models.
  18+ · GamCare" (Spix's figures sentence only where Spix figures show).
- Phone: 44 px tap targets, 12 px type floor, no sideways page scroll at 390 px.
- Lines: cards 2.5/3.5/4.5/5.5 (default 3.5), goals 1.5/2.5/3.5 (2.5), corners 8.5/9.5/10.5/11.5 (9.5),
  player counts 1+/2+/3+ (1+). Over means total > line; "N+" means total ≥ N.

## Review Focus

1. A match with missing statistics: the square shows a dash and the count drops to "6 of 9" — Task 1 test
   `hitCount skips nulls`.
2. Fewer than 10 past matches (a promoted team, an early-season referee): headings say "last 6", never padded —
   Task 2 test `lead sentence uses the real count`.
3. The endpoints answer 404 or fail: rail, bar and match links are absent, the team page still renders — Task 5 test
   `no upcoming means no rail and no bar`, Task 6 test `match view error state`.
4. The phone bar over the slip bar: the bar's bottom offset follows `--slip-bar-h` and hides while the slip sheet is
   open — Task 5 test `bar offset follows the slip bar`.
5. Prices or a read missing: no price column, no "–" price, "Spix's read comes out Thu 19 Nov" — Task 6 test
   `read waiting`, Task 8 test `no price column without prices`.

---

### Task 1: Lines, hits and clearer numbers

**Files:**
- Create: `Scripts/web_app/static/research/insight.js`
- Modify: `Scripts/web_app/static/research/research.js` (formatting, cards, home and away, notices, foot line)
- Modify: `Scripts/web_app/static/research/research.css`, `Scripts/web_app/static/index.html` (tag)
- Create: `Scripts/tests/test_research_insight_web.js`, `Scripts/tests/fixtures/research_team_chelsea.json`
  (the real 15 Nov Chelsea profile), `research_player_palmer.json`, `research_referee_taylor.json`

**Interfaces — Produces (`insight.js`, no imports):**
- `escape(value) -> string`
- `one(value) -> string`: at most one decimal, `'–'` when not finite (`one(2.67) === '2.7'`, `one(3) === '3'`).
- `LINES`, `DEFAULT_LINE`: the spec's lines per market `cards|goals|corners|sot`.
- `fromProfileRow(m) -> MatchRow`: a profile `match_log` row to the B2 `last_matches` shape
  `{event_id, date, venue, competition, opponent, score, cards_for, cards_against, corners_for, corners_against,
  goals_for, goals_against, sot_for, sot_against}` (`cards_for` = `values.cards`, `cards_against` =
  `values.opponent_cards`, `sot_for` = `values.sot_for` …).
- `marketTotal(row, market) -> number|null` (both sides added; null if either side is null).
- `hitCount(values, line) -> {hits, of}`: nulls skipped; `line` 2.5 means > 2.5, `line` 1 (player) means ≥ 1.
- `passes(value, line) -> bool|null`; `strip(values, test, labels) -> html`: the sketch's strip, one 14 px square a
  match (`y` filled, `na` dashed when not recorded), each with a title, the strip labelled "N of M" (Ruling, 16 Nov:
  the sketch's strip replaces the plan's squares with totals inside; the sketch is what the owner approved).
- `labelledColumns(items, {line, label, legend, noun}) -> html`: columns with the value written on top, `zero` mark for 0,
  dashed line at `line`, x labels from `top`/`bottom`, a "Show as table" `<details>`.

- [ ] **Step 1: Write the failing tests** in `test_research_insight_web.js` (import the module with
  `await import(pathToFileURL(...))`):
  - `one`: `one(2.666) === '2.7'`, `one(0) === '0'`, `one(null) === '–'`.
  - `fromProfileRow` on Chelsea's last real row (Sunderland 1–0 Chelsea, 8 Nov): `cards_for 4`, `cards_against 5`,
    `marketTotal(row,'cards') === 9`, `marketTotal(row,'corners') === 4`.
  - `hitCount skips nulls`: `hitCount([5, null, 2, 4], 3.5)` → `{hits: 2, of: 3}`; `hitCount([0,1,2], 1)` → `{hits: 2, of: 3}`.
  - Chelsea's real last 10 at cards 3.5: `hits 9, of 10` (the spec's lead sentence).
  - `hitSquares`: 3 `<li>`, classes `hit`, `none`, `miss`, the null one shows `–`.
  - `labelledColumns`: a 0 value renders class `zero` and the text `0`; every value is written in the markup.
- [ ] **Step 2: Run** `/usr/local/bin/node Scripts/tests/test_research_insight_web.js` — Expected: fails, module missing.
- [ ] **Step 3: Implement `insight.js`.**
- [ ] **Step 4: Run** the test — Expected: passes.
- [ ] **Step 5: Apply the number rules in `research.js`:** `fixed()` gives one decimal; the home-and-away rows write
  both values at the ends (sketch `fix-homeaway.jpg`) and drop "Hover or focus a row"; the xG note loses "green"
  ("A taller cobalt column than steel means…" becomes "When Chelsea's column is taller, they made the better
  chances."); `rankNotice` renders a grey `.quiet` chip line instead of the amber `.notice`; the three disclaimers
  (`card-note` past-matches lines in How often, `dataFoot`, `research-disclaimer`) become one foot line (Global
  Constraints) plus `dataFoot`'s data-basis sentence. `columnChart` writes each column's value on top.
- [ ] **Step 6: Run all research tests and look:** the insight test; then screenshots of Chelsea, Palmer and Taylor
  at 1440 and 390 on 8100 (`--no-reload` uvicorn of this worktree) — Expected: no "green", no amber box, values on
  every bar, home and away values written.
- [ ] **Step 7: Commit** (Done commit, tag `research-20261116a`).

### Task 2: Team page answer first

**Files:** Modify `research.js` (`renderProfile` for teams), `research.css`, `index.html`;
Create `Scripts/tests/test_research_team_web.js`.

**Interfaces:**
- Consumes: Task 1's `fromProfileRow`, `marketTotal`, `hitCount`, `hitSquares`, `labelledColumns`, `LINES`.
- Produces (`insight.js`): `teamLead(name, rows, market, line) -> html`, `teamLeadSub(name, rows, market) -> string`,
  `glance(name, rows, market) -> html` (the sketch's rows) and `stackedColumns(name, rows, market, line) -> html`;
  `research.js` state `state.teamMarket` (`cards|goals|corners`) and `state.lines` (per market), kept in
  `localStorage` key `spix-research-lines` (best effort).

- [ ] **Step 1: Failing tests:**
  - `teamLead('Chelsea', chelseaRows, 'cards', 3.5)` text: "Chelsea's matches are card-heavy: 9 of their last 10 had
    4 or more cards." with `<mark>9 of their last 10</mark>`.
  - `lead sentence uses the real count`: the first 6 rows give "… of their last 6 …".
  - Below 7 hits no "card-heavy": goals 2.5 on the real rows gives "N of their last 10 had 3 or more goals."
  - `stackedCards`: one column per row, total written on top, two segments, the legend holds "Line 3.5".
- [ ] **Step 2: Run** `/usr/local/bin/node Scripts/tests/test_research_team_web.js` — Expected: FAIL.
- [ ] **Step 3: Implement** `teamLead`, `stackedCards`; in `renderProfile('teams')` render, in order: head, lead,
  "Last 10 at a glance" (chips Cards/Goals/Corners, line picker, `hitSquares`), "Cards by match" (shown for Cards;
  Goals and Corners show `labelledColumns` of the match totals), then the old cards inside `<details>` sections
  (Against the league, Home and away, Squad leaders, Chances by match). Chips and lines re-render in place, focus kept.
- [ ] **Step 4: Run** the test — Expected: PASS. Screenshot Chelsea desktop and phone, compare with `desk-team.jpg`.
- [ ] **Step 5: Commit** (Done, tag `b`).

### Task 3: Stat view answer first (players, teams, referees)

**Files:** Modify `research.js` (`render`, `openChart`), `content.html`, `research.css`, `data.js` (per-stat lines),
`index.html`; Create `Scripts/tests/test_research_stat_web.js`.

**Interfaces:**
- Consumes: Task 1's `hitCount`, `labelledColumns`, `one`.
- Produces (`insight.js`): `statAnswer({noun, line, hits, of, unit, who}) -> html`, e.g. "1 or more shots on target
  in <mark>7 of his last 10</mark>." (`who` = his/her/their per subject; players use "their").
- Data (`data.js`): each metric gets `lines` (players' counts `[1,2,3]`; team and referee totals use `LINES`).

- [ ] **Step 1: Failing tests:** `statAnswer` for Palmer's real last 10 on target at 1+ gives "1 or more shots on
  target in 7 of their last 10." (count from the fixture, not hard-coded); for 2+ the count changes; with 3 nulls
  "in N of their last 7 with data".
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement:** the "Explore a threshold" toggle, stepper and season-compare checkbox are removed from
  `content.html`; a line picker (`#lineControl`, segmented buttons from the metric's `lines`) is always shown under
  the answer; the columns are `labelledColumns` with the dashed line; three context tiles replace the side panel's
  text for players (minutes a match and starts; the next opponent's allowance and the price come in Task 5 and 8,
  hidden until then). Per-90 stays under Filters and hides the line picker.
- [ ] **Step 4: Run** — Expected: PASS. Screenshot Palmer On target at 1440 and 390 against `fix-player*.jpg`.
- [ ] **Step 5: Commit** (Done, tag `c`).

### Task 4: Referee this season first, ranks from one season

**Files:** Modify `research.js` (`renderProfile('referees')`, `rankSource`, `rankCard`, `rankNotice`), `research.css`,
`index.html`; Create `Scripts/tests/test_research_ranks_web.js`.

**Interfaces:**
- Consumes: Task 1, Task 3's `statAnswer` and the line picker.
- Produces: `rankSource(d)` returns this season's metrics only; `ranking.provisional` (B5) adds the chip.

- [ ] **Step 1: Failing tests (real Palmer and Taylor fixtures):**
  - Palmer (888 minutes, unranked): the rank card shows 2026/27 values, no dots, the line "Ranked from 900 minutes ·
    888 so far" and a "Last season's ranks ›" link to scope `EPL-2025`; no 2025/26 value appears on the page.
  - Palmer with `ranking.provisional: true` and percentiles set: dots and the chip "Early season · 888 minutes so far".
  - Taylor: the lead is "<mark>N cards</mark> a match this season…" from the profile's own season, the sub-line holds
    the league average when present and last season from `previous_season`, and the line picker defaults to 3.5 with
    "Over 3.5 in N of M".
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS. Screenshots against `fix-referee.jpg`, `fix-ranks.jpg`.
- [ ] **Step 5: Commit** (Done, tag `d`).

### Task 5: Next matches: rail, floating bar and sheet (needs B1)

**Files:** Create `Scripts/web_app/static/research/upcoming.js`, `Scripts/tests/test_research_upcoming_web.js`,
`Scripts/tests/fixtures/research_upcoming_chelsea.json` (B1 shape; Chelsea's real next three fixtures, referees and
read states as the API will give them); Modify `research.js`, `content.html` (`#nextRail`, `#nextBar`, `#sheet`),
`research.css`, `Scripts/web_app/static/app.js` (sets `--slip-bar-h` on `spix-research`), `index.html`.

**Interfaces:**
- Consumes: B1 `research-upcoming.v1`.
- Produces (`upcoming.js`): `readState(read, now) -> string` ("Read ready", "Preliminary", "Read due Thu");
  `nextCard(match, {teamId, active}) -> html` with `data-open-match="{event_id}"`; `railHTML(upcoming, {active})`;
  `barHTML(upcoming)`; `sheetHTML(upcoming)`; `headNext(match, label) -> html` (the player and referee head card).
  `research.js`: `loadUpcoming(kind, id)` → payload or null (404 and errors give null, never throw).

- [ ] **Step 1: Failing tests:**
  - `readState`: ready → "Read ready"; preliminary → "Preliminary"; waiting with `opens_at` Thu 19 Nov → "Read due Thu".
  - `nextCard` for Chelsea v Leeds: "v Leeds", "Sat 21 Nov · 15:00", "Referee Anthony Taylor", figures 2.7 / 10.1 / 4.6;
    away match reads "at Napoli"; no referee → "Referee not named yet"; no read → no figures block.
  - `barHTML`: "Next: v Leeds", "Sat 21 Nov · 15:00 · and 2 more", "Open ›"; with one match no "and N more".
  - `no upcoming means no rail and no bar`: `railHTML(null) === ''`, `barHTML({matches: []}) === ''`.
  - `bar offset follows the slip bar`: the CSS rule for `.next-bar` uses `var(--slip-bar-h, 0px)`; `app.js` source
    sets `--slip-bar-h` where it toggles `.slip-mobile-bar.hidden`, and hides the bar while the slip is open
    (`data-slip-open` on the host).
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement:** team pages load B1 after the profile; ≥1100 px a grid with the 340 px sticky rail;
  <1100 px the fixed bar and the sheet (`role="dialog"`, `aria-modal`, focus moves in and returns to the bar, Escape,
  backdrop and a downward swipe of 60 px close it; body scroll locked while open). Player pages show `headNext` for
  their club's next match; referee pages for the next appointment (`/referees/{key}/upcoming?limit=1`).
- [ ] **Step 4: Run** — Expected: PASS. Screenshots with the fixture served through CDP request interception:
  desktop team page, phone bar with the slip bar shown and hidden, the open sheet; compare with `phone-bar*.jpg`.
- [ ] **Step 5: Commit** (Done, tag `e`).

### Task 6: Match view shell (needs B2)

**Files:** Create `Scripts/web_app/static/research/match.js`, `Scripts/tests/test_research_match_web.js`,
`Scripts/tests/fixtures/research_match_chelsea_leeds.json` (B2 shape: Chelsea's and Leeds's real last 10, Taylor's
real last 10, real meetings; the read's figures copied from a real saved EPL read and labelled so in the file's
`_note`); Modify `research.js` (route `matches/{event_id}/{market}`), `content.html` (`#matchView`), `research.css`,
`index.html`.

**Interfaces:**
- Consumes: B2 `research-match.v1`; Task 1 helpers; Task 5 `railHTML` (open match marked).
- Produces (`match.js`): `MARKETS = ['cards','sot','goals','corners','result','h2h','news']` with labels;
  `matchHead(match)`, `spixReads(read, market) -> html` (one `<mark>` on the open market's figure);
  `marketsRow(active) -> html`; `renderMatch(match, view) -> html` where
  `view = {market, lines: {cards, goals, corners, sot}, sotTeam: 'both'|'home'|'away'}`; a `SECTIONS` map
  `market -> (match, view) => html` that Tasks 7–9 fill (unfilled markets render "Coming soon" only in tests;
  their buttons are hidden until filled).

- [ ] **Step 1: Failing tests:**
  - head: "Chelsea v Leeds", "Premier League · Sat 21 Nov · 15:00", "Referee Anthony Taylor".
  - `spixReads`: "58%", "24%", "18%", "2.7", "10.1", "4.6", "49%"; with market `cards` the `<mark>` wraps 4.6 only.
  - `read waiting`: `read: null` with fixture kickoff gives "Spix's read comes out Thu 19 Nov" and no figures.
  - `marketsRow('cards')`: `aria-current` on Cards; each button `data-market`.
  - `match view error state`: research.js route on a 404 shows "This match could not load" with a back link.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the shell; the markets row is sticky (`top: var(--research-sticky-top, 64px)`), scrolls
  sideways on the phone; the foot line; the route remembers the last market per browser.
- [ ] **Step 4: Run** — Expected: PASS. Screenshot desktop and phone.
- [ ] **Step 5: Commit** (Done, tag `f`).

### Task 7: Cards section and Add to slip

**Files:** Modify `match.js`, `research.css`, `app.js` (`research-add-leg` listener calling the existing
`slip.add(...)` path used by the match page's Explore options), `index.html`; extend `test_research_match_web.js`.

**Interfaces:**
- Consumes: Task 6's `SECTIONS`, Task 1's `hitSquares`, `hitCount`.
- Produces: `cardsSection(match, view) -> html`; event `research-add-leg` with
  `detail = {fixture: {event_id, home, away, kickoff, league}, leg: <recommendations.v1 option>, market}`.

- [ ] **Step 1: Failing tests:**
  - line picker 2.5–5.5 with 3.5 pressed; Over 3.5 and Under 3.5 cards with the read's chance (whole %) and typical
    price; no Add on cards (not a slip market).
  - four hit rows: "Chelsea, last 10", "Leeds, last 10", "Anthony Taylor, last 10", "Meetings" each with "N of M over";
    counts computed from the fixture's rows; no referee → three rows and the "Referee not named yet" line.
  - "Who gets the cards": each team's own cards a match and their opponents', one decimal.
  - referee block: cards a match this season against the league average, matches refereed.
  - booked players: up to 5 per side, "4 yellow · 1 red in 11".
  - `app.js`: a `research-add-leg` listener exists and calls the same slip path as the match page.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS. Screenshots against `phone-match-cards.jpg` and `desk-match.jpg`.
- [ ] **Step 5: Commit** (Done, tag `g`).

### Task 8: Shots on target section

**Files:** Modify `match.js`, `research.css`, `research.js` (Task 3's player price and allowance tiles now filled
from B1/B2 when the player's next match is open), `index.html`; extend `test_research_match_web.js`.

**Interfaces:** Produces `sotSection(match, view) -> html`.

- [ ] **Step 1: Failing tests:**
  - 1+/2+ and Both/Chelsea/Leeds toggles; a table row per shooter: name, a strip of 10 squares, "7 of 10", minutes a
    match (one decimal off: whole minutes), price when present.
  - `no price column without prices`: all `price_sot_*` null → no price header, no "–".
  - "What Leeds allow": shots on target conceded a match (one decimal) and the rank ("9th fewest of 20").
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS. Screenshot against `phone-match-sot.jpg`.
- [ ] **Step 5: Commit** (Done, tag `h`).

### Task 9: Goals, Corners, Result, Head to head, Team news

**Files:** Modify `match.js`, `research.css`, `index.html`; extend `test_research_match_web.js`.

**Interfaces:** Produces `goalsSection`, `cornersSection`, `resultSection`, `h2hSection`, `newsSection`, each
`(match, view) -> html`; Add buttons for goals, corners and result options raise `research-add-leg`.

- [ ] **Step 1: Failing tests:** goals at 2.5: Over/Under chance and price, Add on both, hit rows for both teams and
  meetings; corners at 9.5 likewise; result: three chances with prices and each side's last 10 as W/D/L squares;
  head to head: one row per meeting with score, cards, corners, goals; team news: `lineups: null` gives likely
  starters from `shooters` with "started N of last 5" and "Line-ups are confirmed about an hour before kickoff".
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — Expected: PASS. Screenshots of each section on the phone.
- [ ] **Step 5: Commit** (Done, tag `i`).

### Task 10: Finish: real data, look, review

**Files:** `docs/rework-tracker.md`, this plan, screenshots in the scratchpad.

- [ ] **Step 1:** Once B1 and B2 are merged into main and merged here: run every research view against the real
  endpoints on 8100 (Chelsea, Leeds, Palmer, Taylor, a referee with 2026/27 data, a promoted team, a European match),
  1440 and 390, light and dark. Fix what the real data breaks (`Fix research task N: …`).
- [ ] **Step 2:** All JS tests: `for f in Scripts/tests/test_*_web.js; do /usr/local/bin/node "$f" || echo FAIL $f; done`
  — Expected: no FAIL. Python suite: `.venv/bin/python -m pytest Scripts/tests -q` — Expected: unchanged from main.
- [ ] **Step 3:** Whole-branch review (self-review against `code-reviewer.md`; no subagents unless the owner asks).
- [ ] **Step 4:** Tracker D107 summary line, plan rows Done; tell the orchestrator what changed, the tests and that
  the website needs no restart (static files only) once B1/B2 are live.

## Progress

| Task | Status | Commits | Tests |
|---|---|---|---|
| 1. Lines, hits and clearer numbers | Done | cf8a499.. | insight 1/1 file, all asserts |
| 2. Team page answer first | Done | 6d163c8.. | team + insight, all *_web.js pass |
| 3. Stat view answer first | Not started | | |
| 4. Referee this season first, ranks from one season | Not started | | |
| 5. Next matches: rail, bar, sheet (B1) | Not started | | |
| 6. Match view shell (B2) | Not started | | |
| 7. Cards section and Add | Not started | | |
| 8. Shots on target section | Not started | | |
| 9. Goals, Corners, Result, Head to head, Team news | Not started | | |
| 10. Finish: real data, look, review | Not started | | |

Backend items (spec 6.2), asked 16 Nov: B1 next matches, B2 the match, B3 checks (Chelsea's ranks, Taylor's
2026/27 matches), B4 all-competitions scope (later), B5 early-season ranks.
