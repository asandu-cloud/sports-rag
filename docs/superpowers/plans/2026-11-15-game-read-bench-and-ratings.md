# Game-read engine, phases 0–1: the test bench and four rating models — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A walk-forward test bench that scores any model against the market and against today's Most likely and
Value rules on 14 seasons of football-data history, and the first four rating models (Elo, pi-ratings,
Dixon–Coles with time decay, drifting ratings) plus their average, scored on it.

**Architecture:** A new research package `Scripts/data_platform/game_read/` reads `Index/football_data.sqlite3`
read-only, rebuilds each match's Most likely pool from the market's prices (M1 for lines with no history), runs
each model day by day (predict a day's matches, then learn their results), and scores the forecasts.
`Scripts/ops/game_read_bench.py` runs it and writes reports to `Research/game-read-bench-<date>/`. Nothing in the
live site imports the package; no engine switch, no published pick changes.

**Tech Stack:** Python 3.13.3 (the shared `.venv`), numpy 2.3.3, scipy 1.17.0, scikit-learn 1.8.0 (all installed;
no new dependency), sqlite3, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-game-read-engine-design.md` (approved 15 Nov, tracker D77).

**Start condition:** the owner lifted the code freeze of 14 Nov on 15 Nov and asked for Task 1. Task 14 also needs
the owner's yes after its terms check.

## Progress

Each task has a start commit and a done commit (CLAUDE.md, "Large builds"). To undo a task:
`git revert <its done commit>`.

| Task | What it builds | Status | Start commit | Done commit | Tests |
|---|---|---|---|---|---|
| 1 | History loader | Done 15 Nov | cfc881e | eed3af6 | 11 new, all pass; suite 3,079 passed + the 3 known errors |
| 2 | Margins of our saved reads | Done 15 Nov | cce0817 | c450daa | 5 new, all pass; suite 3,084 passed + the 3 known errors |
| 3 | The rebuilt Most likely pool | Done 16 Nov | 14ba2c2 | 89e0ad2 | 7 new, all pass; suite 3,139 passed + the 3 known errors |
| 4 | Today's rules and the engine's rules | Done 16 Nov | 02ab071 | d87493a | 12 new, all pass; suite 3,151 passed + the 3 known errors |
| 5 | Forecasts, walk-forward runner, zero and control lines | Done 16 Nov | fb94888 | a2a93ed | 8 new, all pass; suite 3,159 passed + the 3 known errors |
| 6 | The scores | Done 16 Nov | c6966dd | 9fbec27 | 10 new + 1 in task 4's, all pass; suite 3,169 passed + the 3 known errors |
| 7 | Bench command, report, Opportunities, locked-run guard | Done 16 Nov | 7c94626 | 0e9fa53 | 6 new, all pass; suite 3,175 passed + the 3 known errors; first tuning run 16 Nov |
| 8 | Elo (model 1) | Done 16 Nov | 5647819 | e3e3266 | 8 new, all pass; suite 3,183 passed + the 3 known errors; k 15, carry 0.9 |
| 9 | Pi-ratings (model 2) | Done 16 Nov | 6cf4cd5 | b004822 | 4 new, all pass; suite 3,187 passed + the 3 known errors; learning 0.05, cross 0.5 |
| 10 | Dixon–Coles with time decay (model 3) | Done 16 Nov | bf9408c | 7471f94 | 5 new, all pass; suite 3,192 passed + the 3 known errors; half-life 365 days |
| 11 | Drifting ratings (model 4) | Done 16 Nov | 3b8eff7 | 31c1810 | 3 new, all pass; suite 3,195 passed + the 3 known errors; gain 3e-6, summer widening 0.01 |
| 12 | The average of the models | Done 16 Nov | f62eb22 | the "Done game-read task 12" commit | 4 new, all pass; suite 3,199 passed + the 3 known errors |
| 13 | Phase 1 runs and the first report to the owner | Not started | | | |
| 14 | ClubElo yardstick (only with the owner's yes) | Not started | | | |

## Global Constraints

- Branch `backend`; small commits, files staged by name, no attribution lines; each task's tracker line in the same
  commit as its code (`docs/rework-tracker.md`, under D77 or a new number from the orchestrator).
- Never stage `Output/`, `Index/`, `Research/`, `.impeccable/`, `.superpowers/`, `.claude/`, `.taste-skill/`.
- `Index/football_data.sqlite3` is opened read-only: `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.
- `Index/platform.db` is read only in Task 2, only inside `data_access()`
  (`Scripts.data_platform.services.refresh_coordination`), never while `pgrep -f refresh-season` finds the refresh,
  in short read-only statements, run with `PYTHONPATH=.` from the repo root.
- No live code path changes: `python -m Scripts.ops.golden_snapshots check` stays unchanged; no engine switch.
- Seasons (start year): warm-up 2012–2018, tuning 2019–2022, locked 2023–2025. Every setting is chosen on tuning.
- Market chance: power method (`Scripts.data_platform.devig.power`) on football-data's average price
  (`Market average` from 2019/20, `BetBrain average` before).
- Most likely floors from `Scripts.data_platform.recommendations`: `MIN_CHANCE` 0.50, `MIN_PRICE` 1.30,
  `MAX_MOST_LIKELY` 3; Value: at least 2% value (`chance × price − 1 ≥ 0.02`), `MAX_VALUE` 2 per match.
- Half lines only in the pools and in the handicap scores; quarter and whole main lines still feed M1.
- Intervals: 95%, resampling whole UTC weeks (`features.benchmarks.evaluation.calendar_block`), 2,000 resamples,
  fixed seed.
- Bench output goes to `Research/game-read-bench-<YYYY-MM-DD>/` (not committed); every locked run is logged in the
  ledger file and the tracker.
- Downloads (Task 14 only) go in their own new folder; Python reading them runs with `-I`.
- Tests: `.venv/bin/python -m pytest Scripts/tests` (switches pinned by `Scripts/tests/conftest.py`).

## Review Focus

1. **A match with one market's prices missing** (no average handicap, say): it must still score in the other
   markets, and a missing price must never become a chance of 0 or 1. Test in Task 1.
2. **A team with no history** (promoted from a league football-data lacks, or its first match): every model returns
   a finite forecast from its prior, never NaN or an error. Tests in Tasks 8–11.
3. **Several matches on one day, a team twice in a window, or postponed games:** all of a day's matches are
   predicted before any of that day's results is learned. Test in Task 5.
4. **Quarter and whole main handicaps:** left out of the half-line pool, the handicap scores and Value, but still
   used to solve M1. Tests in Tasks 3 and 4.
5. **A closing handicap on another line than the pre-kickoff one:** the pick's closing chance comes from M1 solved
   on the closing prices, not from a different line's price. Test in Task 6.

---

## Phase 0: the test bench

### Task 1: History loader

**Files:**
- Create: `Scripts/data_platform/game_read/__init__.py` (empty), `Scripts/data_platform/game_read/history.py`
- Test: `Scripts/tests/data_platform/test_game_read_history.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Match`: `id: int`, `competition: str`, `country: str`, `season: int`,
    `day: datetime.date`, `home: str`, `away: str` (team keys `"<country>:<football-data name>"`, so a team keeps
    its key across the Premier League and Championship), `home_goals: int`, `away_goals: int`, `pre: Prices`,
    `close: Prices`.
  - `@dataclass(frozen=True) class Prices`: `result_avg`, `result_max`: `tuple[float, float, float] | None`;
    `handicap_line: float | None` (home side's line), `handicap_avg`, `handicap_max`: `tuple[float, float] | None`;
    `over25_avg`, `over25_max`: `tuple[float, float] | None` (over, under).
  - `load_matches(path: Path = OUT_DB, *, competitions: Iterable[str] | None = None) -> list[Match]`, ordered by
    day then id; matches without a final score are left out.
  - `block(season: int) -> Literal["warmup", "tuning", "locked"]`.
  - `market_result(prices: Prices) -> tuple[float, float, float] | None`,
    `market_handicap(prices: Prices) -> tuple[float, float] | None`,
    `market_over25(prices: Prices) -> tuple[float, float] | None`: power-method fair chances of the average
    prices, None when a price is missing.
  - `named_result_prices(path: Path = OUT_DB) -> dict[int, list[tuple[float, float, float]]]`: each match's
    pre-kickoff result prices from every named bookmaker (not the averages or maximums), leaving out rows flagged
    `suspect`. Used only for the check in spec point 3.
- `OUT_DB` and `DIVISIONS` come from `Scripts.ops.football_data_import`.

- [x] **Step 1: Write the failing tests.** Build a tiny database in `tmp_path` with the importer's `SCHEMA` and a
  few rows (one 2018 match with `BetBrain average` prices, one 2020 match with `Market average` and
  `Market maximum`, one 2020 match without handicap prices, one Championship and one Premier League match of the
  same team, one match with no score).
  - `test_load_matches_reads_the_average_and_best_prices_before_kickoff_and_at_the_close`: the 2020 match's
    `pre.result_avg`, `pre.result_max`, `close.result_avg` equal the inserted prices.
  - `test_before_2019_the_average_is_betbrains`: the 2018 match's `pre.result_avg` is the `BetBrain average` row.
  - `test_a_missing_market_stays_missing_and_the_others_load` (Review Focus 1): `pre.handicap_avg is None` and
    `market_handicap(pre) is None`, while `market_result(pre)` sums to 1.
  - `test_a_team_keeps_its_key_across_divisions`: both matches give the same `home` key `"England:<name>"`.
  - `test_matches_without_a_score_are_left_out` and `test_block_splits_the_seasons`:
    `block(2018) == "warmup"`, `block(2019) == "tuning"`, `block(2022) == "tuning"`, `block(2023) == "locked"`.
  - `test_the_database_is_opened_read_only`: `load_matches` on the test file leaves its modification time
    unchanged, and a write through the module's connection raises `sqlite3.OperationalError`.
  - `test_named_prices_leave_out_averages_and_suspect_rows`.
- [x] **Step 2: Run them.** `.venv/bin/python -m pytest Scripts/tests/data_platform/test_game_read_history.py -v`
  Expected: FAIL, `ModuleNotFoundError: Scripts.data_platform.game_read.history`.
- [x] **Step 3: Implement `history.py`.** One read-only connection; one query for matches, one for the six price
  sources (`Market average`, `Market maximum`, `BetBrain average`, `BetBrain maximum`, both phases), pivoted in
  Python. Before 2019/20 the best price is `BetBrain maximum`.
- [x] **Step 4: Run them.** Expected: PASS.
- [x] **Step 5: Commit** `history.py`, `__init__.py`, the test, and a tracker line: "D77 build, task 1: history
  loader".

### Task 2: Margins of our saved reads

**Files:**
- Create: `Scripts/data_platform/game_read/margins.py`, `Scripts/data_platform/game_read/margins.json`
- Test: `Scripts/tests/data_platform/test_game_read_margins.py`

**Interfaces:**
- Consumes: `recommendations.options(canonical_results)` (live code: every option with `market`, `side`, `line`,
  `typical_price`).
- Produces:
  - `booksums(options_by_market: Mapping[str, list[dict]]) -> dict[tuple[str, float | None], float]`: for each
    `goals` line, `spreads` home line and `btts`, the sum of `1 / typical_price` over its two sides.
  - `measure(reads: Iterable[Iterable[Mapping]]) -> dict`: the median booksum per key over every read, with `n`.
  - `load() -> dict[tuple[str, float | None], float]` from `margins.json`;
    `booksum(table, market, line) -> float`: the key's median, else the median of the same market's lines, else
    1.05.
  - `export(days, *, now, gate=None) -> dict`: one `data_access()` hold per day, reads shown at kickoff through
    the board's `_shown_at_kickoff`, as `Scripts/ops/late_goal_study.load_read_expectations` does.

- [x] **Step 1: Write the failing tests.**
  - `test_booksum_adds_the_two_sides_of_each_line`: options Over 2.5 at 1.90 and Under 2.5 at 1.95 give
    `booksums(...)[("goals", 2.5)] == pytest.approx(1/1.90 + 1/1.95)`; a handicap pair is keyed by the home line.
  - `test_measure_takes_the_median_per_line_and_counts_reads`.
  - `test_booksum_falls_back_to_the_market_then_to_105`.
- [x] **Step 2: Run them.** Expected: FAIL, module missing.
- [x] **Step 3: Implement** the four functions and a command,
  `python -m Scripts.data_platform.game_read.margins --since 2026-09-05`, that writes `margins.json` with
  `measured_on`, `days` and `n` per key.
- [x] **Step 4: Run them.** Expected: PASS.
- [x] **Step 5: Measure on the real reads.** Check `pgrep -f refresh-season` finds nothing, then run the export
  from the repo root with `PYTHONPATH=.`. Expected: `margins.json` with goals 0.5–4.5, spreads half lines and btts,
  each `n` > 100. Read the table: booksums between 1.02 and 1.15.
- [x] **Step 6: Commit** `margins.py`, `margins.json`, the test, and a tracker line with the table's range.

### Task 3: The rebuilt Most likely pool

**Files:**
- Create: `Scripts/data_platform/game_read/pool.py`
- Test: `Scripts/tests/data_platform/test_game_read_pool.py`

**Interfaces:**
- Consumes: `history.Match`, `history.market_*`, `margins.load/booksum`, `market_goals.solve`,
  `market_goals.handicap_probability`, `market_goals.over_probability`, `market_goals._matrix`.
- Produces:
  - `inverse_power(fair: Sequence[float], booksum: float) -> list[float]`: prices whose implied chances
    `fair_i ** k` sum to `booksum` (k found by bisection, 0 < k ≤ 1), so `devig.power(prices) == fair`.
  - `m1_means(prices: Prices) -> tuple[float, float] | None`: `market_goals.solve` on the market's result, over 2.5
    and main-handicap chances (any main line, quarter and whole included).
  - `build_pool(match: Match, *, phase: Literal["pre", "close"] = "pre", margins=None) -> list[dict]`: options in
    the live shape (`market` in `moneyline`, `goals`, `btts`, `spreads`; `side`; `line`; `chance`;
    `typical_price`), plus `real: bool`. Real options: the result, over/under 2.5 and a half-line main handicap,
    with their market chance and average price. Derived options: goals 0.5–4.5 except 2.5, spreads at the half
    lines −2.5…+2.5 other than the real one, btts; chance from M1's means, price from `inverse_power` and the
    line's booksum. Empty when M1 cannot be solved.
  - `option_key(option) -> tuple[str, str, float | None]`.

- [ ] **Step 1: Write the failing tests.**
  - `test_inverse_power_undoes_the_power_method`: for `fair = [0.62, 0.38]` and booksum 1.06,
    `devig.power(inverse_power(fair, 1.06))` equals `fair` to 1e-9, and the implied chances sum to 1.06.
  - `test_the_pool_has_every_half_line_the_live_site_shows`: the keys cover goals 0.5–4.5, spreads −2.5…+2.5 (half
    lines, both sides), btts yes/no and moneyline home/draw/away; no corners.
  - `test_real_lines_keep_their_real_chance_and_price`: over 2.5's chance is `market_over25(pre)[0]` and its
    `typical_price` the average price; `real` is True only on the real lines.
  - `test_a_quarter_main_handicap_feeds_m1_but_stays_out_of_the_pool` (Review Focus 4): with main line −0.75, no
    option has line ±0.75 and `m1_means` uses it (its means differ from the solve without it).
  - `test_derived_chances_agree_with_m1`: Over 1.5's chance equals `over_probability(*means, 1.5)`.
- [ ] **Step 2: Run them.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement `pool.py`.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 4: Today's rules and the engine's rules

**Files:**
- Create: `Scripts/data_platform/game_read/rules.py`
- Test: `Scripts/tests/data_platform/test_game_read_rules.py`

**Interfaces:**
- Consumes: `recommendations.suggestion`, `most_likely`, `can_win_together`, `same_bet`, `MIN_CHANCE`,
  `MIN_PRICE`, `MAX_MOST_LIKELY`, `MAX_VALUE`; `pool.build_pool`, `pool.option_key`.
- Produces:
  - `todays_most_likely(pool: list[dict]) -> list[dict]`: `most_likely(suggestion(options of each market))`, the
    live functions unchanged.
  - `engine_most_likely(pool, engine: Mapping[key, float], *, strength: Callable[[dict, float], float]) ->
    list[dict]`. An option is backed when `engine[key] > option["chance"]`. Within each market, among eligible
    options (floors as live): backed first, highest `strength` first, then the rest by the live `_rank`. Across
    markets: backed suggestions first by `strength`, then the rest by `_rank`; up to 3 that can win together and
    are not the same bet. Options the engine cannot score count as not backed.
  - `by_edge(option, engine_chance) -> float` = `engine_chance − option["chance"]`;
    `by_blend(model) -> Callable`: the blended chance from a model fitted by `Scripts.data_platform.blend.fit`.
  - `value_candidates(match, *, price: Literal["avg", "max"]) -> list[dict]`: the result's three outcomes, over and
    under 2.5, and a half-line main handicap's two sides, each with its market chance and that price.
  - `todays_value(candidates) -> list[dict]` and `engine_value(candidates, engine) -> list[dict]`: value
    `chance × price − 1 ≥ 0.02` (market chance, or the engine's), at most 2 per match, highest value first, able to
    win together.

- [ ] **Step 1: Write the failing tests.**
  - `test_todays_rule_is_the_live_rule`: on a hand-made pool, `todays_most_likely` equals
    `recommendations.most_likely([recommendations.suggestion(...) per market])`.
  - `test_with_the_market_as_engine_the_lists_are_todays`: `engine = {key: chance}` gives today's list exactly.
  - `test_a_backed_option_beats_a_likelier_one_in_its_market`: Over 1.5 (0.78, 1.30) unbacked and Under 3.5 (0.71,
    1.38) backed: the goals suggestion is Under 3.5.
  - `test_the_list_never_holds_more_than_three_or_options_that_cannot_win_together`.
  - `test_unscored_options_count_as_not_backed`.
  - `test_the_market_makes_no_value_pick_at_the_average_price`: for fair chances from the power method,
    `todays_value(value_candidates(m, price="avg")) == []` on a range of prices (power fair chance × price < 1).
  - `test_value_needs_two_percent_and_keeps_two_a_match`.
  - `test_quarter_main_handicaps_give_no_value_candidates` (Review Focus 4).
- [ ] **Step 2: Run them.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement `rules.py`.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 5: Forecasts, the walk-forward runner, and the zero and control lines

**Files:**
- Create: `Scripts/data_platform/game_read/forecast.py`, `Scripts/data_platform/game_read/walkforward.py`,
  `Scripts/data_platform/game_read/models/__init__.py`, `Scripts/data_platform/game_read/models/market.py`
- Test: `Scripts/tests/data_platform/test_game_read_walkforward.py`

**Interfaces:**
- Produces:
  - `@dataclass class Forecast`: `result: tuple[float, float, float] | None`,
    `goal_difference: Mapping[int, float] | None` (P(home − away = d)), `score_matrix: np.ndarray | None`
    (13×13, sums to 1).
  - `chance(forecast, option) -> float | None`: moneyline from `result`; spreads half lines from `goal_difference`
    or the matrix; goals and btts from the matrix only (None otherwise).
  - `class Model(Protocol)`: `name: str`; `predict(matches: Sequence[Match]) -> list[Forecast]` (one day, before
    its results); `learn(matches: Sequence[Match]) -> None` (the same day, after).
  - `run(model, matches, *, score_from: int = 2019, until: int | None = None) -> list[tuple[Match, Forecast]]`:
    days in order; before `score_from` only `learn`; from it, `predict` the whole day, then `learn` the whole day.
  - `models.market.MarketCopy` (zero line): the market's own result chances and the M1 matrix.
  - `models.market.MarketControl` (control line): M1's means with the Dixon–Coles low-score correction, its rho
    fitted by maximum likelihood on every earlier match's M1 means and score, refitted at each season start.

- [ ] **Step 1: Write the failing tests.**
  - `test_no_forecast_sees_its_own_day` (Review Focus 3): a spy model records the latest day learned at each
    `predict`; over a history with several matches a day and a team playing twice in three days, every recorded
    day is earlier than the predicted day, and each day is predicted whole before any of it is learned.
  - `test_warm_up_seasons_are_learned_not_scored`.
  - `test_the_market_copy_gives_the_markets_chances`: on a pool, `chance(MarketCopy forecast, option)` equals
    each option's `chance` (real lines exactly; derived lines to 1e-9).
  - `test_chance_reads_handicaps_from_a_goal_difference_and_nothing_else_without_a_matrix`.
  - `test_the_control_line_moves_low_scores_only`: with rho < 0, the control's 0–0 and 1–1 chances rise above M1's
    and the matrix still sums to 1.
- [ ] **Step 2: Run them.** Expected: FAIL, modules missing.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 6: The scores

**Files:**
- Create: `Scripts/data_platform/game_read/scores.py`
- Test: `Scripts/tests/data_platform/test_game_read_scores.py`

**Interfaces:**
- Consumes: `walkforward.run` output, `rules.*`, `pool.build_pool`, `blend.fit/predict`,
  `evaluation.calendar_block`.
- Produces (each returns a dict with the estimate, `low`, `high`, `n`, and `by_league`):
  - `week_interval(rows, statistic, *, resamples=2000, seed=20261115) -> tuple[float, float]`.
  - `accuracy(rows, market) -> dict`: mean log loss of the model and the market and their difference, for
    `result`, `handicap` (half-line main lines) and `over25`.
  - `blend_score(tuning_rows, scored_rows, market) -> dict`: `blend.fit` with and without the model on tuning
    forecasts; on the scored rows, `c`, and the log-loss difference blend minus market-only recalibration.
  - `picks_score(rows, *, strength) -> dict`: for today's list and the engine's, picks, wins, expected wins (sum of
    chances), wins minus expected per 100 picks, raw win rate; the paired difference engine minus today; the same
    on real lines only and on derived lines only.
  - `value_score(rows, *, price) -> dict`: for today's Value rule and the engine's, picks, return per unit, and
    closing line value `price × closing fair chance − 1`.
  - `closing_check(rows, *, strength) -> dict`: mean move of the fair chance from pick time to the close, engine's
    Most likely picks minus today's.
  - `closing_chance(match, option) -> float | None`: the closing fair chance of the option's own line; when the
    closing main handicap sits on another line, from M1 solved on the closing prices.
  - `reference_check(matches, named) -> dict`: per season, the mean and 95th percentile of the absolute gap between
    the average-price result chance and the median of the named books' power-method chances (spec point 3).
  - `verdicts(results) -> dict`: pass or fail per model and score, with the spec's rules:
    - blend: c > 0 and the interval below zero;
    - Most likely: the interval above zero, beating the control line on derived lines, and no league worse by
      more than 2 wins per 100 picks;
    - Value: at the average price, closing line value with its interval above zero, a return of at least zero, and
      at least 300 picks.

- [ ] **Step 1: Write the failing tests.**
  - `test_week_interval_resamples_whole_weeks`: rows of two weeks, one all ones, one all zeros: every resample's
    mean is 0, 0.5 or 1.
  - `test_the_zero_line_scores_zero`: with `MarketCopy` rows, `picks_score` gives a paired difference of exactly 0,
    `blend_score`'s log-loss difference is within 1e-6 of 0, and `value_score(price="avg")` has 0 picks for the
    engine.
  - `test_the_market_copy_at_the_best_price_is_todays_value_rule`.
  - `test_a_known_win_counts_once_against_its_chance`: a one-pick list that won at chance 0.6 gives +40 per 100.
  - `test_clv_is_price_times_closing_fair_chance_minus_one`.
  - `test_a_moved_closing_handicap_is_priced_from_m1` (Review Focus 5): pre line −0.5, close line −1.0: the closing
    chance of "home −0.5" is `handicap_probability(*m1_means(match.close), -0.5)`.
  - `test_verdicts_apply_the_spec_rules`: a Value result with 299 picks fails however good it is, and a Most likely
    result with one league at −2.5 per 100 fails.
- [ ] **Step 2: Run them.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement `scores.py`.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 7: The bench command, the report, Opportunities and the locked-run guard

**Files:**
- Create: `Scripts/ops/game_read_bench.py`, `Scripts/data_platform/game_read/report.py`,
  `Scripts/data_platform/game_read/opportunities.py`
- Test: `Scripts/tests/test_game_read_bench.py`

**Interfaces:**
- Produces:
  - CLI: `python -m Scripts.ops.game_read_bench run --models market,control[,elo,pi,dc,drift,average]
    --block tuning|locked [--confirm-locked] [--leagues ...] --out Research/game-read-bench-<date>`.
  - `MODELS: dict[str, Callable[[], Model]]` (Tasks 8–12 add to it).
  - Every run first scores `market` and `control`; if the zero line is not zero (Task 6's tolerances), the run
    stops with exit code 2 and writes nothing else.
  - `--block locked` without `--confirm-locked` exits 2. With it, the run also refuses when
    `git status --porcelain Scripts/data_platform/game_read Scripts/ops/game_read_bench.py` is not empty, and
    appends one line to `Research/game-read-bench/locked-runs.jsonl`: time, models, `git rev-parse HEAD`, and the
    headline scores.
  - `report.write(out_dir, results) -> None`: `summary.json` and `report.md`. Per model and market:
    - accuracy and blend;
    - picks under both ranking rules;
    - Value at both prices;
    - the closing check and the leagues;
    - the verdicts (`scores.verdicts`);
    - the zero and control lines and the reference check;
    - Opportunities.
  - `opportunities.scan(matches) -> dict`, tuning seasons only (it refuses locked matches):
    - the drift from pick time to the close by league, weekday and market;
    - the market's calibration by price band (favourites and longshots);
    - promoted teams' first 10 matches against the market;
    - each with its interval and match count.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_locked_run_must_be_named`, `test_a_locked_run_refuses_uncommitted_bench_code`,
    `test_a_locked_run_is_written_to_the_ledger`.
  - `test_a_broken_zero_line_stops_the_run`: a stub `market` model that adds 0.01 to every chance makes `run` exit 2
    and write no report.
  - `test_opportunities_never_look_at_locked_seasons`.
  - `test_the_report_has_every_section`.
- [ ] **Step 2: Run them.** Expected: FAIL, module missing.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: First real run, tuning only.**
  `PYTHONPATH=. .venv/bin/python -m Scripts.ops.game_read_bench run --models market,control --block tuning --out
  Research/game-read-bench-<date>`. Expected:
  - the zero line passes;
  - about 14,400 tuning matches are scored;
  - the control line's picks score on derived lines is reported;
  - the result market's average-price chance sits close to the named books' median (the check of spec point 3).
- [ ] **Step 6: Commit** the code and a tracker line with the run's headline: the control line's derived-line
  score, today's Most likely and Value records on tuning.
- [ ] **Step 7: Full suite and golden check.** `.venv/bin/python -m pytest Scripts/tests` (expected: all pass
  but the 3 known errors) and the golden check with the switches written out inline. Expected: unchanged.

---

## Phase 1: four rating models and their average

### Task 8: Elo (model 1)

**Files:**
- Create: `Scripts/data_platform/game_read/ordered.py`, `Scripts/data_platform/game_read/models/elo.py`
- Test: `Scripts/tests/data_platform/test_game_read_elo.py`

**Interfaces:**
- Produces:
  - `ordered.fit(x: np.ndarray, y: np.ndarray, classes: int) -> OrderedLogit` and
    `OrderedLogit.probabilities(x) -> np.ndarray`, used for result (3 classes) and the clipped goal difference
    −4…+4 (9 classes).
  - `models.elo.Elo(k=..., home_step=..., carry=..., new_team_offset=...)` implementing `Model`, with:
    - ratings starting at 1500;
    - a goal-difference multiplier: 1 for a margin of 1, 1.5 for 2, (11 + margin) / 8 from 3;
    - home advantage per league, updated online with its own step;
    - each new season, ratings pulled towards the league mean by `carry`;
    - a team new to a league starts at the mean of the teams relegated from it the season before, else the league
      mean plus `new_team_offset`;
    - result and goal difference from ordered logits on the rating gap, refitted at each season start on every
      earlier match.
  - `Forecast` with `result` and `goal_difference`, no matrix.
- Constants: chosen on tuning by a small grid on result log loss (`k` 10–40, `carry` 0.6–1.0), written as the
  defaults with the date.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_bigger_win_moves_the_rating_more`.
  - `test_ratings_are_pulled_towards_the_league_mean_each_summer`.
  - `test_a_team_with_no_history_gets_a_finite_forecast` (Review Focus 2).
  - `test_a_stronger_home_team_is_favoured_and_the_draw_peaks_at_an_even_gap`.
  - `test_ordered_logit_recovers_known_cutpoints` on simulated data.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register `"elo"` in `MODELS`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tune** on tuning seasons and record the chosen constants.
- [ ] **Step 6: Commit** with a tracker line: constants and tuning scores.

### Task 9: Pi-ratings (model 2)

**Files:**
- Create: `Scripts/data_platform/game_read/models/pi_ratings.py`
- Test: `Scripts/tests/data_platform/test_game_read_pi.py`

**Interfaces:**
- Produces `PiRatings(learning=0.035, cross=0.7)` (Constantinou and Fenton 2013, starting values; tuned):
  - each team has a home and an away rating;
  - a rating's expected goal difference is `sign(R) · (10 ** (|R| / 3) − 1)`;
  - the predicted goal difference is the home team's home value minus the away team's away value;
  - the error is weighed by `3 · log10(1 + |e|)`;
  - the venue rating moves by `learning ×` the weighted error, and the other rating by `cross ×` that move;
  - result and goal difference come from `ordered` logits on the predicted goal difference, as in Task 8.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_home_win_bigger_than_expected_raises_both_home_team_ratings_and_lowers_the_away_teams`.
  - `test_the_expected_goal_difference_follows_the_paper` (values for R = 0, ±1, ±3).
  - `test_a_team_with_no_history_gets_a_finite_forecast`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register `"pi"`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tune** `learning` and `cross` on tuning seasons.
- [ ] **Step 6: Commit** with a tracker line.

### Task 10: Dixon–Coles ratings with time decay (model 3)

**Files:**
- Create: `Scripts/data_platform/game_read/models/dixon_coles.py`
- Test: `Scripts/tests/data_platform/test_game_read_dixon_coles.py`

**Interfaces:**
- Produces `DixonColes(half_life_days=..., ridge=1e-3, window_days=1460)`:
  - `log λ_home = μ_league + home_league + attack_home − defence_away` and
    `log λ_away = μ_league + attack_away − defence_home`, as in the A6 design;
  - one fitted rho with the low-score correction;
  - a promoted-team attack and defence shift for a team's first season in a division;
  - matches weighted `exp(−ln 2 · days_ago / half_life_days)`;
  - a small ridge penalty on attack and defence, so teams with little history shrink towards the league;
  - fitted by L-BFGS with the exact gradient, refitted lazily in `predict` when new matches have been learned.
  - The forecast carries the score matrix, so every goal line, handicap and both teams to score.
- `half_life_days` is chosen on tuning from 120, 180, 270, 365 and 540.
- Check, instead of penaltyblog: with no decay, no rho and no ridge, the fitted attack and defence equal scikit-learn's
  `PoissonRegressor(alpha=0)` on the same design.

- [ ] **Step 1: Write the failing tests.**
  - `test_the_gradient_matches_a_numerical_gradient`.
  - `test_without_decay_rho_or_ridge_it_is_the_poisson_regression`.
  - `test_older_matches_count_less`: a team strong two years ago and weak lately is rated weak.
  - `test_the_score_matrix_sums_to_one_and_rho_moves_low_scores`.
  - `test_a_promoted_team_with_no_history_gets_a_finite_forecast` (Review Focus 2).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register `"dc"`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tune** the half-life on tuning seasons (log loss on result and over/under 2.5); note the run time.
- [ ] **Step 6: Commit** with a tracker line.

### Task 11: Drifting ratings (model 4)

**Files:**
- Create: `Scripts/data_platform/game_read/models/drifting.py`
- Test: `Scripts/tests/data_platform/test_game_read_drifting.py`

**Interfaces:**
- Produces `Drifting(gain=..., summer_widening=...)`:
  - model 3's structure;
  - each team's attack and defence move at their own pace: after each day, a filter step on the Poisson score of
    that day's goals, sized by each rating's uncertainty;
  - uncertainty grows between matches and more over the summer;
  - starts from model 3's fit at the end of the warm-up.
- How the filter is built is left to the builder (spec, "Left to the builder"). `gain` and `summer_widening` are
  tuned on tuning seasons.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_team_whose_scoring_jumps_is_followed_faster_than_by_model_3`: simulated team, goal rate 1.0 then 2.0
    from match 30; after 10 more matches, the drifting attack is closer to 2.0 than model 3's at a 365-day
    half-life.
  - `test_uncertainty_grows_over_the_summer`.
  - `test_a_team_with_no_history_gets_a_finite_forecast`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register `"drift"`.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tune** on tuning seasons.
- [ ] **Step 6: Commit** with a tracker line.

### Task 12: The average of the models (model 9, first version)

**Files:**
- Create: `Scripts/data_platform/game_read/models/average.py`
- Test: `Scripts/tests/data_platform/test_game_read_average.py`

**Interfaces:**
- Produces `fit_weights(forecasts_by_model, outcomes, market) -> dict[str, float]` (non-negative, summing to 1, a
  log-linear pool fitted on tuning forecasts per market) and `Average(members, weights)` implementing `Model`.
  Members without a matrix take no part in goal lines or both teams to score.

- [ ] **Step 1: Write the failing tests.**
  - `test_one_perfect_member_takes_all_the_weight` on simulated data.
  - `test_weights_are_never_negative_and_sum_to_one`.
  - `test_goal_lines_come_only_from_members_with_a_matrix`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register `"average"`. Its weights are fitted inside each run on the tuning block, never
  on locked seasons.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 13: Phase 1 runs and the first report to the owner

**Files:**
- Modify: `docs/rework-tracker.md` (results), `docs/superpowers/specs/2026-11-06-game-read-engine-design.md` only if
  a finding changes the design (with the owner).

- [ ] **Step 1: Tuning run of every model.** `... run --models market,control,elo,pi,dc,drift,average --block
  tuning --out Research/game-read-bench-<date>`. Expected: the zero line passes and every section is filled.
- [ ] **Step 2: Read the Opportunities section** and write each finding into the report with its evidence.
- [ ] **Step 3: Freeze the versions.** Commit any last tuning change; the tree must be clean.
- [ ] **Step 4: One locked run.** `... --block locked --confirm-locked`, once, for the frozen versions. Expected:
  one ledger line. Never rerun it after looking at the result; a new model version gets its own single locked run.
- [ ] **Step 5: The tracker entry:** each model's scores on tuning and locked, pass or fail per score, the
  Opportunities, and the ledger line.
- [ ] **Step 6: The first report to the owner**, in plain language:
  - what passed and what did not;
  - what that means for Most likely and Value;
  - the Opportunities;
  - what phase 2 does next.
- [ ] **Step 7: Full suite and golden check** as in Task 7 Step 7. Commit the tracker entry.

### Task 14: The ClubElo yardstick (only with the owner's yes)

**Files:**
- Create: `Scripts/data_platform/game_read/models/clubelo.py`, `Scripts/data_platform/game_read/clubelo_names.json`
- Test: `Scripts/tests/data_platform/test_game_read_clubelo.py`

- [ ] **Step 1: Check ClubElo's terms of use** (clubelo.com) and its coverage of the 10 leagues, and write both
  into the tracker. Stop and ask the owner before any download. A download of one snapshot per matchday of the
  scored seasons is roughly 2,000 requests, spread over days at a polite pace.
- [ ] **Step 2: With the owner's yes,** download into a new folder under `Research/clubelo-<date>/`, and read it
  with `python -I`.
- [ ] **Step 3: Write the failing tests:**
  - `test_names_map_to_football_data_teams`: every scored team maps, or is listed;
  - `test_the_yardstick_uses_the_rating_published_before_the_match`.
- [ ] **Step 4: Run them.** Expected: FAIL.
- [ ] **Step 5: Implement `ClubElo` as a `Model`.** Result and goal difference come from `ordered` logits on its
  rating gap. Register `"clubelo"`.
- [ ] **Step 6: Run the tests.** Expected: PASS.
- [ ] **Step 7: Tuning run beside Elo; commit** with a tracker line: does our Elo match a free public Elo?
