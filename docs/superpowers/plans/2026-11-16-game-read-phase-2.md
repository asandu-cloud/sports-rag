# Game-read engine, phase 2: statistics and machine learning on top — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task,
> inline in one session: no subagents unless the owner asks (CLAUDE.md). Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Test whether richer inputs and machine learning add anything to the market, on the phase 0 bench.
- D12's variables enter models 3–4 one at a time, through the ablation gate.
- The market blend may lean away from a model (the locked-run lead of 16 Nov).
- Model 6: boosted trees on the market's error.
- Model 7: boosted trees on the outcome.
- Then the average of whatever passes.

**Architecture:** Everything stays in the research package `Scripts/data_platform/game_read/` and the bench
`Scripts/ops/game_read_bench.py`; nothing live imports either, and no published pick changes.
- Match statistics are read from `Index/platform.db` once, inside the gate, into a local read-only file. They are
  shots inside and outside the box, kickoff times and every fixture we hold, used for rest. The bench never reads
  `platform.db`.
- A fresh block, the 2026/27 season so far, is added to the history behind the same guard as the locked seasons.

**Tech Stack:** as phase 1, plus LightGBM 4.6.0 (already in the shared `.venv`; no new dependency).

**Spec:** `docs/superpowers/specs/2026-11-06-game-read-engine-design.md` ("Statistics and machine learning on top
(phase 2)"). Phase 1 results: tracker D77, tasks 8–13.

**Start condition:** met. The owner answered the three decisions on 16 Nov (below) and code may start.

**Rough time:** two weeks, to early December (spec, "Phases and timing"). Nothing in it changes a published pick.

## The owner's answers (16 Nov)

1. **xG.** Real xG stays out of phase 2: ours starts with 2023/24, inside the locked seasons. Instead, a derived
   *shot-location xG* is tested in Task 3, beside shots in the box and shots on target, behind the same 0.5% gate.
   - It values each shot by where it was taken (inside or outside the box).
   - Its weights are fitted on the tuning seasons only, then frozen.
   - Shots on target are left out of it: every goal is a shot on target, so they would partly count the goals.
   - If it passes, real xG for the top five (Understat or a licence) becomes a terms check for the owner later.
2. **The 2026/27 CSVs.** I fetch them: football-data.co.uk's 10 free files only, once now (Task 1) and once
   before the final run (Task 9). Each download goes in its own new folder and is read with `python -I`.
3. **The fresh block is the last check:** yes.
4. **The derived xG carries forward (17 Nov),** although it missed the 0.5% gate in Task 3 (−0.39% on the result,
   interval [−0.51, −0.26]). Rest (Task 4) is gated on top of `dc+xg` and `drift+xg`, and the features pass (Task 5)
   uses them as its Dixon–Coles and drifting members.

## Not in this plan

- **D12's cards and corners variables** (referee D1, foul matchup D2, corner pressure D3): football-data has
  cards and corners counts but no prices for those markets, so the bench cannot score them. They go with phase 3,
  on our saved reads.
- **Model 8, the Bayesian model:** the spec builds it only if models 3–4 add anything to the market. Phase 1 showed
  they don't (tracker D77, task 13).
- **ClubElo:** dropped by the owner on 16 Nov.

## Progress

Each task has a start commit and a done commit (CLAUDE.md, "Large builds"). Commit names: "Start game-read phase 2
task N: …", "Done game-read phase 2 task N: … (<tests>, D77)", "Fix game-read phase 2 task N: …". To undo a task:
`git revert <its done commit>`. The local ledger is `.superpowers/sdd/2026-11-16-game-read-phase-2/progress.md`.

| Task | What it builds | Status | Start commit | Done commit | Tests |
|---|---|---|---|---|---|
| 1 | The fresh block (2026/27) and its guard | Done 16 Nov | b602472 | ed8ea5a | 6 new, all pass; suite 3,204 passed + the 3 known errors; 1,084 fresh matches |
| 2 | Match statistics: export and loader | Done 16 Nov | 1be9759 | 2b2f594 | 5 new, all pass; suite 3,209 passed + the 3 known errors |
| 3 | Shot ratings (S3) into models 3–4, through the ablation gate | Done 17 Nov | ddba5eb | 82161ca | 8 new, all pass; suite 3,217 passed + the 3 known errors; no variant passes the gate |
| 4 | Rest (C1) into models 3–4, through the ablation gate | Done 17 Nov | e6b53bc | 77e740a | 4 new, all pass; suite 3,221 passed + the 3 known errors; rest fails the gate |
| 5 | The features pass and the lean-away blend (model 5b) | Done 17 Nov | db4adb9 | 29d9b9e | 4 new, all pass; suite 3,225 passed + the 3 known errors; no lean blend beats the market |
| 6 | Boosted trees on the market's error (model 6) | Done 17 Nov | 5528df6 | the "Done game-read phase 2 task 6" commit | 5 new, all pass; suite 3,230 passed + the 3 known errors; no setting beats the market |
| 7 | Boosted trees on the outcome (model 7) | Not started | | | |
| 8 | The average of what passes (model 9, second version) | Not started | | | |
| 9 | Phase 2 runs (tuning, one locked, one fresh) and the report | Not started | | | |

## The locked seasons and the fresh block

Phase 1's single locked run (ledger line 1) was for phase 1's versions. Phase 2's versions are new, so by the
spec's rule each gets one locked run, all frozen together in Task 9. But 2023–2025 are no longer untouched. Their
phase 1 results are known, and the lean-away lead came from them. So:
- **Every setting is still chosen on the tuning seasons (2019–2022) only,** and every variant tried is counted in
  the report, so the owner can see how hard tuning was pushed.
- **One locked run** for all of phase 2's frozen versions (Task 9), logged in the ledger as before. A model passes
  only by the spec's rules on the locked seasons.
- **One fresh run** on the 2026/27 season so far: about 1,050 matches now, and about 1,400 by early December. It
  is too small to pass a model alone, so it is a check. A model that passes on the locked seasons must point the
  same way on the fresh block; its interval need not leave out zero.
- **The lean-away lead is judged elsewhere.** Its locked result is reported but cannot count. It must show on the
  tuning seasons (fitted only on earlier seasons) and point the same way on the fresh block. Model 6 is ablated
  without the lead's inputs, so the report shows how much of any locked gain comes from it.
- Phase 4's shadow run on live weekends remains the final judge, as the spec says.

## Global Constraints

- Branch `backend`; small commits, files staged by name, no attribution lines; each task's tracker line under D77
  in the same commit as its code. The next free D-number is D112 if a decision needs its own.
- Never stage `Output/`, `Index/`, `Research/`, `.impeccable/`, `.superpowers/`, `.claude/`, `.taste-skill/`.
- `Index/platform.db` is read only in Tasks 1 and 2:
  - only inside `data_access()` (`Scripts.data_platform.services.refresh_coordination`);
  - never while `pgrep -f refresh-season` finds the refresh;
  - in short read-only statements, run with `PYTHONPATH=.` from the repo root.
- `Index/football_data.sqlite3` and the new `Index/game_read_stats.sqlite3` are opened read-only by the bench.
- Downloaded CSVs go in a new folder, are checked (header, row count, no formulas) before they are copied to
  `Research/football-data/raw/`, and the importer runs on them with `python -I`. The repo root is added to
  `sys.path` explicitly.
- No live code path changes: `golden_snapshots check` stays unchanged; no engine switch.
- Blocks (season start years): warm-up 2012–2018, tuning 2019–2022, locked 2023–2025, fresh 2026 on.
- The ablation gate (D12, spec): at least 0.5% better log loss than the model without the variable, interval below
  zero, and no league worse by more than 2%, on the tuning seasons.
- Intervals, resamples, seed, the market's chance, the floors and the pass rules: as phase 1 (`scores.py`).
- Phase 1's frozen versions stay as they are. A variable makes a new model name (`dc+box`), never a new default.

## Review Focus

1. **A match with no statistics** (no linked fixture; Belgium's shots in the box before 2021; a missing kickoff):
   every variable falls back to neutral, never NaN or an error, and the trees get a missing value, which they
   handle. Tests in Tasks 2–4.
2. **Leakage through a statistic.**
   - A side's shot rating uses only matches before the predicted day.
   - Rest uses only earlier kickoffs.
   - The trees train only on seasons before the one they predict.
   Tests in Tasks 3, 4 and 6.
3. **A run that touches the locked or fresh seasons without being named** must refuse and write nothing.
   Adding the fresh season must not change a single 2012–2025 row of the history. Tests in Task 1.
4. **A team with two fixtures close together** (a European midweek match, a postponed game, two matches on one
   day): rest counts every fixture we hold, in any competition, and is never negative. Test in Task 2.
5. **The lean-away blend's weight below zero:** it must produce chances that sum to 1 and stay inside (0, 1), and
   be fitted only on seasons before the one it predicts. Test in Task 5.

---

### Task 1: The fresh block (2026/27) and its guard

**Files:**
- Modify: `Scripts/data_platform/game_read/history.py`, `Scripts/ops/game_read_bench.py`,
  `Scripts/data_platform/game_read/opportunities.py`
- Test: `Scripts/tests/data_platform/test_game_read_history.py`, `Scripts/tests/test_game_read_bench.py`

**Interfaces:**
- Produces `FRESH_FROM = 2026`; `block(season)` returns `"fresh"` from 2026.
- `run --block fresh` needs `--confirm-locked` and committed bench code, like a locked run, and is logged in the
  same ledger with its block. A locked run walks forward only to 2025.
- `opportunities.scan` refuses fresh matches as it refuses locked ones.

- [ ] **Step 1: Write the failing tests.**
  - `test_2026_is_fresh`.
  - `test_a_fresh_run_must_be_named`.
  - `test_a_locked_run_never_predicts_the_fresh_season`.
  - `test_opportunities_never_look_at_fresh_seasons`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Add the season's data** (after the owner's decision 2).
  - The 10 CSVs go to a new folder each and are checked, then copied as `{DIV}_2627.csv` into the bench's own raw
    folder, `Research/game-read-bench/raw/`, beside links to the shared raw CSVs.
  - The importer builds the bench's own file, `Index/game_read_history.sqlite3`, inside `data_access()`, with
    `python -I`. The bench reads only that file (`history.BENCH_DB`).
  - *Ruling (16 Nov):* the shared `Index/football_data.sqlite3` and `Research/football-data/raw/` are left as they
    are. The live pick evidence (D82) reads the shared file from 2024/25 on, and takes this season from our own
    database only after the file's last season. A 2026/27 in the shared file would freeze its evidence at the
    download date.
  - A check script, kept in the scratchpad, confirms that every 2012–2025 match and price the bench reads is
    identical to the shared file's, in the same order. On any difference, stop and report.
  - Record the fresh match count in the tracker.
- [ ] **Step 6: Commit** with a tracker line.

### Task 2: Match statistics: export and loader

**Files:**
- Create: `Scripts/data_platform/game_read/stats.py`
- Modify: `Scripts/ops/game_read_bench.py` (an `export-stats` subcommand)
- Test: `Scripts/tests/data_platform/test_game_read_stats.py`

**Interfaces:**
- `export(out=Index/game_read_stats.sqlite3)`: one gated, read-only pass over `platform.db`.
  - For each football-data match linked to a fixture (`fixture_api_id`): each side's shots inside and outside the
    box, real xG and kickoff (UTC). Real xG is exported only to report how close the derived xG comes on 2022/23
    (a tuning season); no model reads it.
  - For every finished fixture we hold, in any competition: kickoff and both teams, by API-Football team id.
  - Refuses while the refresh runs.
- `@dataclass(frozen=True) class MatchStats`:
  - `home_box`, `away_box: int | None` (shots inside the box);
  - `home_outside`, `away_outside: int | None` (shots outside the box);
  - `home_target`, `away_target: int | None` (shots on target, from football-data's own columns, 2012 on);
  - `home_xg`, `away_xg: float | None`;
  - `home_rest`, `away_rest: int | None`: whole days since the team's previous fixture in any competition we
    hold, from 2019; None before 2019 or when unknown.
- `load(path) -> dict[int, MatchStats]`, keyed by `Match.id`, opened read-only.

- [ ] **Step 1: Write the failing tests** (on a small synthetic platform file and history file):
  - `test_rest_counts_a_european_midweek_match`;
  - `test_rest_is_none_before_2019_and_never_negative`;
  - `test_an_unlinked_match_has_no_box_or_xg`;
  - `test_export_refuses_while_the_refresh_runs`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Export for real**, outside refresh hours, and record coverage by league and season in the tracker.
- [ ] **Step 6: Commit** with a tracker line.

### Task 3: Shot ratings (S3) into models 3–4, through the ablation gate

**Files:**
- Create: `Scripts/data_platform/game_read/models/shots.py`
- Modify: `models/dixon_coles.py`, `models/drifting.py`, `scores.py`, `report.py`, `Scripts/ops/game_read_bench.py`
- Test: `Scripts/tests/data_platform/test_game_read_shots.py`, `test_game_read_scores.py`

**Interfaces:**
- `ShotRatings(kind="box"|"target"|"xg", half_life_days=365)` uses Dixon–Coles's machinery on a side's shot count
  (or derived xG) instead of goals (no rho). It learns after each day and gives each side's expected value for a
  match.
- The derived shot-location xG of one side in one match is `w_in · inside + w_out · outside`. These are per-shot
  values, added up like real xG.
  - `fit_location_xg(stats, matches) -> tuple[float, float]` fits them by Poisson maximum likelihood on goals.
  - They are fitted once on the tuning seasons (2019–2022, about 29,000 team-matches) and frozen in the constant
    `LOCATION_XG`, with the date of the fit.
  - The report states how close they come to real xG on the 2022/23 matches that have it.
  - A side with no shot counts gets None.
- `DixonColes(..., variables=(), stats=None)` and `Drifting(..., variables=(), stats=None)`.
  - Each variable adds a term `beta · x` to a side's log goals: `x` is the log of the side's expected shots minus
    its league's mean.
  - The variable is neutral (0) until a team has 5 matches with the statistic.
  - `beta` is fitted with the ratings in model 3; model 4 takes it from model 3's fit at its start.
  - Names in the bench: `dc+box`, `dc+target`, `dc+xg`, `drift+box`, `drift+target`, `drift+xg`.
- `scores.ablation(base_rows, variant_rows, market) -> dict`: the variant's log loss minus the base's, in % of the
  base's, with the week-block interval and by league. `scores.gate(ablation) -> bool`: estimate ≤ −0.5, high < 0,
  no league above +2.0.
- Three shot variables are tested side by side: shots in the box, shots on target and the derived xG.
  - Shots on target covers 2012 on for most leagues. Shots in the box and the derived xG start in 2019, so their
    ratings are thin in 2019.
  - D12's budget (three goal inputs) keeps only the best of the three that pass.
- *Ruling:* the derived xG's two weights are fitted on the same tuning seasons the gate scores. The owner chose
  this. It is two numbers from about 29,000 team-matches, so the in-sample gain is negligible, and the locked and
  fresh seasons never touch the fit.
- A variable passes when it clears the gate on the result or on over 2.5, and does not make the other worse (its
  interval must not sit above zero). The report gets an "Ablation" section.

- [ ] **Step 1: Write the failing tests.**
  - `test_a_shot_rating_uses_only_earlier_days`.
  - `test_a_team_with_no_shot_history_is_neutral`.
  - `test_beta_is_recovered_on_simulated_matches`: goals simulated with a known effect of expected shots.
  - `test_location_xg_recovers_known_per_shot_values`: goals simulated as Poisson with 0.11 a shot inside the box
    and 0.03 outside.
  - `test_location_xg_is_none_without_shot_counts`.
  - `test_the_gate_needs_half_a_percent_and_no_league_worse_than_two`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; fit `LOCATION_XG` on the tuning seasons from the Task 2 export; register the six
  names.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tuning run** of `dc`, `drift` and the six variants. Record each gate result in the tracker; a
  failure is removed from the next tasks and recorded (D12).
- [ ] **Step 6: Commit** with a tracker line.

### Task 4: Rest (C1) into models 3–4, through the ablation gate

**Files:**
- Modify: `models/dixon_coles.py`, `models/drifting.py`, `Scripts/ops/game_read_bench.py`
- Test: `Scripts/tests/data_platform/test_game_read_rest.py`

**Interfaces:**
- Variable `rest`: each side's `(min(rest, 7) − 7) / 7`. It enters the side's own log goals and the opponent's
  log goals, with two fitted coefficients shared by all leagues. A missing rest is 0.
- It is added on top of whatever passed in Task 3 (`dc+box+rest`, or `dc+rest` if neither shot variable passed),
  and gated against that base.

- [ ] **Step 1: Write the failing tests.**
  - `test_short_rest_lowers_scoring_on_simulated_matches`.
  - `test_missing_rest_is_neutral`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register the names.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tuning run and gate.** Record in the tracker.
- [ ] **Step 6: Commit** with a tracker line.

### Task 5: The features pass and the lean-away blend (model 5b)

**Files:**
- Create: `Scripts/data_platform/game_read/features.py`, `Scripts/data_platform/game_read/models/lean.py`
- Test: `Scripts/tests/data_platform/test_game_read_features.py`, `test_game_read_lean.py`

**Interfaces:**
- `Features(members, stats)` keeps its rating models (Elo, pi, the best Dixon–Coles and drifting variants) and
  asks them to predict every day from 2013, warm-up included. So every match from 2013 has out-of-sample inputs.
  Per match it gives:
  - the members' chances and expected goals;
  - the market's chances, and the market minus each member on the log-odds scale (the lead's inputs);
  - the shot and rest variables;
  - league, month, weekday and the bookmakers' margin.
  Each day is predicted once, and its results are learned after.
- `LeanAway(member)` (model 5b): D20's blend, `logit p = a + b·logit p_market + c·logit p_member`, refitted at
  each season start on all earlier seasons, with c free to fall below zero. Result (three-way) and over 2.5. Names:
  `lean-elo`, `lean-dc`, `lean-average`.

- [ ] **Step 1: Write the failing tests.**
  - `test_features_for_a_day_use_only_earlier_days`.
  - `test_a_lean_away_fit_never_sees_its_own_season`.
  - `test_lean_away_chances_sum_to_one_with_c_below_zero`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register the names.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tuning run.** Record whether leaning away helps on the tuning seasons, out of sample.
- [ ] **Step 6: Commit** with a tracker line.

### Task 6: Boosted trees on the market's error (model 6)

**Files:**
- Create: `Scripts/data_platform/game_read/models/trees.py`
- Test: `Scripts/tests/data_platform/test_game_read_trees.py`

**Interfaces:**
- `Trees(on="error", features=Features(...), params=...)`: one LightGBM model per market.
  - Markets: result (three classes), over 2.5 and the real half-line handicap (two classes).
  - It starts from the market's log-odds (`init_score`) and learns only corrections.
  - It is retrained at each season start on every earlier match from 2013. The number of rounds is set by early
    stopping on the last training season, never the one predicted.
  - Its forecast carries the result and the real over 2.5 and handicap lines; derived lines get no chance.
- Name `trees-error`; `trees-error-nolead` is the same without the lead's inputs.
- Tuned on the tuning seasons:
  - leaves 7 or 15, minimum leaf 200 or 500, L2 1 or 10, learning rate 0.02;
  - every combination tried is counted in the report.

- [ ] **Step 1: Write the failing tests.**
  - `test_with_no_rounds_it_is_the_market`: zero trees give the market's chances exactly.
  - `test_it_learns_a_planted_correction`: simulated matches where the market is off by a known amount in one
    league.
  - `test_it_never_trains_on_the_season_it_predicts`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**; register the names.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Tune** on the tuning seasons; note the run time.
- [ ] **Step 6: Commit** with a tracker line.

### Task 7: Boosted trees on the outcome (model 7)

**Files:**
- Modify: `models/trees.py`
- Test: `Scripts/tests/data_platform/test_game_read_trees.py`

**Interfaces:**
- `Trees(on="outcome", ...)`: the same, but with no `init_score`. The market's chance is one input among the rest.
  Name `trees-outcome`.

- [ ] **Step 1: Write the failing test** `test_outcome_trees_take_the_market_as_an_input_only`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Tune** as Task 6.
- [ ] **Step 6: Commit** with a tracker line.

### Task 8: The average of what passes (model 9, second version)

**Files:**
- Modify: `models/average.py`, `Scripts/ops/game_read_bench.py`
- Test: `Scripts/tests/data_platform/test_game_read_average.py`

**Interfaces:**
- `average.MEMBERS` becomes a bench option, `--average-of`. The members are the phase 2 models that pass on the
  tuning seasons.
- If none passes, the average of the best variant of each family is run as a diagnostic and labelled so in the
  report, as in phase 1.
- Its weights are still fitted on the tuning seasons only.

- [ ] **Step 1: Write the failing test** `test_the_average_takes_the_members_it_is_given`.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it.** Expected: PASS.
- [ ] **Step 5: Commit** with a tracker line.

### Task 9: Phase 2 runs and the report

**Files:**
- Modify: `docs/rework-tracker.md`; the spec only if a finding changes the design (with the owner).

- [ ] **Step 1: Tuning run** of every phase 2 model with `market` and `control`. Expected: the zero line passes
  and every section is filled.
- [ ] **Step 2: Freeze.** Commit any last tuning change; the tree must be clean.
- [ ] **Step 3: Refresh the fresh season's CSVs** (as Task 1 Step 5), then commit nothing new to the bench.
- [ ] **Step 4: One locked run** (`--block locked --confirm-locked`) for all frozen phase 2 versions. Expected: one
  ledger line.
- [ ] **Step 5: One fresh run** (`--block fresh --confirm-locked`). Expected: one ledger line.
- [ ] **Step 6: The tracker entry:**
  - each model's scores on tuning, locked and fresh, and pass or fail per score;
  - the gate results and the number of variants tried;
  - how much of model 6's result comes from the lead's inputs;
  - the Opportunities;
  - the ledger lines.
- [ ] **Step 7: The report to the owner**, in plain language: what passed; what it means for Most likely and Value;
  whether phase 4 (the selector behind `SPIX_GAME_READ`, then the shadow run) has anything to start with.
- [ ] **Step 8: Full suite and golden check;** commit the tracker entry.
