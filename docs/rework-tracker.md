# Engine rework tracker

The single source of truth for the rework's **current** plan, status, decisions and
deviations. The Engine Rework report (claude.ai artifact BHP6rm7e4Xf5a29KGoCiUe) is the
original design; where this file differs, **this file wins**. Update it in the same
commit as the work it describes; the report is refreshed from it at milestones.

Last updated: 2026-10-28 (stage 3 live; refresh retry; history rewrite compared)

## Status

| Step | Status | Commits |
|---|---|---|
| A0 Safety net | Done 15 Oct | 02fa536, f7459f0, c1e9b42 |
| A1 Measurement | **Done 20 Oct** (step 4 closed: equal weights kept, true close live, D39, D40) | see below |
| A2 Canonical features | A2a live since 17 Oct 16:28 (`DOCUMENT_STORE=sqlite`; D11); A2b after A6 (D13) | 7af457b |
| A3 Market anchor and honest uncertainty | Started 16 Oct under D16 (research and shadow only) | see A3 table |
| A4 Calibration | A4.1 dataset done 16 Oct (25,527 graded candidates, 160 fixtures; grading matches settlement on all 1,918 overlapping picks). A4.2 method study done 16 Oct: no calibrator adopted yet (D28), rerun weekly; live as A3.7 stage 5, not before Tue 10 Nov (D29) | 197dc73; see D28 |
| A5 Selection policy | Research started 16 Oct (D30): A5.1 done (Spix's disagreement with the market is almost all error); A5.2 done (pricing against the market wins, D31); A5.3 part 1 done 20 Oct (half the best prices were not on offer at the read, D39); price confirmation built 23 Oct, off until stage 4 (D42); exchange commission dropped, the feed has no exchange prices (D43); confidence labels built 23 Oct behind `SPIX_CONFIDENCE_LABELS`, all low until a spread cut-off passes (D44); stage 4 replay with confirmation 24 Oct (Findings); the goal-market selection rule ships with stage 4 (D33), the rest at stage 6, earliest Tue 17 Nov | see D30, D31, D33 |
| A6–A8, Part B | Not started; A6 prep done 24 Oct (penaltyblog checked, not used: ratings fitted in-house, Findings) | 43b4431 |
| Market expansion (D25) | Planned after A5 | see D25 |

### A1 (order changed 15 Oct, see D6)

| # | Task | Status | Commit |
|---|---|---|---|
| 1 | Handicap grading fix | Done | dfb7f93 |
| 2 | Closing prices from API-Football's final pre-kickoff snapshot | Done | aa2f2ee |
| 3 | Market reference price at publication | Done | 1854f23 |
| 4 | Bookmaker weights for the reference price | Done 20 Oct: no weights, the median stays (owner, D40; accuracy is flat, D15); freshness handled by the true close (D18, D39) and price confirmation (D40) | 405e9c9 |
| 5 | Weekly scorecard | Done (Mondays 08:00) | d1ae597 |
| 6 | Market baseline in the research protocol | Done | see D9 |

A1 gate (report): >90% of finished picks settled and with a closing price ✓ (95.9% closing,
handicap regrade finishing 15 Oct night); first scorecard ✓; first reference weights with
Pinnacle's freshness measured ✓ (20 Oct: freshness D15 and D39, true close live; the owner kept equal weights, D40).

### A3 (started 16 Oct, ahead of A1 step 4 and the A2a switch; rules in D16)

| # | Task | Status | Live? |
|---|---|---|---|
| 3.1 | Margin-removal method: proportional vs additive vs power vs Shin on football-data closing history | Done: power (D17) | No: option `method="power"`, default unchanged until A1's gate |
| 3.2 | M1 market goal expectations per team (handicap + totals) | Done: research (D19) | No: `market_goals.solve`; used by 3.3 and A8 |
| 3.3 | Fitted blend `logit p = a + b·logit p_market + c·logit p_spix` per market | Done: research; gate not passed with the 2023 engine (D20) | No: at go-live, c = 0 for 1X2/O/U 2.5 **until A6** (D21) |
| 3.4 | Gamma-distributed team rates replacing n/(n+8) and the variance floors | Done: research; not adopted (D22) | No: the simpler 3.5 wins |
| 3.5 | Phase 4 dispersions (corners 0.025, SoT 0.01) replace the variance floors, behind a switch | Done: built (D23), all competitions and stages | No: `SPIX_COUNT_DISPERSION=fixed` at go-live |
| 3.6 | Dixon–Coles rho per league | Done: research; not adopted, rho stays −0.10 (D24) | No change |
| 3.7 | Go-live package, staged (D26): build by Mon 19 Oct; stage 1 dispersions Tue 20; stage 2 power margin removal in selection Fri 23; stage 3 market anchor (strict scope) Tue 27; confirm at the Mon 2 Nov scorecard; stage 4 market anchor for all goal markets via M1, earliest Tue 3 Nov (D29); stage 5 A4 calibrators, earliest Tue 10 Nov | **Stage 1 live 20 Oct 11:44 UTC** (`SPIX_COUNT_DISPERSION=fixed` saved in `.env`; the first read after it, 12:09 UTC, records `fixed`; the golden preview on today's code is unchanged: corners 33 → 38, 5 added, 10 moved). **Stage 2 live 23 Oct 07:52 UTC** (`SPIX_DEVIG_METHOD=power` saved in `.env`; reads waited for the refresh rerun: all 90 reads of 08:26–08:32 UTC and the 230 picks they recorded carry `power` and `fixed`; 88 of their 90 totals results have the quoting bookmaker's power fair price exactly, none the proportional one; the other 2 are long shots whose opposite price is below the odds floor, so the raw price is used, as before). **Stage 3 live 28 Oct** (`SPIX_MARKET_ANCHOR=strict` saved in `.env` 01:07 local, owner; website restarted 01:17; the first reads came after the refresh rerun reopened the gate at 23:44 UTC 27 Oct: all 50 of 23:52–23:58 UTC record `fixed`, `power`, `strict`; see Findings). Package build started 17 Oct: switch record done (`core/engine_switches.py`); stage 2 switch built (`SPIX_DEVIG_METHOD=power`); stage 3 switch built (`SPIX_MARKET_ANCHOR=strict`); stage 4 built (`SPIX_MARKET_ANCHOR=goals`); `golden_snapshots preview` summarises a stage's pick changes | Staged, one change at a time |

## Open items with a destination

Things found along the way that must not be lost. Each names where it gets done.

| Item | Goes to | Why deferred |
|---|---|---|
| Settlement for player picks (participant) and period backfill | Part B | No current pick needs it (0 affected) |
| Unit-bet slip link to the legacy `predictions.db` | A7 | Lives in the Discord legacy tracker A7 replaces |
| Phase 6 shadow controller (`com.bettingrag.phase6-shadow`, hourly, runs `Research/phase6-shadow-logo-2026-10-07/controller.py`): fails on every run since A0 changed a file it fingerprints (`alembic/env.py`, 02fa536, 15 Oct): "Operational prediction input changed; review a new release"; the last 148 runs failed, so its diagnostic window (to 20 Oct) lost its last days. Owner: retire it, or move it into `Scripts/` and re-release | Owner, now | Its window ended 20 Oct |
| Done: tag `pre-rework` is on GitHub at cf00af8 (checked 21 Oct with `git ls-remote`) | Done | |
| Keep the Mac awake for scheduled jobs: the daily refresh runs under `caffeinate -i -s` since 17 Oct (it cannot idle-sleep mid-run); still open, a daily scheduled wake before 07:15 (`sudo pmset repeat wakeorpoweron MTWRFSU 07:05:00`), mains power, or a server. On 17 Oct the Mac slept on battery until ~11:48, waking for 2–3 s every ~17 min: the 07:15 refresh ran only in those slices and finished at 12:25 (stage times normal). While asleep the Match Read worker, measurement worker and odds archive do not run either, which matters for this weekend's near-kickoff captures (A1 step 4 freshness rerun, Mon 19 Oct). Again on 20 Oct: on battery, no daily wake scheduled (`pmset -g sched`), the Mac slept 07:00–11:00 with brief wakes and the refresh took 4 h 35 min (04:21–08:56 UTC; 19 Oct: 1 h 3 min). 21 Oct: 6 h 43 min (04:17–11:00 UTC), still no daily wake scheduled. 26 and 27 Oct: the refresh failed on a sleeping Mac both mornings and nothing reran it, so the gate stayed shut 8 h 22 min and 18 h 27 min (no Match Reads, no measurement) until rerun by hand; since 28 Oct the job retries hourly (`--if-due`, see Findings) | Owner (wake, mains, webhook) | Operational risk to every scheduled job |
| Closed 20 Oct: no bookmaker weights; the median stays (owner, D40) | Done | Power margin removal still becomes the default at stage 2 (D17) |
| Make power the default margin-removal method in measurement (`reference_price`, closing consensus, publication reference, scorecard). Owner, 20 Oct: keep proportional through stages 1–3 so every stage is judged on the same yardstick; switch at the Mon 2 Nov scorecard and recompute the history from the stored per-book prices | Mon 2 Nov scorecard | Refit anything fitted on v1 |
| Closed 20 Oct: no weighted reference (D40); the scorecard adds the true close instead (D39) | Done | |
| Refresh the report artifact from this tracker | Now due (A1 done 20 Oct) | |
| Done 23 Oct (D42): the 22 Oct probe met no error up to its end at 23:11 UTC (6.4 h after Juventus v Rennes's kickoff, 4.2 h after Bournemouth v Milan's); OddsPapi rewrites a match's history about 2 h after kickoff (adding in-play markets); `DUE_AFTER` 2 h and `SETTLED` 6 h stay; the 21 Oct evening cause is not reproduced (errors are retried since D41) | Done | |
| Done 23 Oct (D42): OddsPapi's "No historical odds found" answers stayed so. Two of 22 Oct's (William Hill on Celtic v Celta, one Betfair outcome on Juventus v Rennes) were re-asked every 30 minutes until the Mac slept at 23:51 UTC, then once at 07:19 UTC 23 Oct: still 404 at 12.3 h and 14.6 h after kickoff. `SETTLED` (6 h) stays | Done | |
| Done 28 Oct: no, the rewrite keeps every pre-kickoff price (56 fixtures compared, see Findings). Was: Does OddsPapi's post-match rewrite drop soft bookmakers' pre-kickoff lines? 27 minutes before Juventus v Rennes kicked off (22 Oct), Betano's history had 227 markets; after the rewrite 214 have pre-kickoff prices (1xBet 366 → 347, Bet365 230 → 227; Pinnacle unchanged, 106). Save a weekend match's history at kickoff and compare it with the rewritten one (D42). **Capture running since 23 Oct 07:39 UTC** (`ops/history_rewrite_check.py capture`, `Research/history-rewrite-2026-10-24/`): 73 fixtures of 23–25 Oct (top five, Championship, Primeira Liga, Süper Lig), each saved 2–100 min after kickoff and 3–12 h after; Pinnacle, Betano, 1xBet, Bet365 (ES), Marathonbet, William Hill; free histories, 3 fixture lists bought (the true close reuses them); the Mac must be awake (mains, lid open). Then `compare` | Mon 26 Oct | Own-bookmaker closes (`same_book`) come from the rewritten history |
| Stop embedding into Chroma entirely | A7 | Whale chat and `rag_cli_v2` still search Chroma by meaning |
| A6c LightGBM market-error challenger (D14) | After A6's base model passes | Must beat fitted A6 vs the market baseline |
| Rerun the blend study (`Scripts/ops/blend_study.py`) on A6's forecasts: does Spix earn c > 0? | A6 gate | D20 |
| **Temporary (D21):** 1X2 and O/U 2.5 publish the market's probability (power) with c = 0 **only until A6**; rerun the blend study on A6's forecasts and refit c if Spix earns it | A3.7 go-live, then A6 gate | Not permanent: revisit at A6 |
| OddsPapi true close: built 20 Oct (`services/true_close.py`, `ops/true_close.py`; Pinnacle plus the pick's own bookmaker; API-Football's snapshot kept as fallback, D18, D39); first capture run 20 Oct. Job `com.bettingrag.true-close` installed 20 Oct (owner); hourly at half past since 21 Oct (the 10:30 run met the 6 h 43 min refresh and deferred); the scorecard shows value at the true close and true price CLV; repaired 22 Oct (D41): an OddsPapi error is retried next hour instead of being recorded as "no price", Betfair is requested alone per outcome, names with Turkish letters or short forms match, the settled kickoff is used, `--recapture` replaces named fixtures' records | Live | About 13 billable requests a week (one fixture list per competition and week) |
| Done 20 Oct: OddsPapi's quota resets on the 1st of each month (docs, Requests & Quota); `/v4/account` reports the count (52 of 250 used by 20 Oct) | Done | |
| Quote age near kickoff in selection (API-Football prices ~2 h old in the last 3 h, D15) | A5.3 (D30) | Apparent edge may be a price no longer offered |
| Retire `ml_edge.py` and `Index/elo_ratings.json` | A7 | Replaced by A6 |
| Evidence-aware variance (`uncertainty.py`, D22) for fitted ratings' uncertainty | A6 | Not needed with today's means |
| Go-live of stage 1: set `SPIX_COUNT_DISPERSION=fixed`. Review pack ready (17 Oct). Golden preview (`Research/stage1-preview-2026-10-17/`): corner picks 33 → 38 (5 added, 10 moved, none removed; the earlier '59' counted probability changes, not picks); picks move to safer lines at shorter prices. Live replay of the corners selector on the audited reads (`ops/stage1_replay.py`, `Research/stage1-replay-2026-10-17/`): floors 162 picks, said 72.2%, won 62.8%, market 64.4%, value at the close −2.5% (−3.3 to −1.6); fixed 177 picks, said 76.8%, won 67.3%, market 67.9%, value at the close −2.1% (−2.9 to −1.2). Same overstatement (~9.5 pts) and no worse at the close, so no reason to hold; both versions lose value on corners (selection, A5) | Live 20 Oct 11:44 UTC (first read with it 12:09 UTC) | Confirm on the live scorecard (split by engine switches) |
| Scorecard splits picks by rollout stage: done 17 Oct (`scorecard.engine_stage`, section "By engine switches"; groups by the prediction-changing switches, so `DOCUMENT_STORE` does not split; picks before 17 Oct count as defaults) | Done | Read it at each stage's scorecard check |
| Done 21 Oct: the test suite pins the prediction switches to their defaults (`Scripts/tests/conftest.py`): with stage 1 in `.env`, two golden tests had started failing; 2,444 pass with no switch forced | Done | Tests stay independent of what is live |
| Done 20 Oct: `golden_snapshots preview --baseline SWITCH=VALUE` replays each fixture under the baseline settings and the candidate's, so a stage is previewed on top of the stages already live; regression `check` still runs with the switches forced off | Done | Use it for stages 3 and 4 too |
| Done 20 Oct 15:39 local: website restarted (owner); `/api/health`, `/`, `/app`, `/api/leagues`, `/api/track-record` answer 200, no errors in its log | Done | |
| Go-live of stage 2: set `SPIX_DEVIG_METHOD=power` (built 17 Oct, off). Golden preview with stage 1 off: no pick added or removed; 4 of 54 fixtures move a pick to a better-priced quote (3 corners, 1 goals; one corners line 9.25 → 9.5), one Asian handicap label medium → low; the fair probability shown changes on 293 results. Preview on top of stage 1 done 20 Oct (`Research/stage2-preview-2026-10-20/`): no pick added or removed, 1 corners pick moved (over 9.25 @ 1.36 → over 9.5 @ 1.50), 3 price only; the fair probability shown changes on 309 of 378 results (median 1.07 pts, largest 5.45 on short-priced favourites); 1 handicap label medium → low. Ready to go live: add `SPIX_DEVIG_METHOD=power` to `.env` | Live 23 Oct 07:52 UTC (`.env` saved; first reads with it 08:26 UTC, after the refresh rerun) | Selection's four margin-removal sites (totals, handicap, BTTS, 1X2) use `data_platform/devig.power` when on |
| Go-live of stage 3: set `SPIX_MARKET_ANCHOR=strict` (built 17 Oct, off). 1X2 and goals O/U 2.5 take the market's consensus (median across books of each book's fair price, with selection's margin removal, so it assumes stage 2 is live; 1X2 rescaled to sum to one); Spix's own probability kept as `_spix_prob` / `spix_prob` and a `market_anchor` record in the result context. Golden preview, stage 3 alone: all 7 1X2 picks in the 54 fixtures become no-bets (at the market's number no price clears the 1X2 thresholds); the only O/U 2.5 pick of 38 goals picks moves to over 2.75, still priced on Spix's numbers (strict scope; stage 4 closes that). Preview on top of stages 1–2 done 23 Oct (`Research/stage3-preview-2026-10-23/`): the same picks change (7 1X2 picks become no-bets; Eredivisie 1552197 over 2.5 @ 1.38 → over 2.75 @ 1.39); BTTS, corners and handicap picks all kept; the O/U 2.5 probability moves in all 54 fixtures (median 5.15 pts, largest 25.64; Spix's over was above the market's in 8 of 54). **Gap closed 23 Oct:** the 1X2 numbers the surfaces showed came from `projection.components`, which keeps Spix's own (the A5.1 study reads it there); the market's are in `context.market_anchor.market`. Now `core/market_result.published_result_probabilities` and `probability_source` read the anchor record, and the Match Read metrics and briefing ("The market's consensus gives …", `source: market` in the facts), the website card's result balance, the web API (`probability_source` on the 1X2 prediction), the text renderer ("1X2 (market)", "Market probability") and the Discord cards ("📊 Market") show the market's numbers when anchored; nothing changes while the switch is off (golden check unchanged, 54 fixtures). **Stage 4 gap closed 24 Oct:** the BTTS yes probability, expected goals and goal difference the surfaces showed stayed Spix's under M1. The service now records M1's figures beside the fit (`market_anchor.goal_figures`, from `market_goal_model.figures`: each team's and the total expected goals, goal difference, BTTS yes), and `core/market_result.published_projection` reads them. They are shown by the Match Read metrics (goals and team-goal split with `source: market`; Spix's season and recent blends left out), the briefing ("The market's consensus projects …"; a total and a split from different sources are never joined), the website card's figures, the web API (`projection_source` and `probability_source` on the totals, BTTS and handicap predictions), the text renderer and the Discord cards ("Projected total (market)", "BTTS Yes (market)", "Goal difference (market)"). Golden fixtures under stage 4 (stages 1–2 on): all 162 goal-market results fitted and published M1's figures (e.g. EPL 1557431, goal difference Spix +0.09, market +0.81). **Found and fixed with it (also a stage 3 gap):** the pick explanation (website card and Discord hub) and the website's Match Read selection called every probability "model probability"; under the anchor they now say "market probability" (`stored_probability_source` on the stored result; the card carries `probability_source`). Nothing changes while the switches are off (golden check unchanged, 54 fixtures). The website labels the balance and figures only with the fair-price card (D36). **At go-live, restart the website** (`com.bettingrag.website`, running since 20 Oct): it builds each card per request with the code it loaded. Then re-record (D16 rule 4) | Live 28 Oct (a day late, owner; see Findings) | Other goal lines, Asian handicap and BTTS stay on Spix's probability until stage 4 (D29) |
| Go-live of stage 4 as a bundle (D33 (b), D35): `SPIX_MARKET_ANCHOR=full`, `SPIX_SELECTION_POLICY=market_value`, `SPIX_BOOKMAKER_SET=romania`, all built 17 Oct and off. The goal-market part (`SPIX_MARKET_ANCHOR=goals`, built 17 Oct, off; `core/market_goal_model.py`). M1 is solved per read from the read's own prices (A3.2 targets: 1X2, O/U 2.5, the main Asian handicap; median of each book's fair price) and prices every goal line, the Asian handicap and BTTS; 1X2 and O/U 2.5 keep stage 3's direct consensus; line scoring and the handicap goal-edge rule use M1's means; Spix's probabilities kept (`_spix_prob`) and the fit recorded in `market_anchor.goal_model`. Golden preview (stages 1–2 off), recommended picks before → after: goals 38 → 5 (34 removed, 4 moved, 1 added), Asian handicap 47 → 3, BTTS 12 → 0, 1X2 7 → 0, corners 33 unchanged. D29 gate passed 17 Oct (D34); ships together with A5.2's selection rule for these markets (D33 (b)) | A3.7 stage 4, earliest Tue 3 Nov | Gate (D29): M1 beats Spix's log loss per market on the audited lines |
| Rerun the D29 gate (`python -m Scripts.ops.m1_gate_study --out Research/m1-gate-<date>`) with the weeks since 17 Oct. Rerun 23 Oct (`Research/m1-gate-2026-10-23/`, 15,606 lines, 295 fixtures, M1 solved on every read): passes in every market again; M1 against Spix's log loss: goals other than 2.5 −12.0%, Asian handicap −10.7%, BTTS −8.9% (all intervals below zero); against the bookmakers' own price: goals −1.4% (−0.0121 to −0.0016), handicap −0.2% and BTTS 0.0% (both intervals include zero). Rerun 28 Oct (`Research/m1-gate-2026-10-28/`, 20,793 lines, 393 fixtures, M1 solved on every read): passes in every market again; against Spix: goals other than 2.5 −10.2%, Asian handicap −9.0%, BTTS −7.5% (all intervals below zero); against the bookmakers' own price: goals −1.4% (−0.0109 to −0.0024), handicap −0.2% and BTTS −0.1% (both include zero). Run once more before stage 4 | Before stage 4 goes live | Passed 17 Oct on one audited week (D34) |
| OddsPapi key: a one-off cache file of 16 Oct (`Index/oddspapi/v4_account.json`, git-ignored, never committed) kept the key that OddsPapi's account answer echoes; it was printed into a working session on 23 Oct and redacted in the file. Owner: issue a new key and replace `ODDSPAPI_API_KEY` in `.env` | Owner | The current client and `ops/true_close.py` do not store account answers |
| Done 23 Oct (D43): Betfair is not carried. OddsPapi has no Betfair sportsbook history open to our plan (`betfair-spb` 403 RESTRICTED_ACCESS; `betfair.es` / `betfair.it` 404 on Serie A, LaLiga and UEL matches, while a clone can have one: Betano RO 264 markets), so `oddspapi.BOOKS` no longer maps Betfair. Its picks keep Pinnacle's true close without an own close; price confirmation turns them down as not carried. 82 published picks in 43 fixtures had an own close from the exchange: recaptured 23 Oct 08:33 UTC from the cache (no billable request; `Research/betfair-recapture-2026-10-23/`): the 814 other picks of those fixtures unchanged; of 88 Betfair picks, Pinnacle's close unchanged on all, the exchange close gone on 82, and 12 now unavailable (Pinnacle has no price on their line); 70 final, 18 unavailable | Done | True close, price check and price confirmation of Betfair picks rely on it |
| Price confirmation before publishing (D40 (a)): **built 23 Oct (D42)**, `services/price_confirmation.py`, run by the Match Read dispatcher on stored reads between evaluation and compilation; acts only on reads made under `SPIX_SELECTION_POLICY=market_value`, so nothing changes before stage 4. **Owner: choose the freshness limit** (`MAX_AGE`, 30 minutes now): on 670 replayed picks of 8–21 Oct it publishes 224 at 30 minutes, 264 at 60, 339 with no limit; a price confirmed from an older record was still there at the bookmaker's next check 83–86% of the time (fresh: 91%) On the stage 4 bundle's own picks (24 Oct replay, 2%): 18 / 22 / 32 published at all books and 4 / 6 / 9 Romanian at 30 / 60 minutes / no limit, of 159 and 40 before confirmation. Rerun 28 Oct with the histories downloaded (9–27 Oct, 1 pick without one): 49 / 58 / 71 at all books and 18 / 22 / 27 Romanian, of 225 and 61, still worth +1.6% / +1.7% / +2.8% and +2.6% / +1.8% / +1.9% at the close (Findings) | Owner, before stage 4 (earliest Tue 3 Nov) | No limit publishes half again as many picks |
| Stage 4 bundle: A5.3 part 1 (D39, corrected in D41): Betano's best prices were real at the read in 56% of checked picks, Pinnacle's 53%; price rules decided (D40): (b) built behind the stage 4 switch, (a) price confirmation before publishing must be built before stage 4 (built 23 Oct, D42; freshness limit to choose); bundle preview with (b), 54 golden fixtures of 17–18 Oct (Betano absent those days), all books: goals 38 → 8, corners 33 → 7, BTTS 12 → 4, 1X2 7 → 3, Asian handicap 47 → 0; Romanian books only: 2, 3, 1, 0, 0; bundle replay on 9–23 Oct done 24 Oct (`ops/stage4_replay.py`, see Findings): at 2% before confirmation 159 published picks at all books, 40 Romanian; after confirmation with no limit 32 and 9 (histories downloaded so far); rerun 28 Oct on 9–27 Oct with histories (Findings): 225 and 61 at 2%, after confirmation with no limit 71 and 27; owner sets the published value threshold (1–5% in the replay; Romania volume in D35) | Before stage 4, earliest Tue 3 Nov | Built 17 Oct (D35) |
| A5.3: part 1 done 20 Oct (D39, `ops/price_check.py`). Exchange commission dropped 23 Oct: the feed's Betfair is the sportsbook (D43). Labels designed 23 Oct with the owner (D44): high when the market and Spix agree, market firmness until Spix earns its vote per market. **Built 23 Oct behind `SPIX_CONFIDENCE_LABELS` (off)**; checks run the same day (finding of 23 Oct): price age supported, bookmaker spread not yet (no cut-off, so every label would be low), Spix check passes for the handicap only (a candidate). Rerun both checks weekly (`ops/confidence_study.py`, `ops/confidence_age_study.py`); owner: whether labels may go on price freshness alone until the spread passes, and whether to approve the handicap's Spix vote after a confirming week. **Rerun 28 Oct** (`Research/confidence-2026-10-28/`, `Research/confidence-age-2026-10-28/`; Findings): the bookmaker spread now has a cut-off that passes on the holdout (0.0159, 24% high; +1.6%, +0.2 to +3.1), though not on its training half (+0.7%, −0.6 to +2.0) and the other way on cards (−3.4%, −5.3 to −1.6); price age supported again (+7.0 pts, +0.5 to +14.3); the Spix check passes in no market, the handicap now +1.9% (−0.0 to +4.1): the confirming week does not confirm its vote | A5.3 (D31, D44) | Half the best-price books' quotes were unchanged or missing at the close (D31) |
| Bookmaker sets per country: Romania first (D35); built, off. Used once Romanian prices are available (D36) | After Romanian prices exist | Since 16 Oct the feed has no Romanian-licensed bookmaker |
| Ask ONJN in writing whether a subscription analysis service that names no bookmaker and takes no commission needs a Class II licence; and whether analysis content counts as gambling promotion under the pending 06:00–24:00 bill | Owner, before launch | Applies to every launch option in Romania |
| OddsPapi (parked until paying users, owner 19 Oct): free-plan check of showing prices inside Spix; written permission request drafted 20 Oct (display to paying members, pre-revenue, B2B later) | Owner sends; then after paying users | Free Romanian prices found 19 Oct (D36) |
| Romanian prices on the card from OddsPapi's free endpoint (fixture mapping exists in `pinnacle_check`; market-id mapping for O/U, handicap, BTTS, corners to build) | After the card, if OddsPapi agrees | Within the free allowance (~50 billable a month) |
| Fair-price card output: per fixture the main lines with fair probability, fair odds and minimum odds worth taking (fair × 1.02), Spix's take beside it; member-facing text in English with a Romanian translation ready to switch on | Next build (D36, D37) | EU-wide launch format |
| EU VAT on subscriptions (One-Stop-Shop, per-country rates) and euro pricing in Stripe; entry-tier price | Owner, before launch (D37) | Selling across the EU |
| Price list on the card from bookmakers licensed in the reader's country: reader country at sign-up or as a Discord role; per-country bookmaker lists confirmed on each regulator's register (UK done); a "checked at" time on every price. API-Football's Bet365 prices are not the UK site's (D39): UK Bet365 prices need another source | Card build, after A5.3 (D38) | Replaces the single engine-wide `SPIX_BOOKMAKER_SET` for display |
| Headline rule for the fair-price card (market decisiveness, line movement since opening, or other) | Owner, with the card build (D36) | Value cannot rank angles without bookmaker prices |
| Odds source for Romanian bookmakers (Superbet, Fortuna, Casa Pariurilor, Unibet, NetBet, MaxBet, WinBet, Mozzart, Betano): OddsPapi carries them; free plan too small for live picks; paid from $15 a month for Superbet RO, $25 with Fortuna RO (football, pre-match, 20 Oct finding) | Owner, 20 Oct: Superbet RO + Betano RO with player props ($50 a month quoted; $35 without props, $95 with live odds), not yet; subscribe when the owner is ready | Since 16 Oct API-Football sends no Romanian-licensed bookmaker (Betano and Betfair dropped) |
| API-Football support: Superbet and Unibet answered 18 Oct (discontinued for /odds since ~Sep 2026); Betano, Betfair, SBO and Dafabet were missing 17–18 Oct and are back (19 Oct partly, 20 Oct: Betano 187 of 193 fixtures); the follow-up email is no longer needed unless they drop again | Owner | Back 19–20 Oct |
| Done 23 Oct: bookmaker coverage watch (`ops/bookmaker_coverage.py`, job `com.bettingrag.bookmaker-coverage` at 09:45, 13:45, 17:45, 21:45 local, installed 23 Oct). Per UTC day, each bookmaker's share of the fixtures kicking off within 48 h in the odds archive's raw responses; alert when today's share is under half its median over the previous 7 days (median at least 20%; at least 20 fixtures today and 3 earlier days). Replay on 7–22 Oct: alerts on 16, 17 and 18 Oct for Betano, Betfair, SBO and Dafabet only. Alerts go to `REFRESH_ALERT_WEBHOOK_URL` (a private Discord webhook), the job log and `Index/bookmaker_coverage/latest.json`. **Owner: the webhook is not set in `.env`, so neither this nor the refresh-blocked alert reaches Discord** (the 23 Oct refresh alert was not delivered) | Done (owner: webhook) | |
| Red cards: the selector also outputs the bets to skip (prices below fair, boosts that are not, high-margin bet builders) | A5.4 (D32) | Part of the Daily Card |
| Bet builder and combination pricing from the score model (correlated legs), anchored to the market | After A5, with the D25 expansion (D32) | Price Check and red cards need it |
| Price Check for singles: the member's odds against the fair price (good price / fair / overpaying) | Next build, after the card (D36) | Needs only the fair probabilities |
| The Ledger (member bet tracking, settlement, overpayment against the fair price) | Product backlog (D32) | Web app work; reuses settlement and reference prices |
| How to show Spix's own view when it disagrees with the market (≥10 pts on 3.4 markets per read; ≥25 pts on half the reads), given A5.1 found it mostly error | Owner, before A5.4 | Product decision (D31) |
| If A5.4 prices corners and cards against the market too, A4's calibrators (stage 5) lose their purpose except where no two-sided price exists. Stage 1 replay (17 Oct): Spix's corner picks lose ~2% at the close under either variance; A5.2's market rule kept +4.2% on corners. Candidate to join the stage 4 bundle (owner) | Decide with A5.4 (D31) | Calibrators fit Spix's own probability |
| A5.4: build the chosen policy behind a switch (`line_selection`, `match_read_compiler`, `confidence_from_edge`), golden preview; proposal in D31 | A5; live at A3.7 stage 6, earliest Tue 17 Nov | Gate (report): published picks' calibration error ≤ 3 pts and closing line value at least neutral over 4+ live matchweeks |
| Rerun the A5.1 study (`python -m Scripts.ops.disagreement_study --out Research/disagreement-<date>`) as weeks accrue | A5, with each A5 step | Rerun 23 Oct on two weeks (`Research/disagreement-2026-10-23/`, 32,855 lines, 295 fixtures): see the finding of 23 Oct on corners. Rerun 28 Oct (`Research/disagreement-2026-10-28/`, 44,196 lines, 393 fixtures): corners again carry information (c +0.38, +0.08 to +0.63); no other market's interval excludes zero |
| Rerun the A4.2 study weekly (`python -m Scripts.ops.calibration_study --out Research/calibration-<date>`); calibrator decision with ≥3 weeks of audited fixtures | A4, decision at A3.7 stage 5 (not before Tue 10 Nov, D29) | Not adopted 16 Oct: one audited week, no market passes the gate (D28). Rerun 23 Oct (`Research/calibration-2026-10-23/`, up to 238 fixtures): Platt passes the gate on goals (−5.77%) and Asian handicap (−3.41%), which leave A4's scope at stage 4 (M1, D29); not on corners (+0.21%) or cards (+0.00%), the markets A4 keeps; BTTS not adopted (389 rows). Fixed corner dispersions still no worse than the floors on counts. Rerun 28 Oct (`Research/calibration-2026-10-28/`, up to 336 fixtures): no market passes; Platt on goals −3.78% (−0.0433 to +0.0031) and the handicap −2.47% (−0.0286 to +0.0011) no longer clear zero, corners +0.14%, cards −0.60%, BTTS −6.40% (542 rows, interval includes zero); Spix's log loss is above the market's in every market |
| Qualify the card settlement definition (yellow + red vs bookmakers' bookings rules, second yellows) before calibrating or scoring cards | A4, before stage 5 | Cards are graded yellow + red (indicative); the market's own card slope is 1.40 on that grading (D28) |
| Cards and goals variance floors (no qualified dispersion yet) | A6 | Cards outside research qualification; goals use the score matrix |
| Market expansion wave 1: team goal totals, double chance, draw no bet (D25) | After A5 | No new modelling; anchored by M1 |
| Market expansion wave 2: team corners totals, corners handicap (D25) | After wave 1 | Team corner projections exist; priced on ~half of fixtures |
| Market expansion wave 3: first-half goals O/U and handicap (D25) | After wave 2 | Needs a half-split model (football-data half-time scores) and storing half-time scores in `fixtures` |
| The n/(n+8) early-season prior on means | A6 | 3.4 changed widths only, not means |
| penaltyblog (MIT licence, Cython) for A6's Dixon–Coles ratings: check time-decay weights, per-league home advantage, promoted-team offset and clean install; use pinned if it covers them, else in-house fit validated against it. **Checked 24 Oct (see Findings): it does not cover them; A6 fits in-house, validated against penaltyblog's Poisson fit and its Dixon–Coles fit with numerical gradients** | A6 start | Not used in A3: margin removal in-house (D17); 3.6 fits rho only, with the engine's own Dixon–Coles code |
| First-round variables (D12): M1 | A3, A8 | Selected 16 Oct |
| First-round variables (D12): S1–S4, C1, D1–D3 | A2b builder, A6 models | Selected 16 Oct; each must pass the ablation gate |
| Second-round variables: M2 line movement | A3 blend | Needs 6–8 weeks of odds archive (from ~mid-Nov 2026) |
| Second-round variables: M3 bookmaker disagreement | A5.3 confidence labels (D30) | With A5 |
| Not selected for now: C2 stakes, C3 derby, L1 lineup strength, L2 player load | Revisit at Part B (L1, L2) or after A6 | Owner chose market, strength/form, discipline/set pieces + C1 |
| Correct `CLAUDE.md`: Elo "signal active at 0.15" (never called) and ML R² 0.43/0.47 (saved R² ≈ 0) | Next docs pass | Stale claims mislead new sessions |
| Done 17 Oct 16:28: production reads from the document store (`DOCUMENT_STORE=sqlite` in `.env`). The first 6 reads record `DOCUMENT_STORE: sqlite` and their projections equal the previous Chroma read of the same fixture and stage (7 of 7 markets each). Verification before the switch: every refresh stage completed; same 129,801 documents; all 881 recorded engine queries return the same documents in the same order; 341 documents differ only in a last floating-point digit; engine outputs on the 54 golden fixtures identical from both. Fallback: remove the line | Done | Unblocks the OddsPapi true-close capture (D16 rule 3, A1 step 4) |

## Decisions and deviations from the report

Newest last. Each: what changed, why (evidence), and what it affects later.

**D1 — No bookmaker trusted by default (14 Oct).** Pinnacle's collected prices are unreliable
since 23 Jul 2025 (football-data.co.uk); the anchor is a measured consensus, weights earned by
freshness and accuracy. Affects A1.4, A3.

**D2 — Historical odds from football-data.co.uk, downloaded by hand (15 Oct).** 140 CSVs, 10
leagues, 2012/13–2025/26, in `Research/football-data/raw/` (git-ignored). Importer
`Scripts/ops/football_data_import.py` builds `Index/football_data.sqlite3` (cf00af8). Betfair
history dropped; no email sent. Pinnacle closing 1X2 every season, full closing set from
2019/20; Pinnacle columns flagged `suspect` from 2025/26. No European cups on the site.
Affects A1.4 (accuracy per book), A1.6 and A3 (backtests).

**D3 — Closing price = API-Football's final pre-kickoff snapshot (15 Oct).** A true kickoff
price is not available from this provider: its last pre-match refresh lands a median 59 min
before kickoff (IQR 28–78, max 3.7 h; 2 h 32 min for the 15 Oct UEL games). The odds stay
available a few days after the match (gone by ~24 days), so they are fetched from kickoff + 2 h.
League/date and per-fixture queries return identical data; the limit is the provider's refresh
cadence, not our query. Every closing record stores `minutes_before_kickoff`. Affects every
CLV figure and A1.4. Raw post-match odds for the backfill: `Index/closing_backfill/` (local).

**D4 — Model version = deployed system version (15 Oct).** `match_reads.model_version` and
`published_recommendations.model_version` were always empty; now filled from
`prediction_system_manifest()` unless a model version is given. Changes on every code edit.

**D5 — Engine determinism fix changed live outputs (15 Oct).** Team-name variants were tried in
hash order; the same fixture could get different probabilities per process (one fixture: 4,857
values). Now a fixed priority order. Golden snapshots replay in a fresh process; bundle in
`Index/golden/` is local only.

**D6 — A1 re-ordered (15 Oct).** The report's "settlement for any selection" fixed nothing
(0 player picks, 0 picks without a period); the real blocker was handicap picks storing the team
name as the side (644 ungraded). New order: grading fix, closing prices (time-critical: odds
expire), publication reference, weights, scorecard, research baseline.

**D7 — Count selections, not publications (15 Oct).** Re-reads republish the same pick:
2,688 publications = 986 distinct selections in the measured period. Figures quoted before the
scorecard (e.g. "2,071 finished picks", "+1.6% CLV", "11.3 pts above market") counted
repeats. The scorecard counts each selection once at its first publication, plus the official
first-match-read cohort.

**D8 — Step 4 split around the weekend (15 Oct).** OddsPapi free tier (250 requests, key
`ODDSPAPI_API_KEY` in `.env`) for ~20 Pinnacle spot checks, which must run before kickoff on
17–18 Oct; freshness report rerun and weights on Mon 19 Oct.

**D12 — First-round engine variables selected (16 Oct).** From the
[variables report](https://claude.ai/artifact/KUTZD7evrLbcBM7BrHdwYV), the owner chose the market,
strength-and-form and discipline-and-set-pieces families, plus C1:
- **M1** market goal expectations per team (handicap, total, team-total prices) → A3 anchor, A8.
- **S1** opponent-adjusted, time-decayed goal ratings with per-league home advantage; **S4** promoted-team
  offset (model structure of A6).
- **S2** xG ratings (xG ~90% from 2023, so tested on 2024 onward); **S3** shots-in-box ratings (2019 onward).
- **C1** rest days and congestion, fitted (replaces the 0.85–1.15 multipliers).
- **D1** referee effect fitted and shrunk to the league; **D2** foul matchup (fouls drawn from 2021);
  **D3** corner pressure (blocked shots + shots in box) replacing the control/dominance indices.
- New inputs per model (budget ≤3): goals S2+S3+C1; SoT S2+S3+C1; cards D1+D2+C1; corners D3.
  M1 enters through A3's anchor, not as a model input.
- Second round: M2 (needs archive history), M3 (with A5). Every variable passes the gate in the
  report: ablation against the market baseline, ≥0.5% log-loss gain, interval below zero, no slice
  worse by >2%; a failure is removed and recorded here.

**D13 — A6 before A2b's legacy retirement (16 Oct).** Owner agreed. A6's fitted models use raw
stats (D12: S2, S3, D2, D3), so A2b only has to build those raw inputs with an as-of cutoff; it no longer
recomputes the legacy control, aggression and form indices. The legacy pipeline (per-league scripts,
`Output/`, normalise) keeps feeding the document store until A6 replaces the hand-set projections, then
A2b retires it. Order: A1 step 4 → A3 → A4/A5 → A6 (with the D12 variables) → A2b → A7.

**D14 — Where ML returns (16 Oct).** The old layer (58 hand-made features → LightGBM/Ridge predicting
raw totals, saved R² ≈ 0) is not revived. ML comes back as fitted models that must beat the market
baseline: A3 logistic blend of market and Spix (scikit-learn); A4 calibration (Platt, isotonic from ~1,000
settled); A6 fitted ratings with the D12 variables (penaltyblog Dixon–Coles MLE with time decay; PyMC or
NumPyro hierarchical counts); **A6c (new): a LightGBM challenger predicting the market's error (not the
total) from the D12 variables only, time-ordered cross-validation, kept only if it beats the fitted A6
model against the market baseline; built after A6's base model passes**; Part B player models (GLMs first,
boosting as challenger). `ml_edge.py` and `Index/elo_ratings.json` retire at A7.

**D15 — Step 4 evidence: accuracy is flat, freshness is the issue (16 Oct).**
- *Accuracy* (football-data closing 1X2, 2019–2025, `Research/bookmaker-accuracy-2026-10-16/`): against
  the other books' consensus, Betfair Exchange −0.10% and Pinnacle −0.08% log loss (both intervals below
  zero); Bet365, William Hill, Bet&Win, Interwetten, Coral, Ladbrokes +0.06 to +0.14% (worse). Total spread
  ≈ 0.25%, so accuracy weighting barely moves the reference price.
- *Freshness* (`Scripts/ops/pinnacle_check.py`, OddsPapi free history, 3,417 archived Pinnacle captures,
  141 fixtures, 12 billable requests): within 1 pt of Pinnacle's real price 90% when >24 h out, 72% at
  3–24 h, **43% in the last 3 h**; median age of our price in the last 3 h **130 min**. The plan's rule
  (≥90% within 1 pt) fails near kickoff: API-Football is a delayed copy (its refresh cadence), and this
  applies to every book we get from it.
- *Closing*: our API-Football "closing" vs Pinnacle's true close (OddsPapi, 95 fixtures): median gap
  0.95 pts (p90 2.7), 52% within 1 pt; ours a median 60 min before kickoff, the true close 1.6 min.
- *OddsPapi free plan*: 250 requests; historical odds free (since Jan 2026, ≤3 books per call, 5 s
  cooldown); fixture lists 1 request each; Pinnacle and all ten of our live books available.
  Key `ODDSPAPI_API_KEY`; cache `Index/oddspapi/`.

**D16 — Rules for starting A3 out of order (16 Oct).** A3 starts before A1 step 4 (weights, Monday) and the
A2a production switch (17 Oct). To keep the order break harmless:
1. *Frozen gate:* A3 is judged against `market-baseline.v1` (football-data closing history), which neither
   the bookmaker weights nor a change of closing source can move.
2. *Versioned reference:* every A3 fit records the `reference_price` method version it used and is refitted
   when that version changes (weights, OddsPapi close, a new margin-removal method).
3. *One production change at a time:* A2a switches first; A3 stays research and shadow, behind switches that
   default off, until A1's gate is closed. With the switches off, the golden snapshots must stay identical.
4. *Golden re-records:* only after a market-by-market review of the change summary, each logged here with its
   reason.

**D17 — Power margin removal (16 Oct, A3.1).** `Scripts/data_platform/devig.py` (proportional, additive,
power, Shin; written in-house, no penaltyblog dependency); `Scripts/ops/devig_comparison.py`; results in
`Research/devig-comparison-2026-10-16/`. Football-data closing history, 25,057 fixtures, 291 weeks, median
across books: 1X2 log loss vs proportional — power −0.075% (interval below zero), additive −0.074%, Shin
−0.064%; calibration error 0.0083 → 0.0036 with power (proportional overstates longshots). O/U 2.5: all
within −0.02% (power's interval touches zero). Power chosen for all markets. `reference_price` gains
`method=` with versions (`power_devig_median.v2`); default stays proportional v1 until go-live (D16).

**D18 — Closing source: OddsPapi true close (16 Oct).** Owner approved. Closing prices come from
Pinnacle's last pre-kickoff price in OddsPapi's free history (optionally two more books, ≤3 per call),
with API-Football's final snapshot as fallback when a fixture can't be matched. Budget: one fixture list per
league per week (~50 billable requests/month); historical calls confirmed free (count stayed at 14 after
~141 history calls). Monthly reset of the 250 limit is a vendor claim, unconfirmed; even without it the
remaining 236 last over four months. Built with step 4 on Monday; live after the A2a switch; CLV and
scorecard record which source each closing price came from.

**D19 — M1 beats the engine's goal means (16 Oct, A3.2).** `Scripts/data_platform/market_goals.py` solves
each team's goal mean (independent Poisson, ≤12 goals) so the model reproduces the market's fair 1X2, total
and Asian handicap probabilities (exact zero-profit fair price for quarter lines). `Scripts/ops/market_goals_study.py`,
`Research/market-goals-2026-10-16/`: on 2023 (2,783 fixtures, 41 weeks; 312 without prices, mostly European
cups), M1 from closing prices (power, median across books) scores actual home and away goals **2.27% better**
(Poisson log loss) than the frozen control's means (95% −0.078 to −0.056); median fit error 0.0004. Closing
prices are the best case: publication-time prices (A3.3) will be weaker. Dixon–Coles rho not applied (A3.6).

**D20 — The blend gate is not passed: Spix adds nothing to the market for 1X2/O/U (16 Oct, A3.3).**
`Scripts/data_platform/blend.py` (logarithmic pool, exact gradient), `Scripts/ops/blend_study.py`,
`Research/blend-2026-10-16/`. Frozen 2023 control forecasts; market at publication = football-data
pre-closing prices (power, median); time-ordered folds (fit on earlier quarters), 1,677 out-of-sample
fixtures, 29 weeks. Blend vs market alone: 1X2 +0.05% (interval includes 0), O/U 2.5 **+0.41% worse**
(interval above 0). Blend vs market with c = 0: +0.02% / +0.04% (no gain). Fitted c unstable across
folds (1X2 −0.02 → 0.19; O/U −0.09 → 0.07). Recalibrating the market alone also hurts O/U (+0.37%).
Spix alone is 4.3% / 1.8% worse than the market; closing beats the blend by 0.6–0.8%. Consequence: for
these markets the honest A3 output is the market's probability (power, c = 0); Spix's model must earn a
weight in A6, checked by rerunning this study. Corners, cards, SoT and BTTS have no historical prices;
their blends wait for our own archive.

**D21 — Interim rule for 1X2 and O/U 2.5: the market with Spix's weight at zero, until A6 (16 Oct).** Owner
decision on D20. **This applies only before A6.** When A3 goes live (after A1's gate, D16), the published
probability for 1X2 and over/under 2.5 is the market's fair probability (power margin removal, reference
consensus) with Spix's weight c = 0; Spix's own probability is still computed, stored and shown as a
disagreement where it differs (A5). At A6's gate the blend study is rerun on A6's forecasts: if Spix earns a
weight (blend beats the market alone and the c = 0 recalibration on unseen weeks), c is refitted and this rule
ends; if not, it is reviewed again with the owner. Other markets are unaffected by this rule.

**D22 — Variance: the floors are too wide; the fixed dispersion is enough (16 Oct, A3.4).**
`Scripts/data_platform/uncertainty.py` (Var = μ + μ²(α + (1+α)/(κk)), k = the engine's effective matches),
`Scripts/ops/uncertainty_study.py`, `Research/uncertainty-2026-10-16/`. Frozen control corners/SoT forecasts
joined to their snapshots; fitted on 2022 (≈3,600), evaluated on 2023 (≈3,030). Against today's floors:
fixed Phase 4 dispersion −1.03% corners, −1.41% SoT (intervals below zero, every season stage);
evidence-aware variance −0.96% / −1.38%; evidence vs fixed +0.07% / +0.04% (no difference, even at 0–7
matches; the fit sets α ≈ 0 and κ ≈ 4.6–6.7, reproducing the fixed dispersion). Decision: 3.4's evidence
term is not adopted (simplest model wins, variables-report rules); 3.5 replaces the floors with the fixed
dispersions. Means, including the n/(n+8) prior, are unchanged.

**D23 — Fixed dispersions for corners and SoT in every competition and stage (16 Oct, A3.5).**
`core/prediction_guardrails.py`: with `SPIX_COUNT_DISPERSION=fixed`, corners and SoT totals use
Var = μ + αμ² (α 0.025 / 0.01, `count-dispersion.v1`) instead of the floors; cards and goals unchanged; the
floor confidence cap no longer applies to them. Default off (D16). Scope widened beyond Phase 4's qualified
scope (five leagues, ≥8 matches) because on 2023 every competition and season stage improved or was flat
against the floors (corners −0.4 to −1.8%; SoT −0.01 to −2.9%), none worse; this is development-period
evidence, so the live scorecard confirms it after go-live. Golden preview with the switch on: only corners
(52,131 values, 59 pick-level changes) and SoT (444 values, no pick changes) differ; switched off, the
bundle is identical.

**D24 — Rho stays at −0.10 (16 Oct, A3.6).** `Scripts/data_platform/dixon_coles_rho.py` (maximum
likelihood within the valid range, same formula as `prob_models`), `Scripts/ops/rho_study.py`,
`Research/rho-2026-10-16/`. Frozen control goal means, actual scorelines; fit 2022 (3,645), test 2023
(3,095). Pooled fitted rho ≈ −0.006; per-league values −0.14 to +0.19 (noise). Against −0.10: pooled
−0.066% (interval −0.0038 to +0.00003), per league −0.009%, per league shrunk −0.052%: all below the
protocol's 0.5% bar and not reliably better; some slices slightly worse (Bundesliga +0.26%). Inconclusive,
so the control value stays. Rho can be refitted jointly with A6's ratings, where it belongs.

**D25 — Market expansion, planned after A5 (16 Oct).** Owner asked to add it. Spix publishes 7 of the
114 market types our feeds price (odds archive, last 165 fixtures). Each new market needs settlement rules
(`settlement.MARKET_KEYS`), closing capture and its own scorecard slice, and is published only after the
scorecard shows it holds up (CLV, calibration). Waves, in order:
1. **Team goal totals** (home/away; 7 books, all fixtures), **double chance** (8), **draw no bet** (6):
   from the score matrix and M1, no new modelling; settle from the final score.
2. **Team corners totals** (4 books), **corners Asian handicap** (4), corners 1X2 (3): priced on 78 of 165
   fixtures; engine already projects team corners; settle from `fixture_team_stats`.
3. **First-half goals O/U** (9 books), **first-half handicap** (7), HT/FT (5): needs a half-split model fitted
   on football-data half-time scores (25,055 linked matches); API-Football fixture payloads include the
   half-time score (verified in `raw_payload_archive`) but `fixtures` does not store it yet.
The angle: use the sharp main-market information (M1, power margin removal) to judge how soft books price
these side markets, rather than Spix's own model (D20). Not planned: exact/correct score, odd/even, minute
windows, first/last scorer team, offsides and fouls (7–13 of 165 fixtures, 1–2 books).

**D26 — A3.7 go-live: staged schedule and strict scope (16 Oct).** Owner approved.
- *Schedule* (one production change at a time, D16): Fri 17 Oct A2a switch after the 07:15 refresh check,
  package build starts; Mon 19 Oct A1 closed and package ready (each stage previewed on the golden fixtures,
  changes reviewed); **stage 1** Tue 20 Oct fixed dispersions (corners, SoT); **stage 2** Fri 23 Oct power
  margin removal in pick selection; **stage 3** Tue 27 Oct market anchor; Mon 2 Nov weekly scorecard confirms
  all three. A few days per stage is a safety check: the gains were proven on history, and a week of live picks
  (~300) can catch a fault, not prove a 1% gain. If a check fails, the following stages move back.
- *Strict scope for stage 3:* the market's fair probability with Spix's weight c = 0 (D21) replaces Spix's
  probability **only for 1X2 and over/under 2.5**, the markets history tested. Other goal lines, Asian handicap
  and BTTS keep today's behaviour until a scorecard check after stage 3; extending them through M1's score
  matrix is a separate, later decision. Spix's own probability is still stored for the disagreement view.
- *Prerequisite:* every pick records which engine switches were active, because the model version fingerprints
  code files only.

**D27 — A4 calibration: scope and timing (16 Oct).** Owner asked to start A4. Conditions so it does not
collide with A3.7:
1. *Scope:* only markets where Spix's own probability is published: corners, cards, BTTS, goal totals other
   than 2.5, Asian handicap. Not 1X2 or O/U 2.5 (market-anchored, D21; recalibrating the market hurt, D20).
   SoT has almost no priced candidates yet.
2. *Fit on the probabilities that will be live:* corners (and SoT) calibrators use probabilities recomputed
   with the fixed dispersion of stage 1 (from each result's stored mean), not today's floors.
3. *Go-live as A3.7 stage 4,* after stage 3, not before the Mon 2 Nov scorecard (one change at a time).
   (Moved to stage 5 by D29.)
Data: decision audits since the 6 Oct release hold every evaluated candidate (model probability and exact
settlement profile) for 241 fixtures; lines within a fixture are correlated, so Platt scaling per market
first; isotonic only once a market has ~1,000 settled forecasts. Expectation: the published picks'
calibration slope of 0.42 is mostly selection (winner's curse, A5's job); calibrating all candidates fixes
the model's own miscalibration, so A4 alone will not move that slope much. Phase 4's history also found
small calibration gains.

**D28 — A4.2: no calibrator adopted yet (16 Oct).** Method: Platt scaling per market on one reference
side per line (over, home team, yes; a fit on both complementary sides forces the intercept to zero and
cannot see a directional bias), corners priced with the stage 1 fixed dispersion, each kickoff day
predicted by a fit on the days before it (5 test days, ~100 fixtures), intervals resample fixtures;
isotonic as a challenger. Code `data_platform/calibration.py`, `ops/calibration_study.py`; output
`Research/calibration-2026-10-16/`. Out of sample, no market passes the D27 gate (error ≤ 3 pts, slope
0.8–1.2) after Platt:

| Market | Tested rows (fixtures) | No calibration: error / slope | Platt: error / slope | Platt log loss, 95% |
|---|---|---|---|---|
| Goals (not 2.5) | 3,128 (103) | 15.8 / 0.77 | 8.8 / 0.83 | −10.3% (−0.102 to −0.021) |
| Asian handicap | 2,939 (103) | 7.4 / 0.65 | 3.4 / 0.79 | −4.3% (−0.050 to +0.002) |
| Corners | 4,390 (99) | 5.7 / 0.65 | 5.1 / 0.68 | +0.6% (−0.013 to +0.020) |
| Cards | 1,209 (101) | 4.6 / 1.12 | 4.4 / 0.94 | +1.0% (−0.021 to +0.031) |
| BTTS | 156 (103) | 15.8 / −0.65 | 7.1 / 0.39 | −14.6% (−0.232 to +0.007) |

Isotonic is worse than Platt in every market. Decision: nothing goes live; rerun weekly and decide at
stage 4 (not before Mon 2 Nov) with at least three audited weeks. The same data show why calibration
alone cannot close the gap. On lines priced two-sided, Spix is worse than the bookmaker's de-vigged
price in every in-scope market: log loss +8.7% corners, +10.2% BTTS, +10.3% AH, +10.8% cards,
+18.2% goals. The intervals exclude 0 except AH (−0.0003 to +0.128). Spix's slopes are 0.42–0.68
against the market's 1.02–1.13 (cards 1.40, possibly because yellow + red grading differs from the
books' card rules; not verified). The selector's own picks stay badly calibrated after Platt (slopes
0.16–0.48), as D27 expected: that is selection, which is A5's job.

**D29 — Goals market anchor brought forward to stage 4 (16 Oct).** Owner decision after D28 (Spix is
18.2% worse than the market on goal lines other than 2.5, and its goal means run low). The D26
extension becomes **stage 4**, after stage 3 and ahead of any A4 calibrator. It is the market anchor for
all goal markets through M1's score matrix: O/U lines other than 2.5, Asian handicap and BTTS, with c = 0,
and Spix's own probability kept for the disagreement view.
- *Build* in the A3.7 package behind its own switch, off by default, and previewed on the golden fixtures.
  `market_goals` has 1X2, handicap and totals today; BTTS from the score matrix is added.
- *Gate before go-live.* History cannot test these markets: football-data holds only 1X2, Asian handicap
  and O/U 2.5, and M1 already takes the main handicap and O/U 2.5 prices as inputs. So the check runs
  on the audited lines (A4.1 dataset). In each market, M1's probabilities must beat Spix's on log loss,
  with the fixture-resampled interval below zero. They are also reported against the bookmaker's own
  de-vigged price on the same lines.
- *Schedule.* Stage 4 goes live earliest Tue 3 Nov, after the Mon 2 Nov scorecard confirms stages 1–3,
  and is checked at the Mon 9 Nov scorecard. If an earlier stage fails its check, stage 4 moves back with
  it (D26).
- *Effect on A4.* The calibrators become **stage 5**, earliest Tue 10 Nov. Once stage 4 is live, goal
  lines, Asian handicap and BTTS publish market-anchored probabilities and leave A4's scope, as 1X2 and
  O/U 2.5 did (D27 rule 1). A4 then covers corners and cards, plus SoT once it is priced. If stage 4 fails
  its gate, those markets return to A4's scope; the weekly A4.2 reruns keep them until then.

**D30 — A5 research plan, and A5.1: Spix's disagreement is almost all error (16 Oct).** Owner asked to
start A5 research while A3.7 rolls out. Plan:
- A5.1 (done): is Spix's disagreement with the market information or error?
- A5.2: simulate selection policies on the audited candidates.
- A5.3: evidence-based confidence labels (sample size, uncertainty width, bookmaker disagreement M3, quote
  age D15), checked against closing line value and calibration.
- A5.4: build behind a switch, live as A3.7 stage 6 (earliest Tue 17 Nov, after stage 5's scorecard).

A5.1 method (`data_platform/disagreement.py`, `ops/disagreement_study.py`, output
`Research/disagreement-2026-10-16/`): every line the engine evaluated in the latest pre-kickoff read
per fixture and stage, 16,009 lines across 160 fixtures (9–15 Oct). Each line is taken on the side Spix
prefers, and 1X2 keeps all three outcomes, with Spix's probabilities from the projection, since its
candidates store none. The market is the power-de-vigged median across bookmakers at the read; the close
is API-Football's final pre-kickoff snapshot (114 fixtures). Two side-symmetric tests, with intervals
that resample fixtures:
- *Outcomes:* the weight results give Spix's deviation from the market is c = +0.03 (95% −0.17 to
  +0.22). No market is above 0: corners +0.20, AH +0.15, 1X2 +0.05, goals −0.18, cards −0.45,
  BTTS −0.46; every interval includes 0.
- *Closing:* from pre-match reads (median 10.7 h before kickoff) the market closes k = +0.029 (+0.014 to
  +0.044) of Spix's deviation, i.e. about 3%: real but small. Lineup reads come ~20 min before kickoff
  and show no movement (k ≈ 0).

At every size of disagreement, results followed the market:

| Spix above the market by | Lines | Spix said | Market said | Happened |
|---|---|---|---|---|
| 0–2.5 pts | 3,173 | 50.0% | 48.8% | 48.7% |
| 2.5–5 pts | 2,646 | 44.5% | 40.7% | 39.3% |
| 5–10 pts | 4,020 | 47.1% | 39.8% | 36.5% |
| 10–15 pts | 2,423 | 50.4% | 38.2% | 38.1% |
| 15+ pts | 3,484 | 59.5% | 35.4% | 36.2% |

The selector's own choices (846 lines) show the same thing: above 15 pts, Spix said 75.0%, the market
49.5%, and 52.5% happened. So ranking by edge ranks the engine's error, which is the report's A5
premise, now measured live. For A5.2 this means pricing against the market consensus, showing large
disagreements as "disagreements to check" rather than picks, and allowing Spix's view at most a small
tilt (c and k ≈ 0.03) until A6. Limits: one week; returns at the best price are saved but are too
noisy to read (every line of a fixture settles on the same score).

**D31 — A5.2: price against the market, not against Spix (16 Oct).** Method
(`data_platform/selection_policies.py`, `ops/selection_study.py`, output `Research/selection-2026-10-16/`):
every policy is replayed on every side of every evaluated line in the latest pre-kickoff read per fixture
and stage (160 fixtures, 9–15 Oct). Each side gets the policy's probability; a side is a candidate when
its expected value at the best price on offer is at least 2%; one pick per market, three per fixture.
`current` replays the engine's own selector choices, ranked by Spix's expected value (the compiler's exact
order is not replayed). Pre-match reads (median 10.7 h before kickoff):

| Policy | Picks (fixtures) | Stated | Happened | Happened − stated, 95% | Value at the close, 95% |
|---|---|---|---|---|---|
| Current selector | 459 (158) | 73.3% | 59.6% | −13.6 (−18.5 to −8.8) | −1.4% (−1.9 to −0.9) |
| Spix's expected value | 478 (160) | 31.5% | 14.3% | −17.2 (−20.4 to −13.7) | −25.2% (−27.1 to −23.3) |
| Same, disagreement capped at 10 pts | 473 (160) | 33.0% | 23.7% | −9.2 (−12.6 to −5.9) | −24.1% (−26.4 to −21.6) |
| Market price (c = 0) | 300 (150) | 72.2% | 69.1% | −3.1 (−8.6 to +1.7) | **+3.5% (+2.8 to +4.3)** |
| Tilt c = 0.03 (A5.1's fit) | 303 (150) | 70.6% | 68.3% | −2.3 (−7.0 to +2.7) | +3.4% (+2.6 to +4.3) |
| Tilt c = 0.10 | 354 (154) | 57.3% | 54.6% | −2.8 (−7.8 to +2.4) | +2.1% (+1.1 to +3.1) |
| Tilt c = 0.25 | 436 (158) | 44.3% | 37.4% | −6.9 (−11.4 to −2.5) | −1.1% (−2.4 to −0.0) |

"Value at the close" is the closing consensus probability times the price taken, minus one, so it does not
depend on results. The current selector's overstatement matches the scorecard (−12.8 pts, A1). Market
pricing at a 4% minimum: 96 picks, +5.8% (+3.8 to +7.8); at 0%: 468 picks, +2.5%. On lineup reads
(~20 min before kickoff; no closing value possible) market pricing overstated by 8.4 pts (−16.1 to −1.2,
115 picks, 51 fixtures), which is noisy and must be watched.

**The value is price shopping, so A5.3 must first check that the prices were real.** For the market-price
policy, the best price came from two-sided quotes in 296 of 300 picks, mostly Betano (156) and Pinnacle
(63; our Pinnacle copy is a median 130 min stale, A1.4). At the close, the best-price book's own price was
shorter in 71 picks, the same in 77, longer in 32 and not quoted in 29. Picking the highest price
regresses under every policy (the current selector: 115 shorter, 70 longer), so "shorter" proves little,
and unchanged quotes may be stale (D15).

**Proposal for A5.4 (owner decides after A5.3).** Publish the market's consensus probability (power, median;
a tilt of at most c = 0.03 until A6 earns more). Pick where the best available price beats it by at least
2%, one per market and three per fixture. Show Spix's own view as a disagreement, not use it to pick. This
extends stages 3–4's c = 0 to every market. Spix is ≥10 pts above the market on 3.4 markets per pre-match
read, ≥20 pts on 1.4 (114 of 160 reads), ≥25 pts on 0.86 (82 reads) and ≥30 pts on 0.51 (53 reads), so how
to show it is a product decision.

**D32 — Product direction for the engine (17 Oct).** Owner and Claude agreed the offer, pinned as "What Spix
Sells" (https://claude.ai/artifact/FwVJYpWojqaGL8t7nDXG9X). The promise is "Know the real odds. Never
overpay." Members get the Daily Card (green to take, red to skip), the Match Read with Ask Spix, and Price
Check with the Ledger. The year-one goal is €5,000 a month. What it means for the engine:
- *Keep building and keep trying ML* (A6, A6c, Part B). The rule is unchanged: a model sets the published
  probability only in markets where it beats the market on the scorecard; otherwise it shows as "Spix's
  take". The likeliest edge is in thin markets: player props, corners, cards and bet builders.
- *A5 gains requirements:* best prices only from books available in the member's country; prices fresh
  enough to bet (D15); and red cards, the bets to skip, as a second output of selection.
- *New engine work after A5:* pricing bet builders and other combinations from the score model, anchored to
  the market, for Price Check and red cards. Price Check for singles and the Ledger are web-app work that
  reuses settlement and the reference prices.

**D33 — Stage 4 leaves almost no goal-market picks until A5 (17 Oct, owner to decide).** With the market's
probability in every goal market but today's selection thresholds (edge against the quoting book's own fair price of
at least 2 points, minimum probability, a minimum goal edge for handicaps), the golden preview publishes 8 goal-market
picks across 54 fixtures instead of 104; corners (33) stay on Spix's numbers. Most removed picks were short-priced
lines Spix rated far above the market (median stated probability: goals 0.79, handicap 0.81). Between stage 4
(earliest 3 Nov) and A5's selection rule (stage 6, earliest 17 Nov) the Daily Card would therefore lean on corners,
the market where Spix is least calibrated (D28). Options for the owner: (a) accept two thin weeks; (b) bring the A5.2
rule (price against the consensus at the best available price) forward to go live with stage 4, as one change; (c)
hold stage 4 until A5.4. Decide after the D29 gate and A5.3 (are the best prices real?).
**Owner decision (17 Oct): (b),** subject to A5.3 confirming the best prices were real. The A5.2 rule for the
anchored goal markets is therefore built before stage 4 and ships with it as one change; A5's other parts (labels,
bookmaker sets per country, red cards) stay at stage 6.

**D34 — D29 gate passed: M1 beats Spix in every goal market stage 4 would price (17 Oct).** Method
(`data_platform/m1_gate.py`, `ops/m1_gate_study.py`, output `Research/m1-gate-2026-10-17/`): for every audited
read (latest before kickoff, per fixture and stage; 167 fixtures, 7,927 lines), M1 is solved again from the
price lists the decision audit stored, with the engine's own `market_goal_model.solve` (solved for every read,
median fit error 4.6e-4). Each evaluated line is taken once; Spix's probability is the one the engine computed,
the market is the power-de-vigged median across books. Intervals resample fixtures.

| Market | Lines (fixtures) | Log loss Spix / M1 / market | M1 vs Spix, 95% | M1 vs market, 95% | Calibration error / slope, Spix → M1 |
|---|---|---|---|---|---|
| Goal lines other than 2.5 | 4,342 (167) | 0.547 / 0.470 / 0.475 | −14.0% (−0.117 to −0.040) | −1.1% (−0.013 to +0.002) | 13.1 / 0.75 → 5.2 / 0.93 |
| Asian handicap | 3,358 (167) | 0.610 / 0.540 / 0.541 | −11.6% (−0.128 to −0.017) | −0.2% (−0.006 to +0.003) | 13.8 / 0.82 → 1.9 / 1.06 |
| Same, without the line M1 is fitted to | 3,139 (167) | 0.597 / 0.530 / 0.531 | −11.2% (−0.123 to −0.014) | −0.3% | 13.3 / 0.83 → 1.6 / 1.06 |
| BTTS | 227 (167) | 0.755 / 0.691 / 0.691 | −8.5% (−0.110 to −0.021) | +0.0% (−0.009 to +0.008) | 13.9 / −0.17 → 6.6 / 0.22 |

Pre-match reads alone pass in every market (−12.3%, −10.6%, −7.7%); lineup reads (60 fixtures) point the same way
with wider intervals. M1 prices lines it was not fitted to as well as the bookmakers' own consensus on those
lines, so the matrix extrapolates coherently. One audited week (9–15 Oct plus fixtures finished since); rerun
before stage 4 goes live.

**D35 — Stage 4 selection rule choices, and Romania's bookmaker gap (17 Oct).** Owner chose, for the
market-priced markets in the stage 4 bundle:
1. No minimum probability, as A5.2 tested.
2. Drop the Spix-based filters: the edge against the quoting book's own fair price, and the handicap minimum goal edge.
3. Rank picks within a fixture by value against the market, and price corners against the market in the same bundle.
4. First country: **Romania**. Picks only at bookmakers a Romanian bettor can legally use; the consensus keeps
   every book (more books, sharper consensus).

Finding (ONJN register of Class I licensees, onjn.gov.ro/licentiati-clasa-i, fetched 17 Oct). Of the 10 bookmakers
in our API-Football feed, only **Betano** (ro.betano.com) and **Betfair** (betfair.ro, sportsbook and exchange) are
licensed in Romania. Bet365's rights were annulled in 2015. Pinnacle, William Hill, BetVictor, Marathonbet, 1xBet,
Dafabet and SBO are not on the register.

API-Football's catalogue includes Superbet, Unibet, NetBet and 888Sport, but none of them appeared in its odds for
our fixtures since 16 Oct (raw archive). Betano appeared for 176 and Betfair for 159 of about 335 fixtures.

OddsPapi lists Superbet RO, Fortuna RO, Casa Pariurilor RO, Betano RO (marked as a clone of Betano's feed), MaxBet
RO, WinBet RO, Mozzart, NetBet, Unibet, 888Sport and the Betfair Exchange. Its free plan (250 requests a month)
cannot feed live picks.

Consequence: until a source for Romanian bookmakers is added, Romanian members' picks can only use Betano and
Betfair. Owner decision needed: the odds source for Romanian bookmakers.

*Update, same day.* Correction: across the whole odds archive (356 fixtures since 6 Oct) Betano is present for 324
fixtures and Betfair for 306; the "about half" above came from a partial sample. The owner set the Romanian book list
to **Betano and Superbet**. Superbet has no prices in our feed: none in any archived API-Football response since
6 Oct or any closing snapshot; asking API-Football explicitly for bookmaker 34 on 10 upcoming fixtures returned
none; the last Superbet prices were in a few UCL reads on 8–9 Sep. It counts as soon as a source provides it.

*Update, evening.* **API-Football stopped sending four bookmakers on 16 Oct around 09:00–15:00 UTC.** Across all
fixtures captured from 14 Oct to 16 Oct 15:00 UTC (213 fixtures) versus since (194 fixtures), appearances fell:
Betano 3,042 → 92, Betfair 2,889 → 97, SBO 2,670 → 54, Dafabet 2,582 → 60; Bet365, Pinnacle, 1xBet, William Hill,
Marathonbet and BetVictor are unchanged. For the 54 golden fixtures Betano went 54 → 43 → 4 → 0 within six hours
and has not returned. Published picks priced at Betano fell from 62–240 a day (6–15 Oct) to 17 on 16 Oct and
fewer than 10 on 17 Oct. Superbet and Unibet also came from API-Football earlier, which its support confirmed on 18 Oct
(coverage discontinued from about September 2026); the 8 Sep Superbet quote carries our API-Football corners key
(`totals_corners_over_under`). The Odds API (added 8 Feb, removed 9 Sep 2026) is not needed to explain them. Consequences: today
the feed carries no Romanian-licensed bookmaker; the bundle preview above uses 15 Oct prices, when Betano was
present; A5.2's best prices (Betano in 156 of 300 picks) are no longer flowing. Owner actions: ask API-Football
support whether Betano, Betfair, Superbet and Unibet coverage is coming back; the second odds source (OddsPapi)
becomes a dependency for Romania, not an option.

*Update, 18 Oct.* **API-Football support confirmed** that Superbet (34) and Unibet (16) coverage was discontinued
for the /odds endpoint across all our competitions (EPL, La Liga, Serie A, Bundesliga, Ligue 1, Championship,
Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, UCL, UEL, UECL) from about September 2026; both stay listed
in /odds/bookmakers without returning fixture data. So the catalogue does not show what is actually covered, and
Superbet prices for Romania need another source (after launch, D36). Betano, Betfair, SBO and Dafabet: still absent
at 10:26 UTC on 18 Oct (Betano 0, Betfair 0 in that day's captures; 16 and 20 on 17 Oct); support's answer on
these four is pending.

**Stage 4 bundle built (17 Oct, all off).** `SPIX_MARKET_ANCHOR=full` adds corners at each line's consensus to
stage 4; `SPIX_SELECTION_POLICY=market_value` applies the A5.2 rule to market-priced lines (eligible when the line
has a two-sided market price, the bookmaker is in the active set and the value against the market is at least 2%;
best line per market by value; the compiler ranks market-priced picks first by value, Spix-priced picks after);
`SPIX_BOOKMAKER_SET=romania` (`core/bookmaker_sets.py`) publishes picks only at Betano and Superbet, while the
consensus keeps every book. Golden preview with stages 1–4 on, recommended picks across 54 fixtures:

| Market | Today | Bundle, all books | Bundle, Romania set |
|---|---|---|---|
| Goals | 38 | 9 | 2 |
| Asian handicap | 47 | 20 | 0 |
| Corners | 33 | 7 | 3 |
| BTTS | 12 | 4 | 1 |
| 1X2 | 7 | 3 | 0 |
| Total | 137 | 43 | 6 |

Betano prices every market in all 54 fixtures, so the thin Romanian card is not missing data: one soft bookmaker
rarely beats the sharp consensus by 2% after its own margin. With Betano alone, the threshold sets the volume:
6 picks on 5 fixtures at 2%, 17 on 14 at 1%, 31 on 23 at 0% ("at least a fair price"). Only the 2% rule was tested
(A5.2, all books: 0% kept +2.5% at the close on 468 picks). Owner decisions before go-live: the Romanian odds
source (more books, more value found) and the published threshold.

**D36 — Romania launches on a fair-odds card; no bookmaker prices needed (17 Oct, owner).** No paid odds source
before launch (too expensive pre-launch), and since 16 Oct API-Football sends no Romanian-licensed bookmaker
(D35). So Spix publishes the fair price instead of naming a bookmaker: for each angle the market's fair
probability, fair odds and the minimum odds worth taking (fair odds × 1.02, the 2% value threshold A5.2
tested), for example "fair 1.85; worth it at 1.89 or better". Members check their own Superbet or Betano app.
The same numbers power Price Check (a member enters the odds their app shows: good price, fair, or overpaying)
and red cards ("don't take this below 1.70"). Spix's own take is shown beside the fair price, labelled as such.
- *Engine:* the fair probabilities come from the market anchors already built (stage 3 consensus, stage 4 M1 for
  goal markets, corners at the line consensus), from every bookmaker in the feed. The bookmaker-set rule
  (`SPIX_BOOKMAKER_SET=romania`) and value-at-a-named-book picks stay built but switched off until Romanian prices
  are available (API-Football restored, or a paid source after launch); the card then adds "Superbet has it at X".
- *Build next:* the fair-price card output (per fixture, the main lines: 1X2, O/U goals, BTTS, main handicap, main
  corners line); the rule for which angles to headline, decided with the owner (without bookmaker prices value
  cannot rank them; candidates are how decisive the market is, or how far it moved since opening); Price Check
  for singles.
- *Trade-off accepted:* the card says what a bet is worth, not where to place it.
- *Free source for Romanian prices found (19 Oct).* OddsPapi's free `/v4/historical-odds` (not billed; up to 3
  bookmakers per call, 5 s cooldown) returned live Romanian prices for UCL Fenerbahçe v Slavia Prague (20 Oct):
  Superbet RO 299 markets, last change 2 min before the call; Casa Pariurilor 122 markets, 10 min; Betano RO 215
  markets, 3 h. Finding OddsPapi's fixture ids costs 1 billable request per competition and date window (14 of 250
  used), so a weekly fixture list for 12 competitions is about 50 a month. Terms (oddspapi.io/legal/terms): no resale
  or redistribution as a standalone product; access can be ended at any time without notice; nothing on free-plan
  or commercial use. The Odds API has no Romanian bookmakers in any region (checked 19 Oct). Candidate add-on to
  the card ("Superbet has it at X"), after OddsPapi confirms in writing that showing prices in Spix is allowed.

**D37 — Launch EU-wide in English; build the Romanian version alongside (19 Oct, owner).** The fair-odds card (D36)
needs no country's bookmaker prices, so Spix launches across Europe rather than in Romania alone, in the manner of
Numbers Game (English, Europe-wide, Discord, tiers), but selling the real chance and fair price with a public
calibration record instead of a hit rate.
- *Language:* English only in production. The Romanian version is built as we go (all member-facing text and card
  output written so a second language can be switched on) and released later.
- *Coverage:* our 13 competitions (top 5, Championship, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig,
  UCL, UEL, UECL); every fixture has a fair price.
- *Rules from day one, every country:* name or link no bookmaker (changed by D38: only bookmakers licensed in the
  reader's country); no affiliate deals; no hit-rate or profit claims;
  18+ and "you can lose your stake" messaging; no paid advertising aimed at Italy (Dignity Decree); UK marketing
  follows the ASA's tipster guidance; the ONJN questions stay open for the Romanian release.
- *Business:* prices in euros; EU VAT charged at each customer's country rate through the One-Stop-Shop (Stripe can
  calculate it); a lower-priced entry tier is the owner's call (Numbers Game's starts at £3.99 a month).

**D38 — Bookmaker prices shown, only from books licensed in the reader's country (20 Oct, owner).**
- *Data:* the owner confirms that API-Football's bookmaker prices may be shown to paying members. (Their terms page
  could not be read here: HTTP 403 on 20 Oct.)
- *Rule (replaces D37's "name or link no bookmaker"):* the card may show and name bookmakers' prices for a bet, but
  only bookmakers licensed in the reader's country, and only where that country's rules allow it. Never an
  unlicensed bookmaker (1xBet, Pinnacle, SBO, Dafabet are unlicensed in most of Europe). Still no affiliate links
  or deals, and no hit-rate or profit claims.
- *Per-country lists from our feed today:* UK: Bet365, William Hill, BetVictor (UKGC register, active). Spain and
  Italy: Bet365, William Hill; Germany: Bet365 (third-party lists, to confirm on each regulator's register before
  use). Romania: none (Betano and Betfair are licensed but dropped by the feed on 16 Oct). Italy: no bookmaker
  names until the Dignity Decree's reach is checked. Romania: names only after the ONJN Class II answer.
- *Conditions before prices go live:* the reader's country is known (asked at sign-up, or a Discord role); A5.3
  confirms the prices shown are current; every price carries the time it was checked.

**D39 — A1 step 4 and A5.3 part 1: true close built; half the best prices were not on offer (20 Oct).**
- *Price reality (A5.3 part 1).* Method (`ops/price_check.py`, output `Research/price-check-2026-10-20/`): every pick
  the A5.2 market-price policy (value ≥ 2%) and the current selector make on the audited reads (latest read before
  kickoff per fixture and stage; 239 fixtures 9–19 Oct, 214 matched in OddsPapi) is checked against the quoting
  bookmaker's own timestamped prices (OddsPapi's free history): at the moment of the read, was its own price at least
  the quoted one (0.5% tolerance)? Market-price policy: 632 picks, 435 checkable (BetVictor is not in OddsPapi, 142
  lines are not in the bookmaker's history there, 51 fixtures unmatched). "Value at own price" counts a withdrawn or
  suspended quote as no bet; the consensus is still the read's own (partly stale) quotes, so it is indicative. (Sample
  corrected 22 Oct, D41: 538 checkable, 53% real; the readings below stand.)

| Slice | Checked | Real | Stale (median minutes since it moved) | Suspended | Value claimed | Value at own price | Still ≥ 2% |
|---|---:|---:|---|---:|---:|---:|---:|
| All | 435 | 54% | 107 (165) | 86 | +5.1% | +3.3% | 61% |
| Pre-match reads | 298 | 58% | 66 (217) | 53 | +4.9% | +3.4% | 63% |
| Lineup reads (~20 min before kickoff) | 137 | 45% | 41 (92) | 33 | +5.4% | +3.0% | 56% |
| Last 3 h before kickoff | 287 | 47% | 77 (95) | 74 | +5.3% | +3.0% | 57% |
| Goals / cards / corners | 171 / 94 / 61 | 65% / 72% / 59% | | 24 / 2 / 15 | | +2.1% / +7.9% / +1.9% | |
| Asian handicap | 87 | 18% | 24 | 45 | +5.7% | +3.0% | 22% |
| 1X2 | 20 | 20% | 16 | 0 | +3.4% | −2.1% | 20% |
| Pinnacle | 168 | 52% | 38 (76) | 43 | +6.4% | +5.0% | 62% |
| Betano | 139 | 58% | 40 (217) | 19 | +3.7% | +1.7% | 63% |
| Bet365 (Spanish site) | 78 | 68% | 18 (302) | 1 | +3.0% | +1.2% | 72% |
| SBO and Dafabet | 23 | 0% | 7 | 16 | | | |
| Goals, cards, corners at the main books | 317 | 68% | | | | +3.8% | 76% |

  The current selector's quotes: 60% real of 639 checked. Readings: about half the best prices API-Football showed
  were no longer on offer when we read them, and many had been withdrawn (Asian handicap alternate lines, Pinnacle's
  especially); the value does not vanish at the bookmakers' real prices (+3.3%, 61% still clear 2%; totals markets at
  the main books +3.8%, 76%); SBO's and Dafabet's quotes were never real.
- *Bet365 country sites.* API-Football's Bet365 prices match Bet365's Spanish, Italian, Greek and Dutch sites, not the
  UK or German sites, which price these selections lower: 2% of the 82 checked quotes were available there.
- *Proposals (owner decides, for stage 4 and A5.4):* (a) before a value pick is published, confirm its price on the
  bookmaker's own OddsPapi history (free; 3 bookmakers per call, 5 s each; about 10–15 picks a day) and publish only
  confirmed prices with the time checked (also D38's "checked at"); (b) until (a) exists: no best prices from SBO or
  Dafabet, and Asian handicap and 1X2 value only on the main line; (c) D38's UK list cannot use API-Football's Bet365
  prices.
- *True close (D18) built.* `data_platform/oddspapi.py` (shared client, market and line mapping; the Asian handicap
  market carries the home team's line, OddsPapi's "bookings" totals match our cards lines), `services/true_close.py`
  and `ops/true_close.py`: once a match is over, Pinnacle's last pre-kickoff prices for the pick's line (margin removed,
  proportional and power) and the pick's own bookmaker's last pre-kickoff price, in `extras["true_close"]`; a line whose
  last entry before kickoff is a suspension has no close (v2: Pinnacle's live lines end a median 1.2 minutes before
  kickoff; the inactive endings were lines withdrawn ~22 days earlier). API-Football's `closing_capture` is unchanged.
  The scorecard adds value at the true close (odds × Pinnacle's closing probability, power − 1) and true price CLV.
  First capture (20 Oct, picks published since the measurement boundary): 3,385 records, 2,737 with a close, 648
  without (line not in either history), 22 fixtures unmatched; Pinnacle's close a median 1.6 minutes before kickoff
  (p90 15); it differs from API-Football's closing consensus by a median 1.1 pts (p90 3.3); own-book closes for
  2,016 (Betano 762, Pinnacle 541, 1xBet 366, Bet365 287). 26 billable requests (fixture lists, two weeks × 13
  competitions). Scorecard of 20 Oct, each selection once: **value at the true close −2.0% (−2.8 to −1.0) on 774
  picks**, true price CLV +1.5%, against price CLV +1.8% and fair CLV +0.7% from API-Football's snapshot: measured
  against the sharp close, today's selector does not beat the market (as the stage 1 replay found).
- *Bookmaker weights (the step's original task):* D15's accuracy spread is 0.25%, so weighting barely moves the
  reference price; the problem is freshness, handled by the true close (closing) and proposal (a) (publication).
  Proposal: keep the median (equal weights) and close the weights task; owner to confirm.

**D40 — Owner's decisions on D39 (20 Oct).** All four approved:
1. *Daily true close:* `com.bettingrag.true-close` installed (10:30 local; about 13 billable OddsPapi requests a week).
2. *Bookmaker weights:* none; the reference stays the median across bookmakers. A1 step 4 is closed and **A1 is done**.
3. *Price rules for stage 4 and A5.4:* (a) confirm each market-value pick's price on the bookmaker's own history before
   it is published, with the time checked; this must exist before stage 4 goes live, because the main-line rule does not
   touch 1X2, whose best prices were stale rather than withdrawn. (b) Built behind `SPIX_SELECTION_POLICY=market_value`
   (`bookmaker_sets.UNCONFIRMED`, `line_selection._main_handicap_line`): no price from SBO or Dafabet, and Asian
   handicap value only on the main line (the home line whose median fair probability is closest to even); with the
   switches off the 54 golden fixtures are unchanged. (c) D38: API-Football's Bet365 prices are not shown to UK readers.
4. *Website restarted* on today's code and `.env` (15:39 local).

**D41 — True close repaired: OddsPapi errors had been recorded as missing prices (22 Oct).**
- *What went wrong.* The OddsPapi client handed back error answers as if they were price histories, and the true close
  read them as "no price". Of the 57 fixtures affected since 20 Oct (1,188 picks; found because a successful answer is
  always cached, so a request with no cached answer had failed):
  - *Betfair, 37 fixtures.* OddsPapi gives the exchange's (`betfair-ex`) history only on its own and for one outcome per
    request; asked together with other bookmakers it answered 400, and the bookmakers in the same request were lost too.
  - *"No historical odds found" (404), 8 fixtures.* William Hill has no OddsPapi history for some fixtures (QPR v Derby
    and 5 others; a 6th likely), nor did one Betfair outcome (Hoffenheim v Beşiktaş); the 404 failed the whole fixture.
  - *21 Oct evening, 8 fixtures (UCL, LaLiga; 141 picks).* Every request failed 2.5–2.75 h after kickoff except one;
    the same requests succeed now. The answer was not kept, so the cause is open (probe, see open items).
  - *4 Serie A fixtures, 20 Oct.* Failed for a reason not kept; the same requests succeeded on 22 Oct.
  Separately, 24 fixtures (450 picks) never matched an OddsPapi fixture and were retried every hour: Turkish letters were
  dropped instead of converted (Kasımpaşa against Kasimpasa; Beşiktaş, Fenerbahçe, Başakşehir, Gençlerbirliği), short
  names share no word with the full ones (Wolves, QPR, Köln against Cologne, Rennes against Stade Rennais), and Casa Pia
  v Santa Clara was published for 14:30 UTC and played at 19:30 (the only moved match among 4,087 settled picks).
- *Fixes* (`oddspapi.py`, `services/true_close.py`, `ops/true_close.py`, `ops/price_check.py`). Any answer other than 200
  raises `OddsPapiError` (never cached; its message has the status and code, never the URL, which holds the key);
  "rate limited" waits as told and retries; an error leaves the fixture's picks due for the next hourly run, and only
  3 days after kickoff is it recorded as unavailable with the error (`--recapture` redoes such records once a cause is
  fixed). A 404 counts as "no prices from that bookmaker" once the match is 6 h past kickoff, sooner it is retried
  (provisional until the probe). Betfair is requested alone per picked outcome (`oddspapi.histories`, now shared by the
  true close and the price check, and by price confirmation later). Names are compared in plain letters, and one team
  matching word for word is enough when no other fixture at that kickoff has it (on the cached lists: 23 of the 24 now
  match, the 24th through its settled kickoff; all 177 fixtures already captured map to the same OddsPapi fixture).
  The settled kickoff replaces the published one. An unreachable account check is reported without a traceback.
- *Repair.* The 16:30 UTC run matched all 24 fixtures: 353 closes. The 57 fixtures were recaptured: of their 1,188
  picks, 990 now have a close (829 before), 834 Pinnacle's (731), 798 the own bookmaker's (418); no Pinnacle close
  already recorded changed; 198 have none (the line is priced neither by Pinnacle nor by the pick's bookmaker). All
  measured picks: 4,087 records, 3,357 with a close (Pinnacle 2,692, own bookmaker 2,802), 730 without. Scorecard of
  22 Oct: **value at the true close −2.0% (−2.8 to −1.1) on 939 picks** (774 on 20 Oct), true price CLV +1.4%; the
  conclusion of D39 stands.
- *Price check (A5.3 part 1) corrected.* The same Betfair failure and the name matching had shrunk D39's sample: rerun on
  the same 632 market-policy picks (`Research/price-check-2026-10-22/same-picks-as-20-oct/`), all 239 fixtures match
  (214) and 538 picks are checkable (435). **Real at the read 53% (54%)**; value claimed +4.8%, at the bookmakers' own
  prices +3.0% (+3.3%); 60% still clear 2% (61%). By market: Asian handicap 17% real (18%), 1X2 26% (20%), goals 61%
  (65%), cards 73% (72%), corners 64% (59%). By bookmaker: Pinnacle 53% (52%), Betano 56% (58%), Bet365's Spanish site
  61% (68%; 2% at its UK and German sites, as before), SBO and Dafabet 0%. The current selector: 60% real of 802 (639).
  D39's readings and D40's decisions stand. A rerun over everything audited to 21 Oct (259 fixtures, 713 picks,
  `Research/price-check-2026-10-22/`) gives 52% real of 572.
- *For price confirmation (D40 (a)).* OddsPapi's pre-match history is close to live for Pinnacle (latest entry 72 s old,
  2 h before Bournemouth v Milan); Betano's latest was 21 minutes old, 1xBet's 11, Bet365's 33. The history records
  changes only, so that is either an unchanged price or slower polling: to settle before confirmation relies on it.
  A request narrowed to one outcome works for up to three bookmakers (7 kB instead of 3–7 MB per request).
  Confirmation must bypass the client's permanent cache, or the true close would later read a pre-kickoff copy.

**D42 — Price confirmation built; OddsPapi's schedule and the post-match probe (23 Oct).**
- *How fresh OddsPapi is (to settle D41's question).* OddsPapi records **changes only** and checks each bookmaker on
  its own schedule. On Juventus v Rennes and Bournemouth v Milan (22 Oct), in the last 3 hours: Pinnacle every few
  seconds, 1xBet about every 3 minutes, Bet365's Spanish site about every 6, Betano every ~10; each Betano check changed
  100–200 of its ~500 prices, so its 20–56 minute gaps are checks OddsPapi missed, not prices that stood still. Across
  858 cached histories (8–22 Oct), OddsPapi had seen the bookmaker within 30 minutes of a read moment, in the last 3 h /
  6–3 h / 24–6 h before kickoff: Pinnacle 98% / 93% / 57%, Betano 73% / 66% / 34%, Bet365 (ES) 77% / 69% / 44%, 1xBet
  84% / 78% / 76%. Far from kickoff every bookmaker's records are hours apart (Pinnacle too), where prices move less,
  so a gap there does not prove a missed check (`Research/price-confirmation-2026-10-23/`).
- *Post-match probe (D41 open item).* Every 10 minutes from 16:17 UTC to 23:11 on two 22 Oct matches: no error up to
  the end, 6.4 h after Juventus v Rennes's kickoff and 4.2 h after Bournemouth v Milan's (the only failures were "rate
  limited" during our own recapture; the market counts did not change after the rewrite). About 2 h after kickoff OddsPapi rewrites the history (2 h 05 and
  2 h 01): the added markets are in-play only; Pinnacle's markets with pre-kickoff prices are unchanged (106 before,
  106 after; Bournemouth v Milan 103). The 21 Oct evening failures
  (2.5–2.75 h after kickoff) did not recur at the same point; their answers were not kept, so their cause stays
  unknown, and since D41 such an error is retried rather than recorded. `DUE_AFTER` (2 h) and `SETTLED` (6 h) stay.
  Two of 22 Oct's "No historical odds found" (William Hill on Celtic v Celta, one Betfair outcome on Juventus v Rennes)
  are re-asked to 04:00 UTC to see whether such an answer ever turns into prices. Soft bookmakers' rewritten
  histories have slightly fewer markets with pre-kickoff prices than 27 minutes before kickoff (Betano 214 against
  227): open item.
- *Built* (`oddspapi.py`, `services/price_confirmation.py`, `rendering/match_read_dispatch.py`). `OddsPapiClient.fresh`
  asks without reading or writing the cache; `oddspapi.last_seen` is a bookmaker's newest record on any line. For a
  stored read made under `market_value` (the read's own switch record decides), every recommended result is checked
  before compilation against the quoting bookmaker's full history as it is now (one request per three bookmakers per
  fixture; free; about 5 s apart): **confirmed** (its price now is at least the quote, 0.5% tolerance) keeps the quote;
  **repriced** (shorter, but the value at it still clears the selection's 2%, owner 22 Oct) publishes at the
  bookmaker's own price, with value and fair probability recomputed from its own prices now; anything else is a
  **no-bet** (price gone, line suspended or not listed, bookmaker not seen within `MAX_AGE`, bookmaker or match not in
  OddsPapi, an exchange (commission not modelled yet), or an OddsPapi error). The evidence (prices, when the price was
  set, when the bookmaker was last seen, when checked) goes to `context["price_confirmation"]`, for a "checked at" time
  on the card (D38). Fixture lists come from their own cache (`Index/oddspapi/confirmation`), so the true close keeps
  fetching its lists after the matches; one client per worker run keeps OddsPapi's 5 s cooldown across fixtures.
  Previews and shadow runs make no request. Not live: nothing changes until stage 4 sets `market_value`.
- *Replay on the price check's picks* (8–21 Oct, market policy, SBO and Dafabet left out as at stage 4: 670 picks;
  each read checked against the history as it stood then). Published (confirmed + repriced) by freshness limit:
  15 min 186 (Betano 49 of 236), 30 min 224 (69), 60 min 264 (89), no limit 339 (138). With no limit the rest are
  no-bets for price gone 100, suspended 89, not listed 75, match not in OddsPapi's lists 32, bookmaker not carried 22,
  exchange 13. Was a price confirmed from an older record real? Share still on offer at the bookmaker's next check
  after the read: OddsPapi's last check within 30 minutes 91% (190 picks), 30–60 minutes 83% (35), over an hour 86%
  (72); without confirmation about half the best prices were real at the read (D41). **Owner decides `MAX_AGE`**
  (30 minutes now): a limit trades about a third of the picks for a few points of reliability.

**D43 — API-Football's "Betfair" is Betfair's sportsbook, not the exchange; the exchange-commission step is dropped (23 Oct).**
- *Evidence.* Median 1X2 margin in our archived API-Football odds since 19 Oct: Betfair 7.1% more than 24 h before
  kickoff, 6.8% at 3–24 h, 6.9% in the last 3 h (1,383 captures); goals O/U 2.5 6.6%. Betfair Exchange's closing 1X2
  margin in football-data is 0.6% (6,876 matches; its early prices 3.7%), Betfair's sportsbook 8.1% at the close
  (5.8% early). In the 22 Oct price check, OddsPapi's exchange (`betfair-ex`) price at the read was above our Betfair
  quote on 30 of 43 picks (3 equal; median +1.9%), as an exchange beats a sportsbook before commission.
- *So.* No exchange price reaches selection, so there is no commission to model: A5.3 goes straight to evidence-based
  labels. `oddspapi.BOOKS` maps "Betfair" to `betfair-ex`, which is wrong: the true close of Betfair picks used the
  exchange's prices, the price check's Betfair rows (13 market-policy picks, 43 current) compared sportsbook quotes with
  the exchange (its "81% real" is not evidence), and price confirmation turns Betfair picks down as "exchange" (13 in
  the replay). OddsPapi lists Betfair's sportsbook only as Betfair ES and Betfair IT (`betfair.es`, `betfair.it`, both
  clones of one feed `betfair-spb`); no Romanian or UK site. Next: a free pilot, as for Bet365 (20 Oct), comparing
  API-Football's Betfair with `betfair.es` / `betfair.it` at the same moments; map to it if one matches, else mark
  Betfair not carried (price confirmation then fails closed on Betfair picks, the true close falls back to
  API-Football's snapshot). Betfair is one of the two Romanian-licensed books in the feed (D35), so this matters for
  stage 4's `romania` set.
- *23 Oct, after the pilot.* OddsPapi's Betfair sportsbook feed (`betfair-spb`, parent of Betfair ES and IT) answers
  403 RESTRICTED_ACCESS on our plan, and `betfair.es` / `betfair.it` answer 404 "No historical odds found" on Udinese v
  Lecce, Osasuna v Racing Santander and Juventus v Rennes; a clone site can have its own history (Betano RO: 264
  markets, as Betano). Nothing to compare with, so Betfair is not carried (`oddspapi.BOOKS`); the exchange code path
  stays for a future source. The scorecard's true price CLV leaves Betfair picks out (no own close).

**D44 — A5.3 confidence labels: high means the market and Spix agree; until Spix earns it, how firm the market is
(23 Oct, owner).**
- *The percentage* a member sees is the chance this bet wins in this match. It is the published fair probability,
  unchanged by this step. Calibration is the check behind it, not what it says.
- *The label* is high or low:
  - **High:** the market and Spix agree it is a good bet.
  - **Low:** still a good bet against the market's price, but less sure.
  - The label never changes which bets are published.
- *Spix's vote counts market by market,* once that market passes a check: picks where Spix agrees keep more value at
  the close and are no worse calibrated. The list of such markets starts empty and changes only by an owner-approved
  decision here. Corners (c = +0.49, 23 Oct) is the first candidate.
- *Until then, the owner chose "market firmness" labels* over no labels:
  - **High:** the bookmakers agree closely on the line (M3) and the price is fresh (price confirmation's
    `bookmaker_seen_at`, from stage 4).
  - **Low:** scattered bookmakers, or an old or unknown price.
  - Firmness cut-offs ship only if high beats low at the close on the audited picks, confirmed on a held-out week.
    If not, the result comes back to the owner.
- *Replaces* `confidence_from_edge`, whose inputs are Spix's own numbers (D30: mostly error). D30's sample size and
  uncertainty width are left out until a market earns Spix's vote.
- *Build:* behind `SPIX_CONFIDENCE_LABELS` (`edge` default, `evidence`), its own rollout stage after stage 4.
- Design: `docs/superpowers/specs/2026-10-23-a5-3-confidence-labels-design.md`.
- *Built and checked 23 Oct* (plan `docs/superpowers/plans/2026-10-23-a5-3-confidence-labels.md`): see the finding of 23 Oct.
  The Match Read compiler ranks a fixture's picks on the label an evidence label replaced (`confidence_evidence.edge_label`),
  so labels never change which three picks a read publishes (the legacy ranking puts confidence first).

## Findings that shape later steps

- **Stage 4 replay rerun with the price histories downloaded: confirmed picks keep their value (28 Oct).**
  `ops/stage4_replay.py --from 2026-10-09 --to 2026-10-28 --confirm --fetch-histories`
  (`Research/stage4-replay-2026-10-28/`; final read per fixture, 396 fixtures; 1 pick without a history; histories are
  free, the billable count stayed at 65; the studies buy no fixture list, so 5 fixtures whose weekly list was not
  cached are left out of confirmation). Today's rule: 833 published, value at the close −1.4% (−1.8 to −1.1), stated
  75.7% / won 64.9%. Stage 4 at 2%: all books 225, +1.6% (+0.7 to +2.5), stated 48.3% / won 48.6%; Romanian books 61,
  +4.3% (+3.0 to +5.6). After price confirmation at 30 / 60 minutes / no limit: all books 49 / 58 / 71 picks, worth
  +1.6% / +1.7% / +2.8% at the close; Romanian 18 / 22 / 27, +2.6% / +1.8% / +1.9%. So over the 19 days confirmation
  keeps about 2.6–3.7 picks a day at all books and 1–1.4 at Romanian books, and what it keeps still beats the close.
  (24 Oct, with 26 picks lacking a history: 18 / 22 / 32 and 4 / 6 / 9.)
- **The bookmaker-spread label passes on its holdout for the first time; the handicap's Spix vote is not confirmed
  (28 Oct).** `ops/confidence_study.py` on 1,044 market-price picks (348 fixtures; at least 4 bookmakers; train 672,
  holdout 372): spread cut-off 0.0159, 24% of picks high; on the holdout high picks are worth +5.1% at the close
  against +3.5% (+1.6%, +0.2 to +3.1), on the training half +4.9% against +4.2% (+0.7%, −0.6 to +2.0). By market high
  is better on the handicap (+3.6%, +1.7 to +5.6) and 1X2 (+3.2%, +0.3 to +6.7) and worse on cards (−3.4%, −5.3 to
  −1.6). The Spix check passes in no market; the handicap, last week's candidate, is +1.9% (−0.0 to +4.1). The age
  check (`ops/confidence_age_study.py --fetch-histories`, 445 picks on offer at the read, 243 fixtures): a fresh price
  (seen within 30 minutes) was still there at the next check 95% of the time, an older one 88% (+7.0 pts, +0.5 to
  +14.3), value at the close +3.7% and +2.9%. One passing week on a holdout of 111 high picks; recommended: rerun next
  week before the labels use the spread (owner's call, D44).

- **Stage 3 is live and prices 1X2 and O/U 2.5 at the market (28 Oct).** The 50 reads of 23:52–23:58 UTC 27 Oct (the
  first after the gate reopened; 19 fixtures, 8 leagues) against the 33 of the 48 h before: every result records
  `fixed`, `power`, `strict` (350 of 350); every 1X2 result carries the market's probabilities, summing to one, and
  prices on them (50 of 50); 1X2 picks 9 → 0 (the preview's expectation); O/U 2.5: 661 of 666 options take the
  market's consensus exactly (median of 5–8 books' power fair prices) and keep Spix's as `_spix_prob`; BTTS, corners,
  handicap and other goal lines stay on Spix's numbers, as scoped. The other 5 options are one read (PSV v Willem II,
  1552209, 31 Oct) whose over 2.5 (1.08–1.15) is below the totals odds floor: `line_selection` forms the fair pairs
  from the options left after the floor, so the line has no pair and no anchor, and the long-shot under 2.5
  (5.5–6.0) stays on Spix's 27.3%. Not picked (the read took under 6.5 @ 1.22), but a lopsided 2.5 line's long shot
  can still be priced on Spix's number: candidate fix, form the pairs before the floor. The restarted website serves
  the stage 3 card: its 1X2 balance equals the fair-price card's (93.1 / 4.6 / 2.3%) and the briefing says "favourites
  in the market consensus"; the website labels the balance only with the fair-price card design (D36).

- **The daily refresh failed on a sleeping Mac two mornings running; it now retries hourly (28 Oct).** On 26 Oct the
  07:15 run started at 08:52 local when the Mac woke and failed at 09:10 in `platform_incremental_sync` with "database
  is locked" (the stage's own clock counted 11.7 s of those 18 minutes: the Mac slept in between). On 27 Oct it started
  at 07:17 on battery and failed at 08:16 in the same stage, after API-Football connection errors every ~15 minutes
  (the Mac's brief dark wakes) and seven "database is locked" fixture updates (43 s of process time in 59 minutes).
  Nothing reran it, so the gate stayed shut (no Match Reads, no measurement, no true close) for 8 h 22 min on 26 Oct (rerun 16:26,
  done 17:15) and 18 h 27 min on 27–28 Oct (rerun 01:02, done 01:44). Neither alert reached Discord
  (`notification_delivered: False`, no webhook). What held the lock is not established; both failures ran while the
  Mac slept. Built (`services/refresh_retry.py`, `refresh-season --if-due`): the job fires at a quarter past every
  hour; from 07:15 local it runs when no run has started since 07:15 that day or the last full refresh did not
  complete; an hour is skipped, and not counted, when the Mac is on battery with its lid shut or API-Football's host
  does not resolve; at most 6 started runs a day; nothing before 07:15; the gate is unchanged. Installed 28 Oct 01:45
  (`ops/upstream_refresh_launchd.py install --job data`); a manual start at 01:45 skipped ("before the daily refresh
  time"), exit 0, gate untouched. A kickstart of the job now follows the same rule; to force a run, use
  `caffeinate -i -s .venv/bin/python -m Scripts.data_platform refresh-season --include-europe` (no `--if-due`).
- **OddsPapi's post-match rewrite keeps the pre-kickoff prices (28 Oct).** The weekend capture
  (`ops/history_rewrite_check.py`, `Research/history-rewrite-2026-10-24/report.json`) saved 56 of its 73 fixtures of
  23–25 Oct twice, 2–100 minutes after kickoff and 3–12 h after (13 missed the first save and 2 the second, while the
  Mac slept or was offline). For Pinnacle, Betano, 1xBet, Bet365 (ES) and Marathonbet every last pre-kickoff price was
  the same in both saves (1xBet 33,323 lines, Betano 24,941, Pinnacle 14,551; none changed, dropped or added; the same
  markets had a close); William Hill had no history. Every pick's true close was the same wherever either save had one
  (Pinnacle 745, own bookmaker 858; 510 and 397 had none in either). So own-bookmaker closes (`same_book`) taken from
  the rewritten history stand. The 22 Oct difference (Betano 227 → 214 markets) compared a save 27 minutes before
  kickoff, not at kickoff; its cause is not established.

- **Fair-price card numbers and Price Check built as data and an API (24 Oct; D36).** `data_platform/fair_prices.py`:
  per read, the main line of 1X2, goals, BTTS, the handicap and corners (the line most bookmakers price on every
  side, then the most balanced; at least two bookmakers), each outcome's fair probability (median across bookmakers of
  each one's power-devigged price, 1X2 rescaled to one), fair odds, minimum odds worth taking (fair odds × 1.02,
  rounded up to the cent: D36's "fair 1.85; worth it at 1.89 or better") and Spix's own probability beside it. The
  Match Read card carries it as `fair_prices` (no bookmaker named); `GET /api/match-reads/fixtures/{id}/price-check
  ?market=&side=&line=&odds=` rates a member's odds against the effective read's fair price for any priced outcome:
  good price from +2% value, overpaying from −2%, fair between (`FAIR_BAND`, **owner to confirm the band**), 410
  once the read is no longer actionable, as the card. On the 249 reads since 18 Oct every card has all five lines
  (about 9 ms per card). Not built: the card's design, the headline rule (owner, D36), a Discord command, the
  Romanian text.

- **Card settlement: the bookmakers count a straight red as two (24 Oct; before stage 5).** The four bookmakers that
  price cards in our reads are Betano (422 of 556 reads since 9 Oct), Bet365 (385), 1xBet (379) and Pinnacle (366).
  Their rules: yellow 1, red 2, a second yellow ignored (at most 3 per player), regulation time only, cards to
  non-players not counted. Bet365 (help centre, card markets) and Pinnacle (betting rules) from their own pages as
  quoted by search; Betano from Romanian guides quoting its regulation (2022–24); 1xBet not confirmed (one
  third-party page says it counts cards to players and coaches). The pages could not be opened here (a self-signed
  certificate intercepts HTTPS on this machine).
  - *Our count* is the team statistics' yellows + reds, which count both yellows of a two-yellow sending-off (1,752
    of 1,811 cases agree with the events), so it equals the bookmakers' rule except that a straight red counts 1, not
    2. On the 2021–2026 card events (20,030 fixtures, `Research/card-settlement-2026-10-24/measure.py`): +0.113 cards
    per match, totals differ in 10.4% of matches, the chance of an over rises by 0.4 (2.5) to 2.0 points (5.5).
  - *It does not explain the cards miscalibration.* On the 133 market-price cards picks of 9–22 Oct (104 overs,
    mostly over 4.5 and 3.5; consensus stated 45.3%, won 32.3%) the bookmakers' rule would add under a point to
    "won". On the same picks every bookmaker's own fair price overstated them: Pinnacle 39.4% (won 31.8%), Betano
    44.8%, Bet365 46.1%, 1xBet 46.0%. 113 of the 133 picks took Pinnacle's price: the rule picks overs where Pinnacle
    is longest against a median that the three softer books pull towards overs. For A5.4 and stage 5, the cards
    reference price should not be the four-book median (Pinnacle alone, or no cards picks); decide with A5.4.
  - *To settle and model cards by the bookmakers' rule* the engine needs to know straight reds, which only the
    events feed gives: card events stop at 20 Sep (the 30 Sep history download), and the daily refresh fetches
    `/fixtures/players` and `/fixtures/statistics` but not `/fixtures/events`. **Owner:** add `/fixtures/events` to
    the daily refresh (one API-Football request per finished match, as the player statistics) and backfill from 20
    Sep; then the cards outcome becomes yellows + reds + straight reds, and A4's cards calibrator and A6's cards
    model use it.

- **penaltyblog checked for A6 (24 Oct).** Version 1.13.1 (5 Oct), MIT licence, wheels for Python 3.13 on macOS
  arm64 and Linux; checked in a throwaway environment (`Research/penaltyblog-check-2026-10-24/`), the project's
  untouched. Against the tracker's four needs:
  - *Time decay:* yes. Every model takes `weights`; `dixon_coles_weights(dates, xi, base_date)` gives exp(−xi · days).
  - *Per-league home advantage:* only as one fit per league; a model has a single home-advantage parameter, so a
    joint fit across leagues (needed for the European cups: 149–286 teams with few matches each per competition, and
    for connecting leagues) cannot give each league its own. Fitted per competition (since 1 Jul 2024, xi 0.0018),
    home advantage ranges from 0.10 (Serie A) to 0.29 (LaLiga), so per-league values matter.
  - *Promoted-team offset:* none (no mention anywhere in the package); the hierarchical Bayesian model only pulls
    thin-data teams to the league average.
  - *Clean install:* the full install is 83 packages, 535 MB (plotly, kaleido, matplotlib, scrapers, statsbombpy,
    pulp), and `import penaltyblog` fails without them; the models alone run on numpy, scipy and pandas only by
    bypassing the package's `__init__`.
  - *Correctness:* its Poisson fit matches an in-house weighted maximum-likelihood fit to 0.0008 goals and 0.0002
    in 1X2 probability (EPL, 830 matches). Its Dixon–Coles fit does not: the analytic gradient is wrong (at the
    optimum it reads −1.22 for home advantage where the true slope is −0.003; the same without weights), so the
    default `fit()` stops early (log-likelihood 1184.92 against 1184.90; expected goals off by up to 0.034, 1X2 by
    up to 0.7 points). With `fit(use_gradient=False)` it reaches the in-house optimum exactly. Worth reporting
    upstream (owner's call: an issue on github.com/martineastwood/penaltyblog).
  - *Decision under the open item's rule:* it does not cover them, so A6 fits in-house (scipy, ~0.1–0.2 s per
    league with numerical gradients), with per-league home advantage, a promoted-team offset and a joint fit
    across leagues, and is validated against penaltyblog's Poisson fit and its Dixon–Coles fit with
    `use_gradient=False`, in a separate test environment rather than as a project dependency.

- **Stage 4 replay on two weeks: price confirmation leaves few picks (24 Oct).** `ops/stage4_replay.py`
  (`Research/stage4-replay-2026-10-24/`) re-runs the engine's own selectors on the prices stored by each fixture's final
  read (307 fixtures, kickoffs 9–23 Oct) under the bundle's switches, honours each read's stored guardrail verdict,
  and publishes at most three compatible picks per Match Read by the compiler's rule. Checked against the engine on
  the 54 golden fixtures: 28 of 28 picks identical (the 17 Oct preview's 22 at all books and 6 Romanian). Published
  picks, value at the close at the published price:
  - *Today's rule:* 608 picks (225 fixtures), stated 75.3%, won 63.8%, value at the close −1.4% (−1.8 to −1.0).
  - *Stage 4 before confirmation,* value threshold 1 / 2 / 3 / 5%: all books 287 / 159 / 102 / 43 picks, value at
    the close +0.9 / +2.2 / +2.8 / +3.7% (2%: +1.0 to +3.5; stated 48.9%, won 49.8%); Romanian books (Betano only:
    the feed has no Superbet) 77 / 40 / 28 / 14, +2.8 / +5.5 / +6.2 / +7.1%. Shots on target produced no pick;
    cards are blocked by the guardrail in every read.
  - *After price confirmation at the read* (the quoting bookmaker's OddsPapi history as it stood then; downloaded
    histories only, 0 requests: billable 61 and 1,243 cache files before and after), at 2%: all books 159 → 18 at
    30 minutes, 22 at 60, 32 with no limit (26 without a downloaded history); Romanian 40 → 4, 6, 9 (7 unknown). Of
    the 33 Romanian picks with a history, 18 were already shorter at Betano at the read and 6 suspended; at all
    books 37 of 159 were at bookmakers OddsPapi does not carry (confirmation fails closed).
  - *Reading:* the rule picks the largest gaps to the market, and those are mostly API-Football's lagging copies of
    prices the bookmaker had already moved: 27% of the Romanian picks with a history were confirmed with no limit,
    against 58% of D42's Betano picks (a broader set). As built, stage 4 with confirmation publishes about 2 picks
    a day at all books and under 1 a day at Betano (no limit; fewer at 30 minutes). The threshold and `MAX_AGE`
    decisions should weigh this; taking the selection's prices from a fresher source (OddsPapi's own Betano prices
    rather than API-Football's copy) would change the picture and is not built. Rerun Mon 26 Oct with `--confirm
    --fetch-histories` once the history-rewrite capture is idle (free requests for the 26 missing histories); the
    rewrite compare that day also says whether histories cut at the read can be trusted.

- **A5.3 checks, first run (23 Oct; D44).** On the market-price policy's picks of 9–22 Oct (824 picks, 274
  fixtures; the earlier week chooses, the last week confirms):
  - *Bookmaker spread: not yet.* The best cut-off on the first week (interquartile range 0.0157, 25% of picks high)
    kept +2.2% more value at the close there (+0.7 to +3.7), but only +0.4% on the held-out week (−1.5 to +2.1). So
    `CUTOFFS.max_spread` stays unset and every label would be low. By market it points the right way for the handicap
    (+3.3%, +0.9 to +5.2 over both weeks), 1X2 and goals, and the wrong way for cards (−3.3%, −6.0 to −0.4).
    Per-market cut-offs chosen and checked the same way (`per_market_exploratory.json`, exploratory only): only 1X2
    passes, on 24 held-out picks.
  - *Price age: supported* (`ops/confidence_age_study.py`, 297 picks whose quoted price was on offer at the read,
    8–21 Oct, cached OddsPapi histories, no billable request). Prices whose bookmaker OddsPapi had seen within 30
    minutes were still there at its next check 95% of the time, older ones 85%: +9.5 pts (+1.0 to +18.5). Value at
    the close: +4.9% against +2.9%. `fresh_minutes` stays 30. Rerun the same evening to 23 Oct from downloaded data
    only (fixture lists from the true close's weekly ones; billable count and cache unchanged): 308 picks, 177
    fixtures, 95% against 85%, +9.8 pts (+1.8 to +18.7). Most later histories were downloaded several bookmakers to a
    request, so `--fetch-histories` (free requests) would add them; run it when the weekend capture is idle.
  - *Spix check:* the handicap passes (Spix agreeing +2.2% at the close, +0.2 to +4.8; calibration not worse).
    Goals (+0.1%), corners (+1.1%) and cards (−0.8%) do not; 1X2 and BTTS have too few picks. It is one pass in four
    tested markets, so a candidate for the owner after a confirming week, not a vote yet (`SPIX_VOTE_MARKETS` stays
    empty).
  - *For the owner:* the age part of the label only matters if price confirmation's `MAX_AGE` is no limit (Claude's
    recommendation). At 30 minutes every published pick is fresh by definition.
  - Output: `Research/confidence-2026-10-23/`, `Research/confidence-age-2026-10-23/`.
- **Fixed 23 Oct: the fixture lookup used by the price check, the Pinnacle check and the age check**
  (`pinnacle_check.our_fixtures`) read `platform.db` outside the refresh gate and never closed its connection. It
  now takes the gate itself, and its callers read their audited lines first through `price_check.load_inputs`, so
  the gate is never taken twice. A nested gate fails when the refresh is waiting.
- **`golden_snapshots check` reads the live `.env` (23 Oct).** A plain `check` inherits the switches saved in `.env`
  (stages 1–2 since 20 and 23 Oct), so it reports thousands of changed values with no code change (2,530 on the first
  fixture). Run it with every switch at its default, e.g. `SPIX_COUNT_DISPERSION=floors SPIX_DEVIG_METHOD=proportional
  SPIX_MARKET_ANCHOR=off SPIX_CONFIDENCE_LABELS=edge python -m Scripts.ops.golden_snapshots check`. On 23 Oct that
  gives no changes across 54 fixtures. (Earlier notes saying the check forces the switches off were wrong.)
- **Spix's corner disagreements now carry information, but tilting toward them adds no value at the close (23 Oct).**
  A5.1 rerun on two weeks (295 fixtures, 9–22 Oct; stage 1's fixed dispersions live since 20 Oct): the weight results
  give Spix's deviation from the market is c = +0.14 overall (−0.01 to +0.31), and **corners +0.49 (+0.18 to +0.80)**,
  the first market whose interval excludes zero (BTTS −0.42, cards −0.15, goals −0.13, 1X2 +0.12, handicap +0.10, all
  including zero). The market still barely moves toward Spix before kickoff (k = +0.01). The A5.2 selection study,
  rerun and split by market (`Research/selection-2026-10-23/`, pre-match reads, 2% minimum): pricing against the
  market keeps the most value at the close in every market. Corners: market +3.7% (+2.6 to +5.0), tilt c = 0.10
  +3.3%, c = 0.25 −0.6%, the current selector −2.5%. Overall: market +2.9% at 0%, +7.3% at 4%; current −1.6%. One
  more caution: in cards the market price policy states 45.1% and wins 33.7% (−11.3 pts, −21.6 to −1.3) on the
  yellow + red grading, which the card settlement definition must qualify first. A5.4's proposal (D31) stands;
  corners' c is one to watch at each rerun.
- **The research studies read `platform.db` in short statements behind the refresh gate (23 Oct).** The A4.2, A5.1
  and D29 studies (`calibration_study`, `disagreement_study`, `m1_gate_study`; also `selection_study`, `stage1_replay`) each selected every match read's JSON
  in one statement: 3,335 reads of about 0.75 MB each (about 2.5 GB), which would hold the database's read lock far
  past the jobs' 2 s busy timeout. They now page through the reads' other columns 500 rows at a time, pick the latest
  read per fixture and stage, and fetch each chosen read's JSON alone (`calibration_data.latest_read_rows`); the
  closing snapshots likewise (`disagreement.closing_snapshots`). Each study's reads run inside `data_access()`.
- **The odds archive pauses while the Mac sleeps (23 Oct).** Its 3-hourly job (`StartInterval` 10800) last captured
  at 23:16 UTC on 22 Oct; launchd's interval timer does not count sleep (asleep 23:51–07:10 UTC), so the next run
  came about 3 awake hours later. Line movement (M2) and the coverage watch both read this archive: one more reason
  for the scheduled wake or a server.
- **The 23 Oct daily refresh failed because of an ungated read (23 Oct).** `canonical_kb_refresh` failed at 07:43 UTC
  with "database is locked" while a working session read all 5,681 published picks from `Index/platform.db` outside
  the refresh gate (4.0 s; the database is in rollback-journal mode with a 2 s busy timeout, so a reader blocks a
  writer's commit). The gate stayed closed (no Match Reads, no measurement) until the job was rerun at 07:48 UTC.
  Rule: ad-hoc reads of `platform.db` go through `data_access()`. The 19 Oct refresh also failed with "database is
  locked" (in `incremental_refresh`); its cause is not known.
- **The dropped bookmakers are back in API-Football (20 Oct).** Fixtures priced per day in our raw `/odds` responses
  (odds archive): Betano 189 of 191 on 14 Oct, 134 of 165 on 16 Oct, 4 of 188 and 4 of 158 on 17–18 Oct, 96 of 174
  on 19 Oct, 187 of 193 on 20 Oct; Betfair, SBO and Dafabet the same pattern. Superbet and Unibet stay at 0. So the
  16 Oct drop lasted about two days, and Betano (licensed in Romania) is again in the feed we pay for. (The archive's
  per-quote table stops at 6 Oct; coverage has to be counted from the raw bodies.)
- **The website process has run since 8 Oct (20 Oct).** `com.bettingrag.website` (uvicorn, PID started 8 Oct
  16:54) has not reloaded code or `.env` since, so its on-demand `/api/fixtures` and `/api/best-bets` predictions
  run 8 Oct code with Chroma and no engine switches. Published picks are unaffected (the Match Read worker starts
  a fresh process every 10 minutes). Restart with the owner.
- **Alternatives to OddsPapi for Romanian bookmakers (20 Oct; vendor pages, not tested).** Odds-API.io lists
  Superbet RO, Betano RO, Fortuna RO, Mr Bit RO and Casa Pariurilor (no country given); plans £49 a month for 2
  bookmakers, £99 for 5, £179 for 10, no free tier; Superbet page names match result, handicap, totals and BTTS,
  not corners. Sportmonks' premium odds feed: only Betano.de found, no Romanian bookmaker confirmed (EUR 129–199 a
  month, second-hand figure). OpticOdds: Betano listed, no published prices (sales call). The Odds API and
  API-Football: none (19 Oct, 16 Oct). Apify scrapers: Superbet (RO, PL, RS, HR; match result only; 1 user; from
  $1 per 1,000 results) and Betano (13 countries incl. Romania, all markets; $0.0015 a call or $3 a day); both
  scrape the bookmakers' sites and leave compliance with the bookmakers' terms to the user; none found for Fortuna
  or Casa Pariurilor. Superbet and Betano publish no odds API of their own. OddsPapi stays the cheapest for the
  same bookmakers.
- **OddsPapi's paid plans are priced per bookmaker; the smallest Romanian plan is $15 a month (20 Oct).** Their plan
  calculator (oddspapi.io/en/pricing; quotes from `api.oddspapi.io/checkout/calculate-price` with our key, nothing
  bought): monthly price = (chosen bookmakers + sports) × request tier. Football $5. Tiers: Normal ×1 (about 5,000
  requests a month), Dev ×2 (100,000), Pro ×5 (500,000), Ultra ×10 (1,500,000); 10% off from 10 bookmakers.
  Bookmakers a month: Superbet RO, Fortuna RO (efortuna.ro), MaxBet RO, WinBet RO, NetBet $10 each; Betano RO,
  Unibet, William Hill $20; Casa Pariurilor RO, Bet365, Betfair exchange $30. BetVictor is not carried; Pinnacle is
  not sold separately (price −1 in their list). Quotes for football, pre-match, Normal tier, in USD: Superbet RO 15;
  plus Fortuna RO 25; plus Betano RO 45; plus Casa Pariurilor 75; Superbet RO with live odds 25; Bet365 + William
  Hill 55. All nine Romanian bookmakers on our list 135 (no volume discount below 10 books); three of them (Unibet,
  NetBet, Mozzart) are international feeds, not Romanian sites. Romanian-site feeds: Superbet, Fortuna, MaxBet,
  WinBet $10; Betano, 888sport, Mr Bit, PokerStars $20; Casa Pariurilor $30. Still open: whether about 5,000 requests a month cover our refreshes (untested on a paid plan),
  OddsPapi's written OK to show prices in Spix (their terms bar resale or redistribution as a standalone product),
  and the ONJN answer before naming Romanian bookmakers (D38).
- **Switching launch country does not fix the bookmaker gap (19 Oct).** Stage 4 bundle on the 54 golden fixtures
  (15 Oct prices), value picks with only each country's licensed bookmakers from our feed: Romania today 0, Romania
  if Betano returns 6, UK (Bet365, William Hill, BetVictor; all active on the UKGC register, accounts 55148, 39225,
  39576) 5, Spain or Italy (Bet365, William Hill; third-party lists) 5, Germany (Bet365; third-party) 5, every book
  43. All five UK picks are at Bet365. The 43 come from 1xBet 17, Pinnacle 13, Betano 6, Betfair 3, SBO 2,
  Bet365 2. Since the 16 Oct drop all 204 fixtures have at least two bookmakers (median six, Pinnacle in 177), so a
  fair price is always available. Italy's Dignity Decree bans nearly all gambling advertising. Romania: affiliates
  need an ONJN Class II licence (ONJN filed a criminal complaint against one in March 2026), and a Senate-passed bill
  would ban online gambling-promoting content from 06:00 to 24:00 (Chamber status unknown). Report:
  https://claude.ai/artifact/PjF3wM3TvL1G4nPoe3tjRy. It confirms D36.

- **Goal means run low (16 Oct).** Spix projected 2.82 goals per match against 3.16 scored over 321
  finished fixtures since 5 Sep, and 2.68 against 3.28 in the audited week of 9–15 Oct. That week was
  high-scoring and the market under-read it too (O/U 2.5 overs: market 0.581, Spix 0.489, scored
  0.662), but Spix sits about 10 pts below the market on overs on average (other goal lines 0.384 vs
  0.481). This strengthens the case for extending the market anchor to all goal lines (D26 open item)
  and for A6's fitted goal model; a Platt intercept corrects it only in part (D28).

- **Fixed corner dispersion holds on live data (16 Oct).** On the full corner-count distribution, the
  stage 1 fixed dispersion beats the floors on the live engine's means: 317 finished fixtures (5 Sep
  to 15 Oct), −0.020 nats per fixture (95% −0.045 to +0.008), consistent with A3.4. On the audited
  lines alone (156 fixtures, 9–15 Oct), the floors score better (line log loss 0.564 vs 0.573,
  interval −0.003 to +0.024, includes 0); cause not established. Not a reason to move stage 1.
  The corner mean barely tracks the actual count (correlation 0.09–0.12).

- **Unused signals (16 Oct).** `get_elo_edge` is imported by `core/projections.py` but never called; the
  58-feature ML regressions have saved validation R² ≈ 0 (goals 0.018, corners −0.010, cards 0.006,
  SoT 0.027) and blend weight 0. More features alone did not help; new variables follow the gate in the
  variables report (fitted, ablated against the market, ≤3 per model per round).

- **The engine has never beaten the market (D9, 15 Oct).** On 2023 (2,783 fixtures, 41 weeks) the
  frozen control is 4.4% worse than the closing market on 1X2 log loss and 2.1% worse on over/under
  2.5; the Phase 4 finalists that "qualified" against the control are still 1.4–3.7% worse.

- **First scorecard (15 Oct, each selection once, 571 settled):** won 61.2% vs Spix said 74.0%
  (−12.8 pts, 95% −19.6 to −6.5); market said 62.8%. Log-loss gap +0.052 (Spix worse than the
  market, interval excludes 0). Calibration slope 0.42 (strongly overconfident). Fair CLV
  +0.9% (+0.1 to +1.8): the market moves toward Spix's picks after publication, a small signal
  A3's blend can use. Goals most overconfident (said 81.5%, won 62.9%).
- Pinnacle on API-Football moves with the market (1X2 100%, AH 93%, O/U 84% of moves) and
  changes price most often; accuracy not yet measurable (14 Oct freshness study).

**D9 — Market baseline added to the research protocol (15 Oct).** Amendment
`docs/phase4-market-baseline-amendment-2026-10-15.md` (`market-baseline.v1`): every research
comparison also scores the candidate against the closing market (football-data history, 1X2 and
over/under 2.5, same reference-price method, paired week bootstrap). Corners/SoT/BTTS have no market
price in the history; handicap compared in a later version. A3 must beat the market alone; A6 keeps
only components that improve the gap to the market. Code: `features/market_baseline.py`,
`ops/market_baseline_report.py`.

**D10 — A2 in two stages (16 Oct).** The report assumed one database source; the engine actually
reads a mix: of 18,878 team-match documents in the golden reads, 9,595 come from the legacy
pipeline (per-league scripts → `Output/` CSVs → `Index/normalized_*.json`, most of the 2025/26 and
2026/27 seasons) and 9,283 from the platform's `kb_documents`. Legacy rows carry fields the
platform docs lack (control/aggression/form indices; corners, shots, SoT conceded). No match is
counted twice. Replaying the 881 recorded queries on `kb_documents` alone: 210 team-match and 72
profile queries return fewer documents than Chroma. Hence:
- **A2a:** serve the engine's exact documents from SQLite (`kb_documents` + `normalized_*.json`),
  switch by flag, golden snapshots must be identical; then drop the embedding step from the daily
  refresh (removes OpenAI from the prediction path). Chroma stays only for semantic-search features
  (Whale chat, `rag_cli_v2`) until A7.
- **A2b:** database-only feature builder recomputing the legacy-only fields; every golden difference
  explained; then retire the per-league scripts, `Output/` and the normalise step.
Owner also asked (16 Oct) for a report on candidate new variables; selection happens before A2b/A6.

**D11 — A2a: SQLite document store (16 Oct).** `Scripts/data_platform/vectors/document_store.py`,
`Index/documents.sqlite3` (186 MB; Chroma is 18 GB): Chroma's 129,677 documents exported without
embeddings, in Chroma's order. Parity: all 881 recorded engine queries return identical results
from the store and live Chroma; on the same data the engine's outputs for all 54 golden fixtures are
identical on both. `DOCUMENT_STORE=sqlite` switches the engine's two lookups
(`team_resolution.get_collection_handle`, `lineup_context._get_chroma_collection`). The daily refresh
now writes the store as a required step (legacy `pull_season --load-documents` with the embed's skip
rules; `kb-refresh --vector-backend sqlite+chroma`, store first, Chroma best effort) and embeds into
Chroma as a separate stage whose failure is a warning. Finding: Chroma's result order is not
guaranteed (20–22 of 881 queries came back in a different order in two runs with no writes, 0 in
three later runs; cause not established); the store's order is fixed.

## How to resume in a new session

1. Read this file, then `git log --oneline` since the last commit listed above.
2. Check the open-items table before starting any step it names.
3. When a finding changes the plan: add a D-entry, move items in the tables, commit with the work.
