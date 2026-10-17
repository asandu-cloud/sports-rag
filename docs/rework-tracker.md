# Engine rework tracker

The single source of truth for the rework's **current** plan, status, decisions and
deviations. The Engine Rework report (claude.ai artifact BHP6rm7e4Xf5a29KGoCiUe) is the
original design; where this file differs, **this file wins**. Update it in the same
commit as the work it describes; the report is refreshed from it at milestones.

Last updated: 2026-10-17 (D32)

## Status

| Step | Status | Commits |
|---|---|---|
| A0 Safety net | Done 15 Oct | 02fa536, f7459f0, c1e9b42 |
| A1 Measurement | In progress (steps 1, 2, 3, 5, 6 done; step 4 left) | see below |
| A2 Canonical features | A2a live since 17 Oct 16:28 (`DOCUMENT_STORE=sqlite`; D11); A2b after A6 (D13) | 7af457b |
| A3 Market anchor and honest uncertainty | Started 16 Oct under D16 (research and shadow only) | see A3 table |
| A4 Calibration | A4.1 dataset done 16 Oct (25,527 graded candidates, 160 fixtures; grading matches settlement on all 1,918 overlapping picks). A4.2 method study done 16 Oct: no calibrator adopted yet (D28), rerun weekly; live as A3.7 stage 5, not before Tue 10 Nov (D29) | 197dc73; see D28 |
| A5 Selection policy | Research started 16 Oct (D30): A5.1 done (Spix's disagreement with the market is almost all error); A5.2 done (pricing against the market wins, D31); A5.3 next (are the best prices real?); live as A3.7 stage 6, earliest Tue 17 Nov | see D30, D31 |
| A6–A8, Part B | Not started | |
| Market expansion (D25) | Planned after A5 | see D25 |

### A1 (order changed 15 Oct, see D6)

| # | Task | Status | Commit |
|---|---|---|---|
| 1 | Handicap grading fix | Done | dfb7f93 |
| 2 | Closing prices from API-Football's final pre-kickoff snapshot | Done | aa2f2ee |
| 3 | Market reference price at publication | Done | 1854f23 |
| 4 | Bookmaker weights for the reference price | **In progress** 16 Oct: historical accuracy + spot-check script now; spot checks Sat/Sun 17–18 Oct; freshness rerun and weights Mon 19 Oct | |
| 5 | Weekly scorecard | Done (Mondays 08:00) | d1ae597 |
| 6 | Market baseline in the research protocol | Done | see D9 |

A1 gate (report): >90% of finished picks settled and with a closing price ✓ (95.9% closing,
handicap regrade finishing 15 Oct night); first scorecard ✓; first reference weights with
Pinnacle's freshness measured ✗ (step 4).

### A3 (started 16 Oct, ahead of A1 step 4 and the A2a switch; rules in D16)

| # | Task | Status | Live? |
|---|---|---|---|
| 3.1 | Margin-removal method: proportional vs additive vs power vs Shin on football-data closing history | Done: power (D17) | No: option `method="power"`, default unchanged until A1's gate |
| 3.2 | M1 market goal expectations per team (handicap + totals) | Done: research (D19) | No: `market_goals.solve`; used by 3.3 and A8 |
| 3.3 | Fitted blend `logit p = a + b·logit p_market + c·logit p_spix` per market | Done: research; gate not passed with the 2023 engine (D20) | No: at go-live, c = 0 for 1X2/O/U 2.5 **until A6** (D21) |
| 3.4 | Gamma-distributed team rates replacing n/(n+8) and the variance floors | Done: research; not adopted (D22) | No: the simpler 3.5 wins |
| 3.5 | Phase 4 dispersions (corners 0.025, SoT 0.01) replace the variance floors, behind a switch | Done: built (D23), all competitions and stages | No: `SPIX_COUNT_DISPERSION=fixed` at go-live |
| 3.6 | Dixon–Coles rho per league | Done: research; not adopted, rho stays −0.10 (D24) | No change |
| 3.7 | Go-live package, staged (D26): build by Mon 19 Oct; stage 1 dispersions Tue 20; stage 2 power margin removal in selection Fri 23; stage 3 market anchor (strict scope) Tue 27; confirm at the Mon 2 Nov scorecard; stage 4 market anchor for all goal markets via M1, earliest Tue 3 Nov (D29); stage 5 A4 calibrators, earliest Tue 10 Nov | Package build started 17 Oct: switch record done (`core/engine_switches.py`); stage 2 switch built (`SPIX_DEVIG_METHOD=power`); stage 3 switch built (`SPIX_MARKET_ANCHOR=strict`) | Staged, one change at a time |

## Open items with a destination

Things found along the way that must not be lost. Each names where it gets done.

| Item | Goes to | Why deferred |
|---|---|---|
| Settlement for player picks (participant) and period backfill | Part B | No current pick needs it (0 affected) |
| Unit-bet slip link to the legacy `predictions.db` | A7 | Lives in the Discord legacy tracker A7 replaces |
| Move Phase 6 shadow controller from `Research/` into `Scripts/` | After 20 Oct 2026 | Its diagnostic window runs to 20 Oct |
| Push tag `pre-rework` (cf00af8) | Owner | Local only |
| Keep the Mac awake for scheduled jobs: the daily refresh runs under `caffeinate -i -s` since 17 Oct (it cannot idle-sleep mid-run); still open, a daily scheduled wake before 07:15 (`sudo pmset repeat wakeorpoweron MTWRFSU 07:05:00`), mains power, or a server. On 17 Oct the Mac slept on battery until ~11:48, waking for 2–3 s every ~17 min: the 07:15 refresh ran only in those slices and finished at 12:25 (stage times normal). While asleep the Match Read worker, measurement worker and odds archive do not run either, which matters for this weekend's near-kickoff captures (A1 step 4 freshness rerun, Mon 19 Oct) | Owner | Operational risk to every scheduled job |
| Reference weights replace the median in `reference_price.py` | A1 step 4 | Method is versioned (`proportional_devig_median.v1`) |
| Make power the default margin-removal method (`reference_price`, closing consensus, publication reference, scorecard) | A3.7 go-live, after A1's gate (D16, D17) | Refit anything fitted on v1 |
| Scorecard switches to the weighted reference when step 4 lands | A1 step 4 | |
| Refresh the report artifact from this tracker | At end of A1 | |
| Stop embedding into Chroma entirely | A7 | Whale chat and `rag_cli_v2` still search Chroma by meaning |
| A6c LightGBM market-error challenger (D14) | After A6's base model passes | Must beat fitted A6 vs the market baseline |
| Rerun the blend study (`Scripts/ops/blend_study.py`) on A6's forecasts: does Spix earn c > 0? | A6 gate | D20 |
| **Temporary (D21):** 1X2 and O/U 2.5 publish the market's probability (power) with c = 0 **only until A6**; rerun the blend study on A6's forecasts and refit c if Spix earns it | A3.7 go-live, then A6 gate | Not permanent: revisit at A6 |
| Build OddsPapi true-close capture (Pinnacle ± 2 books), API-Football snapshot as fallback (D18) | A1 step 4, Mon 19 Oct | Goes live after the A2a switch (D16 rule 3) |
| Check whether OddsPapi's request count resets monthly (vendor says 250/month; unconfirmed) | ~14 Nov 2026 | 14 used by 16 Oct; ~50/month planned |
| Quote age near kickoff in selection (API-Football prices ~2 h old in the last 3 h, D15) | A5.3 (D30) | Apparent edge may be a price no longer offered |
| Retire `ml_edge.py` and `Index/elo_ratings.json` | A7 | Replaced by A6 |
| Evidence-aware variance (`uncertainty.py`, D22) for fitted ratings' uncertainty | A6 | Not needed with today's means |
| Go-live of 3.5: set `SPIX_COUNT_DISPERSION=fixed`; review the 59 corner pick changes in the golden preview, then re-record (D16 rule 4) | A3.7 stage 1, Tue 20 Oct | Confirm on the live scorecard (development-period evidence) |
| Scorecard splits picks by the engine switches recorded on them (`tracking.engine_switches`, from 17 Oct; earlier picks ran on the defaults) | Before the Mon 2 Nov scorecard | The record exists since 17 Oct (A3.7 package); new A3.7 switches register in `core/engine_switches.py` |
| Go-live of stage 2: set `SPIX_DEVIG_METHOD=power` (built 17 Oct, off). Golden preview with stage 1 off: no pick added or removed; 4 of 54 fixtures move a pick to a better-priced quote (3 corners, 1 goals; one corners line 9.25 → 9.5), one Asian handicap label medium → low; the fair probability shown changes on 293 results. Preview again on top of stage 1, then re-record (D16 rule 4) | A3.7 stage 2, Fri 23 Oct | Selection's four margin-removal sites (totals, handicap, BTTS, 1X2) use `data_platform/devig.power` when on |
| Go-live of stage 3: set `SPIX_MARKET_ANCHOR=strict` (built 17 Oct, off). 1X2 and goals O/U 2.5 take the market's consensus (median across books of each book's fair price, with selection's margin removal, so it assumes stage 2 is live; 1X2 rescaled to sum to one); Spix's own probability kept as `_spix_prob` / `spix_prob` and a `market_anchor` record in the result context. Golden preview, stage 3 alone: all 7 1X2 picks in the 54 fixtures become no-bets (at the market's number no price clears the 1X2 thresholds); the only O/U 2.5 pick of 38 goals picks moves to over 2.75, still priced on Spix's numbers (strict scope; stage 4 closes that). Preview again on top of stages 1–2, then re-record (D16 rule 4) | A3.7 stage 3, Tue 27 Oct | Other goal lines, Asian handicap and BTTS stay on Spix's probability until stage 4 (D29) |
| Extend the market anchor to all goal markets via M1's score matrix (other O/U lines, Asian handicap, BTTS), behind its own switch; add BTTS from the score matrix to `market_goals` | A3.7 package (built switched off); live at stage 4, earliest Tue 3 Nov (D29) | Beyond what history tested: must pass the gate on audited lines first (D29) |
| A5.3 first: are the best prices real? Quote age at the read (odds archive `source_age_seconds`), OddsPapi spot checks of the best-price book, exchange commission; then evidence-based labels | A5.3, next (D31) | Half the best-price books' quotes were unchanged or missing at the close (D31) |
| Bookmaker sets per country: the best price only from books a member can legally use, and the first country (owner) | A5.4 requirement (D32) | A third of A5.2's best prices came from books most European bettors cannot use |
| Red cards: the selector also outputs the bets to skip (prices below fair, boosts that are not, high-margin bet builders) | A5.4 (D32) | Part of the Daily Card |
| Bet builder and combination pricing from the score model (correlated legs), anchored to the market | After A5, with the D25 expansion (D32) | Price Check and red cards need it |
| Price Check for any single and the Ledger (member bet tracking, settlement, overpayment vs best price) | Product backlog after A5 (D32) | Web app work; reuses settlement and reference prices |
| How to show Spix's own view when it disagrees with the market (≥10 pts on 3.4 markets per read; ≥25 pts on half the reads), given A5.1 found it mostly error | Owner, before A5.4 | Product decision (D31) |
| If A5.4 prices corners and cards against the market too, A4's calibrators (stage 5) lose their purpose except where no two-sided price exists | Decide with A5.4 (D31) | Calibrators fit Spix's own probability |
| A5.4: build the chosen policy behind a switch (`line_selection`, `match_read_compiler`, `confidence_from_edge`), golden preview; proposal in D31 | A5; live at A3.7 stage 6, earliest Tue 17 Nov | Gate (report): published picks' calibration error ≤ 3 pts and closing line value at least neutral over 4+ live matchweeks |
| Rerun the A5.1 study (`python -m Scripts.ops.disagreement_study --out Research/disagreement-<date>`) as weeks accrue | A5, with each A5 step | One audited week so far (D30) |
| Rerun the A4.2 study weekly (`python -m Scripts.ops.calibration_study --out Research/calibration-<date>`); calibrator decision with ≥3 weeks of audited fixtures | A4, decision at A3.7 stage 5 (not before Tue 10 Nov, D29) | Not adopted 16 Oct: one audited week, no market passes the gate (D28) |
| Qualify the card settlement definition (yellow + red vs bookmakers' bookings rules, second yellows) before calibrating or scoring cards | A4, before stage 5 | Cards are graded yellow + red (indicative); the market's own card slope is 1.40 on that grading (D28) |
| Cards and goals variance floors (no qualified dispersion yet) | A6 | Cards outside research qualification; goals use the score matrix |
| Market expansion wave 1: team goal totals, double chance, draw no bet (D25) | After A5 | No new modelling; anchored by M1 |
| Market expansion wave 2: team corners totals, corners handicap (D25) | After wave 1 | Team corner projections exist; priced on ~half of fixtures |
| Market expansion wave 3: first-half goals O/U and handicap (D25) | After wave 2 | Needs a half-split model (football-data half-time scores) and storing half-time scores in `fixtures` |
| The n/(n+8) early-season prior on means | A6 | 3.4 changed widths only, not means |
| penaltyblog (MIT licence, Cython) for A6's Dixon–Coles ratings: check time-decay weights, per-league home advantage, promoted-team offset and clean install; use pinned if it covers them, else in-house fit validated against it | A6 start | Not used in A3: margin removal in-house (D17); 3.6 fits rho only, with the engine's own Dixon–Coles code |
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

## Findings that shape later steps

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
