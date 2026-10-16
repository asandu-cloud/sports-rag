# Engine rework tracker

The single source of truth for the rework's **current** plan, status, decisions and
deviations. The Engine Rework report (claude.ai artifact BHP6rm7e4Xf5a29KGoCiUe) is the
original design; where this file differs, **this file wins**. Update it in the same
commit as the work it describes; the report is refreshed from it at milestones.

Last updated: 2026-10-16 (D22)

## Status

| Step | Status | Commits |
|---|---|---|
| A0 Safety net | Done 15 Oct | 02fa536, f7459f0, c1e9b42 |
| A1 Measurement | In progress (steps 1, 2, 3, 5, 6 done; step 4 left) | see below |
| A2 Canonical features | A2a built 16 Oct, switch after the 17 Oct refresh check (D11); A2b after A6 (D13) | 7af457b |
| A3 Market anchor and honest uncertainty | Started 16 Oct under D16 (research and shadow only) | see A3 table |
| A4–A8, Part B | Not started | |

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
| 3.5 | Phase 4 dispersions (corners 0.025, SoT 0.01) replace the variance floors, behind a switch | **Next** (D22: ~1.0% / 1.4% better than the floors) | No: switch default off until go-live |
| 3.6 | Dixon–Coles rho per league | Open | No |
| 3.7 | Shadow run, then go-live after A1's gate is closed and A2a is switched | Open | – |

## Open items with a destination

Things found along the way that must not be lost. Each names where it gets done.

| Item | Goes to | Why deferred |
|---|---|---|
| Settlement for player picks (participant) and period backfill | Part B | No current pick needs it (0 affected) |
| Unit-bet slip link to the legacy `predictions.db` | A7 | Lives in the Discord legacy tracker A7 replaces |
| Move Phase 6 shadow controller from `Research/` into `Scripts/` | After 20 Oct 2026 | Its diagnostic window runs to 20 Oct |
| Push tag `pre-rework` (cf00af8) | Owner | Local only |
| Reference weights replace the median in `reference_price.py` | A1 step 4 | Method is versioned (`proportional_devig_median.v1`) |
| Make power the default margin-removal method (`reference_price`, closing consensus, publication reference, scorecard) | A3.7 go-live, after A1's gate (D16, D17) | Refit anything fitted on v1 |
| Scorecard switches to the weighted reference when step 4 lands | A1 step 4 | |
| Refresh the report artifact from this tracker | At end of A1 | |
| Switch production to the document store: add `DOCUMENT_STORE=sqlite` to `.env` | Owner, after A2a review | Chroma remains the fallback (remove the line) |
| Stop embedding into Chroma entirely | A7 | Whale chat and `rag_cli_v2` still search Chroma by meaning |
| A6c LightGBM market-error challenger (D14) | After A6's base model passes | Must beat fitted A6 vs the market baseline |
| Rerun the blend study (`Scripts/ops/blend_study.py`) on A6's forecasts: does Spix earn c > 0? | A6 gate | D20 |
| **Temporary (D21):** 1X2 and O/U 2.5 publish the market's probability (power) with c = 0 **only until A6**; rerun the blend study on A6's forecasts and refit c if Spix earns it | A3.7 go-live, then A6 gate | Not permanent: revisit at A6 |
| Build OddsPapi true-close capture (Pinnacle ± 2 books), API-Football snapshot as fallback (D18) | A1 step 4, Mon 19 Oct | Goes live after the A2a switch (D16 rule 3) |
| Check whether OddsPapi's request count resets monthly (vendor says 250/month; unconfirmed) | ~14 Nov 2026 | 14 used by 16 Oct; ~50/month planned |
| Quote age near kickoff in selection (API-Football prices ~2 h old in the last 3 h, D15) | A5 | Apparent edge may be a price no longer offered |
| Retire `ml_edge.py` and `Index/elo_ratings.json` | A7 | Replaced by A6 |
| Evidence-aware variance (`uncertainty.py`, D22) for fitted ratings' uncertainty | A6 | Not needed with today's means |
| Cards and goals variance floors (no qualified dispersion yet) | A6 | Cards outside research qualification; goals use the score matrix |
| The n/(n+8) early-season prior on means | A6 | 3.4 changed widths only, not means |
| First-round variables (D12): M1 | A3, A8 | Selected 16 Oct |
| First-round variables (D12): S1–S4, C1, D1–D3 | A2b builder, A6 models | Selected 16 Oct; each must pass the ablation gate |
| Second-round variables: M2 line movement | A3 blend | Needs 6–8 weeks of odds archive (from ~mid-Nov 2026) |
| Second-round variables: M3 bookmaker disagreement | A5 confidence labels | With A5 |
| Not selected for now: C2 stakes, C3 derby, L1 lineup strength, L2 player load | Revisit at Part B (L1, L2) or after A6 | Owner chose market, strength/form, discipline/set pieces + C1 |
| Correct `CLAUDE.md`: Elo "signal active at 0.15" (never called) and ML R² 0.43/0.47 (saved R² ≈ 0) | Next docs pass | Stale claims mislead new sessions |
| Verify the store after the first refresh with the new stages (17 Oct 07:15), then switch | Claude, then owner adds `DOCUMENT_STORE=sqlite` | Parity on the 881 engine queries + counts |

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

## Findings that shape later steps

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
