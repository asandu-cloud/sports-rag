# Spix's game-read engine: design

Date: 6 Nov 2026, updated 15 Nov. Status: **approved by the owner on 15 Nov**, with their answers to the two
questions of 6 Nov and their direction of 14 Nov (Value is scored on the same bench). Tracker: D77.
Branch: `backend`. Plan for phases 0–1: `docs/superpowers/plans/2026-11-15-game-read-bench-and-ratings.md`.

## Why

Most likely today is the market's shortest prices. Each option's chance is the market's fair chance, and the list
is the top three by that chance that pay at least 1.30 and can win together. Nothing from Spix decides what is shown.
The owner: "Anyone can just go on a bookmaker's website and pick the lowest odds… that doesn't work." Spix's own
engine is "a priority because otherwise we don't differentiate ourselves".

Spix's own numbers cannot do that job yet:
- The current engine is hand-set averages fed into a fixed scoreline formula. Its ML layer has R² of about 0.02 and
  zero weight.
- The market beats it in all five priced markets (A4.2 reruns, 16 Oct to 3 Nov).
- When it disagrees with the market, it is mostly wrong (A5.1, D30).
- Stage 3 picks won 63.3% where Spix said 76.2% and the market 67.3%.
- Calibrating it does not help (A4: no calibrator passed in corners or cards, and the markets where one helped are
  now priced from the market).

So the engine has to be rebuilt from fitted models, and it must prove itself before it chooses anything.

**The owner's direction (14 Nov).** "The current engine tries to find bets for every single match, which is not
possible… We could explore using the proprietary engine for value betting or less bets but more value." The
product stays led by Most likely: `docs/identity.md` is written for casual bettors who look at every match, and
it names serious value bettors as not the audience. So one engine and one test bench serve both uses:
- **Most likely:** which of the market's likely options to put first.
- **Value:** fewer picks, each worth its price. A match with no value pick says so.

Value stays rare by design.

## What success looks like

The owner: "a higher win rate. that's it. the whole idea of the project is, if your judgement agrees with spix,
then its probably a good pick."

A higher raw win rate is easy to fake by picking shorter prices. So the measure is **wins above the market's chance**:
- run the same pool of options through today's rule and through the engine's rule;
- count each list's wins minus the wins the market expected (the sum of its fair chances);
- the engine passes when its lists beat today's on matches it has never seen.

The raw win rate is reported alongside.

For **Value**, the measure is money:
- the return per unit staked at the price when the pick is made;
- closing line value: how the pick's price compares with the fair price at kickoff.

The engine's Value list passes when it beats today's Value rule on both, on matches it has never seen. Fewer picks
is fine.

## Goals, non-goals, anti-goals

**Goals**
- A Spix engine built from fitted models that reads each game from team statistics.
- The engine chooses Most likely among the market's options at 50% or more that pay at least 1.30. The chance shown
  stays the market's.
- A test bench that judges every model the same way, on past matches with the prices of the time.
- Each pick's note comes from the reasons the engine chose it (replacing D75's form lines).
- Try several model types, as the owner asked: ratings, statistical models, boosted trees, a Bayesian model and an
  average of the best.
- A Value selector from the same engine, scored against today's Value rule (owner, 14 Nov).
- An "Opportunities" section in every report: anything the tests turn up that something outside our toolkit could
  use (owner, 14 Nov). See its own section below.

**Non-goals (for this design)**
- Changing the live Value list before a model passes the Value score. It stays market-based (A5) until then.
- A "Bolder reads" list for picks under 50%. It comes later, measured separately (owner, 5 Nov). Until then, sub-50%
  reads show in Explore the match, marked with their note.
- Player markets, lineups and absences. These come with the player engine (Part B, below).
- Neural networks. There is not enough data (owner, 6 Nov).
- Any change to the match page's design (owner: "The current design for the match reads inside of fixtures is
  fine").

**Anti-goals**
- Shipping a list that looks smarter but wins less than today's.
- A model that only won the test by luck: tuning on the locked seasons, or rerunning them until something passes.
- Showing Spix's own chance as the number on the card.
- Any published pick changing before a model passes the gate and the stage rule gives it a weekly slot.

## What we have to test on

| Source | What it holds | What it can test |
|---|---|---|
| `Index/football_data.sqlite3` (D2) | 49,714 league matches, 10 leagues, 14 seasons (2012/13 to 2025/26). Prices from before kickoff, collected Friday afternoons for weekend games and Tuesday afternoons for midweek games: result 49,684, main Asian handicap 49,590, over/under 2.5 49,682. Closing prices: result 49,661; handicap and O/U 2.5 from 2019/20 (about 25,000). Shots, shots on target, fouls, corners and cards on about 44,000; referee on 12,495 | Result, handicap and goals 2.5, with prices, on 14 seasons. Outcomes only (no prices) for corners, cards and both teams to score |
| The same matches linked to our database | 25,057 carry their API-Football fixture id | xG (from 2023), shots in the box (from 2019), player statistics (fouls drawn from 2021) |
| Our saved match reads (`platform.db`) | 694 finished matches (5 Sep to early Nov 2026) with every market we show, priced before kickoff; 157 are European cups; growing by about 100–150 a week | Both teams to score, corners, other goal lines, cards (after 10 Nov), and the European cups |
| API-Football player statistics (saved) | Per player per match: position, minutes, rating, shots, passes, tackles, interceptions, duels, fouls committed and drawn, cards. No pitch positions or heatmaps | Part B, and the "alternate data" track |

The 10 leagues are:
- Premier League and Championship;
- Bundesliga, Serie A, LaLiga and Ligue 1;
- Eredivisie, Primeira Liga, Belgian Pro League and Süper Lig.

The European cups are not in football-data. Teams from other leagues, such as Plzen or Lillestrom, have no history
there at all.

## The models

Numbered 1–9 so they don't clash with the rework's A-steps or D12's variable codes.

**Ratings, from results only (phase 1)**
1. **Elo.** The owner asked for it.
   - One number per team, updated after every match, with bigger wins moving it more.
   - Home advantage per league.
   - Ratings are carried over the summer with a pull towards the league average. Promoted teams start at the level
     of the teams they replace.
   - Win, draw or lose comes from the rating gap through a fitted model, so the draw chance depends on the gap.
   - The handicap comes from a fitted goal-difference model on the same gap.
2. **Pi-ratings.** Separate home and away ratings, measured in goals (Constantinou and Fenton, 2013). They give an
   expected goal difference, and from it the result and the handicap.
3. **Dixon–Coles ratings with time decay.** This is A6's base model (S1 and S4).
   - An attack and a defence rating per team, with home advantage per league and a promoted-team offset.
   - Fitted across all leagues at once, with older matches counting less.
   - It gives every scoreline, and so the result, the handicap, every goal line and both teams to score.
   - Fitted in-house and checked against penaltyblog (24 Oct finding).
4. **Drifting ratings.** Model 3, but each team's attack and defence move at their own pace over time, as after a new
   manager or an injury crisis (Rue and Salvesen 2000; Koopman and Lit 2015).

Models 1 and 2 give no goal totals, so they are tested on the result and the handicap only.

An outside yardstick, not built by us: **ClubElo's** published ratings, if its terms allow. If our Elo can't match a
free public Elo, that tells us something.

**Statistics and machine learning on top (phase 2)**
- **D12's variables** enter models 3 and 4 one at a time, each through the ablation gate (at least 0.5% better log
  loss, interval below zero, no slice worse by more than 2%):
  - xG ratings (S2) and shots-in-the-box ratings (S3);
  - rest and congestion (C1);
  - for cards and corners: the referee (D1), the foul matchup (D2) and corner pressure (D3).
- 5. **Ratings plus the market.** D20's blend, `logit p = a + b·logit p_market + c·logit p_model`, rerun on each
  rating model. It answers "does Spix earn a weight above zero?" (the D20/D21 open item).
- 6. **Boosted trees on the market's error** (A6c, LightGBM). The trees start from the market's chance and learn
  only corrections, from the D12 variables plus the Elo and pi-ratings as inputs.
- 7. **Boosted trees on the outcome**, with the market's chance as one input among the rest.
- 8. **A Bayesian hierarchical count model**, where thin-data teams borrow strength from their league. It is built
  last, and only if models 3–4 add anything to the market: it needs PyMC or NumPyro, a heavy install that is in
  neither environment today.
- 9. **The average of the models that pass.** A weighted average on the log-odds scale (the existing `blend.py`),
  with weights fitted on the tuning seasons.

## The test bench (how every model is judged)

1. **Walk-forward.** Before each matchday, every model is refitted on matches from earlier days only, then predicts
   that day. No prediction ever sees its own result or anything after it.
2. **Three blocks of seasons.**
   - **Fitting and warm-up:** 2012/13–2018/19.
   - **Tuning:** 2019/20–2022/23, about 14,400 matches. Every setting and every choice of variables is made here.
   - **Locked:** 2023/24–2025/26, about 10,700 matches. Each model version runs on it once, at the end, and every
     locked run is written into the tracker.
3. **The market at pick time** is football-data's pre-kickoff price, with the margin removed by the power method,
   as on the live site. For the handicap and over/under 2.5, football-data names only two bookmakers in most
   seasons (Bet365 and Pinnacle), and Pinnacle's prices are suspect from 2025/26. So for every market, the market's
   chance is the power-method fair price of football-data's average price (BetBrain's average before 2019/20). On
   the result, where many bookmakers are named, the bench reports how far this sits from the median of the named
   books, as a check. Closing prices are used only for the closing checks, never as an input.
4. **The Most likely pool on the history (finding of 15 Nov).**
   - football-data has one line per market: the result, the main handicap and over/under 2.5. With one line, at
     most one option in a market clears 50%, so a match has at most three eligible options, one per market.
   - With the owner's answer (a), today's rule and the engine's rule would then show the same options on nearly
     every match. The picks score would read zero whatever the engine does.

   So the bench rebuilds the live pool:
   - **Goal means.** M1, the market's goal model (A3.2), turns each match's market chances for the result, the main
     handicap and over/under 2.5 into a goal mean per team.
   - **Every line.** From these means comes the market's chance on every half line the live site shows: goals 0.5
     to 4.5, handicaps −2.5 to +2.5, and both teams to score. The real lines keep their real chance and price.
   - **Prices.** A line with no history gets a typical price from its chance and the margin our saved reads show
     for that market and line, put back with the power method in reverse.
   - **Corners** are left out. football-data has the counts but no prices.

   **The control line.** M1 is a plain Poisson model, which is known to be off on low scores. An engine with a
   low-score correction could "beat" M1 on lines such as under 1.5 or both teams to score just by fixing that,
   while real bookmakers already price it.
   - So a control line runs beside the zero line: M1's own goal means with Dixon–Coles's low-score correction.
   - On the derived lines, a model must beat the control line, not zero.
   - The real-price lines and the saved reads (point 9) are the check.
5. **Five scores per model and market:**
   - **Accuracy:** log loss against the market's, with intervals that resample whole weeks.
   - **Adds to the market:** model 5's blend. It passes when c is above zero and the blend beats the market alone,
     with the interval below zero, on the locked seasons.
   - **Most likely picks (the owner's measure).**
     - *Today's rule* uses the live code's own functions: one suggestion per market (the likeliest option with a
       chance of at least 50% and a price of at least 1.30), then up to 3 that can win together.
     - *The engine's rule* follows answer (a): within each market and across markets, the options Spix backs come
       first, strongest first, then the rest by market chance.
     - *The score* is wins minus the market's expected wins, per 100 picks.
     - *It passes* when, on the locked seasons:
       - the engine's lists beat today's with the interval above zero;
       - they beat the control line on the derived lines;
       - no league is worse by more than 2 wins per 100 picks.
   - **Value.**
     - *Today's Value rule* is A5.2, live since stage 4: the market's fair chance against the best price, at least
       2% value.
     - *The engine's rule* is the engine's chance against the price, at least 2% value.
     - *Two prices.* Each rule is scored at football-data's best price, which is what the live rule uses but holds
       stale prices (half the live best prices were gone when checked, D39). It is also scored at the average
       price, which any bettor can get.
     - *Scores:* the number of picks, the return per unit, and closing line value against the fair price of the
       closing average.
     - *It passes* when, on the locked seasons and at the average price, closing line value has its interval above
       zero and the return is not below zero, on at least 300 picks.
   - **Second check:** do closing prices move towards the engine's Most likely picks more than towards today's?
6. **A zero line.** A "model" that copies the market must score zero on the blend and the Most likely score, and
   make no Value picks at the average price. Every run checks this first. If it fails, the bench is wrong.
7. **Two ranking rules,** reported side by side: by edge (the engine's chance minus the market's, in points) and by
   the blended chance. The owner chooses with the results in front of them.
8. **A guard on the locked seasons.**
   - The bench refuses a locked run unless the run is named as one.
   - It writes every locked run (date, model, code version, result) to a ledger file.
   - Each locked run also goes into the tracker.
9. **The other markets and the cups** run on the saved reads with the same scores.
   - Their models can still learn from 14 seasons of outcomes: corner counts on about 44,000 matches, and both teams
     to score from the scores.
   - Only the comparison with the market needs the saved reads, so their intervals stay wide until enough weeks
     build up.
   - The saved reads have real prices on every line, so they also check the derived lines of point 4.

**An honest expectation.** Published studies find that rating models on their own score worse than bookmaker prices
(for Elo, Hvattum and Arntzen 2010). Our chance is in adding to the price at pick time, which is hours or days before
the close, in some leagues or markets. It is possible that nothing passes. Then Most likely stays as it is, and we
know where the engine stands. That is a result, not a failure.

## Phases and timing

Code starts when the owner lifts the code freeze of 14 Nov, after reading the phase 0–1 plan. Phases 0–3 change
nothing on the site, so they run alongside the cards weekend of 20–22 Nov and its Mon 23 Nov scorecard.

| Phase | What | Rough time | Changes published picks? |
|---|---|---|---|
| 0 | The test bench: history loader, the rebuilt Most likely pool, walk-forward runner, the five scores, today's two rules simulated, the zero and control lines, the locked-run guard | 4 working days | No |
| 1 | Models 1–4 and the ClubElo yardstick on result, handicap and goals 2.5, plus their average. First report to the owner | 1 week; the first report about two weeks after code starts | No |
| 2 | D12 variables and models 5–8, then the average (9) | 2 weeks, to early December | No |
| 3 | The other markets and the cups on the saved reads, rerun weekly as matches build up | Alongside phase 2 | No |
| 4 | The selector behind `SPIX_GAME_READ` (off), notes from the engine's reasons, a 2–3 weekend shadow run graded in the owner dashboard, staging on 8100, then a free weekly slot | From early December; live mid-December at the earliest, only for the markets and leagues that passed | Yes, one weekly slot |

Each phase ends with a short report to the owner. Phase 2 runs whatever phase 1 shows, because the owner wants the
machine-learning versions tested. Phase 4 starts only for what passed: Most likely, Value, or both.

## Opportunities (a section in every report)

The owner, 14 Nov: "always look for possibilities during testing, we might find out something that could
potentially be exploited by something that's not currently in our toolkit… I want to explore those as well."

Every bench report ends with an Opportunities section. An opportunity is a pattern the bench sees that no model
in this design uses, for example:
- a league, market or day of the week where the price before kickoff is regularly beaten by the close;
- a kind of team (promoted, after a manager change, in a congested week) the market misprices for a while;
- a part of the price range where favourites or longshots win more or less than their price says;
- a data source that would explain what the models get wrong.

Each one is written down with its evidence, its interval and how many matches it rests on. An opportunity is
found on the tuning seasons and never on the locked ones, and it changes nothing by itself. Anything worth
following becomes a new candidate that goes through the same bench and gate. The owner decides which ones to
follow. Ideas that fall outside Spix go to a separate track, like the late-goal study (D103).

## Alternate data (a track alongside phases 1–3)

Before any download, each source is checked for:
- coverage of our 13 competitions;
- at least three seasons of history;
- whether it is known before kickoff;
- cost;
- terms of use.

A source that clears these then goes through the same ablation gate. Anything paid, or any new account, is the
owner's call.

| Source | What it could add | First use | To check |
|---|---|---|---|
| ClubElo | Daily Elo for European clubs over decades, linked across leagues by European matches | Yardstick for model 1; ratings for cup teams from leagues we don't hold | Terms; coverage of Conference League teams |
| Understat | Shot-by-shot xG for the top five leagues since 2014 | S2 xG ratings tested on 10 seasons, not 3 | Terms (a website with no official API); top five leagues only |
| Our own player statistics | Position, duels, tackles, fouls committed and drawn, cards per player | Who is in the duels: the question the owner's heatmap idea asks, for player fouls and bookings (Part B) | Seasons covered per league |
| Player positions or heatmaps | Where each player plays and makes contact | Player fouls and bookings (the owner's idea) | Held by event and tracking providers (Opta, StatsBomb, Wyscout; Sofascore shows heatmaps): paid or restricted terms; StatsBomb's free data covers few competitions. Revisit at Part B if the proxies above show player bookings can be predicted |
| Squad market values | Squad strength, a known strong predictor in research | An input to models 6–7 | Terms: scraping may not be allowed |
| Price movement up to publication | How the market moved before our pick (D12's M2) | An input to models 6–7 | Needs our own archive (since Sep); football-data's movement runs past pick time, so it can't be used |
| Travel distance, weather | Away travel in the cups; heavy rain or wind | Small inputs | Low expected gain; weather terms |

## How the player engine joins later (Part B)

When the player models exist (generalised linear models first, boosting as a challenger), they join through the
same bench and gate:
1. **Who plays.** Confirmed lineups (about an hour before kickoff) and absences adjust each team's ratings. The
   picks can update then.
2. **Player markets.** Anytime scorer, player booked, and player shots enter the pool with the market's chance. One
   at 50% or more can make Most likely; under 50% goes to Bolder reads.
3. **Picks that fit together.** "Haaland to score" must fit the same scoreline story as "City to win"; the
   win-together check is extended to players.
4. **Notes.** For example: "has scored in 6 of his last 8 starts; their first-choice centre-back is out."

Player-market prices have almost no history (goalscorer prices for 96 matches in the odds archive). Saving them for
every match from now would give Part B something to be tested on. The request cost has to be checked first, and the
owner decides.

## What this changes elsewhere

- **A4 calibration:** closed as a stage of its own (owner, 15 Nov). The weekly A4.2 reruns stop once the bench
  runs.
- **A5:** unchanged live. The A5.4 cards switch is live since 9 Nov, and the Value list stays market-based. If an
  engine-based Value rule passes the bench, it takes its own weekly slot.
- **A6 and A6c:** this design is A6 and A6c, widened with Elo, pi-ratings, drifting ratings, the Bayesian model and
  the average. D14's order is kept: A6c comes after the base ratings.
- **D20 and D21:** model 5 is the blend study rerun. The live c stays at 0 until a model passes.
- **D75 and D76:** pick notes will come from the engine's reasons in phase 4. `SPIX_PICK_NOTES` stays off, and the
  frontend's D76 display waits.
- **A7:** the new Elo replaces `ml_edge.py`'s Elo and `Index/elo_ratings.json`.
- **The stage rule:** phases 0–3 change no published pick. Phase 4 takes one weekly slot and is judged at the
  Monday scorecard.

## Constraints

- **The safety rules in `CLAUDE.md`.** `platform.db` is read only inside `data_access()`, never during the refresh,
  in short read-only statements. Saved reads are exported once per run into a research file.
  `football_data.sqlite3` is opened read-only.
- **The golden check stays unchanged** while `SPIX_GAME_READ` is off.
- **Downloads are untrusted.** Each goes in its own folder, and any Python that reads them runs with `-I`. No paid
  data or new accounts without the owner.
- **The API-Football budget** is about 7,500 requests a day, so any backfill is spread over days.
- **Public copy says "statistical models"**, and notes make no promises.
- **Bench output** goes to `Research/game-read-bench-<date>/`, which is not committed. Each run's summary goes into
  the tracker.

## Decided by the owner

- Spix's read chooses Most likely among options at 50% or more that pay at least 1.30. The chance shown is the
  market's (5 Nov).
- Success is a higher win rate, measured as wins above the market's chance, against today's list (5 Nov).
- Reads under 50% go to Explore the match now; a "Bolder reads" list comes later (5 Nov).
- The engine comes first. A5 and A6 go ahead, and several machine-learning specifications get tested (5 Nov).
- Try Elo and similar ratings, at least for the winner (6 Nov).
- No neural networks for now (6 Nov).
- Lineups, injuries and absences come with the player model (5 Nov).
- The match page's design stays (5 Nov).
- When Spix backs none of a match's options, Most likely still shows up to 3: the options Spix backs first,
  strongest first, then the rest by market chance (answer (a), 14 Nov).
- Start with the 10 domestic leagues. The cups keep today's rule until phase 3 (14 Nov).
- Most likely leads the product. Value is scored on the same bench, and fewer Value picks is fine (14–15 Nov).
- Every report has an Opportunities section (14 Nov).
- A4 calibration is closed as a stage of its own (15 Nov).

## My calls (for the owner to check)

- Locked seasons 2023/24–2025/26, tuning seasons 2019/20–2022/23.
- Historical tests use half lines only, to match the live pool. The 1.30 floor uses football-data's average price
  where a line has one, as the live list's typical price uses the middle of the books' prices.
- The market's chance comes from football-data's average price for every market (point 3 of the bench).
- Lines with no history are priced from M1 and the margins of our saved reads, with a control line against M1's
  known low-score error (point 4).
- The Value pass rule: at the average price, closing line value above zero with its interval, return not below
  zero, at least 300 picks.
- Intervals resample whole weeks, because matches on the same weekend are not independent.
- The edge is measured in percentage points.
- Model 8 comes last, and only if models 3–4 add anything.
- ClubElo and Understat are the first sources to check.
- Code goes in `Scripts/data_platform/game_read/`, run by `Scripts/ops/game_read_bench.py`.

## Left to the builder

- How often to refit within a matchday week.
- Optimisers.
- Elo and pi-rating constants (tuned on the tuning seasons).
- How drifting ratings are fitted (a filter or a Bayesian fit).
- LightGBM settings.
- The report layout.

## Open questions for the owner

None open. The two questions of 6 Nov were answered on 14 Nov (see "Decided by the owner"): (a), and start with
the domestic leagues. New questions go to the owner as the reports raise them.
