# Bet-building assistant — design (D128)

Owner decision 24 Nov 2026, from the plan "Build the bet, not the model" (18 Nov,
https://claude.ai/artifact/4ij8vSVmfHEpMPsZX32VN5). Status: design, waiting for the owner's approval. No code yet.

## Intent and why

Spix stops trying to out-predict the market and helps the Sunday bettor (identity, section 2) build a slip of their
own in a few minutes. The slip should win often enough to stay fun and be worth showing a friend.

- **No model beats the market.** Game-read phase 2 found none of six versions passed (D77). Spix's chances are
  already honest because they are the market's: Most likely said 68.9% and landed 68.6% (9 Oct – 8 Nov).
- **The missing piece is the help between a chance and a bet:** which legs, how many, how they fit together.
- **Two slips at the same chance can cost very different amounts.** The bookmaker keeps more on long prices, on
  corners and with every extra leg. A builder that picks the cheaper route to the same payout gives the bettor more
  back without knowing more than the market:
  - on 11 seasons (2012/13–2022/23), aimed at about 1 in 3, it gives back 96.2 of every 100 and lands a win in 84% of
    months;
  - a casual four-leg slip gives back 86.1 and lands a win in 27% of months;
  - stacking today's Most likely picks at the same chance gives back 4.0 less.
- **The business case:** in 10,000 simulated futures the assistant came first in 70%, reaching about €1,447 a month
  by January 2029 against €1,043 for ready-made slips and €466 for chasing an edge (no paid marketing).

## The owner's six choices (24 Nov)

1. **Direction:** the bet-building assistant. A ready-made slip appears only as a starting point to edit.
2. **The builder's default size:** about 1 in 3. The bettor can change it.
3. **"A leg on every match I picked":** off by default, one tap to turn on. The cheapest slip comes first.
4. **Engine work:** game-read phase 3 onwards is parked. The bench stays as the yardstick for any later idea.
5. **The €5,000-a-month goal:** a date is set after 8 weekends with real members.
6. **Most likely with cost in mind:** tested on the owner dashboard first. It goes live only if the owner says yes,
   in its own stage week.

## Goals

- "Build my bet": the bettor picks matches and how big a win (about 1 in 2, 3, 4 or 6). Spix fills a slip from its
  own options. The bettor swaps, removes or adds any leg, and "about 1 in N" updates.
- Each option's cost is stored with the recommendations block and kept under the hood (identity, section 8). The
  bettor sees a chance in words and a typical price, never "you are overpaying".
- Same-match goal legs are priced from a score grid that is right on draws and "both teams score" before any
  same-match slip shows "Worth it at".
- The study behind the plan becomes a weekly check, so we know the builder still earns its place.
- Later: a share card and a Monday "How your slips did", losses included.

## Non-goals

- No new models, no new Value work, no new markets, no player props (parked, per choice 4).
- Corners or cards are never combined with goals from the same match until D51's factor is measured.
- No ready-made slip as the paid product.

## Anti-goals: it fails even if it works

- **It pushes people to bet more:** reminders, streaks, stake suggestions, or nudges towards bigger slips. None are
  built, and nothing we measure rewards them. Slips per member, stakes and slip size are never targets.
- **"About 1 in 3" reads as a promise.** Chances are in plain words, losses are shown next to wins, and 18+ and a help
  line sit on every slip and card.
- **It reads like a tipster.** No "lock" or "banker", and the bet stays the bettor's (identity, section 6).
- **It breaks a published pick.** Nothing in phases 1–3 changes Most likely, Value or the engine. The golden check
  must stay unchanged.

## Constraints

- **The stage rule (D26, D29).** Only step 4 (Most likely with cost) changes published picks, in its own week, not
  before Tue 8 Dec. Everything else ships behind a switch that starts off.
- **Locked seasons.** Seasons from 2023 on stay locked for the engine's checks. The weekly check uses history before
  2023 only, and the grid fix is fitted on 2019–2022 only.
- **Data format.** `recommendations.v1` changes only on the backend branch, and only by adding fields. The frontend
  asks there for anything it needs.
- **Public copy rules.** "Statistical models", never "AI". No profit or win-rate promises. 18+ and help lines.
- **No new paid data source.** Everything comes from the reads Spix already stores.

## Approaches considered

**What to build** (the owner picked the first):

| Approach | Trade-off |
|---|---|
| **Bet-building assistant** | Passes all five identity questions; first in 70% of futures; most word of mouth |
| Ready-made slips | Simple, but fails identity question 2 (a slip to copy is a tipster's product); everyone wins or loses together |
| Research hub | Strong data, but it is homework, the thing Spix exists to remove |
| Keep chasing an edge | No model passes; Value has nothing to bet on in 57% of weekends |

**Where the builder runs.** This is my call, stated so the owner can correct it:

| Option | Trade-off |
|---|---|
| **A. A backend function and a web route** (chosen) | One source of truth, tested in Python, and shared with the weekly check and the dashboard test. One request per build. |
| B. In the browser, beside today's slip code | No server work and instant, but the search and the grid would live twice (JS and Python), and the weekly check could not reuse it. |
| C. Slips worked out ahead for each board | Cheap to serve, but only ready-made slips: it can't take the bettor's own matches and size. |

The grid moves the same way. The slip's `jointChance` in `app.js` stays as the display's quick check. The builder and
"Worth it at" use the corrected Python grid, and a test keeps the two in step on the four main pairs.

## The design

### 1. Option cost (backend, additive)

`recommendations.py`: every option in `markets[].options` gets `"cost"`, the share of each stake the bookmaker keeps
at the typical price: `1 − chance × typical_price`, rounded to 3 places. The display never shows it. Most likely and
Value are untouched.

### 2. The score grid, corrected (backend)

Today the grid is two independent goal counts from the fitted rates (`score_grid`). On 16,000 historical matches it
rates the draw about 2 points low (23.5% against 25.3%) and "both teams score" about 2 points low (50.6% against
52.8%). A new `Scripts/data_platform/score_grid.py` builds the scoreline table with a low-score correction (Dixon–Coles
ρ, one value, or one per league if that is clearly better), fitted on 2019–2022 matches only.

Target: within 2 points on every common pair of result and goals legs. Until it meets that, same-match slips show no
"Worth it at".

### 3. The slip builder (backend)

`Scripts/data_platform/slip_builder.py`, a pure function. It takes the recommendations blocks of the chosen matches,
a target chance, "a leg on every match" on or off, and any legs the bettor has fixed. It returns up to 3 slips, each
with its legs, joint chance, typical combined price and cost.

- **Legs:** options of the layer's markets with a chance from 0.25 to 0.92. At most one leg per match, or two goal
  legs from one match through the grid. Corners never go with goals from the same match.
- **Joint chance:** matches multiply; same-match legs come from the grid.
- **Near the size:** the joint chance is within ±20% of the target (for about 1 in 3, between 0.27 and 0.40). With
  too few matches, it says so and offers the nearest size.
- **Choosing:** the lowest cost wins. Within 0.5 points of cost, prefer more different markets, so slips are not
  always favourites to win. Then prefer fewer legs.
- **Search:** exact over each match's options (choose one, two or none), keeping the best few slips per chance band.
  The plan pins the algorithm and a time limit (target under 200 ms for 10 matches).

### 4. The web route (backend, behind a switch)

`POST /api/slip/build`, only while `SPIX_SLIP_BUILDER=on` (off at first, so `.env` needs no line until launch). It
reads the stored reads of the chosen matches and never calls a bookmaker. It answers with legs, "about 1 in N",
typical price, and "Worth it at" for same-match slips once the grid passes. It never returns the cost. It is public,
read-only and rate-limited like the other public routes (the cyber layer).

### 5. The weekly check (backend)

`Scripts/ops/slip_builder_study.py` turns the plan's bettor simulation into a check: 200 bettors, 415 weekends from
2012/13 to 2022/23, at typical prices. It reports:

- the builder against stacked Most likely picks at the same chance;
- said against landed;
- months with a win.

It must stay at least +2.0 back of every 100 at about 1 in 3, and said against landed within 2 points.

### 6. Screens (frontend, after 1–4)

Mockups on real matches first, with two or three versions side by side for the owner. Then "Build my bet" on the
slip: the bettor picks matches and a size, Spix fills the slip, and every leg can be swapped, removed or added.

### 7. Share and Monday (frontend, backend and cyber, later)

- **The share card:** the legs and "about 1 in N", with no stake, 18+ and a help line.
- **"How your slips did":** losses included, built on the wins reel's grading (D99).
- **Saved slips on an account:** cyber defines the account side.

### 8. Most likely with cost in mind (backend; changes picks; its own week)

When two markets' suggestions are within about 2 points of chance, the cheaper one is listed first. It runs on the
owner dashboard beside today's order first, showing how many picks would change and what they would have cost. It
goes live only on the owner's yes, in a week with no other pick change, and is judged at the Monday scorecard.

## How we will know

| Measure | Target |
|---|---|
| Builder slips, said against landed | Within 2 points |
| Builder against stacked Most likely, same chance | At least +2.0 back of every 100 |
| Same-match grid against what happened | Within 2 points on every common pair |
| From launch: free to paid each month | 3% or more |
| From launch: first renewal | 55% or more |
| From launch: monthly leaving after that | 10% or fewer |
| From launch: sign-ups from a shared slip or a friend | 5% or more |

## Left to the builder (the plan settles these)

- The exact search and how it caches.
- The grid's correction form: one ρ or per league.
- The route's request shape and limits.
- How the study's numbers sit on the owner dashboard.

## Order

1. Builder engine: items 1–5, behind the switch, no pick change.
2. Screens: item 6.
3. Share and Monday: item 7.
4. Most likely with cost: item 8, not before Tue 8 Dec.
5. Launch readiness: the server move (D115), sign-in and payments (cyber), and a test with about 10 friends over two
   weekends.
