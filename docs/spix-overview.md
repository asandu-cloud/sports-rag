# Spix: what it does and what it will do

*Summary as of 29 October 2026. Sources: `docs/rework-tracker.md` (the source of truth for the engine plan), the
agreed offer page "What Spix Sells" (17 Oct), and `docs/brand/color-palette.md`. Where this file and the tracker
disagree, the tracker wins.*

## In one line

**Know the real odds. Never overpay.** Spix is a football betting research service. For each match it works out
what a bet is really worth, and shows bettors which bets are worth taking at their bookmaker and which to skip.

## What Spix is, and what it is not

- **It is** a calibrated second opinion: the honest chance of each outcome, the fair price, and the lowest price
  worth taking. Bettors decide for themselves.
- **It is not** a tipster selling a win rate, a "beat the bookies" system, or a promise of profit. Spix makes no
  hit-rate or profit claims, takes no affiliate money, and publishes its full record, losses included.
- **Who it's for:** casual and regular football bettors in Europe, mostly on their phones, who want to bet with
  more knowledge and stop overpaying.

## What Spix does today

### The engine
- Covers 13 competitions with bookmaker prices: the Premier League, La Liga, Serie A, Bundesliga, Ligue 1,
  Championship, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, and the Champions, Europa and Conference
  Leagues.
- Reads prices from many bookmakers through API-Football and removes each bookmaker's margin to find the fair
  price. Since 23 Oct this uses the "power" method, which tested best on years of closing prices.
- For the main markets (match result, over/under 2.5 goals), it takes the market's consensus as the honest chance.
  The research showed that where Spix's own model disagreed with the market, Spix was nearly always the one that
  was wrong. Spix's own model has to prove itself on the scorecard before it gets a say.
- Projects goals, corners, cards and shots on target for every match. Uncertainty for corners and shots is now
  modelled properly (live since 20 Oct).
- Publishes **Match Reads** (the analysis for each match) automatically through the day.

### Keeping itself honest
- Every published pick is settled after the match and compared with the **closing price** (the price just before
  kickoff), including a "true close" from a second odds source (OddsPapi).
- A **weekly scorecard** every Monday shows whether Spix's picks beat the closing price, by market and by engine
  change. It is the yardstick for every decision.
- Every engine change ships behind a switch, one stage at a time, with a preview against saved matches first.

### The website
- **Matchday board:** today's matches with Spix's read, fair prices and the pick.
- **Research area:** teams and players.
- **Best Bets:** the strongest picks of the day.
- **Methodology page:** how to read the analysis, plus responsible-gambling, privacy and terms pages.
- **New logo since 29 Oct:** a text-only "spix" with a football for the dot on the i, which drops onto the i when
  you change page.

### Discord
- The Spix's Picks bot, with match, market and parlay commands, scheduled posts and a premium AI chat. It predates
  the engine rework and moves onto the new engine in a later step (A7).

### Operations
- Scheduled jobs on the Mac handle the daily data refresh, odds archiving, true-close capture, Match Read
  publishing, measurement and the Monday scorecard.
- Failures and the day's rhythm go to the owner's Telegram.

## What Spix will do

### Next few weeks: finishing the engine rework
| When | Step |
|---|---|
| Mon 2 Nov | The scorecard judges stages 1–3; measurement switches to the power method. |
| From Tue 3 Nov | **Stage 4:** the market's view for all goal markets, and picks chosen by value (the best real price against the fair price). Every price is re-checked just before publishing, so Spix never shows a price that has already gone. |
| From Tue 10 Nov | **Stage 5:** calibration, so a "60%" really happens about 60% of the time. |
| Later | **A6:** Spix's own team-strength ratings, fitted in-house. Spix's own view gets more weight only where it beats the market on the scorecard. |
| Later | **A7–A8:** move the Discord bot and the older AI chat onto the new engine, and retire the old ML layer. |

### The member offer (agreed 17 Oct; details still open)
1. **The Daily Card:** green cards to take, red cards to skip. Each card shows the honest chance, the fair odds and
   the **minimum odds worth taking** (fair odds × 1.02), so members can check their own bookmaker's app. Plus a
   public record.
2. **The Match Read + Ask Spix:** the research for each match done for you, and the original "ask Spix" idea,
   built on honest numbers.
3. **Price Check & the Ledger:** check whether any single bet, bet builder or boost is fairly priced, and a personal
   tracker showing how much you overpaid.
- **Free tier (proposal):** Price Check on singles, one Match Read a day, and the public record.
- **The match card** is being redesigned by the owner in Claude Design; the build follows their design.

### More markets (after the selection work)
Spix publishes 7 of the 114 market types its feeds price. New markets arrive in waves, each published only once the
scorecard shows it holds up:
1. Team goal totals, double chance, draw no bet.
2. Team corners, corners handicap, corners 1X2.
3. First-half goals and handicap, half-time/full-time.

The edge is most likely in thinner markets (corners, cards, player props, bet builders), where bookmakers price
less sharply.

### Launch
- **EU-wide, in English first.** A Romanian version is built alongside and released later.
- Only bookmakers licensed in the reader's country are shown or named.
- Euro pricing with EU VAT (One-Stop-Shop) through Stripe.
- Before launch the owner:
  - writes to ONJN (Romania's gambling regulator) to confirm that the service needs no licence;
  - asks OddsPapi for permission to show its prices to paying members;
  - sets the price and tiers.
- A paid odds source (OddsPapi) only once there are paying members.
- **Year-one goal:** €5,000 a month in subscriptions (about 200 members at about €25), grown by word of mouth from a
  product people trust.

## Principles every decision is checked against
- **Genuine:** real chances and real prices, never inflated claims.
- **Value for the bettor:** spot the bets to take and the bets not to take.
- **Earn it on the scorecard:** a model gets a say only where it measurably beats the market.
- **Scalable:** works across countries and leagues without hand-tuning.
- **Responsible:** no promise of profit, no pressure to bet, and responsible-gambling help always one tap away.
