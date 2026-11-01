# Spix identity

**Status:** version 1, first draft from the owner's description, 1 Nov 2026. Section 10 lists what is still open.

This file says what Spix is, who it is for and why they pay. When there is doubt about what to build, say or cut,
start here. Only the owner changes it. Every change gets a line in section 11, so we can see how the identity moved.

## 1. In one line

**Spix is the one place football bettors go to build a good bet without the homework.**

It is a hub, a haven for the bettor: the research is done, everything is in one place, and the bettor still makes
the choices.

## 2. The picture

Sunday morning. A Spix subscriber wants to watch the football and put 20 on the game. They are hungover and plan
to spend the day on the couch. Having money on the game makes it more interesting.

They open Spix and look through the matches and what Spix recommends. They pick the legs they like, drop the ones
they don't, and build their own parlay. A few minutes later they place it at their usual bookmaker and go back to
the couch.

It felt easy, and it still felt like their bet. They did the research, they looked at the recommendations, and they
made the choices. Spix just made that much easier.

After a month they think: **how was I betting before this?** The 20 euros a month feels like great value.

That thought is what Spix is trying to earn. It is the test of the whole product.

## 3. Who it is for

**For:** casual football bettors, 18+, for whom a bet is part of enjoying the game. They bet for fun, on small
stakes, usually on matches they are going to watch. They want to make informed bets, and they do not want betting
to feel like a second job.

**Not for:**
- Professional or serious value bettors who track closing prices and move between bookmakers.
- People looking for a system to make money.
- Anyone for whom betting has stopped being fun. Spix does not try to win them, and it tells them where to get help.

## 4. Why they pay

1. **It saves the work.** Everything they would have looked up is already there, in one place.
2. **It gives them good options.** They never start from a blank page, because Spix recommends which bets to look
   at and says why in plain words.
3. **The bet is still theirs.** They choose, and the choosing is part of the fun.
4. **They can trust it.** The numbers are honest, including when a bet is a long shot.

## 5. What Spix must do well

In order of importance:

1. **Recommendations on the matches people care about.** Several sensible options for a match, not just one pick,
   and nothing that makes the bettor feel foolish for choosing.
2. **Building a bet.** A simple way to take legs from those recommendations, mix them with their own ideas and
   see what the whole bet looks like.
3. **The research, in one place.** Form, team news, head to head, goals, corners, cards and players, with the short
   version first and the detail on request.
4. **Honest numbers underneath everything.** See section 7.

## 6. Agency is the product

Spix recommends. The bettor decides. This is what separates Spix from a tipster.

- Spix shows options and reasons. It does not hand down a single answer to copy.
- The bettor can always change, swap or remove a leg, and can add one Spix did not suggest.
- A ready-made bet is a starting point to edit, never the finished thing.
- Spix does not lecture. If a choice is a long shot, it says so once, plainly, and lets them carry on.

## 7. Honesty and care

These rules do not change with the selling point:

- **No promises of profit.** No win-rate claims, no "beat the bookies", no guaranteed anything.
- **Honest chances.** If Spix says 60%, that has to happen about 60% of the time. A model only sets the published
  chance where it beats the market on the scorecard.
- **The record stays honest.** Every pick is measured, losses included. The scorecard remains how we judge the
  engine, whether or not bettors ever read it.
- **No affiliate money.** Spix is paid by its members, not by bookmakers.
- **Spix never pushes people to bet more.** No nudges to stake more, no "you missed this" messages, no chasing
  losses, no streaks or rewards for betting often. Fun is the point, and only for people for whom it stays fun.
- **Help is easy to find.** 18+ everywhere, with local help lines.
- **The Sunday picture is for us, not for marketing.** Being hungover or drinking never appears in anything public.

## 8. What stays out of the front of the product

Bettors do not care about these, so they do not lead:

- **Which bookmaker is 2% better.** They will use their own book anyway. Spix does not ask them to shop around.
- **Margins, fair prices and closing prices.** These stay under the hood. They make the numbers honest, and the
  methodology page can explain them for anyone curious.
- **Anything that needs betting maths to understand.** If it needs a glossary, it is not on the main screen.

"Know the real odds. Never overpay." was the promise from 17 Oct. It is retired as the main selling point.

## 9. How to use this file

**Before building or writing anything, ask:**

1. Does it make the Sunday morning easier, or at least not harder?
2. Does the bettor still feel the bet is theirs?
3. Would a casual bettor understand it without any betting maths?
4. Is it honest?
5. Would it ever push someone to bet more than they meant to?

The first four need a yes and the last needs a no. If a feature fails, it needs a very good reason or it waits.

**Which document wins:**

| Question | Source |
|---|---|
| What Spix is, who it is for, why they pay | This file |
| Engine plan, status and decisions | `docs/rework-tracker.md` |
| How it looks | `DESIGN.md`, `docs/brand/` |

`PRODUCT.md`, `docs/spix-overview.md`, `docs/spix-marketing-brief.md` and the "What Spix Sells" page follow this
file. If they disagree with it, they are out of date.

## 10. Open questions

1. **What counts as a recommendation?** Today's rules give too few. In the stage 4 preview, 54 matches produced 22
   picks and no Asian handicaps. Value alone cannot fill a Sunday, so Spix needs a rule for "sensible options on
   this match", with value as one thing it considers.
2. **Whose price do we show?** Bettors use their own book, and our feed does not carry every book in every country
   (D36, D38). What should a bet show when the price at their book is unknown?
3. **How honest is the parlay screen?** Parlays are long shots and each leg adds bookmaker margin. Proposal: show
   the combined chance once, in plain words such as "about 1 in 6", with no warnings beyond that.
4. **What is the price?** The picture says 20 euros a month. The earlier plan was about 25, with the year-one goal
   of €5,000 a month. At 20 euros that means 250 members, not 200. Tiers are also undecided.
5. **Where is home?** The website, Discord or both. The website slip only adds up odds today, while the Discord bot
   already builds parlays.
6. **What happens to Price Check and the Ledger?** They were two of the three member offers. Proposal: drop them
   from the launch offer and keep the idea for serious users later.
7. **What is the public line?** The line in section 1 is a working description, not a slogan.

## 11. Change log

| Date | Change |
|---|---|
| 1 Nov 2026 | First version, from the owner's Sunday-morning description. Replaces "Know the real odds. Never overpay." as the main selling point. |
