# Recommendation layer: browse a match, see what Spix would take

**Status:** approved by the owner, 3 Nov 2026 (tracker D52, D53). Answers to section 8 are at the end.
**Reads with:** `docs/identity.md` (sections 2, 5, 6 and 8 decide this design), tracker D50 and D51.

## 1. Why

Stage 4 went live today and is doing its job: a pick is published only where a price beats the market by 2% and
the price is confirmed. On 3 Nov that left 5 of 18 matches with a pick. That suits a unit bettor. It does not suit
the person in `identity.md` section 2, who is watching one match on Sunday and wants to put 20 on it. For them, a
match with "no bet" is a dead end, and today most matches are dead ends.

The match page is also built around one released selection. Everything else is read-only ("Other model angles,
not released selections"), and the slip only takes a leg Spix released, with a fresh quote. The bettor can look
but cannot choose. That is the opposite of section 6, "Spix recommends. The bettor decides."

This design turns each match into something to browse. Every match shows what Spix would take in each market and
why, plus a short list of the most likely bets. The bettor can step through the other options, add any of them
and build their own bet. The Value picks stay exactly as stage 4 makes them.

## 2. What the bettor sees

### On the board (the list of matches)

Each match card shows Spix's top Most likely suggestion with its chance and typical price, and a count such as
"3 suggestions". A match is never written off while it has priced markets. The card says "No value picks today"
only on the match page, inside the Value list, where it is true and useful.

### On the match page, top: "Spix's picks for this match"

Two short lists, as D50 set them:

- **Most likely**, up to 3: bets Spix thinks are likely to land, from different markets, at a price that still
  pays. Each one has the bet in plain words ("Over 1.5 goals"), its chance ("about 3 in 4", with 76% beside it),
  the typical price, one sentence of why, and Add.
- **Value**, up to 2: the stage 4 picks, unchanged. When there are none, one plain line says so: "No value picks
  on this match today. The prices are about right."

### Below it: "Explore the match"

This is the interactive part. A row of market chips: **Result, Goals, Both teams to score, Handicap, Corners.**
Each market opens in place and shows:

- **Spix's suggestion for that market**, marked. This is the option Spix would take there, chosen by the same rule
  as the Most likely list (section 3) but limited to that market. A market where nothing qualifies says "Nothing
  stands out here" and still shows its options.
- **The options to step through.**
  - Result: three tiles (home, draw, away), each with its chance and typical price.
  - Both teams to score: yes or no.
  - Goals, Corners and Handicap: a line stepper. The bettor taps − / + (or swipes on a phone) through the lines
    the bookmakers price, such as 1.5, 2.5 and 3.5. Over and under, or each side of the handicap, update as
    they go: the chance gets lower as the price gets higher. This is the moment the bettor sees the trade-off for
    themselves, with no maths.
- **Add on every option** with a typical price, not only on Spix's suggestions. The bettor can take Spix's pick,
  pick a riskier line or go against Spix. Nothing tells them they are wrong. A long shot (chance under 25%) says
  so once, in the option: "a long shot, about 1 in 5".

Closed by default except the first market, so the page stays roomy (the owner's taste: essentials first, detail
when asked). The research sections already on the page stay below.

### The slip

Any added option goes to the slip at its typical price. The slip total is shown as "about" (`identity.md` section
8). Under the stake, one plain line: "Check the price in your bookmaker's app before you place it."

Same-match legs: goal-market legs from one match can be combined (owner's answer 1). Their combined chance comes
from the match's score grid, not from multiplying. A corners leg with a goals leg from the same match is refused
with a plain line until D51's factor is measured: "Corners and goals from one match can't be combined yet."

## 3. How Spix chooses (working defaults, to be tuned after real matches)

All numbers come from what each Match Read already stores. No new engine output is needed (section 5).

- **The options:** every line of Result, Goals, Both teams to score, Handicap and Corners that at least 2
  bookmakers price on every side. This is the same rule as the fair-price card (`fair_prices.MIN_BOOKS`), except
  that it uses every line instead of only the main one. Goals, Corners and Handicap use half lines only (1.5,
  2.5, −0.5, +1.5), so no bet can end with the stake returned or split. Quarter and whole lines need betting maths
  to explain. *(My call.)*
- **The chance:** the market's fair chance for that outcome, the median across bookmakers of each one's price
  with the margin taken out (power method), exactly as `fair_prices` computes it. Under stage 4 this is also what
  Spix publishes for goal markets and corners, because Spix's own numbers did not beat the market (power
  scorecard, 2 Nov: Spix said 76%, picks won 63%, the market said 67%). Honest chances first.
- **The typical price:** the median of the bookmakers' prices for that outcome, to two decimals, with no
  bookmaker named (D50).
- **Most likely:**
  - chance of at least 50%, and a typical price of at least 1.30, so the list never fills with
    "Over 0.5 goals at 1.03";
  - at most one per market;
  - the three must be able to win together. Check every pair of goal-market legs against the scorelines 0–0 to
    10–10: at least one scoreline must win both. Corners have no conflict with goal markets;
  - ranked by chance, ties going to the higher price.
- **Spix's suggestion in one market:** the same rule within that market, so the Most likely list is the top
  three markets' suggestions.
- **Value:** the read's published selections, as stage 4 releases them after price confirmation.
- **The why sentence:** a fixed template filled from the read's own figures (projected goals, each team's goals,
  projected corners, the result balance). For example: "Spix expects about 3.3 goals; Villarreal and PSG both
  score in most of the games that leads to." No generated text, and no "AI" (public copy rule: statistical models).
- **Cards and shots on target are left out** of this version. Spix's own cards chances are off (the
  consensus said 45%, won 32%), and their settlement still counts a straight red as one card. They come back when
  stage 6 prices them against the market and the settlement fix is in.

## 4. Honesty and care

- **Measured like everything else.** Every Most likely option shown is logged with its stated chance and typical
  price, then settled. The options measured are the ones in the last read before kickoff, so each match counts
  once *(my call)*. The scorecard gets its own section: "Most likely: said x%, happened y%", by market. This is
  its own cohort and does not touch the Value cohort or the stage measurement.
- **No tipster words.** No "lock", "banker", "sure thing" or "hot". No streaks, no "you missed this", no
  push to add more legs or stake more (`identity.md` section 7).
- **The chance is labelled plainly.** It is shown as "chance" on screen. The methodology page says it is the
  bookmakers' own view without their margin, and that Spix's work is choosing, explaining and spotting value.
  The page never presents a market favourite as Spix's own discovery.
- **Prices age.** Each list says "Prices as of 14:05", from the read's time. Every Add is at the typical price as
  a guide, never as a quote.

## 5. Where it lives in the code

- **No engine change and no engine switch.** The lists are compiled from stored reads at card-build time, next to
  `fair_prices.fair_card`, in a new `Scripts/data_platform/recommendations.py` that reuses `fair_prices` (every
  line, not only the main one) and adds the typical price. The card from `match_read_cards.build_match_read_card`
  gains a `recommendations` block (schema `recommendations.v1`: `most_likely`, `value`, `markets` with each
  market's options and suggestion, `prices_as_of`).
- **Because the published Value picks do not change,** this can ship before the Mon 9 Nov scorecard without
  breaking the one-change-a-week rule (D26, D29). Stage 4's measurement is untouched.
- **A display switch,** `SPIX_RECOMMENDATIONS` (off by default), so the website can be compared with it on and off
  and turned back in one step, as D46's `?card=classic` did.
- **Front end:** `Scripts/web_app/static/app.js` (`_renderMatchReadSummary`, the board card, `slip`) and the
  stylesheet. The read-only "Other model angles" block goes.
- **Logging and settlement:** the shown Most likely options go into the predictions store with a
  `list = "most_likely"` tag. Settlement reuses the existing market resolvers.

## 6. Not in this version

Same-match pricing beyond goal markets, player bets, cards and shots on target, the free and
paid split (built with the two tiers before launch; the lists count as recommendations), Discord (A7), a bookmaker
chooser, and the Betfair fallback for Value picks.

## 7. Left to the builder

The exact thresholds can be tuned within section 3's spirit, as can the wording of the why templates (the owner
signs off the copy), the chance-to-words table ("about 3 in 4") and how the stepper animates. Tests pin the
selection rules and the "can win together" check with fixed reads.

The look comes from the owner. The fixture card is being reworked in Claude Design (D47), so this page should be
mocked to match it and shown with real data before it ships (feedback: a mockup the owner liked failed in the
browser).

## 8. Questions for the owner

1. **Legs from the same match.** The Sunday bettor will often want two or three legs from the one match they are
   watching. Stage 4 already fits a score grid to each match's prices, which is what D51 proposed for combining
   goal-market legs. Should this project include the bet builder for goal-market legs from the same match, or
   should it stay one leg per match until the bet builder gets its own design?
   *Recommendation: include it for goal markets only. Corners with goals wait for D51's measured factor.*
2. **The Value price.** A value pick is value because a few bookmakers price it above fair, so the typical price
   is usually not value. Show "Worth it at 2.10 or better" (the fair-price card's minimum odds, no bookmaker
   named) instead of the typical price?
   *Recommendation: yes. The bettor checks their own app against one number.*
3. **The look.** Do you want to design this page in Claude Design, as with the card, or should I mock two or three
   versions side by side with real matches for you to choose from first?

### Answers (3 Nov 2026, owner)

1. Yes: include same-match legs for goal markets only.
2. Yes: Value picks show "Worth it at X or better".
3. Mockups first: 2–3 versions side by side with real matches; the owner picks one.
