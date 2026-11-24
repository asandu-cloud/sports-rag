# Best bets on the match cards — design (D129)

Owner decision 24 Nov 2026, after the mockup on the real Champions League boards
(https://claude.ai/artifact/E8jykPRRawU697SsfPKU4Y) and a backtest on the cards shown at kickoff from 4 to 22 Nov.
Short design for a small change.

## Intent and why

Each market's suggestion is the likeliest option that pays at least 1.30 (D52). Ranking by chance alone drags every
pick down to the floor:
- on the board card, the pick averaged 1.34 and was under 1.45 on 97% of matches;
- corners led a third of all picks.

The owner wants the cards to show the best bets available for each match, which the bettor then takes as singles or
uses in a parlay: bets that win often and are worth taking.

The card and the match page keep their design (owner, 24 Nov: "we are not changing the design of the match card
itself. we are changing whats inside it"). Only the bets inside change.

## The rule

1. **The pool.** Every option the read prices (the layer's five markets, every half line) with a typical price
   from 1.30 to 1.99.
2. **Fair only.** Drop an option when the bookmaker keeps more than 6 of every 100 staked at the typical price
   (`1 − chance × typical price > 0.06`). The chance is the market's own, so this compares the bookmakers' cut
   between lines. It makes no claim to know better than the market.
3. **The three bets** ("Most likely" on the page), from three different markets, each able to win alongside the
   others and never the same bet under two names:
   1. the fairest under 1.60;
   2. the fairest from 1.60;
   3. the next fairest.
   "Fairest" means the bookmaker keeps least, with ties going to the higher chance. They are listed fairest first,
   so the fairest one goes on the board card.
4. **Each market's suggestion** (marked in "Explore the match") is that market's bet among the three when it has
   one. Otherwise it is the market's fairest option in the pool that is not the same bet as one of the three.
5. Value, "Worth it at", the score grid and the block's format are unchanged. Nothing is added to
   `recommendations.v1` and nothing is removed.

## Approaches

| Approach | Trade-off |
|---|---|
| Likeliest at 1.30 or more (today) | Honest chances, but every pick slides to the floor price, and corners dominate |
| **Fairest in each payout range, all able to win together, up to 1.99** (chosen) | Spread from about 1.30 to 1.90, any two bets go in a parlay, and the board card leads with the fairest bet |
| The same, plus "pays more" (2.00–2.59) | On 15 of 16 matches tested, the fair bet at 2.00 or more backed the other side of another pick. Bigger payouts are left to combining matches (the Bet Builder, D128) |

## Switch and timing

- **A new engine switch, `SPIX_BEST_BETS`:** `likeliest` (the default, today) or `fairest`.
- **The rule follows the read, not the website.** Each read records the engine switches in effect when it is made
  (A3.7). `block()` reads the value from the read's own record, and a read without the key uses `likeliest`. So
  cards already shown, the before-kickoff record (D70), the wins reel (D99) and the owner dashboard keep the bets
  they really showed. After the flip, only new reads use the new rule.
- **The flip changes published picks,** so it needs its own stage week, judged at a Monday scorecard (D26, D29).
  The owner decides when. Cards have their first real test on 27–29 Nov, judged Mon 30 Nov.
- **No frontend change.** The heading "Most likely" stays until the owner decides on its wording.

## Evidence so far (backtest, 4–22 Nov, 128 matches, cards as shown at kickoff)

| | Likeliest (today) | Fairest |
|---|---|---|
| Average price | 1.40 | 1.54 |
| Landed / said | 71.4% / 69.0% | 64.8% / 63.5% |
| Back per 100 staked (95%) | −0.2 (±6.4) | −1.3 (±7.5) |
| Board card bet: average price | 1.34 | 1.45 |
| Board card bet: back per 100 | −6.2 (±10.6) | −5.4 (±12.1) |
| Board cards led by corners | 52 of 128 | 1 of 128 |

- **Chances stay honest.**
- **The money is the same within noise.** The bookmaker keeps 3.5 of every 100 on average against 3.8, and 2.6
  against 3.4 on the board card.
- **The gain is better payouts and more variety,** not money saved against today's rule.

## Left to the builder

- Exact names of the constants and helpers.
- Whether the owner dashboard's version column should carry the rule; the scorecard can already split picks by the
  read's recorded switches.
