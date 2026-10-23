# A5.3 confidence labels: design (23 Oct 2026)

Rework step A5.3 (tracker D30, D31). Agreed with the owner on 23 Oct (tracker D44). Calls marked
**[Claude's call]** were not discussed and are open to the owner's correction.

## Intent

Every published pick shows a member two things:

1. **A percentage:** the chance this bet wins, in this match. It is the published fair probability: the market's
   consensus wherever the market anchor is on (stages 3–4), otherwise Spix's own. This build does not change it.
   It stays honest only if bets shown at 70% win about 70% of the time over many matches. That is the check behind
   the number; members never see it.
2. **A label, high or low:** how sure we are that this is a good bet. Every labelled pick is still a pick, and
   the label never changes which bets are published.

The owner's definition (23 Oct):
- **High:** the market and Spix agree this is a good bet.
- **Low:** it is still a good bet against the market's price, just less sure.

Spix's opinion has to earn its place, market by market (the rework's rule, as in D21). Until it has in a market,
that market's label measures **how firm the market's own price is**:
- **High:** the bookmakers agree closely and the price is fresh.
- **Low:** the bookmakers are scattered, or the price is old or unknown.

Once a market passes the Spix check below, Spix's agreement joins its label: high needs a firm market **and** Spix
agreeing. The owner expects the rework (A6's fitted models) to make Spix's opinion count in more markets over time.

**Why replace today's label.** `confidence_from_edge` (`core/line_selection.py`) gives high when Spix's probability
is at least 65% and its edge at least 8%, and medium at 55% and 3%. Both inputs are Spix's own numbers, and A5.1 found
Spix's disagreement with the market is almost all error (D30), so today's "high" mostly marks a bigger Spix mistake.

## Goals

- A label that is honest today, using only evidence that does not depend on Spix's unearned opinion.
- A per-market check that lets Spix's agreement into the label once it has earned it, and keeps it out until then.
- The label's evidence recorded on every result, so research can test it and the card (D36) can say why.
- Built behind a switch that is off, previewed on the golden fixtures, live only as its own rollout stage.

## Non-goals

- Which bets are picked: that is selection (stage 4's `market_value` rule, A5.4 at stage 6).
- The percentage shown: that is the market anchor (stages 3–4).
- The card's design and wording: that is the fair-price card build (D36, D37).
- Stake sizing.
- Spix's sample size and uncertainty width (D30's other two inputs) are left out of the label for now: they describe
  Spix's own model, whose vote does not count yet. Revisit them when a market earns Spix's vote. **[Claude's call]**

## Anti-goals (a failure even if it works) **[Claude's call, from the rework's "Spix earns its place" rule]**

- A "high" that does no better than "low" on the check: a label that is decoration.
- Spix's unearned opinion entering the label by another route.
- A label that turns a pick into a no-bet, or a no-bet into a pick.

## The label

Three pieces of evidence per recommended result:

1. **Bookmaker agreement (M3)** on the picked line at the read:
   - Each bookmaker with a two-sided price on that line has its margin removed with selection's method (power once
     stage 2 is live).
   - The spread is how far apart their fair probabilities for the picked side are.
   - "Agree closely" means the spread is under a cut-off, with at least a minimum number of bookmakers. Fewer
     bookmakers than that counts as not agreeing. **[Claude's call]**
   - The prices are already stored in every read's decision audit (`input_options`).
2. **Price age:** how long before publication the quoting bookmaker's price was last seen.
   - It is known only when price confirmation runs (D42, from stage 4): the record's `bookmaker_seen_at`, against the
     time of the check.
   - "Fresh" means under a cut-off. An unknown age counts as not fresh. **[Claude's call]** This includes legacy
     surfaces and reads before stage 4, where no confirmation runs.
3. **Spix's agreement**, counted only in markets that passed the Spix check:
   - Spix's own probability for the picked side (`_spix_prob` under the anchor, the model probability otherwise)
     also gives at least the selection's minimum value (2%, `MARKET_VALUE_MIN_EV`) at the published price.

Rule:
- **High** when the market is firm (bookmakers agree and the price is fresh) and, in markets where Spix's vote
  counts, Spix agrees. The market must also give value at the price: with the stage 4 selection that always holds.
  Under today's selector, a pick only Spix likes is labelled low.
- **Low** otherwise.

Two levels, as the owner described. "Medium" is no longer produced under the switch, and the surfaces show
whatever label the decision carries.

## The checks

Both use recorded data only: no new requests, no billable OddsPapi calls. They run as research studies that read
`platform.db` behind the refresh gate in short statements (`calibration_data.latest_read_rows`), and their output
goes to `Research/`.

**Measures, per label:**
- Value at the close: the closing consensus probability times the price taken, minus one. It does not depend on
  results.
- Stated against happened: the shown percentage against the win rate (weighted for half results), with intervals
  that resample whole fixtures.

**Firmness check (sets the cut-offs):**
- On the audited market-value picks (the A5.2 replay rows, pre-match and lineup reads), compare high against low.
- High must keep at least as much value at the close as low, with the difference's interval above zero, and must be
  no worse on stated against happened.
- The cut-offs are chosen on the earlier weeks and confirmed on the latest week, so the check does not fit noise.
  Each label must hold at least about a fifth of the picks. **[Claude's call]**
- The price-age part can only be checked on picks with OddsPapi history at the read: the price-check replay
  (`ops/price_check.py`, D39, D41), mainly Betano, Pinnacle, 1xBet and Bet365.
- **If no cut-off passes, nothing ships.** The result goes to the owner instead, with what the data showed.

**Spix check, per market:**
- Among the market-value picks, picks where Spix agrees must keep more value at the close than those where it does
  not (interval above zero), and be no worse on stated against happened.
- It needs a minimum sample per market. **[Claude's call]**
- It is rerun weekly with the A5.1 and A5.2 studies, and again at the A6 gate.
- A market enters the "Spix's vote counts" list only by an owner-approved tracker decision. The list starts
  empty. Corners (c = +0.49 on 23 Oct) is the first candidate, but it must pass this check first.

## Constraints

- **Switch:** `SPIX_CONFIDENCE_LABELS`, `edge` (default, today's label) or `evidence`. It is registered in
  `core/engine_switches.py`, so every read records it and the scorecard splits picks by it.
- **Where the label is set:**
  - The engine records the evidence it has (bookmaker spread, Spix's value) in the result's context.
  - The final label is set after price confirmation in the Match Read dispatcher, where the price age is known.
  - Surfaces that call the engine directly get the engine's label, with age unknown.
- **One change at a time (D26):**
  - The labels go live as their own stage, after stage 4 (price confirmation must be on for the age).
  - The tracker's current plan puts the rest of A5 at stage 6, earliest Tue 17 Nov. The owner sets the date.
- **No behaviour change with the switch off:** golden `check` unchanged, and the preview reports how labels move.
- **Repo rules:** the tracker is updated in the same commit; `Index/`, `Output/`, `Research/` are never committed;
  `platform.db` is read only inside `data_access()`.

## Left to the builder

- The spread measure: for example the range or the interquartile range of the bookmakers' fair probabilities. Also
  the minimum number of bookmakers.
- The exact cut-offs and the minimum samples, from the checks above.
- One spread cut-off for all markets or one per market (spreads differ between, say, 1X2 and corner lines), as the
  data supports.
- Whether a repriced pick (D42: the bookmaker's price was shorter but still had value) is treated differently. It is
  recorded either way; use it in the rule only if the check shows it matters.
- File layout, names and tests, following the surrounding code.
