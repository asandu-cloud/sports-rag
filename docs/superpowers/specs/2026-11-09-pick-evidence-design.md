# Pick evidence on the match card: design (tracker D82)

The owner approved the second version of the pick-notes mockup (artifact
https://claude.ai/artifact/C732K7mQTdXUxUFQ44wUez, "Easier to read") on 9 Nov: "What should go live is everything
that is in the new match cards within the artifact. That includes How Spix reads it. I like the design as well as the
structure of the information." This replaces the for/against notes of D75, whose wording the owner rejected on 5 Nov.

## What the bettor sees

On the match page, for a domestic-league match with Most likely picks:

1. **How Spix reads it.** One short lead sentence, then a small table: the chance of each result (home, draw,
   away), goals expected for each team, and corners expected.
2. **Each Most likely pick** keeps its label, chance, price and Add. Under the chance:
   - a bold **verdict**, set by a rule on the counts below, never by a writer;
   - one **detail** line of at most 25 words that adds what the rows do not show (an average, a won-drawn-lost
     record, or the safer line and its price);
   - **Past matches at similar odds**: three rows (the home team's matches, the away team's matches, the
     league's matches), each with how many of its matches the pick would have won, as "9 of 12" and a bar.
3. **A tap on a row opens a sheet** listing those matches: date and season, the match, the price, the goals,
   corners or score, and whether the pick would have won.

Value picks, Explore the match and the board card are unchanged.

## Why it does not change any pick

Everything here is read after the picks are made, from the card's own `recommendations.v1` block and from finished
matches. The engine, the Match Read worker, the recommendation lists and the stored records are untouched. A test
builds the same card with the switch on and off and checks every existing field is identical. The pinned golden
check must stay unchanged.

## The evidence groups (chosen by price alone)

Tonight's prices are the card's moneyline typical prices (the median of the bookmakers' best prices). Each group
is chosen only by how the bookmakers priced past matches, never by whether they agree with the pick:

- **Home team group:** the home team's 12 league matches (any of the ten domestic leagues) before tonight whose
  own win odds are closest to tonight's home win odds, by distance in log price; ties go to the newer match.
- **Away team group:** the same for the away team and tonight's away win odds.
- **League group:** the 40 matches in tonight's league whose home and away odds are closest to tonight's (distance
  in log price on both), before tonight.

A team group needs at least 8 matches and the league group at least 20; a group short of that is left out. A pick
needs at least two groups, or it shows no evidence.

**History:** football-data's two finished seasons (2024/25 and 2025/26, `Index/football_data.sqlite3`, the
"Market average" pre-closing 1X2 prices) and this season's finished matches from the platform database (goals,
corners from the team statistics, and the 1X2 typical price from each match's final pre-kickoff odds snapshot,
stored since 9 Oct). Matches without prices or corners are left out. A match counts once.

**What "won" means, per market,** from the picked side's view (margin = picked team's goals minus the other's):

| Pick | Won when |
|---|---|
| Over/Under L goals or corners | the match total is over/under L |
| Both teams to score yes/no | both scored / at least one did not |
| Team to win; Draw | the team won; the match was drawn |
| Team ±L handicap | margin + L > 0 |

In the other team's group the same event is read from that team's side (e.g. "Forest did not win by more than
2"); in the league group from "the home side" or "the away side".

## The verdict rule

A group backs the pick when it won in at least 60% of its matches, and leans against it under 50%.

| Counts | Verdict (three groups) | Verdict (two groups) |
|---|---|---|
| all back | All three groups agree. | Both groups agree. |
| all against | All three groups lean against it. | Both groups lean against it. |
| two against | Two of the three groups lean against it. | — |
| two back | Two of the three groups agree. | — |
| one back, one against | The groups are split. | The groups are split. |
| one back | Only one group clearly agrees. | Only one group clearly agrees. |
| otherwise | The groups don't settle it. | The groups don't settle it. |

## The words

The mockup's lead and detail lines were written by Claude from the facts, under the same checks GPT would face.
The OpenAI account has had no credit since 5 Nov, so production starts with **fixed wording built from the same
facts** (this design), in the same shape: a lead of at most 14 words and a detail line of at most 25.

- **Lead:** who the favourite is ("X are clear favourites" at 60% or more; "X are favourites" when 12 points or
  more ahead; "X are slight favourites" when 4 points or more ahead; otherwise "Too close to call") and the goals
  ("with plenty of goals expected" at 3.4 or more in all; "with few goals expected" at 2.3 or less; "with most goals
  expected from X" when one side expects at least 1.5 times the other's; otherwise "with goals expected at both
  ends"). The figures in the table are rounded to one decimal.
- **Detail, first sentence:** goals or corners: "Similar matches averaged A to B goals." (the three groups'
  averages; one figure when they agree); a result or handicap: the picked team's group as "X won W, drew D and lost
  L of their N matches at similar odds." (a zero reads "none"); both teams to score: "X scored in A of their N, Y in B of theirs."
- **Detail, second sentence:** when the same market has a likelier option on the same side priced under 1.30,
  "Under 12.5 is safer but pays only 1.17." That says why Most likely lists this line.

GPT writing these two lines (with the fixed wording as the fallback) is a later step, once the OpenAI account has
credit and the owner wants it.

## Where it runs

In the website process, when the card is built, behind the new display switch **`SPIX_PICK_EVIDENCE`** (off by
default and pinned off in the tests). The history is loaded once and kept for an hour; this season's prices are
read incrementally (all 406 stored snapshots parse in about 0.6 s). Any failure leaves the card as it was, with a
logged warning that names only the error type. No new job, table, migration or paid call.

## The data the front end gets (`pick-evidence.v1`)

On each card's `recommendations` block, while the switch is on:

- `story` (any league, when the figures exist): `{"lead", "result_chances": {"home","draw","away"},
  "goals_expected": {"home","away"}, "corners_expected"}` (corners null when the read has no corners projection).
- `evidence` (domestic leagues, when tonight's 1X2 is priced and the teams are known):
  `{"version": "pick-evidence.v1", "since": "2024-08-01", "competition", "tonight": {"home","draw","away"},
  "groups": [...]}`. Each group: `{"key": "home"|"away"|"league", "team" (null for the league), "price_from",
  "price_to", "matches": [...]}`; each match `{"date", "season", "competition", "home", "away", "home_price",
  "draw_price", "away_price", "home_goals", "away_goals", "corners"}` and for a team group also `"venue"` ("H"/"A").
- Each Most likely item with evidence gains `evidence: {"verdict", "detail", "groups": [{"key", "won", "of",
  "means", "results": [true, false, ...]}]}`, `results` in the order of that group's `matches`.

The shared match lists sit once per card (about 15 KB), not once per pick.

## Going live

1. Backend: build and test behind the switch; the orchestrator merges.
2. Frontend: builds the display from the mockup (the artifact above), reading `pick-evidence.v1`. A card without
   `story` or `evidence` renders as today.
3. The owner adds `SPIX_PICK_EVIDENCE=on` to `.env`; the website is restarted (asked first).

`SPIX_PICK_NOTES` (D75) stays off; its code can be removed in a later clean-up.
