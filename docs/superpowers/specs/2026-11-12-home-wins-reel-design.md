# Home page wins reel: the data (12 Nov 2026; tracker D99)

The owner wants a slow, endless reel of last week's best real wins on the website's home page: cards like the match
cards, drifting right to left with small offsets and rotations. The frontend designs the cards and the motion (taste
skill), and the owner approves them before any code goes to production. This document is the backend's part: which
wins, from which record, in which format.

## The owner's choices (12 Nov)

- **Which wins:** the highest typical prices that came in, from both lists (Most likely and Value), one per match,
  about 12 cards.
- **Honesty:** one plain "said vs happened" line beside the reel, because a reel of wins alone is selective and
  identity.md section 7 says every pick is measured, losses included. For example: "Every pick counts, losses too.
  From 4 to 8 Nov Spix gave its Most likely picks about 7 in 10, and 7 in 10 came in."

## Which picks count

- **Only what the site showed at kickoff.** This is the board's own before-kickoff record (D70): the last read made
  before kickoff that passed the release check a second before kickoff. A pick published earlier and withdrawn
  before kickoff is not a win Spix can show. Example: Valencia v Real Madrid had a Value pick on 6 Nov
  (Real Madrid −1.75), and the later reads said no bet, so at kickoff the site showed none.
- **Only from 4 Nov, 19:50 UTC,** when the lists went on the site (D68).
- **Won** means settled at full time on 90 minutes. A push, a half result, extra time or a missing score is not a
  win, as in the owner dashboard's grading.
- **Best** means the highest typical price the match card showed at kickoff:
  - for a Most likely pick, its own typical price;
  - for a Value pick, the same option's typical price in the card's markets.

  Keep one pick per match, break ties by kickoff, and show at most 12.
- **The week** is the latest full Monday-to-Sunday week (London time) with a graded pick. In a break with no
  matches, the reel keeps the last week that had them. "Full" (decided in the build): otherwise a Monday-night
  match would start a week of one or two wins, and the section would hide until the weekend. `week.from` and
  `week.to` are the first and last days with a graded pick.
- **Half results** (a quarter or whole line on a Value pick): a half won counts with the wins and a half lost with
  the losses; a push is void, as the owner dashboard shows them. Only a full win with a typical price goes on the
  reel.
- **The record line** covers both lists, as shown at kickoff that week: picks, won, lost, void, the average chance,
  the expected wins (the chances added up) and the share that came in. The frontend asked for Value in it (12 Nov):
  a 4.10 Value winner next to a Most likely-only record would read as selective. For small samples the expected
  wins read more honestly than a percentage: "10 Value picks; Spix expected about 4 or 5 to win; 3 did, 2 were
  void."

## The data contract (`home_wins.v1`)

```json
{"version": "home_wins.v1", "week": {"from": "2026-11-04", "to": "2026-11-08"},
 "record": [{"list": "most_likely", "picks": 353, "won": 248, "lost": 105, "void": 0, "said": 0.6897,
             "expected_wins": 243.5, "happened": 0.7025},
            {"list": "value", "picks": 10, "won": 3, "lost": 5, "void": 2, "said": 0.4518,
             "expected_wins": 4.5, "happened": 0.3}],
 "wins": [{"fixture": {"event_id": "...", "league": "Eredivisie", "home_team": "Cambuur",
                       "away_team": "PSV Eindhoven", "kickoff": "2026-11-08T15:45:00Z"},
           "visuals": {"home_team_logo": "...", "away_team_logo": "..."},
           "score": {"home": 2, "away": 5}, "list": "value",
           "pick": {"label": "Over 5.5 goals", "market": "goals", "side": "over", "line": 5.5,
                    "chance": 0.2265, "typical_price": 4.1},
           "result": "7 goals"}]}
```

- `result` is filled for goal and corner totals ("7 goals", "13 corners") and is null otherwise; the score says
  the rest.
- An empty `wins` list means the section hides.
- `said` is the average chance over every pick, `expected_wins` the chances added up, and `happened` won divided
  by picks. A void pick (a push) is neither won nor lost.
- In the site's chance words, 69% reads "about 2 in 3" (there is no "7 in 10" step).
- The frontend may still ask for other fields while it designs.

## How it is built (12 Nov, after the owner agreed the design)

- `Scripts/data_platform/services/home_wins.py` builds the block. It uses the board's before-kickoff path
  (`_shown_at_kickoff` in the match reads router) for the week's days and public leagues, plus the fixtures' final
  scores and corners. It reads `platform.db` only, never the owner dashboard's file, so the site does not depend on
  a local tool.
- Rebuilding the week's reads is too slow for a page request (about 120 matches, 32 s). So
  `python -m Scripts.data_platform home-wins` runs once a day as the best-effort refresh stage `home_wins_file`
  (while `SPIX_RECOMMENDATIONS` is on), writing `Index/home_wins.json`. The public endpoint `GET /api/home/wins`
  serves that file, or an empty reel when there is none.
- No migration, no `.env` line, and no change to picks or to the engine switches.

## Real data used for the sketches (4–8 Nov)

- 118 matches had picks on the board at kickoff.
- Most likely: 353 picks, 248 won. The average chance given was 69.0%, and 70.3% came in.
- Value: 10 picks were still showing at kickoff, of 61 Value publications for those matches. Of the 10, 3 won,
  5 lost and 2 were void, against about 4.5 expected wins. The 3 wins were Over 5.5 goals at Cambuur v PSV
  (7 goals), Over 9.5 corners at Villarreal v Getafe (13), and Under 3.5 goals at Cagliari v Frosinone (1).
- The 12 cards, highest price first, start at 4.10 (that Value win, given 23%) and end at 1.56.
- The sample file is passed to the frontend.
