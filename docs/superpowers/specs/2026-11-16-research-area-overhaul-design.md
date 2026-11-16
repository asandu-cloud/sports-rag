# Research Area overhaul: next matches, the match view and clearer numbers (D107)

Owner approval: 16 Nov 2026, on the clickable sketch and the list of 35 improvements ("the sketch looks really good.
i dont think i have anything i want to change for now. you can get started on implementing the things in the
renders. for the phone, use option C (floating design) for the next matches").

The sketch is the reference for every screen in this document. Its clickable pages (`sketch/index.html`, with
`?m=bar` for the phone's floating bar, and `sketch/fixes.html`) and its screenshots (`*.jpg`) are in
`docs/superpowers/specs/assets/2026-11-16-research/`. Its numbers are dummy, shaped on real records.

## 1. Why

The owner does not use the Research Area for their own bets: "Right now its nowhere near ready for customer use."
Their two test bets show why:

- **Over 2.5 cards in Chelsea's next match.** Today this takes three separate searches (Chelsea, the opponent, the
  referee). No profile knows the next match, and the hit rate for a line hides behind "Explore a threshold".
- **A player's shots on target.** The player page answers a season question ("0.91 per 90") rather than the bet's
  question ("1 or more in 7 of his last 10").

The overhaul turns the research into the step before a bet: pick a team, see its next matches, open one, and read
every market for that match with the numbers that decide it.

## 2. Goals, non-goals, anti-goals

Goals:
1. A team page shows the team's next three matches, in every competition. Desktop: a sticky rail on the right.
   Phone: a floating bar (the owner's option C) that opens a sheet with the three.
2. A match view for any of those matches: Spix's figures for it, then one section per market (Cards, Shots on
   target, Goals, Corners, Result, Head to head, Team news), each opening on the line the bettor picks.
3. Every profile answers first: the bettor's line and how often it was reached ("7 of his last 10"), then the
   detail.
4. Numbers a bettor can read on a phone: written on the bars, one decimal at most, zeros drawn, no hover needed.

Non-goals (this build):
- No paywall. The free and paid split in the review stays a proposal until the payments work (D50). The match
  view's sections are built as separate blocks, so a later gate can lock some without a redesign.
- No new picks. Cards and shots on target stay out of Most likely and Value (D61). The match view shows Spix's
  figures and chances for them, as the sketch does, but never labels them a pick.
- No new predictions. Every Spix figure comes from the match's Match Read; the research adds past matches only.
- The Research Assistant tab, the board and the match page are unchanged, apart from the slip bridge (section 6).

Anti-goals (a failure even if it works):
- A page that pushes a bet: no "hot", no streak language, no tipster words, no profit or win-rate claims.
- Numbers from two seasons or two scopes side by side without saying so (Palmer's 23% against 12%).
- Hover-only numbers, or a dash where the true value is 0.
- Three disclaimers on one page. One quiet line per page says what the numbers are.

## 3. What the owner decided

- The sketch, as drawn: team page, next-matches rail, match view, the four smaller fixes. No changes requested.
- Phone: option C, the floating bar. The review recommended the strip; the owner chose the bar. The bar must not
  sit on top of the slip bar (section 5.2).

## 4. Screens

### 4.1 Team page (answer first)

Order, top to bottom (sketch: `desk-team.png`):
1. Head: crest, name, competition and season, the season picker.
2. Lead sentence from the last 10, about the market the bettor last chose (Cards by default):
   "Chelsea's matches are card-heavy: 9 of their last 10 had 4 or more cards." The count is the one marked figure.
   The wording is factual: "card-heavy" only when 7 or more of 10 cleared the line; otherwise "N of their last 10".
3. **Last 10 at a glance.** Chips for Cards, Goals and Corners, a line picker (cards 2.5 to 5.5, goals 1.5 to 3.5,
   corners 8.5 to 11.5), and one square per match: filled when the match total went over the line, with the total
   written in it, opponent and H/A underneath.
4. **Cards by match.** Stacked columns, the team's cards and the opponent's, the total written on top, and a dashed
   line at the chosen line. The line's label sits in the legend, not on the chart.
5. The existing sections, collapsed under their headings: Against the league (ranks), Home and away, Squad
   leaders, Chances by match. One tap opens each.

### 4.2 Next matches

Desktop (wider than 860 px, as the sketch draws it): a rail on the right of the team page, 340 px wide, 300 px between
861 and 1100 px, sticky under the top bar. One card per match (sketch: three):
- crest, "v Leeds" or "at Napoli", competition, day and time;
- the read's state: "Read ready", "Preliminary" or "Read due Thu";
- the referee once named ("Referee Anthony Taylor"), else "Referee not named yet";
- when the read is out, Spix's expected goals, corners and cards for the match, in three small figures;
- the whole card opens the match view.

At 860 px and below the rail is not shown. The phone and small tablet use the floating bar (option C):
- a bar fixed near the bottom: the next match's crest, "Next: v Leeds", "Sat 21 Nov · 15:00 · and 2 more", "Open ›";
- tapping it opens a bottom sheet titled "Chelsea's next matches" with the same three cards; tapping a card opens
  the match view and closes the sheet; the backdrop, Escape and a swipe down close it;
- the bar sits 8 px above the slip's mobile bar when that bar shows, and 8 px above the page bottom (plus the
  safe area) when it does not; the page gets bottom padding so the bar never covers the last row;
- the bar shows on team pages only. On a player page the next match is the card in the head (4.4); on a referee
  page it is the next appointment.

### 4.3 Match view

Route: `#research/matches/{event_id}/{market}`, with `{market}` one of `cards`, `sot`, `goals`, `corners`, `result`,
`h2h`, `news`. It opens at the market the bettor came from (Cards from a team page that was on Cards). The rail
stays on the right on desktop, with the open match marked.

Top to bottom (sketch: `desk-match.png`, `phone-match*.png`):
1. Back link to the team, the two crests, "Chelsea v Leeds", competition, day, time, referee.
2. **How Spix reads it**: result chances (58 / 24 / 18), expected goals, expected corners, expected cards,
   both teams to score. One figure marked: the one for the open market. Before the read is out, this block says
   when it comes out and the sections below still show the past matches.
3. **Markets row**, sticky under the top bar: Cards, Shots on target, Goals, Corners, Result, Head to head, Team
   news. On a phone it scrolls sideways.
4. The open market's section:
   - **Cards**: line picker 2.5 to 5.5; an Over and an Under card with Spix's chance and the typical price (and Add
     when the slip can take it); hit rows, one square per match, for the home team's last 10, the away team's last
     10, the referee's last 10 and the last meetings, each with "N of 10 over"; "Who gets the cards" (each team's
     own cards a match and the cards their opponents get); the referee (cards a match this season against the
     league average, matches refereed); the players booked most often on each side.
   - **Shots on target**: 1+ and 2+, both teams or one; a table of the likely starters with their last 10 as a
     strip of squares, "7 of 10", minutes a match and, when a price is known, its typical price; under it "What
     Leeds allow" and "What Chelsea allow": shots on target conceded a match and the rank.
   - **Goals** and **Corners**: line picker, Spix's chance and typical price for Over and Under, hit rows for both
     teams' last 10 and the last meetings.
   - **Result**: the three chances with typical prices, each team's last 10 results as squares, home and away form.
   - **Head to head**: the last meetings, with score, cards, corners and goals.
   - **Team news**: likely starters from recent starts; confirmed line-ups when the read has them; a plain line
     when neither is known.
5. One quiet line at the foot: "Past matches are records, not a forecast. Spix's figures come from statistical
   models. 18+ · GamCare".

### 4.4 Smaller fixes (sketch: `fix-*.png`)

- **Player page, answer first**: the next match as a card in the head ("Next match v Leeds · Sat 21 Nov, 15:00 ·
  Open ›"); stat tabs (Shots, On target, Goals, Assists, Booked, Fouls); Last 5 / Last 10 / Season; a line picker
  (1+, 2+, 3+); the answer in large type ("1 or more shots on target in 7 of his last 10"), the dates and starts
  under it; labelled columns with zeros drawn and the line dashed; three context tiles (minutes a match, what the
  next opponent allows, the typical price when known).
- **Referee page, this season first**: cards a match this season, marked, with the league average and last season
  in the sub-line; next appointment in the head; labelled columns against a line picker (2.5 to 5.5), "Over 3.5 in
  7 of 9".
- **Ranks from one season**: the rank card uses the same season as everything else on the page. Under the minutes
  threshold it still shows, with a quiet "Early season · 888 minutes so far" chip, instead of switching to last
  season. Last season's ranks are one link away.
- **Home and away**: the home and away values written at both ends of each row.

### 4.5 Clearer numbers everywhere

- One decimal at most (2.67 becomes 2.7); percentages whole.
- Counts as "N of M" next to any rate.
- Values written on or above bars; tooltips repeat them, never hold them alone.
- A zero is drawn as a thin base mark with "0" above it; a dash only when the data is missing.
- Old copy that names colours that no longer exist ("A taller green bar") is rewritten.
- Notices are quiet grey chips, not amber boxes.
- Colour: cobalt for the bettor's selection and hits only; the lime marker on one figure per screen.

## 5. Behaviour details

### 5.1 Lines and hits

A match "went over" a line when its total is greater than the line (lines are halves, so there are no pushes). For
player counts, "1+" means at least 1. Hit squares show the total. Missing statistics show an empty outlined square
with a dash and do not count in "N of M"; M drops accordingly ("6 of 9").

### 5.2 The floating bar and the slip

The slip's mobile bar is 56 px plus the safe area and shows below 1100 px. The research element cannot see it
through the shadow root, so the app tells it: `app.js` sets a CSS custom property `--slip-bar-h` on the research
host (56 px plus the safe area when the bar is visible, 0 when hidden). The floating bar sits at
`bottom: calc(var(--slip-bar-h, 0px) + 8px + env(safe-area-inset-bottom))`. When the slip sheet is open, the bar
hides.

### 5.3 Adding to the slip from the match view

The research element raises a `research-add-leg` event with the same leg shape the match page's Add uses. `app.js`
passes it to the slip unchanged, so the slip's rules (one leg per market per match, same-match goals combined) are
the slip's own. Add shows only for options the read offers in `recommendations.v1`. Cards lines show their chance
and price without Add, since cards are not a slip market today.

### 5.4 When data is missing

- No read yet: "Spix's read comes out Thu 19 Nov" in place of the figures; the history sections still show.
- No referee named: "Referee not named yet. Appointments usually come two to three days before."; the referee row
  is left out of the hit rows.
- Fewer than 10 matches: "last N", never padded.
- No price: the price column and tile are left out, not shown as a dash.

## 6. Data

### 6.1 What exists

- `GET /api/research/teams/{id}`: a team's matches in one competition and season, each with cards, opponent cards,
  corners for and against, goals, shots on target for and against. Enough for the team page (4.1).
- `GET /api/research/players/{id}`: a player's matches with minutes, shots, shots on target, cards.
- `GET /api/research/referees/{key}`: a referee's matches in all competitions with cards and fouls.
- Match Read cards (`match-read-card.v2`) with `recommendations.v1`.

### 6.2 What the backend adds (asked on 16 Nov)

Read-only, public like the rest of `/api/research`:

**B1. Next matches.** `GET /api/research/teams/{team_id}/upcoming?limit=3` and
`GET /api/research/referees/{referee_key}/upcoming?limit=1`, schema `research-upcoming.v1`:

```json
{"schema_version": "research-upcoming.v1",
 "subject": {"type": "team", "id": 12, "name": "Chelsea"},
 "matches": [{
   "event_id": "1557601", "fixture_id": 32311, "kickoff": "2026-11-21T15:00:00+00:00",
   "competition": {"code": "EPL", "name": "Premier League"},
   "home": {"id": 12, "name": "Chelsea", "short_code": "CHE", "logo_url": "..."},
   "away": {"id": 31, "name": "Leeds", "short_code": "LEE", "logo_url": "..."},
   "venue": "home",
   "referee": {"name": "Anthony Taylor", "referee_key": "anthony-taylor"},
   "read": {"status": "ready", "read_at": "...", "opens_at": null,
            "expected": {"goals": 2.7, "corners": 10.1, "cards": 4.6}}}]}
```

`read.status` is `ready`, `preliminary` or `waiting` (no read yet; `opens_at` says when it can come out).
`referee` is null until named. `expected.cards` is null when the read has no cards figure.

**B2. The match.** `GET /api/research/matches/{event_id}`, schema `research-match.v1`:

- `fixture`: as one B1 match, plus `status`.
- `read`: null, or `status`, `read_at`, `prices_as_of`, `result` (home, draw, away chances), `goals` (home, away),
  `btts_yes`, `corners_expected`, `cards_expected`, and `lines` per market (`cards`, `goals`, `corners`): a list of
  `{line, over: {chance, typical_price}, under: {chance, typical_price}}` from the read, plus each market's
  `suggestion` and the `recommendations.v1` options (for Add). Nothing is computed afresh.
- `teams.home` and `teams.away`, each:
  - `last_matches`: the last 10 finished matches in all competitions, newest last: `event_id`, `date`,
    `competition`, `venue`, `opponent`, `score`, and `cards_for`, `cards_against`, `corners_for`,
    `corners_against`, `goals_for`, `goals_against`, `sot_for`, `sot_against` (null when not recorded);
  - `booked`: up to 5 players with the most cards this season: `player_id`, `name`, `yellow`, `red`, `appearances`;
  - `shooters`: up to 8 likely starters (most starts in the last 5): `player_id`, `name`, `position`, `starts_last_5`,
    and their last 10 appearances as `{date, opponent, minutes, sot, started}`, plus `price_sot_1plus` and
    `price_sot_2plus` (`{price, bookmakers}` or null);
  - `allowed`: shots on target conceded a match this season and its rank in the league;
- `referee`: null, or `name`, `referee_key`, this season's `matches` and `cards_per_match`, the league's average,
  and `last_matches` (up to 10: `date`, `home`, `away`, `competition`, `cards`);
- `h2h`: up to 6 past meetings since 2023: `date`, `competition`, `home`, `away`, `score`, `cards`, `corners`,
  `goals`;
- `lineups`: null, or the confirmed line-ups when the read has them;
- `data_basis`: `latest_finished_kickoff`.

**B3. Checks.** Why Chelsea's 2026/27 ranks show "stale or missing", and why Anthony Taylor has no 2026/27 matches
while Michael Oliver's run to 8 Nov.

**B4 (later).** An all-competitions scope for player and team profiles, so the player page's "All competitions"
toggle works. Until it exists, the toggle is not shown.

**B5. Early-season ranks.** For a player under the 900-minute threshold, this season's `rank`, `percentile` and
`peer_count` in each metric, with `ranking.provisional: true`. Today they are null (Palmer at 888 minutes), so the
page can only show last season's. Until B5 exists, the rank card shows this season's values without dots, says
"Ranked from 900 minutes · 888 so far", and links to last season's ranks; it never mixes the two seasons.

### 6.3 Building before B1 and B2 are merged

The frontend builds against the schemas above with saved test payloads made from real matches. The rail, the bar
and the match view stay hidden while the endpoints answer 404, so the frontend can merge first without showing
anything broken.

## 7. Testing

- JS tests for the pure parts (hits over a line, the lead sentence, number formatting, the rail and bar markup,
  the match view's sections) with saved real payloads, in the style of `Scripts/tests/test_*_web.js`.
- Real-browser screenshots on 8100 at 1440 and 390, light and dark, of the team page, the match view (Cards and
  Shots on target), the bar and its sheet over the slip bar, and the four fixes. Before shots from today's site.
- No sideways scroll at 390; 44 px tap targets; keyboard reaches every control; the sheet traps focus and returns it.

## 8. Calls left to the builder (marked as the frontend's own)

- The lead sentence's threshold for "card-heavy" (7 of 10) and the default lines (cards 3.5, goals 2.5, corners
  9.5, shots on target 1+).
- Remembering the bettor's last market and lines per browser (local storage, best effort).
- The tablet range (641 to 1099 px) uses the phone's floating bar, since the rail needs 1100 px.
- The whole overhaul is logged as D107 with one tracker line per task, so no new D-numbers clash with the backend's.
