# Played-match report from the glance squares (D121)

**Goal.** On a team page, each square of "Last 10 at a glance" names its match: pointing at it shows a card (score,
the row's figure, what Spix said), clicking opens a report of that played match. On a phone a tap opens the card as a
bottom sheet with "Open the match".

**Design.** Owner-approved renders (17 Nov): https://claude.ai/artifact/U5LRo9EBaPtGFjR2tTeeGf — the hover card (1a),
the phone sheet (1b), page A (2a), the miss (3a) and no-read (3b) cases. Panel B (2b) was not chosen.

**Data.** Everything but Spix's picks comes from the team profile's `match_log` row (both teams' goals, xG, shots, shots
on target, corners, cards; the team's possession and fouls). The picks come from the backend's D122 field
`spix_before_kickoff` on each row: absent = the section is left out; null = "no read kept"; else `read_at`,
`most_likely`, `value`, each pick with `label`, `market`, `chance` and `outcome` (won, lost, push, half_won, half_lost,
or null when not settled). The site never grades a pick itself.

**Route.** `#research/played/teams/<id>/<scope>/<fixture_id>/<market>`. The report draws in the match view's body with
no rail; its back link returns to the team page.

## Tasks

1. **The report** (`research/played.js`, pure; research.js route; CSS): head, the last-10 strip, the market chips, the
   lead for the open market, "What Spix said before kickoff" (all three cases), "The match in numbers" with the team's
   usual over the last 10, earlier and later matches. Tests: `Scripts/tests/test_research_played_web.js`.
2. **The card and the sheet**: the glance squares become buttons; pointing or focusing shows the card and rings the
   match in every row; a click opens the report where the screen can hover, else the sheet with "Open the match".
3. **Finish**: real data on 8100 (Como, Bayern) at 1440 and 390, light and dark; the Spix section against D122's real
   field once it is merged.

## Progress

| Task | Status | Commits | Tests |
|---|---|---|---|
| 1. The report | Done | 3745f89..this commit | played + all 19 *_web.js pass; 8100 at 1280 and 390, light and dark |
| 2. The card and the sheet | Done | 97a057f..this commit | played + all 19 *_web.js pass; 8100: mouse hover and click at 1280, taps at 390, light and dark |
| 3. Finish | In progress | | |
