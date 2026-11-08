# Owner dashboard: how Most likely and Value picks did

4 Nov 2026. Owner's request, shaped in conversation with mockups (tracker D67). For the owner only.

## Why

The owner wants to see how Spix's picks perform over time, market by market, before Most likely goes public
and after: what Spix said and what happened. The Monday scorecard covers the published Value picks in text;
nothing records Most likely at all, and nothing shows either over time.

## What the owner decided

- **Two pages.**
  - *Report* (mockup C, "trend first"): a week-by-week line of "said" (the average chance Spix gave) against
    "hit" (how often the picks came in), then one tile per market. Any market can be opened on its own, and
    the page switches between Most likely and Value. Same colours and formatting as the history page and the
    live site.
  - *History* (mockup A, "one line per pick"): looks like the live fixtures page (day strip, league groups,
    match cards), for matches already played. Each match lists the picks Spix made, one line each: which list,
    the pick, the chance Spix gave and the price, what actually happened ("Under 11.5 corners · 10 corners ·
    Hit"), and Hit, Miss or Pending.
- **Local only.** Its own port, **8300**, started when the owner wants it, never part of the live website;
  visitors never see it.
- **No interference with what runs.** No change to the engine or to any running job. It logs the picks and
  presents them.
- **Most likely is logged from now, while the display is still off** (D67), **and rebuilt for past matches**
  that have a stored read (playback accepted 4 Nov).
- **Value means stage 4's published picks** (selection policy `market_value`, from 3 Nov). Earlier picks came
  from older engine versions.

## Non-goals

- No public page, no login, nothing on port 8000.
- No money figures (profit, return); offered, not asked for. Hit against said only. *Lifted for Value picks on
  8 Nov at the owner's request: units and ROI per day and per week (D79).*
- No editing or hiding picks. No cards or shots-on-target in Most likely (the layer does not cover them).
- No legacy (pre-stage-4) picks.

## Builder's calls (the owner can overrule any)

- **Its own database file**, `Index/owner_dashboard.db` (SQLite). `Index/platform.db` is only read, inside
  `data_access()`, in short statements; nothing is ever written to it. So there is no migration and no change
  to the measurement cycle: this replaces the recommendation plan's task 6 (a ledger table in `platform.db`,
  migration 0011, called from `measurement_cycle`) and stands in for task 7 (a scorecard section), which is not
  built unless the owner asks.
- **The log is built from stored reads.** Match reads are never deleted, so the last read evaluated before the
  match's own kickoff (`fixtures.kickoff_utc`, either stage: what the site showed last) can be found at any time.
  Most likely is that read's `recommendations.block(...)["most_likely"]`, computed once per match, when the
  match is over (`FT`, `AET`, `PEN`), and stored with the block's version. A postponed match waits, so a later
  read for its new date is the one logged.
  D67's `shown` flag is dropped: a log built on demand cannot know the display switch at kickoff. When the
  lists go public, the launch date splits the record.
- **Updating.** The dashboard updates its file when it starts and when the owner clicks "Update now". Reads
  come in batches of 20, each batch under its own `data_access()`, so the refresh can take the gate between
  them. If the refresh holds the gate, the update is skipped and the pages say when they were last updated;
  the pages themselves only read the dashboard's own file.
- **Grading Most likely.** Full time (`FT`) only. Result, goals, both teams to score and handicap from the final
  score (`recommendations.wins`); corners from both teams' statistics. Pending, with a reason, when the match
  went to extra time (`AET`, `PEN`: the stored score includes extra time) or its corners are not in yet. Options are half lines, so there are no pushes. A match is logged only once it
  is over, so "not played yet" never appears.
- **Value picks.** The scorecard's rows (`PredictionRepository.get_recent`, published, all scopes), stage 4 only,
  each selection once (`scorecard.cohort(rows, "selections")`). Only the days still needed are read: from the
  oldest unsettled stored pick, or 1 Nov on the first run. "Said" is the published `model_prob`; the result is
  the settlement's own outcome; hit rates use `scorecard.outcome_weight` (half results count half, pushes and
  voids carry no weight).
- **What happened** comes from the match itself: the score, corners (home–away), cards (yellow, red) and shots on
  target from both teams' statistics; a settled Value pick also shows its settlement's `actual_result`.
- **Server.** Python's standard library, bound to `127.0.0.1` only; pages rendered on the server; one small
  inline script for the chart's hover readout. Start: `python -m Scripts.ops.owner_dashboard`.
- **Look.** The live site's colour tokens (light and dark) and its font (Archivo, read from the live site's
  font files if present, else the system font), copied into the dashboard's own stylesheet, so a front-end
  change cannot break it.
- **The chart.** Weeks are ISO weeks by kickoff. Hit is the site's chart colour 1, solid; Said is chart colour 2,
  dashed (validated pair, light `#0A6D37`/`#4D97DE`, dark `#47AA62`/`#2773C0`, all checks pass). Legend, direct
  labels at the line ends, a crosshair readout on hover and focus, and the weekly numbers in a table under the
  chart. A week with no settled pick has no point.

## Smaller calls (from the builder check)

- The history lists only matches with at least one logged pick. Days and kick-off times are the Mac's local
  time. It opens on the most recent day with a settled pick; the day strip moves a week at a time.
- "Said" and "Hit" in the report are both taken over the settled picks only; pending picks are counted beside
  them. Markets with no picks are left out.
- Each line's price: the typical price for Most likely, the odds taken and the bookmaker for Value.
- Value picks are labelled in the layer's words ("AS Roma −1.5", "Under 10 corners", "Lens to win"), built from
  the pick's market, side and line and the match's teams; the engine's own wording is the fallback.
- League names are the competitions' full names (Premier League, Champions League, …).
- One update at a time; a second "Update now" while one runs waits for it.
