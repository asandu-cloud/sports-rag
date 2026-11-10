# Stats questions: design

Date: 10 Nov 2026. Status: **the prototype was built (D92). On 10 Nov the owner asked for it on the live website as
its own tab, "Research Assistant", for signed-in members: see "The website tab" (D94), which overrides "Where it
lives" and the phases below.** Tracker: D91 (renumbered from D90 at the merge of 10 Nov; the frontend took D90
first).
Branch: `backend`. Nothing is built before the owner approves. An implementation plan follows, one per phase, with
superpowers:writing-plans.

## Why

The owner, 10 Nov: "all of our stats are currently being stored inside of SQL right? This means in theory we could
have a sort of rag that instead of looking for embeddings, writes an sql query and returns the statistics the user
is looking for".

They are. `Index/platform.db` holds every finished match of the 13 competitions since 2019/20: 32,247 matches, a
team row per side (about 64,500), about 916,000 player rows, the referee for all but 15 matches (943 referees under 1,599
spellings), and card events since 20 Sep. Embedding search finds text that looks similar; it cannot count or
average. The database can, exactly.

It fits the identity. Section 5.3 asks for "the research, in one place … with the short version first and the
detail on request". Today the Research Area shows fixed profiles. A bettor with one specific question ("how often do
Brentford's away games go over 9.5 corners?") has to work it out from tables, which is the homework Spix exists to
remove.

## Where it lives

~~A fourth tab at the top of the Research Area, next to Players, Teams and Referees (owner, 10 Nov).~~ Replaced the
same day: its own tab in the website's top bar, "Research Assistant" (see "The website tab").

## The picture (my guess, for the owner to correct)

Sunday. The bettor is looking at Brentford v Spurs and likes Over 9.5 corners but wants to check first. They open
the Research Area's question tab and type "how often do brentford away games go over 9.5 corners". About two
seconds later the tab shows:

> **Brentford away, Premier League 2026/27, 6 matches**
> Over 9.5 corners in 4 of 6. Average 10.3 corners a match; the league's average is 9.8.
> *See the 6 matches*

(Numbers made up for the example.) They add the leg, or they don't. Either way it is still their bet.

## What success looks like

- **Every number is right.** Numbers come from tested queries, never from the language model.
- **It understands the question.** In the owner's trial (phase 2), at least 9 in 10 questions reach the right
  question type with the right team, venue, season and line. Every answer starts by saying what it counted, so a
  misreading shows on the card instead of hiding inside a number.
- **It is quick.** About 3 seconds from question to answer.
- **It costs the rest of Spix nothing.** The daily refresh, the picks and the live database's locks are unaffected.

## Goals, non-goals, anti-goals

**Goals**
- A bettor asks a plain question about teams, players, referees or a match and gets exact numbers from Spix's data,
  with the matches behind them one tap away.
- The owner gets the same box with free-form SQL, to explore the data and to find the questions the fixed set
  misses.

**Non-goals**
- Predictions, picks or advice. The question box describes the past. Spix's view of the match stays in its Match
  Read and recommendations.
- Bookmaker prices or odds history (later, from `football_data.sqlite3`, if wanted).
- Live, in-play stats.
- General chat. Off-topic questions get one line saying what the box can answer.
- Discord. Whale chat stays as it is until A7, which can point it at the same question types.
- Any change to `platform.db`: no new table, no migration.

**Anti-goals** (failures even if it works)
- A confident wrong number.
- An answer that reads like a tip ("back the over").
- Anything that keeps people betting: no "want another bet?" follow-ups, no prompts to ask more.
- A slower or failing daily refresh.

## How it works

### 1. The stats file

A separate SQLite file, `Index/stats.db`, built from `platform.db` once a day. Questions never touch `platform.db`.
That matters because a long read on `platform.db` blocks writers (the 23 Oct and 29 Oct lessons, and D88's). One
slow question from a chat box must never hold up the refresh, the Match Read worker or measurement.

- **Built by the daily refresh**, as a new stage after the sync and before the health check. The refresh already
  holds the gate, so nothing else is writing while the build reads. A separate job would read for a long time while
  other jobs write, which is the 29 Oct "busy evening" pattern.
- **Best effort.** The stage accepts exit codes 0 and 1, like `chroma_embed` did: if the build fails, yesterday's
  file stays, the refresh still succeeds, and the failure shows as a warning.
- **Swapped in whole.** It is written to a temporary file and then renamed over the old one, so a reader never
  sees half a file. Readers already open keep the old file until they finish.
- **Flat, pre-joined tables**, so every query is simple and the hard joins are written once, in tested code:
  - `matches`: one row per finished match (competition, season, date, round, home, away, score, referee).
  - `team_matches`: one row per team per match, with the team's and the opponent's numbers side by side
    (`corners_for`, `corners_against`, `goals_for`, `xg_against`, and so on), venue and result. Getting "for"
    and "against" or home and away the wrong way round is the commonest mistake in hand-written football SQL;
    this layout settles it once.
  - `player_matches`: one row per player per match played, with minutes, started or not, and their numbers.
  - `teams`, `players`, `referees`: names and the spellings people type.
  - `build`: when it was built, from which refresh, and the row counts. The card shows "Data up to <date>".
- **The same rules as the Research Area**, so the two never disagree: `normalize_research_row`'s count-null policy,
  the referee alias map, and cards as yellows plus reds.
- **Size:** about 100–200 MB (estimate; most of it player rows).

### 2. Question types

A fixed set of tested queries, each with typed parameters (team, player or referee by id, a stat from a fixed
list, venue, competition, season or last N matches, a line).

| # | Type | Example |
|---|---|---|
| 1 | Team average | "How many corners do Brentford get away from home?" |
| 2 | How often over or under a line | "How often do Brentford's away games go over 9.5 corners?" |
| 3 | Recent form | "How have Spurs been doing lately?" |
| 4 | Head to head (since 2019/20) | "How do Arsenal v Chelsea games usually go?" |
| 5 | Referee | "How many cards does this referee give?" |
| 6 | Player | "How often does Saka have 2+ shots on target?" |
| 7 | Leaders | "Which La Liga team wins the most corners?" |

Every answer carries the competition's average for the same stat, so a number always has a reference point.

### 3. The language model's job: choose, never count

The model reads the question and picks one to three question types with their parameters, as tool calls. The code
then checks each parameter against its fixed list or a real id, runs the query and builds the answer. The model
writes no numbers at all.

- **No match is given**, because the tab sits in the Research Area. When the question names no competition or
  season, a team's question counts its domestic league this season, a player's counts their club's league this season,
  and a referee's counts every competition this season. The scope line says which, and the bettor can ask again
  with another.
- **Names** go through the existing Research search (teams, players, referees with their aliases). If a name matches
  more than one, the card asks which one.
- **One follow-up.** The last question's choices are kept, so "and at home?" works.
- **Questions it cannot answer** get one line on what it can answer.
- **"What should I bet?"** points to Spix's recommendations on the matches page, using Whale chat's existing betting-request
  check (`whale_chat_policy.py`) as the starting point.
- **Prompt tricks do little harm.** Whatever the question says, the model can only ask for the allowed queries
  with checked parameters. Questions are limited to 300 characters.

**The model: `gpt-6-luna`** (owner asked for the cheapest GPT model that is enough, 10 Nov). From OpenAI's pages on
10 Nov:
- It is OpenAI's own pick for "focused, high-volume tasks", supports function calling, and lets the builder turn
  reasoning off or low.
- Price per million tokens: $0.10 in ($0.01 when the repeated instructions are cached) and $0.50 out.
- Only `gpt-5-nano` is cheaper ($0.05 and $0.40), and OpenAI shuts it down on 11 Dec 2026. `gpt-4o-mini`, which
  Whale chat's code defaults to, shuts down on 14 Jan 2027; its named replacement is `gpt-6-luna`. No shutdown is
  listed for `gpt-6-luna`.
- Estimated cost: about 2,000 tokens in (instructions and the seven question types) and up to 300 out per question,
  so about $0.0004 a question, or about 40 cents per 1,000. The trial measures the real figure.
- If it falls short of 9 in 10 in the trial, the next step up is `gpt-6.1-terra` ($1 and $5, about ten times the
  cost, still well under a cent a question).
- It needs OpenAI credit, which has been empty since 5 Nov. Topping up is the owner's call.

### 4. The answer card

A new data block, `stats_answer.v1`, made on the backend and drawn by the frontend:

- **The scope line:** what was counted ("Brentford away, Premier League 2026/27, 6 matches").
- **The headline:** one sentence from a fixed template per question type.
- **A small table** where the type needs one (two teams side by side, a top five).
- **The matches behind it**, on request.
- **A small-sample note** under 5 matches.
- **"Data up to <date>"**.

Past counts are worded as counts ("in 4 of 6 matches"), never as chances. Spix's chances belong to the Match Read.

### 5. Free SQL, owner only

In the owner dashboard (local, port 8300), a second mode lets the model write its own SQL over `stats.db`. It is
the owner's original idea, kept away from bettors until it proves accurate. Each query:
- runs on `stats.db` only, opened read-only, with `PRAGMA query_only`;
- passes an authorizer that allows reading and nothing else (no ATTACH, no PRAGMA, no writes);
- stops after 2 seconds and returns at most 500 rows;
- shows the SQL under the answer, so the owner can check it.

Questions that need free SQL more than once become new fixed question types.

### 6. Logging and limits

- **A log of every question** in its own small file, `Index/stats_questions.db`, never in `platform.db`. Each entry
  records the question, the types and parameters chosen, the answer, the model, the time taken and the cost.
- **Daily limits:**
  - free members get one question a day (owner, 10 Nov);
  - paid members get 50 a day, a cost guard rather than a product limit (owner, 10 Nov: "75 cents a user is fine.
    Lets cap it at 50 a day even though I doubt anyone will even reach that");
  - the whole site stops at 2,000 questions a day (my call, a ceiling on the bill, about 80 cents a day at the
    estimate above).

  The free and paid split works once the two tiers are in the code (an open item before launch). Until then only
  the owner uses it.
- **Cost per member** (owner asked, 10 Nov), at a cautious $0.0005 a question. The input estimate is checked
  against Whale chat's 15 tools, which come to about 1,500 tokens with a 630-token prompt.
  - A free member at the limit every day: 30 questions, about 1.5 cents a month.
  - A paid member asking 5 questions a weekend: about 1 cent a month.
  - A paid member asking 5 a day: about 7.5 cents a month.
  - A paid member at the 50-a-day cap every day: about 75 cents a month, 3% of the 25 euros.

  Nothing else grows with members: the questions use no API-Football requests, and the stats file is built once a
  day on the Mac.
- **Questions need an account**, free or paid, so the limits can be counted.
- **The owner reviews a sample** of the log in the trial: right type, right names, right scope.

### 7. Switches and the stage rule

Nothing here changes a published pick, so it can run alongside the cards weekends. Two switches, both off by
default: `SPIX_STATS_STORE` (the refresh stage) and `SPIX_STATS_QUESTIONS` (the website endpoint). The golden check
stays unchanged.

## Phases

1. **The stats file and the question types**, backend only, no language model. The seven types as plain functions
   with tests. Their numbers are checked against the Research Area for the same team, player or referee. Nothing
   visible changes.
2. **The owner's trial, in the Research Area's new tab.** The backend adds the endpoint behind
   `SPIX_STATS_QUESTIONS`. The frontend builds the tab, with mockups on real answers first (the UI rules). During
   the trial the tab shows only through a preview link (as `?preview=evidence` did, D84), and the endpoint answers
   only the owner. Free SQL sits in the owner dashboard (local, port 8300). The trial runs for one to two weeks of
   the owner's own questions, and it passes at 9 in 10 understood with no wrong numbers.
3. **Bettors**: the tab is shown to everyone signed in, with the daily limits.
4. **Later, if wanted**: free SQL for bettors, if phase 2 shows it is accurate; price history from
   `football_data.sqlite3`; Whale chat on the same types at A7.

## Identity check (section 9)

1. **Easier Sunday:** yes. The answer to a homework question comes in one line, next to the profiles they would
   otherwise read through.
2. **Still their bet:** yes. It gives facts, not picks.
3. **No betting maths:** yes. It speaks in counts and averages, with no probabilities and no fair prices.
4. **Honest:** yes. The numbers come straight from the data, every card says what it counted, small samples are
   flagged, and the matches are one tap away.
5. **Pushes to bet more:** no. It answers only what was asked, never suggests a bet, and never invites another
   question.

Public copy follows the rules: no "AI" (for example "Ask a question about any team, player or referee. Answers come
straight from Spix's match data."), and 18+ and help lines as everywhere.

## What it reuses

- `research_stats.normalize_research_row`, the count-null policy.
- `research_profiles`: its metric names and labels, the referee aliases, and `add_known` for cards.
- The Research search for names.
- `whale_chat_policy.py`'s betting-request patterns.
- The refresh plan's best-effort stage (`RefreshStage`, `accepted_exit_codes=(0, 1)`).

## Calls I made (for the owner to check)

- A separate stats file, built inside the refresh, never questions on `platform.db`.
- The model chooses, the code counts. Numbers never come from the model.
- Free SQL for the owner only at first, in the owner dashboard.
- The owner's trial comes before bettors see anything, behind a preview link.
- The default scope when a question names none: the team's or player's domestic league this season, and every
  competition for a referee.
- The site stops at 2,000 questions a day.
- Cards are counted as in the Research Area: a red is one card. Spix's cards picks settle with a red as two
  (`yellow_1_red_2_max_3_per_player`, D54), so a "cards" answer here can differ by one from a cards pick's
  settlement when a red was shown. The card says "yellow and red cards".
- English only.

## Left to the builder

- The exact stat lists per question type, and the headline templates.
- How the build adds only new or changed matches (`fixtures.updated_at` after the last build), if a full build takes
  over 2 minutes. Measure the full build first.
- Indexes in `stats.db`.
- How long questions stay in the log.
- How a name with several matches is shown.
- How the endpoint knows the owner during the trial.

## The owner's answers (10 Nov)

1. **Where:** "i would first test it as a separate part of research area, so a 4th thing to press on at the top
   essentially". I had suggested the match page; the Research Area it is.
2. **Free members:** "lets do one question a day for a free member".
3. **Model:** the owner asked which is "the cheapest gpt model that would be enough". The answer is `gpt-6-luna`
   (section 3).

Still open: topping up OpenAI's credit, which is needed from phase 2. Phase 1 needs no model.

## The website tab (owner, 10 Nov; D94)

The owner tried the prototype on 10 Nov: "i like it. works well. lets add it as a separate tab to the production
website. the one at port 8000. name the tab "Research Assistant"". Asked where and who:

- **Where:** its own tab in the top bar, after the Research Area: Fixtures · Research Area · Research Assistant · Best
  Bets.
- **Who:** signed-in members. A free account gets 1 answered question a day, a paid one 50, and the whole site stops
  at 2,000 (about 40 to 80 cents a day). Visitors see the tab with a sign-in prompt. This skips the owner-only trial
  of phase 2; the owner's questions on the local page were the trial.
- **Paid** means a paid tier (starter, pro, elite) with an active or trial subscription. Cancelled or unpaid (past
  due) counts as free. This is my call, from the site's own fields; the two-tier price (D50) changes nothing here.
- **A day** runs from midnight UTC (midnight in the UK in winter). A question that fails, is empty or is too long does
  not use up a member's question. A pick from a "Which one?" card is free too.

### The data the tab reads (backend, `Scripts/web_app/routers/research_assistant.py`)

All three answer only while `SPIX_STATS_QUESTIONS=on`. Errors come as `{"detail": {"kind", "message"}}`, with a
message fit to show.

- `GET /api/research-assistant/status` (sign-in optional): `{"enabled": false}` while the switch is off, so the tab
  stays hidden. Otherwise `{"enabled": true, "signed_in", "data_up_to"}`, plus for a member `"plan"` (`free` or
  `paid`), `"daily_limit"`, `"asked_today"` and `"remaining"`. `data_up_to` is a label such as "Sun 8 Nov 2026", or
  null when there is no stats file.
- `POST /api/research-assistant/ask` with `{"question", "previous"}`: `question` up to 300 characters; `previous` is
  the last answer's `{"question", "calls"}`, for one follow-up ("and at home?"), refused with 422 over 4,000
  characters; the model sees only its known choices with short values. Returns `{"question", "calls",
  "answers", "remaining"}`. Each answer is a `stats_answer.v1` card. Errors: 401 `sign_in`, 400 `empty` or
  `too_long`, 429 `member_limit` or `site_limit`, 503 `unavailable` (no stats file, or the model failed; the reason
  stays in the owner's log).
- `POST /api/research-assistant/choose` with `{"type", "params"}`: only a pick from a "Which one?" card this member
  was shown today. Build it from that answer's call: its `params`, unchanged, with the option's `field` set to its
  `value` (a `"#id"`). Anything else is refused. Returns `{"calls", "answers"}`. Uses no model and no question.
  Errors: 401 `sign_in`, 400 `not_a_pick`.

**A card** (`stats_answer.v1`): `type` is one of the seven question types, or `which`, `declined`, `invalid` or
`not_found`. It has `scope` (what was counted; null on the last four), `headline` (one sentence), `detail` (lines),
`table` and `matches` (each `{"columns", "rows"}` or null; `matches` are the matches behind the count, shown on
request), `note` (the small-sample note or a leaders' rule), `data_up_to`, and for `which` an `options` list of
`{"label", "value", "field"}`. `calls[i]` is what the model chose for `answers[i]`: `{"type", "params"}`, so the tab
can say how it read the question.

### What the tab shows (frontend; mockups on real answers first)

- A question box (300 characters), a few example questions, and the questions left today.
- The answer cards as on the local page: scope, headline, detail, the table, the matches on request, the note and
  "Data up to". "Which one?" cards offer their names as buttons, which call `choose`.
- How it read the question, in words (for example "Brentford · corners · away · Premier League"), not the raw
  parameters.
- Visitors: the tab and a sign-in prompt. At the limit: the API's message.
- Not on the website: free SQL, the question log, the Right/Wrong verdicts and any cost. They stay on the owner's
  local page (port 8400), whose log also lists the website's questions.
- Copy: past counts, never chances; "statistical models", never "AI"; no tipster words; nothing that pushes anyone to
  ask or bet more; 18+ and the help line, as elsewhere on the site.

### Switches and going live

- `SPIX_STATS_STORE=on`: the daily refresh rebuilds the stats file just before its health check (about 20 s, best
  effort: a failure is a warning, and the website keeps yesterday's file).
- `SPIX_STATS_QUESTIONS=on`: the website's Research Assistant. A website restart is needed after the merge and after
  the switch.
- **Privacy** (the owner's call on the wording): questions are sent to OpenAI to be read (`store=False`, so OpenAI is
  asked not to keep them), and the owner's log keeps each question with the member's id. The privacy page should say
  so before members use it.

## The team's next match (owner, 10 Nov; D95)

The owner: a bettor uses the Research Assistant to judge a line they see in their betting app ("he sees the over 7.5
corners in his betting app and wants to see what Brentford usually have in their games when they play away"). Spix
already has a pick for that market in the team's next match, so the answer shows it, small. Of four mockups on real
data (artifact 8VvYQo5NhZimpjRGjSixAu, page "Subtle options"), the owner chose **B, a row in the table**, with the
pick highlighted in the lime marker, and turned down a line ladder ("clutters the page").

- **When.** A `team_average` or `line_record` answer about **corners or goals** that counts this season (or the last N
  matches). Cards and shots on target stay out, as they stay out of the picks (D61). A question about a line gets the
  same row and pick: never the asked line's own chance (owner).
- **Which match.** The team's next unplayed fixture in a league Spix reads (`PUBLIC_MATCH_READ_LEAGUES`), whatever the
  venue the question asked about, from the stored schedule (`FixtureScheduleRepository.next_for_team`).
- **Which figures.** The card the match page shows for it, read the same way (`match_reads._actionable_reads`, so a
  read the board withholds shows nothing): "Corners expected" or "Goals expected", and the market's suggestion in the
  `recommendations.v1` block. Nothing is worked out afresh. Spix prices the whole match's corners only, so a corners
  row has a match figure and dashes for the team and its opponent; a goals row has all three.
- **Before the read.** Reads appear up to 96 hours before kickoff (`MATCH_READ_OUTLOOK_HOURS`). Until then the answer
  carries a note in place of the row: "Spix's read of Brentford v Everton (Mon 23 Nov) comes out from Thu 19 Nov."
- **No model call, nothing extra to pay, no pick changes.** A failure leaves the answer as it was.

### The data (backend, `stats_answer.v1`)

`team_average` and `line_record` cards gain `subject`: `{"team_id", "team", "stat", "current"}` (the platform's team
id; `current` is false for a past season). The website adds `next_match` to such a card when there is one
(`Scripts/data_platform/services/next_match.py`):

```json
"next_match": {
  "fixture": {"event_id": "1557458", "league": "EPL", "home_team": "Brighton", "away_team": "Brentford",
              "kickoff": "2026-11-08T14:00:00+00:00"},
  "venue": "away", "opponent": "Brighton", "market": "corners",
  "status": "read",
  "label": "Spix expects v Brighton, Sun 8 Nov",
  "expected": {"for": null, "against": null, "match": 9.5},
  "pick": {"label": "Over 8.5 corners", "side": "over", "line": 8.5, "chance": 0.704, "typical_price": 1.36},
  "note": null,
  "prices_as_of": "2026-11-08T13:56:26"
}
```

- `status` is `read` or `waiting`. While waiting, `label`, `expected`, `pick` and `prices_as_of` are null and `note`
  says when the read comes out (or "isn't out yet" inside the 96 hours).
- `pick` is null when Spix has no pick in that market; `note` then says "No pick from Spix in corners for that match."
- `expected` is to one decimal; `for` is the asked team, `against` its opponent. It can be null if the card lacks
  the figure.
- Dates are UK dates. `prices_as_of` is UTC without an offset, as in `recommendations.v1`.

### What it shows (frontend)

Option B: one more row in the card's For / Against / In the match table, `label` then the three `expected` figures
(a dash for null, the figures in plain bold ink), and under the table one line: "Spix's pick for that match:
**Over 8.5 corners** (lime marker), about 2 in 3, at 1.36." with a link to the match page. The marker is the answer's
one marked figure (DESIGN.md's One Highlighter Rule); cobalt stays on links. A `line_record` card has no such table, so
its row is the same sentence-sized line under its details (the frontend's call). A `waiting` entry shows only its
note, in grey, under the table.
