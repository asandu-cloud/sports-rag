# Stats questions: design

Date: 10 Nov 2026. Status: **draft sketch, three questions open with the owner** (end of this file). Tracker: D90.
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

## The picture (my guess, for the owner to correct)

Sunday, the match page for Brentford v Spurs. The bettor likes Over 9.5 corners but wants to check first. They type
"how often do brentford away games go over 9.5 corners". About two seconds later the page shows:

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

- **On a match page**, the model is told the match: both teams, the referee, the competition. "Home", "away", "this
  ref" and "their striker" then resolve without guessing.
- **Names** go through the existing Research search (teams, players, referees with their aliases). If a name matches
  more than one, the card asks which one.
- **One follow-up.** The last question's choices are kept, so "and at home?" works.
- **Questions it cannot answer** get one line on what it can answer.
- **"What should I bet?"** points to the match's own recommendations, using Whale chat's existing betting-request
  check (`whale_chat_policy.py`) as the starting point.
- **Prompt tricks do little harm.** Whatever the question says, the model can only ask for the allowed queries
  with checked parameters. Questions are limited to 300 characters.

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
- **A daily cap per member** keeps the cost bounded (number to be set by the owner).
- **The owner reviews a sample** of the log in the trial: right type, right names, right scope.

### 7. Switches and the stage rule

Nothing here changes a published pick, so it can run alongside the cards weekends. Two switches, both off by
default: `SPIX_STATS_STORE` (the refresh stage) and `SPIX_STATS_QUESTIONS` (the website endpoint). The golden check
stays unchanged.

## Phases

1. **The stats file and the question types**, backend only, no language model. The seven types as plain functions
   with tests. Their numbers are checked against the Research Area for the same team, player or referee. Nothing
   visible changes.
2. **The owner's trial**, in the owner dashboard: the question box with the model choosing, plus free SQL. One to
   two weeks of the owner's own questions. It passes at 9 in 10 understood and no wrong numbers.
3. **Bettors**: an endpoint on the website behind `SPIX_STATS_QUESTIONS`. The frontend builds the box, with mockups
   on real matches first (the UI rules).
4. **Later, if wanted**: free SQL for bettors, if phase 2 shows it is accurate; price history from
   `football_data.sqlite3`; Whale chat on the same types at A7.

## Identity check (section 9)

1. **Easier Sunday:** yes. The answer to a homework question comes in one line, on the page they are already on.
2. **Still their bet:** yes. It gives facts, not picks.
3. **No betting maths:** yes. It speaks in counts and averages, with no probabilities and no fair prices.
4. **Honest:** yes. The numbers come straight from the data, every card says what it counted, small samples are
   flagged, and the matches are one tap away.
5. **Pushes to bet more:** no. It answers only what was asked, never suggests a bet, and never invites another
   question.

Public copy follows the rules: no "AI" (for example "Ask about this match. Answers come straight from Spix's match
data."), and 18+ and help lines as everywhere.

## What it reuses

- `research_stats.normalize_research_row`, the count-null policy.
- `research_profiles`: its metric names and labels, the referee aliases, and `add_known` for cards.
- The Research search for names.
- `whale_chat_policy.py`'s betting-request patterns.
- The refresh plan's best-effort stage (`RefreshStage`, `accepted_exit_codes=(0, 1)`).

## Calls I made (for the owner to check)

- A separate stats file, built inside the refresh, never questions on `platform.db`.
- The model chooses, the code counts. Numbers never come from the model.
- Free SQL for the owner only at first.
- The owner's trial comes before bettors see anything.
- Leagues default to the match's own competition. Other competitions are counted only when the question asks.
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

## Questions for the owner

1. **Where does the box go first:** on each match page, answering about that match, or in the Research Area, for
   any team, player or referee? Recommendation: the match page. That is where the Sunday bettor is when the
   question comes up, and knowing the match makes names easy to read.
2. **Which language model:** top up OpenAI (Whale chat's code already calls it, and it has had no credit since
   5 Nov), or move to Claude (a new key)? Phase 1 needs neither.
3. **Who gets it:** paid members only, or a few questions a day for free members too? And how many a day for paid
   members?
