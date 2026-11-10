# Betting RAG — Project Guide

## Start here (every session)

### Read these before working
1. **`docs/identity.md`**: what Spix is, who it is for, why they pay. Spix is the one place football bettors go to
   build a good bet without the homework: a hub where a casual bettor (the "Sunday on the couch" picture, section 2)
   browses Spix's recommendations and builds their own bet. Spix recommends; the bettor decides. Section 9's five
   questions gate every feature. Only the owner changes this file.
2. **`docs/rework-tracker.md`**: the source of truth for the engine plan, status and decisions (D1…D52 and on), and the
   "Open items with a destination" table. Where it disagrees with this file or the Engine Rework report, the tracker
   wins. **Log every finding or plan change there in the same commit as the work.**
3. **`DESIGN.md`** and **`docs/brand/`** (incl. `color-palette.md`, `logo/`): how it looks. `PRODUCT.md`,
   `docs/spix-overview.md` and `docs/spix-marketing-brief.md` are out of date where they disagree with the identity.
4. **Specs and plans**: `docs/superpowers/specs/` (designs) and `docs/superpowers/plans/` (implementation plans).

Much of the rest of this file describes the **legacy** engine (`rag_cli_v2.py`, Chroma, Discord bot). The
**new engine** is the Match Read pipeline in `Scripts/data_platform/` and `Scripts/rag_ingest/core/`
(`market_service.py`, `line_selection.py`, `market_goal_model.py`), staged behind switches (below). The website runs
on it. The Discord bot still runs the old engine until A7.

### Current status (as of 11 Nov 2026; the tracker is newer if it disagrees)
- **Engine stages 1–4 live.** Stage 4 went live 3 Nov: `SPIX_MARKET_ANCHOR=full` and
  `SPIX_SELECTION_POLICY=market_value`, every bookmaker, 2% value threshold, price confirmation at 60 minutes,
  failing closed. Value picks are rare by design: 5 of 18 matches on 3 Nov.
  - Passed its safety check at the 9 Nov scorecard: 72 picks (3–8 Nov), said 51.0%, won 53.4%, log-loss gap
    −0.001 against the market, price CLV +2.0%, value at Pinnacle's true close −0.9%. One week shows only that
    nothing is broken (D26).
- **Recommendation layer live 4 Nov (D52, approved D53, live D68).**
  - Each match gets Most likely (up to 3) and Value (up to 2) lists, plus "Explore the match": every market with
    Spix's suggestion marked, a line stepper, and Add on any option.
  - Same-match goal legs are included. Value picks show "Worth it at X or better". Mockups with real matches come
    before any front-end change.
  - Design: `docs/superpowers/specs/2026-11-03-recommendation-layer-design.md`.
  - Plan: `docs/superpowers/plans/2026-11-03-recommendation-layer.md` (8 tasks, the data block first).
  - Built and merged 4 Nov: the `recommendations.v1` block (D56, Value fix D65), the look (version B, tiles, D63),
    the match page, board card and slip (D59, D60). The owner added `SPIX_RECOMMENDATIONS=on` and the website was
    restarted on 4 Nov (D68). `?recs=off` shows the previous card.
  - Task 6 became the owner dashboard (D67): Most likely and Value picks logged from now, said against hit, local
    only. Start it with `.venv/bin/python -m Scripts.ops.owner_dashboard` (http://127.0.0.1:8300/). Since 7 Nov
    it also shows each league, and a league can be opened alone (D78); since 8 Nov, Value picks also show profit
    in units and ROI, day by day and week by week, at one unit a pick (D79). Task 7 (a scorecard section) is dropped,
    and no migration 0011 is needed.
  - "Worth it at" uses the chance a Value pick was made with (D68, live 4 Nov).
  - Started matches keep their picks from before kickoff, with a live status (D70, D71): live 4 Nov at 22:41
    (`SPIX_BEFORE_KICKOFF=on`, `SPIX_LIVE_STATUS=on`; the status costs at most 400 API-Football requests a day).
    Since 5 Nov they sit in their own Live tab, in play first, then finished (D73).
  - Pick notes (D75): merged 5 Nov behind `SPIX_PICK_NOTES` (off). The owner rejected the for/against wording on
    5 Nov (the two lines can contradict each other); a new approach that explains the pick from the data is being
    worked out. The frontend's display (D76) was taken off its branch on 9 Nov and waits for it. Keep the switch
    off.
  - Pick evidence live 9 Nov at 18:51 (D82, D83, `SPIX_PICK_EVIDENCE=on`). "How Spix reads it" sits above
    Most likely. Each Most likely pick in the domestic leagues gets a verdict, a detail line and past matches at
    similar odds, with a sheet of those matches (the owner's "Easier to read" card). Match page only. Open for
    the owner: "How Spix reads it" repeats the chances and goals of the Match Read's figures panel further down.
    `/app?preview=evidence` puts a saved real card (Brighton v Brentford, 8 Nov) on today's board for that
    browser tab, labelled as a preview; `?preview=off` ends it (D84, link only, writes nothing).
- **Cobalt palette live 9 Nov (D80).** Cobalt `#0246A4` replaced the turf green, lime cream `#D5F097` the
  highlighter, and mint cream `#F2FFFA` is the light ground; the dark theme uses a light cobalt `#88B4F2`. Since
  9 Nov evening the dark theme is "cobalt night" (D85): near-black greys with a faint cobalt tint (`--bg #111316`)
  and mint cream text. Cache tags are `nextmatch-20261110a` (style.css, app.js; D96), `crest-20261110a`
  (research.js, research.css; D90, 10 Nov: tall crests such as Tottenham's stay inside their boxes) and
  `night-20261109a` (theme.js). Open for the owner: the steel blue of the Away segment and the "preliminary" label now sits close to
  the cobalt.
- **Home page hero live 10 Nov (D87).** The website's home page shows the landing preview's hero animation
  (Arsenal v Chelsea, labelled Example) in place of the "Illustrative example" card. Since D89 it is light on
  the light page and dark in the dark theme. Files: `hero-run.css` (tag `hero-20261110b`) and `hero-run.js`
  (`hero-20261110a`).
- **OpenAI credit topped up 10 Nov (owner)**, after running out on 5 Nov 18:04. Match Read briefings go back to the
  model's wording; they cost roughly $5–10 a week at the 3–5 Nov pace. Picks never depended on it.
  OpenAI shuts down `gpt-4o-mini` on 14 Jan 2027: Whale chat and the Discord slash commands fall back to it when
  `WHALE_CHAT_MODEL` or `RAG_CHAT_MODEL` is empty (both are set; their values are the owner's to check, D91).
- **Stats questions prototype (D91 design, D92 build, merged 10 Nov).** A local page where a bettor's question about
  past numbers is turned by gpt-6-luna into one of seven tested question types, and code counts the answer. The owner
  also gets read-only free SQL. Separate from the website and the jobs (owner): start it with
  `python -m Scripts.ops.stats_questions serve` (http://127.0.0.1:8400/). It reads its own stats file
  (`Index/stats_questions/stats.db`) and needs OpenAI credit for the model step.
- **Research Assistant backend live 10 Nov at 20:37 (D94, owner).** `/api/research-assistant/{status,ask,choose}`
  for signed-in members: free accounts 1 answered question a day, paid 50, the site 2,000. A pick answers only a
  "Which one?" card the member was shown that day. `SPIX_STATS_QUESTIONS=on` and `SPIX_STATS_STORE=on` are in
  `.env`; the second adds a best-effort `stats_questions_file` stage (about 20 s) to the daily refresh, before
  `platform_health`. The tab (D93, live 10 Nov) sits after Research Area at `/app#assistant` and shows only while
  the status says enabled. Since 11 Nov at 00:23 (D95, D96, owner; website restarted, pid 39692), a team's corners
  or goals answer also shows the team's next match: what Spix expects in it and Spix's pick in that market, or
  when its read comes out.
  Open before real users: the privacy line (questions are logged with the account id), and auth and security (the
  owner will discuss them separately). The website listens on the home network only.
- **No more Chroma updates (D81, owner, 9 Nov).** The daily refresh no longer embeds into Chroma, because nothing
  running reads it; `refresh-season --embed` brings the step back. `Index/chroma` (19 GB) stays until A7.
  Confirmed on the 10 Nov refresh.
- **Refresh lock fix (D88, owner, 10 Nov).** The refresh failures of 27 Oct, 4 Nov and 10 Nov 13:15 were the
  measurement job's heartbeat write landing under the sync's fixture updates. The sync now ends a fixture's
  transaction on a lock and retries up to 3 times, and measurement writes nothing while the refresh holds the gate.
- **Cards work that does not change picks (D54):**
  - add the events to the refresh and backfill them from 20 Sep (done 4 Nov: migration 0010, 554 matches
    backfilled, the refresh saves events from now on, D64);
  - count a straight red as two cards (done 3 Nov in the studies: yellow 1, red 2, at most 3 per player);
  - rerun the calibration study (done 3 Nov: no calibrator passes in corners or cards, so stage 5 has nothing to
    ship);
  - build A5.4 behind an off switch (merged 4 Nov: `SPIX_CARDS_REFERENCE`; cards priced from Pinnacle when on,
    D61, D62).
- **Cards stage (A5.4) live 9 Nov at 18:51**, a day ahead of its Tue 10 Nov slot (owner): `SPIX_CARDS_REFERENCE=pinnacle`.
  No club match is played before 20 Nov, so no pick changed early.
- **Next dates.**
  - No club matches from 9 to 19 Nov (international break). Cards picks start with the 20–22 Nov weekend, once
    cards prices appear (about 18–19 Nov). Then check that new reads record `pinnacle`, and ask the backend for
    checks 4–7 (tracker, 9 Nov). The Mon 16 Nov scorecard has no new matches.
  - Mon 23 Nov 08:00: the first scorecard to judge cards.
  - Tue 24 Nov: earliest next stage (owner, 9 Nov: 17 Nov would share cards' first weekend).
- **Update this block** in the same commit as the tracker whenever a stage goes live or the current project
  changes.

### The stage rule (D26, D29)
Only one change that alters published picks goes live per week. Each is judged at the Monday scorecard after a
weekend of matches. Work that does not change published picks (display, measurement, research, building behind
an off switch) can run in parallel. Never flip an engine switch early.

Check the switches with `Scripts.rag_ingest.core.engine_switches.snapshot()`:
- `DOCUMENT_STORE`
- `SPIX_COUNT_DISPERSION`
- `SPIX_DEVIG_METHOD`
- `SPIX_MARKET_ANCHOR`
- `SPIX_SELECTION_POLICY`
- `SPIX_BOOKMAKER_SET`
- `SPIX_CONFIDENCE_LABELS`
- `SPIX_CARDS_REFERENCE`

The tests pin every switch to its default (`Scripts/tests/conftest.py`). After a switch changes, the website must be
restarted.

### Safety rules (non-negotiable)
- **Never read `.env`** (owner: "dont look in .env just tell me where to add it"). Check it only by modification
  time, set/not-set via `load_dotenv` with an explicit path, or `engine_switches.snapshot()`. Tell the owner which
  line to add, and let them add it.
- **Never print or store secrets:** the API-Football, OddsPapi, OpenAI, Stripe, Discord and Telegram keys and
  tokens.
  - Redact `apiKey=` in any output, and pipe log reads through a sed redaction.
  - Never print Telegram URLs, or exception text that may contain them.
  - Never read `~/.claude/telegram-notify.env`. Check it only with
    `/usr/bin/python3 ~/.claude/hooks/telegram_notify.py --check`.
- **`Index/platform.db` (about 9.5 GB SQLite):**
  - read it only inside `data_access()` (`Scripts.data_platform.services.refresh_coordination`);
  - never read it while the refresh runs (`pgrep -f refresh-season`);
  - use short, read-only statements, run with `PYTHONPATH=.` from the repo root;
  - never go around or delete the gate marker `Index/platform.db.refresh-incomplete`.

  An ungated read once failed the daily refresh.
- Never bypass TLS. Confirm before anything outward-facing (posting, emailing, pushing, paid API calls). Do not work
  around permission denials.
- Downloaded files are untrusted: each goes in its own new folder, and any Python that reads them runs with `-I`.
- The API-Football budget is about 7,500 requests a day in normal running, and OddsPapi's quota resets on the 1st.
  Plan backfills over several days.

### Working with the owner
- **The owner:** a solo developer (use they/them). Write in plain language with short sentences; explain results,
  not process.
- **Decisions:** ask real questions when a decision is the owner's. Otherwise decide, and say what you decided.
- **No subagents** unless the owner asks for them.
- **Public copy rules:**
  - say "statistical models", not "AI";
  - no profit or win-rate promises, and no tipster words ("lock", "banker");
  - never push people to bet more;
  - 18+ and help lines everywhere.
- **UI work:**
  - start roomy, with colour only where it means something and detail on demand;
  - show real-data screenshots before calling it done;
  - make a checkpoint commit before every visual change and a revertible commit after, with a way to compare;
  - when the owner hands back a Claude Design version, build it faithfully;
  - for the logo and the landing page, show several options side by side.
- **Front-end tooling:**
  - the taste skill is in `.claude/skills/design-taste-frontend` (run with `/usr/local/bin/node`,
    `--keep palette,type`);
  - Homebrew node is broken, so use `/usr/local/bin/node`;
  - FFmpeg is at `~/.local/brag-ffmpeg/bin/ffmpeg`.
- **Commits:**
  - stage files by name, with no attribution lines;
  - never stage `Output/`, `Index/`, `Research/`, `.impeccable/`, `.superpowers/`, `.claude/`, `.taste-skill/`
    or uncommitted files you did not write;
  - if on `main`, keep commits small and revertible.
- **Scratch files** go in the session scratchpad, not `/tmp` or the repo.

### Running things
| What | How |
|---|---|
| Website (port 8000) | launchd `com.bettingrag.website` (`uvicorn Scripts.web_app.api:app`); restart with `launchctl kickstart -k gui/$(id -u)/com.bettingrag.website` |
| Daily refresh | launchd `com.bettingrag.daily-data-refresh` (`python -m Scripts.data_platform refresh-season --include-europe --if-due`, retries hourly) |
| Match Reads | launchd `com.bettingrag.match-read-worker` (`python -m Scripts.data_platform match-read-cycle --once --mode website`) |
| Weekly scorecard | launchd `com.bettingrag.scorecard`, Mondays 08:00 (`python -m Scripts.ops.scorecard --notify`) |
| Other jobs | `com.bettingrag.{true-close,prediction-measurement,odds-archive,odds-archive-near-kickoff,fixture-schedule,bookmaker-coverage,phase6-shadow}` in `~/Library/LaunchAgents` |
| Tests | `.venv/bin/python -m pytest Scripts/tests` |
| Stage previews | `python -m Scripts.ops.golden_snapshots preview --baseline SWITCH=VALUE` |
| Weekly studies | `Scripts.ops.calibration_study`, `disagreement_study`, `confidence_study`, `confidence_age_study`, `m1_gate_study` (each `--out Research/<name>-<date>`) |
| Landing page (staging) | `landing-preview/`; build `.venv/bin/python -B landing-preview/src/build_page.py`, serve `.venv/bin/python -B landing-preview/serve.py` (port 8010) |

The Mac must stay on mains power with the lid open for the scheduled jobs. Refresh failures alert to Telegram.

### Parallel sessions (set up 3 Nov 2026)
The owner runs up to three Claude instances at once. Each one has its own folder, and there is only ever one
instance per folder.

| Folder | Branch | Port to test on | Owns |
|---|---|---|---|
| `~/Desktop/Betting_RAG` | `main` | 8000 is the live website | **Production.** Every scheduled job and the website run from here. Nobody develops here; only merge here, then run the tests and restart the website if what it serves changed |
| `~/Desktop/Betting_RAG-backend` | `backend` | 8200 | `Scripts/data_platform`, `Scripts/rag_ingest/core`, `Scripts/ops`, tests, engine switches, the cards work (D54), the `recommendations.v1` data format |
| `~/Desktop/Betting_RAG-frontend` | `frontend` | 8100 (app), 8011 (landing preview) | `Scripts/web_app/static`, `landing-preview/`, `website/`, `DESIGN.md`, `docs/brand/` |
| `~/Desktop/Spix-Content` | its own repo | none | Content creation. Not part of this repo, and never writes to it |

- **Which folder am I in?** Run `git branch --show-current`. On `main`, only merge.
- **Shared things.** The worktrees link back to the main folder's `.env`, `.venv`, `Index/` (including
  `platform.db`), `Output/`, `Research/` and `.claude/`. `Index/` is left out of their checkout (sparse checkout),
  so the link is not seen as a change. Data, settings and switches are therefore shared with production: a switch
  changed from any folder changes the live site.
- **Staying current.** At the start of each session, and after anything is merged into main, run `git merge main`
  in the worktree. To merge a branch, run `git merge backend` (or `frontend`) in the main folder.
- **Data rules.**
  1. Frontend and content never write to the database. Content takes its figures from the website.
  2. Backend bulk writes (backfills, re-grading) run first on a copy: point the command at it with
     `DATABASE_URL=sqlite:///<copy>` on the command line, never in `.env`. Only then run it on the real database,
     inside `data_access()` and outside refresh hours. Disk space is tight (about 36 GB free on 3 Nov), so delete
     copies after use.
  3. No schema change to the shared database before its code is merged to main. New tables are fine; renaming or
     dropping waits for the merge.
- **The tracker.** Each branch logs its own work. D-numbers get settled at merge: if two branches took the same
  number, renumber the second. In a tracker merge conflict, keep both entries.
- **This status block** is only updated on main.
- **The data format between them.** `recommendations.v1` changes only on the backend branch; the frontend asks
  for changes there.

#### Committing and merging
- **Backend and frontend** commit only on their own branch, in small steps, each with a clear message and its
  tracker entry in the same commit. They never commit on `main` and never merge into it. When a piece is ready,
  they tell the owner in one short message: what changed, the test results (and screenshots for the frontend),
  and whether it needs a migration, a new `.env` line or a website restart.
- **The orchestrator** is the only instance in `~/Desktop/Betting_RAG`, and the only one that merges into `main`.
  It reviews with `git log main..<branch>` and `git diff --stat main...<branch>`, then reads the diffs that
  matter. It does not write product code, and it never edits files in the worktrees: it writes problems down for
  that branch's instance.
- **Checks before every merge** (on a trial merge):
  - the branch merges cleanly;
  - the full Python suite and the JS tests pass;
  - `python -m Scripts.ops.golden_snapshots check` is unchanged, so no published pick moves with the switches off;
  - none of the never-stage folders are in the branch;
  - no tracker D-number clashes;
  - no rename or drop of anything the live code uses.
- **The merge itself:** `git merge --no-ff <branch>`, so each merge can be reverted in one step. Ask the owner
  before each merge until they say merges that pass the checks may go ahead. Always ask before restarting the
  website or running a migration.
- **After a merge:** the orchestrator updates the status block above and the tracker status if needed, and tells
  the other branch to run `git merge main`.
- **Docs-only changes on `main`** (this file, the tracker status) are made by the orchestrator and committed
  directly, small and revertible.
- **Content** commits in its own repo and never in this one.

## What This Is
A football betting assistant that uses RAG (Retrieval-Augmented Generation) over a Chroma vector database to score and recommend parlays, totals, goal intervals, handicap lines, BTTS, moneyline, correct scores, player props, and team comparisons. Covers the top 5 European domestic leagues (EPL, LaLiga, SerieA, Bundesliga, Ligue1) and 3 European competitions (UCL, UEL, UECL). Delivered via CLI, Streamlit UI, Discord bot (Spix's Picks), live betting bot (SpixLive), and a FastAPI-powered web app (SpixBot).

**Brand:** Spix (website and identity); the Discord server is still Spix's Picks (formerly Spick's Picks, renamed April 2026).

## Architecture Overview

### Data Pipeline
```
API-Football ──▶ Scripts/{League}/*_stats_for_teams.py --season YYYY
                  Scripts/{League}/*_player_stats*.py   --season YYYY
        │
        ▼
*_feature_engineering.py   (per-league derived features)
        │
        ▼
Output/*_feature_engineering/   (per-league engineered stats, multiple seasons)
        │
        ▼
00_normalize_to_docs.py --all   (normalize → Index/normalized_*.json)
        │
        ▼
01_embed_and_upsert.py --all-leagues [--skip-existing]   (embed → Chroma)
        │
        ▼
Chroma collection "football_top5"   (Index/chroma/)
        │
        ▼
rag_cli_v2.py / rag_streamlit.py / Discord bot / FastAPI web app
```

### Canonical Data Platform (`Scripts/data_platform/`)
Additive PostgreSQL layer introduced alongside the season-snapshot pipeline.
See `docs/PLATFORM_V1.md`, `docs/PLATFORM_V2.md`, `docs/PLATFORM_V3.md` for the full write-ups.
- **V1**: competitions, seasons, teams, players, fixtures, per-fixture team/player stats, sync bookkeeping (`sync_runs`, `sync_watermarks`), raw payload archive, team/player feature snapshots, canonical app users/subscriptions/referrals; incremental API-Football sync with watermarks; `PayloadArchiver` (local disk / S3-ready); feature-flagged shim in `web_app/users.py`
- **V2**: predictions, parlay_sessions, trial_users, trial_parlays, live_fixtures + snapshots + alerts, kb_documents + kb_refresh_queue, odds_snapshots; service + repository layer (`data_platform/services/`, `data_platform/repositories/`); drop-in shims for `rag_ingest/prediction_tracker`, `rag_ingest/core/parlay_service.ParlaySessionStore`, `live/live_store.LiveStore`; unified FastAPI (`/api/live/*` mounted from `web_app/routers/live.py`); entity-based KB refresh (content-hashed docs, `spixctl kb-refresh`)
- **V3**: config-driven competition registry (`registry/competitions.yaml` + loader), `feature_builds` lineage table + `feature_version`/`feature_build_id`/`built_at`/`as_of_fixture_api_id` columns, replay/repair tooling (`replay/` + `spixctl replay`, `compare_values`), observability (`observability/` + `HealthReport`, 8 data-quality checks, `/api/health/platform` endpoint), vector backend abstraction (`vectors/` with `VectorBackend` protocol, `ChromaVectorBackend`, `InMemoryVectorBackend`), complete `resolve_outcomes` shim
- CLI: `python -m Scripts.data_platform init-db | bootstrap | refresh | build-features | migrate-users | migrate-predictions | migrate-parlays | migrate-live | kb-enqueue-all | kb-refresh | kb-status | status | health | registry | replay`
- Activation: `PLATFORM_ENABLED=1` in env routes legacy modules through Postgres while preserving every import + Discord UX. `VECTOR_BACKEND=chroma|inmemory` selects the vector layer
- Chroma remains the retrieval layer but is now downstream of `kb_documents` and accessed via `VectorBackend` protocol (swappable with pgvector/Weaviate/etc. by adding an implementation)

### Odds Provider
**API-Football is the sole provider of pre-match odds for reads.** The Odds API has been fully removed from all active code paths. Odds are fetched via `odds_provider.py` which wraps API-Football's `/odds` endpoint. **OddsPapi** is used only for measurement and checks:
- the true close (`services/true_close.py`);
- price confirmation before publishing (`services/price_confirmation.py`);
- price checks.

It carries no Betfair history (D43).

- `odds_provider.fetch_events_apifootball(league, target_date)` — fetches fixtures + odds for a specific date
- `odds_provider.enrich_event_odds(event, league)` — re-fetches odds per fixture to discover additional markets (corners, cards, SoT)
- Events are cached for 10 minutes (`_CACHE_TTL = 600`)
- Card and SoT totals markets typically appear 1-2 days before kickoff; earlier queries show model-only projections

**Date-aware fetching:** `rag_cli_v2.fetch_events()` accepts `target_date` parameter. When no date is specified, `answer_once()` parses the date from the query first (via `parse_explicit_date`), then falls back to a 7-day lookahead if today has no fixtures.

**Orchestrator:** `python Scripts/pull_season.py --season 2023 2024 --normalize --embed --retrain-ml` runs all scripts for given seasons across all 8 competitions, then normalizes, embeds, and retrains ML models.

### Key Files

| File | Purpose |
|------|---------|
| `Scripts/rag_ingest/rag_cli_v2.py` | **Main scoring engine** (~7700 lines). Thin wrappers delegating to `core/projections.py`. Odds fetching, KB retrieval, `kb_leg_quality()` scoring, `select_parlay()` optimizer, 10+ market intents, comparisons, totals, evidence rendering, LLM explanation. |
| `Scripts/rag_ingest/core/projections.py` | **Source of truth for all projection functions** (~57KB). Match-level totals, per-team projections, BTTS, correct score, moneyline, goal difference. All blend formulas live here. |
| `Scripts/rag_ingest/core/weights.py` | **Centralized weight configuration** (~14KB). All scoring/projection weights in one place. |
| `Scripts/rag_ingest/core/team_resolution.py` | **Team data resolution** (~33KB). Profile meta, recent stats, fixture rows, variance — all from Chroma. |
| `Scripts/rag_ingest/core/parlay_service.py` | **Structured parlay builder** — `build_parlay_request()`, `build_parlay()`, constraint spec, candidate filtering, session storage. |
| `Scripts/rag_ingest/core/parlay_models.py` | **Parlay data models** — `ParlayBuildRequest`, `ParlayBuildResult`, `ParlayLegResult`, `ParlayLegRef`. |
| `Scripts/rag_ingest/odds_provider.py` | **API-Football odds wrapper** — translates API-Football bet types to internal format. `_BET_ID_TO_MARKET_KEY` maps bet IDs to market keys. |
| `Scripts/rag_ingest/rag_streamlit.py` | Streamlit UI. Template-based queries (12 categories), league/date selectors, multi-turn chat. |
| `Scripts/rag_ingest/00_normalize_to_docs.py` | Normalization pipeline (~1300 lines). Reads `*_feature_engineering/` dirs, produces 5 doc types, computes 17 opponent-join aggregations. |
| `Scripts/rag_ingest/01_embed_and_upsert.py` | Embeds normalized JSON docs (OpenAI text-embedding-3-large) and upserts into Chroma. Supports `--skip-existing` for incremental updates. |
| `Scripts/pull_season.py` | **Orchestrator** — 5-stage pipeline: data pull → feature engineering → normalize → embed → retrain ML. |
| `Scripts/rag_ingest/prob_models.py` | **Probabilistic primitives** — Poisson, negative binomial, Dixon-Coles bivariate Poisson, implied probability, value edge, EV, Kelly criterion. Pure stdlib. |
| `Scripts/rag_ingest/referee_data.py` | **Referee data layer** — builds profiles + `Index/fixture_referee_map.json`. Deduplicates abbreviated names via `_dedup_referee_names()` (first-initial + last-name grouping). |
| `Scripts/rag_ingest/ml_edge.py` | **ML edge layer** (~3500 lines) — Elo ratings + Ridge/LightGBM regression. 58-feature vector. Cumulative tracker enrichment during training. |
| `Scripts/rag_ingest/player_projections.py` | **Player prop modeling** — per-player projections (goals, assists, shots, cards). Stricter gating: min expected minutes, bench skip, rotation caps, role heuristics. |
| `Scripts/rag_ingest/lineup_context.py` | **Lineup + card risk profiles** — `CardRiskProfile` dataclass, `compute_card_risk_profile()` from player Chroma docs. |
| `Scripts/rag_ingest/league_context.py` | **League context layer** — rest days, congestion proxy, stake context, regime shift (last-5 vs season delta). Bounded adjustments [0.85, 1.15]. |
| `Scripts/rag_ingest/prediction_tracker.py` | **Prediction logging** — tracks predictions, resolves outcomes, daily breakdown. |
| `Scripts/rag_ingest/backtest.py` | **Backtesting harness** — MAE, RMSE, bias, skill vs naive, hit rates, calibration, confidence/referee slices. |
| `Scripts/rag_ingest/cards_backtest.py` | **Dedicated cards backtest** — walk-forward cards evaluation with referee slices and confidence calibration. |
| `Scripts/rag_ingest/newsletter.py` | **Newsletter generation** — matchday intel for Whale-tier. |
| `Scripts/discord_bot/` | **Discord bot** — 8 cogs, 18 slash commands, 8+ background tasks, Stripe subscriptions, Whale AI chat with 17 tools. |
| `Scripts/live/` | **Live betting engine** — async poller, Bayesian projection engine, live odds ingestion, alert generation, FastAPI server. |
| `Scripts/live_discord_bot/` | **Live Discord bot** — separate bot for live match alerts and analysis. 4 slash commands. |
| `Scripts/web_app/` | **FastAPI web app (SpixBot)** — structured JSON prediction API, auth, Stripe webhooks, referral system, track record API. |

### Document Types in Chroma
- **team_profile** — season-level aggregates per team (goals_for_pm, corners_pm, cards_per_90_team, sot_for_pm, possession, dominance_index, control_index, venue splits, opponent-join stats, variance stats, yellows_pm, reds_pm)
- **team_fixture** — per-match stats with opponent-joined fields (yellow_cards, red_cards, cards_total, fouls_committed)
- **player_fixture** — per-match player stats
- **player_profile** — season-level player aggregates
- **data_dictionary** — global glossary of variable meanings

## Scoring Engine (`rag_cli_v2.py`)

### Parlay Flow
1. `parse_constraints()` — extracts ConstraintSpec from user query
2. `parse_explicit_date()` — extracts target date BEFORE fetching events
3. `fetch_events()` — gets odds from API-Football for the target date; 7-day lookahead if no date specified and today has no fixtures
4. `enrich_events_for_groups()` — per-event odds re-fetch via `odds_provider.enrich_event_odds()` for corners/cards/sot/btts
5. `build_candidates()` — creates CandidateLeg objects. Applies MIN_LEG_ODDS (1.10) and MIN_TOTALS_LEG_ODDS (1.25) floors
6. `select_parlay()` — optimizer with quality gate:
   - Hard filter: `_PARLAY_MIN_ODDS = 1.30`
   - **Quality gate**: computes `kb_leg_quality()` for all candidates, rejects legs with quality <= `min_kb_quality` (0.0) — eliminates legs where the model disagrees with the bet
   - Pool reduction: top 220 by odds proximity + quality bonus
   - Solver: brute-force (≤50k combos) or beam-search (60-wide)
   - Combo scoring: `kb_quality_weight` = 1.2 (heavily favors high-quality legs)
7. `leg_evidence()` — generates supporting statistics per leg
8. LLM explanation step (GPT) — synthesizes recommendation text

### Projection Architecture
All projection functions live in `Scripts/rag_ingest/core/projections.py` (source of truth). `rag_cli_v2.py` contains thin wrappers that delegate to core.

**Per-Team Projections (anchored to match total):**
- `projected_team_corners()` — match-total anchored via share model (0.55) + direct stabilizer (0.45). Feature scoring for share: season/recent blend with control, dominance, possession signals.
- `projected_team_cards()` — match-total anchored via cards_share_model weights (0.55 anchor / 0.45 stabilizer). Includes cards_induced, fouls_drawn, aggression signals.
- `projected_corners(home_meta, away_meta)` — 0.6 own + 0.4 opp, 50% venue blend
- `projected_cards(home_meta, away_meta, ref_mod)` — 0.70 own + 0.30 opp + 0.05 agg_nudge, ref modifier multiplicative
- `projected_sot(home_meta, away_meta)` — 0.6 own + 0.4 opp, 50% venue blend
- `projected_goals(home_meta, away_meta)` — 0.6 own + 0.4 opp, 50% xG blend, 50% venue blend

**Match-Level Projections (with dynamic blend shifting):**
- `projected_total_goals()` — 85% season / 15% recent. xG integration (50/50). Knockout modifier
- `projected_total_corners()` — 80% season / 20% recent. Dynamic shift when divergence > 20%
- `projected_total_cards()` — 85% season / 15% recent. Referee strictness modifier. Foul-card blend (0.15)
- `projected_total_cards_detail()` — rich dict: total_cards, total_yellows, total_reds, uncertainty, confidence_bucket, warnings, lineup_card_context
- `projected_total_sot()` — 80% season / 20% recent. Dynamic shift when divergence > 20%

**Outcome Probabilities:**
- `projected_btts_prob()` — Dixon-Coles bivariate model (primary), independent Poisson fallback
- `projected_correct_score_probs()` — Dixon-Coles scoreline matrix (primary), independent Poisson fallback
- `projected_moneyline_probs()` — from scoreline matrix sums
- `projected_goal_difference()` — for spreads/handicap

### Referee Data Layer
Two-layer architecture + ML training integration:
- **Static:** `Index/referee_profiles.json` (~274 unique referees after dedup) + `Index/fixture_referee_map.json` (7,426 fixture→referee mappings across 3 seasons)
- **Deduplication:** `_dedup_referee_names()` merges abbreviated names ("M. Oliver") with full names ("Michael Oliver") via (first_initial, last_name) signature grouping. Reduced 424 → ~274 unique referees.
- **Dynamic:** Live API-Football call for referee assignments, cached 15 min
- **ML training:** `fixture_referee_map.json` loaded during training to populate referee features (strictness_ratio, cards_per_foul) per fixture — 73% coverage
- **Modifier formula:** `multiplier = 1.0 + (strictness - 1.0) * confidence * weight` (weight=0.35)

### ML Edge Layer (`ml_edge.py`)
Two components: Elo ratings + regression models.

**Elo ratings:** Per-team per-stat (goals, corners, cards, SoT). ELO_START=1500, ELO_K=32.

**Regression models:** 58-feature vector (38 base + 16 v2 + 4 stat-specific direct).
- **Base features (38):** possession, control_index, dominance_index, form_index, aggression, fouls, xG, shots, SoT ratio, opponent-adjusted stats, Elo diff, league one-hot
- **V2 features (16):** venue-split rates, opponent defensive cross-context, recent form (last 5), rest days, referee strictness, season stage, H2H history
- **Stat-specific direct features (4):** own production avg + opponent's production avg per team (from cumulative tracker)
- **Algorithm:** Ridge for corners (`FORCE_RIDGE_STATS = {"corners"}`), LightGBM for others when samples >= 5000
- **LightGBM regularization:** `reg_alpha` + `reg_lambda` in param grid to prevent overfitting
- **Training enrichment:** `_TeamCumulativeTracker` builds running per-team averages during chronological training loop, enriches fixture rows with `*_pm` fields. Snapshots taken BEFORE each fixture (no future leakage). Min 2 matches before enrichment.

**Saved R2 scores (`Index/ml_models/model_r2.json`, checked 3 Nov 2026):** goals 0.018, corners −0.010, cards 0.006,
SoT 0.027. All four saved models are Ridge (`*_ridge.joblib`). The R2 figures of 0.43/0.47 once written here were
wrong.

**ML blend configuration:** `dynamic_blend: True`, `skip_stats: {"goals"}`. Blend weight formula: `min(0.30, (R2 - 0.20) * 0.75)`. R2 < 0.20 → 0%, so **the ML layer has zero weight today**. `elo_signal_weight` (0.15) is defined in `core/weights.py` and `rag_cli_v2.py` but never read. `ml_edge.py` and the Elo file are to be retired at A7, replaced by A6's fitted ratings.

## Data Pipeline Per League

All 5 domestic leagues now use aligned logic (matched to EPL's approach):
- **Aggregation merge key:** `fixture_id` (not fixture name string)
- **String column exclusions:** `fixture_id`, `fixture_date`, `fixture_date_utc`, `final_score_string` protected from numeric conversion
- **Merge duplicate resolution:** `_x`/`_y` suffix columns resolved (prefer fixture stats)
- **Rounding:** `.round(2)` across all leagues
- **Team name aliases:** SerieA (`"Verona" → "Hellas Verona"`), Bundesliga (dot-space normalization for "1.FC Köln")

| League | Dir | ID | Team Stats | Player Stats | Feature Eng |
|--------|-----|-----|-----------|--------------|-------------|
| EPL | `Scripts/Premier_League/` | 39 | `PL_stats_for_teams.py` | `prem_player_stats_per_gw.py` | `PL_feature_engineering.py` |
| LaLiga | `Scripts/La_Liga/` | 140 | `LaLiga_stats_for_teams.py` | `laliga_player_stats_per_gw.py` | `LaLiga_feature_engineering.py` |
| SerieA | `Scripts/Seria_A/` | 135 | `SeriaA_team_stats.py` | `SeriaA_player_stats.py` | `SeriaA_feature_engineering.py` |
| Bundesliga | `Scripts/Bundesliga/` | 78 | `bundesliga_stats_for_teams.py` | `bundesliga_stats_for_players.py` | `Bundesliga_feature_engineering.py` |
| Ligue1 | `Scripts/Ligue_1/` | 61 | `Ligue1_stats_for_teams.py` | `Ligue_1_Stats_for_players.py` | `Ligue1_feature_engineering.py` |
| UCL | `Scripts/Champions_League/` | 2 | `UCL_stats_for_teams.py` | `UCL_player_stats.py` | `UCL_feature_engineering.py` |
| UEL | `Scripts/Europa_League/` | 3 | `UEL_stats_for_teams.py` | `UEL_player_stats.py` | `UEL_feature_engineering.py` |
| UECL | `Scripts/Conference_League/` | 848 | `UECL_stats_for_teams.py` | `UECL_player_stats.py` | `UECL_feature_engineering.py` |

**Data completeness:** All 5 domestic leagues have `fixture_date`, `fixture_id`, `home_team`, `away_team` across all 3 seasons (2023, 2024, 2025). Only 2 rows total with merge issues (SerieA 2024 reversed fixture name).

## Discord Bot (Spix's Picks)

### Architecture
**8 cogs** loaded at startup, `discord.py` v2.x with async/await:

| Cog | File | Purpose |
|-----|------|---------|
| `slash_cmds` | `slash_cmds.py` | Core prediction commands including rebuilt `/analyze` |
| `auto_push` | `auto_push.py` | Scheduled background tasks (dynamic timing, 2h before kickoff) |
| `parlay_builder` | `parlay_builder.py` | Interactive parlay builder with follow-ups |
| `subscribe` | `subscribe.py` | Stripe subscription management |
| `whale_chat` | `whale_chat.py` | Whale-tier conversational AI with 17 tools |
| `trial` | `trial.py` | Free trial funnel, graduation, `post_trial_parlays(force=)` |
| `tier_welcome` | `tier_welcome.py` | Tier welcome DMs + new member onboarding (`on_member_join`) |
| `promo_parlay` | `promo_parlay.py` | Admin-only `/promo` command for marketing content |

### Embed Visual Design
- Matchup composite image (both team crests) via Pillow, shown as thumbnail
- Clean fixture name as author text (no icon)
- League badge in footer
- Graceful fallback: league logo if composite generation fails
- `_build_matchup_composite()` in `embeds.py` generates 256×96 PNG with "vs" text
- `get_matchup_file()` with in-memory cache `_composite_cache`
- All embeds return `Tuple[List[discord.Embed], List[discord.File]]`

### `/analyze` Command (Rebuilt)
Full comprehensive match analysis pipeline — calls ALL projection functions directly instead of routing through RAG prompt:
1. Runs every projection: goals, corners, cards, SoT, BTTS, moneyline, spreads, correct score, per-team corners, per-team cards
2. Extracts best odds lines from API-Football for every market
3. Collects team profiles, recent form (last 6), referee data
4. Packages into structured JSON payload
5. Sends to GPT with a report-writing system prompt → written narrative covering every market
6. Rendered as Discord embed

### Whale Chat AI
Private threads in #whale-ai. GPT-4o-mini with **17 tools** for natural language betting analysis:

**Projection tools:** `project_goals`, `project_corners`, `project_cards`, `project_sot`, `project_btts`, `project_moneyline`, `project_goal_difference`, `project_team_corners`, `project_team_cards`

**Data tools:** `get_fixtures`, `get_team_profile`, `get_recent_form`, `get_referee_info`, `get_h2h_stats`, `get_odds`

**Action tools:** `build_parlay` (full optimizer with scored legs), `recommend_line` (best over/under for any stat), `recommend_spread` (best handicap line with edge/confidence)

### Slash Commands (18)

**Prediction Commands (7):** `/fixtures`, `/analyze` (Elite), `/market`, `/compare`, `/players`, `/teamlines`, `/help`

**Parlay Commands (2):** `/parlay` (Pro), `/build` (Pro)

**Subscription Commands (4):** `/subscribe`, `/account`, `/referral`, `/cancel`

**Trial Commands (2):** `/trial`, `/trial-trigger` (Owner)

**Admin Commands (1):** `/promo` (admin-only, single-match ~2.5x parlay for marketing — spreads + totals + cards)

**Other (2):** `/chat` (Whale), `/trigger` (Owner — includes free_picks, newsletter, everything)

### Tier System
Legacy. To be replaced by two tiers, free and paid at 25 euros a month (D50, `docs/identity.md` section 4), before
launch.

| Tier | Role | Cost | Commands/Day |
|------|------|------|-------------|
| Free | None | Free | 1 |
| Rookie (Starter) | ROOKIE | $9.99/mo | 5 |
| Tipster (Pro) | TIPSTER | $14.99/mo | 15 |
| Whale (Elite) | WHALE | $29.99/mo | Unlimited |

### Background Tasks (Auto Push)
- **Market channels** — polls every 30 min, posts 2-3.5h before first kickoff
- **Daily parlays** — 3 presets (Lock ~1.8x, Safe ~2.5x, Standard ~5x) with cross-parlay leg dedup via `excluded_legs`
- **Player props** — 30 min before each kickoff using confirmed lineups
- **Newsletter** — polls every 30 min, posts ~2h before first kickoff (dynamic, not fixed 9 AM)
- **Free picks (trial)** — 2 low-odds parlays to #free-picks, `post_trial_parlays(force=True)` for manual trigger
- **Best picks recap** — 9:00 AM, resolves yesterday's predictions
- **Track record** — 12:00 PM, posts overall accuracy summary
- **CLV capture** — captures closing odds for grading
- **New member onboarding** — DM on `on_member_join` with trial/subscribe/commands guide

## Live Betting Engine (SpixLive)

### Architecture
Separate from the pre-match pipeline. Runs as its own process with an optional Discord bot.

```
API-Football /fixtures?live=all  ──▶  Discovery (60s interval)
                                        │
                                        ▼
                                  LiveMatchState created
                                  Pre-match priors from rag_cli_v2
                                        │
                                        ▼
API-Football /fixtures?id=X      ──▶  Detail Polling (45s interval)
API-Football /odds/live?fixture=X ──▶  Live Odds Fetch
                                        │
                                        ▼
                                  Bayesian Projection Engine
                                  (Gamma-Poisson conjugate updating)
                                        │
                                        ▼
                                  Alert Generation (edge-based picks)
                                        │
                                        ▼
                              Discord Bot + FastAPI (:8001)
```

### Key Files

| File | Purpose |
|------|---------|
| `Scripts/live/live_poller.py` | Discovery + detail polling, state management, card backfill from events |
| `Scripts/live/live_engine.py` | Bayesian projection engine — Gamma-Poisson conjugate updating, pace decay, situational adjustments |
| `Scripts/live/live_odds.py` | Live odds ingestion from API-Football `/odds/live` endpoint with in-play bet-ID remap |
| `Scripts/live/live_alerts.py` | Actionable alert detection — model prob vs fair odds edge (≥0.04), cooldown 5 min |
| `Scripts/live/live_state.py` | Data classes (LiveMatchState, LiveSnapshot, MatchEvent) |
| `Scripts/live/live_store.py` | SQLite persistence layer (`Index/live.db`) |
| `Scripts/live/live_config.py` | Configuration constants |
| `Scripts/live/live_api.py` | FastAPI server (port 8001) — `/api/live/health`, `/api/live/fixtures`, snapshots |
| `Scripts/live/backtest_live.py` | Live engine backtesting harness |
| `Scripts/live_discord_bot/bot.py` | Separate Discord bot (requires `DISCORD_LIVE_BOT_TOKEN`) |

### Live Odds Pipeline
API-Football's `/odds/live` endpoint returns a different response shape from pre-match `/odds`:
- **Pre-match:** `row.bookmakers[].bets[].values[]` with `value: "Over 2.5"`
- **In-play:** `row.odds[].values[]` with `value: "Over"` + `handicap: "7.5"` (separate fields)

**Solution:** `_wrap_inplay_row_as_bookmaker()` in `live_odds.py`:
1. Remaps in-play bet IDs to pre-match equivalents (`_INPLAY_TO_PREMATCH_BET_ID`)
2. Splices `handicap` back into `value` string for Over/Under markets
3. Filters out suspended markets
4. Wraps in a synthetic "API-Football Live" bookmaker for the existing transformer

**In-play bet ID remap:**
| In-play | Pre-match | Market |
|---------|-----------|--------|
| 59 → 1 | Fulltime Result → Match Winner (h2h) |
| 25 → 5 | Match Goals → Goals Over/Under |
| 69 → 8 | Both Teams to Score → BTTS |
| 37 → 45 | Total Corners → Corners O/U |
| 119 → 80 | Total Cards → Cards O/U |
| 150 → 87 | Total shots on goal → SoT O/U |

### Card Stats Backfill
API-Football's `statistics` array lags `events` by 2-5 minutes. `_card_counts_from_events()` counts yellow/red cards per team from the events feed and applies `max(stats_count, events_count)` as the authoritative value. Prevents the engine from recommending Under lines based on stale zero counts.

### Live Discord Bot Commands
- `/livescores` — all currently live fixtures with scores
- `/live <match>` — full live analysis for one fixture
- `/liveprob <match> <market>` — live probability for a specific market
- `/livestats <match>` — raw live stats (corners, shots, cards, possession)

### Alert Generation
Fires when 10 ≤ elapsed ≤ 85 minutes:
- `_priced_total_candidate()` — model prob vs fair_prob edge ≥ 0.04
- `_priced_btts_candidate()` — BTTS odds/model mismatches
- `_priced_moneyline_candidate()` — moneyline value opportunities
- 5-minute cooldown per alert type per fixture

## Weights
All scoring weights centralized in `core/weights.py` (source of truth) and `SCORING_WEIGHTS` in `rag_cli_v2.py`. Key values:

| Sub-dict | Key params | Purpose |
|----------|-----------|---------|
| `combo` | `kb_quality_weight` **1.2**, `around_target_log_penalty` 12.0, `conservative_threshold` 1.25, `conservative_penalty` 24.0 | Parlay combination scoring |
| `pool` | `min_kb_quality` 0.0, `default_anchor` 1.55, `cap` 220, `beam_width` 60 | Candidate pool + quality gate |
| `ml` | `dynamic_blend` **True**, `skip_stats` **{"goals"}**, `elo_signal_weight` 0.15 | ML integration (cards 12.5%, SoT 20%, corners/goals 0%) |
| `referee` | `modifier_weight` 0.35, `min_sample_size` 8, `full_confidence_sample` 20 | Referee modifier |
| `projection` | `goals_own` 0.6/`opp` 0.4, `blend_goals_season` 0.85/`recent` 0.15, `xg_blend` 0.50 | Projection blending |
| `cards_share_model` | `match_anchor_weight` 0.55, `direct_stabilizer_weight` 0.45 | Team cards allocation |
| `corners_share` | `season_blend` 0.65, `recent_blend` 0.35, `shrink` 0.06, `floor` 0.28, `ceiling` 0.72 | Team corners share scoring |
| `corners_output` | `match_anchor_weight` 0.55, `direct_stabilizer_weight` 0.45 | Team corners allocation |
| `group_preference` | `corners` 0.72, `totals` 1.05, `sot` 0.95, others 1.0 | Market group preferences |
| `diversity` | `concentration_penalty` 3.5, `unique_group_bonus` 0.6, `max_same_group` 2 | Parlay diversity |

## Environment
- **APIs**: API-Football (`API-FOOTBALL-KEY`), OpenAI (`OPENAI_API_KEY`)
- **Vector DB**: Chroma (persistent local at `Index/chroma/`, or remote via `CHROMA_HTTP_HOST`)
- **LLM**: GPT (configurable via `RAG_CHAT_MODEL`, whale chat via `WHALE_CHAT_MODEL`)
- **Embedding**: `text-embedding-3-large`
- **Python**: 3.11+, dependencies in `requirements.txt`
- **Database**: SQLite at `Index/predictions.db` (shared between web app and Discord bot), `Index/live.db` (live engine)
- **Discord Bots**: Main bot (`DISCORD_TOKEN`), Live bot (`DISCORD_LIVE_BOT_TOKEN`)
- **Live channels**: `DISCORD_CHANNEL_LIVE_ALERTS`, `DISCORD_CHANNEL_LIVE_SCORES`

## Common Operations

```bash
# Re-normalize all leagues
cd Scripts/rag_ingest && python 00_normalize_to_docs.py --all

# Incremental embed (skip existing fixtures/players)
python 01_embed_and_upsert.py --all-leagues --skip-existing

# Full pipeline: pull + FE + normalize + embed + retrain
python Scripts/pull_season.py --season 2023 2024 2025 --normalize --embed --retrain-ml

# Single league
python Scripts/pull_season.py --season 2025 --league EPL

# Train ML models
cd Scripts/rag_ingest && python ml_edge.py --train

# Rebuild referee profiles + fixture map (all seasons)
python referee_data.py --build

# Backtest
python backtest.py --league EPL --season 2025 --skip-first 10

# Cards-specific backtest
python cards_backtest.py --league EPL --season 2025

# Run Discord bot
cd Scripts/discord_bot && python bot.py

# Run live betting bot
cd Scripts/live_discord_bot && python bot.py

# Run live API server
cd Scripts/live && uvicorn live_api:app --port 8001

# Run FastAPI web app
cd Scripts/web_app && uvicorn api:app --reload
```

## Index Directory Structure

| Path | Purpose |
|------|---------|
| `Index/chroma/` | Chroma vector DB (~18 GB) |
| `Index/ml_models/` | Ridge/LightGBM models + R2 + feature importance + training history |
| `Index/elo_ratings.json` | Elo ratings per team per stat (293 teams) |
| `Index/referee_profiles.json` | Referee card/foul profiles (~274 referees after dedup) |
| `Index/fixture_referee_map.json` | Fixture ID → referee name mapping (7,426 entries, 3 seasons) |
| `Index/predictions.db` | SQLite DB — users, subscriptions, referrals, predictions |
| `Index/live.db` | SQLite DB — live match states, snapshots, odds history |
| `Index/normalized_*.json` | Normalized docs per league/season (8 files, ~61k docs) |
| `Index/variable_dictionary.txt` | Data dictionary of all variables |

## Known Architecture Notes
- API-Football is the sole odds provider. The Odds API code (`odds_get`, `ODDS_API_BASE`, `LEAGUE_TO_ODDS_SPORT`) still exists as dead code but is never called
- `answer_once()` parses the target date from the query BEFORE fetching events, preventing empty results when asking about future dates
- When no fixtures exist for the queried date and no explicit date is given, a 7-day lookahead fetches the next available fixtures
- Card and SoT totals markets from API-Football typically appear 1-2 days before kickoff; queries made 3+ days out show model-only projections (no bookmaker odds)
- Dixon-Coles bivariate Poisson is the primary model for BTTS and correct score (rho=-0.10, not calibrated per league)
- `_PARLAY_MIN_ODDS = 1.30` — minimum odds per parlay leg
- Quality gate in `select_parlay()`: legs with `kb_leg_quality <= 0` are rejected (model disagrees with bet direction)
- `FORCE_RIDGE_STATS = {"corners"}` — corners forced to Ridge because LightGBM consistently overfits on this stat
- Cumulative tracker enrichment during ML training: `_TeamCumulativeTracker` builds running per-team averages, snapshots BEFORE each fixture, enriches rows with `*_pm` fields so v2 features have real values
- `projected_total_goals()` recent component uses `xg_for_avg` (not actual goals) for the recent blend — known inconsistency, documented but not yet fixed
- `fouls_per_90_team` and `fouls_avg` blended without unit conversion in `projected_total_cards()` foul-card estimate — known issue
- `margin_std = 1.25` hardcoded for spread cover probability — should be per-league/per-matchup but currently static
- Corners pre-match prediction has 0.03 correlation with next-match actual corners (verified). R2=0.12 is the honest ceiling with current features
- Live odds from API-Football `/odds/live` use different bet IDs and value shapes from pre-match `/odds`. Handled by `_wrap_inplay_row_as_bookmaker()` in `live_odds.py`
- Live stats (statistics array) lag events by 2-5 min. Card counts backfilled from events feed via `_card_counts_from_events()`. Corners/SoT/shots only available via lagging stats
- Auto-push parlay leg dedup: `used_leg_refs` accumulated across consecutive builds, injected as `excluded_legs` for subsequent requests. Same pattern in trial free picks
