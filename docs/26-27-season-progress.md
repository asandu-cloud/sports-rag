# 26/27 Season Progress

## Final vision

Build a trustworthy football betting product where the algorithm produces a
small number of clear, well-explained recommendations and users return for a
real matchday community—not just another stream of picks.
z
The product has two equal homes:
=
- **Discord is the clubhouse.** It provides the matchday experience, discussion,
  community identity, live conversation, and timely curated information.
- **The website is the product record.** It provides the complete match slate,
  evidence behind every decision, account value, saved information, and a
  transparent performance history.

The principle for every surface is: **cover every match, but do not force a bet
for every match.** A clear no-bet is useful information.

---

## Current state

### Completed foundation

- [x] Canonical pre-match pipeline for goals, corners, cards, shots on target,
  BTTS, moneyline, and spreads.
- [x] One structured `MarketResult` format with fixture, projection, price,
  recommendation/no-bet/unavailable status, confidence, evidence, and
  provenance.
- [x] Main Discord `/market` flow, scheduled market posts, daily top-pick
  selection, website fixtures, and website best-bets use that canonical path.
- [x] Regression coverage for normal recommendations, no-bets, missing odds,
  missing profiles, website delivery, and Discord card delivery.
- [x] Existing tests currently pass locally.

### Match Read foundation and controlled delivery

- [x] Compile all canonical results for one fixture into one deterministic
  Match Read: a thesis, one core selection, up to two compatible supporting
  selections, review-only alternatives, or a clear no-bet/unavailable state.
- [x] Preserve the same-game insight without inventing a combined price,
  probability, or expected value for correlated legs.
- [x] Persist immutable, versioned pre-match/confirmed-lineup Match Reads with
  canonical results, exact odds, model provenance, and later amendments.
- [x] Add an explicit after-delivery bridge that records a visible Match Read
  and links each displayed selection to the existing official tracking and
  settlement cohort; visible no-bets are auditable without becoming bets.
- [x] Generate a matchday slate with one fixture fetch/enrichment pass per
  league rather than seven separate market fetches; expose a review-only
  `match-reads` CLI and an opt-in `MATCH_READ_MODE=shadow` bot path.
- [x] Add a shared Match Read card contract used by the website and Discord
  hub renderers; review-only alternatives remain distinct from released picks.
- [x] Add a separate, explicit website release command plus a public-delivery
  gate: a persisted shadow row cannot leak to a public board, and a started or
  stale fixture cannot remain actionable.
- [x] Build the public website Match Read board/detail API and a five-fixture
  Best of Today cap; the normal page reads persisted cards only and never
  re-runs the model for a visitor.
- [x] Install the local macOS Match Read scheduler: it runs the one-shot
  worker every ten minutes in `website` mode, records durable run history,
  and publishes only fresh, pre-kickoff website cards. It remains a local-Mac
  bridge until the same command is moved to cloud scheduling.
- [x] Reliability steps 1–2: add stage timings and timeout diagnostics, batch
  historical opponent lookups, reuse the fixture quality audit, and invalidate
  derived profile caches after Chroma changes. Six preserved profile cases
  match exactly with 89.3% fewer collection reads; see
  [verification and operational limits](match-read-performance.md).
- [x] Implement reliability steps 3–4: fair scheduling and durable retry backoff,
  full-dispatch budget admission, early verified-lineup checks, revision-keyed
  statistical reuse, short incremental transactions and local refresh/worker
  coordination. See [implementation and limits](match-read-performance.md).
- [ ] Complete reliability steps 5–6: failure alerts and clear website
  availability states, followed by busy-matchday operational acceptance.
  A loaded scheduler alone does not establish that cards stay current.
- [x] Harden upstream refresh downloads (25 September): explicit network
  timeouts/three-attempt retry ceiling, validated API responses, fail-fast
  legacy team/player collectors, previous-fixture coverage protection and
  wall-clock stage deadlines with descendant cleanup. Interrupted refreshes
  retain the safety gate until a complete rerun succeeds.
- [x] Add a prolonged-refresh watchdog to existing scheduled gate checks:
  persistent local incident, health warning, deduplicated optional private
  Discord webhook and recovery notice. See
  [configuration](terminal-commands.md#refresh-timeout-and-blockage-protection-25-september-2026).
- [ ] Configure `REFRESH_ALERT_WEBHOOK_URL` with the owner's private operations
  webhook and verify remote delivery. Local alerting works without it; do not
  use public betting channels. Busy-matchday/card-availability acceptance
  remains separate from this blockage fix.
- [x] Recover the stalled 23 September refresh without deleting its gate or
  historical data: full additive rerun completed on 25 September at 15:49,
  document queue drained with zero errors/pending embeddings, and both
  measurement and website Match Read workers completed post-refresh cycles.
  No eligible live fixtures were present, so busy-matchday acceptance remains
  open. See [recovery evidence and remaining setup](worker-refresh-recovery-2026-09-25.md).
- [x] Investigate upcoming fixture coverage: live API checks confirmed a calendar
  gap beyond the old seven-day fetch window, not dropped imports. Added 392
  future fixtures through October 26; earliest supported fixture is October 9.
  Expanded schedule coverage to 35 days without widening the 96-hour card window.
- [x] Install separate local upstream schedules: six-hour fixture-only coverage
  refresh (hourly due checks), plus daily joined results/player/profile/embedding
  refresh at 07:15 local time. Existing ten-minute website worker is unchanged.
- [x] Verify schedule-to-generation-to-publication-to-HTTP-card-to-XI-amendment
  wiring in an isolated end-to-end test; verify installed jobs and a live empty
  cycle. No synthetic fixture or prediction was published to the real website.
- [ ] Verify real provider-backed cards and lineup amendments when fixtures enter
  the four-day window (earliest currently October 5 for October 9 fixtures).
- [x] Promote Championship, Süper Lig, Eredivisie, Primeira Liga, and Jupiler
  Pro League from shadow-only data support into the public Match Read path:
  their website filters, scheduled-card scope, Best of Today eligibility, and
  routine domestic refresh are now connected. The Best of Today cap remains
  five fixture cards.
- [x] Build an opt-in `MATCH_READ_MODE=hub` Discord path: one edited hub per
  league/day, rather than loose market-channel posts. It records only cards
  Discord accepted and retries incomplete tracking writes.
- [x] Prevent the website's generic slip from constructing a made-up same-game
  price: it accepts one selection per fixture until a genuine bookmaker SGP
  quote is available.
- [ ] Run the controlled public-delivery acceptance check on a real matchday
  before enabling `MATCH_READ_MODE=hub` as the normal daily-picks path.
- [ ] Implement the agreed amended-card reporting views: headline ROI retains
  every as-published exposure, while a separate final-card report will show
  the last pre-kickoff card per fixture. Earlier releases must never be
  rewritten or hidden from the audit trail.
- [x] Wire a verified current-season lineup source into the confirmed-lineups
  dispatcher. It uses each canonical API-Football fixture ID directly and
  requires two matching 11-player XIs; in explicit hub mode the bot then edits
  the existing league hub and labels the card as updated.

### Important limits still present

- [ ] Current-season data, odds, and profiles have not yet been refreshed and
  verified for operating use.
- [ ] The product is not deployed as a managed cloud service.
- [ ] Published standalone picks are not yet all recorded and settled through
  one automatic tracking path.
- [ ] Several specialist Discord and live paths still use legacy rendering or
  RAG behaviour.
- [ ] The current Discord information architecture and onboarding are too
  complicated for a new user.

---

## Priority 1 — Season-ready data and trustworthy picks

### Data operations

- [ ] Choose the authoritative providers for fixtures, odds, lineups, results,
  and historical odds snapshots.
- [ ] Configure and securely store production API credentials.
- [ ] Run the full data refresh for every supported domestic and European
  competition.
- [ ] Verify team aliases, promoted/relegated teams, competition mappings,
  kickoff times, and market coverage.
- [ ] Establish scheduled ingestion for fixtures, odds, lineups, referee data,
  results, and settlement data.
- [ ] Add freshness checks and alerts for missing fixtures, stale odds, failed
  jobs, and empty market coverage.
- [ ] Keep an auditable snapshot of every input used for a published pick.

### Accuracy and tracking

- [x] Recover verified local 2025 canonical history for the original five
  leagues and UCL/UEL/UECL without replacing existing records; replay the new
  5,083-fixture candidate export (2026-09-22). This is not model promotion.
  See [coverage and remaining Phase 1 gates](phase1-history-recovery-2026-09-22.md):
  ambiguous card/zero statistics and full numerical baseline replay remain open.
- [x] Complete the small Phase 1 historical diagnostic (2026-09-22): audit 5,083
  local FT fixtures and spot-check 52 against API-Football, within 120 attempts.
  Card-label selection bias is confirmed; no production data or models changed.
  See [findings and Phase 2 handoff](phase1-history-diagnostic-2026-09-22.md).
  Phase 2 tracking/settlement is next, retaining pending states for missing data
  and unresolved card-market definitions. Full modelling sign-off remains open.
- [x] Automatically record each selected canonical recommendation when its
  public Match Read is released on Discord or the website; previews and shadow
  candidates do not enter the official published cohort.
  Hardened 2026-09-23: atomic Match Read publication/linking, cross-surface dedup,
  prospective system identity, and initial-versus-amendment reporting. No
  historical grades or versions were invented. See the
  [Phase 2 tracking implementation](phase2-publication-tracking-2026-09-23.md).
- [ ] Settle those records from final fixture results and show hit rate, ROI,
  sample size, and market/league breakdowns.
  Implemented 2026-09-24: fixture-ID-based resolution, conservative outcome grading,
  pending reasons and correction evidence. No live rows were graded. Independent
  scheduling, verified bookmaker policies and provider-backed acceptance remain
  open; see [safe settlement](phase2-safe-settlement-2026-09-24.md).
  Later 2026-09-24: independent prospective measurement scheduler installed,
  closing capture implemented, and 8 provider fixtures checked read-only using
  16 requests. The 35 sampled old publications remain pending for missing market
  periods. Applicable bookmaker policy approval, historical recovery and future
  matchday acceptance remain open. See
  [measurement automation](phase2-measurement-automation-2026-09-24.md).
  Remaining local work completed later that day: full 467-publication recovery
  audit, guarded offline plan/apply, website/Discord reporting fixes and isolated
  end-to-end acceptance. The 461 overdue old selections lack periods, so they
  remain pending; no ROI is fabricated. Live prospective validation and
  bookmaker/site policy approval still block full sign-off. See
  [local completion report](phase2-local-completion-2026-09-24.md).
  Latest 2026-09-24: the owner approved product-rule measurement instead of
  requiring bookmaker-specific settlement. New publications snapshot the
  participation-card policy (minutes > 0, weights 1/2/3) and abandonment-as-void.
  Implemented with conservative missing-data gates and labelled API/Discord
  reporting. No historical grades or prediction formulas changed. Live acceptance
  and card-model target alignment remain open; see
  [participation policy](participation-settlement-policy.md).
- [ ] Add archived historical fixture/odds/profile recordings to supplement the
  current synthetic regression scenarios.
- [x] Add a chronological recorded-decision walk-forward report with ROI,
  calibration, sample-size and pre-match-timestamp checks. It must still be
  run on accumulated shadow/live records before changing weights or thresholds.
- [ ] Review calibration by probability bucket; do not optimise only on hit
  rate or a small ROI sample.
- [x] Define a release policy for intentional model changes: baseline revision,
  rationale, backtest result, and rollback path.

### Early-season confidence and lineup guardrails

- [x] Audit every market's pre-match inputs for sample size, profile season,
  variance, odds freshness, and as-of timestamp; persist those facts with the
  prediction.
- [x] Blend sparse current-season team data with time-safe prior-season and
  league priors rather than treating one or two new-season fixtures as a full
  profile.
- [x] Supply a defensible variance estimate when a team profile has too little
  data; do not fall back to an over-confident distribution because variance is
  missing.
- [x] Correctly price Asian half- and quarter-line outcomes in probability and
  expected-value calculations.
- [x] Make recommendation confidence depend on data quality and calibration as
  well as model probability and apparent edge.
- [x] Introduce per-market eligibility gates: suppress a recommendation when
  data is too sparse, stale, incomplete, or uncalibrated instead of merely
  presenting it as a weak pick.
- [x] Align European fixture projections and eligibility audits around one
  domestic-anchored profile context. Early UCL/UEL/UECL cards now use covered
  domestic current/prior-season form as their primary evidence, retain the
  continental sample and source weights in the audit trail, and are capped at
  low confidence until that competition has enough current-season matches.
  Teams without a supported domestic or European profile remain unavailable or
  suppressed rather than receiving an invented fallback.
- [x] Prevent temporal leakage by ensuring every historical prediction uses
  only data available before that fixture's kickoff.
- [x] Implement chronological walk-forward and reliability reporting for
  captured canonical decisions; compare the revised selector with any recorded
  baseline source before changing thresholds.
- [ ] Accumulate and review a sufficient shadow sample before changing release
  mode or relying on the revised selector for public picks.
- [x] Implement a shadow-mode release switch and documented rollback path that
  prevents candidate picks from reaching Discord or the website.

### Lean manager and formation context (after guardrails)

- [ ] Record the active manager, start date, and source confidence for each
  team; flag recent manager changes as an uncertainty factor.
- [ ] Maintain manager formation frequency and recency from historical fixture
  lineups, with an explicit sample count rather than a claimed tactical score.
- [ ] Use confirmed starting XIs and formations when they become available
  shortly before kick-off to rerun affected predictions against fresh odds.
- [ ] Support an expected-lineup state before confirmation and a confirmed-
  lineup state afterwards; make the distinction visible to users.
- [ ] Backtest whether manager/formation context materially improves
  calibration before using it to adjust probabilities.
- [ ] Defer detailed manager-style ratings and player-by-formation performance
  adjustments until sufficient data demonstrates an improvement.

**Launch standard:** a user can see exactly what was recommended, at which
price, why, and how the same type of recommendation has performed.

---

## Priority 2 — Website: the clear matchday product

### Matchday experience

- [ ] Build the main matchday page around a simple question: “What are today’s
  best bets?”
- [ ] Show a top section capped to the strongest qualified bets across the
  slate; avoid presenting every model output as a bet.
- [ ] Show every fixture below it with one primary qualified recommendation or
  an explicit no-bet state.
- [ ] Provide a clean fixture-detail view: available markets, model
  probabilities, odds, edge, confidence, evidence, lineup/referee status, and
  kickoff time.
- [ ] Clearly separate **recommended**, **no bet**, and **data unavailable**.
- [ ] Make dates, competitions, odds source/capture time, and update state
  obvious to the user.

### Product value and account experience

- [ ] Finish authentication, account pages, tiers, subscription lifecycle, and
  billing flows with real production configuration.
- [ ] Add a transparent public/member track-record view.
- [ ] Add a member’s saved fixtures/picks or matchday watchlist only if it
  serves a clear user need; do not build a feature for its own sake.
- [ ] Make website links first-class targets from Discord matchday posts and
  fixture discussions.
- [ ] Complete responsive/mobile design before launch; most matchday use will
  be on phones.

### Website acceptance checklist

- [ ] A new user can understand today’s best bets within one screen.
- [ ] A user can inspect any fixture without needing Discord context.
- [ ] A no-bet is understandable and does not look like a broken card.
- [ ] The site never displays stale data without saying when it was updated.
- [ ] Website recommendations exactly match the corresponding Discord pick.

---

## Priority 3 — Discord: community without clutter

### Target server structure

Keep permanent channels few and purpose-led. Avoid one channel per market.

- [ ] `#start-here` — concise explanation, membership/help links, and one
  “how it works” path.
- [ ] `#matchday-hub` — one curated matchday post, not a stream of every
  market output.
- [ ] `#match-chat` — fixture discussion threads and general pre-match talk.
- [ ] `#live-lounge` — live conversation plus only meaningful live alerts.
- [ ] `#community-picks` — member reasoning and shared bet slips in a clear,
  moderated format.
- [ ] `#results-and-review` — settled model picks, lessons, and transparent
  track record.
- [ ] Small support/announcement space; add tier-specific rooms only when they
  provide genuine social value.

### Matchday publishing design

- [ ] Publish one matchday hub message per slate: strongest picks, key fixture
  links, and a simple explanation of confidence/no-bet states.
- [ ] Use fixture threads or buttons for discussion instead of creating market
  channels or posting repeated messages.
- [ ] Limit public bot posts to high-signal information: the daily slate,
  material lineup changes, genuinely actionable live alerts, and results.
- [ ] Keep complete market details available on demand via the website or
  private/on-demand bot commands.
- [ ] Define a predictable publishing rhythm: matchday preview, lineup update
  where warranted, live only when meaningful, and post-match review.
- [ ] Prevent duplicate posts and make amendments visible rather than silently
  replacing a published recommendation.

### Community and retention

- [ ] Design fixture threads so conversation starts easily: a short model read,
  key question, and link to evidence—not a wall of numbers.
- [ ] Give members a safe, consistent way to share their own reasoning without
  encouraging indiscriminate betting or spam.
- [ ] Establish moderation rules, spoiler/live-score etiquette, and a clear
  responsible-betting standard.
- [ ] Define moderator tools and escalation rules before opening the community
  widely.
- [ ] Create a weekly review/discussion rhythm so community value continues on
  non-matchdays.

---

## Priority 4 — Onboarding and product understanding

The existing problem is not merely too many commands; it is too much to learn
before a user receives value.

### New-user journey

- [ ] Reduce onboarding to one short, guided path: **what this is → where to
  find today’s picks → how to read a pick → where to discuss it**.
- [ ] Make the default first action a matchday view or curated Discord hub, not
  a long command list.
- [ ] Explain the three decision states in plain language:
  - **Recommended:** a model projection and an available price both qualify.
  - **No bet:** the fixture was assessed, but the current price is not good
    enough.
  - **Unavailable:** information required for a safe decision is missing.
- [ ] Show one real, annotated example pick instead of a dense manual.
- [ ] Clearly distinguish the main experience from optional advanced tools
  such as player props, parlays, and detailed market scans.
- [ ] Add a first-week sequence: welcome, first matchday guide, how to join a
  fixture discussion, and where to see settled results.

### Bot design

- [ ] Make the core commands few and obvious; hide advanced commands behind
  menus, help, or appropriate tiers.
- [ ] Send exploratory/detail-heavy answers privately or ephemerally where
  possible.
- [ ] Give every bot output a next action: discuss, view evidence, save/watch,
  or see results.
- [ ] Remove or deprecate commands/features that duplicate the website or
  create routine noise.

### Onboarding acceptance checklist

- [ ] A new member can explain the product and find a useful pick in under five
  minutes without staff help.
- [ ] A user understands why a no-bet is not a product failure.
- [ ] A member can join a relevant fixture discussion in one action.
- [ ] The channel list is understandable without reading an external guide.

---

## Priority 5 — Production infrastructure and low-maintenance operations

- [ ] Choose cloud hosting for the website/API, Discord worker(s), scheduled
  ingestion jobs, and any live worker.
- [ ] Containerise or otherwise define reproducible application runtime(s).
- [ ] Replace local production persistence with a managed database; decide how
  the large vector/index data is stored, refreshed, backed up, and restored.
- [ ] Implement environment-specific configuration and secret management.
- [ ] Add CI checks: tests, formatting/static checks where appropriate, and
  deployment build validation.
- [ ] Add production logs, error reporting, health checks, uptime alerts, job
  failure alerts, and basic dashboards.
- [ ] Define backups, retention, migration, rollback, and incident procedures.
- [ ] Load-test the matchday workload: concurrent Discord commands, scheduled
  posts, API reads, and provider-rate-limit failures.
- [ ] Add a staging environment with test Discord channels before any public
  deploy.

**Launch standard:** a provider or bot failure is visible quickly, recoverable
without editing production by hand, and does not silently publish stale picks.

---

## Priority 6 — Remaining pipeline migration and cleanup

- [ ] Create structured contracts for per-team corners/cards and migrate
  `/teamlines` plus scheduled team-line posts.
- [ ] Create structured contracts for correct score and goal intervals, or
  explicitly keep them as non-recommendation analytics features.
- [ ] Route parlay candidate generation and evidence through canonical market
  results.
- [ ] Feed live polling with an explicit canonical pre-match snapshot while
  keeping live-state calculations separate.
- [ ] Review recommendation-facing Whale/chat and legacy commands; route them
  through the canonical selector or make them analytics-only.
- [ ] Remove dormant duplicate logic from `rag_cli_v2.py` only after the
  replacement path has been deployed and observed safely.

---

## Launch sequence

Do not launch broad paid acquisition before these are true:

1. [ ] Fresh data and odds flow successfully for a representative matchweek.
2. [ ] Website matchday and fixture experiences are clear on mobile.
3. [ ] Discord has the new small, conversation-first structure and publishing
   rhythm.
4. [ ] Onboarding has been tested with people who did not build the product.
5. [ ] Every published canonical pick is tracked and settles correctly.
6. [ ] Billing/authentication and support flows work in production mode.
7. [ ] Cloud deployment, monitoring, backups, and alerts are proven in a
   staging run.
8. [ ] A closed beta has run through real matchdays, with clutter, confusion,
   and failure points addressed.

## Decision log

Record major choices here as they are made.

| Date | Decision | Why | Owner / follow-up |
|---|---|---|---|
| 2026-08 | Discord remains a primary community product, not merely a distribution channel. | Long-term retention should come from useful matchday conversation alongside trustworthy information. | Redesign server and onboarding before broad launch. |
| 2026-08 | Website becomes the durable source of match detail and performance history. | It reduces Discord clutter while preserving full evidence and transparency. | Build matchday and fixture views first. |
| 2026-08 | A no-bet is a valid visible outcome. | Avoids forcing low-quality recommendations merely to fill a slate. | Carry through website, Discord, onboarding, and tracking. |
| 2026-08 | Start with lean manager and formation context, not a full manager-profile model. | Confirmed lineups and recent formation usage are actionable; detailed tactical/player-fit ratings need more data and backtesting. | Implement after early-season confidence guardrails are validated. |
| 2026-09 | Start Match Reads in shadow-only mode. | The compiler/persistence path can be reviewed against real fixtures without adding Discord noise or contaminating the public tracked cohort. | Review one or two matchweeks; then decide public website/Discord delivery policy. |
| 2026-09 | Use a public Match Read as a fixture-level card, not a synthetic same-game parlay. | Multiple compatible angles can be useful, but their odds/probability must not be multiplied without a real bookmaker SGP quote. | Keep generic slips to one selection per fixture; add SGP support only with real quotes. |
| 2026-09 | Public Match Read cards are allowed only while fresh and pre-kickoff. | A persisted/delivered record is historical evidence, not perpetual permission to show an actionable price. | Enforce the same release gate on website promotion, website reads, and Discord hubs. |
