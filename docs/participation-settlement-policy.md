# Spix participation-based settlement policy

**30 September 2026 status:** production remains on the v1 rule below. The owner
approved a staged [v2 research definition](phase4-card-policy-v2-results-2026-09-30.md)
using recorded minutes **>= 2**. It will be promoted together with the new engine
and aligned profiles, not activated independently. Existing publication policies
and grades remain tied to their original version. Adding an optional parser
minimum does not switch the current production callers or policy snapshots.

Approved by the project owner on 24 September 2026. Implemented as
`spix-participation-settlement.v1` for **new canonical publications only**.

This is an explicitly approximate product measurement convention, not a claim
that every bookmaker would settle the same selection identically. The recorded
bookmaker, market, period, odds and original prediction remain unchanged.

## Rules

- Regulation time plus stoppage time only. Extra time and penalties are excluded.
- Weighted total cards: yellow = 1, straight red = 2, yellow then straight red =
  3, two yellows and the resulting dismissal = **3 total**, not 4.
- Eligibility is **recorded minutes > 0**, not proof the player was on the pitch
  when booked. Recorded zero-minute players are excluded, including players whose
  very short appearance the provider rounded to zero. Staff are not player rows.
- Missing/invalid minutes for a carded player leave the outcome pending. A row
  with explicit zero yellow AND zero red cards cannot change the card total,
  even if minutes are unknown; this does not convert unknown minutes into zero.
- Missing counts for an eligible/possibly eligible player remain unknown. Two
  yellows without the corresponding recorded red are treated as inconsistent,
  not repaired by inference. Nulls, duplicated player IDs, mismatched teams and
  partial rosters cannot be silently counted as zero.
- Both exact teams must be present, each with at least 11 distinct player rows
  and at least seven recorded positive-minute participants. These checks reject
  obvious partial/placeholder responses, not certify universal feed completeness.
- Existing Asian-line full/half win/loss/push arithmetic is unchanged.
- Confirmed `ABD` (abandoned) is void under our policy. `PST`, `CANC`, `SUSP`,
  `AWD` and `WO` are **not** silently treated as abandonment or automatic refunds.
- Unknown evidence stays pending and outside graded hit-rate/ROI denominators.

This weighted rule applies only to `totals_cards_over_under`. Yellow-only
(`totals_yellow_cards`) and other card products are **not reinterpreted** using
weighted totals. Their automatic grading still requires a separate definition.

## Data and known limitations

The worker fetches `/fixtures/players` once per due fixture that has an eligible
weighted-card recommendation. It uses player IDs, `games.minutes`, `cards.yellow`
and `cards.red`, not team-level `yellow + red` and not season-level totals.
See the provider's [fixture-player endpoint description](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).

Only completed `FT` fixtures are eligible for player-total card settlement.
`AET`/`PEN` player totals are not period-segmented, so they remain pending rather
than including extra time. Goal markets can still use the saved regulation score.
The compact source evidence and each player's inclusion/exclusion reason are
retained with the settlement; private player evidence is not returned by the
public tracking API.

There is deliberately **no events/substitution/lineup timeline reconstruction**.
Cards shown before coming on or after being substituted may therefore count if
the player eventually recorded minutes. FT player totals are used as the
provider's regulation-match totals: they do not independently establish whether
the provider included a card shown after the final whistle. This is an additional
limit of aggregate evidence, not verified post-whistle exclusion. Compare a real
sample with manual results before claiming bookmaker-equivalent accuracy.

## Wiring and safety

- `PublicationService` snapshots the full policy in the newly created prediction's
  extras. Retries and cross-surface delivery of an existing recommendation do not
  add, replace or upgrade its policy. No schema migration or historical backfill.
- The canonical settlement service and the independent scheduled measurement
  worker use that recorded policy. Old rows without it retain their prior gates.
- The player endpoint shares the existing 20-attempt/cycle and 500-attempt/UTC-day
  measurement limits, timeout and retry schedule. Typically this adds one player
  request per card-bearing fixture, not per player or selection. No event or
  lineup requests are added; no full-season download is triggered.
- A player-data failure holds the card bet, not goal/corner/SoT bets with adequate
  evidence. Incomplete evidence retries with the existing backoff. Abandoned
  fixtures need no player download to void under the saved policy.
- Settlement stores policy version, basis, evidence and per-player contributions.
  A concurrent policy/selection change prevents a stale settlement write.
- Shared tracking summaries/daily breakdowns and the website API expose
  `product_rule_settlements`, `participation_card_settlements` and
  `settlement_policy_note`. Discord's shared summary includes the same note.
  Recent API entries expose policy/basis but no private evidence or raw responses.

## What this does not do

This changes **measurement/settlement**, not model formulas, probabilities,
selection thresholds or old training labels. Existing card projections based on
other definitions have not been retrained or validated against this weighted
target. That alignment is still required before claiming calibrated weighted-card
predictions. Legacy/custom-database and separate unit-slip grading are unchanged.

No paid provider calls, production settlements, historical edits or public test
posts were run to implement this. Tests use isolated stores and mocked transport.
Live prospective acceptance remains open; this does not complete all Phase 2 gates.

Verification on 24 September 2026:

- **281 tests passed** across the focused settlement, publication, reporting,
  measurement-worker, Match Read, Asian-line and scheduler regression suite.
  Compile checks and `git diff --check` also passed.
- Read-only adapter checks on existing normalized player rows produced totals
  for fixtures `1570356`, `1570368`, `1570333` and `1570341`. Fixtures `1550091`
  and `1047460` correctly remained pending for missing carded-player minutes.
  This was a local adapter check, not a fresh provider test or bookmaker comparison.
- The before/after SHA-256 over prediction IDs, outcomes, actual values, grading
  timestamps and extras was unchanged:
  `7e4b510e2b8297d3b7aa8ede68b9ba678188c21dfee6f8b35999d905b58afb6d`.
- The read-only acceptance audit still found 467 historical publications missing
  their saved market period and zero prospective publications after activation.
  Reporting and visible-selection links passed; live acceptance remains pending.
  Artifact: `Index/prediction_experiments/phase2-participation-policy-2026-09-24-audit.json`.

The already installed one-shot measurement job loads fresh code each invocation;
it will use the rule for newly recorded eligible publications once the refresh
gate allows it. Restart long-running website/bot/publication workers to load the
new publication snapshot code. No reinstall, database rebuild or regrade command
is required. Do not remove refresh locks or change publication mode for this.
