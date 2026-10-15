# Spix player-prop engine — technical plan

Plan date: **26 September 2026**.

Status: **future work; documentation only**. The owner has asked to finish the
current team-market Phase 3 work before building the player engine. Creating
this document does not authorize implementation, imports, provider collection,
dependency installation, training, changes to delivery modes or promotion.
The choices labelled “proposed” below require review when this work starts.

This plan extends the existing player code and shared prediction infrastructure.
It does not replace the [Phase 3 technical plan](phase3-technical-plan-2026-09-26.md)
or change its [completed Batch A dataset](phase3-dataset-2026-09-26.md). Team
benchmarks proceed first. Player work can subsequently reuse the numerical,
calibration, selection and measurement infrastructure as it becomes available;
its own qualification gates still apply.

## 1. Objective and boundaries

Build reproducible pre-match player forecasts and, where evidence supports
them, price-aware player-prop recommendations. A reliable engine must:

- Forecast participation, playing time and the distribution of a player's
  statistical production using information available at the forecast cutoff.
- Produce calibrated probabilities for precisely defined selections, with
  separate estimates of uncertainty and betting value.
- Return no-bet or unavailable when data, role, quote or settlement evidence
  is inadequate; coverage is not a requirement to recommend every player.
- Preserve enough evidence to replay the decision, identify the exact quote,
  capture an equivalent closing price and settle the actual selection.
- Demonstrate improvement or acceptable performance against simple baselines
  on chronological tests and subsequent real shadow observations.

Predictive qualification and betting-policy qualification are separate. A
useful statistics engine may qualify while price-based recommendations remain
unavailable. Better count accuracy alone does not establish profitable bets.
Retaining a simpler model, restricting scope or declining activation are valid
outcomes. No sample size or number of elapsed matchdays guarantees readiness.

API-Football remains the intended data and odds provider. Do not reintroduce
The Odds API. Numerical training and inference use structured data; OpenAI
embeddings are not a prerequisite. An optional editorial model may explain
saved numerical results after selection, but must not choose bets, invent
statistics or supply probabilities when the numerical service fails.

Out of initial scope: in-play props, first/last scorer sequencing, fantasy
scoring, a full manager/formation model, tracking-data models, same-game joint
pricing and simultaneous deployment across all 13 competitions. Richer player
features for team predictions are a separate later ablation, not an unmeasured
change to the frozen team benchmark.

## 2. Current foundation and evidence limits

The following reflects read-only repository inspection on 26 September. Counts
are diagnostic observations, not an immutable player-dataset manifest. Recheck
them when work starts. Existing local tests were inspected, not rerun as part
of this documentation task.

| Area | What exists | What remains unqualified |
|---|---|---|
| Player projections | Per-90 production, heuristic minutes/role adjustments, opponent context, probabilities and evidence in `player_projections.py`. | Historical accuracy, calibration, a learned participation/minutes distribution and exact quote-based value. |
| Player storage | Canonical nullable fixture-player statistics and derived player feature snapshots. | Complete historical rosters, target semantics, original availability and uniform coverage. |
| Product paths | Player commands and scheduled Discord paths; profile and lineup helpers. | A canonical, persisted, measured player lifecycle shared with the website. |
| Canonical markets | `MarketResult` has a generic participant field; Match Reads, delivery identity and amendments exist. | Typed player identity, participant-aware compatibility and complete quote provenance through every layer. |
| Measurement | Publication, closing-price and settlement services for supported match markets. | Closing-price matching and settlement currently reject participant selections. |
| Verification | Local player heuristic tests and a small inspection of saved provider payloads. | Representative provider validation and actual published player-prop lifecycle acceptance. |

### 2.1 Canonical data observed

- 251,474 fixture-player rows across 6,133 completed FT fixtures and 9,844
  distinct players represented in those rows. These are not 251,474 independent
  matches or equally usable training examples.
- 188,931 rows have positive recorded minutes; 60,458 have unknown minutes and
  2,085 have zero minutes. Row presence alone does not establish participation.
- Both teams have some player rows in each of the 6,133 fixtures. This does not
  certify roster completeness or every statistic.
- Saved player-response archives cover 352 distinct fixtures in 367 responses.
  Most normalized rows therefore lack a corresponding archived player response
  in the inspected archive inventory.
- The eight-season history collector explicitly used `--no-players`. The
  [history import](eight-season-history-import-2026-09-26.md) preserved existing
  player history; it did not generate eight seasons of player statistics.

Player fixture coverage in the canonical database, by provider season:

| Competition | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|
| EPL | 0 | 0 | 0 | 50 |
| LaLiga | 0 | 0 | 0 | 69 |
| SerieA | 0 | 0 | 0 | 50 |
| Bundesliga | 0 | 0 | 0 | 36 |
| Ligue1 | 0 | 0 | 0 | 45 |
| Championship | 557 | 556 | 556 | 95 |
| SuperLig | 380 | 342 | 306 | 54 |
| Eredivisie | 320 | 317 | 308 | 63 |
| PrimeiraLiga | 308 | 308 | 308 | 62 |
| BelgianProLeague | 310 | 315 | 319 | 63 |
| UCL | 0 | 0 | 0 | 18 |
| UEL | 0 | 0 | 0 | 18 |
| UECL | 0 | 0 | 0 | 0 |

There are no canonical player rows for 2019–2022. This is not a claim that all
other player history is absent: a filename inventory found 32 fixture-player
JSON exports under `Output/`, covering the original five leagues and UCL/UEL/
UECL for 2023–2026. Their completeness and import eligibility remain unchecked.
Provider years identify seasons, not necessarily calendar-year matches.

### 2.2 Nulls, zeros and legacy processing

Among 169,385 positive-minute appearances in canonical seasons 2023–2025:

| Statistic | Known values | Explicit zeros | Unknown values |
|---|---:|---:|---:|
| Goals | 13,454 | 0 | 155,931 |
| Total shots | 65,107 | 0 | 104,278 |
| Shots on target | 37,016 | 0 | 132,369 |

Every known value in these columns is positive. Dropping null labels would
create a selected positive-only sample; replacing every null with zero would
assert facts the current evidence does not establish. Three checksum-verified
local provider responses illustrated differing behavior: EPL/Ligue1 examples
included explicit zeros, while a SerieA example included many nulls and positive
values. That tiny inspection does not establish a universal provider rule.

Existing code requires review before reuse as a training source:

- The canonical upsert preserves nullable statistics. Some legacy projection
  helpers instead convert missing recent counts into zero.
- The legacy EPL fixture-player exporter stores a display pairing such as
  `Arsenal vs Chelsea` instead of the provider fixture ID and drops appearances
  of one minute or less. Such files cannot establish the full player universe
  or nonparticipation denominator.
- Legacy profile code uses a 60-minute threshold as a proxy for starting,
  divides starts by appearances, sorts fixture display strings for recent
  context and can select a player's most frequent team across a sample.
- A season collector sums minutes across transfer-related blocks while taking
  some event counts from one representative block. Event/exposure denominators
  must come from the same observations.
- Current RAG player retrieval filters short appearances and sorts retrieved
  records by minutes; it is not a chronological, cutoff-safe feature builder.
- Some stored snapshots contain imported season aggregates. An `as_of` field
  does not prove the aggregate existed at that historical forecast time.

The new engine must rebuild qualified historical features from earlier
observations. Preserve legacy profiles as evidence of current behavior; do not
silently reinterpret them as historical ground truth or rewrite them now.

## 3. Proposed initial scope

Choose competitions from the data and quote audit, not prestige or an assumed
provider coverage flag. Report coverage for all 13 competitions while allowing
only separately qualified market/competition/stage combinations to progress.

| Order | Market | Entry conditions |
|---|---|---|
| Pilot | Player total shots and shots on target; initially confirmed starters. | Valid zero/positive labels, certified regulation scope, historical support, exact participant identity and usable prices for betting claims. |
| Next | Player goals and anytime scorer. | Qualified goal/own-goal semantics, sufficient event support, playing-time uncertainty and documented participation rules. |
| Later | Assists, fouls, tackles, passes and player bookings. | A separate target, provider coverage, market equivalence, price and settlement review for each. |

Prefer a small number of supported competitions for the initial public pilot.
The research dataset can be broader if every contribution is qualified and
league effects are represented. A competition with statistics but no usable
player quotes can support numerical research without actionable props.

Starting with confirmed starters reduces one source of uncertainty; it does
not remove substitution risk, late withdrawals or the need for a minutes
distribution. Preliminary forecasts can be an explicitly separate shadow
cohort until they qualify. Player bookings are not automatically disqualified
forever by the team-card problem, but they need their own verified labels and
rules. The approved weighted team-card policy is not a player-booking policy.

## 4. Data audit, reconciliation and collection

### 4.1 Local inventory first

Use a consistent read-only database snapshot and hash relevant source files.
Inventory canonical rows, raw player/lineup/event archives, the older `Output/`
exports and existing derived profiles separately. Produce coverage by market,
competition, season, fixture status, role and source, including null/zero rates,
short appearances, transfers, duplicate identities and conflicting revisions.

Build a reviewed mapping from legacy records to exact provider fixture,
player and team IDs. A display name or team pairing is supporting evidence,
not an identity key. Repeated pairings, ambiguous dates, aliases and multiple
competition matches must remain unresolved unless independently disambiguated.
Preserve a mapping ledger with source hashes, evidence, rule version and
review status. Do not choose the nearest-looking fixture or player name.

Validate regulation-period evidence independently of row presence and `FT`.
The team Batch A report already records missing period evidence for 5,781
fixtures; player rows on those fixtures do not repair that gap by themselves.
Check player-stat scope separately, including provider corrections and
possible extra-time aggregates.

### 4.2 Zero and participation evidence

For each target retain the original value and one of:

`explicit_zero`, `explicit_positive`, `explicit_null`, `field_omitted`,
`response_unavailable`, `verified_derived`, `conflicting`.

A derived zero requires a versioned reconstruction rule and evidence of
completeness appropriate to that statistic. Missing a goal event is not proof
of zero without a complete, correctly attributed event record. A team total
alone is generally insufficient to allocate player statistics. Never distribute
an unexplained remainder among players or infer all unknown players had zero.
Re-downloading may reproduce the same nulls; quantify that before scaling up.

Distinguish confirmed start, listed substitute, confirmed appearance,
confirmed nonappearance, absent and unknown. A missing row or null minutes is
not automatically DNP. Rounded zero minutes can include a very short appearance.
Minutes above 90 may reflect stoppage-time accounting; do not classify them as
extra time or clamp them to 90 without the applicable provider convention.

Maintain separate eligibility for historical context, count targets, minutes
targets, participation targets and settlement. A row may qualify for one use
and fail another. Labels remain unknown when their evidence fails; never impute
targets. Market-specific rates use only matched, known count/exposure pairs.

### 4.3 Bounded provider verification, then optional collection

API-Football documents fixture-player statistics, lineup endpoints and
season-specific coverage. A positive coverage flag is not a guarantee for each
fixture. Its pre-match odds guide describes limited historical retention, so
future quote capture must be planned separately from historical statistics.
Recheck current capabilities and the account plan before collection.
[Provider reference](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).

After the local audit, propose a separately approved provider sample with:

- Exact fixture IDs, endpoints, competition/season strata and a request ceiling
  counting retries and pagination, plus an estimated duration/storage budget.
- Random controls alongside targeted null/zero, transfer, short-appearance,
  late-substitution and period cases. Do not sample only positive events.
- Player statistics, lineup and event requests only where each is necessary
  to answer a declared evidence question; no implicit all-season enrichment.
- Raw request parameters, response bodies, provider/capture timestamps,
  checksums, coverage flags and a resumable success/failure ledger.

Only then define a larger missing-data collection if justified. Estimate
requests from missing fixture/endpoint combinations and verified batching
support, not from number of players. Eight seasons of team data does not imply
that eight seasons of every player field are necessary or obtainable.

Use isolated staging, bounded timeouts/retries, quotas and fail-fast validation.
Validate archives and normalized coverage before any canonical import. A later
approved additive import needs a consistent backup, the normal writer gate,
an explicit conflict plan, transactional application and before/after row
fingerprints. Preserve revisions; a newer null must not silently erase an
earlier evidenced value. Do not delete refresh locks or incomplete markers.

## 5. Proposed versioned contracts

Names below describe intended contracts, not implemented APIs or migrations.
Review the existing models and migrations again before deciding storage layout.

| Contract | Required content |
|---|---|
| `player-evidence.v1` | Provider fixture/player/team/competition IDs, provider season, original and revised kickoff, match status/period, raw value/state, source URI/hash, parser version, provider/capture/availability timestamps and correction lineage. |
| `player-eligibility.v1` | Per-use and per-market decisions, history support, missingness, period and identity checks, deterministic primary and overlapping exclusion reasons. |
| `player-features.v1` | Ordered numerical schema, missingness, player/team membership at cutoff, source snapshot, feature policy, forecast stage and reconstruction class. |
| `player-forecast.v1` | Start/substitute/DNP probabilities, conditional minutes distribution, count distribution, raw/calibrated probabilities, uncertainty, support, versions and fallback reason. |
| `player-market.v1` | Typed participant plus exact player/team/fixture IDs, statistic/definition version, side, line, period, participation rule and bookmaker market identity. |
| `player-quote.v1` | Exact market identity, bookmaker/provider IDs, raw outcome text, decimal price, provider quote time, captured time, mapping evidence, status and source hash. |
| `player-settlement.v1` | Immutable publication-time rule definition, evidence requirements, grading states, correction history and explicit pending reasons. |
| Candidate bundle | Data/feature/minutes/count/calibration/policy/identity/settlement versions, supported scope, training cutoff, dependencies, seed, hashes and qualification report. |

Use stable IDs in hashes and persistence; names are display fields. Keep
forecast identity, recommendation identity and delivery identity distinct.
Retain the exact player universe considered and reasons for rejecting players,
quotes and selections. An unavailable prediction is a saved result, not a row
that disappears from coverage reporting.

Store observation time separately from event time and assumed availability.
Backfilled final statistics may support an explicitly reconstructed cohort;
they do not become original pre-match captures. For retrospective final-data
availability, reuse a justified versioned policy only where appropriate and
record its limitations. The team dataset's kickoff-plus-three-hours convention
does not establish when historical player data or lineups actually arrived.

## 6. Shared pre-match player features

Implement one pure feature builder for research and inference, with explicit
input snapshots and no hidden live provider, Chroma or current-profile reads.
Cache by all relevant snapshot/policy hashes, not only player name or season.

Proposed feature groups:

- **Player production:** market-specific counts and exposure, shrunk per-minute
  rates, recent chronological form, uncertainty/support and source missingness.
- **Participation:** verified earlier starts, substitute appearances, minutes
  distributions, substitution timing where available, rest and congestion.
- **Role:** time-valid position, observed starter/bench status and verified
  penalty/set-piece role when known. Missing role evidence stays explicit.
- **Team and opponent:** cutoff-safe production/concession rates, venue,
  expected team output, opponent strength and competition context.
- **Membership and change:** team as of the fixture, recent transfer or role
  change, time since last appearance and cold-start status.
- **Lineup stage:** exact known teammates/opponents and roster completeness
  only if their lineup snapshot was available before the cutoff.

Initial history-window and decay choices must be bounded and declared before
development scoring. A training-example lookback, a player's rate-history
window and a team's profile window are different choices. Do not import all
three approved team-profile variants as an unbounded player search.

Use verified fixture membership to handle transfers, loans and multiple
competitions; current club names must not rewrite past observations. If
cross-competition history is allowed, declare its scope and adjustments.
Retain separate counts for observations and minutes contributing to each rate.
Sparse or new players use conservative pooled role/league priors or unavailable
states, with support visible to the policy.

Do not use the forecast fixture's actual minutes, final lineup, rating, shots,
goals or post-match formation as pre-match predictors. Earlier player outcomes
may update later features only after the declared label-availability cutoff.
Earlier matches inside an evaluation block can update features under a frozen
policy; fitting parameters remains confined to permitted training windows.

If team-model forecasts become player inputs, create them out of time using
team models fitted and selected only on earlier data. In-sample team forecasts
and final match totals are not legitimate substitutes. Retain their bundle IDs.
Avoid circular inference: lineup context may feed a team model, then a player
model may consume that frozen team forecast; final player predictions must not
feed back into the same team forecast without a separately validated model.

## 7. Participation, minutes and count modelling

### 7.1 Participation and minutes

Define the candidate universe from timestamped rosters, squad membership,
lineups or quoted players available at the forecast stage. Selecting only
eventual participants would hide nonappearance risk. If historical evidence
cannot reconstruct that universe, do not claim to have validated preliminary
participation forecasts on those rows.

Model mutually exclusive states `starts`, `substitute appearance` and `DNP`,
then conditional playing-time distributions for the playing states. Compare
simple role/history frequencies against regularized logistic or multinomial
models. A minutes baseline should reproduce observed earlier role-specific
durations; more complex discrete-time, distributional or survival approaches
are later bounded candidates when substitution evidence supports them.

Use observed starting-XI identity rather than “played at least 60 minutes”.
Separate pre-lineup and confirmed-lineup forecast stages. A retrospectively
known starting XI can support a labelled conditional reconstruction experiment;
without a saved pre-kickoff timestamp it is not evidence of real-time lineup
availability. Continue to represent early substitutions and late withdrawals.

Actual minutes may be a training outcome or count-model exposure. At forecast
time, integrate over predicted minutes rather than supplying eventual minutes.
Version regulation/stoppage-time exposure semantics; extra-time player totals
are excluded unless regulation observations are independently recoverable.

### 7.2 Production conditional on exposure

Use pooled models that share information across players while accounting for
role, league, team and opponent. Do not train a separate complex model for
every player. Player effects or encodings must be regularized, computed only
inside the fitting slice and have an explicit unseen-player fallback.

Proposed comparisons, to be budgeted after eligibility is known:

| Model family | Purpose and constraints |
|---|---|
| Role/league baseline | Earlier rate with conservative pooling, combined with the same minutes forecast used in direct comparisons. |
| Player rate baseline | Shrunk historical player rate with simple context; quantify the value of player-specific history. |
| Existing statistical reconstruction | Frozen legacy projection arithmetic using qualified historical inputs; document omitted adjustments and label it a reconstruction. |
| Regularized Poisson / negative binomial | Conditional count rates with verified exposure; fit dispersion on allowed data rather than assume it. |
| Ridge | Optional expected-count/rate comparator with declared nonnegative transformation; not a complete probability model by itself. |
| LightGBM, XGBoost, CatBoost | Bounded count-mean candidates; comparable features/search budgets and isolated pinned dependencies. Dispersion and probability calibration need their own treatment. |
| Logistic regression | Participation and declared binary-event comparators, such as at least one shot on target; distinct from a probability calibrator. |
| Statistical/ML blend | Convex weights learned from earlier out-of-sample predictions; zero ML weight is allowed. |

The libraries selected for team Phase 3 provide useful adapters, but their
exact team grid and qualification thresholds are not automatically approved
player specifications. Freeze a player search budget, seeds and parameter
grids before fitting. Unsupported dependencies must be reported, not silently
replaced or installed into the publishing runtime.

For a log-link exposure model, an actual training exposure offset is legitimate
only with a matched known count and valid participation/minutes. Do not divide
by null/zero exposure or count an unknown target as a zero event. At inference,
evaluate the fitted rate at each permitted future exposure. If a library does
not support the required offset, specify and test an appropriate alternative;
never smuggle future minutes into its feature vector.

### 7.3 Full distributions and coherence

For information available at cutoff `x`, integrate uncertainty explicitly:

```text
P(Y = k | x) = sum over participation states r and minutes m:
              P(r | x) * P(m | r, x) * P(Y = k | m, r, x)
```

For sporting-count forecasts, confirmed nonparticipation has zero on-pitch
production. For a bet, nonparticipation may instead void the selection under
its saved rule. Carry those cases separately into payoff calculations.

Evaluate a common minutes forecast with competing rate models, then paired
ablations of improved minutes versus improved rates. Evaluating a count
distribution only at mean minutes generally loses uncertainty; retain the
mixture, normalized probabilities and explicit tail treatment.

For compatible definitions, require nonnegative integer support and monotone
over-line probabilities. A joint shots/SoT model should enforce `SoT <= shots`;
an optional conditional-binomial construction can be compared with separate
count models. Independent marginal forecasts cannot be sold as a coherent
joint distribution. Check player/team aggregate plausibility under matching
definitions, but do not force sums to match by allocating missing events or
ignoring own goals, unknown players and minutes conventions.

Zero-inflation, hurdles, shot-quality/xG inputs and richer role models are
later predeclared experiments if the simpler comparisons justify them.
Do not treat provider ratings as an xG substitute or assume event coordinates
and player xG exist in the current dataset.

## 8. Chronology, preprocessing and protected tests

Use a separate immutable player dataset and split manifest. Every player,
market, quote, stage and revision associated with one provider fixture must
remain in the same partition. Simultaneous kickoffs stay together. Preserve
the team dataset and its existing assignment; the player work must not reopen,
move or tune against reserved team periods.

Proposed player sequence:

`initial training → forward development → player-model confirmation (counts/raw distributions) → calibration/policy reserve → final player-system test`

Choose dates from eligible player coverage and calendar structure before any
candidate scores. Use separate earlier confirmation and later final-system
tests, consistent with the owner's two-test principle. Do not mechanically
copy team dates onto player history that cannot support them. If overlapping
fixtures or team-model inputs would compromise a shared untouched test,
reserve a later common period or qualify the player system separately with
clear limits. Record consumed periods in a shared test-access ledger.

Reusing a period already used to choose an upstream model is not an untouched
test of the combined system. Availability counts may be audited; model scores
remain behind explicit opening operations tied to frozen specifications and
candidate hashes. After a test informs a change, record it as consumed and
use genuinely later evidence for renewed qualification.

Use nested forward development where support permits. All preprocessing,
category encodings, priors, dispersion, feature selection, early stopping,
hyperparameters and blend weights must fit using only the allowed earlier
slice. Enforce label availability at each fit cutoff. Missing indicators and
zero values remain distinguishable through fitting and saved-model inference.

Proposed fitting unit: give each fixture equal total weight within a market
before elapsed-time recency, distributing that weight across its eligible
player examples. Review this choice before freezing it: equal-player-row
weighting would give greater influence to fixtures with more captured players.
Report both player-level and equal-fixture summaries. Exposure in a count
model and observation weights have different meanings and must not be
interchanged. Report raw rows, unique players/fixtures/matchdays, total
qualified exposure, weighted effective sample size and league contributions.

Retaining players across chronological train/test periods is expected for a
forecasting task. Additionally report unseen-player, new-team and transfer
slices. A separate player-held-out diagnostic may assess cold starts, but it
does not replace chronological evaluation.

## 9. Evaluation, calibration and decision criteria

Freeze the scoring specification before development runs. Proposed metrics:

| Component | Primary proposed score | Required secondary checks |
|---|---|---|
| Participation | Multiclass log loss | Brier score, reliability, start/sub/DNP confusion and candidate-universe coverage. |
| Minutes | Distribution score suited to the chosen support, such as CRPS | MAE, bias, interval coverage/width; conditional-on-playing and unconditional results separately. |
| Expected counts | RMSE per statistic | MAE, signed bias, R² where defined; no pooling of unlike units. |
| Count distribution | Count log loss | Ranked probability score, tail mass, overdispersion, zero frequency and interval coverage. |
| Declared prop events/outcomes | Binary log loss for sporting events; multiclass log loss for complete betting outcomes | Brier score, reliability and discrimination; retain void/DNP, push and half-outcome states under the saved rule. |
| Recommendation policy | Predeclared portfolio/value criteria | Flat-stake ROI, turnover, drawdown, unique fixtures, CLV coverage, voids, pending cases and concentration. |

A half-line or scorer bet can still void under its participation rule. Binary
scoring applies to a defined sporting event, or an explicitly conditional-on-
action comparison with participation assessed separately. It must not turn
a void into a binary win or loss when assessing the complete betting outcome.

Compare candidates on identical eligible player-fixture-stage rows. Report
wider coverage, failures and abstentions separately. A method may not improve
its headline result merely by omitting difficult players. For executable-bet
comparisons use identical quote opportunities, with quote-availability losses
visible rather than reconstructed.

Slices include market, league, season/stage, forecast stage, position/role,
minutes band, support, source class, null/zero policy, transfer/cold start and
price/line band. Report rare-event counts as well as row counts. Use paired
calendar-block resampling with all fixture players grouped; assess sensitivity
to dependence across repeated players/teams and successive matchdays. Do not
treat player rows or multiple lines from one fixture as independent samples.

Predeclare a bounded ablation set for player history, minutes, team/opponent,
role/lineup and recency. Rebuild features when changing a history policy. Keep
comparison inputs controlled so that a better rate model is not credited for
a simultaneously improved minutes forecast.

Fit raw distribution parameters only on allowed training/development data.
Then fit calibrators on dedicated held-out or chronological out-of-fold
predictions, with pooled fallbacks for sparse slices. If a base model is refit,
generate valid new calibration predictions; old predictions do not remain
held out if the new base model learned those labels.

The calibration/policy reserve needs a temporal subdivision or chronological
cross-fitting: fit the calibrator earlier and develop policy on later
predictions. Selecting bets on calibrator-fitted outcomes is not independent
policy evidence. Freeze all components before opening the final system test.
If authentic quotes or sample support are insufficient, wait for prospective
data rather than consume the final reserve during development.

Store raw and calibrated probabilities, sample support, calibrator version,
uncertainty and any fallback. Calibration must preserve normalized support and
monotonicity across lines. Separate event probability, estimated value and
confidence in the estimate. Avoid unqualified confidence percentages inferred
from arbitrary heuristic scores.

Before the first fit, specify numerical minima for independent fixtures/dates,
event/non-event counts, exposure, weighted support, interval precision,
improvement margins and tolerated slice regressions. The existing team
minima and search budget do not certify player models. Unset criteria block
qualification; they must not be filled in after seeing favorable test results.

## 10. Exact prices, selection and dependence

Audit actual player-market availability under the current API-Football account.
Existing provider mappings for several player markets do not establish usable
prices for every league, player or side. Capture timestamped raw quotes
prospectively in isolated storage when that collection is separately approved.
Never calculate historical ROI from today's prices or invented past lines.

Replace fragment/surname matching with exact identity resolution. If a quote
has only a name, require a unique verified fixture-roster mapping and preserve
its evidence. Ambiguous initials, homonyms, accents, transfers, wrong-team
players or conflicting IDs remain unavailable. Preserve the original value,
line and side; never mistake an under quote for an over quote or match 1.5 to
2.5 because the player name agrees.

Verify bookmaker, market definition, participant, period, participation rule,
line, side, price status and pre-cutoff freshness before comparing value.
Provider time and local capture time serve different purposes. A missing time
must not be silently replaced with the time of a later API request. Revalidate
the quote following a lineup change and before a recommendation is delivered.

Definitions require evidence. For example, Opta distinguishes shots on target,
ordinary blocked attempts and shots hitting the frame; a name such as “shots”
does not by itself prove equivalence. This illustrates the need for a mapping,
not an assertion that API-Football or every bookmaker uses Opta's definitions.
[Definition reference](https://www.statsperform.com/opta-event-definitions/).

Compute value from the complete payoff distribution under the exact rule:

```text
unit-stake EV = (odds - 1) * (P(full win) + 0.5 * P(half win))
               - P(full loss) - 0.5 * P(half loss)
```

Push and void contribute zero net return. Partition states so their total
probability is one; DNP belongs to the state required by the saved rule.
Do not use a probability conditional on playing or avoiding a push as if it
were an unconditional full-win probability. Only use de-vigged market
comparators where equivalent, sufficiently complete outcomes are available.

Selection should share the later versioned Spix policy: calibrated value,
uncertainty, executable prices, data support and concentration limits. Log all
evaluated candidates and rejection reasons. Policy thresholds are selected
on development evidence and frozen before the final test.

Extend Match Read compatibility to recognize player identity, nested lines,
opposite selections, related shots/SoT/goals and player/team relationships.
Two different players in one market are not automatically duplicates; two
correlated selections are not automatically independent. Preserve the current
maximum of three compatible selections per fixture and the five-fixture
headline shortlist unless the owner later changes that product rule. Player
selections count toward the existing fixture cap. Do not manufacture combined
same-game probability, price or EV through multiplication.

## 11. Canonical delivery, closing capture and settlement

### 11.1 Structured delivery

Extend the canonical contract and persistence with typed player identity and
saved rule/quote references. The current generic participant string is not
sufficient on its own. Review identity hashes, serializers and database
constraints together; additive versioned changes must preserve old records.

Upgrade lineup evidence to retain exact team/player IDs, starter/substitute
membership, capture time, source hash and revision. Existing eleven-name
validation is useful scaffolding but is not an exact player-identity contract.
Partial XIs, duplicate IDs, wrong teams and stale data cannot qualify a
confirmed-lineup prediction.

Route website views, slash commands and scheduled posts through the same
persisted structured decisions. Ordinary page requests must not rerun the
model. Render explicit preliminary, confirmed, amended, expired, unavailable
and no-bet states. Player identity, line, odds, lineup stage and freshness
must be clear without exposing internal implementation details.

Replace legacy text parsing and post-render player filtering in the migrated
path. A numerical failure must return unavailable; remove the LLM pick fallback
from that official path. Preserve older standalone recommendations as their
own cohort rather than rewriting them as new canonical publications.

### 11.2 Publication identity and amendments

Record actual delivery through the shared publication bridge. Retries and
website/Discord delivery of the same recommendation must not create duplicate
stakes. Save immutable initial selection, quote, model and rule evidence.

Current headline measurement uses selections from the first delivered Match
Read per fixture. A player selection added after lineup confirmation to an
earlier delivered card is an amendment; an earlier no-bet card cannot be
replaced retrospectively. A separate “first player release” report would need
an explicitly approved prospective cohort definition. This plan does not
change the existing headline identity or reporting policy.

A withdrawn starter, changed kickoff or fresh quote may produce a new version
and an edited message, while preserving the earlier delivery and its outcome.
Keep candidate/shadow records inaccessible to public release queries.

### 11.3 Closing and settlement

Both current closing capture and settlement reject participant markets.
Extend them before public activation; player projections alone do not close
the measurement gap. Closing quotes require the same player, market, side,
line, period, bookmaker and rules with qualifying pre-kickoff timestamps.
Missing, changed-line or stale closes remain unavailable; report CLV coverage.

Create a reviewed market/bookmaker rule matrix covering:

- Starts-required versus any-appearance terms, unused substitutes, late
  withdrawals, rounded-zero-minute appearances and interrupted appearances.
- Regulation/stoppage time, extra time, shootouts, abandonment, postponement,
  cancellation and suspension, including applicable rescheduling windows.
- Goals and own goals; assist definitions; total shots/SoT; booking types and
  timing, substitutions and any applicable bench/post-whistle treatment.
- Integer/half/quarter lines where offered, pushes, half outcomes, voids,
  provider corrections and missing/conflicting evidence.

Store the rule version at publication and grade according to that version.
Do not retroactively add newly approved semantics to old selections. If
bookmaker equivalence cannot be established, either keep the market unavailable
for betting or obtain approval for an explicitly labelled product measurement
convention before publishing it. The
[existing participation policy](participation-settlement-policy.md) does not
provide blanket authorization for new player rules.

Use exact participant evidence and retain raw evidence references plus a
compact reasoned grading record. Ambiguous participation, statistic nulls,
unsegmented AET/PEN totals or conflicting revisions remain pending. Do not
equate missing data with a losing over, winning under or automatic void.
Corrections append a revision trail and preserve the initial grade/reporting
history according to a declared correction policy.

## 12. Runtime, storage and promotion

Training runs offline in an isolated environment and produces an immutable
candidate bundle. It is never part of the ten-minute publishing cycle.
Snapshot/quote collection, inference, closing capture and settlement can share
bounded provider caches and scheduled infrastructure without fitting models.
Failed training leaves the active bundle and published records intact.

Retain compressed raw archives with hashes, relational normalized data,
immutable research exports and small deployable bundles as separate storage
classes. The website and bot runtime need qualified current features and
bundles, not every historical training export or its embeddings. Measure
storage/query/runtime costs before choosing cloud retention. Historical raw
evidence and backups require durable storage and a tested restore path; do
not discard them merely to reduce deployment size.

Respect refresh gates, database writer coordination, deadlines and retry
limits. Candidate work must not bypass an incomplete refresh or silently
change the legacy worker's behavior. Extend failure monitoring to player
coverage, lineup lateness, quote identity/freshness, minutes-distribution
shifts, null/zero rates, fallback, extreme edges, no-bet rates and settlement
backlogs. Verify private alert delivery before relying on it operationally.

Promotion sequence:

1. Offline data/model qualification with preserved artifacts and review.
2. Separately authorized prospective snapshots and prices, with baseline and
   candidate predictions using the same input cutoffs.
3. Isolated shadow comparison and real closing/settlement/reporting evidence.
4. Explicit approval for the exact market/competition/stage scope.
5. Atomic activation of one consistent bundle, with tested rollback and
   long-running-process reload behavior.

One worker cycle must use one consistent version set. Concurrent reloads,
restarts and failed loads must not mix feature/model/calibrator/policy versions.
Rollback restores future inference behavior without erasing publications,
settlements or the evidence from the previous bundle.

## 13. Implementation batches and deliverables

These batches start after the current team Phase 3 work is reviewed. Their
names avoid renumbering the existing project phases. Collection and public
activation remain separate decisions at the relevant gates.

| Batch | Work | Deliverable and exit gate |
|---|---|---|
| P0 — Evidence and scope | Recheck repository state; audit local player history, definitions, quotes and existing code; specify any bounded provider verification. | Dated data/market feasibility report, coverage/exclusion tables, identity ledger and reviewed initial scope. No claim of eligible training data from row counts alone. |
| P1 — Usable player dataset | Reconcile eligible local history; perform only separately authorized missing-data collection/import; implement evidence, eligibility and pure feature contracts. | New immutable dataset, exact lineage, replayable features, grouped chronology, two protected tests and frozen scoring/search specification. Unsupported cohorts remain explicit. |
| P2 — Baselines and candidates | Build participation/minutes baselines and rate/distribution adapters, then bounded forward comparisons and ablations. | Out-of-sample predictions, equal-cohort comparisons, uncertainty, candidate bundle and earlier confirmation review. No activation. |
| P3 — Probability and price policy | Calibrate, validate line consistency, normalize exact quotes, freeze participation/payoff rules and learn a conservative selection policy. | Calibrated bundle, price/settlement contract tests and frozen system ready for its final untouched evaluation. If quotes are unavailable, descriptive qualification only. |
| P4 — Integration and shadow | Extend canonical persistence, lineups, Match Read compatibility, closing, settlement and reporting; migrate legacy callers into an isolated mode. | No-publication integration tests plus real prospective shadow lifecycle evidence and operational acceptance. Prospective quote capture may start earlier if separately authorized. |
| P5 — Review and limited release | Open the declared final test when eligible, assess shadow evidence, approve scope, exercise activation/rollback and observe initial public delivery. | Signed-off scope and rollback target; actual publication-to-settlement traceability. Weak/insufficient results keep the candidate inactive. |

P3 here is a player-workstream batch label, not completion of the project's
team Phase 3. Candidate changes prompted by any inspected test consume that
test and require later evidence; batch labels do not waive this rule.

Each experiment preserves a manifest, schema, eligibility version, complete
exclusion/coverage report, source/code/dependency hashes, seed, split membership,
fit cutoffs, weighting and search budget, OOF predictions, baseline/candidate
metrics, calibration/policy artifacts, inspection ledger and limitations.

Suggested future module boundaries, all **proposed, not created here**:

| Responsibility | Existing area to extend or pattern to reuse |
|---|---|
| Player evidence/eligibility/datasets | `Scripts/data_platform/features/`, using the Phase 3 evidence/split patterns with player-specific contracts. |
| Shared pure player features | `Scripts/rag_ingest/core/`, alongside existing canonical feature builders. |
| Isolated research CLI/adapters | `Scripts/ops/`, using the benchmark manifest/verify/replay design; avoid adding player fitting to legacy training commands. |
| Statistical comparator | `Scripts/rag_ingest/player_projections.py`, via frozen pure arithmetic rather than current live wrappers. |
| Exact quotes and market service | `Scripts/rag_ingest/odds_provider.py`, canonical market contracts and `Scripts/data_platform/services/`. |
| Publication/closing/settlement | Existing shared services and identity/reporting modules, with backward-compatible participant extensions. |
| Product renderers | `Scripts/web_app/` and `Scripts/discord_bot/`, consuming persisted decisions. |

## 14. Required tests and acceptance evidence

These tests are future implementation requirements; none are claimed to pass
because this plan was written.

| Area | Required cases |
|---|---|
| Evidence | Explicit zero/null/omission/unavailable responses; incomplete archives; hash mismatch; conflicting corrections; no inferred zeros without qualified evidence. |
| Identity | Duplicate/wrong player or team IDs; repeated fixture display names; aliases/homonyms; transfers and competing source revisions. |
| Participation | Missing row versus verified DNP; starter substituted early; substitute playing over 60 minutes; rounded-zero appearance; incomplete squad; withdrawn starter. |
| Period and labels | Regulation versus stoppage/extra time; own goals; incompatible assist/card definitions; unavailable labels never imputed; matched event/exposure denominators. |
| Time and leakage | Future results/lineups/corrections cannot change a saved earlier snapshot; actual future minutes excluded; label availability enforced; upstream team predictions fitted out of time. |
| Isolation and replay | Held-out data cannot affect preprocessing/encodings/hyperparameters; all fixture rows stay together; simultaneous fixtures remain grouped; repeated execution reproduces membership and hashes. |
| Models | Sparse/unseen players; valid exposure; raw margin versus count scale; serialization; normalized participation/count distributions; stable tails and monotone line probabilities. |
| Calibration and EV | Separate calibrator/policy data; no conditioning mistake for DNP/push; full/half outcome arithmetic; no use of test labels for dispersion or calibration. |
| Quotes | Wrong player/team/fixture, side, line, bookmaker, period, rule or timestamp; stale/suspended prices; ambiguous mapping; late lineup correction requires revalidation. |
| Compatibility | Duplicate/nested selections; multiple players in one market; player/team relationships; shared fixture cap; no independent same-game multiplication. |
| Publication | Two surfaces yield one recommendation/stake; first-publication and amendment separation; first no-bet immutable; shadow data cannot pass public release gates. |
| Measurement | Exact pre-kickoff closing selection; pending versus void; supported participation rules; corrections preserve lineage; unrelated valid markets settle despite one player's missing evidence. |
| Operations | Restart/concurrency/retry idempotency; incomplete refresh gate; quota/deadline failure; atomic bundle reload; failed candidate load; rollback and backup restore. |

Local test completion is distinct from provider verification and real delivery
acceptance. Before public release, observe actual busy and quiet matchdays,
lineup amendments, stale/missing prices, postponements and relevant provider
corrections. Empty worker cycles and mocked outcomes do not close those gates.

Maintain an evidence matrix per market/competition/forecast stage with separate
statuses for local implementation, real-provider verification, offline model
qualification, prospective shadow qualification and actual public lifecycle
acceptance. No pooled pass automatically qualifies every league or player role.

## 15. Decisions to freeze when implementation starts

The owner has approved documenting this roadmap and keeping team Phase 3 first.
The following remain future decisions, with recommended starting positions:

| Decision | Recommended starting position |
|---|---|
| Pilot markets and competitions | Shots/SoT, confirmed starters, competitions selected from actual eligible data and exact quote coverage. |
| Additional acquisition | Reconcile local exports first; bounded provider verification before an explicitly sized larger collection. |
| Historical stage validity | Label reconstructed starter scenarios honestly; prospective timestamped lineups establish operational availability. |
| Search and weighting | Bounded pooled models with role/league context; proposed equal fixture mass before recency, reviewed for the player universe. |
| Support and statistical gates | Freeze numerical minima, practical improvements, precision and slice tolerances after coverage audit and before scoring. |
| Dates and shared holdouts | Two separate player tests with an access ledger; protect team reserves and require untouched evidence for any combined system. |
| Market settlement | Exact supported participation/statistic rules; any approximate product convention requires explicit approval. |
| Presentation and measurement | Preserve three total selections per fixture and existing first-publication/amendment reporting; separately review any dedicated player cohort. |
| Activation | Market/league/stage-specific approval after offline and real shadow evidence, with an exercised rollback. |

These decisions can be resolved together at the start of the relevant batch.
They are not a request for implementation permission now, and no arbitrary
numbers should be filled in simply to make the plan look complete.

## 16. Repository reference map

Paths are relative to this document. They record the inspected foundation,
not a promise that every existing helper is safe to reuse without changes.

- [Project improvement plan](ml-prediction-improvement-plan.md),
  [Phase 3 specification](phase3-technical-plan-2026-09-26.md),
  [frozen team dataset](phase3-dataset-2026-09-26.md) and
  [history import](eight-season-history-import-2026-09-26.md).
- [Existing player engine](../Scripts/rag_ingest/player_projections.py),
  [legacy RAG entry points](../Scripts/rag_ingest/rag_cli_v2.py) and
  [player tests](../Scripts/tests/test_player_projections.py).
- [Fixture/player models](../Scripts/data_platform/models/fixtures.py),
  [feature snapshots](../Scripts/data_platform/models/features.py),
  [canonical upserts](../Scripts/data_platform/sync/upserts.py),
  [canonical feature aggregation](../Scripts/data_platform/sync/features_from_canonical.py)
  and [legacy snapshot importer](../Scripts/data_platform/sync/features_from_snapshot.py).
- [Legacy fixture-player exporter](../Scripts/Premier_League/prem_player_stats_per_gw.py),
  [legacy player profiles](../Scripts/Premier_League/Prem_player_profile.py),
  [season player collector](../Scripts/top_5_leagues_players.py) and
  [historical team-data collector](../Scripts/ops/prediction_history_download.py).
- [MarketResult](../Scripts/rag_ingest/core/market_result.py),
  [odds adapter](../Scripts/rag_ingest/odds_provider.py),
  [lineup context](../Scripts/rag_ingest/lineup_context.py) and
  [Match Read compiler](../Scripts/data_platform/services/match_read_compiler.py).
- [Publication identity](../Scripts/data_platform/publication_identity.py),
  [delivery bridge](../Scripts/data_platform/services/match_read_delivery.py),
  [release gates](../Scripts/data_platform/services/match_read_release.py),
  [reporting](../Scripts/data_platform/publication_reporting.py),
  [closing capture](../Scripts/data_platform/services/closing_prices.py) and
  [settlement](../Scripts/data_platform/settlement.py).
- [Approved team-card measurement policy](participation-settlement-policy.md)
  and [measurement automation](phase2-measurement-automation-2026-09-24.md).
- [Discord player commands](../Scripts/discord_bot/cogs/slash_cmds.py) and
  [scheduled delivery](../Scripts/discord_bot/cogs/auto_push.py).

Writing this plan performed no player import, provider collection, embedding,
training, model promotion or public publication. Implementation status remains
as recorded in section 2 until subsequent dated evidence supersedes it.
