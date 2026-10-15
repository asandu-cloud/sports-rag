# Local card reconstruction and fixed backtest

30 September 2026. Authorized scope: reconstruct and backtest cards; stop before
weight optimisation. This protocol is recorded before this run's card outcomes
are inspected. No provider calls, database repairs, production changes or promotion.

## Evidence and access

Retain the pre-2024 development boundary from the September 28 card protocol:
only completed fixtures with kickoff + three hours strictly before 1 January
2024. Fixture/team/name/date/archive metadata may be inspected outside that period
to resolve identities and reject reserved rows **before decoding their outcomes**.
The final-system test and prospective reserve remain protected. A provider season
2023 file contains 2024 matches too; its filename does not authorize all its rows.

Use a read-only SQLite transaction, the checksum-verified Phase 3 regulation
history, local raw-response references, and structurally filtered legacy player
exports. Preserve exact provider IDs. A unique exact home/away-name match within
the competition and provider season can produce a documented provisional identity;
Existing explicit competition-registry aliases may also resolve team names when
they map to exactly one of the already identified provider team IDs; preserve
the registry source/hash. Ambiguous names, duplicate meetings, conflicting IDs
and unmapped rows are excluded.
Never choose between ambiguous fixtures using their card outcomes.

Reuse `phase4-participation-card-target.v1` and the approved participation parser.
Keep source-certified targets, provisional arithmetic and unknowns separate.
Normalized exports cannot certify their own source completeness. In particular,
exports that filtered out one-minute participants cannot establish the approved
positive-minute target, even when their remaining rows pass roster checks.
Do not fill missing red cards, minutes or missing players with zero, synthesize
original envelopes, or treat duplicate CSV/JSON exports as independent evidence.
Keep source disagreements visible. Record source hashes, exact admitted input
rows and exclusion reasons so the reconstruction can be replayed offline.

Search local source locations and archive metadata before concluding new
collection is required. Only open player/event payloads after exact fixture and
period attribution. Unattributed blobs remain an explicit inventory limitation.
Provider collection, if needed, is a separate unapproved task.

## Dated inputs and fixed comparison

Only qualified targets may enter numerical fitting, team/referee histories or
scoring. Team histories use the current and immediately previous provider season;
every contributing match must precede the forecast's UTC calendar day and have
assumed label availability (kickoff + three hours) strictly before its cutoff.
This is an `assumed_final` reconstruction, not original published predictions.

Use exact normalized referee names within a competition. Do not automatically
merge initial/surname aliases; report their fragmentation. No full-season referee
averages may be attached retrospectively to earlier fixtures. Derive league,
team, opponent-induced and referee weighted-card rates from the same qualified
history. Sparse/unknown referee input is neutral, with its reason saved.

The fixed aligned statistical reference uses existing configured card coefficients
(own/opponent, season/recent, referee), captured before scoring. It is a research
reconstruction aligned to participation targets, not a replay of original public
card predictions. Compare team-only against team-plus-referee on identical
fixtures. Do not fit or search weights, half-lives, windows or distributions.
Unreconstructable lineup, foul and live context must be explicit. A reference
with these omissions is not advertised as the full frozen production control.

Retain the existing calendar-2022 fitting/support period and calendar-2023
evaluation period. Minimums remain 500 fitting fixtures / 26 ISO weeks and 200
evaluation fixtures / 20 weeks; per-league conclusions need 200 / 20. Chronological
profiles can advance from strictly earlier qualified outcomes during evaluation.
Require both teams to have at least eight qualified current-season observations
for the fixed comparison; report resulting exclusions separately. No-red and
unknown-red coverage must be reported; never select fixtures for positive reds.

Primary diagnostic: paired count negative log likelihood; secondary MAE, bias,
RMSE, Brier/log loss at over 4.5, reliability and central 80% interval coverage.
Use the same fixed Poisson distribution for both means to isolate referee effects;
this does not qualify a card dispersion model. Save paired ISO-week bootstrap
intervals (1,000 draws, seed 20260930) and league/season-stage/referee-coverage slices.
No ROI, CLV, promotion or claimed improvement from unsupported cohorts.

If qualified evidence or chronological support fails, output a blocked backtest
report and exact missing prerequisites, not metrics based on provisional targets.
Synthetic tests verify implementation only and must be reported separately from
real-data backtesting. Preserve an immutable experiment, protocol, code/dependency
hashes, evidence ledger, coverage, predictions (possibly empty) and checksums.
