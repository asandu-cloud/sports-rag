# Phase 1 historical-data diagnostic — 22 September 2026

Follow-up (25 September): legacy team collectors now preserve nulls and explicit
zeros. Five evidence-backed card fields were repaired offline, raising 2025
computable card targets from 305 to 307 (one fully known zero-red fixture).
The dated diagnostic below is preserved; unrestricted card modelling remains
unqualified. See [repair evidence and limits](card-stat-missingness-repair-2026-09-25.md).

## Decision

**The small diagnostic is complete. Proceed with Phase 2's measurement and
settlement work; do not sign off unrestricted card-model training yet.**

The local fixture/team links passed the checks performed, and sampled corner
and shots-on-target values agree with the provider where both are available.
Card-label missingness is demonstrably selective, not harmless random gaps.
Re-downloading all team statistics is not, by itself, an evidenced solution.

No production records, prediction rules, active models, published Match Reads,
profiles, embeddings, or scheduler settings were changed. No models were fitted.
New code is an isolated diagnostic utility with regression tests. Phase 2 has
not been implemented by this diagnostic.

## Scope and evidence

- Read-only, transaction-consistent snapshot of `Index/platform.db`: provider
  seasons **2025 and 2026**, all 13 supported competitions. This is the existing
  current/prior-season modelling window, not a change to its length.
- Local audit: **5,083 FT fixtures**, including 4,460 in 2025 and 623 in 2026.
  AET/PEN outcomes are not assumed to be regulation-time labels. Player-history
  completeness and original historical pre-match data vintages are outside this
  small team-stat diagnostic.
- Provider sample: **52 fixtures**, with seed `20260922`; 25 randomly selected
  controls, one per populated competition-season cohort, plus 27 targeted/extra
  fixtures. There are 27 sampled 2025 fixtures and 25 from 2026. UECL has no FT
  2026 fixtures in this snapshot, so that cohort cannot supply a completed-game
  comparison. All 13 competitions are represented; 17 sampled fixtures came
  from legacy recovery and 35 from the canonical provider path.
- **117 successful API calls**: 13 competition-coverage requests, 52 team-stat
  requests and 52 complete event-timeline requests. The durable ledger records
  **118 attempts**, including one failed sandbox DNS attempt, below the hard
  120-attempt limit. No player downloads or large backfill ran.
- Raw responses, capture times, request parameters, response hashes, and the
  frozen local inputs are saved separately. Credentials are not recorded.
  Restarts reuse successful responses; failures consume the persisted budget.

The sample is deliberately stratified and partly targeted. Its pooled rates
are **not estimates of error prevalence across the entire database**. Random
controls and targeted/extra comparisons are reported separately.

Private evidence directory:
`Index/prediction_experiments/phase1-diagnostic-2026-09-22/`.

- `PREPARED.json` verifies `local-snapshot.json`, `local-audit.json`, `sample.json`
  and `request-plan.json`.
- `provider/` contains the request ledger and checksum-verified raw responses.
- **`analysis-002.json` is the reviewed report**. It distinguishes 52 processed
  fixtures from 48 with comparable team statistics. The initial analysis is
  preserved, but its `compared_fixtures: 52` counted empty responses as processed
  cases; use revision 002's explicit denominators instead.
- `local-archive-crosscheck.json` records the follow-up checks of earlier raw
  payloads for the discrepant/recoverable fixtures.

Do not commit or publicly serve this evidence directory.

## Findings

### 1. Local structure passes; completeness is market-specific

SQLite quick-check returned `ok`. There were no foreign-key violations,
duplicate provider fixture IDs, missing catalogue links, orphan team-stat rows,
or quarantined fixture/team pairings. The scoped FT checks found no invalid
numeric ranges, team-score disagreements, or shots-on-target exceeding total
shots. These checks do not establish every statistic's real-world correctness.

| Provider season | FT fixtures / goal labels | Corner labels | SoT labels | Card labels |
| --- | ---: | ---: | ---: | ---: |
| 2025 | 4,460 | 4,045 | 4,029 | 305 |
| 2026 | 623 | 623 | 623 | 241 |
| Total | 5,083 | 4,668 | 4,652 | 546 |

Labels here mean numerically known results, not approved betting-market
semantics or sufficient training features. Goals were checked against local
fixture scores; this pass did not independently re-download final-score lists.

### 2. Card-training selection bias is confirmed

**All 305 usable 2025 card labels contain a recorded positive red-card count.**
There are no fully known zero-red fixtures in that season's card-label subset.
For 2026, the 241 card labels contain 81 recorded-positive-red fixtures and 160
recorded-zero-red fixtures. Other fixtures still lack required components.

The current candidate card target requires known yellow and red counts for
both sides. Conservative handling of lossy legacy zeros, together with missing
provider values, leaves a subset enriched for recorded dismissals. Training or
evaluating a general card model on that subset would be misleading. This is a
finding about the **candidate historical dataset**, not a measured accuracy
result for the current live prediction model.

### 3. Most comparable statistics agree; four responses are empty

Of 52 sampled fixtures, **48 returned exactly the expected two team IDs**,
giving 96 comparable team sides. The other four returned empty statistics, not
conflicting identities:

- UCL 2025: fixture `1405445`.
- UECL 2025: fixtures `1410024`, `1410010`, `1383506`.

All four still returned event timelines. All 26 competition-season coverage
records advertise fixture-statistics and events coverage; that does not ensure
statistics exist for every fixture. API-Football's own guide also distinguishes
coverage flags from guaranteed availability and warns that individual
statistics can be null when uncollected. It does not establish a universal
null-means-zero rule. [Official API-Football guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide).

Within the 96 comparable team sides:

- Corners and SoT: **94 agreements and two still-unknown values each**, no
  numeric disagreements. Both fields agreed on all 46 comparable sides from
  the random-control subset.
- Yellow cards: 94 agreements and two newly available explicit zeros.
- Red cards: 35 agreements, **56 still unknown**, three newly available explicit
  zeros, and two stored zeros for which the fresh response now returns null.
- Total shots: 92 agreements, two numeric differences, two still unknown.
- xG: 74 agreements, two numeric differences, four newly available values,
  16 still unknown. Pass accuracy also has four newly available values.

There are **13 newly available scalar values across five fixtures**, retained
as review candidates, not applied: `1552160`, `1387850`, `1575487`, `1451274`,
`1421213`. A newer null must not automatically erase an earlier evidenced zero.

### 4. Some differences are verified provider revisions

| Fixture / API team | Field | Stored | Fresh provider | Earlier raw evidence |
| --- | --- | ---: | ---: | --- |
| `1558639` / `259` | Total shots | 8 | 9 | Archived response also says 8 |
| `1575487` / `214` | Total shots | 7 | 6 | No matching local raw archive found |
| `1584413` / `564` | xG | 2.78 | 2.74 | Archived response also says 2.78 |
| `1584413` / `997` | xG | 0.46 | 0.52 | Archived response also says 0.46 |

The two relevant 21 September raw payloads passed their stored checksums
(archive IDs `781` and `728`). Those three fields changed between provider
responses; this is not evidence of a parsing error for those archived values.
The Primeira Liga difference remains unexplained without its earlier raw
response. Store correction provenance and observation time; do not claim a
fresh value was available before the original prediction.

### 5. Card events cannot yet substitute for missing team-card totals

The sampled timelines contain 199 `Yellow Card` and 15 `Red Card` events, with
**no second-yellow-specific event detail**, so that edge case remains untested
on real evidence. All 58 currently null red-card sides in the comparable sample
have no listed red event. That is suggestive, but absence from a timeline does
not prove complete coverage or an authoritative zero.

There are also direct disagreements between event counts and statistics:

- QPR, fixture `1563175`: team statistics say two yellows; the timeline lists
  three. Two yellow events have no player ID, so actor eligibility needs review.
- Team `4160`, UEL fixture `1421213`: team statistics say one yellow; the
  timeline lists three.

Do not resolve those discrepancies by choosing whichever count supplies a
training label. Second bookings, bench/staff eligibility, period boundaries,
and the bookmaker's exact card-counting rules require explicit definitions.

### 6. Phase 2 must preserve the precise card market

Code inspection found two distinct provider market keys:
`totals_cards_over_under` and `totals_yellow_cards`
(`Scripts/rag_ingest/odds_provider.py`). Both are accepted by the generic
`cards` total extractor (`core/odds_extraction.py`) and closing-price matcher
(`prediction_tracker.py`), while the candidate target sums yellow + red.

This is a concrete definition-mixing risk, not proof that every prior card
selection was incorrectly graded. Phase 2 must carry the exact market identity,
bookmaker, period and settlement definition through publication, grading and
closing-price matching. Ambiguous old rows cannot safely be assigned a rule
from their eventual result.

## Next work and remaining sign-off conditions

1. **Proceed with Phase 2:** trace existing publication/tracking/resolver paths,
   preserve version and quote identity, and make settlement independent of the
   Discord process. Keep published, shadow and legacy cohorts separate.
2. Add an explicit pending reason for missing statistics or unresolved market
   definitions. Unsupported/ambiguous card outcomes stay pending or excluded
   from eligible reports; never manufacture zero, a loss, or a void. Automatic
   grading requires a verified market definition and sufficient final evidence.
3. Keep the 13 fresh-value candidates and observed revisions as a separately
   reviewed repair proposal. No database repair or candidate re-export was
   performed by this diagnostic.
4. Before approving card training, resolve null/zero and event-count semantics
   with provider clarification and verified edge cases, then show representative
   coverage by league/season. Do not launch the deferred 2,636-request backfill
   merely to repeat null statistics. Legacy zero-corner/SoT missingness also
   remains a coverage issue despite the sampled non-null agreements.
5. Build Phase 3 benchmarks only on explicitly eligible, temporally valid
   cohorts; sparse/unresolved markets are withheld. Offline engineering need
   not wait for the next matchday. Prospective evidence and the separate full
   Phase 0 numerical-input/quote replay gate remain open.

## Verification and reuse

**71 focused tests passed** across this diagnostic, historical recovery,
canonical dataset loading, and pre-match feature building. Tests cover read-only
access, missing versus zero, identity errors, deterministic sampling, empty
responses, preserved event evidence, checksums, budget persistence, no automatic
retries, and non-overwriting offline analysis.

A final read-only comparison found all 5,083 audited FT fixture records and
their team rows unchanged from the frozen snapshot. Separately, 395 upcoming
`NS` fixtures had newer fetch timestamps in the live database, with no other
field differences. The diagnostic has no database write path and uses its
frozen inputs regardless of concurrent refresh activity; it does not claim the
entire live database stayed byte-for-byte unchanged while it ran.

Re-analyse the captured evidence without API calls or production writes:

```bash
./.venv/bin/python Scripts/ops/prediction_history_diagnostic.py analyze \
  --experiment Index/prediction_experiments/phase1-diagnostic-2026-09-22
```

This writes a new numbered analysis, preserving earlier files. No command needs
to be added to the weekly refresh chain. Any new sample or repairs should be
separately approved rather than silently extending this bounded diagnostic.
