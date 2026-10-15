# Research profile follow-up verification — 8 October 2026

The backend now applies the specified observation rules to summaries, peer builds, frequencies and chart values. Canonical source statistics remain unchanged. All requested real-data checks match except the approximate Haaland peer count: the correct count under the new rule is **61**, explained below.

## Files changed in this follow-up

- `Scripts/data_platform/research_stats.py`: exact eight-field team observation test, both-side match gate, played-player count filling, separate fixture-score results, versioned interpretation.
- `Scripts/data_platform/repositories/research.py`: fixture-driven team queries with optional statistics, referee exclusion support, available scopes, squad aggregates, subject details, recent-activity search and indexed player filtering.
- `Scripts/data_platform/services/research_profiles.py`: aggregation and coverage, role/context, structured comparisons and headlines, scopes, leaders, shooting metrics/funnel, opponent abbreviations and last-10 values.
- `Scripts/tests/data_platform/test_research_profiles.py`: updated interpretation checks and new contract tests; 45 test cases.
- This report and JSON evidence in `docs/research-profiles/follow-up-2026-10-08/`: `haaland_2025.json`, `haaland_2026.json`, `arsenal_2025.json`, `taylor_2025.json`, three `*_search.json` responses, `verification.json`, `source-audit.json`, `build.json`, and `test-results.json`.

No schema change or migration was needed. The existing router and model suffice. The full build replaced derived research ranks/build metadata/referee aliases only. No prediction, odds, Match Read, selection or frontend logic changed. SHA-256 checks found no changes to the 46 protected frontend/configuration/documentation/migration files captured before this task. No frontend files were added. Existing unrelated working-tree changes were preserved; no commit was created.

## Response contract and full examples

These are complete responses from the isolated FastAPI research router using the real `Index/platform.db`, including the full metrics, coverage, frequencies, last-10 and scopes:

- [Haaland, EPL 2025/26](haaland_2025.json) — `/api/research/players/52?competition=EPL&season=2025`
- [Arsenal, EPL 2025/26](arsenal_2025.json) — `/api/research/teams/7?competition=EPL&season=2025`
- [Anthony Taylor, 2025/26](taylor_2025.json) — `/api/research/referees/a-taylor-68af826d?season=2025`
- [Haaland, early 2026/27](haaland_2026.json) — includes labelled previous-season ranks.

Shared top-level fields remain `schema_version`, `subject`, `competition`, `season`, `totals`, `metrics`, `ranking`, `frequencies`, `headline`, `data_basis`, `last_10`, and `previous_season`; `available_scopes` is added. Each metric now supplies `metric_label`, `display_value`, `peer_label`, `comparison`, `comparison_share`, `rank_phrase`, and `comparison_text` alongside its numerical result, rank and coverage.

- Player context is under `subject.club`, `subject.position: {code, label}` and `subject.photo_url`. `role_summary` supplies role, starts, appearances, start rate and average minutes; `starts_basis` states flag/fallback use. `shooting_funnel` supplies shots/on-target/goals; `metrics.shooting_accuracy_pct` and `metrics.conversion_pct` use sum-over-sum calculations.
- Team context includes `subject.short_code` and `subject.logo_url`. `squad_leaders` has `goals`, `assists`, `shots_on_target`, `yellow_cards`, each containing up to three `{player_id, name, value, appearances}` entries. Aggregation uses appearances for that team in the selected competition-season, including before a transfer; ties use fewer minutes, then name, then ID for determinism.
- Referee context includes `subject.country` and aliases. `competition_breakdown` supplies observed/finished/excluded match counts and `small_sample`. Referee `available_scopes` uses `{code: "ALL", name: "All competitions"}` and one entry per season year.
- Search returns `latest_competition`, `latest_season`, `last_activity_at` and distinguishing context. Player search includes `club_name`, `club`, `position`; team search includes country; referee search includes country and finished-match count across all seasons in the search's competition scope. Results are sorted by latest finished activity before applying `limit`, then name and stable identity. Search position is from the latest appearance; profile position is the most common position in the selected scope.
- `headline` carries structured comparison fields plus deterministic descriptive text. Team headlines use `rank_phrase` and a null headline `percentile`; their metrics retain numerical percentiles for the lower-to-higher track and deterministic headline selection. No banned words occur in headline prose.

## Observation and coverage rules

All finished player appearances with minutes greater than zero are observed. Missing counts are zero; missing rating is excluded from its average. Rate denominators use total played minutes or summed opportunities as appropriate.

Team statistics require both sides to have at least one non-null value among shots total/on target, corners, fouls committed, yellow/red cards, possession and passes total. Zero is an observed value. Goals, blocked shots or xG alone do not establish observation. Missing rows and rows empty under this exact definition are excluded from all match-statistics metrics, even if the other side is populated. Missing xG and possession stay unknown within observed matches. Results use fixture scores independently.

`data_basis` reports `finished_matches`, `stats_observed_matches`, `excluded_matches`, the interpretation version and rank build identity. Per-metric coverage counts only genuinely unavailable observations. Frequencies retain count, denominator and unknown count. Player last-10 non-events are zero. Arsenal's empty-statistics final match retains its 2–1 result but has null statistics. Taylor's excluded match remains visible in the chronological last 10 with null statistics and `stats_observed: false`; no zero-height card bar should be inferred from it.

Headlines append the missing-observations sentence only when the selected metric has genuine missing observations. Haaland and Arsenal's result-based headline have no caveat; Taylor's foul headline does because one finished match has empty statistics.

## Acceptance: expected versus actual

### Haaland — player 52

| Check | Expected | Actual |
|---|---|---|
| EPL 2025 appearances | 35 | 35 |
| Minutes | 2,958 | 2,958 |
| Goals | 27 | 27 |
| Shots / on target | 102 / 59 | 102 / 59 |
| Starts | 34 | 34; all 35 appearances use the substitute flag |
| 1+ on target | 29/35 | 29/35, unknown 0 |
| 2+ on target | 16/35 | 16/35, unknown 0 |
| 3+ shots | 18/35 | 18/35, unknown 0 |
| Scored | 20/35 | 20/35, unknown 0 |
| Assisted | 7/35 | 7/35, unknown 0; 8 total assists |
| Booked | 2/35 | 2/35, unknown 0 |
| Club | Manchester City | Manchester City, database ID 13 |
| Position / role | F / Forward; regular starter | F / Forward; `regular_starter` |
| Average minutes | About 85 | 84.5143 |
| Headline metric / rate | Goals per 90 / 0.82 | `goals_per_90` / 0.8215010142; display `0.82` |
| Headline comparison | 99th percentile; about 58 peers | 99; rank 1; **61 peers** |
| Missing-observations note | None | None |
| EPL 2026 appearances / minutes | 5 / 450 | 5 / 450 |
| EPL 2026 goals / shots / on target | 5 / 19 / 10 | 5 / 19 / 10 |
| EPL 2026 ranking | Unranked, previous 2025 ranks | `below_minutes_threshold`, 2025 ranks included |
| Available scopes | EPL 2026/2025 and UCL seasons | 12 scopes, newest first; includes EPL/UCL 2026 and 2025 |

The 61-player population includes Anthony Elanga (1,338 minutes), J. Bruun Larsen (1,038) and Wilson Odobert (990). Each has a most-common F position and no explicitly recorded goal count. The new rule makes their goals zero, so they qualify for the goals comparison. Excluding them would recreate the previous missing-count bias. Haaland remains at 99 and rank 1. See the independent raw-row arithmetic in [source-audit.json](source-audit.json).

Actual headline: “0.82 goals per 90 minutes, more than 99% of Premier League forwards”.

### Arsenal — team 7, EPL 2025

| Check | Expected | Actual |
|---|---|---|
| Finished matches | 38 | 38 |
| Won / drawn / lost | 26 / 7 / 5 | 26 / 7 / 5 |
| Goals for / against | 71 / 27 | 71 / 27 |
| Clean sheets | 19 | 19 |
| Stats-observed matches | 37 | 37 |
| Excluded fixture | 6910 | 6910 |
| Cards per match | About 1.351 | 50 / 37 = 1.3513513514 |
| 4+ match cards | 18/37 | 18/37; unknown 1 |
| 10+ match corners | 16/37 | 16/37; unknown 1 |
| Goals-against rank | 1 of 20 | 1 of 20 |
| Headline metric | Goals against | `goals_against`, 27 / 38 = 0.7105263158 |
| Rank phrase | “the fewest of 20 teams” | “the fewest of 20 teams” |
| Short code | ARS | ARS |
| xG-against rank, additional check | Fewest | 1 of 20; 27.49 / 37 = 0.7429729730 |

Actual headline: “0.71 goals conceded per match, the fewest of 20 teams”. Result-based frequencies use 38 matches: over 2.5 goals 20/38, both scored 18/38, clean sheet 19/38. The two observed matches with null opponent corners count those corners as zero.

### Anthony Taylor — 2025, all competitions

| Check | Expected | Actual |
|---|---|---|
| Alias merge | A. Taylor + Anthony Taylor, England | Both use `a-taylor-68af826d` |
| Finished / observed / excluded | 40 / 39 / 1 | 40 / 39 / 1; excluded fixture 6907 |
| Cards per match | About 4.256 | 166 / 39 = 4.2564102564 |
| Fouls per match | About 21.03 | 820 / 39 = 21.0256410256 |
| 4+ cards | 25/39 | 25/39; unknown 1 |
| 5+ cards | 19/39 | 19/39; unknown 1 |
| A red shown | 4/39 | 4/39; unknown 1 |
| Country | England | England |
| EPL / UCL observed matches | Included | 30 / 5; neither small-sample |
| UEL observed matches | 3, small-sample | 3, `small_sample: true` |
| UECL observed matches | 1, small-sample | 1, `small_sample: true` |
| Headline | Cards or fouls; more/fewer comparison | Fouls; fewer than 85% of 120 eligible referees |

Actual headline: “21.03 fouls per match, fewer than 85% of referees. Some match observations are missing.” The final sentence refers to the genuinely excluded match, not null counts within the 39 observed matches.

### Search and rebuilt rank coverage

| Check | Expected | Actual |
|---|---|---|
| `players&q=Haaland` | Erling first, Manchester City | Erling ID 52 first, Manchester City, Forward, EPL 2026; M. Haaland ID 24671 second, Brann, Midfielder, UEL 2025 |
| `teams&q=Arsenal` | Arsenal first, England, EPL | Arsenal ID 7 first, England, EPL 2026; Arsenal Tivat and Arsenal Tula follow |
| Referee cards versus yellows rank rows | Roughly equal after rebuild | **972 cards / 972 yellows** |
| Other referee metric rank rows | Available with observed matches | Reds 972, fouls 972, cards per foul 972 |

These are subject-season rank rows across the full build, not 972 distinct people. Eligibility now uses at least 15 stats-observed matches, rather than merely finished fixtures with existing rows.

## Tests

Commands were run as plain pytest, without an alarm/timeout helper:

```sh
.venv/bin/python -m pytest -q Scripts/tests/data_platform
node Scripts/tests/test_phase5_communication_web.js
```

Final full data-platform suite: **1,109 passed, 2 skipped, 2 failed, 78 errors**, in 41.85 seconds. All 45 Research profile cases pass. The only non-passing cases are the explicitly acknowledged pre-existing problems:

- 78 benchmark setup errors because `xgboost` is unavailable.
- `test_team_components_delegate_one_native_count_estimator_and_reject_baseline_inputs[team_catboost]` fails because `catboost` is unavailable.
- `test_tracking_contract.py::test_published_cohort_filters_and_quote_versions_are_auditable` fails its date-relative published-cohort assertion.

The earlier focused Research run passed all 45 cases in 2.83 seconds. The requested JavaScript test passed its visible selection, exact quote, escaping, identity, expiry and structured-rendering checks. `git diff --check` passed. No dependencies were installed or changed.

Old-policy tests updated explicitly:

- `test_empty_player_row_is_unknown_despite_participation_metadata` became `test_played_player_is_observed_even_when_all_count_fields_are_null`: played appearances now have observed zero counts.
- `test_zero_is_populated_but_non_count_measurements_remain_nullable`: observation established on both sides; xG/possession remain null.
- `test_empty_team_side_is_independent_of_opponent_and_fixture_score`: fixture goals no longer establish statistical observation; the opposite side is gated too, while results survive.
- `test_opponent_join_direction_league_average_and_team_frequencies`: includes the missing-side finished fixture in results and unknown statistical coverage.
- `test_referee_pair_requirement_cross_competition_and_previous_season`: retains finished fixtures with a missing side for exclusion accounting.
- `test_queries_charts_peer_builds_and_frequencies_share_count_policy`: played-player empty counts become zero; empty team/referee match statistics remain unknown consistently across paths.
- `test_missing_build_returns_counts_without_manufacturing_ranks`: fixture-score results remain available without opponent statistics.

Added coverage includes role boundaries, rating and shooting denominators, exact team observation fields, score-only results, referee match thresholds after exclusions, headline priorities/rank phrases/banned words, actual-team scope and transfer handling, squad ties, scopes/early-season fallback, chart context, country/small samples, and search ordering/disambiguation before limits. Existing 404/422, read-only, rank freshness, atomic replacement/rollback and migration round-trip checks still pass.

## Build and latency

Full command: `.venv/bin/python -m Scripts.data_platform research-build`.

Full rebuild: **19.566 seconds**, 110 scopes, 264,138 metric-rank rows, 1,599 raw referee aliases. Build ID: `profiles-v2:9138da19bc5d47648103e55d4eadfeb0`. Interpretation: `observed-match-count-zero.v2`. The version marker withholds pre-fix ranks until rebuilding; no model/table change was needed.

Measurements below use FastAPI `TestClient` against the actual local SQLite database, including routing, service queries and JSON serialization. Each has a first request followed by 20 sequential samples. There is no response/result cache. The OS/database file cache was already warm from rebuilding and verification; these are not network or load-test timings.

| Endpoint | First request ms | Median ms | p95 ms | Maximum of 20 ms |
|---|---:|---:|---:|---:|
| Haaland EPL 2025 | 22.48 | 4.28 | 5.15 | 5.26 |
| Haaland EPL 2026, including prior ranks | 6.43 | 5.77 | 6.07 | 6.16 |
| Arsenal EPL 2025, including squad leaders | 21.92 | 8.34 | 8.68 | 9.00 |
| Taylor 2025, including all scopes | 28.30 | 6.60 | 8.82 | 27.46 |
| Haaland search | 8.64 | 7.28 | 8.08 | 8.32 |
| Arsenal search | 28.59 | 25.41 | 26.56 | 26.59 |
| Taylor search | 2.43 | 1.45 | 1.62 | 1.78 |

All measured profile requests satisfy 300 ms. The initial run also stayed under 300 ms for profiles (maximum first-request time 143.53 ms). It exposed an unnecessarily broad player-search plan, at 445 ms median. Filtering matching player IDs before the appearance window allowed the existing `(player_id, fixture_id)` index to be used; the final search timings are above. No new index or migration was necessary. See [verification.json](verification.json) and [build.json](build.json).

## Reconciliation and presentation notes

- The only numerical acceptance discrepancy is the approximate peer count, fully reconciled as 61 rather than 58. Raw source checks independently reproduce Arsenal and Taylor's observed/excluded counts and rates.
- Missing xG, possession, ratings, and excluded team-match statistics remain unknown by design. They were not repaired in the source database or represented as zero.
- A player chart's `shots_off_target` means **shots minus shots on target, including blocked shots**, because the player table does not supply a separate off-target count. This definition is explicit in `data_basis`; the UI should not imply it is a distinct provider off-target statistic.
- Cards remain yellow plus red, with `booked` meaning either, as stated in `data_basis`; squad bookings use yellow cards as requested. These are descriptive stored statistics, not a bookmaker settlement contract. AET/PEN data remain as stored, including extra time when present.
- Referee identity keeps the existing first-initial/surname merge behaviour. Country comes from a unique non-empty alias suffix; absent or conflicting suffixes return null. No country was inferred from league membership. Arsenal Tivat/Tula country fields remain null where the source has no country.
- `available_scopes.ranked` describes an available rank snapshot with matching sample counts and the current policy version. The selected profile additionally checks metric values and peer group before attaching ranks. Normal periodic rebuilds are still required after data updates.
- This verifies the backend and rebuilt local research tables. It does not claim that a frontend consumes the added fields or that a running website process was restarted. No PostgreSQL server or concurrent-load benchmark was available; query code remains portable SQLAlchemy without SQLite-specific SQL.
