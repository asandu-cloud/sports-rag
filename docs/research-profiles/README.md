# Research profiles backend, 8 October 2026

Implemented database-backed player, team and referee profiles. Updated on 8 October 2026 to apply the user-specified count-null rule: empty counts in populated rows mean zero; wholly empty stats rows remain unknown. Earlier results are retained in `before-count-null-policy/`. No frontend, prediction, odds, Match Read, selection, provider archive or model artifact changes. Nothing committed. Existing unrelated changes were preserved. Static files were SHA-256 checked against the start of this task and are unchanged. The running website was not restarted; the new router was tested in isolation against the real database.

## Files

Added:

- `Scripts/data_platform/services/research_profiles.py`: pure calculation functions, profile orchestration, scoped rank builds.
- `Scripts/data_platform/repositories/research.py`: portable SQLAlchemy queries and transactional replacement.
- `Scripts/data_platform/research_stats.py`: shared research-only count normalization and row-availability rules.
- `Scripts/data_platform/models/research.py`: peer statistics, scope build provenance and referee aliases.
- `Scripts/data_platform/alembic/versions/0009_research_profiles.py`: new tables and lookup indexes.
- `Scripts/web_app/routers/research.py`: four read API routes and parameter validation.
- `Scripts/referee_names.py`: extracted legacy normalization and deduplication functions.
- `Scripts/tests/data_platform/test_research_profiles.py`: 26 isolated tests.
- This report, four complete real profile responses, a search response, timings and build verification under `docs/research-profiles/`.

Changed:

- `Scripts/data_platform/cli.py`: `research-build` registration and command.
- `Scripts/data_platform/models/__init__.py`: register the three research models.
- `Scripts/data_platform/models/fixtures.py`: declare the same lookup indexes as migration 0009.
- `Scripts/rag_ingest/referee_data.py`: import the extracted helpers. Both function ASTs were compared with HEAD and are unchanged.
- `Scripts/web_app/api.py`: mount the research router alongside Match Reads.

## Operation and storage

```bash
.venv/bin/python -m Scripts.data_platform init-db
.venv/bin/python -m Scripts.data_platform research-build
.venv/bin/python -m Scripts.data_platform research-build --season 2025 --competition EPL
```

Migration 0009 was applied to local `Index/platform.db`. It only adds research tables and indexes. No earlier migration was modified. Run the migration before starting an application process with this router, and run a build before expecting referee discovery or peer ranks. Rebuild after canonical data updates; no schedule or provider call was added.

`research_peer_stats` contains subject type/key, competition and season IDs, season year, peer group, metric, value, percentile, rank, peer count, sample minutes/matches, league average, build ID and timestamp. `research_profile_builds` records scope provenance even when nobody meets a threshold. `research_referee_aliases` supports indexed discovery and joins without normalizing every historical fixture on each request.

A full build replaces every stored competition-season scope and every cross-competition referee year. A competition-filtered build replaces that competition's player/team scopes and refreshes referee ranks across **all competitions** for its affected years. All replacements, including aliases, commit in one transaction. Errors roll back the replacement. Unrelated rank scopes remain intact. Builds are intended to run serially; no distributed job scheduler was introduced.

Referee keys use the legacy first-initial/last-name signature plus a short deterministic hash. Thus `A. Taylor` and `Anthony Taylor, England` both use `a-taylor-68af826d`, which remains stable when longer display names arrive. Country differences do not change the inherited identity grouping. This is a heuristic, not a provider-certified identity system; different officials sharing the signature can still be merged.

## API contract

Routes:

| Route | Selection |
| --- | --- |
| `GET /api/research/players/{player_id}?competition=EPL&season=2025` | Database player ID; optional competition and starting year |
| `GET /api/research/teams/{team_id}?competition=EPL&season=2025` | Database team ID; optional competition and starting year |
| `GET /api/research/referees/{referee_key}?season=2025` | All competitions in the selected starting year |
| `GET /api/research/search?type=players&q=Haaland&competition=EPL&limit=20` | `players`, `teams` or `referees`; literal case-insensitive substring |

Player/team defaults use the most recent season **with finished subject data**, then most minutes/matches, then competition code to break ties. They never use `is_current`. Referees default to their latest qualifying-data year. Search defaults to 20 results, maximum 100, ordered by name then identity. Results distinguish duplicate player/team names by database ID. Competition codes are exact catalogue codes. Seasons must be integer starting years in 1900–2100. Blank search, invalid type/ID/year/limit and unknown competition codes return 422. Unknown subjects or valid scopes without subject data return 404. Reads perform no inserts, builds or provider requests.

All profile responses use this envelope; objects below are shape descriptions, not another real response:

```text
{
  schema_version: "research-profile.v1",
  subject: {type, id/name/api_football_id/... OR referee_key/name/aliases},
  competition: {id, code, name} | null,
  season: {year, label, id: integer | null},
  totals: {...},
  metrics: {metric_name: Metric},
  ranking: {ranked, reason, peer_group, sample_minutes, sample_matches,
            minimum_minutes, minimum_matches},
  frequencies: {threshold_name: {count, denominator, unknown}},
  last_10: [{fixture_id, date, status, home:{id,name}, away:{id,name},
             score:{home,away}, stats_available:{...}, venue/opponent OR competition, values:{...}}],
  headline: {subject_type, metric, value, percentile, rank, peer_count,
             partial_observations, text} | null,
  data_basis: {source, matches, minutes, competition, season, finished_statuses,
               match_scope, generated_at, build_id, ranks_built_at,
               null_policy, count_null_policy_version, stats_rows, frequency_scope, cards_definition, ...},
  previous_season: {season, competition, ranking, metrics, data_basis} | null
}

Metric = {
  value: number | null,
  unit: "per_90" | "per_match" | "percent" | "rating" | "cards_per_foul",
  higher_is_more: boolean,
  coverage: {observed_matches, missing_matches},
  basis: string,
  rank: integer | null, percentile: integer | null,
  peer_count: integer | null, league_average: number | null,
  numerator/denominator: numbers for ratio metrics only
}
```

The profile-specific shapes are:

- **Player:** totals include appearances, minutes, starts, goals, assists, shots, shots on target, fouls won/committed, yellow/red/total cards. Metrics are `goals_per_90`, `assists_per_90`, `shots_per_90`, `shots_on_target_per_90`, `fouls_won_per_90`, `fouls_committed_per_90`, `cards_per_90`, yellow/red card rates, `duel_win_pct`, `pass_accuracy_pct`, `average_rating`. Adds `starts_basis` with method and counts of flag/fallback observations. Chart values are minutes, shots, shots on target, goals and yellows. Frequencies: `shots_on_target_1_plus`, `shots_on_target_2_plus`, `shots_3_plus`, `scored`, `assisted`, `booked`.
- **Team:** totals include matches, W/D/L, unknown results, clean sheets and recorded sums for each count metric. Metrics: `goals_for`, `goals_against`, `xg_for`, `xg_against`, `shots_on_target_for`, `shots_on_target_faced`, `corners_for`, `corners_against`, `possession`, `cards`. Adds `last_5_form` and `home_away.home/away`, each with match counts, totals, metrics and frequencies. Chart values are goals, xG and corners for/against plus own cards. Frequencies: `over_2_5_goals`, `both_teams_scored`, `clean_sheet`, `corners_10_plus`, `cards_4_plus`; goals/corners/cards threshold totals combine both teams, while clean sheet is the subject team's.
- **Referee:** totals and metrics cover cards, yellows, reds and fouls; metrics also include `cards_per_foul`. Adds `competition_breakdown`, an array of competition code plus that competition's counts, metrics and frequencies. Chart rows identify both fixture participants, score and competition, with cards/yellows/reds/fouls values. Frequencies: `cards_4_plus`, `cards_5_plus`, `red_card_shown`, `fouls_25_plus`.

Search shape:

```json
{
  "schema_version": "research-search.v1",
  "type": "players",
  "query": "Haaland",
  "results": [{"id": 52, "name": "Erling Haaland", "api_football_id": 1100}],
  "count": 1,
  "data_basis": {
    "source": "canonical_database", "competition": "EPL",
    "season": null, "matches": null, "minutes": null, "build_id": null,
    "generated_at": "2026-10-08T09:30:38.126361+00:00",
    "scope": "subjects_with_finished_data; referee_aliases_from_latest_build"
  }
}
```

Team search items use the same ID/name/provider-ID shape; referee items are `{referee_key, name}`. Search is a catalogue operation and has no aggregate matches/minutes or rank build, hence those fields are null. Its real captured response is [search.json](search.json).

Complete real responses, without truncating the last ten or metrics:

- [Haaland, EPL 2025](haaland_2025.json)
- [Haaland, EPL 2026 including prior-season ranks](haaland_2026.json)
- [Arsenal, EPL 2025](arsenal_2025.json)
- [Anthony Taylor, 2025 across competitions](taylor_2025.json)

## Calculation details and limitations

Only FT/AET/PEN fixtures count. Player appearances require minutes > 0. AET/PEN observations retain the canonical stored values; this is descriptive historical research, not regulation-time bookmaker settlement. No bookmaker equivalence is implied. Player and referee rates use the requested thresholds of 900 minutes and 15 matches; no additional team minimum was specified, so teams with at least one finished observation can rank.

Player position is the modal valid G/D/M/F appearance position in that competition-season, with alphabetic tie-breaking. No master-player style tags or archetypes are used. Starter status uses the substitute flag when present, otherwise minutes >= 60. Pass/duel rates sum numerators and denominators after applying the count-null rule. A zero denominator remains undefined (null), not zero percent. Rating is the unweighted mean of recorded appearance ratings.

Percentiles use `round(100 * (below + .5 * equal) / peer_count)` with Python's half-to-even rounding. Rank is competition rank: tied values share a rank and leave gaps. Percentiles always ascend numerically. Rank 1 is highest production or lowest conceded production according to `higher_is_more`; it is not a universal quality score. Team league averages weight observed team-match values equally. Possession is converted from its stored fraction to percent exactly once.

All threshold frequencies cover the whole selected season, not just the last ten. Their denominator includes only observations needed to decide the threshold; unknown observations are counted separately. Cards mean yellows plus reds. `booked` means either a yellow or red, not a bookmaker-specific settlement definition. Last ten and last-five form are oldest to newest, with fixture ID as the date tie-breaker. Opponent statistics come from the opponent row of that exact fixture. W/D/L uses fixture scores; recorded statistical totals use team-stat rows.

**The active count-null rule is `populated-row-count-zero.v1`.** For a player who played, or a finished team observation, missing count fields are zero when that stats row contains at least one numeric statistic. Explicit zero is a populated value. A missing row or a row with all numeric stats absent stays unknown. Player IDs, team IDs, minutes, position, captain/substitute flags and the fixture scoreboard do not establish row availability. The canonical team-stats row's own `goals` value does count as a statistic, including zero. Therefore a team-stats row containing only goals is populated under this rule; its other empty counts become zero.

The repository reads all numeric stat columns to judge the whole row, including fields outside the displayed charts. Each team side is assessed independently. This prevents a populated opponent row from filling an entirely empty subject row, and prevents a referee's fixture score from filling absent team statistics. Totals, frequencies, charts, peer ranks and the previous-season view share the same normalization. Source rows are never modified.

Count fields include goals, assists, shots, passes attempted/completed, duels, tackles, interceptions, fouls and cards where present on that entity. Missing xG, goals prevented, possession, rating and recorded percentages remain nullable because they are measurements, not event counts. Undefined ratios also remain null. These are distinct from an unknown count in a wholly empty row.

`data_basis.count_null_policy_version` identifies the rule. `data_basis.stats_rows` reports populated/empty row counts and the number of count fields filled with zero for the subject/opponent or referee's home/away sides. Each chart entry includes `stats_available`. Metric coverage reflects availability **after** normalization; it does not claim that every zero was explicitly recorded by the provider.

Player per-90 count rates still divide summed counts by all appearance minutes, as specified. Whole empty rows remain excluded from frequency denominators, with their unknown counts exposed. Haaland's scoring frequency is now **20/35 with zero unknowns**, rather than 20/20 plus 15 unknown observations. All-zero scorers with populated rows now enter the eligible peer cohort. This is the requested research interpretation, not a claim that provider evidence independently certified every converted zero.

The headline chooses the greatest distance from percentile 50, ties by its declared priority list. It is descriptive, uses only built ranks and adds a missing-observation caveat when applicable. No model or LLM writes it. Prior-season ranks appear for players/referees below their sample threshold, only for the immediately preceding year and the same competition (or the same all-competition referee scope). Current values are never labelled with previous-season ranks.

Ranks are snapshots, labelled by build ID/time. New build IDs carry the `counts-v1:` prefix, and old-policy ranks are withheld as `rank_policy_changed_rebuild_required` until rebuilt, even if a subject's numerical value happens to match. A changed subject sample, position or metric causes ranks to be withheld as `ranks_stale_or_missing`; a read never rebuilds them. Changes to other peers are reflected at the next build, not recomputed on every request. Unknown position, insufficient sample, absent build or no recorded metrics also have explicit reasons.

## Acceptance results and data differences

| Subject | Database-backed result |
| --- | --- |
| Erling Haaland, database ID 52 / provider ID 1100, EPL 2025 | **35 appearances, 2,958 minutes, 27 goals, 102 shots, 59 on target**. Goals/90 = **0.8215010142**, rank **1**, percentile **99**, **61** goal-rate peers. |
| Haaland, EPL 2026 | **5 appearances, 450 minutes, 5 goals, 19 shots, 10 on target**. `ranked:false`, `below_minutes_threshold`, with **2025/26** ranks included. |
| Arsenal, database ID 7, EPL 2025 | **38 matches**, **26 W / 7 D / 5 L**, **71–27 goals**, **19 clean sheets**. Fewest goals conceded, rank **1/20**; fewest xG conceded per observed match, rank **1/20**. |
| Anthony Taylor, `a-taylor-68af826d`, 2025 | **40** finished fixtures with both team rows: **37** under `A. Taylor`, **3** under `Anthony Taylor, England`. Both names resolve to one key. |

Reconciliation details under the updated rule:

- All **61** EPL 2025 forwards meeting 900 minutes have usable goal rates. The three formerly all-null goal histories (J. Bruun Larsen, Anthony Elanga and Wilson Odobert) now contribute zero goal rates because their stats rows are populated. Haaland remains rank 1 and percentile 99. This differs from the approximate 59 peers in the initial acceptance estimate.
- Haaland's requested totals are unchanged. Goals, shots and shots-on-target coverage is now **35/35**. He scored in **20/35**, recorded 1+ shots on target in **29/35**, and 2+ in **16/35** appearances.
- Arsenal still has **38 matches**, **71–27 goals** and **19 clean sheets**. It now has **50 cards**, **1.315789474 per match**, with full count coverage under this rule. The final 24 May 2026 match is a **2–1 win away to Crystal Palace** and accounts for the extra win/goals versus the original 37-match estimate. The team-stat goals fields make that observation populated for counts. xG remains absent there, so xG averages continue to use **37 observations**.
- Taylor still has **40** qualifying fixtures. Under the new rule, totals are **166 cards** (162 yellows + 4 reds) and **820 fouls**, or **4.15 cards** and **20.5 fouls per match**. Cards reached 4+ in **25/40**, 5+ in **19/40**, a red appeared in **4/40**, and fouls reached 25+ in **9/40**. The final fixture's team-stat goals fields make the rows populated even though other counts are blank.
- Canonical metadata such as `zero_policy: legacy_zero_is_unknown` is preserved. The research response now explicitly applies the requested interpretation over those canonical values. No ingestion change, historical source rewrite or prediction-engine change was made.

## Validation and performance

The required plain command was run, without any SIGALRM/timeout wrapper:

```bash
.venv/bin/python -m pytest -q Scripts/tests/data_platform
```

Final full suite: **1,090 passed, 2 skipped, 2 failed, 78 setup errors**, in **40.70 seconds**. Both failures and all errors are the declared unrelated cases:

- 78 benchmark setup errors: unavailable `xgboost`.
- One `team_catboost` benchmark failure: unavailable `catboost`.
- `test_tracking_contract.py::test_published_cohort_filters_and_quote_versions_are_auditable`: date-relative fixture.

An initial sandbox run also failed an existing Poisson test because CPU discovery was restricted. The test and full suite were rerun outside the sandbox; it passed without code changes. All other tests are green. All **26 research tests pass** (latest targeted run 3.01 seconds), including percentiles/ties, thresholds, starts, sum-over-sum ratios, nulls, opponent joins, aliases, early-season fallback, default scope choice, frequencies, headlines, stale ranks, read-only HTTP and 404/422, scoped idempotency and failed-build rollback. Added checks cover populated-row count nulls, all-zero rows, metadata-only empty rows, missing/non-count measurements, unplayed/unfinished exclusions, independent team-side availability, consistent charts/ranks/frequencies, preservation of canonical values and rejection of old-policy ranks. Migration upgrade/downgrade/re-upgrade was tested on a private database. Four warnings come from the existing Alembic `path_separator` configuration.

The new migration also compiles to PostgreSQL DDL. Queries use portable SQLAlchemy constructs; a live PostgreSQL server was not available for runtime validation.

Real local database measurements, isolated FastAPI TestClient, one first request plus 20 repeats, including JSON serialization but excluding TCP transport:

| Endpoint | First request | Median repeat | P95 repeat |
| --- | ---: | ---: | ---: |
| Haaland 2025 | 49.51 ms | 4.91 ms | 6.41 ms |
| Haaland 2026 + previous ranks | 7.86 ms | 6.90 ms | 7.83 ms |
| Arsenal 2025 | 12.64 ms | 5.51 ms | 6.43 ms |
| Taylor 2025 | 8.27 ms | 4.94 ms | 7.52 ms |
| Player search | 50.79 ms | 8.29 ms | 14.01 ms |

Default-season profiles and team/referee searches also passed the 300 ms target; all measurements are in [timings.json](timings.json). No in-memory profile cache was introduced. New indexes cover `(player_id, fixture_id)` and `(referee, season_id)`.

Full build under the updated count rule: **19.735 seconds**, **110 scopes**, **237,736 peer rows** and **1,599 referee aliases**. The final rank-content checksum and policy version are saved in [build-verification.json](build-verification.json). Scoped repeat-build and failed-build rollback tests pass. Earlier policy timings/results remain archived in `before-count-null-policy/`.
