# Betting RAG — Project Guide

## What This Is
A football betting assistant that uses RAG (Retrieval-Augmented Generation) over a Chroma vector database to score and recommend parlays, totals lines, handicap lines, and team comparisons. Covers the top 5 European leagues: EPL, LaLiga, SerieA, Bundesliga, Ligue1.

## Architecture Overview

### Data Pipeline
```
Output/*_feature_engineering/   (source: per-league engineered stats)
        │
        ▼
00_normalize_to_docs.py --all   (normalize → Index/normalized_*.json)
        │
        ▼
01_embed_and_upsert.py --all-leagues --reset   (embed with text-embedding-3-large → Chroma)
        │
        ▼
Chroma collection "football_top5"   (Index/chroma/, ~32k vectors)
        │
        ▼
rag_cli_v2.py / rag_streamlit.py   (query engine + Streamlit UI)
```

### Key Files

| File | Purpose |
|------|---------|
| `Scripts/rag_ingest/rag_cli_v2.py` | **Main scoring engine** (~3800 lines). Odds fetching, KB retrieval, `kb_leg_quality()` scoring, `select_parlay()` optimizer, comparisons, totals lines, evidence rendering, LLM explanation. |
| `Scripts/rag_ingest/rag_streamlit.py` | Streamlit UI. Template-based queries, league/date selectors, renders parlay cards. |
| `Scripts/rag_ingest/00_normalize_to_docs.py` | Normalization pipeline. Reads `*_feature_engineering/` dirs, produces 4 doc types (player_fixture, player_season, team_fixture, team_profile), computes opponent-join aggregations. |
| `Scripts/rag_ingest/01_embed_and_upsert.py` | Embeds normalized JSON docs and upserts into Chroma. |
| `Scripts/rag_ingest/chroma_backend.py` | Chroma client factory (persistent local backend). |
| `Scripts/rag_ingest/prob_models.py` | **Probabilistic primitives** — Poisson/negative binomial distributions, implied probability, value edge, EV, Kelly criterion. Pure functions, no external deps. |
| `Scripts/rag_ingest/referee_data.py` | **Referee data layer** — builds historical referee profiles from API-Football, live assignment lookup at query time, computes card projection modifiers. |
| `Scripts/rag_ingest/heuristics.py` | `top_markets()` heuristic — ranks market opportunities from team profile metadata. |
| `Scripts/rag_ingest/02_query_smoke.py` | Smoke test for Chroma retrieval. |

### Document Types in Chroma
- **team_profile** — season-level aggregates per team (goals_for_pm, corners_pm, cards_per_90_team, sot_for_pm, possession, dominance_index, control_index, etc.)
- **team_fixture** — per-match stats with opponent-joined fields (corners_against, sot_against, shots_against)
- **player_fixture** — per-match player stats
- **player_season** — season-level player aggregates

### Opponent-Join Aggregations (computed in 00_normalize)
These are season-level averages computed by joining each team's fixture rows with their opponent's rows:
- `corners_against_pm` — avg corners conceded per match (via `compute_corners_against_pm()`)
- `opp_cards_induced_pm` — avg cards opponents receive against this team (via `compute_cards_induced_pm()`)
- `corners_home_pm` / `corners_away_pm` — venue-split corner averages (via `compute_corners_home_away_pm()`)
- `corner_edge_pm` — corners_for - corners_against per match
- `cards_home_pm` / `cards_away_pm` — venue-split card averages (via `compute_cards_home_away_pm()`)
- `sot_against_pm` — avg shots on target conceded per match (via `compute_sot_against_pm()`)
- `corners_var`, `goals_var`, `cards_var`, `sot_var` — season-level sample variance per stat (via `compute_stat_variance()`), used to parameterize negative binomial distributions when variance > mean

## Scoring Engine (`rag_cli_v2.py`)

### Market Groups
Classified by `market_group_from_key()`: `moneyline`, `spreads`, `totals`, `corners`, `cards`, `sot`.

### Parlay Flow
1. `parse_constraints()` — extracts ConstraintSpec from user query (target odds, leg count, hard/soft group preferences)
2. `fetch_events()` — gets odds from The Odds API (default markets: h2h, totals, spreads)
3. `enrich_events_for_groups()` — per-event API discovery for missing market groups (corners, cards)
4. `build_candidates()` — creates CandidateLeg objects from all event odds
5. `kb_leg_quality()` — scores each leg using KB metadata (multi-signal composite per market group)
6. `select_parlay()` — optimizer: pool reduction → brute-force or beam-search → best combo by KB quality + odds proximity
7. `leg_evidence()` — generates human-readable evidence lines per leg
8. LLM explanation step (GPT) — synthesizes final recommendation text

### Scoring Blocks in `kb_leg_quality()`
- **Moneyline/Spreads**: form_index, control_index, dominance_index, SoT edge
- **Goals Totals**: pace_goals (goals only) + sot_nudge (directional, separated from offset comparison) + line_fit
- **Corners**: projected corners (venue-blended), recent form, possession asymmetry, style-clash (dominance x possession x shots), attack volume proxy, side edge
- **Cards**: cards_per_90 (venue-blended), fouls_per_90, cards_per_foul, aggression_index, opponent-induced cards, matchup adjustment, foul momentum, side edge
- **SoT**: projected_total_sot (season 65% + recent 35%), offset comparison, line_fit

### Comparison Workflow
Separate from parlays. Functions: `comparison_signals_cards()`, `comparison_signals_corners()`, `comparison_signals_sot()`. Each produces a composite score for "which team" questions.

### Totals Line Workflow
`projected_total_corners()`, `projected_total_cards()`, `projected_total_sot()` — blends season (65%) + recent (35%) projections. `projected_cards()` uses opponent induction + venue splits matching the corners/SoT pattern.

### Handicap Line Workflow
Standalone spread/handicap recommendation per fixture. Flow: `projected_goal_difference()` (season goals_for_pm diff + form/dominance nudge blended 65/35 with recent xG diff) → `extract_spread_line_options()` (pulls spread/handicap lines from bookmaker data) → `choose_best_spread_line()` (scores lines by edge = projected_diff + point, odds quality, proximity penalty) → `render_spreads_line_answer()` (per-fixture output with projection evidence, recommended line, edge, and confidence). Routed via `INTENT_SPREADS_LINE` (score 0.68).

### Probabilistic Modeling Layer
Converts deterministic point estimates into calibrated probabilities for better hit-rate and value detection.

**Module:** `prob_models.py` — pure Python, no external dependencies beyond stdlib.

**Distributions:**
- **Poisson** — default for count events (goals, corners, cards, SoT). `poisson_over_prob(lam, line)` → P(X > line).
- **Negative binomial** — automatic fallback when season-level variance exceeds the mean (over-dispersion). `over_prob(lam, line, variance)` selects distribution automatically.
- **Normal** — used for spread cover probability (goal difference is continuous). `_normal_cdf()` approximation.

**Value detection chain:**
1. `implied_prob(odds)` — extracts raw implied probability from decimal odds (1/odds)
2. `remove_vig_two_way(odds_a, odds_b)` — removes bookmaker vig, returns fair probabilities summing to 1.0
3. `value_edge(model_prob, implied)` — difference between model and bookmaker. Positive = value bet.
4. `expected_value(model_prob, odds)` — EV per unit staked. `(prob × odds) - 1`. Positive = profitable long-term.
5. `kelly_fraction(model_prob, odds)` — Kelly criterion optimal bet fraction.

**Integration points:**
- `choose_best_total_line()` — adds probability-aware scoring (`value_edge_weight`, `negative_ev_penalty`) and attaches `_model_prob`, `_implied_prob`, `_value_edge`, `_ev` to returned dict
- `choose_best_spread_line()` — uses normal distribution for cover probability, attaches same probability data
- `confidence_from_edge()` — upgraded with `model_prob` and `value_edge_pct` parameters for probability-calibrated confidence (high: ≥65% & ≥8% edge, medium: ≥55% & ≥3% edge)
- `render_totals_line_answer()` / `render_spreads_line_answer()` — display "Model: P(Over X) = Y% | Books imply: Z% | Value edge: ±W%" per fixture
- `kb_leg_quality()` — probability bonus for count-based markets: `(model_p - 0.50) × prob_quality_weight`
- `score_combo()` — joint probability and combo EV scoring for parlays
- `leg_evidence()` — probability evidence line appended for count-based markets

**Variance sources:**
- **On-the-fly:** `get_team_recent_variance()` computes sample variance from last 6-8 fixtures (immediate, no re-embed needed)
- **Season-level:** `compute_stat_variance()` in `00_normalize_to_docs.py` computes from all fixtures, stored in team_profile metadata as `corners_var`, `goals_var`, `cards_var`, `sot_var`

### Referee Data Layer
Separate from Chroma — referee profiles are too small (~100-200 entries) and too dynamic for vector search.

**Two-layer architecture:**
- **Static (weekly):** `Index/referee_profiles.json` — historical referee card/foul averages and strictness ratios. Built by cross-referencing API-Football fixture referee names with `team_engineered_features` card data (matched via `fixture_id`). Rebuild: `cd Scripts/rag_ingest && python referee_data.py --build`
- **Dynamic (query time):** Live API-Football call (`GET /fixtures?league={id}&date=YYYY-MM-DD`) fetches referee assignments for upcoming fixtures. Cached in-memory for 15 min.

**Modifier formula:** `multiplier = 1.0 + (strictness - 1.0) * confidence * weight`
- `strictness` = referee avg cards / league avg cards (>1.0 = strict, <1.0 = lenient)
- `confidence` = linear ramp from 0 at 5 matches to 1.0 at 20 matches
- `weight` = `SCORING_WEIGHTS["referee"]["modifier_weight"]` (default 0.25)

**Integration:** Applied multiplicatively to `projected_cards()`, `kb_leg_quality()` cards block, `comparison_signals_cards()`, and rendered in `leg_evidence()` and `render_totals_line_answer()`.

**Graceful degradation:** Every failure (no API key, API down, ref not announced, unknown ref, <5 matches) returns modifier=1.0. System works identically without referee data.

## Environment
- **APIs**: The Odds API (`ODDS-API` key), OpenAI (`OPENAI_API_KEY`), API-Football (`API-FOOTBALL-KEY`)
- **Vector DB**: Chroma (persistent, local at `Index/chroma/`)
- **LLM**: GPT for explanation text (model configurable via `RAG_CHAT_MODEL`)
- **Embedding**: `text-embedding-3-large`
- **Python**: 3.11+, dependencies in `requirements.txt`

## Common Operations

### Re-normalize all leagues
```bash
cd Scripts/rag_ingest
python 00_normalize_to_docs.py --all
```

### Re-embed and upsert (resets collection)
```bash
cd Scripts/rag_ingest
python 01_embed_and_upsert.py --all-leagues --reset
```

### Run Streamlit UI
```bash
cd Scripts/rag_ingest
streamlit run rag_streamlit.py
```

### Run CLI directly
```bash
cd Scripts/rag_ingest
python rag_cli_v2.py
```

## Weights
All scoring weights are centralized in `SCORING_WEIGHTS` dict at the top of `rag_cli_v2.py` (~line 91). Sub-dicts: `combo`, `kb_quality`, `totals_line`, `spreads_line`, `projection`, `projection_spreads`, `comparison_cards`, `comparison_corners`, `comparison_sot`, `projection_sot`, `projection_cards`, `confidence`, `referee`, `prob`, `pool`.

## Known Architecture Notes
- `contradictory()` prevents stacking 2 legs from the same market group in the same fixture
- Moneyline and spreads can never mix in the same parlay
- `is_full_game_market()` filters out half-time, lay, and player-prop markets
- All parlay events are always enriched with corners/cards/sot so the optimizer evaluates all market types
- SoT is a standalone market group with its own scoring block, evidence rendering, and constraint parsing
- `projected_sot()` blends own sot_for_pm (60%) with opponent sot_against_pm (40%) for projections
- `projected_cards()` blends own cards_per_90 (60%, venue-adjusted) with opponent opp_cards_induced_pm (40%) + aggression nudge
- `confidence_from_edge()` uses stat-specific thresholds: cards (0.45/0.20), corners (0.60/0.25), SoT (0.50/0.20), goals (generic 0.75/0.35)
- `INTENT_SPREADS_LINE` routes handicap/spread queries to standalone line advice (score 0.68, between comparison 0.7 and totals_line 0.65)
- Spread edge math: home side `edge = projected_diff + point`, away side `edge = -projected_diff + point`
- `prob_models.py` is pure stdlib (no scipy needed) — Poisson/NegBin via log-gamma, portable and fast for ranges 0-30
- Negative binomial auto-selected when `variance > mean` (over-dispersion), otherwise Poisson
- Spread cover probability uses normal distribution (goal difference is continuous), default `margin_std = 1.25`
- `prob` weights: `value_edge_weight` (2.0), `negative_ev_penalty` (3.0), `min_value_edge` (0.02), `min_model_prob` (0.45), `margin_std_default` (1.25), `prob_quality_weight` (1.5), `ev_combo_weight` (0.5)
- `referee_data.py` is kept separate from Chroma — profiles stored as `Index/referee_profiles.json`, loaded lazily once per session
- Referee modifier applied multiplicatively after matchup adjustment in all cards scoring paths
- `referee` weights: `modifier_weight` (0.25), `min_sample_size` (5), `full_confidence_sample` (20), `evidence_threshold` (0.02)
- Cross-reference uses `fixture_id` for exact matching between API-Football and team_engineered_features data
