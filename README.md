# Betting_RAG — Football Parlay/Bet Recommender (Terminal)

This repository builds a Premier League (EPL) data pipeline and a local RAG query tool that recommends bets via terminal prompts. It pulls match and player stats from API-Football, engineers features, normalizes those into documents, embeds them into Chroma, and then serves recommendations via a CLI.

Below is the restart guide so you can pick up quickly after a break.

## Quick Restart (EPL Only)
1. Confirm `.env` has valid keys:
   - `API-FOOTBALL-KEY`
   - `OPENAI_API_KEY`
2. Refresh EPL raw data:
   ```bash
   python Scripts/Premier_League/prem_player_stats_per_gw.py
   python Scripts/Premier_League/PL_stats_for_teams.py
   ```
3. Re-run EPL feature engineering:
   ```bash
   python Scripts/Premier_League/PL_feature_engineering.py
   ```
4. Build EPL player/team profiles:
   ```bash
   python Scripts/Premier_League/Prem_player_profile.py
   python Scripts/Premier_League/prem_team_profiles.py
   ```
5. Normalize to RAG docs, then embed into Chroma:
   ```bash
   python Scripts/rag_ingest/00_normalize_to_docs.py
   python Scripts/rag_ingest/01_embed_and_upsert.py
   ```
6. Query the RAG (interactive):
   ```bash
   python Scripts/rag_ingest/rag_cli.py --league EPL
   ```
7. Query the RAG (one-shot):
   ```bash
   python Scripts/rag_ingest/rag_cli.py --league EPL --once "Arsenal vs Chelsea props"
   ```

## Project Layout (What Matters Most)
- `Scripts/Premier_League/prem_player_stats_per_gw.py`
  - Pulls per-fixture player stats from API-Football.
  - Writes `Output/Prem_output/player_fixture_stats_2025.*` and totals.
- `Scripts/Premier_League/PL_stats_for_teams.py`
  - Pulls per-fixture team stats from API-Football.
  - Writes `Output/Prem_teams/team_fixture_stats_2025.*` and aggregates.
- `Scripts/Premier_League/PL_feature_engineering.py`
  - Merges player + team fixture data.
  - Engineers player features, team features, and player context features.
  - Writes `Output/Prem_feature_engineering/*engineered_features_2025.*`.
- `Scripts/Premier_League/Prem_player_profile.py`
  - Aggregates player fixture features into player profiles.
  - Writes `Output/Prem_feature_engineering/player_profiles_2025.*`.
- `Scripts/Premier_League/prem_team_profiles.py`
  - Aggregates team fixture features into team profiles.
  - Writes `Output/Prem_feature_engineering/team_profiles_2025.*`.
- `Scripts/rag_ingest/00_normalize_to_docs.py`
  - Normalizes the engineered outputs into RAG-ready documents.
  - Writes `Index/normalized_EPL_*.json` and `Index/variable_dictionary.txt`.
- `Scripts/rag_ingest/01_embed_and_upsert.py`
  - Embeds docs using OpenAI and upserts into `Index/chroma`.
- `Scripts/rag_ingest/rag_cli.py`
  - Terminal RAG assistant for bet recommendations.

## Environment Setup
1. Create `.env` at repo root (if missing):
   ```bash
   API-FOOTBALL-KEY=your_api_football_key
   OPENAI_API_KEY=your_openai_key
   ```
2. Ensure Python dependencies are available:
   - `pandas`, `requests`, `python-dotenv`, `chromadb`, `openai`, `numpy`

## EPL Season Configuration
- EPL scripts currently use:
  - `LEAGUE_ID = 39`
  - `SEASON = 2025`
- API-Football interprets `SEASON = 2025` as the **2025/26** season.
- If you ever need to shift seasons, update:
  - `Scripts/Premier_League/prem_player_stats_per_gw.py`
  - `Scripts/Premier_League/PL_stats_for_teams.py`
  - `Scripts/Premier_League/PL_feature_engineering.py`
  - `Scripts/Premier_League/Prem_player_profile.py`
  - `Scripts/Premier_League/prem_team_profiles.py`

## Data Outputs (EPL)
- Raw player fixtures:
  - `Output/Prem_output/player_fixture_stats_2025.json`
  - `Output/Prem_output/player_fixture_stats_2025.csv`
- Raw team fixtures:
  - `Output/Prem_teams/team_fixture_stats_2025.json`
  - `Output/Prem_teams/team_fixture_stats_2025.csv`
- Engineered features:
  - `Output/Prem_feature_engineering/player_engineered_features_2025.json`
  - `Output/Prem_feature_engineering/team_engineered_features_2025.json`
- Profiles:
  - `Output/Prem_feature_engineering/player_profiles_2025.jsonl`
  - `Output/Prem_feature_engineering/team_profiles_2025.jsonl`
- RAG normalized docs:
  - `Index/normalized_EPL_*.json`

## RAG Notes
- `Scripts/rag_ingest/rag_cli.py` expects the Chroma collection `football_top5`.
- If the CLI says the collection is missing, re-run:
  ```bash
  python Scripts/rag_ingest/01_embed_and_upsert.py
  ```
- To reset the collection:
  ```bash
  python Scripts/rag_ingest/01_embed_and_upsert.py --reset
  ```

## Common Troubleshooting
- **No fixtures returned**: API-FOOTBALL key is invalid or the season is wrong.
- **Empty merges in feature engineering**: team/fixture naming mismatches or missing fixture stats.
- **RAG has no results**: normalization or embedding not run, or season/league filters too strict.
- **Season label mismatch**: `Scripts/rag_ingest/00_normalize_to_docs.py` maps year to `YYYY/YY` using `year-1`. Adjust if needed.

## Next Leagues
The same pattern applies for La Liga, Serie A, Bundesliga, and Ligue 1. Look in:
- `Scripts/La_Liga`
- `Scripts/Seria_A`
- `Scripts/Bundesliga`
- `Scripts/Ligue_1`

<<<<<<< HEAD
=======
Scripts for each league:
1. Run {league}_player_stats_per_gw.py
   - Feeds into feature engineering for each league. 
3. Run {league}_stats_for_teams.py
   - Feeds into feature engineering for each league. 
4. Run {league}_feature_engineering.py
>>>>>>> 393f70fc8680f60dc0386f16cbb47f74fd6477e1
