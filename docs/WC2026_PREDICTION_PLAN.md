# World Cup 2026 Prediction Engine

## Context
The existing prediction engine covers 8 European club competitions with a 65% backtested hit rate. The 2026 FIFA World Cup (June 11 - July 19, 48 teams, 16 venues across USA/Mexico/Canada) requires a fundamentally different approach because: (1) international match data is sparse, (2) player-club context doesn't transfer cleanly to national teams, (3) tournament dynamics produce more upsets than league football, (4) alternative contextual data (weather, altitude, fatigue, travel) has outsized impact in this specific tournament.

## Prediction Stack: Three Signals Blended

```
Signal 1: Elo ratings (40% weight)
    → Overall team strength, handles sparse data gracefully
    → Source: eloratings.net (free) or self-computed from Mart Jurisoo dataset

Signal 2: Market odds (35% weight)
    → Aggregated market intelligence, incorporates info we don't have
    → Source: Odds API (existing subscription)

Signal 3: Our model (25% weight)
    → Stat-specific decomposition (goals, corners, cards, SoT separately)
    → Sources: player-aggregated profiles + referee data + tournament modifiers + contextual factors

For sparse-coverage teams (Saudi Arabia, etc.): 20% Elo / 60% market / 20% model
```

## Data Sources (All Free — $0 additional)

### Already have
- API-Football (WC fixtures, squads, qualifying data, odds, match stats)
- Odds API (bookmaker odds for WC markets)
- Chroma DB (player profiles for 5 domestic leagues + 3 European comps)

### Free sources to add
- **eloratings.net** — national team Elo ratings (scrape)
- **Mart Jurisoo dataset** (GitHub) — 49,000+ international results since 1872 (CSV download)
- **FBref.com** — international xG data for qualifiers (scrape, rate-limited)
- **StatsBomb open data** (GitHub) — event-level data for 2018 + 2022 World Cups
- **Transfermarkt** — squad market values as quality proxy (scrape)
- **OpenWeatherMap API** — match-day temperature, humidity, wind (free tier: 1000 calls/day)

## Alternative/Contextual Data Edge

These factors are individually small (1-3% each) but compound in specific scenarios. Bookmakers don't systematically price all of them simultaneously — that's where the edge lives.

### Weather (HIGH impact for 2026 specifically)
- **Source:** OpenWeatherMap API (free tier)
- **What it affects:** Houston/Miami/Dallas in June = 35°C + 80% humidity. European pressing teams physically deteriorate in 2nd half → more fouls → more cards, fewer SoT, lower pressing intensity
- **Implementation:** Fetch 48hr forecast before each match, apply stat-specific modifiers:
  - Temperature >30°C: goals -5%, cards +10%, SoT -8% (2nd half effect)
  - Rain: cards +8-12%, corners from crosses +5%, passing accuracy down
  - Wind >25km/h: set piece quality down, fewer corners from sustained possession
- **Edge:** Bookmakers adjust moneyline for weather but rarely adjust prop markets (cards, corners)

### Altitude (HIGH impact for Mexico venues)
- **Source:** Static lookup table (16 venues)
- **What it affects:** Estadio Azteca at 2,240m, Guadalajara at 1,566m. Sea-level teams (England, Netherlands, Japan) lose 15-20% aerobic capacity at altitude in first exposure
- **Implementation:** Modifier based on team's home altitude vs venue altitude:
  - Altitude difference >1500m: goals -8%, cards +12%, SoT -10%
  - Altitude difference >1000m: goals -4%, cards +6%, SoT -5%
  - Teams with altitude experience (Bolivia, Ecuador, Colombia, Mexico): no penalty
- **Edge:** Well-documented in sports science, inconsistently priced in betting markets

### Rest Days + Travel Distance (HIGH impact for 48-team format)
- **Source:** Tournament schedule (published) + Google Maps API (free tier)
- **What it affects:** Some teams play 3 group matches in 9 days with cross-country flights (NYC→LA = 4,500km). Others get 4 days rest in the same city.
- **Implementation:**
  - 2 rest days + >2000km travel: cards +8%, SoT -6%, goals -5%
  - 3 rest days + <500km travel: no penalty (full recovery)
  - Extra time in previous match: treat as -0.5 rest days
  - Cumulative tournament fatigue: -2% per match after match 3
- **Edge:** Schedule asymmetry is known before the tournament. We can pre-compute fatigue advantages for every possible knockout matchup.

### Player Cumulative Season Minutes (MEDIUM-HIGH)
- **Source:** Already in Chroma (domestic league data)
- **What it affects:** Players who played 50+ club matches + UCL deep run arrive depleted. Squads where starters' seasons ended in April are fresher.
- **Implementation:** Compute total minutes played in 2025/26 season for each squad's expected starters. Flag squads with >80th percentile cumulative load as "fatigue risk."
- **Edge:** Bookmakers know about individual star players' form but don't systematically model squad-wide fatigue loads.

### Squad Depth Quality Gap (MEDIUM)
- **Source:** Compute from existing player profiles
- **What it affects:** When managers rotate (dead rubbers, short recovery), the gap between XI and bench matters. France's bench ≈ other teams' starting XI. Saudi Arabia's bench is a cliff.
- **Implementation:** Compute "first XI composite rating" vs "second XI composite rating" from player club stats. Depth ratio = second/first. Apply when rotation is expected.
- **Edge:** Market prices the first XI matchup. Squad depth is a secondary factor bookmakers handle inconsistently.

### Set Piece Specialist Presence (MEDIUM)
- **Source:** FBref (free kick/corner delivery stats) + player profiles
- **What it affects:** WC goals from set pieces = ~35% (vs ~25% domestic). Teams with elite dead-ball specialists over-index because open-play chemistry is reduced but set piece routines can be drilled in limited training time.
- **Implementation:** Flag teams with top-tier set piece takers. Apply +5-8% goals modifier for strong set piece squads.
- **Edge:** Set piece quality is poorly captured in team-level aggregates.

### Scenario: Compounding Example
England vs Mexico at Estadio Azteca, altitude 2,240m, 32°C, England played extra time 3 days ago and flew from New York, referee is strict (5.1 cards/match avg):

| Factor | Cards modifier |
|--------|---------------|
| Base international modifier | +20% |
| Altitude (sea-level team) | +12% |
| Temperature >30°C | +10% |
| Rest days (2) + travel (3500km) | +8% |
| Strict referee | +15% (from referee module) |
| **Cumulative** | **~+65% on base card projection** |

## Tournament-Specific Adjustments

### International football modifier (applied to ALL WC matches)
- Goals: ×0.88 (WC avg ~2.5 vs ~2.8 domestic)
- Corners: ×0.85
- Cards: ×1.20
- SoT: ×0.82

### Phase modifiers
- Group MD1: cautious (goals ×0.92, cards ×1.10)
- Group MD3: depends on scenario (dead rubber vs must-win)
- Round of 32: goals ×0.88, cards ×1.15
- Quarterfinal+: goals ×0.82, cards ×1.25

### Underdog adjustment
- Compress probabilities toward 50/50 for ranking gaps >15
- Hard cap: never output >85% win probability
- Draw floor: minimum 15% draw probability in group stage
- Penalty shootouts: model as ~50/50 regardless of team quality

### Group stage game theory
- Both qualified: rotation expected, depth matters, moderate intensity
- Both eliminated: avoid betting, unpredictable
- One must-win: classic upset territory if must-win team is underdog
- Both must-win: favors the better team

## Markets Priority

| Market | Priority | Why |
|--------|----------|-----|
| **Cards O/U** | PRIMARY | Referee data is our unique edge + all contextual factors compound on cards |
| **Goals O/U / BTTS** | SECONDARY | Dixon-Coles with Elo priors, moderate edge |
| **Moneyline** | SELECTIVE ONLY | Only bet when model disagrees with market by >5% |
| **Corners** | LOW PRIORITY | Signal too weak internationally |

## Files to Create

| File | Purpose |
|------|---------|
| `Scripts/world_cup/wc_elo.py` | National team Elo model |
| `Scripts/world_cup/wc_profiles.py` | Synthetic national team profiles from player club data |
| `Scripts/world_cup/wc_modifiers.py` | All contextual modifiers: weather, altitude, fatigue, travel, phase, underdog |
| `Scripts/world_cup/wc_referee.py` | FIFA referee panel profiles |
| `Scripts/world_cup/wc_data.py` | Data collection: squads, qualifying stats, FBref xG scraping |
| `Scripts/world_cup/wc_simulator.py` | Monte Carlo tournament simulation for futures markets |
| `Scripts/world_cup/wc_weather.py` | OpenWeatherMap integration for match-day conditions |

## Files to Modify

| File | Changes |
|------|---------|
| `Scripts/rag_ingest/rag_cli_v2.py` | Add WC2026 dispatch in `_profile_meta()` and `_recent_stats()` (~30 lines) |
| `Scripts/rag_ingest/core/weights.py` | Add `SCORING_WEIGHTS["wc"]` sub-dict with all modifiers |
| `Scripts/discord_bot/config.py` | Add WC2026 to league lists |

## Implementation Phases

### Phase 1: Foundation (April 2026)
- Download Mart Jurisoo dataset, build Elo model
- Collect historical WC data from API-Football (2014, 2018, 2022)
- Scrape FBref for qualifying xG data
- Build Dixon-Coles goal model with Elo priors
- Extend referee_data.py for FIFA referee panel
- Build venue lookup table (altitude, capacity, country)
- Integrate OpenWeatherMap API

### Phase 2: Squad Data (May 2026)
- Ingest preliminary + final squad lists from API-Football
- Build synthetic profiles for all 48 teams
- Compute squad depth scores, cohesion matrices, cumulative minutes
- Curate manager tactical profiles (manual, 48 entries)
- Build travel distance lookup between all 16 venues
- Validate against bookmaker odds

### Phase 3: Pre-Tournament (June 1-10, 2026)
- Process final friendlies
- Full end-to-end testing of all markets
- Build Monte Carlo tournament simulator
- Deploy to Discord with WC data quality badges
- Pre-compute fatigue/travel advantages for all group matches

### Phase 4: Live Operations (June 11 - July 19)
- Post-matchday profile updates
- Fetch weather 48hrs before each match
- Recalibrate phase modifiers after group MD1
- Bayesian update team strength after each match
- Track model accuracy vs domestic baseline
- Shift season/recent blend toward tournament data (85/15 → 55/45 by knockouts)

## Honest Expectations
- Cards O/U: 57-62% hit rate (our best WC market)
- Goals O/U / BTTS: 55-58%
- Moneyline (selected): 58-63% on bets we take
- Corners: 52-55% (low priority)
- Overall ROI on selected bets: 3-7% edge with proper bankroll management
- Domestic comparison: ~5-8% lower hit rates than domestic leagues across all markets
