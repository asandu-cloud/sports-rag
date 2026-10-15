# Frontend Architecture: Betting RAG Website

## 1. Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **Next.js 15 (App Router)** | SSR for SEO on match pages, ISR for pre-computed predictions, API routes as thin Python bridge, mobile-first routing |
| Styling | **Tailwind CSS 4 + shadcn/ui** | Rapid iteration, dark mode built-in, consistent component primitives, small bundle |
| State | **Zustand** (parlay slip) + **React Query** (server data) | Zustand is 1KB, persists to localStorage; React Query handles caching/stale-while-revalidate |
| Data strategy | **Hybrid: pre-computed JSON (ISR) + on-demand API** | Pre-compute fixture predictions every 15 min via cron; parlay builder hits live API |
| Backend bridge | **FastAPI wrapper** around `rag_cli_v2.py` functions | Thin HTTP layer; no rewrite of scoring engine; deployed same server |
| Hosting | **Vercel** (frontend) + **Railway/Fly.io** (FastAPI + Chroma) | Vercel for edge CDN + ISR; backend needs persistent disk for Chroma |

---

## 2. System Architecture

```
Browser (Next.js)
    |
    |-- Static/ISR pages (fixtures, match detail)
    |       served from Vercel edge cache, rebuilt every 15 min
    |
    |-- Client-side API calls (parlay builder, live odds)
    |       |
    |       v
    |   Next.js API Routes (/api/*)
    |       |
    |       v
    |   FastAPI Backend (same origin via proxy, or separate domain)
    |       |
    |       |-- rag_cli_v2.py functions (projections, scoring, odds)
    |       |-- Chroma (persistent local)
    |       |-- API-Football (rate-limited, cached)
    |       |-- API-Football (referee data, cached 15 min)
    |       |-- OpenAI (LLM explanations, on-demand)
    |
    v
localStorage (parlay slip state, recent views)
```

### Why not call rag_cli_v2.py directly from Next.js API routes?

Next.js API routes run in Node.js. The scoring engine is Python with Chroma, scikit-learn, and numpy dependencies. A Python-to-Node bridge (child_process, WASM) would be fragile and slow. A FastAPI wrapper is the cleanest separation: Python stays Python, HTTP is the contract.

---

## 3. FastAPI Backend (API Contract)

The FastAPI server is a thin wrapper. Each endpoint calls one or two existing `rag_cli_v2.py` functions and returns structured JSON instead of the text rendering the CLI produces.

### Base URL: `/api/v1`

### 3.1 Fixtures & Predictions

#### `GET /fixtures/{league}/{date}`
Returns all fixtures for a league on a given date with pre-computed predictions.

**Parameters:**
- `league`: `EPL | LaLiga | SerieA | Bundesliga | Ligue1 | UCL | UEL | UECL`
- `date`: `YYYY-MM-DD`

**Response shape:**
```typescript
interface FixtureListResponse {
  league: string;
  date: string;
  fixtures: FixtureSummary[];
  generated_at: string;  // ISO timestamp
}

interface FixtureSummary {
  event_id: string;
  home_team: string;
  away_team: string;
  kickoff: string;          // ISO datetime
  league: string;

  // Pre-computed headline predictions
  moneyline: {
    home_prob: number;       // 0.42
    draw_prob: number;       // 0.28
    away_prob: number;       // 0.30
    recommended: string;     // "Home" | "Draw" | "Away"
    confidence: string;      // "high" | "medium" | "low"
    best_odds: number | null;
    value_edge: number | null;
  };

  goals: {
    projected_total: number; // 2.67
    recommended_side: string; // "Over" | "Under"
    recommended_line: number; // 2.5
    model_prob: number;       // 0.623
    confidence: string;
    best_odds: number | null;
    value_edge: number | null;
  };

  btts: {
    prob_yes: number;         // 0.58
    recommended: string;      // "Yes" | "No"
    confidence: string;
    best_odds: number | null;
    value_edge: number | null;
  };

  // Signal strength (used for "top picks" sorting)
  max_value_edge: number;     // highest edge across all markets
  top_confidence: string;     // best confidence level across markets
}
```

**Backend implementation:**
```python
# fastapi_app.py — pseudo-code
@app.get("/api/v1/fixtures/{league}/{date}")
def get_fixtures(league: str, date: str):
    events, notes = rag.fetch_events(league, set(rag.DEFAULT_MARKETS))
    target = datetime.date.fromisoformat(date)
    filtered = rag.filter_events_by_exact_date(events, target)

    results = []
    for ev in filtered:
        home = ev["home_team"]
        away = ev["away_team"]

        # Moneyline
        p_home, p_draw, p_away, h_proj, a_proj = rag.projected_moneyline_probs(home, away, league)
        ml_odds = rag.extract_moneyline_odds(ev)
        ml_result = rag.choose_best_moneyline_side(p_home, p_draw, p_away, ml_odds, home, away)

        # Goals totals
        proj_total, season_total, recent_total = rag.projected_total_goals(home, away, league)
        goals_options = rag.extract_total_line_options(ev, "goals")
        goals_best = rag.choose_best_total_line(goals_options, proj_total, combined_var)

        # BTTS
        p_btts, h_g, a_g, _, _ = rag.projected_btts_prob(home, away, league)
        btts_options = rag.extract_btts_odds(ev)
        btts_best = rag.choose_best_btts_side(btts_options, p_btts)

        results.append(build_fixture_summary(ev, ml_result, goals_best, btts_best, ...))

    return {"league": league, "date": date, "fixtures": results}
```

#### `GET /fixture/{league}/{event_id}`
Full detail for a single fixture (all market types).

**Response shape:**
```typescript
interface FixtureDetail {
  event_id: string;
  home_team: string;
  away_team: string;
  kickoff: string;
  league: string;

  moneyline: MoneylineDetail;
  goals_totals: TotalsDetail;
  corners_totals: TotalsDetail;
  cards_totals: TotalsDetail;
  sot_totals: TotalsDetail;
  btts: BTTSDetail;
  correct_score: CorrectScoreDetail;
  goal_intervals: GoalIntervalDetail;
  handicap: HandicapDetail;
  team_corners: TeamTotalsDetail;
  team_cards: TeamTotalsDetail;
  referee: RefereeDetail | null;

  evidence: EvidenceLine[];    // 3-5 human-readable reasoning points
  data_source: string | null;  // European competition data source note
}

interface TotalsDetail {
  projected_total: number;
  season_total: number | null;
  recent_total: number | null;
  recommended_side: string;
  recommended_line: number;
  model_prob: number;
  implied_prob: number | null;
  value_edge: number | null;
  ev_per_unit: number | null;
  confidence: string;
  best_odds: number | null;
  bookmaker: string | null;
}

interface MoneylineDetail {
  home_prob: number;
  draw_prob: number;
  away_prob: number;
  home_proj_goals: number;
  away_proj_goals: number;
  home_form: number | null;
  away_form: number | null;
  home_dominance: number | null;
  away_dominance: number | null;
  recommended: string;
  all_sides: MoneylineSide[];
  confidence: string;
}

interface MoneylineSide {
  team: string;
  model_prob: number;
  best_odds: number | null;
  bookmaker: string | null;
  value_edge: number | null;
  ev: number | null;
}

interface BTTSDetail {
  prob_yes: number;
  home_goal_proj: number;
  away_goal_proj: number;
  recommended: string;        // "Yes" | "No"
  model_prob: number;
  implied_prob: number | null;
  value_edge: number | null;
  confidence: string;
  best_odds: number | null;
}

interface CorrectScoreDetail {
  home_proj_goals: number;
  away_proj_goals: number;
  most_likely: string;         // "1-1"
  most_likely_prob: number;
  top_scores: {
    score: string;
    prob: number;
    odds: number | null;
    bookmaker: string | null;
    value_edge: number | null;
    ev: number | null;
  }[];
  result_probs: {
    home_win: number;
    draw: number;
    away_win: number;
  };
}

interface GoalIntervalDetail {
  projected_total: number;
  bands: {
    label: string;             // "0-1 goals"
    probability: number;
    odds: number | null;
    bookmaker: string | null;
  }[];
  recommended_band: string;
  recommended_prob: number;
}

interface HandicapDetail {
  projected_diff: number;
  recommended_side: string;    // "Arsenal -1.5"
  recommended_line: number;
  cover_prob: number;
  best_odds: number | null;
  value_edge: number | null;
  confidence: string;
}

interface TeamTotalsDetail {
  home: { team: string; projected: number; recommended_side: string; recommended_line: number; model_prob: number; confidence: string; };
  away: { team: string; projected: number; recommended_side: string; recommended_line: number; model_prob: number; confidence: string; };
}

interface RefereeDetail {
  name: string;
  avg_cards: number;
  strictness: number;          // 1.15 = 15% stricter than league avg
  strictness_label: string;    // "strict" | "lenient" | "average"
  sample_size: number;
  cards_per_foul: number;
}

interface EvidenceLine {
  text: string;
  category: string;           // "form" | "projection" | "referee" | "elo" | "value"
}
```

### 3.2 Parlay Builder

#### `POST /parlay/build`
Runs the parlay optimizer.

**Request:**
```typescript
interface ParlayBuildRequest {
  league: string;                    // or "all" for cross-league
  date: string;
  leg_count: number;
  target_odds_min?: number;
  target_odds_max?: number;
  preferred_markets?: string[];      // ["totals", "corners", "cards"]
  excluded_markets?: string[];
  fixture_filter?: string[];         // event_ids to restrict to
}
```

**Response:**
```typescript
interface ParlayBuildResponse {
  legs: ParlayLeg[];
  combined_odds: number;
  league_distribution: Record<string, number>;
  constraints_applied: string[];
  constraints_unmet: string[];
  explanation: string;               // LLM-generated reasoning
}

interface ParlayLeg {
  event_id: string;
  fixture: string;
  home_team: string;
  away_team: string;
  league: string;
  market_label: string;              // "Over 2.5 Goals"
  odds: number;
  bookmaker: string;
  evidence: string[];                // 2-3 stat-backed reasons
  model_prob: number | null;
  value_edge: number | null;
  confidence: string;
}
```

#### `POST /parlay/evaluate`
Scores a user-assembled parlay (from the interactive builder).

**Request:**
```typescript
interface ParlayEvaluateRequest {
  legs: {
    event_id: string;
    league: string;
    market_key: string;
    outcome: string;
    point?: number;
  }[];
}
```

**Response:**
```typescript
interface ParlayEvaluateResponse {
  legs: (ParlayLeg & { kb_quality_score: number })[];
  combined_odds: number;
  combined_model_prob: number;
  combined_ev: number;
  overall_confidence: string;
  warnings: string[];                // e.g., "2 legs from same fixture"
  suggestion: string | null;         // "Consider swapping leg 3..."
}
```

### 3.3 Search & Discovery

#### `GET /value-picks/{date}`
Top value picks across all leagues for a date (sorted by value_edge).

#### `GET /leagues`
Available leagues with current fixture counts.

---

## 4. Frontend File Structure

```
betting-rag-web/
├── src/
│   ├── app/                              # Next.js App Router
│   │   ├── layout.tsx                    # Root layout (nav, theme, parlay slip)
│   │   ├── page.tsx                      # Landing: today's top picks + league selector
│   │   ├── [league]/
│   │   │   ├── page.tsx                  # League fixture list for today
│   │   │   └── [date]/
│   │   │       ├── page.tsx              # League fixtures for specific date
│   │   │       └── [eventId]/
│   │   │           └── page.tsx          # Match detail page (all markets)
│   │   ├── parlay/
│   │   │   └── page.tsx                  # Parlay builder (auto-build + manual)
│   │   ├── picks/
│   │   │   └── page.tsx                  # "Value Picks" — sorted by edge
│   │   └── api/                          # Next.js API routes (proxy to FastAPI)
│   │       ├── fixtures/
│   │       │   └── [league]/[date]/route.ts
│   │       ├── fixture/
│   │       │   └── [league]/[eventId]/route.ts
│   │       ├── parlay/
│   │       │   ├── build/route.ts
│   │       │   └── evaluate/route.ts
│   │       └── value-picks/
│   │           └── [date]/route.ts
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Nav.tsx                   # Top nav: logo, league tabs, parlay count badge
│   │   │   ├── LeagueTabs.tsx            # Horizontal scrollable league selector
│   │   │   ├── DatePicker.tsx            # Today / Tomorrow / Next 7 days strip
│   │   │   └── MobileBottomBar.tsx       # Mobile: Home / Picks / Parlay (with leg count)
│   │   │
│   │   ├── fixtures/
│   │   │   ├── FixtureList.tsx           # Scrollable list of MatchCards
│   │   │   ├── MatchCard.tsx             # Single fixture summary card
│   │   │   ├── MatchCardSkeleton.tsx     # Loading skeleton
│   │   │   └── LeagueSection.tsx         # League header + fixture list group
│   │   │
│   │   ├── predictions/
│   │   │   ├── ConfidenceBadge.tsx       # Green/Yellow/Gray pill for high/med/low
│   │   │   ├── ValueEdgeBar.tsx          # Horizontal bar showing model vs books
│   │   │   ├── ProbabilityRing.tsx       # Circular progress showing probability %
│   │   │   ├── ProjectionBar.tsx         # Horizontal bar: projected total vs line
│   │   │   ├── MoneylineTriple.tsx       # Three-column Home/Draw/Away with probs
│   │   │   ├── GoalIntervalChart.tsx     # Bar chart of interval band probabilities
│   │   │   ├── CorrectScoreGrid.tsx      # 5x5 grid heatmap of scoreline probs
│   │   │   ├── TotalsCard.tsx            # Over/Under recommendation card
│   │   │   ├── BTTSCard.tsx              # Yes/No recommendation card
│   │   │   ├── HandicapCard.tsx          # Spread recommendation card
│   │   │   ├── RefereeCard.tsx           # Referee profile with strictness meter
│   │   │   └── EvidenceList.tsx          # Bulleted evidence lines
│   │   │
│   │   ├── parlay/
│   │   │   ├── ParlaySlip.tsx            # Floating/slide-out parlay slip
│   │   │   ├── ParlayLegRow.tsx          # Single leg in the slip (swipe to remove)
│   │   │   ├── ParlayOddsBar.tsx         # Running combined odds display
│   │   │   ├── ParlayBuilderModal.tsx    # Auto-build settings modal
│   │   │   ├── AddToParlay.tsx           # "+" button on any bet card
│   │   │   └── ParlayShareCard.tsx       # Screenshot-ready parlay summary
│   │   │
│   │   └── ui/                           # shadcn/ui primitives
│   │       ├── badge.tsx
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── dialog.tsx
│   │       ├── drawer.tsx                # Mobile bottom drawer
│   │       ├── skeleton.tsx
│   │       ├── tabs.tsx
│   │       └── tooltip.tsx
│   │
│   ├── hooks/
│   │   ├── useFixtures.ts               # React Query wrapper for fixtures
│   │   ├── useFixtureDetail.ts          # React Query wrapper for single fixture
│   │   ├── useParlaySlip.ts             # Zustand store for parlay state
│   │   ├── useValuePicks.ts             # React Query wrapper for value picks
│   │   └── useLeague.ts                 # URL-derived league context
│   │
│   ├── lib/
│   │   ├── api.ts                       # Fetch wrapper with base URL, error handling
│   │   ├── constants.ts                 # League names, colors, icons, market labels
│   │   ├── formatting.ts               # odds formatting, probability display, edge display
│   │   ├── parlayMath.ts               # Client-side combined odds, EV calculation
│   │   └── types.ts                     # All TypeScript interfaces (mirrors API shapes)
│   │
│   └── styles/
│       └── globals.css                  # Tailwind base + custom properties
│
├── public/
│   ├── league-logos/                    # EPL.svg, LaLiga.svg, etc.
│   └── og/                             # Open Graph images for social sharing
│
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── .env.local                          # NEXT_PUBLIC_API_URL, etc.
```

---

## 5. Page Designs & Routes

### 5.1 Landing Page (`/`)

**Purpose:** Answer "What should I bet on today?" in under 3 seconds.

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  [Logo]  EPL  LaLiga  SerieA  Bundes  L1  UCL  UEL │  ← LeagueTabs (scrollable)
├─────────────────────────────────────────────────────┤
│  Today    Tomorrow    Sat 15th    Sun 16th    ...   │  ← DatePicker strip
├─────────────────────────────────────────────────────┤
│                                                     │
│  TOP PICKS TODAY                    [Build Parlay]  │  ← Value picks banner
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│  │ Arsenal v CHE│ │ RM v Barca   │ │ Bayern v BVB ││  ← Top 3 by value edge
│  │ Over 2.5  ●● │ │ BTTS Yes  ●● │ │ Home Win  ●  ││     (ConfidenceBadge dots)
│  │ +8.3% edge   │ │ +6.1% edge   │ │ +5.2% edge   ││
│  │ [Add to slip] │ │ [Add to slip]│ │ [Add to slip]││
│  └──────────────┘ └──────────────┘ └──────────────┘│
│                                                     │
│  PREMIER LEAGUE                           See all → │  ← LeagueSection
│  ┌─────────────────────────────────────────────────┐│
│  │ Arsenal vs Chelsea      15:00                   ││  ← MatchCard
│  │ 42% Home  28% Draw  30% Away                   ││     (MoneylineTriple inline)
│  │ Goals: O 2.5 (62%)  BTTS: Yes (58%)  ●● HIGH   ││     (headline picks)
│  │                              [View] [+ Parlay]  ││
│  ├─────────────────────────────────────────────────┤│
│  │ Liverpool vs Man Utd    17:30                   ││
│  │ ...                                             ││
│  └─────────────────────────────────────────────────┘│
│                                                     │
│  LA LIGA                                  See all → │
│  ┌─ ... ─┐                                         │
│                                                     │
├─────────────────────────────────────────────────────┤
│  [Home]        [Picks]        [Parlay (3)]         │  ← MobileBottomBar
└─────────────────────────────────────────────────────┘
```

**Data strategy:** ISR with 15-minute revalidation. `getStaticProps` calls `/api/v1/fixtures/{league}/today` for each league, assembles the page. Stale-while-revalidate means the page loads instantly from cache while refreshing in background.

### 5.2 Match Detail Page (`/[league]/[date]/[eventId]`)

**Purpose:** Full analysis dashboard for a single fixture.

**Layout (mobile: stacked tabs; desktop: 2-column):**
```
┌─────────────────────────────────────────────────────┐
│  ← Back to EPL                         Mar 13, 2026 │
├─────────────────────────────────────────────────────┤
│             Arsenal  vs  Chelsea                     │
│             1.85        3.40       4.20              │  ← live odds
│          ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░              │  ← probability bar
│          42% Home    28% Draw    30% Away            │
│                                                      │
│  [Winner]  [Goals]  [Corners]  [Cards]  [More ▼]    │  ← market tabs
├─────────────────────────────────────────────────────┤
│                                                      │
│  GOALS TOTALS                                        │
│  Projected: 2.67 goals                               │
│  ┌──────────────────────────────────────┐            │
│  │  [====projected===|==line===]        │ 2.5 line   │  ← ProjectionBar
│  └──────────────────────────────────────┘            │
│  Recommended: Over 2.5 @ 1.87 (bet365)              │
│  Model: 62.3%  |  Books: 53.5%  |  Edge: +8.8%     │  ← ValueEdgeBar
│  Confidence: ●● HIGH                                │
│  [+ Add to Parlay]                                   │
│                                                      │
│  Season projection: 2.72                             │
│  Recent 6-match: 2.58                                │
│                                                      │
│  Evidence:                                           │
│  • Arsenal avg 2.1 goals/match, Chelsea concede 1.4 │
│  • Combined xG suggests 2.8 expected goals           │
│  • Both teams scoring in 7 of last 10 fixtures      │
│                                                      │
├─────────────────────────────────────────────────────┤
│  BTTS                      CORRECT SCORE             │
│  P(Yes) = 58%              Most likely: 1-1 (14.2%) │
│  Rec: Yes @ 1.72           1.  2-1  12.8%  @ 8.50  │
│  Edge: +5.1%               2.  1-1  14.2%  @ 6.00  │
│  [+ Parlay]                3.  1-0  11.1%  @ 7.50  │
│                             4.  2-0   8.3%  @11.00  │
│                             5.  0-1   9.2%  @ 9.00  │
├─────────────────────────────────────────────────────┤
│  GOAL INTERVALS                                      │
│  ┌──────────────────────────────────────┐            │
│  │  0-1  ██████████  28.3%              │            │  ← GoalIntervalChart
│  │  2-3  ████████████████  48.1%        │            │
│  │  4-5  ██████  18.4%                  │            │
│  │  6+   ██  5.2%                       │            │
│  └──────────────────────────────────────┘            │
│                                                      │
├─────────────────────────────────────────────────────┤
│  REFEREE                                             │
│  Michael Oliver                                      │
│  4.2 cards/match (strict, 1.15x league avg)         │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  0.142 cards/foul              │
│  Based on 47 matches                                │
└─────────────────────────────────────────────────────┘
```

**Data strategy:** ISR with 5-minute revalidation for the match detail. Uses `generateStaticParams` for known upcoming fixtures. Falls back to SSR for unknown event IDs.

### 5.3 Parlay Builder (`/parlay`)

**Purpose:** Two modes: (1) Auto-build with constraints, (2) Manual assembly from browsing.

**Auto-build flow:**
1. User selects league(s), date, leg count, target odds range
2. Optionally selects preferred/excluded market types
3. Clicks "Build Parlay"
4. Backend runs `select_parlay()` and returns optimized combo
5. Result displayed as ParlayShareCard with evidence per leg

**Manual flow:**
1. User browses fixtures, clicks "+" on any bet recommendation
2. Leg added to ParlaySlip (persistent bottom drawer on mobile, sidebar on desktop)
3. Slip shows running combined odds, combined model probability, combined EV
4. "Evaluate" button sends assembled legs to `/parlay/evaluate` for KB-backed analysis

**ParlaySlip Zustand store:**
```typescript
interface ParlaySlipState {
  legs: ParlaySlipLeg[];
  addLeg: (leg: ParlaySlipLeg) => void;
  removeLeg: (index: number) => void;
  clearSlip: () => void;

  // Derived
  combinedOdds: number;
  legCount: number;
}

interface ParlaySlipLeg {
  event_id: string;
  fixture: string;
  league: string;
  market_label: string;    // "Over 2.5 Goals"
  odds: number;
  bookmaker: string;
  model_prob: number | null;
  value_edge: number | null;
  confidence: string;
}
```

**Persistence:** Zustand `persist` middleware writes to `localStorage`. Parlay survives page refreshes and browser restarts. A "Share Parlay" button generates a URL-encoded parlay link (`/parlay?legs=base64encodedLegs`) for social sharing.

### 5.4 Value Picks Page (`/picks`)

**Purpose:** "What are the best bets today?" sorted by value edge.

Shows all fixtures across all leagues, ranked by highest value edge. Each row shows: fixture, market, model probability, book odds, edge, confidence. Filterable by league, market type, minimum confidence.

---

## 6. Key Component Specifications

### 6.1 MatchCard

The most-reused component. Appears on landing page, league pages, and search results.

```tsx
// Condensed, not full implementation — shows the data contract
interface MatchCardProps {
  fixture: FixtureSummary;
  onAddToParlay: (leg: ParlaySlipLeg) => void;
  compact?: boolean;  // true on landing page, false on league page
}
```

**Compact mode (landing page):** Team names, kickoff time, moneyline probabilities as a single bar, best pick headline with confidence dots.

**Full mode (league page):** Adds goals O/U, BTTS, top value edge with "Add to Parlay" buttons.

### 6.2 ConfidenceBadge

Translates "high" / "medium" / "low" into a visual signal without technical jargon.

| Confidence | Visual | Label | Color |
|-----------|--------|-------|-------|
| high | Three filled dots | "Strong Pick" | Green (`#22c55e`) |
| medium | Two filled + one outline | "Moderate" | Amber (`#f59e0b`) |
| low | One filled + two outline | "Speculative" | Gray (`#6b7280`) |

No mention of "probability," "model," or "edge" in the badge itself. The technical details are available one level deeper in the match detail view.

### 6.3 ValueEdgeBar

Shows bookmaker implied probability vs model probability as a visual comparison.

```
Books say:  ████████████████░░░░░░░░  53.5%
Model says: ██████████████████████░░  62.3%
                              Edge: +8.8%
```

For casual users, the label is "Our model sees more value here than the bookmakers." For those who tap/hover, the percentages appear.

### 6.4 ParlaySlip

**Desktop:** Fixed sidebar on the right (320px wide), always visible when legs > 0.

**Mobile:** Bottom drawer (sheet). Collapsed state shows "Parlay (3 legs) 4.2x" as a bar. Tap to expand full slip. Swipe down to collapse.

The slip calculates combined odds client-side using `parlayMath.ts`:

```typescript
// lib/parlayMath.ts
export function combinedOdds(legs: { odds: number }[]): number {
  return legs.reduce((acc, leg) => acc * leg.odds, 1.0);
}

export function impliedProbability(odds: number): number {
  return 1 / odds;
}

export function combinedProbability(legs: { model_prob: number | null }[]): number | null {
  const probs = legs.map(l => l.model_prob).filter((p): p is number => p !== null);
  if (probs.length !== legs.length) return null;
  return probs.reduce((acc, p) => acc * p, 1.0);
}
```

---

## 7. Data Flow & Caching Strategy

### 7.1 Pre-Computation Layer (Cron Job)

A background process runs every 15 minutes:

```python
# cron_precompute.py
for league in ALL_LEAGUES:
    for date in [today, tomorrow]:
        fixtures = compute_all_fixture_predictions(league, date)
        write_json(f"cache/{league}_{date}.json", fixtures)
```

This pre-computed JSON is what ISR pages read. The expensive work (Chroma queries, projection calculations, odds fetching) happens in the background, not on page request.

### 7.2 Caching Layers

| Layer | TTL | What |
|-------|-----|------|
| Vercel ISR | 15 min | Fixture list pages, match detail pages |
| React Query | 5 min | Client-side fixture data (stale-while-revalidate) |
| FastAPI in-memory | 10 min | API-Football fixture and odds responses (rate-limit protection) |
| FastAPI in-memory | 15 min | Referee assignments (API-Football) |
| Chroma | Persistent | Team profiles (re-embedded weekly) |
| localStorage | Permanent | Parlay slip state |

### 7.3 Rate Limit Protection

API-Football has request limits. The architecture protects against this:

1. **Pre-computation** means page loads never hit API-Football directly.
2. **FastAPI caches** odds responses for 5 minutes. Multiple users requesting the same league/date share one upstream call.
3. **Parlay builder** is the only user-initiated flow that might trigger fresh odds fetches. The builder UI shows a "Last updated X min ago" timestamp and a manual refresh button (rate-limited to once per 2 minutes client-side).

---

## 8. Translation Layer: Technical to Casual

The backend produces technical output. The frontend must translate this for casual bettors without losing accuracy.

| Backend Term | Frontend Display |
|-------------|-----------------|
| `model_prob: 0.623` | "62% chance" (inside a ProbabilityRing) |
| `value_edge: 0.083` | "8% better odds than the books" |
| `confidence: "high"` | "Strong Pick" (green badge) |
| `projected_total: 2.67` | "We expect about 2.7 goals" |
| `ev_per_unit: 0.156` | "Good value" (only shown on detail page) |
| `strictness_ratio: 1.15` | "This ref gives 15% more cards than average" |
| `P(BTTS Yes) = 0.58` | "58% both teams score" |
| `cover_prob: 0.61` | "61% chance of covering the spread" |
| Poisson/NegBin distribution | Never shown. Abstracted to probability % |
| R-squared, Elo rating | Never shown. Drives confidence badge internally |
| `dominance_index: 0.72` | "Strong control of matches" (in evidence list) |

**Progressive disclosure pattern:** Landing page shows zero technical terms. Match detail page shows probabilities and edges. A collapsible "Technical Details" section at the bottom of match detail shows season/recent splits, variance, Elo ratings, and model methodology for power users.

---

## 9. Mobile-First Design Decisions

1. **MatchCard** is a full-width card with large tap targets. The "+" button for parlay is 44x44px minimum.
2. **LeagueTabs** scroll horizontally with snap behavior. Active league is always centered.
3. **DatePicker** is a horizontal strip (not a calendar popup) since users only care about the next 7 days.
4. **ParlaySlip** is a bottom drawer, not a sidebar. It uses `vaul` (Drawer component from shadcn).
5. **Match detail tabs** (Winner / Goals / Corners / Cards / More) are swipeable on mobile.
6. **Font sizes:** Match scores use 24px, odds use 18px, evidence text uses 14px. All above mobile accessibility minimums.
7. **No hover-dependent interactions.** Everything that shows on hover also shows on tap.

---

## 10. SEO Strategy

Match detail pages should be indexable for queries like "Arsenal vs Chelsea predictions."

1. **Dynamic metadata** per match page:
   ```tsx
   export async function generateMetadata({ params }): Promise<Metadata> {
     const fixture = await getFixture(params.league, params.eventId);
     return {
       title: `${fixture.home_team} vs ${fixture.away_team} Predictions | BettingRAG`,
       description: `Our model gives ${fixture.home_team} a ${(fixture.moneyline.home_prob * 100).toFixed(0)}% chance. Goals projection: ${fixture.goals.projected_total.toFixed(1)}. Best value: ${fixture.goals.recommended_side} ${fixture.goals.recommended_line}.`,
       openGraph: {
         images: [`/api/og/${params.eventId}`],  // Dynamic OG image
       },
     };
   }
   ```

2. **Structured data** (JSON-LD) for sports events:
   ```json
   {
     "@type": "SportsEvent",
     "name": "Arsenal vs Chelsea",
     "startDate": "2026-03-13T15:00:00Z",
     "homeTeam": { "@type": "SportsTeam", "name": "Arsenal" },
     "awayTeam": { "@type": "SportsTeam", "name": "Chelsea" }
   }
   ```

3. **Static generation** of match pages via ISR means search engines get fully rendered HTML.

---

## 11. Performance Budget

| Metric | Target | How |
|--------|--------|-----|
| LCP | < 1.5s | ISR pre-renders HTML; league logos as SVG inline |
| FID | < 50ms | Minimal JS on landing page; parlay logic lazy-loaded |
| CLS | < 0.05 | Skeleton placeholders match final card dimensions |
| Bundle (initial) | < 120KB gzip | Code-split per route; shadcn tree-shakes; no moment.js |
| TTI (3G) | < 3s | SSR HTML is usable before JS hydrates |

**Code splitting plan:**
- Landing page: MatchCard, LeagueTabs, DatePicker (core bundle)
- Match detail: TotalsCard, GoalIntervalChart, CorrectScoreGrid, RefereeCard (lazy)
- Parlay builder: ParlayBuilderModal, ParlayShareCard (lazy)
- Heavy components (GoalIntervalChart uses recharts or similar) loaded via `next/dynamic`

---

## 12. Implementation Phases

### Phase 1: Core Read-Only (2-3 weeks)
- FastAPI wrapper with `/fixtures` and `/fixture` endpoints
- Landing page with MatchCard, LeagueTabs, DatePicker
- Match detail page with all market sections
- ConfidenceBadge, ValueEdgeBar, EvidenceList
- Mobile-responsive layout
- ISR caching

### Phase 2: Parlay Builder (1-2 weeks)
- ParlaySlip Zustand store with localStorage persistence
- AddToParlay buttons on all bet recommendations
- ParlaySlip drawer (mobile) and sidebar (desktop)
- `/parlay/build` auto-builder endpoint and UI
- `/parlay/evaluate` for manual parlay scoring

### Phase 3: Discovery & Polish (1 week)
- Value Picks page with cross-league sorting
- Share parlay as URL / screenshot card
- Dynamic OG images for match pages
- Structured data for SEO
- Dark mode

### Phase 4: Real-Time & Social (future)
- WebSocket for live odds updates
- User accounts (save parlays, track results)
- Parlay leaderboard
- Push notifications for value alerts

---

## 13. Environment Variables

```env
# .env.local (Next.js)
NEXT_PUBLIC_API_URL=http://localhost:8000   # FastAPI backend
NEXT_PUBLIC_SITE_URL=https://bettingrag.com

# FastAPI backend (existing .env)
API_FOOTBALL_KEY=<key>
OPENAI_API_KEY=<key>
API-FOOTBALL-KEY=<key>
CHROMA_COLLECTION=football_top5
RAG_CHAT_MODEL=gpt-5.4
```

---

## 14. Key Files to Create First

1. **`backend/fastapi_app.py`** — FastAPI wrapper around rag_cli_v2.py functions. This is the critical bridge. Start here.
2. **`src/lib/types.ts`** — All TypeScript interfaces. Define the contract before building components.
3. **`src/components/predictions/ConfidenceBadge.tsx`** — The visual language of the entire site.
4. **`src/components/fixtures/MatchCard.tsx`** — Most-reused component.
5. **`src/hooks/useParlaySlip.ts`** — Zustand store. Enables the parlay builder from day one.
6. **`src/app/page.tsx`** — Landing page. First thing users see.
