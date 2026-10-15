# Matchwise — Website & Prediction Communication Plan

## Executive Summary

A consumer-facing website that translates complex statistical predictions into instant, visual bet recommendations. The core principle: **lead with the pick, back it with evidence, let users build parlays from what they see.**

---

## 1. Brand Identity

**Name:** Matchwise
**Tagline:** Every pick. Every reason.
**Positioning:** Evidence-based betting companion — not a tipster, not a sportsbook, not a stats tool. A smart friend who watches every league and shows their work.

### Visual System

| Element | Specification |
|---------|--------------|
| Background | Deep Navy `#0F1923` |
| Cards/Surfaces | `#1A2836` |
| Primary Accent | Emerald Teal `#00D4AA` |
| Warning Accent | Warm Amber `#FFB347` |
| Muted | Steel `#8899AA` |
| Negative/Avoid | Coral `#E74C6F` |
| Body Font | Inter (free, tabular numbers for odds) |
| Display Font | Space Grotesk (headers, personality) |
| Data Font | JetBrains Mono (odds, percentages) |

### Confidence Language (used everywhere)

| Internal Level | User-Facing Label | Color | Visual |
|---------------|-------------------|-------|--------|
| High (P >= 65%, edge >= 8%) | **Strong Edge** | `#00D4AA` | Filled badge + bar |
| Medium (P >= 55%, edge >= 3%) | **Leaning** | `#FFB347` | Partial badge + bar |
| Low (below thresholds) | **Speculative** | `#8899AA` | Outline badge + bar |

### Value Communication

| Edge Range | Label | Shown to User |
|-----------|-------|---------------|
| >= 8% | "Strong Value" | "Books are underpricing this by 8%" |
| 5-8% | "Good Value" | "5% edge over bookmakers" |
| 3-5% | "Fair Value" | "Slight edge at current odds" |
| < 3% | "Market Price" | "Close to fair odds" |
| Negative | "No Value" | "Overpriced — pass or wait" |

---

## 2. Information Architecture

### The 3-Second Rule

Every prediction card must answer three questions in 3 seconds:
1. **What to bet** — the pick (e.g., "Over 2.5 Goals")
2. **How confident** — visual badge (Strong Edge / Leaning / Speculative)
3. **What you get paid** — odds (e.g., "@ 1.87")

Everything else (model probability, value edge, projections, evidence) is behind one tap.

### Progressive Disclosure (3 Levels)

**Level 1 — Card (always visible):**
```
Arsenal vs Chelsea  |  3:00 PM  |  EPL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[STRONG EDGE]  Over 2.5 Goals @ 1.87
"Both teams average 1.4+ goals. 7% edge."
                              [+ Add to Slip]
```

**Level 2 — Expanded (one tap):**
```
Model: 62%  |  Books: 55%  |  Edge: +7.2%
Projection: 2.72 total goals
Season avg: 2.81  |  Recent 6: 2.64
EV: +$0.17 per $1 staked
Referee: M. Oliver (strict, 1.15x avg cards)
```

**Level 3 — Full Analysis (separate page):**
```
Complete breakdown: team profiles, H2H, venue splits,
defensive adjustments, xG data, Elo ratings, variance
modeling, all bookmaker lines, alternative markets.
```

### Simple Mode vs Expert Mode

**Simple Mode (default):** Confidence badges, one-line picks, "Good Value" labels, pre-built parlays.
**Expert Mode (toggle):** Model probabilities inline, value edge percentages, EV per unit, Elo signals, referee modifiers, Poisson vs NegBin indicators, Kelly fraction.

---

## 3. Page Structure

### 3.1 Landing Page (`/`)

**Hero:** "Every pick. Every reason." + [See Today's Picks] CTA

**Today's Best Parlays** — swipeable carousel of 2-3 pre-built parlays (from `select_parlay()`):
- 3-leg Safe parlay (~2.5x)
- 4-leg Standard parlay (~4x)
- 5-leg High-value parlay (~7x)

**Fixtures by League** — horizontal league tabs (EPL | LaLiga | SerieA | ...), scrollable fixture cards underneath.

**Model Accuracy Banner** — "Season Record: 412-287 (58.9%) on recommended picks" with link to full track record.

### 3.2 Fixtures Page (`/fixtures/[league]/[date]`)

**League tabs** at top (horizontal scroll on mobile).
**Date selector** — Today | Tomorrow | +5 day buttons.
**Fixture cards** — one per match, showing the headline pick (highest-confidence market).

Each card shows:
- Teams + kickoff + league badge
- Best pick with confidence badge
- Odds
- One-line reasoning
- "More Markets" chevron (expands to show all market picks)
- "+ Add" button (adds headline pick to parlay slip)

### 3.3 Match Detail Page (`/match/[league]/[event_id]`)

**Hero section:** Teams, kickoff, league, live score area (future).

**Market sections** — scrollable or tabbed:
- Best Pick (highlighted, auto-selected by highest quality score)
- Goals O/U
- Moneyline (3-way probability ring chart)
- BTTS
- Corners O/U
- Cards O/U (with referee badge)
- SoT O/U
- Handicap/Spread
- Correct Score (top 5 scorelines)
- Goal Intervals (probability bar chart)
- Team Lines (per-team corners + cards)

Each market section is a **Bet Card** with:
- Pick + odds + confidence badge
- One-line reasoning
- Expandable evidence (Level 2)
- "+ Add to Slip" button

**Team Comparison sidebar** (desktop) or section (mobile):
- Form index comparison
- Dominance index
- Elo ratings per stat
- Recent results

### 3.4 Parlay Builder Page (`/parlay`)

**AI-Built Parlays tab** — pre-computed by `select_parlay()`:
- Today's Best (3-5 curated parlays)
- By league
- By market type (corners-only, goals-only, mixed)

**My Parlay tab** — user's current slip:
- All added legs with confidence indicators
- Running combined odds
- Parlay quality score (from `score_combo()`)
- "Evaluate" button — sends to backend for full KB-scored assessment
- Individual leg swap (suggest alternative from same fixture)

**Quick Build** — "Give me a [2/3/4/5]-leg parlay for [league] today" button.

### 3.5 Track Record Page (`/accuracy`)

Auto-generated from `backtest.py`:
- Overall hit rate + sample size
- By confidence level (Strong Edge picks should be 65%+)
- By market type (goals, corners, cards, SoT, BTTS, moneyline)
- By league
- Monthly breakdown
- Current streak

### 3.6 How It Works Page (`/how-it-works`)

Plain-language explanation (no code, no formulas):
1. We collect data from 8 European leagues
2. Our models project match statistics using team form, opponent strength, venue, and referee tendencies
3. We compare our projections to bookmaker odds to find mispricings
4. We only recommend bets where our model sees meaningful value
5. Every recommendation shows its full evidence chain

---

## 4. Technical Architecture

### Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | **Next.js 15** (App Router) | SSR for SEO, ISR for pre-computed predictions, mobile-first |
| Styling | **Tailwind CSS 4 + shadcn/ui** | Dark mode built-in, rapid iteration, consistent components |
| State | **Zustand** (parlay slip) + **React Query** (server data) | 1KB state lib, localStorage persistence, SWR caching |
| Backend API | **FastAPI** wrapping `rag_cli_v2.py` | Thin HTTP layer over existing Python scoring engine |
| Data | **Pre-computed JSON (ISR)** + on-demand API for parlay builder | Pre-compute every 15 min, serve from edge cache |
| Hosting | **Vercel** (frontend) + **Railway/Fly.io** (FastAPI + Chroma) | Edge CDN for speed, persistent disk for Chroma |

### Data Flow

```
Cron (every 15 min)
    |
    v
FastAPI calls rag_cli_v2.py projection functions
    |
    v
Writes structured JSON to /api/fixtures/{league}/{date}
    |
    v
Next.js ISR serves from cache (sub-1.5s page loads)

User adds leg to parlay → client-side Zustand store
User clicks "Evaluate" → FastAPI /api/parlay/evaluate (live)
```

### API Endpoints

| Endpoint | Method | Source Function | Cache |
|----------|--------|----------------|-------|
| `/api/v1/fixtures/{league}/{date}` | GET | `fetch_events()` + `filter_events_by_exact_date()` + all `projected_*()` functions | 15 min ISR |
| `/api/v1/fixture/{league}/{event_id}` | GET | All `projected_*()` + `choose_best_*()` + `confidence_from_edge()` | 15 min ISR |
| `/api/v1/parlay/evaluate` | POST | `kb_leg_quality()` + `score_combo()` | No cache |
| `/api/v1/parlay/suggest` | POST | `select_parlay()` + `leg_evidence()` | 5 min |
| `/api/v1/accuracy` | GET | `backtest.py` output | Daily |

### Key Implementation Detail: No Text Parsing

The current CLI renders predictions as text (`render_totals_line_answer()`, etc.) which the Discord bot then regex-parses back into structured data. The website **skips this entirely**. FastAPI calls the underlying projection functions directly (`projected_total_goals()`, `choose_best_total_line()`, etc.) and returns structured JSON. No text rendering → no regex parsing.

---

## 5. Core UI Components

### 5.1 Match Card (Fixture List)

```
┌─────────────────────────────────────────────┐
│  EPL              15:00 UTC                  │
│  Arsenal  vs  Chelsea                        │
│                                              │
│  [STRONG EDGE]  Over 2.5 Goals               │
│  @ 1.87 (Bet365)                             │
│  "7% edge — both teams averaging 1.4+ gpg"  │
│                                              │
│  [More Markets ▾]            [+ Add to Slip] │
└─────────────────────────────────────────────┘
```

- Card border-left color = confidence color
- Touch target for "+ Add" = 44px minimum
- "More Markets" expands to show all market picks inline
- Tap fixture header → navigates to full match detail page

### 5.2 Bet Card (Single Market)

```
┌─────────────────────────────────────────────┐
│  ⚽ Goals Over/Under          [STRONG EDGE]  │
│                                              │
│  Over 2.5 Goals                              │
│  @ 1.87 (Bet365)                             │
│                                              │
│  ██████████████████░░  62%                   │
│  Model: 62%  |  Books: 55%  |  Edge: +7%    │
│                                              │
│  ▾ Show Evidence                             │
│                                [+ Add to Slip]│
└─────────────────────────────────────────────┘
```

- Probability bar fills proportionally (green for strong, amber for lean)
- Evidence section (collapsed by default) shows projection, season/recent split, referee data
- Simple Mode hides the "Model: 62% | Books: 55% | Edge: +7%" line entirely — just shows the badge

### 5.3 Parlay Slip (Mobile Bottom Sheet)

```
┌─────────────────────────────────────────────┐
│  ▬  My Parlay (3 legs)         Combined: 3.4x│
│                                              │
│  1. Arsenal vs Chelsea                       │
│     Over 2.5 Goals @ 1.87  [STRONG]     [✕] │
│                                              │
│  2. Man City vs Newcastle                    │
│     BTTS Yes @ 1.72        [LEANING]    [✕] │
│                                              │
│  3. Bayern vs Dortmund                       │
│     Over 9.5 Corners @ 1.83 [STRONG]    [✕] │
│                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  Combined Odds: 5.87x                        │
│  $10 wins $58.70                             │
│  Parlay Quality: ●●●●○ Good                  │
│                                              │
│  [Evaluate with AI]        [Share / Copy]    │
└─────────────────────────────────────────────┘
```

- Mobile: slides up from bottom, drag handle at top
- Desktop: fixed sidebar on right
- Persists via Zustand + localStorage (survives page navigation and browser close)
- "Evaluate with AI" calls `score_combo()` + `kb_leg_quality()` and returns quality assessment
- "Share" generates a shareable link or image card

### 5.4 Confidence Badge

Three-tier system, used everywhere consistently:

```
Strong Edge:   [████████████] green pill, white text, filled
Leaning:       [████████░░░░] amber pill, dark text, partial fill
Speculative:   [████░░░░░░░░] gray pill, gray text, outline
```

- Works at 3 sizes: small (inline in cards), medium (bet cards), large (match detail hero)
- Includes both color AND shape difference (accessibility for colorblind users)
- Always paired with the label text ("Strong Edge" not just the bar)

### 5.5 Value Badge

When edge >= 3%, show a badge next to the odds:

```
Strong Value (8%+):   [↑ 8% EDGE]  green background
Good Value (5-8%):    [↑ 5% EDGE]  teal background
Fair Value (3-5%):    [↑ 3% EDGE]  amber background
No badge below 3%
```

Plain-language tooltip on hover/tap: "Our model gives this a 62% chance. Bookmakers imply 55%. That 7% gap means the odds are in your favor."

---

## 6. Mobile-First Design Principles

- **80-85% of bettors use mobile.** This is mobile-first, desktop-enhanced.
- Bottom sheet for parlay slip (not sidebar)
- Horizontal scroll for league tabs with snap points
- Swipeable market tabs on match detail
- 44px minimum touch targets (Apple HIG)
- No hover-dependent interactions
- Fixture cards as full-width list items (not grid) on mobile
- Evidence sections collapsed by default (accordion pattern)
- Sticky header with league tabs + date selector

---

## 7. Parlay Builder UX

### Three Modes

1. **AI-First (default):** "Today's Best Parlays" — pre-built by `select_parlay()`, displayed as swipeable cards. User taps "Use This Parlay" or removes/swaps individual legs.

2. **Manual Build:** User browses fixtures and taps "+ Add to Slip" on individual bet cards. System shows "Parlay Quality" score updating in real-time.

3. **Quick Build:** "Build me a [2-5]-leg parlay for [league]" — single button, instant result.

### Handling "Bad" Legs

When a user adds a leg the model rates poorly:
- **Never block** — users hate being told they can't bet
- **Warn visually** — leg appears with gray/red indicator: "Thin edge at these odds"
- **Show impact** — Parlay Quality score drops visibly
- **Suggest alternative** — "Consider Over 2.5 instead (stronger edge)"

### Follow-up Actions

After parlay is built:
- **Adjust Risk** — slider from 2x to 10x target odds, system rebalances legs
- **Swap Weakest** — system identifies lowest-quality leg and offers alternatives
- **Lock + Rebuild** — user locks legs they like, system replaces the rest
- **Share** — generates image card or shareable link

---

## 8. Trust & Track Record

### Model Accuracy Page (auto-generated from backtest.py)

```
Season 2025/26 Track Record
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall:          58.9%  (412/699 picks graded)
Strong Edge:      71.4%  (112/157)  ← this is the key number
Leaning:          53.8%  (201/373)
Speculative:      58.7%  (99/169)

By Market:
  Corners O/U:    61.2%  (287 picks)
  Cards O/U:      58.9%  (203 picks)
  Goals O/U:      54.7%  (341 picks)
  BTTS:           59.4%  (186 picks)
  Moneyline:      52.1%  (230 picks)

Updated daily. All picks tracked, including losses.
```

The "Strong Edge picks hit at 71%" number is the single most powerful trust signal. It proves the confidence calibration works.

### Responsible Gambling

Footer on every page:
> Matchwise provides statistical analysis for entertainment and informational purposes. Never bet more than you can afford to lose. 18+ only. If you need help, contact the National Council on Problem Gambling at 1-800-522-4700.

---

## 9. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- FastAPI wrapper around core projection functions
- Pre-computed fixture JSON (cron every 15 min)
- Next.js app with fixture list + match detail pages (read-only)
- Match cards with confidence badges + bet cards
- Mobile-responsive layout
- Dark theme with brand design system

### Phase 2: Parlay Builder (Weeks 4-5)
- Zustand parlay slip with localStorage persistence
- "+ Add to Slip" buttons on all bet cards
- Bottom sheet (mobile) / sidebar (desktop) parlay slip
- "Evaluate with AI" endpoint
- Pre-built "Today's Best Parlays" carousel
- Share/copy functionality

### Phase 3: Discovery & Trust (Weeks 6-7)
- Track record page (auto-generated from backtest.py)
- Simple/Expert mode toggle
- "Best Bets Today" aggregation page (highest-confidence picks across all leagues)
- SEO optimization for match pages
- How It Works page

### Phase 4: Growth Features (Weeks 8+)
- User accounts (optional, for saving preferences)
- Pick tracking (users mark which picks they followed)
- Community leaderboard
- Push notifications for high-confidence picks near kickoff
- "Value Alerts" (equivalent to Discord value scan)
- Premium tier differentiation

---

## 10. Cross-Channel Consistency

| Element | Website | Discord | Future App |
|---------|---------|---------|------------|
| Confidence labels | Strong Edge / Leaning / Speculative | Same | Same |
| Confidence colors | #00D4AA / #FFB347 / #8899AA | Same | Same |
| Card structure | Pick → Confidence → Odds → Why | Same hierarchy in embeds | Same |
| Value language | "X% edge over bookmakers" | Same | Same |
| Track record | Full page at /accuracy | Summary in bot profile | In-app section |
| Tone | Professional-warm | Slightly more conversational | Notification-brief |

---

## 11. What NOT to Build

- **No sportsbook integration** (placing bets) — regulatory minefield. Show picks, link to sportsbooks.
- **No live/in-play features** (Phase 1) — pre-match is 90% of value and 100% of current capability.
- **No user accounts for browsing** — predictions are public. Accounts only for saving preferences and tracking.
- **No chat/AI assistant on the website** — the structured cards ARE the prediction. Chat is for Discord.
- **No complex filters on launch** — start with league + date only. Market-type filtering in Phase 3.
