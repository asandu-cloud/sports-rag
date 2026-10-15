# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Casual European football bettors, UK included, aged 18+. They follow football and bet occasionally rather than professionally. Their typical moment is matchday: checking which fixtures have analysis, often on a phone, sometimes hours before kickoff and sometimes in the evening. Many are not fluent readers of statistics, so every number needs plain-language context.

## Product Purpose

Spix turns a statistical football model into fixture-level Match Reads: the match outlook, up to three compatible selections or an explicit "no bet", and the limits behind the numbers. Success is a reader who understands why a selection exists or why passing is the sensible choice, before deciding anything.

## Positioning

Spix is analysis that is willing to say "pass". A Match Read can end with no bet, prices are shown as recorded quotes rather than guarantees, and no profit or win-rate claims are made. Sportsbooks and tip channels sell action; Spix sells a clearer view before the decision.

## Operating Context

- Website: FastAPI app (`Scripts/web_app/`) serving a public landing page, info pages (Methodology, Privacy, Terms, Responsible Gambling) and the app: Matchday board, Research Area, Best Bets, Match Read detail and a bet slip.
- Companion Discord community and bots (Spix's Picks); the website links to Discord.
- Odds and fixtures come from API-Football. Covered competitions extend beyond the top five leagues on the board.
- Accounts: email/password and Discord sign-in. Paid membership is "coming soon"; no payment starts on the site.

## Capabilities and Constraints

- Match Reads may be preliminary, published, updated, or awaiting analysis; selections carry confidence labels that are heuristic, not win probabilities.
- The bet slip collects selections but does not place bets; parlay scoring is disabled until it uses verified quotes.
- Research Area currently runs on clearly labelled fictional sample data.
- Open decision: stake currency. The slip uses `$` while the Methodology calculator uses `£`. Not yet decided for a European audience.

## Brand Commitments

- Public name: **Spix** everywhere on the website (not SpixBot, not Matchwise).
- Logo mark: the slanted "S" with two eyes. Its shape is fixed; colour may change with the palette.
- Voice: plain, careful, never hype. Copy names uncertainty and limits directly.
- Responsible gambling is part of the brand: 18+ messaging, GamCare link, "you can lose your stake".

## Evidence on Hand

- Live fixture and Match Read data from the API (`/api/match-reads/*`, `/api/fixtures/*`).
- Illustrative example Match Read on the landing page (labelled illustrative).
- No verified track record, testimonials, customer counts or pricing exist. Future work must not invent them.

## Product Principles

1. Passing is a real outcome; "no bet" is never styled as a failure or a disabled state.
2. Show the reasoning next to the number. A figure without context is not shipped.
3. Never imply certainty: estimates, quotes and confidence labels are named for what they are.
4. Readable on a phone on matchday comes first; density is earned, not default.

## Accessibility & Inclusion

WCAG 2.2 AA as the floor. Readers include novices and people reading in varied light, so text contrast, a 12px minimum for any text, keyboard access and screen-reader parity with visual content are required.
