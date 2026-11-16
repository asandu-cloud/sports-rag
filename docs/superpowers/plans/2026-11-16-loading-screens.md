# Loading screens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the site's plain "Loading…" lines with the six loaders the owner approved on 16 Nov ("i like all of
them. lets implement them"), each where the options page placed it.

**Architecture:** One shared kit: `static/loaders.js`, a classic script whose top-level functions return each
loader's HTML (testable with `Function(source)` like `home-wins.js`), and `static/loaders.css`. The page links both;
the Research Area's shadow root fetches `loaders.css` beside `research.css` and calls the kit through `window`.
Everything moves in CSS: the delay, the status steps, the prices, the still frames. Only the first-load cover (A)
has script timing.

**Tech Stack:** Plain JS and CSS in `Scripts/web_app/static/`; Node tests run with `/usr/local/bin/node`.

**Spec:** the options page, https://claude.ai/artifact/De2WiK5ZfHMYDqvaeSuf9S ("Spix Loading Screens", private),
and the owner's approval of all six with the placements below (16 Nov, D112).

| Loader | Where |
|---|---|
| A Ball drop | First load of the app (`/app`, the board or Best Bets): a cobalt cover over the window |
| B Pitch lines | Research Area match view (`matchBody`) |
| C Last 10 | Research Area profile (`profileBody`) |
| D Price board | Best Bets (`bestBetsContainer`) |
| E Card outline | The matchday board (`fixturesContainer`) |
| F Rolling ball | Research Area home (`researchHome`) and the component's own start |

## Global Constraints

- Frontend files only: `Scripts/web_app/static/**`, `Scripts/tests/*_web.js`, the tracker and this plan.
- Every commit that changes a served file bumps its `?v=` tag; the new files and every file touched use
  `loaders-20261116a`, `b`, `c`… (one letter per task); research files go to `research-20261116q`, `r`….
- Commits: `Start loaders task N: <what>` (row "In progress"), `Done loaders task N: <what now works> (<tests>, D112)`
  with code, tests, the row "Done" and the tracker line. Stage files by name; no attribution lines.
- A loader shows only after 300 ms (a CSS delay, then a 200 ms fade), so a quick load never flashes. The cover (A)
  also stays at least 600 ms once shown.
- The status line names only what that request really loads. Screen readers hear one fixed line (`role="status"`);
  the stepping words are `aria-hidden`.
- Reduced motion: a still frame with the same words.
- Numbers in a loader (C's columns, D's prices) are shapes, not data: D's cells say no team and no bookmaker.
- Colours from the theme tokens only, light and dark; phone width 390 px with no sideways scroll.

## Review Focus

1. A load that finishes in under 300 ms shows nothing — test: the kit's wrapper carries the delay class.
2. A failed load still clears the cover (A) — test: `spixCover` hides on a rejected promise.
3. The board's background refresh (every 90 s) never shows a loader over cards already on screen — test: the
   loader is drawn only on the `loading` state, which background refreshes skip.
4. Reduced motion freezes every loader — test: the CSS has one `prefers-reduced-motion` block naming each kind.
5. The shadow root gets the styles — test: `research.js` fetches `loaders.css`.

## Tasks

### Task 1: The loader kit
- `static/loaders.js`: `spixLoader(kind, {label, steps})` for `ball`, `pitch`, `last10`, `prices`, `card`, `roll`;
  `spixCover` (A's timing: show after 300 ms, hold 600 ms, hide on settle or failure, give up after 15 s).
- `static/loaders.css`; both linked in `index.html` before `app.js`.
- Test: `Scripts/tests/test_loaders_web.js`.

### Task 2: A, the first-load cover
- `appModule.init()` covers the window with A while the first board load runs.

### Task 3: E on the board, D on Best Bets
- `_renderFixtures({loading})` draws E (three cards); `_renderBestBets({loading})` draws D.

### Task 4: B, C and F in the Research Area
- Match view B, profile C, home F, the component's start F; `research.js` fetches `loaders.css`.

### Task 5: Finish: real data, look, review
- Throttled loads on 8100 at 1440 and 390, light and dark; screenshots for the owner.

## Progress

| Task | Status | Commits | Tests |
|---|---|---|---|
| 1. The loader kit | Done | 27b37f0..this commit | loaders + all 18 *_web.js pass |
| 2. A, the first-load cover | Done | 8b289c2..this commit | loaders + all 18 *_web.js pass; 8100 with the API held 2.5 s |
| 3. E on the board, D on Best Bets | Done | 8d05f32..this commit | loaders + all 18 *_web.js pass; 8100 with the API held 2.5 s |
| 4. B, C and F in the Research Area | In progress | | |
| 5. Finish: real data, look, review | Not started | | |
