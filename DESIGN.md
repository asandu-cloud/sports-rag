---
name: Spix
description: Football Match Reads on a careful analyst's matchday sheet. Chalk by day, Floodlit by night.
colors:
  chalk-canvas: "#F3F7F3"
  chalk-well: "#E7EDE8"
  chalk-surface: "#FCFEFC"
  chalk-quiet-fill: "#ECF2ED"
  chalk-hover: "#F0F5F1"
  chalk-rule: "#CFD7D0"
  chalk-control-rule: "#808981"
  chalk-soft-rule: "#DEE5DF"
  ink: "#111E17"
  ink-secondary: "#404B44"
  ink-tertiary: "#5D6660"
  turf: "#0E6336"
  turf-deep: "#10512D"
  on-turf: "#FAFDFA"
  turf-wash: "#DAF1DD"
  turf-wash-strong: "#C6E6CB"
  floodlight-marker: "#F0E488"
  caution-amber: "#905100"
  caution-amber-wash: "#FBE9CA"
  loss-red: "#AA342C"
  loss-red-wash: "#FFE4E1"
  away-blue: "#2D5F87"
  away-blue-wash: "#E0EDF9"
  floodlit-ground: "#0D1511"
  floodlit-well: "#090F0C"
  floodlit-surface: "#151E1A"
  floodlit-quiet-fill: "#1C2521"
  floodlit-hover: "#19231E"
  floodlit-rule: "#2B3631"
  floodlit-control-rule: "#5D6A63"
  floodlit-soft-rule: "#202925"
  chalk-ink: "#EFF1EC"
  chalk-ink-secondary: "#C1C6C0"
  chalk-ink-tertiary: "#99A19A"
  floodlit-turf: "#89CD9B"
  floodlit-turf-hover: "#A1DFB1"
  floodlit-on-turf: "#091A11"
  floodlit-turf-wash: "#1A3424"
  floodlit-turf-wash-strong: "#20462F"
  floodlit-marker: "#E7DB79"
  floodlit-amber: "#F2B966"
  floodlit-amber-wash: "#3A2B16"
  floodlit-loss: "#F4928A"
  floodlit-loss-wash: "#402624"
  floodlit-away-blue: "#98C3E8"
  floodlit-away-blue-wash: "#21303C"
typography:
  display:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "clamp(3.25rem, 7vw, 6rem)"
    fontWeight: 850
    lineHeight: 0.92
    letterSpacing: "0"
    fontVariation: "'wdth' 68"
  headline:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "clamp(2.25rem, 4vw, 3rem)"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.005em"
    fontVariation: "'wdth' 74"
  title:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "0.01em"
    fontVariation: "'wdth' 74"
  figure:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.1
    fontFeature: "'tnum' 1"
    fontVariation: "'wdth' 85"
  body-reading:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.6
  body:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  ui:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: 1.45
  label:
    fontFamily: "Archivo, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0.05em"
    fontVariation: "'wdth' 87"
rounded:
  mark: "2px"
  chip: "4px"
  control: "6px"
  panel-inset: "8px"
  card: "10px"
  panel: "12px"
  sheet: "14px"
  pill: "999px"
spacing:
  xxs: "4px"
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
  gutter: "24px"
  gutter-mobile: "16px"
  topbar: "60px"
  slip-width: "340px"
  max-width: "1280px"
components:
  button-primary:
    backgroundColor: "{colors.turf}"
    textColor: "{colors.on-turf}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.turf-deep}"
    textColor: "{colors.on-turf}"
  button-outline:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.turf}"
    rounded: "{rounded.control}"
    padding: "8px 16px 8px 12px"
    height: "44px"
  button-outline-hover:
    backgroundColor: "{colors.turf-wash}"
    textColor: "{colors.turf}"
  button-outline-added:
    backgroundColor: "{colors.turf}"
    textColor: "{colors.on-turf}"
  chip-date:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.ink-secondary}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "8px 16px"
    height: "44px"
  chip-date-active:
    backgroundColor: "{colors.turf}"
    textColor: "{colors.on-turf}"
  nav-link:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "8px 14px"
    height: "44px"
  nav-link-active:
    backgroundColor: "{colors.turf-wash}"
    textColor: "{colors.turf}"
  card-fixture:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.card}"
    padding: "16px"
  panel-match-read:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "24px"
  figure-mark:
    backgroundColor: "{colors.floodlight-marker}"
    textColor: "{colors.ink}"
    rounded: "{rounded.mark}"
    padding: "0 0.22em"
  status-ready:
    backgroundColor: "{colors.turf-wash}"
    textColor: "{colors.turf}"
    typography: "{typography.label}"
    rounded: "{rounded.chip}"
    padding: "3px 8px"
    height: "26px"
  status-preliminary:
    backgroundColor: "{colors.away-blue-wash}"
    textColor: "{colors.away-blue}"
    rounded: "{rounded.chip}"
    padding: "3px 8px"
    height: "26px"
  status-unavailable:
    backgroundColor: "{colors.caution-amber-wash}"
    textColor: "{colors.caution-amber}"
    rounded: "{rounded.chip}"
    padding: "3px 8px"
    height: "26px"
  status-no-bet:
    textColor: "{colors.ink}"
    rounded: "{rounded.chip}"
    padding: "3px 8px"
    height: "26px"
  confidence-high:
    backgroundColor: "{colors.turf-wash}"
    textColor: "{colors.turf}"
    rounded: "{rounded.chip}"
    padding: "2px 8px"
    height: "24px"
  confidence-medium:
    backgroundColor: "{colors.caution-amber-wash}"
    textColor: "{colors.caution-amber}"
    rounded: "{rounded.chip}"
    padding: "2px 8px"
    height: "24px"
  confidence-low:
    backgroundColor: "{colors.chalk-quiet-fill}"
    textColor: "{colors.ink-tertiary}"
    rounded: "{rounded.chip}"
    padding: "2px 8px"
    height: "24px"
  input:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
    height: "48px"
---

# Design System: Spix

## Overview

**Creative North Star: "The Analyst's Matchday Sheet"**

Spix reads like a careful analyst's matchday sheet: ruled paper, condensed team names written in heavy capitals, small plain notes underneath, and exactly one figure per read run over with a highlighter. The world has two lights. Chalk is the daytime sheet, a cool green-tinted off-white with deep green-black ink. Floodlit is the same sheet under stadium lights, a green-black ground with chalk ink. Both are the same system with the same roles; the reader picks, or the operating system picks until they do.

The surface is flat and quietly ruled. Structure comes from 1px rules and tonal wells, not from shadows. A single deep turf green carries every action and every selected state, so "where can I act" is answered by one colour. Passing is styled as deliberately as picking: the no-bet state is an ink-outlined chip, never a greyed-out or disabled one. Density is moderate and phone-first; every tap target is at least 44px.

The world explicitly refuses the sportsbook default of neon on dark, glows, and soft decorative gradients.

**Key Characteristics:**
- Two themes, one role map: Chalk (light) is every visitor's default; Floodlit (dark) applies only when chosen with the theme toggle (`data-theme="dark"`, remembered per browser). The device's own dark setting is deliberately ignored.
- One action colour (turf green) for buttons, links, selection and active states only.
- One floodlight-yellow marker behind the single key figure of a read.
- Archivo only: condensed heavy capitals for names and titles, normal width for reading, tabular figures for numbers.
- Flat surfaces, 1px rules, tonal wells; shadows only on things that float.

## Colors

A near-monochrome green-tinted paper-and-ink palette with one turf accent, one highlighter, and three quiet semantic washes.

### Primary
- **Turf** (Chalk `turf`, Floodlit `floodlit-turf`): primary buttons, links, active date chip, active nav and league underline, add-to-slip, focus rings, slip counts and the Home segment of the balance bar. Hover deepens in Chalk (`turf-deep`) and lightens in Floodlit (`floodlit-turf-hover`).
- **Turf Wash** (`turf-wash`, `turf-wash-strong`): the tinted ground for active nav, slip toggle, "ready"/"updated" status, high-confidence and strong-value chips, and the recommended market line.

### Secondary
- **Floodlight Marker** (`floodlight-marker`, `floodlit-marker`): the highlighter. Behind the one marked figure in a read (projected total goals), the landing hero's one marked word, the Research Area's main insight figure, and text selection. Ink on marker in both themes.

### Tertiary
- **Away Blue** (`away-blue` + wash): the Away segment of the balance bar, the "preliminary" status, and Research chart bars. It is a data colour, not an action colour.
- **Caution Amber** (`caution-amber` + wash): medium confidence, "unavailable" status, slip quality warnings, and the info-page notice panel.
- **Loss Red** (`loss-red` + wash): errors, destructive hover (clear slip, remove leg). Never used for "no bet".

### Neutral
- **Chalk Canvas / Floodlit Ground** (`chalk-canvas` / `floodlit-ground`): page background; also the browser `theme-color`.
- **Chalk Surface / Floodlit Surface**: cards, panels, modal, slip, inputs.
- **Chalk Well / Floodlit Well**: sunk areas: footer, shared-caveats panel, Research filter panels.
- **Quiet Fill**: low-emphasis chips, pending status, alternatives list, disabled primary.
- **Rules**: `chalk-rule` for card and section borders, `chalk-soft-rule` for inner dividers and crest rings, `chalk-control-rule` (3.3:1) for inputs, outlined controls, list bullets and the Draw segment.
- **Ink tiers**: `ink` (primary text, team names), `ink-secondary` (supporting copy), `ink-tertiary` (meta, labels, timestamps). Floodlit mirrors them as `chalk-ink` tiers.

### Named Rules
**The One Turf Rule.** Turf green means "you can act here" or "this is selected". It never decorates, never colours a heading, and never signals a positive outcome on its own.

**The One Highlighter Rule.** Each read carries at most one marked figure. If two numbers seem to need the marker, the read has not decided what matters.

**The Pass Is Not Grey Rule.** No-bet states use full ink text inside a 1px control-rule outline on a transparent ground. Never dim, strike or disable them.

**The Match Card Tints (29 Oct, owner).** The board's match card tints its four section tiles: Fair prices turf wash, Stats and form away-blue wash, Spix vs market caution-amber wash, Head to head quiet fill; the open tile is solid turf. Recent form uses turf wash for wins, quiet fill for draws and loss-red wash for losses. Here the washes label sections and results; they do not carry their status meanings.

## Typography

**Display Font:** Archivo (variable, self-hosted, width 62–125%, weight 100–900) with system-ui fallback
**Body Font:** Archivo at normal width
**Label/Mono Font:** none distinct; numbers use Archivo tabular figures (`--mono` is a legacy alias of the sans)

**Character:** One family doing two jobs by width. Condensed heavy capitals (74% width, 800) feel like names chalked on a team sheet; normal-width regular text reads calmly at length.

### Hierarchy
- **Display** (850, clamp(3.25rem, 7vw, 6rem), 0.92, 68% width, uppercase): landing hero only. Info-page h1 uses a sibling setting (850, clamp(2.5rem, 6vw, 4rem), 70% width).
- **Headline** (800, clamp(2.25rem, 4vw, 3rem), 1.0, 74% width, uppercase): app page titles ("Matchday"), landing section heads, Match Read team title (up to 3.5rem).
- **Title** (800, 22px, 1.1, 74% width, uppercase, 0.01em): team names on fixture cards (20px on mobile), league names (20px), slip title, info h2 (26px).
- **Figure** (800, 22px, 85% width, tabular): odds, payouts, example stats. Selection picks use 22px at 80% width without uppercase.
- **Body Reading** (400, 18px, 1.6, max 68ch): the Match Read thesis; info pages use 17px/1.7 at 72ch.
- **Body** (400, 16px, 1.5): default text, subtitles (max 65ch).
- **UI** (500–700, 13–15px): nav, chips, buttons, signals list (14px/1.45), meta.
- **Label** (700, 12px, 0.05em, 87% width, uppercase): the single uppercase voice for small block labels (market card titles, "Applies to every selection", alternatives title, auth divider, league badge).

### Named Rules
**The Width Carries Rank Rule.** Hierarchy is set by width and weight, not by colour. Names and titles are condensed and heavy; anything meant to be read at length is normal width.

**The Tabular Numbers Rule.** Every number that can sit beside another (times, odds, percentages, counts, stakes) uses tabular figures.

**The 12px Floor Rule.** No text renders below 12px, including labels and badges.

## Layout

A centred 1280px column with 24px gutters (16px under 768px) under a fixed 60px translucent top bar. The Matchday board flows league groups into CSS columns (360px column width, 24px gap) so short leagues sit side by side; each group is a header row with a 1px rule, then stacked fixture cards 12px apart. Below the page title come a horizontally scrolling date strip and a league tab row with an animated 3px turf underline and edge fades when scrollable.

Match Read detail narrows to 880px; reading measures cap at 65–72ch. The bet slip is a 340px panel fixed to the right edge on desktop that reserves only the space it overlaps; under 1100px it becomes a bottom sheet (max 80dvh) with a 56px summary bar. Under 640px the top nav moves to a fixed bottom nav (48px targets) and the slip toggle moves to the bottom bar. Landing uses a two-column intro (1.1fr / 1fr, 72px gap) that stacks under 760px.

Spacing steps are 4, 6, 8, 12, 16, 20, 24, 32px, with larger section breathing room (48–72px) on the landing page. Minimum interactive size is 44px everywhere; primary form controls are 48px.

## Elevation & Depth

Flat by default. Depth is conveyed by tone (canvas, surface, well) and by 1px rules. Cards, panels and chips carry no shadow at rest or on hover; hover strengthens the border to the control rule instead. Shadows exist only on elements that float above the page: the sign-in modal, toasts, and the mobile bottom-sheet slip. The top bar and bottom nav are translucent with a 16px backdrop blur, falling back to solid canvas under reduced transparency or increased contrast.

### Shadow Vocabulary
- **Float** (`box-shadow: 0 8px 16px rgba(17,30,23,0.08), 0 24px 48px rgba(17,30,23,0.14)`; Floodlit `0 8px 16px rgba(0,0,0,0.35), 0 24px 48px rgba(0,0,0,0.45)`): modal, toast, mobile slip sheet only.
- **Focus ring** (`outline: 2px solid` turf, 2px offset; inputs use a turf border plus `0 0 0 1px` turf): keyboard focus everywhere.

### Named Rules
**The Only Floating Things Cast Shadows Rule.** If it scrolls with the page, it is flat. A shadow means "this is above the sheet".

## Shapes

Gently softened rectangles on a tight radius ladder that grows with the size of the object: 2px for the marker, 4px for chips and badges, 6px for every control (buttons, inputs, tabs, nav links), 8px for inset panels and slip legs, 10px for fixture cards and rows, 12px for the Match Read panel, modal and landing panels, 14px for the top of the mobile sheet. Crests are circles on a white (Chalk) or chalk (Floodlit) disc with a soft-rule inner ring; league logos sit on a 4px rounded tile. List bullets are 5px squares with a 1px radius, not dots. Rules are 1px solid; the one dashed rule separates released selections from "other model angles". The landing benefits use a 2px ink top rule as their only heavier stroke.

## Components

### Buttons
Plain, confident, turf-filled; no gradients, no lift.
- **Shape:** softened corners (6px).
- **Primary:** turf ground, on-turf text, 700 weight; 48px tall in forms and slip, 52px on the landing CTA, 40px in the top bar. Trailing arrow icons nudge 3px right on hover.
- **Hover / Focus:** background deepens to turf-deep (Chalk) or lightens (Floodlit), 150ms; 2px turf focus outline.
- **Outline (Add to slip):** surface ground, turf text and 1px turf border; hover fills turf wash; added state fills turf with a check icon.
- **Icon square:** 44px, control-rule border, plus icon; same hover/added logic.
- **Disabled:** quiet fill with tertiary ink, not-allowed cursor.

### Chips
- **Date and market tabs:** surface ground, 1px rule border, secondary ink; hover strengthens border; active is solid turf with on-turf text.
- **Status (fixture card footer):** 4px radius, 12px bold. Ready = turf wash, Preliminary = away-blue wash, Unavailable = amber wash, Pending = quiet fill, No bet = ink text in a 1px control-rule outline.
- **Confidence:** High turf wash, Medium amber wash, Low quiet fill, No bet outlined; 87% width with slight tracking.
- **Selection role:** outlined; "Core" uses a 1.5px ink outline and 800 weight.

### Cards / Containers
- **Corner Style:** 10px (fixture cards, rows, market cards); 12px (Match Read panel, landing panels).
- **Background:** surface on canvas.
- **Shadow Strategy:** none (see Elevation).
- **Border:** 1px rule; hover strengthens to control rule; keyboard focus on the inner link draws a turf border plus 1px turf ring around the whole card.
- **Internal Padding:** 16px fixture card (14px mobile), 24px Match Read panel (18px mobile), 28px landing panels.

### Inputs / Fields
- **Style:** 48px, surface ground, 1px control-rule border, 6px radius, 16px text, tertiary placeholder.
- **Focus:** border turns turf plus a 1px turf ring; caret is turf.
- **Error:** loss-red text on loss wash in a 6px panel.

### Navigation
- **Top bar:** fixed 60px, translucent canvas with blur, 1px bottom rule. Logo is the slanted two-eyed "S" mark in turf beside an 800 condensed uppercase wordmark (wordmark hides under 420px). Nav links are 14px/500 secondary ink; hover adds a faint ink wash; active is turf text on turf wash at 600.
- **League tabs:** text tabs with crest tiles, a turf underline that slides between tabs, a vertical divider after "All", and edge fades plus arrow buttons when the row overflows.
- **Mobile:** bottom nav bar with equal-width 48px links; theme toggle is a 44px icon button showing moon in Chalk and sun in Floodlit.

### Fixture Card (signature)
Kickoff time and "Updated" meta on top; two rows of crest plus condensed uppercase team name; a soft-rule divider; a square-bulleted signals list where the projected total goals is the one marked figure; the Home/Draw/Away balance bar (8px, 2px gaps, turf / control rule / away blue, with a tabular legend); then status chip and a turf "View Match Read" link with an arrow. Scheduled fixtures without a read keep the card shape with soft-rule border and secondary-ink names, no hover.

### Match Read Panel (signature)
A 12px surface panel: status label, the thesis at reading size with one marked figure, key points, then selections separated by 1px rules. Each selection row carries a condensed pick, role and confidence chips, the price (figure style) with the bookmaker in tertiary ink, and an outlined Add button; the explanation spans below at 68ch. Shared caveats sit in a sunk well under a Label heading; withheld angles follow a dashed rule as quiet-fill chips.

### Toast
Ink ground with canvas text (inverts per theme), 8px radius, Float shadow, a small status dot ringed in turf or loss red; slides up 10px and fades out after 2.2s.

## Do's and Don'ts

### Do:
- **Do** route every colour through the CSS custom properties so both Chalk and Floodlit stay correct; the only fixed colour is the Discord sign-in button (#5865F2, hover #4752C4).
- **Do** keep turf green for actions and selection only, and use the floodlight marker on one figure per read.
- **Do** set team names, page titles and league names in Archivo 800 at 74% width, uppercase; keep reading text at normal width.
- **Do** use tabular figures for every number and keep all text at 12px or larger.
- **Do** build depth from canvas, surface and well tones plus 1px rules; strengthen the border on hover instead of adding shadow.
- **Do** keep tap targets at 44px minimum (48px for primary form controls and mobile nav).
- **Do** style "no bet" as an ink-outlined, full-contrast state.
- **Do** respect reduced motion (no movement, colour and opacity feedback only) and reduced transparency (solid bars, no blur).
- **Do** apply the stored theme before first paint with the inline `spix-theme` snippet and dual `theme-color` metas on every page.

### Don't:
- **Don't** use neon accents on dark, glows, or soft decorative gradients. Hard-stop marker strokes and scroll-edge fades are the only gradients the world uses.
- **Don't** add shadows to cards, chips or anything that scrolls with the page.
- **Don't** introduce a second typeface, a monospace code face for numbers, or a system display face.
- **Don't** colour headings turf, or use turf to mean "good outcome".
- **Don't** grey out, strike through or disable "no bet".
- **Don't** put an uppercase kicker or eyebrow above a headline; the uppercase Label voice is for small block labels only.
- **Don't** mark more than one figure per read with the highlighter.
