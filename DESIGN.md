---
name: Spix
description: Football Match Reads on a careful analyst's matchday sheet. Chalk by day, Floodlit by night.
colors:
  chalk-canvas: "#F2FFFA"
  chalk-well: "#E3F1EC"
  chalk-surface: "#FBFFFD"
  chalk-quiet-fill: "#E9F5F0"
  chalk-hover: "#EEF7F4"
  chalk-rule: "#CFD7D4"
  chalk-control-rule: "#808986"
  chalk-soft-rule: "#DEE5E2"
  ink: "#111E17"
  ink-secondary: "#404B44"
  ink-tertiary: "#5D6660"
  cobalt: "#0246A4"
  cobalt-deep: "#023A88"
  on-cobalt: "#F2FFFA"
  cobalt-wash: "#DCE5F2"
  cobalt-wash-strong: "#C7D6EB"
  lime-marker: "#D5F097"
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
  floodlit-cobalt: "#88B4F2"
  floodlit-cobalt-hover: "#ACCAF6"
  floodlit-on-cobalt: "#08152B"
  floodlit-cobalt-wash: "#1E2B30"
  floodlit-cobalt-wash-strong: "#283842"
  floodlit-marker: "#D5F097"
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
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.on-cobalt}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.cobalt-deep}"
    textColor: "{colors.on-cobalt}"
  button-outline:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.cobalt}"
    rounded: "{rounded.control}"
    padding: "8px 16px 8px 12px"
    height: "44px"
  button-outline-hover:
    backgroundColor: "{colors.cobalt-wash}"
    textColor: "{colors.cobalt}"
  button-outline-added:
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.on-cobalt}"
  chip-date:
    backgroundColor: "{colors.chalk-surface}"
    textColor: "{colors.ink-secondary}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "8px 16px"
    height: "44px"
  chip-date-active:
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.on-cobalt}"
  nav-link:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.ui}"
    rounded: "{rounded.control}"
    padding: "8px 14px"
    height: "44px"
  nav-link-active:
    backgroundColor: "{colors.cobalt-wash}"
    textColor: "{colors.cobalt}"
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
    backgroundColor: "{colors.lime-marker}"
    textColor: "{colors.ink}"
    rounded: "{rounded.mark}"
    padding: "0 0.22em"
  status-ready:
    backgroundColor: "{colors.cobalt-wash}"
    textColor: "{colors.cobalt}"
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
    backgroundColor: "{colors.cobalt-wash}"
    textColor: "{colors.cobalt}"
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

Spix reads like a careful analyst's matchday sheet: ruled paper, condensed team names written in heavy capitals, small plain notes underneath, and exactly one figure per read run over with a highlighter. The world has two lights. Chalk is the daytime sheet, mint cream with deep green-black ink. Floodlit is the same sheet under stadium lights, a green-black ground with chalk ink. Both are the same system with the same roles; the reader picks, or the operating system picks until they do.

The surface is flat and quietly ruled. Structure comes from 1px rules and tonal wells, not from shadows. A single cobalt blue carries every action and every selected state, so "where can I act" is answered by one colour. Passing is styled as deliberately as picking: the no-bet state is an ink-outlined chip, never a greyed-out or disabled one. Density is moderate and phone-first; every tap target is at least 44px.

The world explicitly refuses the sportsbook default of neon on dark, glows, and soft decorative gradients.

**Key Characteristics:**
- Two themes, one role map: Chalk (light) is every visitor's default; Floodlit (dark) applies only when chosen with the theme toggle (`data-theme="dark"`, remembered per browser). The device's own dark setting is deliberately ignored.
- One action colour (cobalt blue) for buttons, links, selection and active states only.
- One lime cream marker behind the single key figure of a read.
- Archivo only: condensed heavy capitals for names and titles, normal width for reading, tabular figures for numbers.
- Flat surfaces, 1px rules, tonal wells; shadows only on things that float.

## Colors

A near-monochrome mint-tinted paper-and-ink palette with one cobalt accent, one highlighter, and three quiet semantic washes.

### Primary
- **Cobalt** (Chalk `cobalt`, Floodlit `floodlit-cobalt`): primary buttons, links, active date chip, active nav and league underline, add-to-slip, focus rings, slip counts, the Home segment of the balance bar, Research chart bars, the pick-evidence bars and checks, and the top bar in both themes. Hover deepens in Chalk (`cobalt-deep`) and lightens in Floodlit (`floodlit-cobalt-hover`). Floodlit uses a light cobalt because cobalt itself reads at only 2.1:1 on the dark ground. Cobalt replaced the turf green on 9 Nov 2026 (D80): the green felt like tennis, and blue reads as trust.
- **Cobalt Wash** (`cobalt-wash`, `cobalt-wash-strong`): the tinted ground for active nav, slip toggle, "ready"/"updated" status, high-confidence and strong-value chips, and the recommended market line.

### Secondary
- **Lime Marker** (`lime-marker`, `floodlit-marker`, both lime cream): the highlighter. Behind the one marked figure in a read (projected total goals), the landing hero's one marked word, the Research Area's main insight figure, the count in a pick-evidence sheet, and text selection. Ink on marker in both themes.

### Tertiary
- **Away Blue** (`away-blue` + wash): the Away segment of the balance bar, the "preliminary" status and the Research "lenient" tag. It is a data colour, not an action colour. Since cobalt became the action colour (D80) the two blues sit close (1.3:1 in Chalk, 1.1:1 in Floodlit); a new Away colour is an open choice.
- **Caution Amber** (`caution-amber` + wash): medium confidence, "unavailable" status, slip quality warnings, and the info-page notice panel.
- **Loss Red** (`loss-red` + wash): errors, destructive hover (clear slip, remove leg). Never used for "no bet".

### Neutral
- **Chalk Canvas / Floodlit Ground** (`chalk-canvas` mint cream / `floodlit-ground`): page background. The browser `theme-color` is the top bar's cobalt.
- **Chalk Surface / Floodlit Surface**: cards, panels, modal, slip, inputs.
- **Chalk Well / Floodlit Well**: sunk areas: footer, shared-caveats panel, Research filter panels.
- **Quiet Fill**: low-emphasis chips, pending status, alternatives list, disabled primary.
- **Rules**: `chalk-rule` for card and section borders, `chalk-soft-rule` for inner dividers and crest rings, `chalk-control-rule` (3.3:1) for inputs, outlined controls, list bullets and the Draw segment.
- **Ink tiers**: `ink` (primary text, team names), `ink-secondary` (supporting copy), `ink-tertiary` (meta, labels, timestamps). Floodlit mirrors them as `chalk-ink` tiers.

### Named Rules
**The One Cobalt Rule.** Cobalt means "you can act here" or "this is selected". It never decorates, never colours a heading, and never signals a positive outcome on its own.

**The One Highlighter Rule.** Each read carries at most one marked figure. If two numbers seem to need the marker, the read has not decided what matters.

**The Pass Is Not Grey Rule.** No-bet states use full ink text inside a 1px control-rule outline on a transparent ground. Never dim, strike or disable them.

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

A centred 1280px column with 24px gutters (16px under 768px) under a fixed 60px cobalt top bar. The Matchday board flows league groups into CSS columns (360px column width, 24px gap) so short leagues sit side by side; each group is a header row with a 1px rule, then stacked fixture cards 12px apart. Below the page title come a horizontally scrolling date strip and a league tab row with an animated 3px cobalt underline and edge fades when scrollable.

Match Read detail narrows to 880px; reading measures cap at 65–72ch. The bet slip is a 340px panel fixed to the right edge on desktop that reserves only the space it overlaps; under 1100px it becomes a bottom sheet (max 80dvh) with a 56px summary bar. Under 640px the top nav moves to a fixed bottom nav (48px targets) and the slip toggle moves to the bottom bar. Landing uses a two-column intro (1.1fr / 1fr, 72px gap) that stacks under 760px.

Spacing steps are 4, 6, 8, 12, 16, 20, 24, 32px, with larger section breathing room (48–72px) on the landing page. Minimum interactive size is 44px everywhere; primary form controls are 48px.

## Elevation & Depth

Flat by default. Depth is conveyed by tone (canvas, surface, well) and by 1px rules. Cards, panels and chips carry no shadow at rest or on hover; hover strengthens the border to the control rule instead. Shadows exist only on elements that float above the page: the sign-in modal, toasts, and the mobile bottom-sheet slip. The top bar and bottom nav are translucent with a 16px backdrop blur, falling back to solid canvas under reduced transparency or increased contrast.

### Shadow Vocabulary
- **Float** (`box-shadow: 0 8px 16px rgba(17,30,23,0.08), 0 24px 48px rgba(17,30,23,0.14)`; Floodlit `0 8px 16px rgba(0,0,0,0.35), 0 24px 48px rgba(0,0,0,0.45)`): modal, toast, mobile slip sheet only.
- **Focus ring** (`outline: 2px solid` cobalt, 2px offset; inputs use a cobalt border plus `0 0 0 1px` cobalt): keyboard focus everywhere.

### Named Rules
**The Only Floating Things Cast Shadows Rule.** If it scrolls with the page, it is flat. A shadow means "this is above the sheet".

## Shapes

Gently softened rectangles on a tight radius ladder that grows with the size of the object: 2px for the marker, 4px for chips and badges, 6px for every control (buttons, inputs, tabs, nav links), 8px for inset panels and slip legs, 10px for fixture cards and rows, 12px for the Match Read panel, modal and landing panels, 14px for the top of the mobile sheet. Crests are circles on a white (Chalk) or chalk (Floodlit) disc with a soft-rule inner ring; league logos sit on a 4px rounded tile. List bullets are 5px squares with a 1px radius, not dots. Rules are 1px solid; the one dashed rule separates released selections from "other model angles". The landing benefits use a 2px ink top rule as their only heavier stroke.

## Components

### Buttons
Plain, confident, cobalt-filled; no gradients, no lift.
- **Shape:** softened corners (6px).
- **Primary:** cobalt ground, on-cobalt text, 700 weight; 48px tall in forms and slip, 52px on the landing CTA, 40px in the top bar. Trailing arrow icons nudge 3px right on hover.
- **Hover / Focus:** background deepens to cobalt-deep (Chalk) or lightens (Floodlit), 150ms; 2px cobalt focus outline.
- **Outline (Add to slip):** surface ground, cobalt text and 1px cobalt border; hover fills cobalt wash; added state fills cobalt with a check icon.
- **Icon square:** 44px, control-rule border, plus icon; same hover/added logic.
- **Disabled:** quiet fill with tertiary ink, not-allowed cursor.

### Chips
- **Date and market tabs:** surface ground, 1px rule border, secondary ink; hover strengthens border; active is solid cobalt with on-cobalt text.
- **Status (fixture card footer):** 4px radius, 12px bold. Ready = cobalt wash, Preliminary = away-blue wash, Unavailable = amber wash, Pending = quiet fill, No bet = ink text in a 1px control-rule outline.
- **Confidence:** High cobalt wash, Medium amber wash, Low quiet fill, No bet outlined; 87% width with slight tracking.
- **Selection role:** outlined; "Core" uses a 1.5px ink outline and 800 weight.

### Cards / Containers
- **Corner Style:** 10px (fixture cards, rows, market cards); 12px (Match Read panel, landing panels).
- **Background:** surface on canvas.
- **Shadow Strategy:** none (see Elevation).
- **Border:** 1px rule; hover strengthens to control rule; keyboard focus on the inner link draws a cobalt border plus 1px cobalt ring around the whole card.
- **Internal Padding:** 16px fixture card (14px mobile), 24px Match Read panel (18px mobile), 28px landing panels.

### Inputs / Fields
- **Style:** 48px, surface ground, 1px control-rule border, 6px radius, 16px text, tertiary placeholder.
- **Focus:** border turns cobalt plus a 1px cobalt ring; caret is cobalt.
- **Error:** loss-red text on loss wash in a 6px panel.

### Navigation
- **Top bar:** fixed 60px, solid cobalt in both themes with a 1px deeper-cobalt bottom rule (`--tb-*` tokens; `?logo=old` brings back the old translucent bar). Logo is the "spix" wordmark in mint cream with a classic football as the dot on the i (D48). Nav links are 14px/500 in pale blue (`#D0DDF2`); hover adds a faint mint wash; active is mint cream text on a translucent mint ground at 600.
- **League tabs:** text tabs with crest tiles, a cobalt underline that slides between tabs, a vertical divider after "All", and edge fades plus arrow buttons when the row overflows.
- **Mobile:** bottom nav bar with equal-width 48px links; theme toggle is a 44px icon button showing moon in Chalk and sun in Floodlit.

### Fixture Card (signature)
Kickoff time and "Updated" meta on top; two rows of crest plus condensed uppercase team name; a soft-rule divider; a square-bulleted signals list where the projected total goals is the one marked figure; the Home/Draw/Away balance bar (8px, 2px gaps, cobalt / control rule / away blue, with a tabular legend); then status chip and a cobalt "View Match Read" link with an arrow. Scheduled fixtures without a read keep the card shape with soft-rule border and secondary-ink names, no hover.

### Match Read Panel (signature)
A 12px surface panel: status label, the thesis at reading size with one marked figure, key points, then selections separated by 1px rules. Each selection row carries a condensed pick, role and confidence chips, the price (figure style) with the bookmaker in tertiary ink, and an outlined Add button; the explanation spans below at 68ch. Shared caveats sit in a sunk well under a Label heading; withheld angles follow a dashed rule as quiet-fill chips.

### Pick Evidence (D83)
From the owner's approved mockup ("Easier to read", 9 Nov), on the match page only, and only when the card carries
`pick-evidence.v1` (`SPIX_PICK_EVIDENCE`); without it the page is as before.
- **How Spix reads it:** a 10px quiet-fill panel above Most likely. A 13px tertiary caption, a bold 18px lead
  sentence, then a three-column list (one column under 600px) of the chance of each result, goals expected and
  corners expected, each figure once, the dots in tertiary ink. Cup cards get this panel only.
- **On each Most likely card:** the "why" line gives way to a bold 16px verdict and a 15px secondary detail line,
  then a soft rule and "Past matches at similar odds" with up to three rows (home team, away team, league). Each row
  is a full-width button: the group's name and price range on the left, "won of N" on the right with a chevron, and a
  4px bar (cobalt on the chart track) beneath. Hover is the surface-hover fill.
- **The sheet:** a native modal dialog (14px radius, Lift shadow, overlay backdrop), up to 680px wide. A label
  naming the pick, a condensed uppercase title, one plain sentence on how the group was chosen, the count marked in
  lime ("9 of 12") with what it means, then a table of the matches: date and season, the match from the group's side
  ("v" at home, "at" away), the price, the goals, corners or score, and Yes with a cobalt check or No in tertiary ink.
  Under 560px the date moves under the match and the last column is headed "Won?", so nothing scrolls sideways.

### Toast
Ink ground with canvas text (inverts per theme), 8px radius, Float shadow, a small status dot ringed in cobalt or loss red; slides up 10px and fades out after 2.2s.

## Do's and Don'ts

### Do:
- **Do** route every colour through the CSS custom properties so both Chalk and Floodlit stay correct; the only fixed colour is the Discord sign-in button (#5865F2, hover #4752C4).
- **Do** keep cobalt for actions and selection only, and use the lime marker on one figure per read.
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
- **Don't** colour headings cobalt, or use cobalt to mean "good outcome".
- **Don't** grey out, strike through or disable "no bet".
- **Don't** put an uppercase kicker or eyebrow above a headline; the uppercase Label voice is for small block labels only.
- **Don't** mark more than one figure per read with the highlighter.
