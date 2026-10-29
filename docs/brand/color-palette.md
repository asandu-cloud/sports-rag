# Spix colour palette

The colours the website uses today (`Scripts/web_app/static/style.css`, light theme "Chalk" and dark theme "Floodlit"),
and the roles proposed for the new text-only logo. Logo concepts: https://claude.ai/artifact/Nj3bB4vjm9pdEFmThZdVxZ ("Spix Logo Concepts", private).

Status, 29 Oct 2026: logo chosen and on the website (D48). Website: the "Pitch" colours (white letters and a classic
football on a turf-green header, in both themes), "Landing" design, Fraunces, Classic ball. Discord avatar: the full
name on a turf tile (`docs/brand/logo/spix-discord-avatar.png`, 512 px). Logo files and the script that builds them:
`docs/brand/logo/`. The round-3 ball options below stay on record; the website uses Classic.

Website header colours (`--tb-*` in `style.css`, the same in light and dark): bar Turf `#0E6336`, border `#0B5530`,
logo and active text `#FAFDFA` (7.2:1), other links `#D3E9D9` (5.6:1), button White on Turf text. The logo colours are proposals until a design and font are picked; the website
colours are the live values. If `style.css` changes, update this file in the same commit.

## Logo colours (proposed)

Round 2 (29 Oct): Fraunces, "Landing" design, with a classic black-and-white football as the dot on the i.
The logo uses two colours: the letters, and the football (white with Ink patches).

| Name | Hex | RGB | Role |
|---|---|---|---|
| Ink | `#111E17` | 17, 30, 23 | Letters on light backgrounds; the football's patches and outline |
| White | `#FFFFFF` | 255, 255, 255 | The football on light backgrounds |
| Chalk | `#EFF1EC` | 239, 241, 236 | Letters and the football on dark backgrounds |
| Turf | `#0E6336` | 14, 99, 54 | Icon tile; optional green patches instead of Ink |

Approved pairings (contrast ratio, WCAG; 3:1 is the minimum for graphics, 4.5:1 for text):

| Letters / ball | Background | Ratio |
|---|---|---|
| Ink letters, Ink patches | Surface `#FCFEFC` | 17.0 |
| Turf patches (option) | Surface `#FCFEFC` | 7.3 |
| Chalk letters, Chalk ball | Dark canvas `#0D1511` | 16.3 |
| Chalk letters, Chalk ball | Dark surface `#151E1A` | 15.0 |
| White letters and ball | Turf tile `#0E6336` | 7.2 |

### Ball colour options (round 3, 29 Oct)

The ball is drawn round (a real football shape turned towards the viewer) with a flat shadow on the lower right and a
small highlight on the upper left. Colour goes in one place only, so it stays a black-and-white football:

| Option | Patches | Front patch | Shadow | Highlight |
|---|---|---|---|---|
| Green heart (recommended) | Ink `#111E17` | Turf `#0E6336` | Turf at 20% | White at 85% |
| Classic | Ink | Ink | Ink at 16% | White at 85% |
| Turf patches | Turf | Turf | Turf at 22% | White at 85% |
| Grass shadow | Ink | Ink | Grass `#47AA62` at 50% | White at 85% |
| Floodlit | Turf | Turf | Turf at 20% | Floodlight `#F0E488` |

`#47AA62` is the site's dark-theme chart green (`--chart-1`). On light backgrounds the white football needs its Ink
outline. One-colour version: everything in Ink on light, or
everything in Chalk on dark (the patches knock out to the background).

## Website, light theme (Chalk)

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#F3F7F3` | Page canvas |
| `--bg2` | `#E7EDE8` | Sunk wells, footer |
| `--surface` | `#FCFEFC` | Cards, panels |
| `--surface2` | `#ECF2ED` | Quiet fills, chips |
| `--border` | `#CFD7D0` | Lines |
| `--border2` | `#808981` | Inputs, controls |
| `--text` | `#111E17` | Main text |
| `--text2` | `#404B44` | Secondary text |
| `--text3` | `#5D6660` | Labels |
| `--accent` | `#0E6336` | Turf: actions and selection only |
| `--accent-dim` | `#DAF1DD` | Green wash |
| `--marker` | `#F0E488` | Floodlight highlighter (one price) |
| `--amber` | `#905100` | Caution |
| `--loss` | `#AA342C` | Losses |
| `--info` | `#2D5F87` | Information |

## Website, dark theme (Floodlit)

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0D1511` | Page canvas |
| `--bg2` | `#090F0C` | Sunk wells, footer |
| `--surface` | `#151E1A` | Cards, panels |
| `--surface2` | `#1C2521` | Quiet fills, chips |
| `--border` | `#2B3631` | Lines |
| `--border2` | `#5D6A63` | Inputs, controls |
| `--text` | `#EFF1EC` | Main text |
| `--text2` | `#C1C6C0` | Secondary text |
| `--text3` | `#99A19A` | Labels |
| `--accent` | `#89CD9B` | Pitch green: actions and selection only |
| `--accent-dim` | `#1A3424` | Green wash |
| `--marker` | `#E7DB79` | Floodlight highlighter (one price) |
| `--amber` | `#F2B966` | Caution |
| `--loss` | `#F4928A` | Losses |
| `--info` | `#98C3E8` | Information |
