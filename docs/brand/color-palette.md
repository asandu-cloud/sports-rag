# Spix colour palette

The colours the website uses today (`Scripts/web_app/static/style.css`, light theme "Chalk" and dark theme "Floodlit"),
and the roles proposed for the new text-only logo. Logo concepts: https://claude.ai/artifact/Nj3bB4vjm9pdEFmThZdVxZ ("Spix Logo Concepts", private).

Status, 29 Oct 2026: logo round 1. The logo colours are proposals until a design and font are picked; the website
colours are the live values. If `style.css` changes, update this file in the same commit.

## Logo colours (proposed)

The logo uses at most two colours at once: one for the letters, one for the ball on the i.

| Name | Hex | RGB | Role |
|---|---|---|---|
| Ink | `#111E17` | 17, 30, 23 | Letters on light backgrounds |
| Turf | `#0E6336` | 14, 99, 54 | The ball on light backgrounds; the brand green; icon tile |
| Chalk | `#EFF1EC` | 239, 241, 236 | Letters on dark backgrounds |
| Floodlight | `#E7DB79` | 231, 219, 121 | The ball on near-black or on Turf |
| Pitch green | `#89CD9B` | 137, 205, 155 | The ball on the dark site header |

Approved pairings (contrast ratio, WCAG; 3:1 is the minimum for graphics, 4.5:1 for text):

| Letters / ball | Background | Ratio |
|---|---|---|
| Ink letters | Surface `#FCFEFC` | 17.0 |
| Turf ball | Surface `#FCFEFC` | 7.3 |
| Turf ball | Canvas `#F3F7F3` | 6.8 |
| Chalk letters | Dark canvas `#0D1511` | 16.3 |
| Floodlight ball | Dark canvas `#0D1511` | 13.1 |
| Chalk letters | Dark surface `#151E1A` | 15.0 |
| Pitch green ball | Dark surface `#151E1A` | 9.1 |
| White `#FAFDFA` letters | Turf `#0E6336` | 7.2 |
| Floodlight `#F0E488` ball | Turf `#0E6336` | 5.7 |

One-colour version: everything in Ink on light, or everything in Chalk on dark.

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
