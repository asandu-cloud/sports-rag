# Protocol amendment: the market as a baseline (`market-baseline.v1`)

Amends `phase4-evaluation.v1` (28 Sep 2026). Added 15 Oct 2026 as rework step A1.6. Nothing in
the original protocol is removed; this adds a second reference to every comparison.

## Why

The original protocol compares each candidate only with the frozen control engine. A candidate
can then "qualify" while still being worse than the betting market, which is what the rework
anchors to (A3). Every research comparison now also reports the candidate against the market.

## What is compared

- **Events:** the 1X2 result (`winner`) and goals over/under 2.5 (`over2.5`), taken from the
  candidate's goal forecast (the same joint score distribution that produces every goal market).
  Corners, shots on target and BTTS have no market price in the history; handicap forecasts exist
  only at fixed lines, unlike the market's line. They are not compared in v1.
- **Market probability:** football-data.co.uk closing prices (`Index/football_data.sqlite3`) for
  fixtures linked to our fixture ids with agreeing scores (2019/20 onward). Each individual
  bookmaker's margin is removed proportionally and the median is taken across bookmakers
  (`Scripts/data_platform/reference_price.py`, the same method as the live reference price).
  Aggregates stand in only when no single bookmaker priced the market; maxima never do; Pinnacle
  prices flagged suspect (2025/26 onward) are excluded. `best_available` timing may add
  pre-closing prices and labels them; primary comparisons use closing prices.
- **Score:** log loss of the observed outcome, forecast minus market, on identical fixtures.
  Uncertainty: the protocol's 2,000-draw paired calendar-week bootstrap (seed 20260928), via
  `phase4_qualification.paired_inference`. Fixtures without a market price are reported as
  coverage loss, never silently dropped.

Code: `Scripts/data_platform/features/market_baseline.py` (`market_probabilities`,
`compare_with_market`); report command `python -m Scripts.ops.market_baseline_report`.

## How it gates the rework

- **A3 (market anchor):** the blended engine must beat the market alone on unseen matchweeks:
  the 95% interval of the gap entirely below zero for `winner` and `over2.5`, or equal log loss
  with better calibration.
- **A6 (fitted team strength):** a component is kept only if it improves the gap to the market
  on unseen seasons, in addition to passing the original criteria against the A3 engine.
- **All other candidates:** the market comparison is reported alongside the control comparison.
  It does not replace the original criteria.

Reserved-period rules are unchanged: computing a market comparison on a period counts as an
access of that period and must be logged as the protocol requires.

## First diagnostic (2023 development period, already inspected)

`Research/market-baseline-2026-10-15/`: the saved 2023 finalist forecasts, 2,783 of 3,095 goal
fixtures with a market price (the rest mostly European cups), 41 weeks.

| Forecaster | 1X2 vs market | Over/under 2.5 vs market |
|---|---|---|
| Frozen control | +4.36% log loss (95% gap +0.033 to +0.051) | +2.05% (+0.009 to +0.019) |
| Component stack | +2.92% (+0.021 to +0.036) | +1.49% (+0.006 to +0.015) |
| Supported stack | +3.69% (+0.027 to +0.044) | +1.43% (+0.006 to +0.014) |

Every forecaster, including the candidates that passed qualification against the control, is
clearly worse than the market at closing.
