# Fixed card backtest from reconstructed player evidence

2 October 2026. Recorded before scoring this run. The owner authorized dated
team/referee inputs and the fixed card backtest, stopping before optimization.

## Frozen method and inputs

Use the sealed `Research/phase4-player-card-reconstruction-2026-10-02` targets.
The earlier reconciliation certified explicit player counts against lineups and
card events; this runner validates their contract and internal arithmetic, not
their original completeness anew. Preserve all target exclusions and source
references. Archive IDs refer to the prepared collection and must retain its
batch identifier; they are not canonical database IDs.

Apply the [original fixed protocol](phase4-card-reconstruction-protocol-2026-09-30.md)
and [two-minute amendment](phase4-card-policy-v2-protocol-2026-09-30.md):

- Targets and histories use participation v2, recorded minutes >= 2, regulation
  time, explicit yellow/red counts and the agreed ordinary league/European
  group-stage scope. No unresolved targets enter profiles or scores.
- Only kickoff + three hours strictly before 1 January 2024 is admitted.
  Calendar 2022 is the support period (500 fixtures / 26 ISO weeks); calendar
  2023 is evaluation (200 / 20). No parameters are fitted in this comparison.
  Earlier 2021 evidence may supply the immediately preceding provider season.
- Forecast cutoff is UTC midnight before the match. Each historical result must
  have kickoff + three hours strictly before that cutoff. Use only the same
  competition and current/immediately previous provider seasons. Both teams
  require eight qualified current-season matches. This gate is retained even
  where it prevents European group-stage forecasts; do not relax it after scoring.
- Use the sealed 30 September reference weights, with unchanged prediction,
  metric, support and referee-name functions checked against the archived source.
  Own/opponent weights are 0.7/0.3, venue weight 0.5, season/recent 0.85/0.15,
  and recent six-match decay 0.85. The existing recent own/opponent 0.6/0.4
  calculation and bounded divergence adjustment also remain unchanged.
- Referee histories use exact normalized names within a competition. Five
  earlier matches start the confidence ramp, reaching full confidence at 20;
  the modifier coefficient is 0.35. The 0.15 confidence-weighted referee anchor
  requires at least eight matches. Unknown/sparse histories are neutral.
  Report possible surname/initial fragmentation without merging identities.
- Compare team-only and team-plus-referee means on identical fixtures, using
  fixed Poisson distributions. Primary score is paired count NLL; secondary
  metrics are MAE, bias, RMSE, over-4.5 Brier/log loss, reliability and central
  80% interval coverage. Use the original paired ISO-week bootstrap: 1,000 draws,
  seed 20260930. Report league, season-stage and referee-support slices; supported
  slices need at least 200 fixtures / 20 weeks. Smaller slices are descriptive.

Persist input targets/rules/source report, frozen weights, protocol, source and
dependency hashes, chronological cohort membership, dated inputs (including
forecast exclusions), paired predictions, scores, coverage and referee identity
audit in a new sealed experiment. Replay from its saved inputs without accessing
databases, API clients, production profiles or later outcomes. Replays test the
same fixed calculation; they are not additional model-selection trials.

## Interpretation and release boundary

These are `assumed_final` historical reconstructions using later captures.
Original referee-assignment availability is unknown. Lineup, foul/aggression and
other live context are omitted: this is an aligned research reference, not the
full frozen production engine or original published predictions. The two-minute
weighted count is a product convention, not established bookmaker equivalence.

No weights, history windows, count dispersion or calibrators are optimized.
Later qualification/final-system/prospective outcomes remain protected. No API
requests, canonical writes, production profile regeneration, recommendations or
promotion are authorized by this run. Production retains settlement v1 pending
the separate agreed joint release. Results cannot establish ROI or profitability.

## Results

The fixed run completed on the permitted 2023 development cohort. No methods or
thresholds were changed after inspecting scores.

- 9,325 qualified historical targets: 2,027 from calendar 2021, 3,452 from 2022
  and 3,846 from 2023. The 2022 support cohort spans 42 ISO weeks.
- Built 3,846 dated team/opponent/league/referee input snapshots, including
  snapshots for forecasts excluded by the history gate.
- Scored **2,771 paired fixtures across 35 ISO weeks**, covering all ten domestic
  leagues. Every domestic league meets the predeclared 200-fixture/20-week minimum.
- The eight-current-season-match gate excluded 1,075 qualified evaluation
  fixtures: 811 domestic fixtures and all 264 European group-stage fixtures
  (UCL 89, UEL 91, UECL 84). European performance remains unassessed. These are
  forecast-support exclusions, separate from the 1,998 source-target exclusions.

| Measure | Team only | Team plus referee |
| --- | ---: | ---: |
| Count negative log likelihood (primary; lower is better) | 2.229656 | 2.220334 |
| MAE in weighted card units | 1.847479 | 1.832552 |
| RMSE | 2.327786 | 2.308381 |
| Mean predicted minus actual count | +0.230206 | +0.228018 |
| Over 4.5 Brier score | 0.234402 | 0.232322 |
| Over 4.5 log loss | 0.661351 | 0.656995 |
| Central 80% count-interval coverage | 83.04% | 83.65% |

The referee-minus-team-only NLL difference is **-0.009322**, a 0.42% reduction.
Its paired ISO-week bootstrap 95% interval is **[-0.012936, -0.005704]**. This is a
modest improvement for the pooled development comparison, not production
qualification or a comparison against the complete production engine.

### League and referee scope

All ten domestic cohorts have sufficient descriptive support. The intervals below
are individual diagnostic intervals, not a multiple-comparison-adjusted rule for
selecting league-specific configurations after seeing results.

| League | Scored fixtures | ISO weeks | Referee NLL difference | 95% interval |
| --- | ---: | ---: | ---: | --- |
| BelgianProLeague | 239 | 33 | -0.014536 | [-0.025622, -0.005053] |
| Bundesliga | 234 | 26 | -0.005370 | [-0.012597, +0.002379] |
| Championship | 436 | 31 | -0.014645 | [-0.024814, -0.004210] |
| EPL | 316 | 31 | -0.000261 | [-0.011958, +0.012754] |
| Eredivisie | 243 | 28 | -0.007854 | [-0.020425, +0.005869] |
| LaLiga | 277 | 30 | -0.022217 | [-0.040969, -0.004394] |
| Ligue1 | 284 | 30 | +0.002615 | [-0.014457, +0.018688] |
| PrimeiraLiga | 220 | 28 | -0.012006 | [-0.025294, +0.001410] |
| SerieA | 303 | 31 | -0.009501 | [-0.016994, -0.001457] |
| SuperLig | 219 | 30 | -0.008193 | [-0.023553, +0.007426] |

Referee names were present for all qualified 2023 targets. A name is not sufficient
history: 167 scored fixtures had sparse referee input and retained exactly the
team-only prediction. Of 2,604 fixtures reaching the five-match reference gate,
2,557 actually changed prediction; the existing confidence ramp starts at zero
at five matches. Possible name fragmentation remains visible: the inventory has
946 competition/name keys and 17 shared-final-token groups. These groups include
potentially different people (for example A. Madley and R. Madley); no aliases
were merged and no original assignment timestamps were invented.

Both season-stage slices improved on pooled NLL: 928 fixtures with 8–15 current
observations (-0.011420) and 1,843 with at least 16 (-0.008266).

### Remaining limitations and next decision

The referee adjustment does not solve calibration. The reference overpredicts
weighted counts by about 0.23 per match. For example, in the referee model's
70–80% over-4.5 probability bin, the mean probability was 74.27% but the observed
frequency was 58.24% across 170 fixtures. Full reliability bins are saved; sparse
extreme bins should not be treated as precise calibration estimates. Count
interval coverage is descriptive and affected by discrete Poisson intervals.
No fitted dispersion or calibrator was evaluated here.

The repaired target cohort now includes 7,773 fixtures without an eligible-player
red and 274 zero-card fixtures. It avoids the earlier positive-red-only target
problem, but source exclusions can still select the sample; universal
representativeness has not been established. European knockouts remain deferred,
and European group stages cannot pass this particular fixed history rule.

The authorized stopping point is reached after verification: review these fixed
results with the owner before agreeing on bounded weight optimization. European
history support and referee aliases require separately justified methods. No
weights, distributions, thresholds or league switches were selected from these
results. Production activation remains a later joint release of the new engine,
compatible card profiles and settlement policy, with explicit approval.

### Artifacts and verification

Experiment: `Research/phase4-card-fixed-backtest-2026-10-02/`.

- `inputs/`: complete sealed source targets, rules, report and source checksums.
- `fixed-weights.json`, `reference.json`, `manifest.json`, `source/`: frozen
  reference provenance, formula fingerprints, dependency versions and source.
- `dated-inputs.jsonl`: exact contributing fixture IDs and weighted own/opponent,
  league and referee histories for all 3,846 qualified evaluation fixtures.
- `cohort-membership.jsonl`: every source fixture's role, target exclusions,
  forecast exclusions, history counts and scored membership.
- `predictions.jsonl`, `backtest.json`: paired predictions, derived profiles,
  scores, reliability, intervals and slices.
- `coverage.json`, `referee-identities.json`: source/forecast coverage and exact
  referee-name audit. `COMPLETE.json` seals the entire experiment.

Verification is recorded separately in
`Research/phase4-card-fixed-backtest-validation-2026-10-02/`, preserving the
original experiment seal. Offline replay reproduced all six result files
exactly. An independent audit checked all 3,846 snapshots and 1,923,646 historical
fixture links against source targets, including exact chronological membership,
competition/season windows, team sides, referee matching, source eligibility and
snapshot hashes. No own-fixture, future, same-day, unresolved or reserved-period
outcome entered a forecast. Source artifact checksums remain unchanged.
All **86 focused tests passed**, covering source contracts, missing versus zero
counts, protected-period decoding, chronological inputs, the eight-match boundary,
neutral referee handling, immutable outputs, replay and network/database isolation.

The runner is `Scripts/ops/phase4_card_backtest.py`. Reproduce the saved run without
opening databases or original raw archives:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_card_backtest replay \
  --experiment Research/phase4-card-fixed-backtest-2026-10-02
```

The `run` command requires a new output directory and refuses to overwrite one.
An offline guard rejects network access, database connections and writes outside
the output directory. Formula fingerprints reject changes to the fixed reference.
