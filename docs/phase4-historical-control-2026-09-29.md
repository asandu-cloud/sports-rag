# Phase 4, Step 1 — historical inputs and frozen-control reproduction

Implemented 29 September 2026 after owner authorization. This completes **Step 1**
of the [five-step plan](phase4-full-probability-comparison-plan-2026-09-28.md).
No candidate parameters, calibrators, blends or selection policies were fitted.

## Result and artifacts

The frozen `phase4-repaired-statistical-control.v1` now has dated historical input
snapshots and reproduced outputs. Two separate immutable experiments preserve
earlier fitting references and the 2022/2023 development comparison:

| Scope | Artifact under `Research/` | Fixtures replayed | Eligible goals | Eligible corners | Eligible SoT |
|---|---|---:|---:|---:|---:|
| 2019–2021 earlier fitting references | `phase4-historical-control-fitting-2026-09-29/` | 10,849 | 8,909 | 8,823 | 8,495 |
| 2022–2023 development | `phase4-historical-control-2026-09-29/` | 8,799 | 6,740 | 6,640 | 6,624 |

All **19,648 serialized-input replays matched**, producing 137,536 canonical
market-result records across seven market groups. A result record can explicitly
be unavailable; these counts are not counts of eligible recommendations.
Cards remain unqualified in both cohorts.

The earlier reference forecasts supply genuinely earlier inputs for the first
2022 forward fold. Creating them does not fit anything, change the frozen
parameter-selection protocol, or label 2023 as untouched. Downstream fitting
must still enforce label availability at each fold cutoff and the protocol's
rules for each candidate family.

Both artifacts contain:

- `history.jsonl`: the 18,737 certified pre-2024 regulation-history rows, shared by
  reference rather than copied into every snapshot. Original observation times
  and null values are retained.
- `requests.jsonl`, `targets.jsonl`, `snapshot-schema.json`: fixed forecast
  membership, separate targets, availability and target definitions.
- `reproduction/snapshots.jsonl`: effective/current/prior profiles, profile-quality
  audits, recent-six aggregates, recent-eight variances, contributing fixture IDs,
  per-field counts, venue membership, and explicit context availability.
- `reproduction/control.jsonl`: unchanged canonical results, team goal means,
  joint goal distribution, count PMFs with variance provenance, coherent derived
  probabilities and five-way Asian diagnostics.
- `reproduction/eligibility.jsonl`, `reproduction/report.json`,
  `integrity-audit.json`, `context-coverage.json`: every exclusion, league/year
  coverage, variance fallbacks, missingness and verified snapshot membership.
- `implementation/`, `validation/`, `manifest.json`, `PREPARED.json`,
  `access-ledger.json`, `COMPLETE.json`: exact implementation copies, tests,
  dependencies, source/data/protocol hashes, access boundary and completion seal.

The development artifact additionally preserves 28 fresh-process replays spanning
all 13 competitions and both years. Reversing the history input order produced
byte-identical snapshot, forecast and eligibility files for those fixtures.

## What the reconstruction actually executes

The worker imports the archived engine from
`Research/phase4-baseline-2026-09-28/control/workspace/` in a fresh process.
It replaces external lookups with exact-ID dated fixture rows, then executes the
unchanged profile, projection, variance, probability and canonical guardrail code.
The separate numerical replay uses serialized aggregates instead of those raw
lookups; every fixture must match before completion can be reported.

Preserved control behavior includes the eight-match prior switch, newest available
earlier prior season, current-season recent-six window with positional .85 decay,
current-season recent-eight variance, European blending, existing variance floors,
fixed goal dependence and zero ML contribution. No Phase 3 reconstructed mean is
substituted for this control. Historical eight-season storage does not make the
control average all eight seasons.

The input adapter preserves missing versus zero and pairs opponent rows by exact
fixture/team IDs. It excludes the entire forecast day and additionally requires
the assumed completion time, kickoff plus three hours, strictly before the cutoff.
Domestic membership is reconstructed from the latest earlier completed domestic
fixture for the exact team ID. This is disclosed historical identity resolution;
it does not replay historical Chroma aliases, retrieval caps or stored-profile
fallbacks without dated evidence.

The control's calendar-based season inference is preserved. Six June UCL
qualifiers in the development cohort have a provider season different from that
calendar inference. They are recorded rather than silently repaired; qualifying
rounds are already outside the primary research scope.

Count PMFs retain the real control mean and variance, with residual mass bounded
below 1e-10 and parametric identity preserved. Goal-derived totals, BTTS, winner
and handicap probabilities share the frozen score distribution. Fixed half,
whole and quarter lines are explicitly diagnostic; they are not historical offers.
No odds, EV, ROI, CLV or delivered recommendation history was fabricated.

## Eligibility and remaining limitations

The frozen Phase 3 market exclusions are retained conservatively, then an
available control distribution is required. A changed reconstruction does not
silently broaden an approved cohort. All fixtures stay in the accounting.
Production recommendation-quality assessments are recorded separately from
research eligibility; neither implies that an executable price exists.

For 2022/2023, distributions can be calculated for 8,783 goals, 8,708 corners and
8,540 SoT fixtures, but only the smaller eligible counts above may enter the
primary comparison. In particular, **909 fixtures lack certified period evidence**
and remain excluded. Exclusion counts overlap because one fixture can fail more
than one requirement. League-level coverage does not establish that every league
meets the later minimum-support or regression tests.

Historical lineup/player adjustments, standings/regime context, referee profiles,
knockout/first-leg context and authentic quote slates are unavailable in this
reconstruction. They are explicitly neutral or unavailable for every system.
Engineered aggression/control/form fields are also unavailable, rather than
invented from unverified inputs. The context matrix is a common input boundary
for later candidates, not a claim that these effects never matter.

Stored final statistics use **`assumed_final` historical availability**. Original
observations were collected later, so this is not an immutable record of what was
known at the original kickoff and is not an exact replay of published forecasts.
Missing xG remains missing and follows the frozen fallback. Weighted card targets
and representative earlier card evidence remain a separate qualification task.

## Validation and preservation

**445 focused/regression tests passed**, including 25 new historical-input tests.
They cover future/same-day result isolation, completion availability, exact IDs,
null versus zero, deterministic membership, prior-season gaps, dated domestic
membership, fitting/development separation, frozen-function parity, independent
arithmetic examples, coherent goal/Asian probabilities and forbidden IO.

The final streaming audit verifies every snapshot hash, contributing fixture's
identity/season/cutoff, recent-window membership, target/forecast alignment and
the complete seven-market result set. No research recommendation or price is
present. The worker prohibits networking, SQLite, dotenv reads, live data reads,
subprocesses and writes outside its output; successful runs recorded zero IO
violations. Bytecode-cache probing initially triggered this guard; that startup
attempt is preserved separately as `phase4-historical-control-2026-09-29-startup-failed/`
and was never marked complete. The loader now compiles the explicitly saved
adapter source without probing a bytecode cache.

The mixed source archive was checksum-verified and parsed through the existing
lexical boundary: **7,067 later history objects were skipped before JSON decoding**.
No 2024+ outcomes were decoded or used. The original frozen control and protocol
remain unchanged. All 456 existing runtime/model files compared with the frozen
source matched. No canonical database, active model, production profile, worker
mode, refresh marker or public delivery setting was modified by this work.

## Verify and continue

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_history verify Research/phase4-historical-control-2026-09-29
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_history verify Research/phase4-historical-control-fitting-2026-09-29
```

New runs use `prepare`, `run`, then `seal` in a **new** directory. Use
`prepare --scope earlier_fitting` for the earlier cohort; the default scope is
2022/2023 development. Existing output directories are refused. Partial/smoke
runs cannot pass the completion seal. Captured worker/adapter source is used for
execution; prepared inputs are checksum-checked before and after execution.

**Next is Step 2:** compare the separately declared gradual prior, corner/SoT
dispersion and coherent goal-distribution changes against these same control
inputs. Keep common omissions and fixed cohort accounting, fit only within the
allowed chronological boundaries, and retain the unchanged control as a valid
winner. This Step 1 result demonstrates reproducibility, not improved predictive
accuracy or profitability. No Step 2 comparison or production promotion has run.
