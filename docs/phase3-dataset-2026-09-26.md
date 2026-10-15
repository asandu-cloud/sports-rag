# Phase 3 Batch A — usable dataset, 26 September 2026

Status: **Batch A complete and verified locally against the preserved provider
archives and reconstructions.** No benchmark model has been trained, scored or
promoted. This is not live-publication or predictive-accuracy validation.

The owner authorized Batch A after agreeing the technical rules, including
pooled league-aware models with equal fixture weights before recency weighting.
This report records dataset preparation, not completion of Phase 3.

## Dataset and eligible scope

The new immutable research dataset is:

`Index/prediction_experiments/phase3-dataset-2026-09-26/`

Dataset identity:
`4362324537087ea2654d27538614166b305647112fa4749e8222134acdf52de0`

A consistent SQLite snapshot was captured at 17:07 UTC on 26 September.
It contains 32,043 fixtures: 31,588 FT history records and 455 NS records.
Only completed FT records enter this retrospective target audit. Provider
status alone does not qualify them: archived regulation scores, identities,
statistic provenance, participation concerns, round scope and actual
contributing pre-match support are checked separately.

| Market | Recorded target available | Eligible across all partitions | Initial training plus development |
|---|---:|---:|---:|
| Goals | 31,588 | 21,035 | 15,649 |
| Corners | 29,491 | 20,431 | 15,463 |
| Shots on target | 28,957 | 20,092 | 15,119 |
| Legacy yellow-plus-red cards | 4,537 | 0 | 0 |

The three market counts overlap: they are not independent matches to add
together. The total eligible column includes reserved periods and is not a
training-set size. Cards remain excluded because their semantics and selective
missingness have not been qualified. No card repair was performed.

| Competition | Eligible goals | Eligible corners | Eligible SoT |
|---|---:|---:|---:|
| EPL | 2,560 | 2,498 | 2,507 |
| LaLiga | 2,584 | 2,519 | 2,507 |
| SerieA | 2,558 | 2,491 | 2,489 |
| Bundesliga | 2,062 | 2,018 | 2,030 |
| Ligue1 | 2,252 | 2,209 | 2,210 |
| Championship | 2,067 | 2,067 | 2,067 |
| SuperLig | 1,329 | 1,329 | 1,329 |
| Eredivisie | 1,081 | 1,080 | 1,081 |
| PrimeiraLiga | 1,145 | 1,145 | 1,145 |
| BelgianProLeague | 1,155 | 1,144 | 826 |
| UCL | 915 | 857 | 861 |
| UEL | 852 | 738 | 732 |
| UECL | 475 | 336 | 308 |

## Exclusions and evidence

The resolver verified 17,766 locally stored source-file hashes with no archive
errors. Target evidence comprises 17,589 raw-provider-archive records, 8,218
verified local reconstructions and 5,781 unresolved records. These remain
distinct source classes. The safe historical-context set has 25,804 fixtures.

- **5,781 lack qualifying saved regulation-period evidence.** These are
  Championship, SuperLig, Eredivisie, PrimeiraLiga and BelgianProLeague
  provider seasons 2023–2025 plus partial 2026. Their canonical results exist;
  the inspected archived evidence cannot certify the market period. They are
  excluded from targets and from historical profiles, Elo and recent form.
- **Three have possible unplayed/administrative-result concerns**: provider
  fixtures 590783, 635812 and 1209260. A 3–0/0–3 result without referee or
  statistics triggers conservative review. This is not an asserted official
  ruling. They are excluded from both targets and context.
- **2,911 otherwise safe fixtures are outside primary round scope**, mainly
  European qualifiers and separate domestic playoffs. Verified context remains
  allowed under the declared same-competition policy.
- Remaining exclusions are market-specific missing targets, missing required
  production rates or fewer than five valid contributing observations for
  either team. Overlapping reasons and a deterministic primary reason are
  preserved; no target is imputed.

Nothing was removed from the canonical database. Original recorded outcomes
remain separate from validated labels in the audit evidence. Explicit zero,
provider null, omitted fields, ambiguous legacy zero and unavailable sources
have distinct provenance states. Optional unavailable fields remain null.

## Frozen chronological partitions

Intervals are UTC, inclusive at the start and exclusive at the end. All
markets, revisions and forecast stages of a fixture share one assignment;
simultaneous kickoffs cannot be split. Conflicting identities/kickoffs are
quarantined. Boundaries use eligible coverage and calendar structure only.

| Partition | Interval | Goals | Corners | SoT |
|---|---|---:|---:|---:|
| Initial training | Before 1 Jan 2022 | 8,909 | 8,823 | 8,495 |
| Forward development | 1 Jan 2022–1 Jan 2024 | 6,740 | 6,640 | 6,624 |
| Phase 3 confirmation | 1 Jan 2024–1 Jan 2025 | 1,949 | 1,776 | 1,790 |
| Calibration reserve | 1 Jan–1 Jul 2025 | 1,125 | 1,065 | 1,071 |
| Final system test | 1 Jul 2025–1 Jul 2026 | 2,105 | 1,926 | 1,911 |
| Later reserved cohort | From 1 Jul 2026 | 207 | 201 | 201 |

The first eligible fixture is 30 August 2019. Seeking at least two elapsed
years of initial history and whole six-month blocks puts development's start
on 1 January 2022. Four six-month development blocks are defined. The first
two provide earlier inner validation blocks; the 2023 blocks have at least
two earlier blocks for nested evaluation. All four pass the raw pooled
training-count, tree-count and validation-count minima for each market using
all earlier eligible data. Recipe-specific lookback/weighted ESS gates remain
to be evaluated; raw counts do not qualify every recipe.

The three confirmation cohorts each cover 39 nonempty weeks and exceed the
1,000-fixture minimum. EPL, LaLiga, SerieA, Bundesliga and Ligue1 meet the
separate 200-fixture/20-date confirmation sample minimum for each market.
UCL, UEL and UECL fall below it. The five newer domestic leagues have zero
eligible confirmation examples. They cannot inherit qualification from pooled
results. All zero-support slices are explicit in `splits.json`.

**The four-year versus all-history question cannot be answered by these
development folds.** At every permitted fitting date their eligible memberships
are identical. Do not use either reserved test to choose between them or
claim evidence for eight-year training. Two-year comparisons can proceed.
The January–June calibration reserve omits early-season stages; later Phase 4
qualification must address that limitation. The later reserved 2026/27 cohort
is still reconstructed historical data, despite the internal partition name
`prospective_reserve`; it is not prospective shadow evidence.

## Features, isolation and artifacts

The existing pure 148-feature builder and its reference two-season policy are
unchanged. The research view selects 126 features, removing 22 card-derived
features and missingness indicators. Competition identity is retained. The
support gate counts only known observations with nonzero contribution to the
actual market profile; an inactive prior cannot inflate its sample support.

Retrospective inputs remain explicitly `assumed_final`: final stored statistics
are assumed available three hours after kickoff. Actual observation timestamps
are preserved separately. This does not recover original provider vintages or
historical lineup/publication stages. Historical referee assignments retain
their availability caveat. Earlier completed matches can update later features
inside an evaluation block under the frozen chronology.

The complete safe history is retained for approved three-/five-season profile
comparisons. Their additional feature policies and bounded comparison recipe
must be frozen before Batch C. This export is the two-season reference; it does
not silently claim those alternative vectors are already implemented.

`development/` contains only initial-training/development features and labels.
Confirmation, calibration, final-system and later targets have separate
`lockbox/` stores. The development reader opens neither those stores nor the
label-rich `audit/` history/snapshot. A fit cutoff cannot pass confirmation's
start, and every training label must be available strictly before its cutoff.
No preprocessing parameters are fitted in Batch A. Weighted preprocessing and
estimator adapters belong to Batch B; existing candidate isolation tests pass.

Lockboxes are an audited workflow boundary, not filesystem secrecy for public
results. Eligibility preparation inspects label validity and counts; no model
scores have been computed on reserved periods. A confirmation-opening command
requiring frozen candidate/specification hashes and a permanent inspection
event remains Batch D work. No such opening has occurred here.

The bundle preserves the database snapshot, source and dependency fingerprints,
feature schema/mask, raw-evidence references/hashes, cleaned history, exclusions,
feature records, split membership/fold hashes, reserve policy and completion
checksums. The old feature export, baseline and active model artifacts are
preserved. The old Phase 1 candidate CLI is not the Phase 3 training runner.

Read-only integrity verification and full frozen feature replay are available:

```bash
./.venv/bin/python Scripts/ops/prediction_benchmark.py verify --experiment Index/prediction_experiments/phase3-dataset-2026-09-26
./.venv/bin/python Scripts/ops/prediction_benchmark.py replay --experiment Index/prediction_experiments/phase3-dataset-2026-09-26
```

These commands do not fit models, request provider data or inspect model test
scores. `prepare` refuses to overwrite an existing experiment; later changes
require a new directory and dataset identity.

## Validation and next work

186 focused/regression tests passed, covering source/period checks, zero/null
handling, future/simultaneous-result isolation, actual contributing support,
fixture grouping, strict availability, train-reader isolation, deterministic
hashes, corruption rejection, frozen-code replay and protected data/models.
All **31,588** feature records replayed exactly from the frozen implementation
and inputs. Repeated and reversed-input split generation reproduced identical
memberships and hashes. An independent metadata-only review checked every
fixture membership, all 675 coverage slices and all 12 market/fold counts.
All 35 dataset artifact checksums passed. Final row fingerprints confirmed
unchanged canonical FT fixtures and associated team statistics; active model
file hashes also remained unchanged. Evidence is recorded in:

`Index/prediction_experiments/phase3-dataset-2026-09-26-validation/`

The next implementation batch is the isolated baseline/candidate runner.
Eligible pooled benchmarks can proceed without another history download.
Broader league qualification requires recovering the missing period evidence
and resolving sparse evaluation scope in a new versioned dataset before
scoring; no provider requests or threshold changes are implied.

Separately, the 17:08 UTC operational audit found the refresh-incomplete gate
active again after an embedding DNS/connection failure. Both workers were
deferring, and remote notification was not delivered. The marker was not
removed and no refresh/recovery was launched. Phase 2 still had 467 ungraded
publications, 461 overdue and zero prospective publications after activation;
the historical pending-note plan remained unapplied and publisher policy reload
unverified. These live acceptance issues do not invalidate the offline snapshot,
but they remain open production work.
