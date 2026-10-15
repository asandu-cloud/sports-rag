# Existing history import — 25 September 2026

Status: **verified local import complete**. Collection of wholly missing older
seasons has not started. No new candidate model was trained or promoted.

The user completed 19 fixture-list requests with the prepared collection
script. All three resulting preparations passed checksum verification and
reconstruction from frozen exports and provider metadata. The import then ran
offline under the existing refresh writer gate with a fresh SQLite online
backup and one atomic transaction.

## Imported records

| Cohort | Fixtures added | Existing fixtures preserved |
|---|---:|---:|
| Original eight competitions, 2023/24 | 2,513 | 0 |
| Original eight competitions, 2024/25 | 2,658 | 0 |
| UCL, 2026/27 | 80 | 18 |
| UEL, 2026/27 | 74 | 18 |
| UECL, 2026/27 | 230 | 0 |
| **Total** | **5,555** | **36** |

Also added: 11,110 fixture-team rows, 15 season catalogue rows and 74 teams.
All pre-existing records across all 38 database tables were preserved, including
existing names, statistics, predictions, publications, player history, feature
snapshots, watermarks and queues. No player history was imported. Original
exports, baseline archives, saved candidate datasets and active model files
were preserved. No profiles or embeddings were rebuilt.

Of the new fixtures, **421 have provider-verified results but no accepted local
event statistics**. Their goals are known and event statistics remain unknown.
The rest have paired local statistics, but this does not imply every statistic
is available. Legacy zeros without independent evidence remain unknown.

## Exclusions and reconciliation

- Excluded UECL 2024 fixture **1208441**, GAP Connah S Quay FC–Bravo. The provider
  labels it `FT`/90 minutes but gives aggregate goals 0–2, fulltime 0–1 and
  extra-time 0–1. It cannot supply a certified regulation-only target.
- Eleven UECL local statistics pairs remain quarantined because `Guimaraes`
  does not exactly match the provider's `Vitória SC` name. Their verified
  fixture/results metadata was imported; statistics were not guessed.
- The previously excluded Ligue1 2025 fixture **1544684** remains excluded.
- Nine positive-value discrepancies found in the earlier audit were already
  supported by archived provider evidence on the canonical side. Those
  existing values were retained.

The original three preparations are unchanged. A separate reviewed subset
applies an additional regulation-score guard before calling the existing
additive importer. That guard requires fulltime scores to match aggregate
goals, no extra-time/penalty score values, FT status and no elapsed value above
90. **The general recovery/bootstrap implementations were not changed**;
future imports must repeat regulation validation, including on FT-labelled
fixtures. The import artifact contains the guard and excluded source evidence.

Some provider league cohorts contain relegation/playoff rounds. Their
eligibility for modelling still needs an explicit decision in Phase 3.

## Validation

- 37 existing recovery and canonical-dataset tests passed.
- 18 offline tests passed for the new isolated download launcher, including
  live/staging separation, error handling, resume and immutable attempt reports.
- The actual CLI `init-db` path passed a smoke check against a disposable
  isolated database. The real download launcher dry run listed exactly 50
  missing pairs and made no provider calls.
- Four additional guard probes passed before any database writes.
- Pre-existing row fingerprints matched for all 38 tables inside the import
  transaction. Only the four intended tables received additions.
- A second application of all 5,591 accepted plan entries was a verified no-op.
- SQLite quick check passed; zero foreign-key violations and no invalid team
  pairings. The excluded extra-time fixture is absent after commit.
- The canonical candidate loader accepted the expanded history without any
  pairing/identity quarantine. Active model hashes were unchanged.
- No additional API calls, feature exports, training or publications were run.

After import, `Index/platform.db` contains **14,764 fixtures**, of which
**14,355 are FT history rows**, across all 13 supported competitions and
provider seasons 2023–2026. These are coverage counts, not claims of complete
statistics, unbiased labels or qualification for deployment. The prior
5,083-fixture candidate export remains an immutable older snapshot.

Card selection bias and the difference between yellow-plus-red candidate
targets and weighted participation settlement remain unresolved.

## Evidence and backup

Private local directory:
`Index/prediction_experiments/local-history-import-2026-09-25/`

- `platform-before.db`, `before.json`: fresh consistent pre-import backup.
- `plan-verification.json`, `reviewed-plans.json`: frozen plan references and
  accepted entries.
- `regulation-quarantine.json`: excluded fixture and exact provider evidence.
- `result.json`, `verification.json`: committed additions, existing-row
  fingerprints, idempotence and integrity checks.
- `post-import-coverage.json`: availability by competition, season and market.
- `import_verified_history.py`: one-off guarded operation, which refuses to
  overwrite a started run's backup.
- `COMPLETE.json`: initial committed import artifacts and checksums.
- `REVIEW_COMPLETE.json`: final report/coverage/launcher verification checksums.

## Remaining older-season collection

For eight seasons including 2026/27, the wholly absent cohorts are:

- EPL, LaLiga, SerieA, Bundesliga, Ligue1, Championship, SuperLig, Eredivisie,
  PrimeiraLiga, BelgianProLeague, UCL and UEL: provider years **2019–2022**.
- UECL: **2021–2022**.

This is 50 competition-season pairs. The next collection should use the
existing bootstrap CLI in an isolated staging database, retaining raw provider
responses and per-season results for review. Direct live bootstrap would hold
the live writer gate throughout the download and can update existing team
names/fixture fields. It also requires checking per-detail errors independently
of its process exit code. Staging collection must not be treated as a completed
canonical import or as certified statistical completeness.

The separately deferred statistics backfill for already represented seasons
has not been started or folded into this older-season collection.

### User-run command

From the repository root:

```bash
./.venv/bin/python -m Scripts.ops.prediction_history_download --execute
```

Without `--execute`, the launcher only prints a network-free plan. It wraps the
existing bootstrap command with `--no-players --archive`, one competition-season
at a time. Data and raw responses go into
`Index/history_staging/eight-seasons-2026-09-25/`, separate from the live database
and its refresh gate. It does not collect players, prices, embeddings or models.

The collection freezes a manifest, retains every completed subprocess report,
and checks that each zero-error result matches its completed staging sync run.
Rerunning the same command skips verified completed seasons. An interrupted or
failed season is downloaded again in full, using the existing bootstrap
behaviour. Empty statistics are reported as unverified coverage, not complete
labels. Code/manifest differences stop resume for review.

The launcher was tested locally but the large real-provider download has
**not** been executed. After collection, the staged raw evidence still needs
validation and a separately recorded additive canonical import.
