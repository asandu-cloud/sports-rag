# Eight-season history import — 26 September 2026

Status: **collection validated and additive canonical import complete**.
The user ran the API-Football collection. Subsequent validation and import
made no provider requests. Phase 3 benchmarks have not started, and no model
was trained or promoted.

## Result

All 50 planned competition-season collections completed. Their completion
reports, source fingerprints, raw-response checksums and staged records were
verified before importing into `Index/platform.db`.

| Record | Added |
|---|---:|
| Accepted completed fixtures | 17,233 |
| Fixture-team rows | 34,466 |
| Season catalogue rows | 50 |
| Teams | 131 |
| Raw-response archive references | 17,312 |

The canonical database now contains **31,997 fixtures, including 31,588 FT
history rows**. All twelve competitions other than UECL have represented
provider seasons 2019–2026 (2019/20 through the ongoing 2026/27 season).
UECL has 2021–2026. This is eight represented seasons, including the current
incomplete season, and six for UECL; it is not eight complete seasons of
universally complete statistics.

| Numerical target | Known in canonical FT history |
|---|---:|
| Goals | 31,588 |
| Corners | 29,491 |
| Shots on target | 28,957 |
| Legacy yellow-plus-red cards | 4,537 |

These are availability counts, **not eligible training counts**. Per-season
and per-competition counts are in `post-import-coverage.json`.

## Exclusions and missingness

Of 17,262 downloaded fixtures, 29 were withheld from canonical fixture
history. Their original payloads and exclusion reasons remain in the frozen
preparation and `quarantine.json`:

- UECL 2021 fixture **728649** is labelled FT/90 but aggregate goals are
  0–3 and fulltime goals are 0–2. Its regulation target cannot be certified.
- **28 SuperLig 2022 fixtures** involving Hatayspor or Gaziantep FK have
  3–0/0–3 results, no statistics and no referee recorded in March–June 2023.
  These are suspected administrative results requiring participation review;
  the audit does not claim to have independently confirmed their official
  status. Exact fixture IDs and evidence are retained in the quarantine.

Among the accepted additions, **840 have results but no provider team
statistics**. Their statistical columns remain NULL; two team rows preserve
fixture identity and provenance without inventing observations. The remaining
16,393 matches have paired statistics responses, which can still contain
individual missing values. Explicit provider zeros are preserved as zeros.

The expanded collection does **not** resolve card-label selection bias.
Missing red-card values remain unknown, and the candidate card target is
still yellow plus red rather than the approved weighted participation
settlement definition. Unrestricted cards remain excluded from the initial
Phase 3 benchmark scope.

Some provider competition cohorts include qualifiers, playoffs or relegation
rounds. Their eligibility needs an explicit Phase 3 rule. The import checks
obvious regulation-period inconsistencies but does not certify every retained
fixture as played or every league-season as complete. Historical provider
responses were collected in 2026, not at the original prediction cutoffs.

## Preservation and verification

The import used the normal refresh writer gate, a fresh consistent SQLite
backup and one atomic transaction. All pre-existing records across all 38
tables were preserved, verified by row fingerprints. Additions were limited
to fixtures, fixture-team statistics, seasons, teams and raw archives.

- All 17,312 archived responses passed checksum verification, including a
  second verification of the canonical archive copies after commit.
- Staged team statistics agreed with normalization of their original raw
  payloads. Team identity, numeric validity and period checks passed for the
  accepted subset.
- 51 focused validator tests passed, including period conflicts, identity
  mismatches, malformed statistics and zero/null/absent distinctions.
- A second application inside the transaction preserved all 17,233 accepted
  fixtures without adding or changing records.
- SQLite quick check passed, foreign-key errors were zero, and the canonical
  candidate loader reported no team-pairing quarantine. All 29 excluded
  fixture IDs remain absent from canonical history.
- Every new fixture-team row contains import provenance, including original
  observation time, payload hashes and canonical archive references.
- Active model hashes were unchanged. Profiles, embeddings, player history,
  publications, queues and existing watermarks were preserved. The refresh
  incomplete marker was absent at the final check.

No player data or team/player profiles were generated. Staging files,
baseline archives and the existing 5,083-fixture feature export remain
unchanged. The production prediction window and publishing settings were
not changed. These checks do not close live publication, settlement or
busy-matchday acceptance gates.

## Evidence and next work

Private local evidence directory:
`Index/prediction_experiments/eight-season-history-import-2026-09-26/`

- `platform-before.db`, `before.json`: consistent pre-import backup and
  active-model fingerprints.
- `staging-snapshot.db`, `collection-evidence.*`, `source-archives.json`:
  frozen collected evidence and raw-archive references.
- `plan.json`, `quarantine.json`, `audit.json`, `PREPARED.json`: accepted
  inputs, explicit exclusions, coverage and preparation checksums.
- `import_staged_history.py`, `code-snapshot.*`: the one-off guarded import
  and frozen implementation. Do not rerun the completed application.
- `result.json`, `verification.json`, `COMPLETE.json`: committed counts,
  preservation/idempotence checks and original completion checksums.
- `post-import-coverage.json`, `REVIEW_COMPLETE.json`: final availability
  audit and checksums including this review.

Raw archives were copied into `Index/raw_archive/staged-history/` with
canonical references; their original staging locations are also preserved.

The next bounded task is Phase 3 dataset eligibility and chronological
splits for goals, corners and shots on target, using a **new immutable feature
export**. It must define market periods, competition-round scope, historical
availability assumptions, exclusions and sufficient sample requirements;
keep all versions of each fixture together; fit preprocessing only on
training data; and reserve a final test period. Split dates should follow
actual eligible coverage. No additional bulk download is needed to begin
that work, and no benchmark accuracy or profitability claim follows from
this import.
