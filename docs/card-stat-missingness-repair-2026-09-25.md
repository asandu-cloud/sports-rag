# Card statistic missingness repair — 25 September 2026

The legacy **team-stat collection bug is fixed** and five historical card
fields were repaired from existing provider evidence. General card-model
training remains unqualified. No provider requests or bulk backfill ran.

## Collection and recovery

All eight registered legacy team collectors now use
`Scripts/team_stat_export.py`. Provider nulls remain null, explicit zero stays
zero, and omitted fields are not invented. Invalid numbers remain unknown;
duplicate statistic names fail instead of silently overwriting a value.

Each new fixture/team export retains `_provider_statistics`: a versioned
envelope containing provider fixture/team IDs, observation time and original
statistic values. Evidence survives JSON export and is excluded from numeric
season averages. Fixture and aggregate exports preserve nulls. Changes take
effect on the next normal collection; existing Output files were not regenerated.

Historical recovery accepts zeros from new exports only when source evidence
and identities match. **Old, lossy exported zeros stay unknown.** A marker
without matching raw evidence cannot certify a zero. The original frozen
2,663-fixture recovery preparation still verifies unchanged.

The canonical provider importer already preserved zeros and nulls. Candidate
label construction still requires known components and never imputes outcomes.
Legacy player/derived heuristic features have separate missing-value defaults;
this change does not certify those paths or change their prediction formulas.

## Historical repair applied

A fresh read-only audit of the canonical raw archive found no supported missing
card-count repairs, no conflicting numeric values and no invalid archives.
The separately saved 22 September diagnostic supplied these explicit zeros:

| Provider fixture | Provider team | Field | Before | After |
| --- | --- | --- | --- | --- |
| 1387850 | 116 | Red cards | unknown | 0 |
| 1421213 | 1165 | Red cards | unknown | 0 |
| 1421213 | 1165 | Yellow cards | unknown | 0 |
| 1421213 | 4160 | Red cards | unknown | 0 |
| 1451274 | 160 | Yellow cards | unknown | 0 |

`Scripts/ops/prediction_card_repair.py` verifies the diagnostic preparation,
request/response identity, body checksum, team pair and current fixture result
before considering a missing count. Nulls and absent events never imply zero.

Application took the refresh writer gate, checked the incomplete marker was
absent, backed up the database and changed the fields in one transaction.
Per-field provenance retains source hashes, actual provider observation time,
application time, previous digest and plan identity. Input availability is not
backdated. Reapplication is idempotent; stale/conflicting fields abort the
transaction. Existing numeric counts are never overwritten.

Before/after comparison verified five count changes across four team rows,
plus evidence/digest/update-time fields. All other team statistics and row
identities were unchanged.

Provider season 2025 computable **yellow-plus-red** targets increased from
305 to **307**; fully known zero-red fixtures increased from zero to **one**.
Coverage is still heavily selected. The 2026 cohort remains 241 card labels,
including 160 zero-red fixtures. The saved 5,083-row feature export is unchanged
and does not contain the repairs; research using them needs a new immutable export.

Private artifacts under `Index/prediction_experiments/`:

- `card-null-audit-2026-09-25/card-audit.json`
- `card-stat-repair-plan-2026-09-25/plan.json`
- `card-stat-repair-applied-2026-09-25/platform-before.db`
- `card-stat-repair-applied-2026-09-25/before.json`
- `card-stat-repair-applied-2026-09-25/result.json`
- `card-stat-repair-applied-2026-09-25/verification.json`

The backup can contain private application data; do not publish it.

## Validation and remaining gates

144 tests passed: all eight collector round trips, historical recovery, repair
evidence/identity checks, rollback, idempotence, refresh safety, canonical
upserts, feature construction, export/replay, candidate isolation and diagnostics.
Collector requests were mocked; no new real provider response was obtained.

Remaining: obtain sufficient authoritative missing-count evidence and assess
selection bias, then build and validate targets matching
`spix-participation-settlement.v1`. Simple team yellow-plus-red totals remain
different from weighted participation targets. Changing to `yellow + 2*red`
alone does not establish second-yellow, participation or regulation semantics.

Active models, baseline archives, historical feature exports, settlement policies,
published recommendations and public delivery modes were preserved. The
deferred bulk statistics backfill was not started.
