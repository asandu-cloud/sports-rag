# Two-minute card candidate and historical response inventory

30 September 2026. Implemented the owner's participation and scope decisions
from the [v2 research amendment](phase4-card-policy-v2-protocol-2026-09-30.md).
No API calls, database mutations, weight search or production policy activation.

## Implemented definition and release boundary

Research now explicitly uses **recorded minutes >= 2**, with yellow = 1,
straight red = 2, and two yellows plus dismissal = 3 total. The shared participation
parser accepts the minimum explicitly; player evidence retains the actual recorded
minutes and identifies below-threshold exclusions. Eligible-player null counts
and unknown minutes for a carded player still remain pending.

The candidate is `spix-participation-settlement.v2`, paired with
`phase4-participation-card-target.v2`. Its historical target, reconciliation,
dated-history validation and fixed comparison use the same version. The old
research defaults remain v1 so previous artifacts can still replay unchanged.

**Production still uses v1 and minutes > 0.** The shared parser's existing call
signature remains valid and preserves its previous output. No publisher or
settlement worker is switched to v2. Joint promotion must include the new engine,
aligned team/opponent/referee profiles, candidate target version and publication
policy. Settlement dispatch must recognize both supported saved policy versions;
it must not merely replace the global active policy and reinterpret old records.
New publications can adopt v2 at that release; old publications and grades retain
their saved definition. Yellow-only products remain separate. Neither policy
automatically establishes equivalence to every bookmaker's card market.

## Scope and regulation evidence

This stage includes ordinary domestic league rounds, numbered integral Belgian
league split groups and European group/league phases. It excludes separate
promotion/relegation playoffs, European qualifiers, knockouts and unknown rounds.
Deferred matches cannot enter either targets or the candidate's dated histories.
The implementation reuses the existing provider-season/round classifier.

An exact saved fixture identity, known league format and completed FT status now
establish regulation-only scope. A second raw regulation certificate is not
required simply because that fixture was missing from the older frozen history.
Conflicting fixture/team/season/date/round metadata and non-FT statuses still fail.
The metadata reference is labelled as a read-only canonical snapshot, not an
original provider response. Its digest is over canonical JSON; the artifact's
COMPLETE manifest independently hashes the serialized file bytes.

The new artifact is `Research/phase4-card-policy-v2-2026-09-30/reconstruction/`.
Previously admitted pre-2024 outcomes came from the sealed September 30 local
reconstruction. Fresh all-season reads were fixture identity/round and player/
archive presence metadata only. Later player outcomes and response bodies were
not decoded. Ambiguous legacy group-stage/knockout rematches remain excluded;
removing knockouts from the target scope does not identify undated legacy rows.

- 17,529 permitted completed fixtures are within the narrower phase scope.
- 2,037 have some admitted player evidence; **2,013 have provisional arithmetic**.
- The two-minute threshold changes **20** comparable provisional totals.
- No scoped fixture retains the previous regulation-certification exclusion.
- **Zero targets qualify for real fitting/evaluation under the remaining player
  source-completeness requirement.** Earlier player history is still absent.
- No real card forecasts were scored. The new run ends at its evidence/support
  gate; the empty predictions and dated-profile files are intentional.

The decline from the previous 2,101 provisional totals to 2,013 is due to the
narrower competition-stage scope, not a loss of arithmetic availability caused
by the threshold. Missing archival evidence is not proof every provisional total
is wrong. This amendment fixes participation alignment and period scope; it does
not automatically certify the original completeness of surviving legacy exports.

## Which historical player responses are missing

The missing response is the fixture-level API-Football `/fixtures/players`
response: exact teams and player IDs, recorded minutes, yellow/red counts and
the original response structure. Normalized rows and exported profiles are
different objects; they do not substitute for a retained source response.

Metadata inventory for **completed fixtures within the newly agreed scope**:

| Provider season | Completed fixtures | Fixtures with normalized player rows | Registered player responses present | Responses not registered |
|---|---:|---:|---:|---:|
| 2019 | 3,593 | 0 | 0 | 3,593 |
| 2020 | 3,980 | 0 | 0 | 3,980 |
| 2021 | 3,988 | 0 | 0 | 3,988 |
| 2022 | 3,919 | 0 | 0 | 3,919 |
| 2023 | 3,896 | 1,852 | 0 | 3,896 |
| 2024 | 3,966 | 1,818 | 0 | 3,966 |
| 2025 | 3,929 | 1,782 | 0 | 3,929 |
| 2026 | 623 | 623 | 352 | 271 |

These are current local fixture coverage counts, not certification of complete
provider seasons. Provider 2026 is incomplete, and UECL has no seasons before
its inception. Archive presence does not certify response contents. Additional
legacy exports exist outside the database; this table deliberately distinguishes
normalized player rows from registered source responses.

The 2019–2022 gap totals **15,480 in-scope fixtures**, with neither normalized
player rows nor registered player responses. The earlier download explicitly
used `--no-players --archive`; it expanded team-statistics history.

`player-response-inventory-NOT-A-DOWNLOAD-PLAN.jsonl` records each in-scope fixture,
competition/season/round, normalized presence, admitted legacy presence, archive
IDs and the exact endpoint/fixture parameter. `player-response-gaps.json` gives
league/season summaries. No collection is authorized or queued by these files.
The existing bounded search of raw/staging locations found no extra unregistered
player responses; external disks and every duplicated research backup were not
exhaustively searched.

For completing the agreed backtest, collection needs an earlier fitting/history
cohort and later evaluation evidence. We need not wait for every older season to
implement and test the calculation. A separately approved diagnostic should first
verify historical endpoint availability and response quality once API access
returns. Preserve full responses before applying profile relevance filters.

## Missing red cards are not automatically zero

The earlier provider diagnostic found 58 sampled team-side null red fields with
no listed dismissal event. That is suggestive, but the same investigation found
event/statistics disagreements and could not establish universal feed completeness.
A null may be unresolved even when the actual count was zero. The implementation
therefore preserves unknown versus explicit zero and reconstructs a zero only
from sufficient evidence. No null team-card field was changed in the database.

## Validation

**260 tests passed** across the new policy, reconstruction, existing qualification,
participation settlement, safe settlement, measurement-cycle and tracking suites.
Outbound connections and dotenv loading were disabled for that run. Tests cover
one/two-minute boundaries, weighted dismissal arithmetic, nulls, inconsistent
cards, unchanged production policy, scoped rounds, metadata conflicts, reserved
outcome isolation and chronological fixed comparisons.

The default parser matched the preserved pre-change parser on 36 combinations
of minutes and card evidence. A separate fresh-process replay reproduced all
19,648 records and the blocked result of the original v1 reconstruction exactly.
An independent review recomputed all 2,013 provisional v2 totals and verified
that the 20 differences equal the excluded one-minute players' contributions.
The scope, inventory and backtest gate were also replayed from saved inputs.
These checks validate local implementation and reconstruction, not public card
performance or completion of the real-data backtest.
