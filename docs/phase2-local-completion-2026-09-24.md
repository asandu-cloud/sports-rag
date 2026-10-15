# Phase 2 — remaining local implementation and acceptance

Date: 24 September 2026.

**Later policy update:** the owner subsequently approved the simplified
[participation-based product policy](participation-settlement-policy.md), now
implemented for new publications only. This supersedes the bookmaker/site
approval prerequisite below for the product-rule cohort, not the historical
unknowns or live-acceptance requirements. Bookmaker equivalence is not claimed.

This batch follows [measurement automation](phase2-measurement-automation-2026-09-24.md).
The remaining locally testable recovery/reporting work is implemented. **Full
Phase 2 sign-off remains open** for bookmaker-policy approval and prospective
live-match acceptance. No prediction formulas, ML training, calibration, betting
thresholds or publication modes were changed.

## 1. Historical recovery: audited, not fabricated

All **467 canonical publications** were audited locally, not just the earlier
35-selection sample. All lack a saved market period; their historical model/
system versions remain unknown. The canonical schedule currently places six
selections for fixture `1570389` on 21 October; they are not due. The other
**461 are overdue but ineligible to grade** without evidence of their original
market period. Saved final scores do not establish that period.

Consequently, this pass recovers **zero historical outcomes**, makes **zero
provider requests**, and invents no historical prices or versions. Missing
identity now appears as an explicit reason in public tracking metrics, even
before pending-note writes can run.

The new `settlement-recovery` command separates planning from application:

- Plan selects only published, ungraded recommendations up to an explicit ID
  boundary. The default is the prospective worker's activation cutoff, 467.
- Result evidence comes only from checksum-verified, saved fixture/statistics
  responses. It is matched by fixture and competition ID. No network fallback.
- Missing identities, card definitions or authoritative results remain pending;
  future/unknown kickoffs are excluded. Already graded rows are not corrected.
- The plan records the exact selected rows and expected evidence. Changed plans,
  changed predictions and changed evidence fail validation before writing.
- Application is explicit, guarded by the shared refresh lock and measurement
  lease, and resumable/idempotent. Successful rows retain the recovery plan ID
  and evidence; concurrent selection changes cannot receive a stale grade.

Private artifacts:

- `Index/prediction_experiments/phase2-historical-recovery-2026-09-24/reviewed-plan.json`
- `Index/prediction_experiments/phase2-historical-recovery-2026-09-24/final-acceptance-audit.json`

The reviewed plan contains 461 pending-note operations and **no outcome writes**.
Its application was attempted but **deferred with zero writes** because the
existing refresh still held the gate. The marker/lock was not removed and the
refresh was not interrupted. This explicit historical plan is not silently
queued into the prospective worker; it can be resumed after the gate clears.

## 2. Reporting consistency and transparency

The website API previously discarded fields already calculated by the tracker.
It now exposes half wins/losses, separate pushes/voids, weighted hit-rate sample
units, flat-stake returns/ROI, valid CLV coverage and pending counts/reasons.
Recent-record requests can include pending selections via `graded_only=false`.
Public responses contain only safe settlement/closing summaries, not private
evidence file paths or raw provider result blobs.

Additional corrections:

- Pushes and voids remain separate counts. Their established refund accounting
  is unchanged. Unknown result labels are not counted as settled bets.
- Pending recommendations are excluded from result/ROI denominators, not
  interpreted as losses. `has_settled_sample=false` distinguishes an unmeasured
  record from a measured zero ROI. Numeric zero fields remain for compatibility.
- Official CLV metrics require a final, exact-selection capture with valid
  pre-kickoff chronology/freshness and a consistent recorded price calculation.
  Bare or unverified historical CLV values do not enter the official aggregate.
- Official daily reports do not silently choose the best-priced amendment when
  auditing all releases. Initial and amended releases retain their own scopes.
- Website summary/daily backend failures return HTTP 503 rather than a fake
  successful empty record. Discord's official summary labels backend failure.
- Discord summaries say **Asian-weighted hit rate** and show half outcomes; they
  no longer imply that `full wins / all settled bets` equals that weighted rate.
  Half-only market/league samples are no longer omitted from recaps.
- Daily reporting retains the existing **publication-date** basis, now explicitly
  labelled. It is not described as the results of matches played on that date.
  Fixture and publication dates remain separately available in recent records.

The official Discord recap/manual-report settlement path now uses the same
bounded prospective worker as the scheduler. It no longer launches the broad
legacy resolver against old publications. API work is offloaded from Discord's
event loop. Unit-bet slip workflows remain a separate cohort and are unchanged.

These are API/Discord reporting changes, not a new website performance-page UI.
Restart the running website/bot processes to load them. The one-shot measurement
job loads code afresh on each scheduled invocation.

## 3. End-to-end acceptance

New isolated tests exercise a full sequence:

1. Publish a multi-selection Match Read to website and Discord; repeat deliveries
   and verify a single tracked stake per recommendation.
2. Capture exact provisional quotes before kickoff, verify the provider reports
   a start, then finalize valid closing evidence without requesting live odds.
3. Restart the worker, settle Asian half outcomes from final fixture statistics,
   retry, and verify outcomes/stakes/evidence are not duplicated.
4. Compare the actual website summary and daily HTTP responses with the record
   returned through the Discord tracker bridge. Non-zero half-outcome/CLV samples
   agree; this is not merely comparing two empty reports.
5. Check recovery tampering, interrupted/resumed application, unknown identities,
   future fixtures, changed selections, private-evidence redaction and backend
   errors. Existing tests retain cancellation/extra-time/correction coverage.

All this uses isolated databases and mocked provider transport. It does not
post test recommendations to real Discord channels or the public website.
The focused regression suite passed **232 tests**. A final fingerprint comparison
confirmed that the historical prediction rows were unchanged. This batch made
zero new API-Football requests and no model/selection changes.

The new read-only `Scripts.ops.phase2_acceptance_audit` reports the remaining
gates without treating an empty live cohort as a pass. Current findings:

- No delivered selection is missing its publication link.
- Initial/amendment/all-release reporting contracts preserve the same metrics.
- There are **zero prospective publications after activation ID 467**, so live
  identity, settlement and closing capture are still awaiting a sample.
- Historical backlog: 467 pending, including 461 currently overdue; reasons,
  ages and next actions are available in the audit report.

## What is genuinely left

1. Specify the applicable bookmaker and country-specific site. Verify its card/
   cancellation rules against available result evidence before approving those
   outcomes. No blanket policy is installed; no policy was inferred from the
   user's approval to finish Phase 2.
2. Complete the existing refresh successfully. The prospective worker retries
   automatically; the explicit historical pending-note plan must be resumed
   separately if those notes are to be persisted.
3. Observe new real recommendations through capture, kickoff, result retrieval,
   retry and both-surface reporting. Review closing-quote freshness/coverage;
   an offline test cannot certify that the provider supplies qualifying prices.

Historical picks without recoverable original identities may remain permanently
ungraded and explicitly excluded. They must not be converted into favourable or
unfavourable ROI evidence merely to make the backlog disappear.
