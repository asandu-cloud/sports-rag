# Refresh blockage recovery — 25 September 2026

## Cause and scope

The daily refresh started on 23 September at 07:21 local time. Its canonical
incremental sync completed, but its legacy bridge stalled in
`SeriaA_player_stats.py`. A native process sample showed an HTTPS socket read;
the legacy collector supplied no request timeout. Its parent processes kept
the exclusive refresh lock, so Match Read generation and prospective
prediction measurement correctly deferred instead of reading mixed inputs.
The exact request/remote cause of the stalled connection was not established.

The observed chain was `11349 → 11436 → 11767` (full refresh, legacy bridge,
Serie A player download), with `caffeinate` PID 11352. Process identities and
lock ownership were rechecked before sending SIGTERM only to PID 11767.
The parents exited nonzero, all four processes disappeared, and their lock
descriptors closed. No database, historical output, or gate file was deleted.

## Implemented safeguards

- Shared API-Football transport: 5-second connect/30-second socket-read
  timeouts; at most three attempts for transport failures, HTTP 429 and 5xx.
  Invalid JSON/envelopes, provider errors returned with HTTP 200, mismatched
  result counts and unconsumed pagination fail explicitly.
- All 16 registered legacy team/player collectors use the transport and
  propagate failed fixture downloads before saving season-level files.
  Fixture/team coverage lost relative to existing outputs also fails before
  replacement. Canonical detail calls share structural payload checks.
- Referee rebuilds fail on request errors; interactive referee lookups may
  still degrade gracefully. The KB refresh CLI now exits nonzero for recorded
  document-processing errors or remaining unsynced documents.
- Legacy child stages default to 30-minute wall-clock deadlines. Outer
  refresh stages default to two hours. A timed-out/interrupted outer stage
  cleans up its process group, including descendants, before returning.
  Nested stages share that group; unrelated worker processes are not killed.
- Failed full refreshes retain the incomplete marker. Retrying does not reset
  the incident age. Only normal completion of the full refresh clears it.
- Existing scheduled gate checks persist prolonged-blockage alerts, expose a
  health warning and optionally notify a private Discord webhook. No new
  scheduler or public channel is used. Alerts are deduplicated, failed sends
  are bounded, and recovery has its own notice.

Configuration and operational commands are in
[terminal commands](terminal-commands.md#refresh-timeout-and-blockage-protection-25-september-2026).
The owner chose a private Discord webhook, to be configured separately using
`REFRESH_ALERT_WEBHOOK_URL`. Until configured, local logs/incident state/health
are active but remote notifications are not.

## Validation refinements found during recovery

Two initial recovery attempts failed closed while exercising newly added
checks. Existing files were retained and the gate stayed closed.

1. Freezing individual player identities was too strict. API-Football had
   changed Mathis Lainé's minutes from 8 to 1 (excluded by the existing minutes
   filter), changed N'Guessan's player ID, and replaced Yaya Diémé's placeholder
   ID of zero. Read-only fixture inspections confirmed these corrections.
   Coverage protection therefore applies to fixtures/teams, not immutable
   player identities. The minutes filter and prediction formulas are unchanged.
2. A minimum eleven-player rule incorrectly turned long-standing sparse
   qualifier coverage into a refresh outage. Fixture 1554385 listed ten
   Floriana players, eight meeting the existing minutes filter—exactly the
   eight already in the saved baseline. Structurally valid sparse rosters now
   produce explicit coverage warnings. Network failures, missing team blocks
   and malformed statistics still fail; sparse data is not fabricated or
   certified complete. Broader model data-quality policy remains separate.

## Verification

- 321 focused/regression tests passed together, covering transport bounds,
  invalid response handling, all 16 collectors failing before partial saves,
  old-file retention, provider corrections, sparse-coverage warnings, real
  hung-child/grandchild cleanup, inherited locks, gate recovery, alert
  deduplication/retries, and existing publication/measurement/settlement paths.
- The old stalled chain was stopped and its locks released, with the safety
  marker retained for recovery.
- Scheduled measurement ticks emitted the new persistent blockage alert
  while safely deferring. Both worker LaunchAgents remained loaded.
- The complete additive refresh succeeded on 25 September, 15:32:37–15:49:52
  Europe/Bucharest (about 17 minutes). All seven stages completed; the health
  stage returned its documented, accepted warning code. European competitions
  were checked first on this recovery rerun; scope was still all 13 leagues.
- Legacy embedding refreshed 2,446 documents and skipped 12,021 unchanged
  fixture documents. Canonical KB refresh drained 13,844 queue items, scanned
  31,489 documents, and embedded 126 changes: zero errors, zero pending embeds.
  Queue status after completion: 71,670 completed, no other statuses.
- Feature build runs 847/848 completed for both data paths. Historical fixture
  counts remained 1,879 (2023), 1,838 (2024), 4,460 (2025), and 1,032 (2026),
  total 9,209. No historical season was removed or recreated.
- At 15:49:52 the full-refresh context cleared its own incomplete marker and
  logged `Refresh gate recovered`. A fresh process then successfully acquired
  the shared gate. Local alert state reports no active/prolonged blockage.
- Measurement run 851 completed at 15:50:02 with zero errors/requests; this
  was a scheduled post-refresh run, separate from the refresh completion hook.
  Its health is `ok`.
- Requested an immediate invocation of the installed Match Read job using
  `launchctl kickstart` without `-k` or changing its schedule. Run 854 completed
  at 15:53:01 in `website` mode, zero errors and no refresh deferral. A further
  run (860) completed at 15:57:59. The final check showed no active Match Read
  lease and measurement health `ok`.
- There were zero eligible fixtures in the 96-hour window and zero prospective
  measurement candidates. These prove that both workers can run again, not
  that a live match's card/lineup amendment or settlement was exercised. The
  earliest upcoming fixture in the database is 9 October.
- Post-recovery quick health still warns on 28 old watermark timestamps
  (15 historical season scopes and 13 separately scheduled fixture-calendar
  scopes). The blockage warning and measurement warning have cleared. This
  recovery did not erase those unrelated warnings or relabel old data fresh.

This is an operational reliability change, not Phase 3 model training,
calibration, or a claim that sparse historical statistics are fully validated.
