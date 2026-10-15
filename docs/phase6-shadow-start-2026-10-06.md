# Phase 6 started — 6 October 2026

The owner chose **26 weeks** for the fixed prospective comparison and authorized starting Phase 6. The first outcome-blind diagnostic shadow collector is now **implemented, tested, installed and loaded**. See the [complete implementation and evidence](../Research/phase6-shadow-2026-10-06/README.md).

- An independent hourly job runs from 6 October, 12:37 to 20 October, 12:37 Europe/Bucharest while the Mac/user session is available. It stops capturing at the fixed end; it does not silently extend missed hours.
- Both policies consume the same saved canonical forecasts and archived UK bet365/William Hill quote observations. The original receipts, forecasts, decisions, exclusions and shortlists are retained in `Index/phase6_shadow/`.
- **625 Python tests plus four subtests passed**, along with inherited website/Discord identity and JavaScript checks. Real subprocess isolation, pre-match capture and scheduled idempotent restart were verified.
- The first capture processed all three fixtures within 96 hours that had saved forecasts. Another 93 fixtures further out in the 14-day schedule had no forecast yet and remain explicit missing-input records. The candidate made zero selections because its evidence and freshness gates failed.
- **The 26-week performance clock has not started.** Its dates require the final supported scope, probability/price/settlement evidence, initial/amendment recording contract and system freeze. Hourly feasibility records are not independent betting observations or actual first publications.

This is a selection-layer shadow runner from immutable canonical forecasts. It does not yet preserve/replay all upstream profile calculations. It opens no historical or prospective outcome data, makes no new provider requests, and does not publish or train/promote models. The existing prediction engine and public delivery remain unchanged. Full Phase 6 promotion, reload and rollback work remains separate.
