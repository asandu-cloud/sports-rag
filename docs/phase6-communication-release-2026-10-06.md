# Communication and decision-audit release — 6 October 2026

**Released on the owner's Mac and verified through the running website at <http://127.0.0.1:8000>.** The owner authorized releasing the independently useful communication, recommendation-identity and logging changes, then explicitly selected this Mac for local website startup.

The website serves release `communication-audit.2026-10-06.v1`, with loaded source identity `43689c2fab61ba36badb418eed936c467ac31c16c87e6063b5d0c8e29ef45b0e`. The release archive is [Research/phase6-communication-release-2026-10-06](../Research/phase6-communication-release-2026-10-06/README.md).

## What users now receive

- Exact selected price and bookmaker, model probability and estimated return when available. A price such as 2.075 stays 2.075.
- Numerical explanations from the saved decision, including an explicit statement when uncertainty is unquantified. Heuristic confidence labels are not described as win probabilities or calibrated confidence.
- Honest price limitations: unavailable quote timestamps and unverified bookmaker availability remain explicit. Forecast-generation time does not become a fabricated quote check.
- Shared recommendation identities in the common website/Discord card contract. Missing values remain missing and negative edges keep their sign.

The three-selection limit, five-fixture headline limit, current confidence/edge selection policy, compatibility rules and legitimate no-bets remain in place. This release does **not** activate the unqualified Phase 5 EV-based policy or change prediction weights, probability distributions, player effects or active ML models.

## What is recorded

Each new canonical market result retains the extracted quote inputs, the exact original selector output, all returned evaluated candidates, eligibility/rank outcomes, the final decision after safeguards, and prediction/input/policy/release identities. The ordinary immutable Match Read transaction persists those records together with fixture selection/exclusion summaries. No database migration is needed.

Coverage is explicit: this starts at normalized selector inputs; original provider parsing remains upstream. Generic selector eligibility failures retain their actual metrics and flags, without inventing finer rejection diagnoses. Historical cards are not backfilled with fabricated candidate logs. Website GET requests remain read-only; this release does not add a journal of every user-requested headline slate.

## Verified activation

- **133 Python tests passed**, including 24 new checks. Fourteen comparisons cover all seven markets with available/unavailable inputs against the preserved pre-release service; additional selector comparisons confirm that the passive audit preserves existing results.
- Actual JavaScript rendering/interaction checks passed for exact prices, identity, missingness, negative edges, escaping and quote expiry handling. Source restore and concurrent-edit protection passed a disposable rollback drill.
- The running website's health endpoint reports the exact deployed hashes. The homepage and static assets returned HTTP 200 with the release cache version.
- One bounded normal refresh processed the existing upcoming Bundesliga, Ligue 1 and La Liga fixtures. It appended Match Reads **553, 554 and 555**, their observations and normal website delivery records: two recommended reads and one no-bet. No synthetic recommendations were published.
- Those three records contain **21 audited market evaluations, 1,368 extracted options and 1,084 returned evaluated candidates**. Their actual website detail responses expose the new explanations and identities.
- The separate Discord adapter is installed and tested. **No Discord bot was started and no Discord message was sent.** Actual Discord delivery is therefore not verified by this release.

The bounded activation refresh used ordinary provider requests and existing production profile calculations. It was not a backfill or a research outcome evaluation. The permanent refresh cadence, odds archive collector and measurement job were not changed. Earlier experiment archives, model artifacts and publication records remain intact.

## Shadow transition and next evaluation

The presentation/audit source change creates a new source identity. The old diagnostic epoch remains intact. Its replacement is [phase6-shadow-communication-2026-10-06](../Research/phase6-shadow-communication-2026-10-06/README.md), with evidence under `Index/phase6_shadow_communication/`. The hourly launch agent now invokes that frozen controller.

Its first hourly slot correctly failed because saved forecasts still carried the old source identity. That failed slot is preserved. After the normal refresh, a separately labelled on-demand verification resolved the new forecasts and original archived quote bytes and successfully replayed both policies for all three fixtures, with zero worker IO violations. It does not replace or relabel the failed scheduled observation. The next hourly invocation can consume those refreshed forecasts.

The diagnostic end remains **20 October 2026, 09:37 UTC**; the transition does not extend it. The owner-approved **26-week performance comparison has not started**. The strict research candidate still has missing evidence/freshness support and selected zero bets in the verification. This release makes no claim of better forecast accuracy, calibrated probabilities or betting returns.

## Operation and rollback

The local website is managed by `com.bettingrag.website`, bound to `127.0.0.1:8000`. It is local-only; it has not been exposed to the internet. It runs while the Mac/user session is available. The Match Read worker remains in website mode.

```sh
.venv/bin/python -B -m Scripts.ops.website_launchd status
.venv/bin/python -B Research/phase6-communication-release-2026-10-06/deploy.py verify
.venv/bin/python -B Research/phase6-shadow-communication-2026-10-06/controller.py status
```

To roll back this release, the following command checks the installed hashes, stops only the three involved jobs, restores their original source/configuration and loaded states, and preserves databases, publications and collected evidence. A subsequent unrelated source edit causes refusal rather than being overwritten. Because the website was stopped before this release, this rollback also stops the local website.

```sh
.venv/bin/python -B Research/phase6-communication-release-2026-10-06/deploy.py rollback
```
