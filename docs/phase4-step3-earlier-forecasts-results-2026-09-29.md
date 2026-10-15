# Phase 4 Step 3 — earlier forecasts and extended calibration results

**Subsequent status:** the [Step 4 finalist review](phase4-finalist-review-results-2026-09-29.md)
is complete. Step 5's outcome-free preflight found insufficient qualification-period
support; its calibration fitting and qualification evaluation have not run.

The specific challenger-calibration support shortage is resolved: all four 2022
forward quarters now qualify, giving **43 observed validation weeks instead of
13**, without moving the 2023 comparison boundary or lowering support thresholds.
No provider download, database update or production change was required.

The owner authorized this extension after the initial Step 3 report. The
[prospective amendment](phase4-step3-earlier-forecasts-protocol-2026-09-29.md)
explicitly permits earlier forecasts for nested base selection and calibration
fitting. Original protocols and experiments remain preserved. This is additional
development analysis; 2023 was already inspected and is not a fresh test.

## Earlier evidence and validation support

Reconstructed all available saved earlier quarters (2019–2021), using the actual
frozen engine inputs and the existing candidate grids. Quarterly strength fits
use only earlier qualified labels and the same current/prior-season policy.
Fitted count dispersion candidates use the saved control rates. No new settings,
features, retrospective prices or tactical context were introduced.

| Forecast family | Earlier fitting fixtures | Earlier observed weeks | 2022 validation fixtures | Validation weeks |
|---|---:|---:|---:|---:|
| Goals / derived BTTS | 8,909 | 99 | 3,645 per outcome | 43 |
| Corners | 8,823 | 99 | 3,612 | 43 |
| Shots on target | 8,495 | 99 | 3,587 | 43 |

All 13 competitions contribute to the pooled earlier evidence. The 99 earlier
weeks initialize fitting; they are not counted as validation weeks. Final binary
calibrators are fitted on 2022 out-of-sample base forecasts only. Final base
settings still reproduce the full-2022 Step 2 lock. No 2023 calibrator refitting.

The earlier experiment produced 208,816 market/setting forecasts and 291 fitted
strength bundles. All 8,909 eligible control replays match exactly; no invalid
forecasts occurred. There were 78 insufficient strength fits. Across the three
goal-strength settings, 4,491 forecast records use missing-bundle fallback and
2,643 use unknown/sparse-team fallback; these are setting-level records, not
distinct fixture counts, and remain included in the common evaluation evidence.
Six independently refitted bundles reproduce exactly; every bundle's training
IDs, target hash and cutoff were separately audited.

All 2022 nested folds chose goal ridge 10, corner alpha .01 and SoT alpha .01.
The unchanged full-2022 final selection uses goal ridge 10, corner alpha .025
and SoT alpha .01. The corner difference is legitimate chronological selection:
the earlier folds cannot know the final full-year choice. All four challenger
sigmoids select C=.01 on 2022 forward predictions.

## Additional calibration result

The table compares each calibrated challenger with that same challenger's raw
probabilities on the unchanged 2023 common cohort. Negative change is favorable.
Each cohort spans 41 observed weeks. These are log-loss changes, not hit-rate or
profit improvements.

| Outcome | Fixtures | Raw log loss | Calibrated log loss | Relative change | Paired 95% difference interval | Development screen |
|---|---:|---:|---:|---:|---|---|
| BTTS | 3,095 | .688471 | .688726 | +0.037% | [-.003033, .003963] | No added improvement |
| Goals Over 2.5 | 3,095 | .681374 | .682120 | +0.109% | [-.003067, .004622] | No added improvement |
| Corners Over 9.5 | 3,028 | .695973 | .689973 | **−0.862%** | **[-.010072, -.002057]** | **Pass** |
| SoT Over 8.5 | 3,037 | .675535 | .676220 | +0.101% | [-.002731, .003582] | No added improvement |

The corner result also passes against the frozen control's Over 9.5 probability:
.694436 to .689973, a **0.643%** reduction, interval [-.007862, -.001188]. Brier
improves from .251186 to .248416 against the raw dispersion challenger. Its
diagnostic calibration slope moves from .408 to 1.086. These diagnostic slope
estimates are reported on evaluation data and are not refitted production models.
The unchanged 0.5% improvement, negative interval and supported-slice regression
criteria apply. No supported slice exceeds the 2% regression limit for this claim.

This qualifies only a binary development claim for Over 9.5. It does not calibrate
the complete corner distribution, other lines or Asian outcomes. The fitted
dispersion improves full-count NLL but its raw Over 9.5 loss is worse than the
control; the table exposes that difference rather than treating every derived
market as improved automatically. The two improvements cannot be added together.

Joint-goal temperature selects **1.0 (identity)** for both control and challenger.
The challenger now makes that choice with 3,645 validation fixtures/43 weeks:
its 2022 joint losses are 2.964316 at T=1 and 2.964370 at T=1.1. This is a supported
development selection of identity, replacing the earlier insufficient-evidence
fallback. The existing opponent-strength goal gain remains a Step 2 result;
temperature adds no improvement to it. Binary goal adjustments are not wired
into the coherent goal distribution.

## Verification and preservation

**523 tests passed**, including six additional checks for explicit earlier-history
authorization, deterministic membership, future-label exclusion, unchanged default
selection behavior, invalid candidate slates and rejection of later forecast dates.
The expanded independent calibration audit verifies **204 fitted bundles** and
**20 nested/final base choices**, and replays **210,633 binary predictions with
zero probability differences**. All 49,930 joint-distribution records meet the
1e-10 transformed-tail bound. All 140,422 control-based binary predictions match
the original Step 3 run exactly, and the prepared 2022/2023 input files are byte
identical to that run. Evaluation membership, label cutoffs, training-only scaling
and the method/bundle lock pass. Detailed evidence is preserved in
`validation/independent-audit.json`.

The earlier forecast artifact has 397 sealed files. Both experiments preserve
source/data/dependency hashes, the amendment, fitting cutoffs and IDs, fallbacks,
OOF forecasts, method/bundle locks, raw/calibrated probabilities and scoring evidence:

- `Research/phase4-step3-earlier-forecasts-2026-09-29/`
- `Research/phase4-step3-calibration-extended-2026-09-29/`

The `...earlier-forecasts-2026-09-29-preflight/` directory contains an unscored
preparation capture, superseded before execution; it is not a completed experiment.
All earlier sealed archives remain unchanged. The extension verifies 458 frozen
runtime/test/model/dependency files against the current checkout with no changes.
No provider, SQLite or publishing access occurs in the forecast/calibration workers.
No 2024+ outcomes, dedicated calibration, qualification or final-system tests are
opened. Historical availability remains `assumed_final`; these forecasts reconstruct
cutoff-controlled inputs, not the exact recommendations originally published.

## What remains

The missing-window prerequisite is closed for these pooled comparisons. Sparse
league and early-season evaluation slices retain their existing limitations:
earlier fitting data does not increase their 2023 evaluation sample. Cards and
richer lineup effects still require their separate evidence qualification.

Next is the Step 4 review of a complete supported configuration, keeping raw
distributions and binary diagnostics distinct. Combined-configuration assessment
and Step 5 calibration/qualification remain outstanding. The full probability
and selection-system final test remains reserved for Phase 5. No public promotion
is authorized or implied by this development result.
