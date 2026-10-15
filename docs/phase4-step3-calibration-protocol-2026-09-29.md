# Phase 4 Step 3 — logistic comparison and calibration execution

Owner authorized Step 3 on 29 September 2026. This clarifies implementation of
frozen evaluation v1 and candidate v2 before new scoring. Their criteria and
reserved periods are unchanged. All work is isolated research; no publisher,
provider request, live database, production bundle or delivery mode is changed.

## Models and fixed budget

1. Four binary logistic comparators: BTTS, goals Over 2.5, corners Over 9.5,
   SoT Over 8.5. Features are the frozen control probability logit, control mean,
   home-minus-away goal means for BTTS/goals, and training-observed league one-hot
   indicators. C=.01/.1/1/10; seed 20260928; uniform training weights. Standardize
   continuous features using training-only population means/standard deviations;
   zero standard deviations become one. No new features or missing-as-zero
   imputation. Unknown leagues use the pooled numeric component (all league
   indicators zero), with an explicit flag.
2. Identity versus pooled monotone sigmoid calibration for those four outcomes,
   separately on control and the three Step 2 finalist families. Use standardized
   raw logit only; its coefficient is constrained >=0. The same C grid applies.
   No league coefficients in these calibrators. Binary outputs remain diagnostic
   for their specified outcome and cannot replace a count PMF or coherent goal
   system. They do not qualify other lines or Asian profiles.
3. Identity versus joint-goal temperatures .75/.9/1/1.1/1.25/1.5 on the control
   and opponent-strength goal family. Raise every joint cell to 1/T, renormalize,
   and regenerate totals, BTTS, winner and nine Asian handicap profiles from that
   single distribution. Temperature 1 is identity. Expand the score support using
   a geometric bound on the *transformed* Poisson tails until omitted normalized
   mass is <=1e-10; simply powering a previously truncated matrix is not sufficient.
   The four Dixon–Coles corrections remain in the transformation. Invalid
   distributions fail explicitly; no clipping of negative cells.

Logistic optimization minimizes mean binary NLL plus ||coef||²/(2*C*n), with
unpenalized intercept, L-BFGS-B, maxiter=2000, ftol=1e-12, gtol=1e-8. Comparator
coefficients are unconstrained; the sigmoid's sole slope has lower bound zero.
Nonconvergence is an explicit failure. Input logits and diagnostic log losses use
a disclosed 1e-12 numeric probability floor; primary logistic losses use stable
logaddexp from the unrounded fitted logit. No probability/value/confidence identity
is collapsed: estimates carry fitting support and version, not a profitability or
confidence claim. Forecast quality gates are not relaxed.

## Chronology and nesting

Use only sealed earlier-control (2019–2021), Step 1 and Step 2 development artifacts.
The four 2022 calendar quarters are forward folds. Labels must have kickoff+3h
strictly before each fitting cutoff; UTC boundaries keep simultaneous fixtures
together. Minimum fitting support is 500 fixtures/26 ISO weeks and 50 of each
binary outcome. No target or held-out feature may affect training transformations.

- Control-based comparators/calibrators can initialize from the existing earlier
  reconstructed control forecasts and then expand through preceding 2022 folds.
- Challenger calibration has no earlier fitted-candidate forecast archive. Do not
  invent one. At each 2022 fold select the base family parameter by primary NLL on
  preceding eligible **2022** forecasts only, respecting 500/26 support. This uses
  the original candidate grid and tie rules; it cannot consult full-2022 or 2023
  selection. Train the calibrator on preceding predictions for that chosen setting.
  Folds without enough prior support are explicitly unavailable for challenger
  calibration tuning. Score all C/temperature choices on the same available folds;
  at least 200 fixtures/20 weeks are required to select a calibration method.
- The control temperature grid uses all 2022 control predictions. The challenger
  temperature grid uses the same nested, supported 2022 base selections as above.
  No base parameter is selected using its own validation-quarter outcomes.
- Choose C/method/temperature using pooled 2022 OOF losses only. Calibration ties
  within 1e-8 prefer identity, then stronger regularization; temperature ties prefer
  identity, then the smallest distance from one. Comparator C ties prefer smaller C.
- Final comparator fitting uses qualified earlier history through 2022. Final
  sigmoid calibration uses **2022** OOF base predictions only. The challenger base
  setting comes from the existing full-2022 Step 2 lock (and is independently
  recomputed). Fit once before 2023; no calibrator refit on 2023 labels. Preserve
  the already-frozen quarterly base team-state updates from Step 2.
- Save and checksum-lock all methods and fitted bundles before the 2023 comparison.
  The training process cannot read the prepared 2023 shard. 2023 is inspected
  development, never an untouched new test. No 2024+ outcomes are accessed.

## Reporting and acceptance

Compare each binary comparator against the frozen control. Compare each calibrated
system against its own identity and, separately, the frozen control. A combined
system can retain a Step 2 gain while calibration itself fails; report those two
claims distinctly. Temperature selection evaluates joint-score NLL, not whichever
binary goal market happens to look best. Retain identity when it wins development.

Use identical fixture membership, explicit unsupported/invalid counts, raw and
calibrated probabilities, fitting IDs/support, method versions and input hashes.
Report binary log loss, Brier, class counts, 10-bin reliability and diagnostic
calibration slope/intercept; for coherent goals also report joint/count/winner/Asian
scores, 80/95 intervals, numerical errors and distribution coherence. Use the
frozen 2,000 paired calendar-week bootstrap, seed 20260928, >=.5% relative primary
improvement, entirely negative 95% difference interval and <=2% supported-slice
regressions. Goal-derived outcomes retain the same regression limits. Slice
support remains 200 fixtures/20 weeks. Unsupported cohorts retain their existing
reference; no moving boundaries after scores are seen. Later qualification's Holm
correction and distinct calibration/qualification/final-system periods remain open.

Cards and richer lineup effects remain evidence-gated. Count sigmoid diagnostics
are not new distribution-level calibrators; fitted dispersion remains the current
count-distribution candidate. New count-CDF transformations would need a separately
specified experiment. No betting ROI is computed from invented prices.

Preserve a new immutable experiment with source/data/dependency hashes, nested
base choices, OOF predictions, final JSON bundles, selection lock, 2023 predictions,
metrics, tests, audits and review. Separate implementation correctness, development
performance and eventual qualification; none constitutes production promotion.
