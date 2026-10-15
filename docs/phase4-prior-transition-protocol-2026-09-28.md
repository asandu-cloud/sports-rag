# Prior-transition development protocol — 28 September 2026

Declared before calculating comparison scores. This bounded experiment repairs the
profile-weight discontinuity; it does not fit probability calibration or reopen
Phase 3 confirmation. The publishing engine stays unchanged during evaluation.

## Fixed comparison

Reference: current production-shaped statistical core reconstruction, previous-season
weight k/(n+k) below n=8 and zero from n=8 onward, k=8. Candidates: continuous prior
weight k/(n+k) for all positive current-season counts, k in {4, 8, 16}. No prior gives
weight zero; no current observations gives the existing prior-only fallback. The
prior is the immediately preceding provider season in research. Production's existing
season resolution, per-field missing fallback, current sample count, recent window,
European mixing, profile support proxy and scoring settings are otherwise unchanged.

The k=8 candidate preserves weights below the old cutoff. It retains 50% prior weight
at n=8, 33.3% at n=16 and 20% at n=32. The strength is a tuning parameter, not a claim
that eight prior observations are optimally informative. No mean is imputed as zero.

## Data and access

Use the frozen phase3-dataset-2026-09-26 development reader and evidence-qualified
history. Decode historical observation records only when kickoff plus three hours
is strictly before 1 January 2024. Skip later records structurally before decoding
statistical fields. Read no lockboxes, production databases or provider APIs. Preserve
all existing experiments and feature contracts. Results are explicitly assumed-final
historical core reconstructions, not original published predictions.

Evaluate only existing eligible goals, corners and shots-on-target fixtures. Keep
identical reference/candidate fixture sets. Cards receive no accuracy qualification.
Earlier results inside a development block may enter a later fixture's inputs only
when their assumed availability precedes that fixture's forecast cutoff.

## Selection and later development check

- Select one shared k on January–December 2022 only. Rank by the unweighted mean of
  the three market RMSE ratios to the reference (each market itself weights fixtures
  equally). Ties within 1e-12 favor k=8, then 4, then 16. Do not select per league.
- Check that frozen choice on January–December 2023. Report all three candidates as
  disclosed sensitivity analyses; do not switch choice using 2023 results.
- A candidate qualifies for a proposed baseline update only if the 2023 mean RMSE
  ratio is <=1; no individual market RMSE regresses by >1%; no supported league/market
  slice (>=200 fixtures and >=20 distinct ISO weeks) regresses by >3%; and no supported
  cutoff-neighborhood market slice (either team's effective primary profile has 5–11
  current matches, >=100 fixtures) regresses by >2%. No threshold changes after scoring.
- Report MAE, RMSE, bias, R², coverage, weights and missing-field dependence, including
  primary-profile stages 0–7, 8–15 and 16+ matches, league, provider season and half-year.
  Count fixtures once per market; use the smaller of the two primary counts for stages.
- Report paired ISO-week block-bootstrap 95% intervals for candidate-minus-reference
  RMSE by market, 2,000 resamples, seed 20260928. Intervals including zero do not establish
  an accuracy improvement. This is reused development evidence, not an untouched test.
- If the frozen choice fails, retain the existing publishing engine and document the
  candidate and blocker. Do not extend this search or consume reserved outcomes.

## Required evidence

Save protocol and code/data hashes, source snapshot, exact fixture/snapshot membership,
reference replay checks, per-fixture predictions, coverage/exclusions, scores, uncertainty,
selection decision, source patches and tests in a new experiment. Tests cover continuous
monotone weights, n=7/8/9, no current/prior observations, zero versus missing, versioned
profile audit, pre-cutoff isolation, reserve rejection, deterministic output and parity
with the frozen reference. Preserve active models, databases, gates and public modes.
