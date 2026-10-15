# Phase 4 Step 2 — forward backtest execution contract

Owner authorized starting the backtest on 29 September 2026. This execution
clarification is written before scoring. It supplements the unchanged frozen v1
protocol and candidate v2 amendment; no reserve access or promotion is authorized.

## Fixed run

- Use only sealed Step 1 snapshots, eligibility, labels and pre-2024 history.
  Initial pre-2022 history may fit team strengths; no 2024+ labels are read.
- Four calendar-quarter forward folds in 2022. Fit attack/defence team state at
  00:00 UTC on each quarter boundary, separately by competition, provider season,
  market and declared ridge. For a new provider season appearing in that quarter,
  use the same quarter-boundary cutoff and its current/prior season window.
  Sparse fits/teams retain the declared fallback; nonconvergence is an invalid
  candidate, never a successful fallback. All bundle IDs and fit cutoffs are saved.
- All fixed mean/distribution settings are predeclared, not fitted to the fixture
  being forecast. Compare their genuine chronological 2022 forecasts to select
  one setting per family and market using primary NLL. Require the frozen pooled
  tuning support (500 fixtures/26 weeks); invalid or incomplete settings cannot
  win. Ties within 1e-8 favor stronger ridge, rho closest to control, then the
  smaller absolute parameter. The unchanged control remains a separate comparator.
- Lock selected settings and their source hashes before any 2023 scoring. Only
  selected settings enter that later development comparison. Deterministic
  quarterly team-state refits in 2023 may use previously available results; ridge,
  candidate choices and all other hyperparameters remain fixed. This is rolling
  state estimation, not 2023 hyperparameter tuning. Log every such fit.
- Within each quarter, profiles/recent rates update from strictly earlier available
  history under the existing Step 1 cutoff contract. Simultaneous fixtures and
  forecast versions cannot enter different partitions. A fixture uses one saved
  reconstructed pre-kickoff stage; other stages remain unavailable.
- All candidates retain control eligibility per market. Save every invalid and
  fallback record; an invalid member disqualifies that setting on the common
  cohort. Do not claim improvement from dropping difficult fixtures.
- Richer lineup candidates are unavailable because historical player evidence is
  unqualified. Do not score their neutral fallback as evidence for lineup effects.
  Cards remain outside this backtest pending target qualification.
- No combined candidate is selected automatically. Review individual development
  evidence before declaring any bounded combination in a separate manifest.

## Numerical execution and validation

Use the preserved engine's functions through the research adapter. The new
rates-only path omits unpriced publication payloads and repeated diagnostic PMFs,
not any numerical mean/variance calculation. Test fast/full candidate parity and
reproduce saved control means/variances for every eligible fixture. Fixed
probability-only settings reuse the identical control rates. Cache input-only
lookups within one fixture; keys include its cutoff and cache state is cleared
before the next fixture. No fitting/inference or candidate cache is shared across
quarter worker processes.

Primary scores use the exact observed count/score likelihood, including outcomes
outside finite diagnostic PMF arrays. PMFs retain the frozen 1e-10 tail limit.
Save reproducible distribution parameters and dated rate evidence with predictions;
these are sufficient to regenerate the entire PMF. Secondary binary/multiclass
log scores use a 1e-15 numerical probability floor, disclosed separately from the
unclipped primary likelihood. Diagnostic lines are the seven frozen totals and
nine handicaps, not historical bookmaker quotes. No ROI, CLV or odds are inferred.

Report NLL, MAE/bias/RMSE/R², count RPS, 80%/95% interval coverage/width, binary
Brier/log loss, winner/Asian categorical scores, class support and ten-bin
reliability. Calibration intercept/slope fitted for reporting are diagnostics
only and are never applied to forecasts. Later calibrator fitting remains Step 3.

Use the unchanged 2,000 paired calendar-week bootstrap with seed 20260928. Report
league, provider season, season stage (minimum current-season matches across the
teams: 0–7/8–15/16+), forecast stage and attacking-xG missingness slices. Other
omitted contexts are constant unavailable in this cohort, so no observed-effect
claim is possible for them. League/slice conclusions need 200 fixtures/20 weeks.
Quarter-level counts/scores are descriptive when the 20-week requirement is not
met; pooled annual development comparisons apply the unchanged thresholds.

A development claim needs at least 0.5% primary improvement, paired 95% interval
entirely below zero, and no supported slice or related goal market deterioration
above 2%. Apply these to all submitted settings without relaxing after scoring.
Conservative total/Asian regression checks are also reported for count markets.
These are development screens, not later-period qualification or proof of profit.

## Artifact and resource boundary

A new immutable experiment directory preserves input/source/dependency hashes,
protocol, quarter membership, fitting bundles, selection lock, compressed forecast
and evidence shards, coverage/failure counts, reports and test evidence. Keep
failed or interrupted attempts distinguishable; never overwrite a completed run.
At most four independent local worker processes run concurrently, with numerical
thread pools limited to one thread each. No API calls, refresh tasks, database
writes, publishing-mode changes or test recommendations are part of this run.
