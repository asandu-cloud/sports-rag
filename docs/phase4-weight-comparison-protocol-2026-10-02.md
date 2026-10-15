# Bounded complete-setup comparisons — 2 October 2026

Owner authorization: test different setups and report back; consult the owner
before any implementation/promotion decision. This experiment makes research
comparisons only. It does not replace the retained reference, enable a model,
start Phase 5, collect data or access later reserves.

## Frozen design, before scoring

At most 60 configurations per market, including reference configurations.
Use deterministic registries, seed 20261002: the unchanged setup, individual
parameter anchors, then unique sampled combinations filling the fixed budget.
No extra candidates, new ranges or changed acceptance thresholds after results.
The machine-readable registry records every exact combination before inference.

Goals/corners/SoT use the saved 2022/2023 historical inputs. Primary scope remains
the five major domestic leagues, both teams with >=8 current-season matches.
Cards use the qualified v2 inputs from the completed card experiment, all ten
domestic leagues with the same >=8-match gate. Per-market eligible fixture
membership is identical across configurations; missing predictions invalidate a
configuration rather than dropping difficult fixtures. Other scopes are reported
as outside this comparison, not certified by pooled improvements.

All profiles use current/immediately preceding provider seasons, strictly prior
UTC-day results and the existing kickoff+3h availability rule. Current/future
results, same-day results and unqualified card targets cannot affect inputs.
No larger season-window expansion is part of this search. More stored seasons
do not silently enter prediction profiles. Historical evidence remains assumed_final.

Tuning: eligible 2022 forecasts, with support >=500 fixtures/26 observed weeks.
Rank by pooled count NLL (joint home/away score NLL for goals), breaking numerical
ties within 1e-8 in favour of the reference, then fewer changes and stable ID.
Save all tuning ranks and one preselected setup per market before 2023 inference.
This statistical preselection is not a decision to implement a setup.

Later development: all eligible 2023 forecasts for goals/corners/SoT;
April–December 2023 for cards, preserving comparison with the earlier calibrated
card candidate. Both periods are already inspected development evidence.
Evaluate all declared setups descriptively; only the 2022-preselected setup is
the primary comparison. The best observed 2023 row must not replace that primary
choice or be described as independently confirmed. Report quarter consistency,
rank changes, uncertainty, league/stage/missingness slices and per-market coverage.
January–June 2025 stays out of this experiment. July 2025–June 2026 and the later
prospective reserve remain closed. This is no new production qualification.

## Configuration axes

- Own/opponent: .50/.50, .60/.40, .70/.30 for goals/corners/SoT;
  .50/.50, .70/.30, .85/.15 for cards. Corner/SoT recent own/opponent arithmetic
  uses the same candidate weight as the season component.
- Recent share: 0, .10, .15, .20, .30; season share is its complement. The frozen
  divergence adjustment is retained. Existing market-specific defaults remain.
- Venue share: .25, .50, .75 (cards also use these explicit shares).
- Recent window: 4, 6, 8, 12 prior matches; positional decay .85 by default.
- Recent calendar half-life: absent (positional), 30, 90, 180 days. Calendar
  weighting applies to the recent component only; season averages stay unchanged.
- Statistical profile mode: unchanged; gradual previous-season prior with
  strength 8/16; league pooling 8/16; venue pooling 8/16. Each mode is explicit
  and mutually exclusive. League pooling uses each field's known support and
  dated league rate. Venue pooling uses known venue support and its team rate.
- Goals: attacking xG share .25/.50/.75; retain the frozen availability fallback.
- Cards: pooling 0/8/16; referee both/modifier-only/anchor-only/residual/none;
  foul blend 0/.075/.15/.30. Keep original v2 target arithmetic and known referee
  identity. No actual historical XI, invented aggression or knockout context.

Profile/window variants are computed once per dated snapshot, then reused across
configurations. Preserve source fixture IDs, known observation support and
effective sample sizes. Missing values never become zero. Controls must reproduce
the archived actual projection functions numerically before candidate scoring.

## Fixed distributions and controls

Goals retain coherent Dixon–Coles rho=-.10. Corners/SoT retain alpha .025/.01;
their original variance-based controls are separate references. Changing a mean
does not inherit the retained components' offline qualification.

Cards' primary reference is the fuller aligned control. Freeze its per-fixture
dispersion parameter calculated from the existing observed-variance rules and
apply that same parameter to changed means. Do not refit dispersion or calibration.
The simple fixed referee/Poisson and previous frozen pool16/NB/calibrated candidate
remain diagnostics. The latter is evaluated only on April–December 2023 because
its calibration used data through March 2023; it cannot enter 2022 selection.

## Assessment and stopping rule

Primary development screen: >=0.5% NLL improvement, paired ISO-week bootstrap
95% difference interval below zero, >=200 fixtures/20 observed weeks and no
supported league/stage/missingness or related-market loss regression >2%.
Use 2,000 bootstrap draws, seed 20261002, and report Holm-adjusted one-sided
centered-bootstrap p-values for the four preselected market comparisons.
The multiplicity screen requires adjusted p<=.05. Report insufficient slices.

Also report MAE, RMSE, bias, interval coverage, Brier/log loss and reliability on
the existing total lines; goals retain coherent BTTS/winner/Asian diagnostics.
No ROI, hit-rate optimization or fabricated historical prices. Distribution
normalization, Asian arithmetic and original control parity remain test gates.

Preserve registry, input/source/dependency fingerprints, code, membership,
predictions, tuning lock, complete result tables, uncertainty, exclusions and
verification. Return the findings to the owner without choosing what to deploy,
changing the retained research bundle or starting another optimization round.
