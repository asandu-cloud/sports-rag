# Gradual season-prior transition — 28 September 2026

Status: **implemented as an isolated candidate, tested and evaluated on permitted
historical development data. Not installed in the publishing checkout.** The bounded
comparison passes its predeclared pooled/supported-slice checks, with qualifications
below. Phase 4's baseline has not been frozen or universally qualified.

The owner authorized replacing the abrupt eight-match transition and evaluating it
before freezing the baseline. The [protocol](phase4-prior-transition-protocol-2026-09-28.md)
was saved before scoring. It selected one shared strength on 2022 and checked it on
2023, without opening 2024 confirmation or any calibration/final/prospective reserve.

## The change

The current engine uses prior weight `8 / (n + 8)` below eight current-season matches
and zero from eight onward. The candidate removes that cutoff and uses:

`prior_weight = k / (n + k)`; `current_weight = n / (n + k)`.

Here `n` is the number of current-season observations and `k` is prior strength,
expressed in match-count units. It does not mean using 16 seasons or restricting the
prior-season average to 16 fixtures. The candidate selected on 2022 uses **k = 16**.

| Current matches | Existing prior weight | Candidate prior weight |
| ---: | ---: | ---: |
| 0, with a prior | 100% | 100% |
| 1 | 88.9% | 94.1% |
| 7 | 53.3% | 69.6% |
| 8 | 0% | 66.7% |
| 9 | 0% | 64.0% |
| 16 | 0% | 50.0% |
| 32 | 0% | 33.3% |
| 38 | 0% | 29.6% |

This is continuous decay, including after the old cutoff. Strength 16 also increases
prior reliance before eight matches; it is a development-selected candidate rather
than a claim that continuity alone requires this stronger prior.

Without a prior, use the current profile. Without current observations, retain the
existing prior-only fallback. Explicit zero remains an observation; missing current
fields retain the existing prior-field fallback. Consequently a missing field may
use 100% prior information even when the overall profile weight is smaller. Research
reports those fallbacks separately; the global weight is not a per-field support count.

The candidate adds `season-prior-continuous.v1`, records strength and version in profile
audits, and identifies the existing effective-sample field as a **legacy support proxy,
not weighted ESS**. Its eight-match support cap is unchanged: selecting k=16 does not
invent eight additional observations or automatically raise support thresholds.
`PROFILE_PRIOR_WEIGHT_STRENGTH=16` controls weighting; the legacy
`EARLY_SEASON_PRIOR_STRENGTH=8` is preserved for that support proxy and frozen Phase 3
reconstruction compatibility. Recent form, league mixing and other forecast settings
are unchanged. The research adapter and candidate production profile function have
numerical parity tests at strengths 4, 8 and 16.

## Development evidence

The immutable Phase 3 dataset and baseline sidecar were reused. The reference was
replayed exactly for **20,004 eligible market/fixture observations**: 6,740 goals,
6,640 corners and 6,624 SoT observations across 2022–2023. Each candidate uses those
same fixtures; there were no candidate-specific unavailable predictions. 8,799 raw
development fixture rows include market-specific exclusions. Saved prediction rows
number 60,012 across the three candidates; these are overlapping markets/candidates,
not 60,012 independent matches.

One strength was selected by the mean of market RMSE ratios on 2022:

| Strength | 2022 mean RMSE ratio to the existing rule |
| ---: | ---: |
| 4 | 0.997786 |
| 8 | 0.995599 |
| **16** | **0.994522** |

The strength was then fixed for the 2023 check. These are reused development periods,
not an untouched final test. Lower RMSE is better.

| Market | 2023 observations | Existing RMSE | Candidate RMSE | Relative change | Weekly-bootstrap 95% interval for RMSE difference |
| --- | ---: | ---: | ---: | ---: | --- |
| Goals | 3,095 | 1.662763 | 1.657577 | −0.312% | [−0.009538, −0.001431] |
| Corners | 3,028 | 3.429111 | 3.405675 | −0.683% | [−0.036453, −0.012255] |
| Shots on target | 3,037 | 3.099025 | 3.094991 | −0.130% | [−0.015661, +0.006506] |

The mean RMSE ratio is **0.996248**, about 0.375% lower. The declared pooled,
cutoff-neighborhood and sufficiently supported league/market regression limits all
pass. The 5–11-match neighborhood improves by 0.214% for goals, 1.387% for corners and
0.862% for SoT. The bootstrap resamples 41 observed ISO weeks 2,000 times, seed 20260928.
Goals and corners show negative intervals; SoT's interval includes zero.

Do not omit adverse details: goals RMSE rises by 0.230% in the 0–7-match stage; SoT
MAE rises slightly (2.432724 to 2.433053); corners' average bias becomes more negative
(+0.003610 to −0.036242). The evidence supports a modest count-prediction change, not
improvement in every metric or stage. No betting ROI or probability-calibration
improvement was calculated.

## Scope and limitations

- EPL, LaLiga, SerieA, Bundesliga and Ligue1 meet the protocol's league/market slice
  minimum of 200 fixtures and 20 observed ISO weeks in 2023. Their checked regressions
  remain inside the declared limits; this is development support, not live qualification.
- Other competitions lack one or both requirements. UECL goals rise **3.336%** in RMSE
  over 93 fixtures/16 weeks; UEL goals rise 1.028% over 83 fixtures/14 weeks. These are
  explicitly adverse, under-supported slices, not cleared by the pooled result.
  Championship has 254 observations but only 19 weeks. No thresholds were relaxed.
- Cards were excluded and remain unqualified. A shared production profile change also
  affects other numeric profile fields, including variances. This evaluation covers
  reconstructed count means; it does not qualify card targets or distribution scores.
- Research uses exact identities, the immediately preceding provider season, safe
  stored history and assumed availability three hours after kickoff. Production retains
  its existing name/date resolution and possible older-prior fallback. Lineup/player,
  stakes/standings and other live contexts are omitted from the reconstruction. This
  is not a replay of originally published predictions or provider vintages.
- The five new domestic leagues have historical coverage gaps. Strength 16 is the best
  of three predeclared choices on 2022, not proof of an optimal weight for all leagues,
  markets, seasons or current squads. New team/manager/player effects are not modeled.

## Implementation and protection checks

Candidate workspace and complete evidence:
[`Research/prior-transition-2026-09-28`](../Research/prior-transition-2026-09-28/).

- `workspace/Scripts/rag_ingest/core/profile_prior.py`: pure continuous-weight function.
- `workspace/Scripts/rag_ingest/core/team_resolution.py`: gradual blending and audit fields.
- `workspace/Scripts/ops/profile_prior_development.py`: bounded comparison runner.
- `candidate.patch`, `rollback.patch`, `changed-files.json`, `before/`: five-file proposal
  with SHA256 identities. Both patches passed a disposable forward/reverse hash check;
  the forward patch passes against the current publishing source, including whitespace checks.
- **378 focused/regression tests passed** in the isolated workspace. Coverage includes
  n=7/8/9 continuity, monotone decay, absent current/prior profiles, zero/missing semantics,
  selected-strength audit, unchanged support proxy, production/research parity, temporal
  isolation, deterministic ordering, 2022-only selection, protected-value skipping,
  profile caching, canonical projections, quality gates and frozen baseline arithmetic.
- 18,737 qualified pre-2024 historical records were decoded. Another 7,067 later records
  were skipped structurally using kickoff metadata before statistical-value decoding.
  The entire source archive hash was verified. No lockbox file was opened.
- The runner's existing offline guard forbids network, SQLite, live-profile modules,
  non-allowlisted data access and writes outside the new experiment. Tests disabled
  dotenv loading and network access. No refresh, embedding, retraining or publication ran.
- All publishing-source fingerprints, active-model fingerprints, dataset development
  inputs, source history and saved baseline artifacts were verified unchanged.

The `evaluation/` directory preserves the original protocol hash, data/code identities,
coverage/exclusions, every fixture/snapshot prediction, metrics, uncertainty and decision.
The selected production-shaped patch was completed after selection; its hashes are
separate from the comparison runner's saved hashes. It has **not been installed**.

## Recommendation before freezing the baseline

Use the strength-16 gradual policy as the leading baseline candidate for further Phase 4
review. Record supported scope explicitly; retain the adverse sparse-league findings
and the open card/distribution checks. Do not silently apply pooled qualification to
all thirteen leagues and all markets. The present publishing engine keeps its existing
rule until a production scope/change is approved. Keep the old code and outputs as the
comparison reference, and retain calibration/final-system reserves for their agreed use.
