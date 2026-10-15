# Bounded card improvements — 2 October 2026

The approved card research is implemented and evaluated. The frozen candidate
improved on the simple fixed reference, but **failed the later qualification
gate against the fuller aligned control**. It remains an isolated research
candidate. Production predictions, profiles, settlement policies and delivery
modes were not changed.

The owner approved the bounded work after reviewing the fixed card backtest,
and explicitly approved reusing January–June 2025 with its historical-test
limitation. This supersedes the earlier stop-before-optimization instruction
for this experiment only. The [development protocol](phase4-card-improvements-protocol-2026-10-02.md)
and [later-check protocol](phase4-card-reused-qualification-2026-10-02.md) record
the candidate budget, periods, thresholds and access rules.

## What was implemented

The research adapter executes the archived engine's actual card projection
functions with saved, dated participation-v2 profiles. It includes team and
opponent rates, venue and recent form, referee adjustments and available foul
context. The original fixed referee/Poisson reference is retained, and its
2,771 original 2023 predictions and input snapshot hashes reproduced exactly.

The fuller control uses the engine's observed-variance/fallback rules as well
as its reconstructed mean. Neither reference is an exact replay of historical
public recommendations: original forecast snapshots and some contexts do not
exist. Aggression/duel context and historical league-regime context remain
omitted consistently. Original referee-assignment timestamps are unknown.

The bounded mean registry contains the unchanged fuller mean plus 15 alternatives:
own/opponent weights, season/recent weights, venue weights, sample-aware pooling
at strengths 8 and 16, calendar half-lives of 90 and 180 days, four referee
alternatives and the foul-component ablation. Referee alternatives include a
shrunk residual adjustment based on earlier outcomes against each earlier
match's own pre-match team prediction. Referee name fragments were not merged
without independent identity evidence.

Earlier training data selected:

- Team/venue pooling with **prior strength 16**, retaining the existing other
  coefficients and referee adjustments.
- A negative-binomial count distribution with **NB2 alpha 0.05**.
- A monotone count-CDF calibrator fitted to out-of-sample predictions:
  `a=0.06798717098172594`, `b=1.0107094387736915`, regularization `C=0.1`.

No additional rate/referee combination earned the protocol's separate 0.5%
training improvement requirement. No league-specific switches were selected.
The final parameters were frozen before the later check; they were not refitted
on 2024 or 2025 outcomes. Historical profiles advance using earlier results,
which is distinct from retuning the model.

All total-card probabilities and full/half win, push and loss probabilities
come from the same normalized count distribution. The actual historical XI was
tested only as a labelled conditional scenario using earlier player evidence.
It is not an input to the selected model and does not establish original
pre-match lineup availability or complete the player-prop engine.

## Data and chronology

Development used the 9,325 previously qualified pre-2024 targets; all have
two-team foul counts in the immutable source backup. Histories retain the
current and immediately preceding provider season, same-competition identity,
and kickoff plus three hours strictly before the forecast day's UTC midnight.
Missing values remain unknown. Targets follow the two-minute participation-v2
convention; it is not established bookmaker equivalence.

| Purpose | Period and support | Interpretation |
| --- | --- | --- |
| Mean/dispersion selection | Earlier forecasts before January 2023; 3,315 fixtures / 46 ISO weeks | Training only |
| Calibration | Nested July–December 2022 plus frozen-recipe January–March 2023 predictions; 1,704 / 26 | Out-of-sample predictions; first two 2022 folds lacked prior support |
| Development comparison | April–December 2023; 1,730 / 23 | Reused development period, already inspected |
| Later historical check | January–June 2025; 1,751 / 21 | Explicitly reused for cards after other-market assessment |
| Final-system test | July 2025–June 2026 | Outcomes remain protected |
| Prospective reserve | July 2026 onward | Outcomes remain protected |

For the later check, the unchanged source-qualification rules admitted 5,499
of 6,473 additional reconstruction records, including 2024 historical context.
There were 1,817 source-qualified January–June 2025 targets; 1,751 met the fixed
forecast-history gate. Exclusions remain in the saved target records and the
preparation report. Extra-time/special-round handling remains deferred.

The source captures are `assumed_final`, acquired later than the matches.
Chronological reconstruction cannot certify exactly what the provider exposed
at the original forecast time. Existing non-null evidence conflicts, incomplete
participation, unknown event times and unresolved identities remain exclusions.

## Development result

Lower count negative log likelihood (NLL), MAE and Brier scores are better.
Positive bias means overpredicting weighted cards. All rows use identical
1,730-fixture membership.

| Model | Count NLL | MAE | Bias | Brier over 4.5 |
| --- | ---: | ---: | ---: | ---: |
| Fixed referee / Poisson reference | 2.221408 | 1.8341 | +0.2028 | 0.234557 |
| Fuller aligned / observed-variance control | 2.217344 | 1.8246 | +0.1839 | 0.230694 |
| Frozen calibrated candidate | 2.197439 | 1.7995 | +0.0619 | 0.229348 |

The candidate reduced NLL by 1.079% against the fixed reference and 0.898%
against the fuller control. Paired ISO-week bootstrap 95% intervals for the
NLL differences were respectively `[-0.033932, -0.013486]` and
`[-0.030822, -0.009882]`. Both pooled development gates passed.

No individual league met both 200 fixtures and 20 weeks in this shorter
development window. This was a pooled pass, not ten separate league
qualifications. The separately reported expanded early-season/European cohort
also lacked individual league support and did not enlarge the qualified scope.

The components did not each establish a substantial independent gain. On this
period, the selected mean improved 0.257% against the fuller Poisson mean;
dispersion improved a further 0.484%; calibration's incremental gain was 0.070%
with an interval including no improvement. The conditional lineup scenario
improved only 0.086% against the fuller control. These diagnostics do not justify
claiming every proposed change helped or enabling the lineup scenario.

## Frozen January–June 2025 check

All rows below use the same 1,751 fixtures across 21 ISO weeks and ten domestic
leagues. January–June 2025 is **reused historical evidence, not an untouched test**.

| Model | Count NLL | MAE | Bias | Brier over 4.5 | 80% interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed referee / Poisson reference | 2.238920 | 1.8531 | +0.2469 | 0.228868 | 81.67% |
| Fuller aligned / observed-variance control | 2.220009 | 1.8503 | +0.2282 | 0.227070 | 89.83% |
| Frozen calibrated candidate | 2.213462 | 1.8359 | +0.1322 | 0.226520 | 85.44% |

| Primary comparison | NLL improvement | Paired 95% interval for NLL difference | One-sided bootstrap p | Gate |
| --- | ---: | --- | ---: | --- |
| Candidate vs fixed reference | 1.137% | [-0.034768, -0.017283] | 0.00050 | Pass |
| Candidate vs fuller control | 0.295% | [-0.016221, +0.003855] | 0.09095 | **Fail** |

The protocol requires at least 0.5% improvement, an interval wholly below zero,
`p<=0.05`, sufficient support and no supported slice regression above 2%, against
**both** controls. The fuller-control comparison fails the improvement and
uncertainty requirements. The overall qualification result is therefore false.

All ten domestic leagues meet this later check's 100-fixture/10-week support
minimum. None has a point regression above the 2% limit against the fuller
control. BelgianProLeague, Championship and EPL have small point regressions;
most individual intervals include no improvement. Full descriptive league
intervals are saved with validation, without selecting favourable leagues
after seeing the results. Sparse-foul and earlier-season slices still lack
support. European markets remain unqualified by this common cohort.

Calibration reduced later bias and had a small incremental NLL improvement
against the raw candidate, but that gain was only 0.114%. It does not rescue the
failed primary comparison. Nominal interval coverage and reliability also
remain imperfect; all per-line reliability bins, RMSE and 95% coverage are saved
in the reports. No historical betting ROI was calculated.

## Verification and preserved incident

- **116 focused/regression tests passed**, covering chronology, future-result
  independence, fitting isolation, missing versus zero, reference parity,
  conditional-lineup isolation, CDF/Asian settlement consistency, protected
  payload refusal, original reader-guard preservation and preparation isolation.
- Both development and qualification replayed their result files exactly.
  The method and calibration locks also reproduced in development replay.
- An independent input audit checked 3,229,746 historical fixture links across
  9,325 development snapshots and 1,068,726 links across 1,751 later snapshots.
  No current, future, same-day or excluded target entered the admitted history.
  Team/opponent/referee membership and source/input hashes were checked.
- Count PMFs and complete Asian settlement vectors passed normalization and
  complementarity checks, with count tail error at most `1e-10`. Frozen method
  and calibration identities remained unchanged throughout qualification.
- Original source seals still verify. No provider calls, active database writes,
  embeddings, profile rebuilds or production changes occurred in this work.

The first later-period preparation reached source reconstruction but failed
while finalizing its report: the offline guard rejected a redundant database
checksum read after entry into the no-database phase. The access record and
partial output are preserved with `PREPARATION_FAILURE.json`. No qualification
forecasts or metrics had run. Hash validation was moved into the read-only
preparation phase, an integration regression test was added, and identical
preparation succeeded in a new `-v2` directory. Statistical methods, periods and
thresholds did not change. This was a resolved preparation failure, not a
discarded statistical result.

## Artifacts and replay

Paths below are relative to the repository root:

| Artifact | Path |
| --- | --- |
| Development inputs | `Research/phase4-card-improvements-inputs-2026-10-02/` |
| Development candidate and results | `Research/phase4-card-improvements-2026-10-02/` |
| Preserved incomplete preparation | `Research/phase4-card-reused-qualification-inputs-2026-10-02/` |
| Completed later inputs | `Research/phase4-card-reused-qualification-inputs-2026-10-02-v2/` |
| Frozen later results | `Research/phase4-card-reused-qualification-2026-10-02/` |
| Independent audits and validation record | `Research/phase4-card-improvements-validation-2026-10-02/` |

The development seal SHA256 is
`3844dc6d8007751e5ae2dd6974670f23b893ecd3a1d1c3ea7bde395f6fae0d19`;
the later-result seal is
`7cfe9dd8b5acfde2217937412625e8ff2b2ec7312f1b420f664cf4248f0db82d`.
Sealed outputs retain protocols, code/dependencies, dated inputs, memberships,
source hashes, exclusions, parameters and predictions.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_card_candidates replay \
  --output Research/phase4-card-improvements-2026-10-02

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m Scripts.ops.phase4_card_qualification replay \
  --output Research/phase4-card-reused-qualification-2026-10-02

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  Scripts/tests/test_phase4_card_candidates.py \
  Scripts/tests/test_phase4_card_qualification.py \
  Scripts/tests/test_phase4_card_backtest.py \
  Scripts/tests/test_phase4_card_reconstruction.py \
  Scripts/tests/test_phase4_card_policy.py
```

This completes the bounded research, not card promotion or all remaining
Phase 4/5 work. Preserve the failed later gate without retuning against these
results. The current evidence supports modest potential gains over the simpler
reference, with insufficient evidence of the required gain over the fuller
control. Any future release still needs qualified scope, prospective validation,
the agreed joint target/profile/engine release and explicit approval.
