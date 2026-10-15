# Phase 4 qualification amendment — January–June 2025

Version `phase4-qualification.v2`, 29 September 2026. The owner authorized this
prospective amendment after the outcome-free April–June support check failed.
It supersedes only the reserved-period/input rules stated below. The original
protocols and sealed experiments remain intact.

## Inspection history and fixed periods

Before this amendment, Phase 4 inspected development through 2023 and only split
membership/eligibility metadata for 2025. No 2025 labels, features or history were
decoded. Calendar 2024 was previously consumed by Phase 3 confirmation.

January–June 2025 is now one qualification period, bounded by 00:00 UTC on
1 January and 1 July. There is no separate January–March calibration fit. Existing
2022 out-of-fold fitted calibration is preserved byte for byte. July 2025–June
2026 remains protected for the completed probability and selection system; later
prospective reserves also remain protected. More data changes sample support,
not the required performance or uncertainty thresholds.

## Explicit 2024 input bridge and fixed fitting

This amendment permits completed 2024 history as dated forecast context and as
training observations for the already selected opponent-strength method. This
explicitly replaces v1's exclusion of 2024 from base fitting. It does not make
2024 fresh evidence, score it again, or permit parameter/method/scope selection.

Fit ridge-10 goal strengths once at 1 January 2025, independently within each of
EPL, LaLiga, SerieA, Bundesliga and Ligue1, for provider season 2024. Use eligible
goals fixtures in provider seasons 2023 and 2024, whose assumed label availability
(kickoff plus three hours) is strictly before the fitting cutoff and whose kickoff
is on an earlier UTC date. Retain the established fit algorithm, equal fixture
weights, convergence checks and unknown/sparse-team fallback. Record the actual
season in the new bundle; never relabel an older bundle. No 2025 base refitting.

Forecast profiles/recent statistics/variances use the frozen engine's actual
current/prior-season rules on strictly earlier, available matches. During the
qualification period these observed inputs advance chronologically for both
control and candidate. Earlier qualification results may enter later deterministic
profiles, as they would operationally; they cannot change fitted coefficients,
calibrators, methods or thresholds. This is sequential forecasting with frozen
parameters. Same-day results and the forecast fixture's own result are excluded.
Availability remains reconstructed `assumed_final`, not original live snapshots.

Freeze five fitted strength bundles, the unchanged development corner sigmoid,
candidate recipe, sources, dependencies, input hashes and this protocol before
decoding any 2025 outcome. Keep permanent pre-2025 input and qualification opening
records. Access the mixed historical archive with a structural timestamp filter:
skip later outcome objects before JSON decoding, never decode the fixtures array.

## Finalists and scoring

Use the previously sealed finalist recipe
`3778878a8ce078700548925b93e79d6f13bff01fb59de3fe19c5da165a62fdac`.
Scope is the five named leagues and both teams with at least eight current-season
matches. Other scope and unknown/sparse goal teams retain the unchanged control.
Goals use ridge 10, rho −.10 and identity temperature; corner/SoT variances are
mu + .025*mu² and mu + .01*mu². All related goal markets share one score matrix.
The monotone sigmoid C=.01 for corner Over 9.5 is a separate binary diagnostic,
never a replacement for the count PMF or its other/Asian lines.

Submit exactly four primary claims: goals joint NLL, corner count NLL, SoT count
NLL, and corner Over 9.5 calibrated log loss against the raw dispersion candidate.
Report the last comparison against the frozen control as well. Report active
scope and complete served cohorts with explicit fallback/exclusion counts.

Keep minimum qualification support at 500 fixtures/10 observed ISO weeks pooled
and 100 fixtures/10 weeks for league conclusions. Other slices retain the existing
200-fixture/20-week development support rule; report unsupported slices explicitly.
At least .5% primary improvement, a wholly negative paired 95% interval, and no
supported slice/related-market loss regression above 2% are still required.
No post-result change of scope. Keep categorical/binary class counts visible.

Use 2,000 paired calendar-week cluster bootstrap draws, seed 20260928. For the
one-sided null test, subtract the observed fixture-weighted mean difference from
each difference, resample whole weeks and calculate the fraction of centered
bootstrap means at or below the observed difference, with the (k+1)/(2000+1)
finite-simulation correction. Apply Holm .05 across all four claims, assigning
p=1 to unavailable claims rather than reducing the family. Also retain the
ordinary paired percentile 95% interval. Week resampling keeps fixtures grouped;
derived markets are not treated as extra independent evidence.

Produce all forecasts before aggregate scoring, lock predictions and run one
scheduled evaluation. Persist scores, uncertainty, league/season/stage/missingness
slices, reliability, interval/numerical diagnostics and coverage. If evaluation
fails, preserve the access event and outputs; never rename the period untouched.
Numerical implementation repairs require an explicit incident record and must
not change the frozen statistical method after outcomes are inspected.

No active models, public delivery modes, databases or frozen archives change.
There is no API access, refresh, public test recommendation, ROI reconstruction or
promotion. Offline qualification is separate from shadow and live acceptance.
