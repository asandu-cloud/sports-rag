"""Fixed-mean count dispersion development experiment; no live model imports."""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
from scipy.stats import nbinom, poisson

VERSION = 'phase4-count-dispersion-development.v1'
ALPHAS = (0., .01, .025, .05, .1, .2, .4, .8, 1.6)
LINES = {'goals': 2.5, 'corners': 9.5, 'sot': 8.5}
SEED = 20260928


def partition(row):
    kickoff = datetime.fromisoformat(row['kickoff'].replace('Z', '+00:00'))
    if kickoff.tzinfo is None:
        raise ValueError('Explicit forecast timezone required')
    available = kickoff + timedelta(hours=3)
    if kickoff.year == available.year == 2022:
        return 'fit'
    if kickoff.year == available.year == 2023:
        return 'evaluation'
    raise ValueError('Outside permitted fitting/evaluation periods')


def distribution(mean, alpha):
    mean = np.asarray(mean, dtype=float)
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid count distribution parameter')
    return poisson(mean) if alpha == 0 else nbinom(1 / alpha, 1 / (1 + alpha * mean))


def arrays(rows):
    means = np.asarray([r['reference'] for r in rows], dtype=float)
    y = np.asarray([r['target'] for r in rows], dtype=float)
    if not len(rows) or not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError('Missing or invalid count targets')
    return means, y


def support(rows, *, fitting=False):
    weeks = len({r['week'] for r in rows})
    return {'n': len(rows), 'weeks': weeks,
            'sufficient': len(rows) >= (500 if fitting else 200) and weeks >= (26 if fitting else 20)}


def fit(rows):
    if any(partition(r) != 'fit' for r in rows):
        raise ValueError('Fitting may only use 2022 development outcomes')
    if not support(rows, fitting=True)['sufficient']:
        raise ValueError('Insufficient fitting cohort')
    mean, y = arrays(rows)
    losses = [(float(-distribution(mean, alpha).logpmf(y).mean()), alpha) for alpha in ALPHAS]
    loss, alpha = min(losses)
    return {'version': VERSION, 'alpha': alpha, 'variance': 'mu + alpha * mu^2',
            'fitting_support': support(rows, fitting=True), 'fitting_nll': loss,
            'grid': [{'alpha': a, 'nll': v} for v, a in losses]}


def predict(rows, alpha, line):
    mean, y = arrays(rows)
    d = distribution(mean, alpha)
    return {'nll': -d.logpmf(y), 'over_probability': d.sf(np.floor(line)),
            'lower80': d.ppf(.1), 'upper80': d.ppf(.9), 'actual_over': y > line}


def scores(rows, values):
    p = np.clip(values['over_probability'], 1e-15, 1 - 1e-15)
    y = values['actual_over'].astype(float)
    actual = np.array([r['target'] for r in rows])
    reliability = []
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        reliability.append({'lower': i / 10, 'n': int(mask.sum()),
                            'mean_probability': float(p[mask].mean()) if mask.any() else None,
                            'frequency': float(y[mask].mean()) if mask.any() else None})
    return {**support(rows), 'nll': float(values['nll'].mean()),
            'brier': float(np.mean((p - y)**2)),
            'binary_log_loss': float(-np.mean(y * np.log(p) + (1-y) * np.log1p(-p))),
            'interval80_coverage': float(np.mean((actual >= values['lower80']) & (actual <= values['upper80']))),
            'interval80_width': float(np.mean(values['upper80'] - values['lower80'])),
            'reliability': reliability}


def week_interval(rows, difference):
    # Resample whole calendar weeks, preserving paired fixture predictions and
    # within-week dependence; report limits, not independence-based standard errors.
    weeks = sorted({r['week'] for r in rows})
    sums = np.array([sum(float(d) for r, d in zip(rows, difference) if r['week'] == w) for w in weeks])
    counts = np.array([sum(r['week'] == w for r in rows) for w in weeks])
    draws = np.random.default_rng(SEED).integers(0, len(weeks), size=(1000, len(weeks)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def evaluate(rows, fitted, market):
    if any(partition(r) != 'evaluation' for r in rows):
        raise ValueError('Evaluation may only use 2023 development outcomes')
    reference = predict(rows, 0., LINES[market])
    candidate = predict(rows, fitted['alpha'], LINES[market])
    interval = week_interval(rows, candidate['nll'] - reference['nll'])
    slices = {}
    for key in ('league', 'season', 'stage', 'missing_fallback'):
        slices[key] = {}
        for value in sorted({str(r[key]) for r in rows}):
            indices = np.array([i for i, r in enumerate(rows) if str(r[key]) == value])
            cohort = [rows[i] for i in indices]
            baseline = scores(cohort, {k: v[indices] for k, v in reference.items()})
            adjusted = scores(cohort, {k: v[indices] for k, v in candidate.items()})
            slices[key][value] = {'reference': baseline, 'candidate': adjusted,
                                 'nll_change_fraction': adjusted['nll'] / baseline['nll'] - 1}
    regressions = [league for league, item in slices['league'].items()
                   if item['candidate']['sufficient'] and item['nll_change_fraction'] > .02]
    report = {'reference_poisson': scores(rows, reference), 'fitted_negative_binomial': scores(rows, candidate),
              'nll_delta_week_bootstrap95': interval, 'supported_league_regressions': regressions,
              'encouraging_development_result': support(rows)['sufficient'] and interval[1] < 0 and not regressions,
              'slices': slices}
    predictions = [{**r, 'alpha': fitted['alpha'], 'line': LINES[market],
                    'reference_probability': float(reference['over_probability'][i]),
                    'candidate_probability': float(candidate['over_probability'][i]),
                    'reference_nll': float(reference['nll'][i]), 'candidate_nll': float(candidate['nll'][i]),
                    'candidate_interval80': [int(candidate['lower80'][i]), int(candidate['upper80'][i])]}
                   for i, r in enumerate(rows)]
    return report, predictions
