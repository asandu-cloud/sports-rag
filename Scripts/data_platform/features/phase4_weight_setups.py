"""Bounded complete-setup research using archived numerical projection functions."""
import ast
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from functools import lru_cache
import math
import random
from types import FunctionType, SimpleNamespace

import numpy as np
from scipy.stats import poisson, nbinom

from . import phase4_candidates as cm
from . import phase4_card_candidates as cards
from . import phase4_candidate_adapter as adapter
from . import phase4_finalists as scope

VERSION = 'phase4-complete-weight-setups.v1'
SEED = 20261002
MARKETS = ('goals', 'corners', 'sot', 'cards')
ALPHAS = {'corners': .025, 'sot': .01}
FUNCTIONS = ('safe_float', '_stat_blend', '_venue_blend', '_trend_adjustment',
             '_blend_total_with_divergence', 'projected_corners', 'projected_sot',
             'projected_goals', 'projected_total_goals', 'projected_total_corners',
             'projected_total_sot', '_goal_market_team_projections')


def registry(market):
    if market not in MARKETS:
        raise ValueError('Unknown market')
    default = dict(own=.7 if market == 'cards' else .6,
                   recent=.15 if market in ('goals', 'cards') else .2,
                   venue=.5, window=6, half_life=None)
    axes = dict(own=[.5, .7, .85] if market == 'cards' else [.5, .6, .7],
                recent=[0., .1, .15, .2, .3], venue=[.25, .5, .75],
                window=[4, 6, 8, 12], half_life=[None, 30., 90., 180.])
    if market == 'cards':
        default.update(pool=0., referee='both', foul_blend=.15)
        axes.update(pool=[0., 8., 16.], referee=['both', 'modifier', 'anchor', 'residual', 'none'],
                    foul_blend=[0., .075, .15, .3])
    else:
        default.update(profile='control')
        axes.update(profile=['control', 'prior8', 'prior16', 'league8', 'league16', 'venue8', 'venue16'])
        if market == 'goals':
            default.update(xg=.5)
            axes.update(xg=[.25, .5, .75])
    result = [dict(id='reference', kind='reference', options=default, changes=0)]
    if market in ALPHAS:
        result.append(dict(id='original_control', kind='original_control', options=default, changes=0))
    if market == 'cards':
        result += [dict(id=k, kind=k, options=default, changes=0)
                   for k in ('fixed_reference', 'previous_card_candidate')]
    seen = {cm.canonical(default)}
    def add(options):
        key = cm.canonical(options)
        if key not in seen:
            seen.add(key)
            result.append({'id': 'setup-' + cm.digest(options)[:12], 'kind': 'setup', 'options': options,
                           'changes': sum(v != default[k] for k, v in options.items())})
    for key, values in axes.items():
        for value in values:
            add({**default, key: value})
    rng = random.Random(SEED + MARKETS.index(market))
    while len(result) < 60:
        add({k: rng.choice(v) for k, v in axes.items()})
    if len(result) != 60:
        raise ValueError('Candidate budget exceeded')
    return result


def load_engine(source, weights_source):
    env = cards.load_engine(source, weights_source)
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if {n.name for n in nodes} != set(FUNCTIONS):
        raise ValueError('Incomplete frozen projection functions')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    env['math'] = math
    exec(compile(ast.fix_missing_locations(tree), '<archived statistical projection functions>', 'exec'), env)
    if env['SCORING_WEIGHTS']['recency']['trend_weight'] != 0:
        raise ValueError('Protocol requires the unchanged zero trend coefficient')
    return env


def admitted(rows, cutoff):
    boundary = cm.utc(cutoff)
    if boundary >= cm.END:
        raise ValueError('Reserved feature cutoff')
    for row in rows:
        t = cm.utc(row['kickoff'])
        if t.date() >= boundary.date() or t + timedelta(hours=3) >= boundary:
            raise ValueError('Current/future/same-day result in input history')


def weighted(values, weights):
    return cm.weighted_summary(values, weights)


class Inputs:
    def __init__(self, history):
        self.history = {}
        self.leagues = defaultdict(list)
        for row in history:
            if cm.utc(row['kickoff']) >= cm.END:
                raise ValueError('Reserved history')
            if row['fixture_id'] in self.history:
                raise ValueError('Repeated history identity')
            self.history[row['fixture_id']] = row
            self.leagues[(row['competition'], row['season'])].append(row)

    @lru_cache(maxsize=4000)
    def league(self, competition, season, day):
        cutoff = cm.utc(day + 'T00:00:00Z')
        rows = [r for r in self.leagues[(competition, season)]
                if cm.utc(r['kickoff']).date() < cutoff.date()
                and cm.utc(r['kickoff']) + timedelta(hours=3) < cutoff]
        result = {}
        for stat in ('goals', 'corners', 'sot'):
            values = [r[s].get(stat) for r in rows for s in ('home', 'away')]
            known = [v for v in values if v is not None]
            result[stat] = {'mean': sum(known) / len(known) if len(known) >= 20 else None,
                            'n': len(known)}
        result['fixture_ids'] = [r['fixture_id'] for r in rows]
        return result

    def features(self, s):
        if not scope.scope(s)['active'] or s['fixture']['competition'] not in scope.LEAGUES:
            raise ValueError('Outside declared statistical comparison scope')
        if cm.utc(s['fixture']['kickoff']) >= cm.END or cm.utc(s['as_of']) > cm.utc(s['fixture']['kickoff']):
            raise ValueError('Reserved/post-kickoff forecast')
        league = s['fixture']['competition']
        profiles = {'control': deepcopy(s['profiles'])}
        recent = {'6:none': deepcopy(s['recent'])}
        support = {}
        for key in ('prior8', 'prior16', 'league8', 'league16', 'venue8', 'venue16'):
            profiles[key] = deepcopy(s['profiles'])
        for team, quality in s['profile_quality'].items():
            evidence = s['history_evidence'][team][league]
            current = [self.history[fid] for fid in evidence['current']['fixture_ids']]
            prior = [self.history[fid] for fid in evidence['prior']['fixture_ids']]
            admitted(current + prior, s['as_of'])
            for rs, season in ((current, evidence['current_rank']), (prior, evidence['prior_rank'])):
                if any(r['competition'] != league or r['season'] != season
                       or int(team) not in (r['home_team_id'], r['away_team_id']) for r in rs):
                    raise ValueError('Historical team/competition/season mismatch')
            component = s['profile_components'][team][league]
            lg = self.league(league, evidence['current_rank'], cm.utc(s['as_of']).date().isoformat())
            support[team] = {'current_ids': [r['fixture_id'] for r in current],
                             'prior_ids': [r['fixture_id'] for r in prior], 'league': lg, 'fields': {}, 'recent': {}}
            for key in profiles:
                if key == 'control':
                    continue
                strength = float(key.removeprefix('prior').removeprefix('league').removeprefix('venue'))
                for field, (stat, opponent, venue) in adapter.RATE_FIELDS.items():
                    if key.startswith('prior'):
                        a, b = (component['current'] or {}).get(field), (component['prior'] or {}).get(field)
                        if b is not None:
                            w = strength / (len(current) + strength)
                            profiles[key][team][field] = (1-w)*a + w*b if a is not None else b
                    elif key.startswith('league') and venue is None and stat in ('goals', 'corners', 'sot'):
                        stats = adapter.summary(current, team, field)
                        profiles[key][team][field] = cm.pool(s['profiles'][team].get(field), stats['ess'], lg[stat]['mean'], strength)
                        support[team]['fields'][field] = stats
                    elif key.startswith('venue') and venue is not None:
                        overall = {'goals': 'goals_for_pm', 'corners': 'corners_pm', 'sot': 'sot_for_pm', 'xg': 'expected_goals'}[stat]
                        if opponent:
                            overall = {'corners': 'corners_against_pm', 'sot': 'sot_against_pm'}[stat]
                        stats = adapter.summary(current, team, field)
                        profiles[key][team][field] = cm.pool(s['profiles'][team].get(field), stats['ess'], s['profiles'][team].get(overall), strength)
                        support[team]['fields'][field] = stats
            for window in (4, 6, 8, 12):
                for half in (None, 30., 90., 180.):
                    key = str(window) + ':' + ('none' if half is None else str(half))
                    rr = current[:window]
                    ww = [2 ** (-(cm.utc(s['as_of'])-cm.utc(r['kickoff'])).total_seconds()/86400/half)
                          if half else .85 ** i for i, r in enumerate(rr)]
                    evidence_recent = {'fixture_ids': [r['fixture_id'] for r in rr], 'weights': ww, 'fields': {}}
                    support[team]['recent'][key] = evidence_recent
                    if key == '6:none':
                        continue
                    recent.setdefault(key, {})[team] = deepcopy(s['recent'][team])
                    recent[key][team]['n'] = len(rr)
                    for field, out in [('expected_goals', 'xg_for_avg'), ('corners_pm', 'corners_for_avg'),
                                       ('corners_against_pm', 'corners_against_avg'), ('sot_for_pm', 'sot_for_avg'), ('sot_against_pm', 'sot_against_avg')]:
                        values = [v for _, v in adapter.observations(rr, team, field)]
                        stats = weighted(values, ww)
                        recent[key][team][out] = stats['mean']
                        evidence_recent['fields'][out] = stats
        result = {'snapshot_id': s['snapshot_id'], 'fixture': s['fixture'], 'as_of': s['as_of'],
                  'profiles': profiles, 'recent': recent, 'support': support,
                  'league_context': s['league_context'], 'knockout_context': s['knockout_context']}
        result['id'] = cm.digest(result)
        return result


def stat_mean(f, options, market, engine):
    fixture = f['fixture']; h, a = str(fixture['home_team_id']), str(fixture['away_team_id'])
    profiles = f['profiles'][options['profile']]
    key = str(options['window']) + ':' + ('none' if options['half_life'] is None else str(options['half_life']))
    recent = f['recent'][key]
    original = engine['SCORING_WEIGHTS']
    w = dict(original); w['projection'] = dict(original['projection']); w['projection_sot'] = dict(original['projection_sot'])
    w['projection'].update({market+'_venue_blend': options['venue'], 'blend_'+market+'_recent': options['recent'],
                            'blend_'+market+'_season': 1-options['recent']})
    if market == 'sot':
        w['projection_sot'].update(own=options['own'], opp=1-options['own'])
    else:
        w['projection'].update({market+'_own': options['own'], market+'_opp': 1-options['own']})
    if market == 'goals':
        w['projection']['xg_blend'] = options['xg']
    engine.update(SCORING_WEIGHTS=w, _profile_as_of=lambda t, *args: profiles[t],
                  _recent_stats=lambda t, *args, **kwargs: recent[t])
    contexts = {'league_ctx': SimpleNamespace(**f['league_context']), 'knockout_ctx': SimpleNamespace(**f['knockout_context']), 'fixture_date': f['as_of']}
    try:
        if market == 'goals':
            result = engine['_goal_market_team_projections'](h, a, fixture['competition'], **contexts)[:2]
        elif options['own'] == .6:
            result = (engine['projected_total_'+market](h, a, fixture['competition'], **contexts)[0],)
        else:
            function = adapter.own_weight_total(SimpleNamespace(**engine), profiles, recent, h, a, market, options['own'])
            result = (function(h, a, fixture['competition'], **contexts)[0],)
    finally:
        engine['SCORING_WEIGHTS'] = original
    if any(v is None or not math.isfinite(v) or v <= 0 for v in result):
        raise ValueError('Invalid mean; candidate cannot drop the fixture')
    return result


def card_inputs(f, options, cached):
    pool = options['pool']
    if pool not in cached:
        cached[pool] = cards.profile_inputs(f, {'pool': pool})
    profiles, _, effective = cached[pool]
    recent = {}
    for side in ('home', 'away'):
        rr = list(reversed(f['inputs']['teams'][side]))[:options['window']]
        half = options['half_life']
        ww = [2 ** (-(cm.utc(f['as_of'])-cards.cards.utc(r['kickoff'])).total_seconds()/86400/half)
              if half else .85 ** i for i, r in enumerate(rr)]
        recent[side] = {'n': len(rr), **{out: cards.weighted([r.get(field) for r in rr], ww)
                         for field, out in [('own', 'cards_avg'), ('induced', 'cards_induced_avg'), ('fouls', 'fouls_avg')]}}
    return profiles, recent, effective


def card_mean(f, options, engine, cached):
    function = FunctionType(cards.full_mean.__code__, {**cards.__dict__, 'profile_inputs': lambda row, spec: card_inputs(row, options, cached)},
                            argdefs=cards.full_mean.__defaults__)
    function.__kwdefaults__ = cards.full_mean.__kwdefaults__
    original = engine['SCORING_WEIGHTS']
    w = dict(original); w['projection_cards'] = dict(original['projection_cards'])
    w['projection_cards']['foul_card_blend'] = options['foul_blend']
    engine['SCORING_WEIGHTS'] = w
    try:
        return function(f, {k: options[k] for k in ('own', 'recent', 'venue', 'referee')}, engine)['mean']
    finally:
        engine['SCORING_WEIGHTS'] = original


def primary_loss(means, target, alpha=None, team_targets=None):
    if team_targets is not None:
        h, a = means[:, 0], means[:, 1]
        yh, ya = team_targets[:, 0], team_targets[:, 1]
        rho = -.1
        tau = np.ones(len(target))
        tau[(yh == 0) & (ya == 0)] = (1-h*a*rho)[(yh == 0) & (ya == 0)]
        tau[(yh == 0) & (ya == 1)] = (1+h*rho)[(yh == 0) & (ya == 1)]
        tau[(yh == 1) & (ya == 0)] = (1+a*rho)[(yh == 1) & (ya == 0)]
        tau[(yh == 1) & (ya == 1)] = 1-rho
        if np.any(tau <= 0):
            raise ValueError('Invalid goal dependence')
        return -poisson.logpmf(yh, h)-poisson.logpmf(ya, a)-np.log(tau)
    mu = means[:, 0]
    alpha = np.broadcast_to(alpha, mu.shape)
    loss = -poisson.logpmf(target, mu)
    positive = alpha > 0
    loss[positive] = -nbinom(1/alpha[positive], 1/(1+alpha[positive]*mu[positive])).logpmf(target[positive])
    return loss


def choose(specs, losses, metadata):
    if any(cm.utc(r['kickoff']).year != 2022 or cm.utc(r['kickoff'])+timedelta(hours=3) >= cm.utc('2023-01-01T00:00:00Z') for r in metadata):
        raise ValueError('Selection requires available 2022 outcomes only')
    weeks = len({r['week'] for r in metadata})
    if len(metadata) < 500 or weeks < 26:
        raise ValueError('Insufficient tuning support')
    means = losses.mean(axis=0)
    valid = [i for i, s in enumerate(specs) if s['kind'] in ('reference', 'setup') and np.all(np.isfinite(losses[:, i]))]
    best = min(means[i] for i in valid)
    selected = min((i for i in valid if means[i] <= best+1e-8), key=lambda i: (specs[i]['kind'] != 'reference', specs[i]['changes'], specs[i]['id']))
    return {'selected': specs[selected]['id'], 'index': selected, 'support': {'fixtures': len(metadata), 'weeks': weeks},
            'ranking': [{'id': specs[i]['id'], 'nll': float(means[i])} for i in sorted(valid, key=lambda i: (means[i], specs[i]['id']))],
            'implementation_decision': 'await_owner_review'}


def paired(rows, delta):
    weeks = sorted({r['week'] for r in rows}); lookup = {w: i for i, w in enumerate(weeks)}
    sums = np.zeros(len(weeks)); n = np.zeros(len(weeks))
    for row, value in zip(rows, delta):
        i = lookup[row['week']]; sums[i] += value; n[i] += 1
    draws = np.random.default_rng(SEED).integers(0, len(weeks), (2000, len(weeks)))
    estimates = sums[draws].sum(axis=1)/n[draws].sum(axis=1)
    mean = float(np.mean(delta))
    p = (1+int(np.sum(estimates-mean <= mean)))/2001 if mean < 0 else 1.
    return {'delta': mean, 'interval95': np.quantile(estimates, [.025, .975]).tolist(), 'p': p}


def holm(values):
    ordered = sorted(values, key=values.get); result = {}; previous = 0.
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., values[key]*(len(ordered)-i)))
        result[key] = previous
    return result
