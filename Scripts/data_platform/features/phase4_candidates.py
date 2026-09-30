"""Research-only candidate contracts, fitted team strengths and probability tools.

No production activation, live IO, automatic parameter search or reserve reader.
The frozen engine supplies the control calculations through the separate adapter.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from Scripts.data_platform.features.count_calibration import ALPHAS, distribution

VERSION = 'phase4-step2-candidates.v2'
END = datetime(2024, 1, 1, tzinfo=timezone.utc)
MARKETS = ('goals', 'corners', 'sot')


def utc(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return t.astimezone(timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def registry():
    result = []
    def add(family, parameter=None, markets=MARKETS):
        item = dict(family=family, parameter=parameter, markets=list(markets), version=VERSION)
        item['id'] = family+(':'+str(parameter) if parameter is not None else '')
        result.append(item)
    add('control')
    add('gradual_prior', 16.)
    for family in ('league_pool', 'venue_pool'):
        for k in (4., 8., 16.): add(family, k)
    add('xg_fallback', markets=('goals',))
    for w in (.25, .5, .75): add('defensive_xg', w, ('goals',))
    for market in MARKETS:
        for ridge in (1., 10., 100.): add('strength_'+market, ridge, (market,))
    for half in (30., 90., 180.): add('calendar_recency', half)
    for window in (4, 8, 12): add('recent_window', window)
    for w in (.5, .7): add('own_weight', w)
    for w in (.25, .75): add('venue_weight', w)
    for w in (.25, .75): add('xg_weight', w, ('goals',))
    for w in (.1, .25, .4): add('recent_weight', w)
    for market in ('corners', 'sot'):
        for alpha in ALPHAS: add('dispersion_'+market, alpha, (market,))
    for rho in (-.15, -.1, -.05, 0., .05): add('goal_rho', rho, ('goals',))
    for alpha in ALPHAS: add('goal_nb', alpha, ('goals',))
    for strength in (.25, .5, 1.): add('lineup', strength, ('goals', 'sot'))
    return result


def validate_spec(spec):
    if spec not in registry():
        raise ValueError('Candidate is outside the declared registry')


def finite_nonnegative(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def available_history(history, cutoff, *, competition=None, seasons=None, eligible_ids=None, end=END):
    """Filter by identity/time before inspecting target values; no future labels."""
    cutoff = utc(cutoff)
    if cutoff > end:
        raise ValueError('Reserved cutoff refused')
    selected, seen = [], set()
    for row in history:
        fid = row['fixture_id']
        if fid in seen:
            raise ValueError('Duplicate fixture history')
        seen.add(fid)
        kickoff = utc(row['kickoff'])
        if kickoff.date() >= cutoff.date() or kickoff+timedelta(hours=3) >= cutoff:
            continue
        if competition is not None and row['competition'] != competition:
            continue
        if seasons is not None and row['season'] not in seasons:
            continue
        if eligible_ids is not None and fid not in eligible_ids:
            continue
        if row['status'] != 'FT':
            raise ValueError('Non-regulation history')
        selected.append(row)
    return sorted(selected, key=lambda r: (utc(r['kickoff']), r['fixture_id']))


def weighted_summary(values, weights=None):
    """Missing observations have no weight; zero remains an observed value."""
    if weights is None: weights = [1.]*len(values)
    if len(values) != len(weights) or any(not finite_nonnegative(w) for w in weights):
        raise ValueError('Invalid observation weights')
    pairs = [(v,w) for v,w in zip(values,weights) if v is not None and w > 0]
    if any(not finite_nonnegative(v) for v,_ in pairs):
        raise ValueError('Invalid production value')
    if not pairs: return {'mean': None, 'n': 0, 'ess': 0.}
    total = sum(w for _,w in pairs)
    return {'mean': sum(v*w for v,w in pairs)/total, 'n':len(pairs),
            'ess':total**2/sum(w*w for _,w in pairs)}


def mixed_support(current, prior, prior_weight):
    if current['mean'] is None: return prior['ess'] if prior_weight > 0 else 0.
    if prior['mean'] is None or prior_weight == 0: return current['ess']
    if prior_weight == 1: return prior['ess']
    return 1/((1-prior_weight)**2/current['ess']+prior_weight**2/prior['ess'])


def pool(value, support, prior, strength):
    if strength <= 0 or support < 0: raise ValueError('Invalid pooling parameters')
    if prior is None: return value
    if value is None: return prior
    return (support*value+strength*prior)/(support+strength)


def fit_strength(history, *, eligible_ids, competition, market, cutoff, season,
                 ridge=10., half_life_days=None, maxiter=500, end=END):
    """Joint Poisson attack/defence fit; two observations per grouped fixture.

    Team effects are centred during fitting for identifiability. Ridge shrinks
    them toward zero within this competition. No league identity is pooled away.
    """
    if market not in MARKETS or ridge not in (1.,10.,100.):
        raise ValueError('Undeclared strength fit')
    if half_life_days is not None and half_life_days not in (30.,90.,180.):
        raise ValueError('Undeclared time weighting')
    rows = available_history(history, cutoff, competition=competition,
                             seasons={season,season-1}, eligible_ids=set(eligible_ids), end=end)
    rows = [r for r in rows if all(r[s].get(market) is not None for s in ('home','away'))]
    if len(rows) < 60: raise ValueError('Insufficient strength fitting cohort')
    y = np.array([r[s][market] for r in rows for s in ('home','away')], dtype=float)
    if np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError('Strength targets must be regulation counts')
    teams = sorted({r[s+'_team_id'] for r in rows for s in ('home','away')})
    mapping = {t:i for i,t in enumerate(teams)}; n = len(teams)
    rr,cc,vv = [],[],[]
    for i,r in enumerate(rows):
        for j,side in enumerate(('home','away')):
            opponent = 'away' if side == 'home' else 'home'
            for col,value in [(0,1.),(1,float(side=='home')),
                              (2+mapping[r[side+'_team_id']],1.),
                              (2+n+mapping[r[opponent+'_team_id']],1.)]:
                rr.append(2*i+j);cc.append(col);vv.append(value)
    x = csr_matrix((vv,(rr,cc)),shape=(len(y),2+2*n))
    weights = np.array([1. if half_life_days is None else
        2**(-((utc(cutoff)-utc(r['kickoff'])).total_seconds()/86400)/half_life_days) for r in rows])
    obs_weights = np.repeat(weights,2); normalizer = obs_weights.sum()
    def centred(theta):
        result = theta.copy()
        result[2:2+n] -= result[2:2+n].mean();result[2+n:] -= result[2+n:].mean()
        return result
    def objective(theta):
        beta = centred(theta); eta = x@beta; mu = np.exp(eta)
        penalty = ridge/normalizer
        loss = float(np.dot(obs_weights,mu-y*eta)/normalizer+.5*penalty*np.dot(beta[2:],beta[2:]))
        gradient = np.asarray(x.T@(obs_weights*(mu-y))/normalizer).ravel()
        gradient[2:] += penalty*beta[2:]
        gradient[2:2+n] -= gradient[2:2+n].mean();gradient[2+n:] -= gradient[2+n:].mean()
        return loss,gradient
    initial = np.zeros(2+2*n);initial[0] = math.log(max(float(np.average(y,weights=obs_weights)),1e-4))
    solution = minimize(objective,initial,jac=True,method='L-BFGS-B',
                        bounds=[(-10.,5.),(-2.,2.)]+[(-4.,4.)]*(2*n),
                        options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-7})
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise ValueError('Strength fitting failed to converge: '+str(solution.message))
    beta = centred(solution.x)
    counts = Counter(r[s+'_team_id'] for r in rows for s in ('home','away'))
    training = [{k:r[k] for k in ('fixture_id','competition','season','kickoff','home_team_id','away_team_id')}
                | {'target':[r['home'][market],r['away'][market]]} for r in rows]
    bundle = {'version':VERSION,'family':'opponent_strength','competition':competition,'market':market,
              'cutoff':cutoff,'season':season,'ridge':ridge,'half_life_days':half_life_days,
              'fixture_ids':[r['fixture_id'] for r in rows], 'training_sha256':digest(training),
              'fixture_count':len(rows),'effective_fixture_count':float(weights.sum()**2/np.dot(weights,weights)),
              'intercept':float(beta[0]),'home_effect':float(beta[1]),
              'teams':{str(t):{'attack':float(beta[2+i]),'defence':float(beta[2+n+i]),'n':counts[t]} for i,t in enumerate(teams)},
              'converged':True,'iterations':int(solution.nit),'publication_enabled':False}
    bundle['id'] = digest(bundle)
    return bundle


def predict_strength(bundle, fixture, as_of, *, end=END):
    if bundle.get('id') != digest({k:v for k,v in bundle.items() if k!='id'}) or not bundle.get('converged'):
        raise ValueError('Invalid or nonconverged strength bundle')
    if utc(bundle['cutoff']) > utc(as_of) or utc(as_of) >= end:
        raise ValueError('Strength bundle is from the future/reserved period')
    if fixture['competition'] != bundle['competition']:
        raise ValueError('Strength bundle competition mismatch')
    if fixture['season'] != bundle['season']:
        raise ValueError('Strength bundle season mismatch; explicit seasonal refit required')
    home = bundle['teams'].get(str(fixture['home_team_id']))
    away = bundle['teams'].get(str(fixture['away_team_id']))
    if not home or not away or min(home['n'],away['n']) < 3:
        return {'status':'control_fallback','reason':'unknown_or_sparse_team','means':None,'bundle_id':bundle['id']}
    means = [math.exp(bundle['intercept']+bundle['home_effect']+home['attack']+away['defence']),
             math.exp(bundle['intercept']+away['attack']+home['defence'])]
    return {'status':'available','means':means,'bundle_id':bundle['id']}


def count_pmf(mean, alpha, *, tolerance=1e-12, maximum=2000):
    if not finite_nonnegative(mean) or not finite_nonnegative(alpha):
        raise ValueError('Invalid distribution parameters')
    if not 0 < tolerance <= 1e-10: raise ValueError('Invalid tail tolerance')
    if mean == 0: return {'pmf':[1.],'omitted_mass':0.,'mean':mean,'alpha':alpha}
    dist = distribution(mean, alpha)  # Reuse the existing fixed-mean count research.
    upper = float(dist.isf(tolerance))
    if not math.isfinite(upper) or upper > maximum: raise ValueError('Count tail exceeds resource bound')
    upper = int(max(upper,1));pmf = np.asarray(dist.pmf(np.arange(upper+1)))
    tail = float(dist.sf(upper))
    if tail > tolerance*1.01 or np.any(pmf < 0) or not np.all(np.isfinite(pmf)):
        raise ValueError('Invalid count PMF')
    return {'pmf':pmf.tolist(),'omitted_mass':tail,'mean':mean,'alpha':alpha}


def goal_matrix(home, away, *, alpha=None, rho=-.1, frozen_probability):
    if alpha is None:
        return frozen_probability.dixon_coles_scoreline_matrix(home,away,rho=rho)
    if rho != 0: raise ValueError('Dixon-Coles correction is not defined for NB candidates')
    h,a = count_pmf(home,alpha),count_pmf(away,alpha)
    if len(h['pmf'])*len(a['pmf']) > 250000:
        raise ValueError('Joint score grid exceeds resource bound')
    matrix = np.outer(h['pmf'],a['pmf']);mass=float(matrix.sum())
    if 1-mass > 1e-10 or np.any(matrix < 0): raise ValueError('Invalid joint goal mass')
    return (matrix/mass).tolist()


def fit_probability_parameter(rows, *, cutoff, family, market, frozen_probability):
    """Declared-grid fit on preceding forecasts only; complete common membership.

    Each row supplies its historical forecast means and subsequently available
    regulation label. No row may disappear because one parameter cannot price it.
    """
    boundary = utc(cutoff)
    if boundary > END or family not in ('dispersion','goal_rho','goal_nb'):
        raise ValueError('Unsupported probability fit')
    if market not in (('corners','sot') if family=='dispersion' else ('goals',)):
        raise ValueError('Probability family/market mismatch')
    if len(rows) < 500: raise ValueError('Insufficient probability fitting cohort')
    seen=set();weeks=set()
    for r in rows:
        if r.get('market')!=market or r.get('period')!='regulation_time':
            raise ValueError('Mixed/unqualified probability target identity')
        if r['fixture_id'] in seen: raise ValueError('Duplicate fitting fixture')
        seen.add(r['fixture_id'])
        kickoff=utc(r['kickoff']);available=utc(r['label_available_at'])
        if utc(r['as_of']) > kickoff or available < kickoff+timedelta(hours=3) or available >= boundary:
            raise ValueError('Label unavailable at fit cutoff')
        weeks.add(kickoff.isocalendar()[:2])
        targets = [r['target']] if family=='dispersion' else r['team_target']
        if any(not finite_nonnegative(y) or int(y)!=y for y in targets): raise ValueError('Invalid fitting label')
    if len(weeks) < 26: raise ValueError('Insufficient fitting weeks')
    grid = (-.15,-.1,-.05,0.,.05) if family=='goal_rho' else ALPHAS
    scores=[]
    for parameter in grid:
        loss=0.;reason=None
        try:
            for r in rows:
                if family=='dispersion':
                    if r['mean']==0: lp=0. if r['target']==0 else -math.inf
                    else: lp=float(distribution(r['mean'],parameter).logpmf(r['target']))
                elif family=='goal_nb':
                    h,a=r['goal_means']; yh,ya=r['team_target']
                    goal_matrix(h,a,alpha=parameter,rho=0,frozen_probability=frozen_probability)
                    lp=sum((0. if y==0 else -math.inf) if mu==0 else float(distribution(mu,parameter).logpmf(y))
                           for mu,y in ((h,yh),(a,ya)))
                else:
                    h,a=r['goal_means']; yh,ya=r['team_target']
                    probability=frozen_probability.dixon_coles_scoreline_prob(int(yh),int(ya),h,a,rho=parameter)
                    lp=math.log(probability) if probability>0 else -math.inf
                if not math.isfinite(lp): raise ValueError('zero_or_invalid_label_probability')
                loss-=lp
        except ValueError as exc: reason=str(exc)
        scores.append({'parameter':parameter,'nll':loss/len(rows) if reason is None else None,'invalid_reason':reason})
    valid=[r for r in scores if r['nll'] is not None]
    if not valid: raise ValueError('All probability candidates invalid on the common cohort')
    best=min(r['nll'] for r in valid)
    tied=[r for r in valid if r['nll'] <= best+1e-8]
    selected=min(tied,key=lambda r:(0 if family=='goal_rho' and r['parameter']==-.1 else 1,
                                    abs(r['parameter']),r['parameter']))
    result={'version':VERSION,'family':family,'market':market,'cutoff':cutoff,'parameter':selected['parameter'],
            'grid':scores,'fixture_ids':sorted(seen),'training_sha256':digest(rows),'n':len(rows),'weeks':len(weeks)}
    return {**result,'id':digest(result)}


def validate_probability_bundle(bundle, spec, as_of):
    validate_spec(spec)
    family=spec['family']
    expected='dispersion' if family.startswith('dispersion_') else family
    if bundle.get('id')!=digest({k:v for k,v in bundle.items() if k!='id'}):
        raise ValueError('Probability bundle identity mismatch')
    if bundle.get('version')!=VERSION or bundle.get('family')!=expected or bundle.get('market') not in spec['markets']:
        raise ValueError('Probability bundle candidate mismatch')
    if bundle.get('parameter')!=spec['parameter'] or utc(bundle['cutoff'])>utc(as_of) or utc(as_of)>=END:
        raise ValueError('Wrong or future probability parameter fit')


def lineup_delta(evidence, fixture, as_of, strength):
    """Apply supplied dated contribution differences; never infer unavailable XI."""
    if evidence is None: return {'status':'control_fallback','reason':'dated_lineup_evidence_unavailable'}
    if strength not in (.25,.5,1.): raise ValueError('Undeclared lineup strength')
    if evidence.get('fixture_id')!=fixture['fixture_id'] or evidence.get('minutes_kind')!='expected':
        raise ValueError('Invalid lineup identity/minutes kind')
    if evidence.get('stage') not in ('expected','confirmed') or not evidence.get('source_id'):
        raise ValueError('Unverified lineup stage/source')
    cutoff=utc(as_of)
    for key in ('observed_at','rates_cutoff','coefficient_cutoff'):
        if utc(evidence[key]) >= cutoff: raise ValueError('Post-cutoff lineup evidence')
    if not evidence.get('coefficient_id'): raise ValueError('Missing pre-fitted lineup coefficient identity')
    delta={};seen_global=set()
    for side in ('home','away'):
        team=evidence[side]
        if team['team_id']!=fixture[side+'_team_id']: raise ValueError('Lineup team mismatch')
        for field in ('reference_minutes','forecast_minutes'):
            minutes=[p[field] for p in team['players']]
            if any(not finite_nonnegative(x) or x>90 for x in minutes) or not math.isclose(sum(minutes),990.,abs_tol=1e-6):
                raise ValueError('Invalid regulation expected-minute allocation')
        result={'goals':0.,'sot':0.}
        for player in team['players']:
            pid=player['player_id']
            if not isinstance(pid,int) or pid<=0 or pid in seen_global: raise ValueError('Duplicate/invalid player identity')
            seen_global.add(pid)
            for market in result:
                rate=player[market+'_per90']
                coefficient=evidence['coefficients'][market]
                if not finite_nonnegative(rate) or not finite_nonnegative(coefficient): raise ValueError('Invalid lineup contribution')
                result[market]+=strength*coefficient*(player['forecast_minutes']-player['reference_minutes'])*rate/90
        delta[side]=result
    return {'status':'available','delta':delta,'evidence_id':digest(evidence),'coefficient_id':evidence['coefficient_id']}
