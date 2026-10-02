"""Bounded, chronological card research. No production imports or live I/O."""
from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
import math
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import nbinom, poisson

from . import phase4_card_reconstruction as fixed
from . import phase4_cards as cards
from .count_calibration import ALPHAS
from .phase4_backtest import profile as asian_profile

VERSION = 'phase4-card-improvements.v1'
SEED = 20261002
PROJECTION_FUNCTIONS = ('_stat_blend', '_venue_blend', '_trend_adjustment',
                        '_blend_total_with_divergence', '_extract_card_risk_context',
                        'projected_cards', 'projected_total_cards_detail')


def load_engine(source, weights_source):
    """Compile unchanged archived numerical functions, with explicit input seams."""
    weights = None
    for node in ast.parse(weights_source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCORING_WEIGHTS' for t in node.targets):
            weights = ast.literal_eval(node.value)
    if weights is None:
        raise ValueError('Missing frozen weights')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in PROJECTION_FUNCTIONS]
    if {n.name for n in nodes} != set(PROJECTION_FUNCTIONS):
        raise ValueError('Missing archived projection functions')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    env = {'SCORING_WEIGHTS': weights, 'safe_float': lambda v: None if v is None else float(v),
           '_resolve_league_context': lambda *a, **k: k.get('league_ctx'),
           '_ml_blend_weight': lambda _: 0., '_HAS_LINEUP_RISK': False}
    exec(compile(ast.fix_missing_locations(tree), '<archived card calculations>', 'exec'), env)
    return env


def registry():
    specs = [{'id': 'fuller_control'}]
    for field, values in (('own', (.5, .85)), ('recent', (0., .3)), ('venue', (0., .75)),
                          ('pool', (8., 16.)), ('half_life', (90., 180.))):
        specs += [{'id': f'{field}:{value}', field: value} for value in values]
    specs += [{'id': 'referee:' + mode, 'referee': mode}
              for mode in ('none', 'modifier', 'anchor', 'residual')]
    specs.append({'id': 'no_fouls', 'fouls': False})
    return specs


def weighted(values, weights):
    valid = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not valid:
        return None
    return float(np.average([v for v, _ in valid], weights=[w for _, w in valid]))


def profile_inputs(f, spec):
    profiles, recent, effective = {}, {}, {}
    k = spec.get('pool', 0.)
    for side in ('home', 'away'):
        history = f['inputs']['teams'][side]
        current = [r for r in history if r['season'] == f['season']]
        previous = [r for r in history if r['season'] != f['season']]
        if not current and not k:
            return None
        def ws(rows):
            half = spec.get('half_life')
            return [2 ** (-(cards.utc(f['as_of']) - cards.utc(r['kickoff'])).total_seconds() / 86400 / half)
                    if half else 1. for r in rows]
        weights = ws(current)
        profile = {}
        for field, output in (('own', 'cards_per_90_team'), ('induced', 'opp_cards_induced_pm')):
            own = weighted([r[field] for r in current], weights)
            if k:
                prior = (sum(r[field] for r in previous) + k * f['league_team_mean']) / (len(previous) + k)
                own = (sum(w * r[field] for w, r in zip(weights, current)) + k * prior) / (sum(weights) + k)
            profile[output] = own
        venue = [r for r in current if r['venue'] == side]
        vw = ws(venue)
        venue_own = weighted([r['own'] for r in venue], vw)
        if k:
            venue_own = (sum(w * r['own'] for w, r in zip(vw, venue)) + k * profile['cards_per_90_team']) / (sum(vw) + k)
        profile['cards_' + side + '_pm'] = venue_own
        profile['fouls_per_90_team'] = weighted([r.get('fouls') for r in current], weights)
        # Aggression, legacy yellow/red display decomposition and league regime
        # cannot be reconstructed safely; none is fabricated from the v2 total.
        profiles[side] = profile
        last = list(reversed(history))[:6]
        rw = ws(last) if spec.get('half_life') else [.85 ** i for i in range(len(last))]
        recent[side] = {'n': len(last), 'cards_avg': weighted([r['own'] for r in last], rw),
                        'cards_induced_avg': weighted([r['induced'] for r in last], rw),
                        'fouls_avg': weighted([r.get('fouls') for r in last], rw)}
        if recent[side]['cards_avg'] is None and k:
            recent[side].update(cards_avg=profile['cards_per_90_team'], cards_induced_avg=profile['opp_cards_induced_pm'])
        effective[side] = sum(weights) ** 2 / sum(w*w for w in weights) if weights else 0.
    return profiles, recent, effective


def full_mean(f, spec, engine, *, lineup=False):
    parts = profile_inputs(f, spec)
    if parts is None:
        return None
    profiles, recent, effective = parts
    w = deepcopy(engine['SCORING_WEIGHTS'])
    if 'own' in spec:
        w['projection_cards'].update(own=spec['own'], opp=1 - spec['own'])
    if 'recent' in spec:
        w['projection'].update(blend_cards_season=1-spec['recent'], blend_cards_recent=spec['recent'])
    if 'venue' in spec:
        w['projection_cards']['venue_blend'] = spec['venue']
    ref = deepcopy(f['referee'])
    mode = spec.get('referee', 'both')
    if mode in ('none', 'anchor'):
        ref['multiplier'] = 1.
    if mode in ('none', 'modifier', 'residual'):
        w['referee']['anchor_weight'] = 0.
    if mode == 'none':
        ref.update(source='unavailable', confidence=0., cards_per_foul=0.)
    if mode == 'residual':
        residual = f['referee_residual']
        ref['multiplier'] = 1 + .35 * residual['n'] / (residual['n'] + 20) * (
            residual['actual'] / residual['expected'] - 1) if residual['expected'] > 0 else 1.
    if not spec.get('fouls', True):
        w['projection_cards']['foul_card_blend'] = 0.
    risk = None
    if lineup and f.get('lineup_scenario'):
        risk = SimpleNamespace(source='lineups', **{s+'_card_risk': SimpleNamespace(**f['lineup_scenario'][s]) for s in ('home','away')})
    original = engine['SCORING_WEIGHTS']
    engine.update(SCORING_WEIGHTS=w, _profile_as_of=lambda team, *a: profiles[team],
                  _recent_stats=lambda team, *a, **k: recent[team])
    try:
        detail = engine['projected_total_cards_detail']('home', 'away', f['competition'],
                    ref_mod=SimpleNamespace(**ref), lineup_ctx=risk, fixture_date=f['as_of'])
    finally:
        engine['SCORING_WEIGHTS'] = original
    mean = detail['total_cards']
    if mean is None or not math.isfinite(mean) or mean <= 0:
        raise ValueError('Nonpositive candidate mean; comparison cannot silently drop it')
    return {'mean': mean, 'profiles': profiles, 'recent': recent, 'effective_sample_sizes': effective}


def build_features(rows, context, weights):
    """One validated, exact chronological index; no per-forecast global rescans."""
    history = fixed.qualified_rows(rows, minimum_recorded_minutes=2)
    by_league, player_history, base_means = defaultdict(list), defaultdict(list), {}
    result = []
    for row in history:
        cutoff = cards.utc(row['kickoff']).replace(hour=0, minute=0, second=0, microsecond=0)
        past = [r for r in by_league[row['competition']] if r['season'] in (row['season']-1, row['season'])
                and cards.utc(r['kickoff']) + timedelta(hours=3) < cutoff]
        inputs = {'cutoff': cutoff.isoformat(), 'teams': {}, 'availability': 'assumed_final',
                  'source_fixture_ids': [r['fixture_id'] for r in past],
                  'league': [{'fixture_id': r['fixture_id'], 'total': r['target']} for r in past],
                  'referee_key': fixed.referee_key(row.get('referee'))}
        for side in ('home','away'):
            team = []
            for old in past:
                old_side = next((s for s in ('home','away') if row[side+'_team_id'] == old[s+'_team_id']), None)
                if old_side:
                    team.append({'fixture_id': old['fixture_id'], 'kickoff': old['kickoff'], 'season': old['season'],
                                 'venue': old_side, 'own': old['team_targets'][old_side],
                                 'induced': old['team_targets']['away' if old_side == 'home' else 'home']})
            inputs['teams'][side] = team
        refs = [r for r in past if inputs['referee_key'] and fixed.referee_key(r.get('referee')) == inputs['referee_key']]
        inputs['referee'] = [{'fixture_id': r['fixture_id'], 'total': r['target']} for r in refs]
        baseline, reason = fixed.fixed_prediction(row, inputs, weights)
        # Save the fixed snapshot identity BEFORE enriching inputs with foul data.
        fixed_id = fixed.digest(inputs)
        for side in ('home','away'):
            for item in inputs['teams'][side]:
                item['fouls'] = context[str(item['fixture_id'])]['fouls'][item['venue']]
        n = len(refs)
        confidence = min(1., max(0., (n-5)/15))
        refmean = float(np.mean([r['target'] for r in refs])) if refs else None
        league_mean = float(np.mean([r['target'] for r in past])) if past else None
        multiplier = 1+.35*confidence*(refmean/league_mean-1) if league_mean and refmean is not None else 1.
        foul_refs = [r for r in refs if all(context[str(r['fixture_id'])]['fouls'][s] is not None for s in ('home','away'))]
        foul_sum = sum(sum(context[str(r['fixture_id'])]['fouls'].values()) for r in foul_refs)
        cpf = sum(r['target'] for r in foul_refs)/foul_sum if len(foul_refs) >= 5 and foul_sum > 0 else 0.
        residual_rows = [r for r in refs if r['fixture_id'] in base_means]
        ref = {'source': 'profile' if n >= 5 else 'unavailable', 'confidence': confidence,
               'multiplier': multiplier, 'sample_size': n, 'avg_cards_per_match': refmean or 0.,
               'cards_per_foul': cpf}
        feature = {k: row[k] for k in ('fixture_id','competition','season','kickoff','target')}
        feature.update(as_of=cutoff.isoformat(), available_at=(cards.utc(row['kickoff'])+timedelta(hours=3)).isoformat(),
                       inputs=inputs, fixed_snapshot_id=fixed_id, baseline=baseline, baseline_exclusion=reason,
                       league_team_mean=league_mean/2 if league_mean is not None else None,
                       referee=ref, referee_foul_matches=len(foul_refs),
                       referee_residual={'n':len(residual_rows),'actual':sum(r['target'] for r in residual_rows),
                                         'expected':sum(base_means[r['fixture_id']] for r in residual_rows),
                                         'fixture_ids':[r['fixture_id'] for r in residual_rows]})
        feature['stage'] = '0-7' if baseline is None else baseline['season_stage']
        feature['foul_state'] = 'available' if cpf and all(any(r['fouls'] is not None for r in inputs['teams'][s]) for s in ('home','away')) else 'missing_or_sparse'
        # Conditional actual-XI scenario. Its availability is never asserted.
        risk = {}
        if past:
            pp = [p for r in past for p in r['player_evidence']['players'] if p['minutes'] is not None and p['minutes'] >= 2]
            rate = sum(p['weighted_cards'] for p in pp)/sum(p['minutes'] for p in pp)
            for side in ('home','away'):
                players = []
                for pid in context[str(row['fixture_id'])]['starters'][side]:
                    earlier = [p for p in player_history[(row['competition'],str(pid))]
                               if p['season'] in (row['season']-1,row['season']) and p['available'] < cutoff]
                    minutes = sum(p['minutes'] for p in earlier)
                    per_min = (sum(p['cards'] for p in earlier)+450*rate)/(minutes+450)
                    expected_minutes = (minutes+5*90)/(len(earlier)+5)
                    players.append({'player_id':pid,'prior_minutes':minutes,'prior_appearances':len(earlier),
                                    'expected_minutes':expected_minutes,'weighted_cards_per_minute':per_min,
                                    'source_fixture_ids':[p['fixture_id'] for p in earlier]})
                risk[side] = {'n_players':len(players), 'players':players,
                              'team_starter_cards_per_90':sum(p['expected_minutes']*p['weighted_cards_per_minute'] for p in players),
                              'high_card_risk_players':sum(p['weighted_cards_per_minute']*90>=.4 for p in players)}
        feature['lineup_scenario'] = risk or None
        # The source IDs retain complete league membership without duplicating
        # millions of league-total dictionaries in the feature artifact.
        inputs.pop('league')
        if baseline:
            base_means[row['fixture_id']] = baseline['team_only_mean']
        feature['input_id'] = fixed.digest({k:v for k,v in feature.items() if k != 'target'})
        result.append(feature)
        by_league[row['competition']].append(row)
        for p in row['player_evidence']['players']:
            if p['minutes'] is not None and p['minutes'] >= 2:
                player_history[(row['competition'],p['player_id'])].append({
                    'fixture_id':row['fixture_id'],'season':row['season'],'available':cards.utc(row['kickoff'])+timedelta(hours=3),
                    'minutes':p['minutes'],'cards':p['weighted_cards']})
    return result


def predict_registry(features, engine):
    records = []
    for f in features:
        if not f['baseline']:
            continue
        derived = {s['id']: full_mean(f,s,engine) for s in registry()}
        predictions = {k:v['mean'] for k,v in derived.items()}
        # Reconstruct the actual season/recent variance inputs using qualified
        # v2 observations. Production blends 0.7 season / 0.3 recent-eight.
        variances = []
        for side in ('home','away'):
            current = [r['own'] for r in f['inputs']['teams'][side] if r['season']==f['season']]
            variances.append(.7*float(np.var(current,ddof=1))+.3*float(np.var(current[-8:],ddof=1)))
        records.append({**{k:f[k] for k in ('fixture_id','competition','season','kickoff','as_of','available_at','target','input_id','stage','foul_state')},
                        'means':predictions, 'fixed_team':f['baseline']['team_only_mean'],
                        'fixed_referee':f['baseline']['team_referee_mean'], 'observed_variances':variances,
                        'effective_sample_sizes':{k:v['effective_sample_sizes'] for k,v in derived.items()},
                        'fuller_profiles':derived['fuller_control']['profiles'],
                        'fuller_recent':derived['fuller_control']['recent'],
                        'lineup_scenario_mean':full_mean(f,{'id':'fuller_control'},engine,lineup=True)['mean']})
    return records


def distribution(mean, alpha):
    mean = np.asarray(mean)
    alpha = np.asarray(alpha)
    if np.all(alpha == 0):
        return poisson(mean)
    if np.any(alpha <= 0):
        raise ValueError('Mixed zero/positive alpha must be evaluated separately')
    return nbinom(1/alpha, 1/(1+alpha*mean))


def training(rows, cutoff):
    boundary=cards.utc(cutoff)
    if boundary > cards.END:
        raise ValueError('Reserved fitting cutoff')
    selected=sorted([r for r in rows if cards.utc(r['available_at'])<boundary],key=lambda r:(cards.utc(r['kickoff']),r['fixture_id']))
    if len({r['fixture_id'] for r in selected}) != len(selected):
        raise ValueError('Repeated training fixture')
    if any(cards.utc(r['available_at']) < cards.utc(r['kickoff'])+timedelta(hours=3)
           or cards.utc(r['as_of']) > cards.utc(r['kickoff']) for r in selected):
        raise ValueError('Invalid forecast or label availability')
    if not fixed.support(selected,fitting=True)['sufficient']:
        raise ValueError('Insufficient chronological fitting support')
    return selected


def select_recipe(rows, features_by_id, engine, cutoff):
    rows=training(rows,cutoff)
    y=np.array([r['target'] for r in rows])
    losses={s['id']:float(-poisson([r['means'][s['id']] for r in rows]).logpmf(y).mean()) for s in registry()}
    specs={s['id']:s for s in registry()}
    base=losses['fuller_control']
    rate_keys=[k for k in losses if k.split(':')[0] in ('own','recent','venue','pool','half_life')]
    ref_keys=[k for k in losses if k.startswith('referee:')]
    rate=min(rate_keys,key=lambda k:(losses[k],k)); ref=min(ref_keys,key=lambda k:(losses[k],k))
    if all((base-losses[k])/base>=.005 for k in (rate,ref)):
        specs['combination:rate_referee']={**specs[rate],**specs[ref],'id':'combination:rate_referee'}
        if (base-losses['no_fouls'])/base>=.005:
            specs['combination:rate_referee_no_fouls']={**specs['combination:rate_referee'],'id':'combination:rate_referee_no_fouls','fouls':False}
    for k,s in specs.items():
        if k.startswith('combination:'):
            means=[full_mean(features_by_id[r['fixture_id']],s,engine)['mean'] for r in rows]
            losses[k]=float(-poisson(means).logpmf(y).mean())
    best=min(losses.values())
    chosen=min((k for k in losses if losses[k]<=best+1e-8),key=lambda k:(k!='fuller_control',len(specs[k]),k))
    means=np.array([full_mean(features_by_id[r['fixture_id']],specs[chosen],engine)['mean'] for r in rows])
    dispersion={str(a):float(-distribution(means,a).logpmf(y).mean()) for a in ALPHAS}
    alpha=min(ALPHAS,key=lambda a:(dispersion[str(a)],a))
    result={'version':VERSION,'cutoff':cutoff,'spec':specs[chosen],'alpha':alpha,
            'mean_losses':losses,'dispersion_losses':dispersion,'training_ids':[r['fixture_id'] for r in rows],
            'training_hash':fixed.digest(rows),'support':fixed.support(rows,fitting=True),'publication_enabled':False}
    return {**result,'id':fixed.digest(result)}


def transform_cdf(cdf, a, b):
    x=np.asarray(cdf,dtype=float)
    if b<=0 or np.any((x<0)|(x>1)):
        raise ValueError('Invalid monotone CDF transform')
    with np.errstate(divide='ignore',invalid='ignore'):
        return expit(a+b*logit(x))


def calibrated_mass(dist, y, a, b):
    # Upper-tail survival differences avoid catastrophic CDF cancellation.
    upper=transform_cdf(dist.cdf(y),a,b)-transform_cdf(dist.cdf(np.asarray(y)-1),a,b)
    lower=transform_cdf(dist.sf(np.asarray(y)-1),-a,b)-transform_cdf(dist.sf(y),-a,b)
    return np.where(dist.cdf(y)<=.5,upper,lower)


def fit_calibration(rows,cutoff='2023-04-01T00:00:00+00:00'):
    rows=training(rows,cutoff)
    if any(cards.utc(r['as_of'])<cards.utc(r['fit_cutoff']) or r['fixture_id'] in r['fit_ids'] for r in rows):
        raise ValueError('Calibration needs genuinely held-out predictions')
    y=np.array([r['target'] for r in rows]); n=len(rows)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['alpha']].append(i)
    def objective(params):
        a,b=params; losses=[]
        for alpha,ii in groups.items():
            d=distribution([rows[i]['mean'] for i in ii],alpha)
            p=calibrated_mass(d,y[ii],a,b)
            if np.any(p<=0):return 1e100
            losses.extend(-np.log(p))
        return float(np.mean(losses)+(a*a+(b-1)**2)/(2*.1*n))
    fit=minimize(objective,[0.,1.],method='L-BFGS-B',bounds=[(-2,2),(.25,4)],options={'maxiter':1000,'ftol':1e-12})
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError('Calibrator failed convergence')
    result={'version':VERSION,'kind':'monotone_count_cdf','a':float(fit.x[0]),'b':float(fit.x[1]),'C':.1,
            'cutoff':cutoff,'support':fixed.support(rows,fitting=True),'training_ids':[r['fixture_id'] for r in rows],
            'training_hash':fixed.digest(rows),'converged':True,'publication_enabled':False}
    return {**result,'id':fixed.digest(result)}


def score(mean,alpha,target,a=0.,b=1.):
    d=distribution(mean,alpha)
    mass=float(calibrated_mass(d,target,a,b))
    if mass<=0 or not math.isfinite(mass):raise ValueError('Invalid target probability')
    limit=max(16,int(mean+10*math.sqrt(mean+alpha*mean*mean)))
    while float(transform_cdf(d.sf(limit),-a,b))>1e-10:
        limit*=2
        if limit>16384:raise ValueError('Unbounded count tail')
    values=np.arange(limit+1)
    pmf=calibrated_mass(d,values,a,b)
    tail=float(transform_cdf(d.sf(limit),-a,b))
    if np.any(pmf<0) or abs(float(pmf.sum())+tail-1)>1e-10:raise ValueError('Invalid count distribution')
    pmf=pmf/pmf.sum()
    expected=float(values@pmf)
    diagnostics={}
    for line in (3.5,4.5,5.5):
        p=float(transform_cdf(d.sf(math.floor(line)),-a,b)); y=target>line
        clipped=np.clip(p,1e-15,1-1e-15)
        diagnostics[str(line)]={'p':p,'y':int(y),'brier':(p-y)**2,'log_loss':float(-math.log(clipped if y else 1-clipped))}
    quantiles=[int(np.searchsorted(pmf.cumsum(),q)) for q in (.025,.1,.9,.975)]
    return {'nll':-math.log(mass),'mean':expected,'raw_mean':mean,'alpha':alpha,'tail_bound':tail,
            'pmf':pmf.tolist(),'diagnostics':diagnostics,'interval80':quantiles[1:3],
            'interval95':[quantiles[0],quantiles[3]],
            'asian':{str(line):{'over':asian_profile(values,pmf,line),'under':list(reversed(asian_profile(values,pmf,line)))}
                     for line in (4.,4.25,4.75,5.)}}


def summary(rows,key):
    scores=[r['scores'][key] for r in rows]
    diff=np.array([s['mean']-r['target'] for r,s in zip(rows,scores)])
    result={**fixed.support(rows),'nll':float(np.mean([s['nll'] for s in scores])),
            'mae':float(np.abs(diff).mean()),'bias':float(diff.mean()),'rmse':float(np.sqrt(np.mean(diff**2))),
            'league_contributions':dict(__import__('collections').Counter(r['competition'] for r in rows))}
    for coverage in ('80','95'):
        result['interval'+coverage+'_coverage']=float(np.mean([s['interval'+coverage][0]<=r['target']<=s['interval'+coverage][1] for r,s in zip(rows,scores)]))
    result['lines']={}
    for line in ('3.5','4.5','5.5'):
        ss=[s['diagnostics'][line] for s in scores]
        bins=[]
        for i in range(10):
            selected=[s for s in ss if i/10<=s['p']<(i+1)/10 or i==9 and s['p']==1]
            bins.append({'lower':i/10,'n':len(selected),'probability':float(np.mean([s['p'] for s in selected])) if selected else None,
                         'frequency':float(np.mean([s['y'] for s in selected])) if selected else None})
        result['lines'][line]={'brier':float(np.mean([s['brier'] for s in ss])),
                              'log_loss':float(np.mean([s['log_loss'] for s in ss])),'reliability':bins}
    return result


def compare(rows,candidate,control):
    if not rows:return {'status':'insufficient_support','fixtures':0}
    cc,bb=summary(rows,candidate),summary(rows,control)
    groups=defaultdict(list)
    for r in rows:groups[cards.utc(r['kickoff']).strftime('%G-W%V')].append(r['scores'][candidate]['nll']-r['scores'][control]['nll'])
    weeks=sorted(groups); sums=np.array([sum(groups[w]) for w in weeks]); counts=np.array([len(groups[w]) for w in weeks])
    draws=np.random.default_rng(SEED).integers(0,len(weeks),(2000,len(weeks)))
    interval=np.quantile(sums[draws].sum(axis=1)/counts[draws].sum(axis=1),[.025,.975]).tolist()
    slices=defaultdict(list)
    for r in rows:
        for field in ('competition','stage','foul_state'):slices[field+':'+r[field]].append(r)
    ss={}
    for k,rr in sorted(slices.items()):
        base=float(np.mean([r['scores'][control]['nll'] for r in rr])); new=float(np.mean([r['scores'][candidate]['nll'] for r in rr]))
        ss[k]={**fixed.support(rr),'relative_regression':new/base-1}
    improvement=1-cc['nll']/bb['nll']
    passed=cc['sufficient'] and improvement>=.005 and interval[1]<0 and all(not s['sufficient'] or s['relative_regression']<=.02 for s in ss.values())
    return {'candidate':cc,'control':bb,'relative_improvement':improvement,'nll_delta':cc['nll']-bb['nll'],
            'paired_week_bootstrap95':interval,'slices':ss,'development_gate_passed':passed,
            'production_qualified':False,'period_inspected_previously':True}
