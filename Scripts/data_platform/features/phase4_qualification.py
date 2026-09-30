"""Fixed January–June 2025 research contract; no IO or method selection."""
from collections import defaultdict
from datetime import timedelta
import numpy as np

try:
    import phase4_candidate_math as cm
    import phase4_finalist_math as finalists
    import phase4_comparison_math as scores
    import phase4_calibration_math as calibration
except ModuleNotFoundError:
    from Scripts.data_platform.features import phase4_candidates as cm, phase4_finalists as finalists
    from Scripts.data_platform.features import phase4_backtest as scores, phase4_calibration as calibration

VERSION = 'phase4-qualification.v2'
START = '2025-01-01T00:00:00+00:00'
END = '2025-07-01T00:00:00+00:00'
CLAIMS = ('goals', 'corners', 'sot', 'corner_over9.5_binary')
CALIBRATOR_ID = '12b3c7435428288ef970d90374e634bb679143d6117cfaab66c55b6fb9bb8a6f'
RECIPE_ID = '3778878a8ce078700548925b93e79d6f13bff01fb59de3fe19c5da165a62fdac'


def qualification_time(value):
    if not cm.utc(START) <= cm.utc(value) < cm.utc(END):
        raise ValueError('Outside the authorized qualification interval')


def preflight(memberships):
    allowed = {'competition','completed','eligible_markets','fixture_id','kickoff','partition','row_count','season'}
    seen, selected = set(), []
    for row in memberships:
        if set(row)-allowed or row['fixture_id'] in seen:
            raise ValueError('Outcomes or duplicate fixture in split metadata')
        seen.add(row['fixture_id'])
        if cm.utc(START) <= cm.utc(row['kickoff']) < cm.utc(END):
            if row['partition'] != 'calibration':
                raise ValueError('Original reserve membership changed')
            selected.append(row)
    result = {}
    for market in cm.MARKETS:
        rows = [r for r in selected if market in r['eligible_markets'] and r['competition'] in finalists.LEAGUES]
        result[market] = support([{'week':week(r['kickoff'])} for r in rows])
    if not all(r['sufficient'] for r in result.values()):
        raise ValueError('Insufficient outcome-free qualification support')
    return {'version':VERSION,'period':[START,END],'supported_league_upper_bounds':result,
            'metadata_only':True,'not_yet_certified_for_season_stage':True}


def week(kickoff):
    y,w,_ = cm.utc(kickoff).isocalendar()
    return f'{y}-W{w:02d}'


def fit_strength(history, eligible_ids, competition):
    if competition not in finalists.LEAGUES:
        raise ValueError('Unapproved strength scope')
    # Reject a contaminated fitting payload even if the underlying fitter would
    # filter future rows. Qualification labels must not enter this operation.
    if any(cm.utc(r['kickoff'])+timedelta(hours=3) >= cm.utc(START) for r in history):
        raise ValueError('Qualification label in the pre-qualification fit')
    return cm.fit_strength(history,eligible_ids=eligible_ids,competition=competition,
        market='goals',cutoff=START,season=2024,ridge=10.,end=cm.utc(START))


def check_bundle(bundle):
    if (bundle.get('cutoff') != START or bundle.get('season') != 2024 or bundle.get('ridge') != 10.
            or bundle.get('market') != 'goals' or bundle.get('half_life_days') is not None
            or bundle.get('competition') not in finalists.LEAGUES
            or bundle.get('id') != cm.digest({k:v for k,v in bundle.items() if k!='id'})):
        raise ValueError('Not the frozen qualification strength method')


def compose(snapshot, control, strength):
    qualification_time(snapshot['as_of'])
    return finalists.compose(snapshot,control,strength,'supported_stack',start=START,end=END)


def calibrated_corner(bundle, snapshot, probability):
    qualification_time(snapshot['as_of'])
    if bundle.get('id') != CALIBRATOR_ID:
        raise ValueError('Calibration refit/substitution forbidden')
    row = {'problem':'corners','period':'regulation_time','as_of':snapshot['as_of'],
           'kickoff':snapshot['fixture']['kickoff'],'league':snapshot['fixture']['competition'],
           'bases':{'dispersion_corners:0.025':{'p':probability}}}
    return calibration.predict_binary(bundle,row,end=cm.utc(END))


def support(rows, kind='pooled'):
    minimum, weeks = (100,10) if kind=='league' else (200,20) if kind=='slice' else (500,10)
    n, observed = len(rows), len({r['week'] for r in rows})
    return {'n':n,'weeks':observed,'minimum_fixtures':minimum,'minimum_weeks':weeks,
            'sufficient':n>=minimum and observed>=weeks,'effective_sample_size':n}


def paired_inference(rows, deltas):
    if not rows or len(rows)!=len(deltas) or not np.isfinite(deltas).all():
        raise ValueError('Invalid paired inference cohort')
    groups = defaultdict(list)
    for row,delta in zip(rows,deltas):
        groups[row['week']].append(float(delta))
    weeks = sorted(groups)
    sums = np.array([sum(groups[w]) for w in weeks])
    counts = np.array([len(groups[w]) for w in weeks])
    mean = float(sums.sum()/counts.sum())
    draws = np.random.default_rng(scores.SEED).integers(0,len(weeks),size=(2000,len(weeks)))
    boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    centered = boot-mean
    return {'paired_week_delta95':np.quantile(boot,[.025,.975]).tolist(),
            'one_sided_p':float((1+np.count_nonzero(centered<=mean))/2001),
            'mean_difference':mean,'bootstrap_draws':2000,'seed':scores.SEED}


def holm(p_values):
    if set(p_values)!=set(CLAIMS) or any(not 0<=p<=1 for p in p_values.values()):
        raise ValueError('Exactly four declared primary claims required')
    result, previous = {}, 0.
    for rank,(name,p) in enumerate(sorted(p_values.items(),key=lambda x:(x[1],x[0]))):
        adjusted = min(1.,max(previous,(len(CLAIMS)-rank)*p))
        result[name] = {'raw_p':p,'adjusted_p':adjusted,'passes':adjusted<=.05}
        previous = adjusted
    return result


def compare(reference, candidate, *, binary=False):
    if ([r['fixture_id'] for r in reference] != [r['fixture_id'] for r in candidate]
            or len({r['fixture_id'] for r in reference})!=len(reference)):
        raise ValueError('Unpaired or duplicate qualification fixtures')
    if any(any(a[k]!=b[k] for k in ('week','league','season','season_stage','forecast_stage','missingness'))
           for a,b in zip(reference,candidate)):
        raise ValueError('Pair metadata differs')
    def metric(rows, kind='pooled'):
        if not rows:return support(rows,kind)
        if binary:
            r = {key:float(np.mean([x['scores'][key] for x in rows])) for key in ('nll','brier')}
            r['reliability'] = scores.reliability([x['scores'] for x in rows])
        else:
            r = scores.metrics(rows,diagnostics=True)
        return {**r,**support(rows,kind)}
    a,b = metric(reference),metric(candidate)
    if not reference:return {'control':a,'candidate':b,'passes_before_holm':False,'one_sided_p':1.}
    delta = np.array([y['scores']['nll']-x['scores']['nll'] for x,y in zip(reference,candidate)])
    inference = paired_inference(reference,delta)
    slices, regressions, related = {}, [], []
    for key in ('league','season','season_stage','forecast_stage','missingness'):
        slices[key] = {}
        for value in sorted({str(r[key]) for r in reference}):
            indices = [i for i,r in enumerate(reference) if str(r[key])==value]
            rows = [reference[i] for i in indices]
            ma,mb = [float(np.mean([collection[i]['scores']['nll'] for i in indices])) for collection in (reference,candidate)]
            s = support(rows,'league' if key=='league' else 'slice')
            change = mb/ma-1
            slices[key][value] = {'support':s,'control_nll':ma,'candidate_nll':mb,'relative_change':change,
                **paired_inference(rows,delta[indices])}
            if s['sufficient'] and change>.02:regressions.append(key+':'+value)
    if not binary and a['sufficient']:
        for line,v in a['totals'].items():
            if b['totals'][line]['nll']>1.02*v['nll']:related.append('total:'+line)
        for name,key in (('btts','logloss'),('winner','nll')):
            if name in a['derived'] and b['derived'][name][key]>1.02*a['derived'][name][key]:related.append(name)
        for line,v in a['derived'].get('handicaps',{}).items():
            if b['derived']['handicaps'][line]['nll']>1.02*v['nll']:related.append('handicap:'+line)
    relative = b['nll']/a['nll']-1
    passed = a['sufficient'] and relative<=-.005 and inference['paired_week_delta95'][1]<0 and not regressions and not related
    return {'control':a,'candidate':b,'relative_change':relative,**inference,'slices':slices,
            'supported_slice_regressions':regressions,'related_market_regressions':related,
            'passes_before_holm':bool(passed)}
