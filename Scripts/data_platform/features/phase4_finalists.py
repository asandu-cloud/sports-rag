"""Pure research composition, evidence scopes and outcome-free reserve gates."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math

VERSION='phase4-finalist-review.v1'
LEAGUES=('EPL','LaLiga','SerieA','Bundesliga','Ligue1')
MARKETS=('goals','corners','sot')
ALPHAS={'corners':.025,'sot':.01}


def utc(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('Explicit timezone required')
    return t.astimezone(timezone.utc)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(encode(value).encode()).hexdigest()


def supported(league,stage):return league in LEAGUES and stage in ('8-15','16+')


def scope(snapshot):
    counts=[q.get('current_season_matches',0) for q in snapshot['profile_quality'].values()]
    if len(counts)!=2 or any(not isinstance(n,(int,float)) or not math.isfinite(n) or n<0 for n in counts):
        raise ValueError('Invalid team support')
    n=min(counts);stage='0-7' if n<8 else '8-15' if n<16 else '16+'
    return {'league':snapshot['fixture']['competition'],'season_stage':stage,
            'active':supported(snapshot['fixture']['competition'],stage)}


def compose(snapshot,control,strength,configuration,*,start='2022-01-01T00:00:00Z',end='2024-01-01T00:00:00Z'):
    if configuration not in ('component_stack','supported_stack'):raise ValueError('Undeclared configuration')
    if not utc(start)<=utc(snapshot['as_of'])<utc(end):
        raise ValueError('Development composition only')
    for rates in (control,strength):
        if rates['fixture_id']!=snapshot['fixture']['fixture_id'] or rates['input_snapshot_id']!=snapshot['snapshot_id']:
            raise ValueError('Mismatched fixture/input identity')
    active=configuration=='component_stack' or scope(snapshot)['active'];r=deepcopy(control)
    r['configuration']=configuration;r['configuration_version']=VERSION
    r['candidate']={'id':configuration,'family':'composite','version':VERSION,'markets':list(MARKETS)}
    r['component_ids']={m:'control' for m in MARKETS};r['evidence']={'fallbacks':[],'scope':scope(snapshot)}
    r['market_fallbacks']={m:[] for m in MARKETS}
    if active:
        r['goal_means']=deepcopy(strength['goal_means']);r['means']['goals']=strength['means']['goals']
        r['component_ids']['goals']='strength_goals:10.0'
        r['evidence']['fallbacks']=deepcopy(strength['evidence']['fallbacks'])
        r['market_fallbacks']['goals']=deepcopy(strength['evidence']['fallbacks'])
        for m,alpha in ALPHAS.items():
            mean=r['means'][m]
            if mean is not None:r['variances'][m]=mean+alpha*mean*mean
            r['component_ids'][m]='dispersion_'+m+':'+str(alpha)
    else:
        r['evidence']['fallbacks'].append('unsupported_league_or_season_stage')
        r['market_fallbacks']={m:['unsupported_league_or_season_stage'] for m in MARKETS}
    r['publication_enabled']=False;r['joint_cross_market_probability']=None
    r['status']='control_fallback' if r['evidence']['fallbacks'] else 'computed'
    return r


def goal_fit_id(configuration,active_scope,bundle_id):
    return bundle_id if configuration=='component_stack' or (configuration=='supported_stack' and active_scope) else None


def reserve_preflight(memberships,leagues=LEAGUES):
    """Only permit outcome-free membership metadata; never read a target store."""
    allowed={'competition','completed','eligible_markets','fixture_id','kickoff','partition','row_count','season'}
    ids=set();rows=[]
    for r in memberships:
        if set(r)-allowed:raise ValueError('Outcome/feature fields in metadata preflight')
        if r['fixture_id'] in ids:raise ValueError('Duplicate fixture membership')
        ids.add(r['fixture_id']);t=utc(r['kickoff'])
        if utc('2025-01-01T00:00:00Z')<=t<utc('2025-07-01T00:00:00Z'):
            if r['partition']!='calibration':raise ValueError('Reserved membership mismatch')
            rows.append(r)
    stages={}
    for name,start,end in [('calibration','2025-01-01','2025-04-01'),('qualification','2025-04-01','2025-07-01')]:
        start,end=(utc(d+'T00:00:00Z') for d in (start,end))
        period=[r for r in rows if start<=utc(r['kickoff'])<end]
        def count(items):
            return {'fixtures':len(items),'observed_iso_weeks':len({utc(r['kickoff']).isocalendar()[:2] for r in items})}
        markets={}
        for m in MARKETS:
            eligible=[r for r in period if m in r['eligible_markets']]
            active=[r for r in eligible if r['competition'] in leagues]
            all_count=count(eligible);candidate_count=count(active)
            reasons=[]
            if candidate_count['fixtures']<500:reasons.append('fewer_than_500_eligible_fixtures')
            if candidate_count['observed_iso_weeks']<10:reasons.append('fewer_than_10_observed_weeks')
            markets[m]={'all_leagues':all_count,'supported_leagues_upper_bound':candidate_count,
                'per_league':{lg:count([r for r in eligible if r['competition']==lg]) for lg in leagues},
                'status':'insufficient' if reasons else 'upper_bound_sufficient_needs_feature_and_class_checks','reasons':reasons}
        stages[name]={'markets':markets,'all_fixture_memberships':count(period),
            'dates':[min((r['kickoff'] for r in period),default=None),max((r['kickoff'] for r in period),default=None)]}
    insufficient=any(v['status']=='insufficient' for stage in stages.values() for v in stage['markets'].values())
    return {'version':VERSION,'thresholds':{'pooled_fixtures':500,'observed_weeks':10,'league_fixtures':100,'league_weeks':10},
        'stages':stages,'status':'blocked_insufficient_metadata_support' if insufficient else 'requires_separate_locked_input_preparation',
        'label_access_permitted':False,'reserved_outcomes_read':False,'calibrators_fitted':False,
        'final_system_test_opened':False,'metadata_limit':'Upper bound; does not certify forecast features, season stage or binary classes.'}


def require_qualification_support(report):
    if report['status']!='requires_separate_locked_input_preparation':
        raise ValueError('Qualification blocked by frozen sample/week requirements')
