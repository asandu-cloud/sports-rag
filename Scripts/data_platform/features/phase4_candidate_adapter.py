"""Execute isolated Step 2 changes around an explicitly supplied frozen engine.

The caller loads phase4_candidate_math by source path in the worker. This module
does not import or edit the active prediction engine.
"""
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import math

try:
    import phase4_candidate_math as cm  # Explicit archived source in the worker.
except ModuleNotFoundError:
    from Scripts.data_platform.features import phase4_candidates as cm

RATE_FIELDS = {
    'goals_for_pm':('goals',False,None),'goals_against_pm':('goals',True,None),
    'corners_pm':('corners',False,None),'corners_against_pm':('corners',True,None),
    'sot_for_pm':('sot',False,None),'sot_against_pm':('sot',True,None),
    'expected_goals':('xg',False,None),'xga_pm':('xg',True,None),
}
for _stat,_prefix in [('goals','goals'),('corners','corners'),('sot','sot'),('xg','xg')]:
    for _venue in ('home','away'):
        RATE_FIELDS[_prefix+'_'+_venue+'_pm']=(_stat,False,_venue)
        if _stat in ('corners','sot'):
            RATE_FIELDS[_prefix+'_against_'+_venue+'_pm']=(_stat,True,_venue)


def observations(rows, team, field):
    stat,opponent,venue=RATE_FIELDS[field];values=[]
    for row in rows:
        side='home' if str(row['home_team_id'])==str(team) else 'away'
        if str(row[side+'_team_id'])!=str(team): raise ValueError('Rate history team mismatch')
        if venue is not None and side!=venue: continue
        source=('away' if side=='home' else 'home') if opponent else side
        values.append((row['fixture_id'],row[source].get(stat)))
    return values


def summary(rows, team, field):
    return cm.weighted_summary([v for _,v in observations(rows,team,field)])


def component_summary(index, team, league, cutoff, field, audit):
    rows=index.before(team,league,cutoff)
    def rank(value):
        return int(str(value)[:4]) if value is not None else None
    current=[r for r in rows if r['season']==rank(audit.get('current_season'))]
    prior=[r for r in rows if r['season']==rank(audit.get('prior_season'))]
    c,p=summary(current,team,field),summary(prior,team,field)
    n=len(current)
    # Source audit rounds its displayed prior weight. Reconstruct the exact
    # numerical weight from the documented mode/count, never the rounded display.
    if audit['profile_mode']=='prior_only': weight=1.
    elif audit['profile_mode']=='current_plus_prior': weight=8/(n+8)
    else: weight=0.
    if c['mean'] is None: mean=p['mean'] if weight>0 else None
    elif p['mean'] is None or weight==0: mean=c['mean']
    else: mean=(1-weight)*c['mean']+weight*p['mean']
    return {'mean':mean,'ess':cm.mixed_support(c,p,weight),'current_n':c['n'],'prior_n':p['n'],
            'current_fixture_ids':[fid for fid,v in observations(current,team,field) if v is not None],
            'prior_fixture_ids':[fid for fid,v in observations(prior,team,field) if v is not None],
            'prior_weight':weight,'current_rank':rank(audit.get('current_season')),
            'prior_rank':rank(audit.get('prior_season'))}


def league_summary(index, league, cutoff, field, support):
    stat,opponent,venue=RATE_FIELDS[field]
    def one(rank):
        rows=cm.available_history(index.history.values(),cutoff,competition=league,seasons={rank})
        values=[r[side].get(stat) for r in rows for side in ('home','away') if venue is None or side==venue]
        result=cm.weighted_summary(values)
        result['fixture_ids']=[r['fixture_id'] for r in rows if any(r[s].get(stat) is not None
                              for s in ('home','away') if venue is None or s==venue)]
        if result['n']<20: result['mean']=None
        return result
    c,p=one(support['current_rank']),one(support['prior_rank'])
    w=support['prior_weight']
    if c['mean'] is None: value=p['mean'] if w>0 else None
    elif p['mean'] is None or w==0: value=c['mean']
    else: value=(1-w)*c['mean']+w*p['mean']
    return {'mean':value,'current_n':c['n'],'prior_n':p['n'],
            'current_fixture_ids':c['fixture_ids'],'prior_fixture_ids':p['fixture_ids']}


def forecast(snapshot, index, spec, engine, *, strength_bundle=None, probability_bundle=None, lineup=None, rates_only=False, end=cm.END):
    cm.validate_spec(spec)
    if cm.utc(snapshot['as_of'])>=end: raise ValueError('Reserved forecast refused')
    p,tr,ms,prob,ls=(getattr(engine,k) for k in ('projections','teams','markets','probability','lines'))
    family,parameter=spec['family'],spec['parameter']
    fixture=snapshot['fixture'];cutoff=snapshot['as_of'];league=fixture['competition']
    home,away=str(fixture['home_team_id']),str(fixture['away_team_id'])
    if cm.utc(cutoff)>cm.utc(fixture['kickoff']): raise ValueError('Post-kickoff forecast refused')
    evidence={'family':family,'parameter':parameter,'field_support':{},'fallbacks':[]}
    if probability_bundle is not None:
        cm.validate_probability_bundle(probability_bundle,spec,cutoff)
        evidence['probability_fit_id']=probability_bundle['id']
    weights=deepcopy(p.SCORING_WEIGHTS)
    if family=='own_weight':
        weights['projection'].update(goals_own=parameter,goals_opp=1-parameter,corners_own=parameter,corners_opp=1-parameter)
        weights['projection_sot'].update(own=parameter,opp=1-parameter)
    if family=='venue_weight':
        for m in cm.MARKETS: weights['projection'][m+'_venue_blend']=parameter
    if family=='xg_weight': weights['projection']['xg_blend']=parameter
    if family=='recent_weight':
        for m in cm.MARKETS:
            weights['projection']['blend_'+m+'_recent']=parameter
            weights['projection']['blend_'+m+'_season']=1-parameter
    original_context=tr.get_team_profile_context
    original_recent=tr.get_team_recent_stats
    original_goals=p.projected_goals
    calculated_profiles={};calculated_recent={};quality={}
    tr.clear_profile_caches()

    def continuous(current,prior):
        n=float(current.get('matches_played') or 0)
        if not prior: return dict(current),0.,'current_only'
        w=parameter/(n+parameter)
        result=dict(prior);result.update(current)
        for key in set(current)|set(prior):
            if key=='matches_played': continue
            c,q=current.get(key),prior.get(key)
            if isinstance(c,(int,float)) and not isinstance(c,bool) and isinstance(q,(int,float)) and not isinstance(q,bool):
                result[key]=(1-w)*c+w*q
            elif c is None and q is not None: result[key]=q
        result['matches_played']=int(n)
        return result,w,'prior_only' if n==0 else 'current_plus_prior'

    def team_context(team,lg,target_date=None):
        meta,audit=original_context(team,lg,target_date=target_date)
        meta=deepcopy(meta);audit=deepcopy(audit)
        if not meta: return meta,audit
        if family=='defensive_xg':
            stats=component_summary(index,team,lg,cutoff,'xga_pm',audit)
            evidence['field_support'][str(team)+':'+lg+':xga']=stats
            actual=meta.get('goals_against_pm')
            if stats['mean'] is not None and actual is not None:
                meta['goals_against_pm']=(1-parameter)*actual+parameter*stats['mean']
            else: evidence['fallbacks'].append(str(team)+':defensive_xg_unavailable')
        if family=='league_pool':
            for field in ('goals_for_pm','goals_against_pm','corners_pm','corners_against_pm','sot_for_pm','sot_against_pm'):
                stats=component_summary(index,team,lg,cutoff,field,audit)
                prior=league_summary(index,lg,cutoff,field,stats)
                evidence['field_support'][str(team)+':'+lg+':'+field]={'team':stats,'league':prior}
                meta[field]=cm.pool(meta.get(field),stats['ess'],prior['mean'],parameter)
        if family=='venue_pool':
            for field in RATE_FIELDS:
                if RATE_FIELDS[field][2] is None: continue
                overall = {'goals':'goals_for_pm','corners':'corners_pm','sot':'sot_for_pm','xg':'expected_goals'}[RATE_FIELDS[field][0]]
                if RATE_FIELDS[field][1]: overall={'corners':'corners_against_pm','sot':'sot_against_pm'}[RATE_FIELDS[field][0]]
                stats=component_summary(index,team,lg,cutoff,field,audit)
                evidence['field_support'][str(team)+':'+lg+':'+field]=stats
                meta[field]=cm.pool(meta.get(field),stats['ess'],meta.get(overall),parameter)
        return meta,audit

    def recent_stats(team,lg,last_n=6,venue=None,target_date=None):
        window=int(parameter) if family=='recent_window' else last_n
        rows=tr.get_recent_team_fixture_rows(team,lg,limit=max(window*3,12) if venue else max(window,4),target_date=target_date)
        if venue: rows=[r for r in rows if r['meta']['home_away']==venue]
        rows=rows[:window]
        observation_weights=([2**(-((cm.utc(cutoff)-cm.utc(r['meta']['fixture_date'])).total_seconds()/86400)/parameter) for r in rows]
            if family=='calendar_recency' else [weights['recency']['alpha']**i for i in range(len(rows))])
        evidence.setdefault('recent_inputs',{})[str(team)+':'+lg+':'+str(venue or 'all')]={
            'fixture_ids':[int(r['meta']['fixture']) for r in rows], 'base_weights':observation_weights,
            'missingness_policy':'renormalize over each field known values'}
        if family!='calendar_recency':
            return original_recent(team,lg,window,venue=venue,target_date=target_date)
        with patch.object(tr,'_weighted_avg',new=lambda values,alpha:cm.weighted_summary(values,observation_weights)['mean']):
            return original_recent(team,lg,window,venue=venue,target_date=target_date)

    def xg_goals(hm,am):
        # Only the explicit fallback candidate supplies overall xG when the
        # original primitive cannot form its two-venue average. Raw profiles
        # remain intact; the calculation strategy is recorded separately.
        copies=[]
        for team,meta in ((home,hm),(away,am)):
            copy=deepcopy(meta)
            if (meta.get('xg_home_pm') is None or meta.get('xg_away_pm') is None) and meta.get('expected_goals') is not None:
                copy['xg_home_pm']=copy['xg_away_pm']=meta['expected_goals']
                evidence['field_support'][team+':xg_fallback']={'strategy':'overall_observation_mean','value':meta['expected_goals']}
            copies.append(copy)
        return original_goals(*copies)

    with ExitStack() as stack:
        patches=[(tr,'_profile_data_revision',lambda:('candidate',spec['id'],fixture['fixture_id'])),
            (tr,'get_team_profile_docs',lambda *a,**k:[]),
            (tr,'_get_all_team_fixture_rows',lambda team,lg:index.rows(team,lg,cutoff)),
            (tr,'_prefetch_opponent_fixture_meta',lambda *a,**k:None),
            (tr,'_get_team_fixture_meta',lambda team,lg,fid,*a,**k:index.metas.get((str(team),str(fid)),{}).get('meta')),
            (tr,'resolve_domestic_league',lambda team:index.domestic(team,cutoff)),
            (p,'get_card_risk_profiles',lambda *a,**k:(None,None)),
            (p,'SCORING_WEIGHTS',weights),(tr,'SCORING_WEIGHTS',weights)]
        if family=='gradual_prior': patches.append((tr,'_blend_sparse_current_profile',continuous))
        if family in ('league_pool','venue_pool','defensive_xg'): patches.append((tr,'get_team_profile_context',team_context))
        if family in ('calendar_recency','recent_window'): patches.append((tr,'get_team_recent_stats',recent_stats))
        for module,name,value in patches: stack.enter_context(patch.object(module,name,new=value))
        for team in (home,away):
            if family in ('gradual_prior','league_pool','venue_pool','defensive_xg'):
                calculated_profiles[team],quality[team]=tr.get_prediction_profile_context(team,league,target_date=cutoff)
            else:
                calculated_profiles[team]=deepcopy(snapshot['profiles'][team]);quality[team]=deepcopy(snapshot['profile_quality'][team])
            calculated_recent[team]=tr._recent_stats(team,league,target_date=cutoff) if family in ('calendar_recency','recent_window') else deepcopy(snapshot['recent'][team])
        def profile(team,*a,**k): return deepcopy(calculated_profiles[team])
        def recent(team,*a,**k): return deepcopy(calculated_recent[team])
        def profile_quality(team,*a,**k): return profile(team),deepcopy(quality[team])
        for module,name,value in [(p,'_profile_meta',profile),(tr,'_profile_meta',profile),(p,'_recent_stats',recent),
                (tr,'get_team_recent_variance',lambda team,*a,**k:deepcopy(snapshot['recent_variance'][team])),
                (ms,'get_prediction_profile_context',profile_quality)]:
            stack.enter_context(patch.object(module,name,new=value))
        if family=='xg_fallback': stack.enter_context(patch.object(p,'projected_goals',new=xg_goals))
        contexts=dict(fixture_date=cutoff,league_ctx=SimpleNamespace(**snapshot['league_context']),
                      knockout_ctx=SimpleNamespace(**snapshot['knockout_context']))
        if family=='own_weight':
            for market in ('corners','sot'):
                function=own_weight_total(p,calculated_profiles,calculated_recent,home,away,market,parameter)
                stack.enter_context(patch.object(p,'projected_total_'+market,new=function))
                stack.enter_context(patch.object(ms,'projected_total_'+market,new=function))
        event={'id':str(fixture['fixture_id']),'home_team':home,'away_team':away,'commence_time':fixture['kickoff'],'bookmakers':[]}
        score,h,a=p.projected_correct_score_probs(home,away,league,**contexts)
        if rates_only:
            # Same frozen numerical functions as market_service, without building
            # unpriced publication/selection payloads. Parity is tested separately.
            projections={'goals':{'value':h+a if score else None}}
            for market in ('corners','sot'):
                value,_,_=getattr(p,'projected_total_'+market)(home,away,league,**contexts)
                variance,details=ms._total_variance(home,away,league,market,value,cutoff)
                projections[market]={'value':value,'variance':variance,'components':{'variance_source':details['source']}}
        else:
            results=ms.evaluate_event(event,league,**contexts,ref_mod=SimpleNamespace(**snapshot['referee']),
                generated_at=cutoff,input_snapshot_id=snapshot['snapshot_id'],
                model_version='phase4-repaired-statistical-control.v1' if family=='control' else cm.VERSION+':'+spec['id'])
            originals=[r.to_dict() for r in results]
            projections={r['market']['group']:r['projection'] for r in originals}
        means={m:projections[m]['value'] for m in cm.MARKETS}
        goal_means=[h,a]
        if family.startswith('strength_'):
            market=family.removeprefix('strength_')
            if strength_bundle is None:
                evidence['fallbacks'].append('strength_bundle_unavailable')
            else:
                if strength_bundle['market']!=market or strength_bundle['ridge']!=parameter or strength_bundle.get('half_life_days') is not None:
                    raise ValueError('Wrong fitted strength candidate')
                strength=cm.predict_strength(strength_bundle,fixture,cutoff,end=end)
                evidence['strength']=strength
                if strength['status']=='available':
                    means[market]=sum(strength['means'])
                    if market=='goals': goal_means=strength['means']
                else: evidence['fallbacks'].append(strength['reason'])
        if family=='lineup':
            adjustment=cm.lineup_delta(lineup,fixture,cutoff,parameter);evidence['lineup']=adjustment
            if adjustment['status']=='available':
                if any(v is None for v in goal_means) or means['sot'] is None:
                    evidence['fallbacks'].append('base_projection_unavailable')
                else:
                    goal_means=[goal_means[i]+adjustment['delta'][s]['goals'] for i,s in enumerate(('home','away'))]
                    means['goals']=sum(goal_means)
                    means['sot']+=sum(adjustment['delta'][s]['sot'] for s in ('home','away'))
                    if min(goal_means)<0 or means['sot']<0: raise ValueError('Lineup produces negative mean; reject instead of clipping')
            else: evidence['fallbacks'].append(adjustment['reason'])
        if rates_only:
            variances={}
            for market in ('corners','sot'):
                variance=projections[market]['variance']
                if means[market] is not None and means[market]!=projections[market]['value']:
                    variance,_=ms._total_variance(home,away,league,market,means[market],cutoff)
                if family=='dispersion_'+market and means[market] is not None:
                    variance=means[market]+parameter*means[market]**2
                variances[market]=variance
            return {'candidate':spec,'fixture_id':fixture['fixture_id'],
                    'input_snapshot_id':snapshot['snapshot_id'],'goal_means':goal_means,
                    'means':means,'variances':variances,'evidence':evidence,
                    'publication_enabled':False,
                    'status':'control_fallback' if evidence['fallbacks'] else 'computed'}
        changed_goals=goal_means!=[h,a]
        if all(v is not None for v in goal_means):
            if family=='goal_nb': matrix=cm.goal_matrix(*goal_means,alpha=parameter,rho=0.,frozen_probability=prob)
            elif family=='goal_rho' or changed_goals:
                matrix=cm.goal_matrix(*goal_means,rho=parameter if family=='goal_rho' else -.1,frozen_probability=prob)
            else: matrix=None
            if matrix is not None: score={(i,j):value for i,row in enumerate(matrix) for j,value in enumerate(row)}
        distributions={};diagnostics={}
        if score:
            totals=defaultdict(float)
            for (i,j),value in score.items(): totals[i+j]+=value
            mean=sum(k*v for k,v in totals.items())
            distributions['goals']={'pmf':[totals[k] for k in range(max(totals)+1)],'mean':means['goals'],
                'variance':sum((k-mean)**2*v for k,v in totals.items()),'source':'shared_candidate_score_distribution',
                'omitted_mass_bound':1e-10}
            diagnostics['winner']={name:sum(v for (i,j),v in score.items() if condition(i,j))
                for name,condition in [('home',lambda i,j:i>j),('draw',lambda i,j:i==j),('away',lambda i,j:i<j)]}
            diagnostics['btts_yes']=sum(v for (i,j),v in score.items() if i>0 and j>0)
            diagnostics['handicaps']={str(line):ls._score_matrix_spread_profile(home,away,league,True,line,score_probs=score)
                                      for line in (-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.)}
        for market in ('corners','sot'):
            mean=means[market]
            if mean is None: continue
            variance=projections[market]['variance'];source=projections[market]['components'].get('variance_source')
            if mean!=projections[market]['value']:
                variance,details=ms._total_variance(home,away,league,market,mean,cutoff);source=details['source']
            if family=='dispersion_'+market:
                variance=mean+parameter*mean*mean;source='fitted_dispersion_candidate'
            alpha=max(0.,((variance or mean)-mean)/mean**2) if mean>0 else 0.
            distribution=cm.count_pmf(mean,alpha)
            distribution.update(variance=variance,source=source)
            distributions[market]=distribution
        for market,centre in [('goals',2.5),('corners',9.5),('sot',8.5)]:
            if market not in distributions: continue
            d=distributions[market];pmf=dict(enumerate(d['pmf']))
            diagnostics[market]={}
            for line in (centre-1.,centre-.5,centre-.25,centre,centre+.25,centre+.5,centre+1.):
                if market=='goals': outcomes={side:prob.asian_total_profile_from_counts(pmf,line,side) for side in ('Over','Under')}
                else: outcomes={side:prob.asian_total_settlement_profile(d['mean'],line,side,d['variance']) for side in ('Over','Under')}
                diagnostics[market][str(line)]=outcomes
        # Canonical control-path records are clearly separated: replaced-strength
        # or distribution candidates must not present stale records as publications.
        output={'version':cm.VERSION,'candidate':spec,'fixture_id':fixture['fixture_id'],
                'input_snapshot_id':snapshot['snapshot_id'],'goal_means':goal_means,'means':means,
                'score_distribution':[[i,j,v] for (i,j),v in sorted((score or {}).items())],
                'distributions':distributions,'diagnostics':diagnostics,'evidence':evidence,
                'profiles':calculated_profiles,'recent':calculated_recent,
                'projection_path_records':originals,'projection_records_role':'profile_path_before_explicit_output_replacements',
                'publication_enabled':False,'prices':'unavailable','status':'control_fallback' if evidence['fallbacks'] else 'computed'}
        output['id']=cm.digest(output)
        return output


def own_weight_total(p,profiles,recent,home,away,market,weight):
    """Named mean candidate replacing the frozen recent 60/40 constants as well."""
    def calculate(_home,_away,_league,knockout_ctx=None,league_ctx=None,fixture_date=None):
        h,a=getattr(p,'projected_'+market)(profiles[home],profiles[away])
        season=h+a if h is not None and a is not None else None
        hf,af=(recent[t].get(market+'_for_avg') for t in (home,away))
        recent_total=None
        if hf is not None and af is not None:
            ha=recent[away].get(market+'_against_avg');aa=recent[home].get(market+'_against_avg')
            if ha is None: ha=a if a is not None else af
            if aa is None: aa=h if h is not None else hf
            recent_total=weight*(hf+af)+(1-weight)*(ha+aa)
        total=p._blend_total_with_divergence(market,season,recent_total)
        if total is not None:
            total+=p._trend_adjustment(recent[home],recent[away],market+'_for_slope')
            if knockout_ctx and knockout_ctx.is_knockout: total*=getattr(knockout_ctx,market+'_modifier')
            if league_ctx is not None: total*=league_ctx.adjustments.get(market,1.)
        return total,season,recent_total
    return calculate
