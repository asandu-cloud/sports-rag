"""Development-only Phase 4 probability scoring and paired comparison.

No IO or model fitting; forecast generation and outcome scoring are separate.
"""
from collections import defaultdict
from datetime import datetime
import math
import numpy as np
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize

VERSION='phase4-forward-comparison.v1'
SEED=20260928
MARKETS=('goals','corners','sot')
LINES={'goals':(1.5,2.,2.25,2.5,2.75,3.,3.5),'corners':(8.5,9.,9.25,9.5,9.75,10.,10.5),'sot':(7.5,8.,8.25,8.5,8.75,9.,9.5)}
OUTCOMES=('full_win','half_win','push','half_loss','full_loss')


def quarter(kickoff):
    t=datetime.fromisoformat(kickoff.replace('Z','+00:00'))
    if t.tzinfo is None or t.year not in (2022,2023): raise ValueError('Development dates only')
    return f'{t.year}-Q{(t.month-1)//3+1}'


def quarter_cutoff(key):
    year,q=key.split('-Q')
    if year not in ('2022','2023') or q not in ('1','2','3','4'): raise ValueError('Unsupported fold')
    return f'{year}-{1+(int(q)-1)*3:02d}-01T00:00:00+00:00'


def support(rows, fitting=False):
    n=len(rows);weeks=len({r['week'] for r in rows})
    return {'n':n,'weeks':weeks,'sufficient':n>=(500 if fitting else 200) and weeks>=(26 if fitting else 20)}


def outcome_class(value,line):
    """Five-way settlement index for an Over total or home handicap margin."""
    lo=math.floor(line*2)/2; hi=math.ceil(line*2)/2
    score=((value>lo)-(value<lo)+(value>hi)-(value<hi))/2
    return {1:0,.5:1,0:2,-.5:3,-1:4}[score]


def profile(values,probabilities,line):
    result=np.zeros(5)
    for x,p in zip(values,probabilities):result[outcome_class(float(x),line)]+=p
    return result.tolist()


def binary(p,y):
    # Finite numerical floor only for secondary binary diagnostics, disclosed.
    p=min(1-1e-15,max(1e-15,float(p)))
    return {'p':p,'y':int(y),'logloss':-math.log(p if y else 1-p),'brier':(p-y)**2}


def score_forecast(rates,spec,market,target,maths,probability):
    """Exact primary density; bounded PMFs for intervals and derived outcomes."""
    family=spec['family'];parameter=spec['parameter'];y=int(target['labels'][market])
    derived={}
    if market=='goals':
        h,a=rates['goal_means'];yh,ya=(int(target['team_labels'][s]['goals']) for s in ('home','away'))
        alpha=parameter if family=='goal_nb' else None;rho=parameter if family=='goal_rho' else -.1
        matrix=np.asarray(maths.goal_matrix(h,a,alpha=alpha,rho=0. if alpha is not None else rho,frozen_probability=probability))
        if alpha is None:
            p=probability.dixon_coles_scoreline_prob(yh,ya,h,a,rho=rho)
            nll=-math.log(p) if p>0 else math.inf
        else:
            nll=0.
            for mu,yy in ((h,yh),(a,ya)):
                d=poisson(mu) if alpha==0 else nbinom(1/alpha,1/(1+alpha*mu))
                nll-=float(d.logpmf(yy))
        ii,jj=np.indices(matrix.shape);pmf=np.bincount((ii+jj).ravel(),matrix.ravel())
        mean=h+a
        derived['btts']=binary(float(matrix[1:,1:].sum()),yh>0 and ya>0)
        winners=[float(matrix[ii>jj].sum()),float(matrix[ii==jj].sum()),float(matrix[ii<jj].sum())]
        winner=0 if yh>ya else 1 if yh==ya else 2
        derived['winner']={'p':winners,'y':winner,'nll':-math.log(max(winners[winner],1e-15)),
                           'brier':sum((p-(i==winner))**2 for i,p in enumerate(winners))}
        derived['handicaps']={}
        for line in (-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.):
            p=profile((ii-jj).ravel(),matrix.ravel(),-line);actual=outcome_class(yh-ya,-line)
            derived['handicaps'][str(line)]={'p':p,'y':actual,'nll':-math.log(max(p[actual],1e-15))}
        distribution={'kind':'independent_nb' if alpha is not None else 'dixon_coles','goal_means':[h,a],
                      'alpha':alpha,'rho':0. if alpha is not None else rho}
    else:
        mean=rates['means'][market];variance=rates['variances'][market]
        if family=='dispersion_'+market: variance=mean+parameter*mean*mean
        alpha=max(0.,((variance or mean)-mean)/mean**2)
        pmf=np.asarray(maths.count_pmf(mean,alpha)['pmf'])
        d=poisson(mean) if alpha==0 else nbinom(1/alpha,1/(1+alpha*mean))
        nll=-float(d.logpmf(y));distribution={'kind':'count_nb','mean':mean,'variance':variance,'alpha':alpha}
    if not math.isfinite(nll) or np.any(pmf<0) or abs(float(pmf.sum())-1)>1e-10:
        raise ValueError('Invalid primary probability/PMF')
    cdf=pmf.cumsum();quantiles=[int(np.searchsorted(cdf,q)) for q in (.025,.1,.9,.975)]
    target_cdf=(np.arange(len(pmf))>=y).astype(float)
    rps=float(np.sum((cdf-target_cdf)**2)+max(0,y-len(pmf)))
    totals={}
    for line in LINES[market]:
        pp=profile(range(len(pmf)),pmf,line);actual=outcome_class(y,line)
        totals[str(line)]={'p':pp,'y':actual,'nll':-math.log(max(pp[actual],1e-15))}
        if line%1==.5:totals[str(line)]['binary']=binary(pp[0],y>line)
    return {'nll':nll,'error':mean-y,'absolute_error':abs(mean-y),'squared_error':(mean-y)**2,'rps':rps,
            'interval80':[quantiles[1],quantiles[2]],'interval95':[quantiles[0],quantiles[3]],
            'coverage80':int(quantiles[1]<=y<=quantiles[2]),'coverage95':int(quantiles[0]<=y<=quantiles[3]),
            'target':y,'distribution':distribution,'totals':totals,'derived':derived}


def reliability(records):
    p=np.array([r['p'] for r in records]);y=np.array([r['y'] for r in records]);bins=[]
    for i in range(10):
        selected=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        bins.append({'lower':i/10,'n':int(selected.sum()),'p':float(p[selected].mean()) if selected.any() else None,
                     'frequency':float(y[selected].mean()) if selected.any() else None})
    slope=intercept=None
    if len(set(y))==2:
        x=np.log(np.clip(p,1e-12,1-1e-12)/np.clip(1-p,1e-12,1))
        def loss(beta):
            z=beta[0]+beta[1]*x
            return float(np.mean(np.logaddexp(0,z)-y*z))
        fitted=minimize(loss,[0.,1.],method='BFGS')
        if fitted.success:intercept,slope=map(float,fitted.x)
    return {'bins':bins,'calibration_intercept':intercept,'calibration_slope':slope,
            'calibration_role':'evaluation_diagnostic_only_not_applied','n':len(records),
            'positive':int(y.sum()),'negative':int(len(y)-y.sum())}


def metrics(rows, diagnostics=False):
    result=support(rows)
    if not rows:return result
    values=[r['scores'] for r in rows]
    for name in ('nll','absolute_error','error','rps','coverage80','coverage95'):
        result[name]=float(np.mean([v[name] for v in values]))
    result['rmse']=math.sqrt(np.mean([v['squared_error'] for v in values]))
    yy=np.array([v['target'] for v in values]);ss=float(np.sum((yy-yy.mean())**2))
    result['r2']=1-sum(v['squared_error'] for v in values)/ss if ss else None
    for q in (80,95):result['width'+str(q)]=float(np.mean([v['interval'+str(q)][1]-v['interval'+str(q)][0] for v in values]))
    for key in ('totals','derived'):
        result[key]={}
    for line in values[0]['totals']:
        vs=[v['totals'][line] for v in values]
        r={'nll':float(np.mean([v['nll'] for v in vs])),
           'class_counts':[sum(v['y']==k for v in vs) for k in range(5)]}
        if 'binary' in vs[0]:
            bs=[v['binary'] for v in vs]
            r.update(logloss=float(np.mean([v['logloss'] for v in bs])),brier=float(np.mean([v['brier'] for v in bs])))
            if diagnostics:r['reliability']=reliability(bs)
        result['totals'][line]=r
    if 'btts' in values[0]['derived']:
        bs=[v['derived']['btts'] for v in values]
        result['derived']['btts']={k:float(np.mean([b[k] for b in bs])) for k in ('logloss','brier')}
        if diagnostics:result['derived']['btts']['reliability']=reliability(bs)
        result['derived']['winner']={k:float(np.mean([v['derived']['winner'][k] for v in values])) for k in ('nll','brier')}
        result['derived']['winner']['class_counts']=[sum(v['derived']['winner']['y']==i for v in values) for i in range(3)]
        result['derived']['handicaps']={line:{'nll':float(np.mean([v['derived']['handicaps'][line]['nll'] for v in values])),
            'class_counts':[sum(v['derived']['handicaps'][line]['y']==i for v in values) for i in range(5)]}
            for line in values[0]['derived']['handicaps']}
    return result


def week_interval(rows,deltas):
    weeks=sorted({r['week'] for r in rows});groups={w:i for i,w in enumerate(weeks)}
    sums=np.zeros(len(weeks));counts=np.zeros(len(weeks))
    for r,d in zip(rows,deltas):i=groups[r['week']];sums[i]+=d;counts[i]+=1
    draws=np.random.default_rng(SEED).integers(0,len(weeks),size=(2000,len(weeks)))
    samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def compare(reference,candidate,diagnostics=False):
    if [r['fixture_id'] for r in reference]!=[r['fixture_id'] for r in candidate]:raise ValueError('Unpaired fixtures')
    a,b=metrics(reference,diagnostics),metrics(candidate,diagnostics)
    if not reference:return {'control':a,'candidate':b,'passes':False}
    delta=np.array([y['scores']['nll']-x['scores']['nll'] for x,y in zip(reference,candidate)])
    interval=week_interval(reference,delta);regressions=[];slices={}
    for key in ('league','season','season_stage','forecast_stage','missingness'):
        slices[key]={}
        for value in sorted({str(r[key]) for r in reference}):
            ids=[i for i,r in enumerate(reference) if str(r[key])==value]
            ar,br=([reference[i] for i in ids],[candidate[i] for i in ids])
            ma,mb=metrics(ar),metrics(br);change=mb['nll']/ma['nll']-1
            slices[key][value]={'support':support(ar),'control_nll':ma['nll'],'candidate_nll':mb['nll'],'relative_change':change}
            if ma['sufficient'] and change>.02:regressions.append(key+':'+value)
    related=[]
    if a['sufficient']:
        for line,r in a['totals'].items():
            if b['totals'][line]['nll']>1.02*r['nll']:related.append('total:'+line)
        for name in ('btts','winner'):
            if name in a['derived']:
                key='logloss' if name=='btts' else 'nll'
                if b['derived'][name][key]>1.02*a['derived'][name][key]:related.append(name)
        for line,r in a['derived'].get('handicaps',{}).items():
            if b['derived']['handicaps'][line]['nll']>1.02*r['nll']:related.append('handicap:'+line)
    relative=b['nll']/a['nll']-1
    return {'control':a,'candidate':b,'relative_change':relative,'paired_week_delta95':interval,
            'supported_slice_regressions':regressions,'related_market_regressions':related,'slices':slices,
            'passes':a['sufficient'] and relative<=-.005 and interval[1]<0 and not regressions and not related}


def select_settings(rows,specs):
    """Only 2022 OOF rows can select parameters; never discard invalid members."""
    if any(r['year']!=2022 for r in rows):raise ValueError('Selection requires 2022 only')
    by=defaultdict(list)
    for r in rows:by[(r['market'],r['candidate_id'])].append(r)
    selected={};table={}
    for market in MARKETS:
        control=by[(market,'control')];ids={r['fixture_id'] for r in control}
        for family in sorted({s['family'] for s in specs if market in s['markets'] and s['family'] not in ('control','lineup')}):
            valid=[];trials=[]
            for spec in specs:
                if spec['family']!=family:continue
                cohort=by.get((market,spec['id']),[])
                complete={r['fixture_id'] for r in cohort}==ids and len(cohort)==len(ids)
                good=complete and all(r['status']!='invalid' for r in cohort) and support(cohort,True)['sufficient']
                score=float(np.mean([r['scores']['nll'] for r in cohort])) if good else None
                trials.append({'id':spec['id'],'nll':score,'complete':complete,'support':support(cohort,True),
                               'invalid':sum(r['status']=='invalid' for r in cohort)})
                if good:valid.append((score,spec))
            key=market+':'+family;table[key]=trials
            if valid:
                best=min(v[0] for v in valid)
                tied=[s for loss,s in valid if loss<=best+1e-8]
                def tie(s):
                    p=s['parameter']
                    if family.startswith('strength_'):return -p
                    if family=='goal_rho':return abs(p+.1)
                    return abs(p or 0)
                selected[key]=min(tied,key=tie)['id']
            else:selected[key]=None
    return {'selected':selected,'grid':table,'selection_year':2022,'evaluation_year':2023,'publication_enabled':False}
