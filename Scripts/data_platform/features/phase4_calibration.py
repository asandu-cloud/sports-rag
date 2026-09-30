"""Research-only chronological logistic comparison and coherent calibration."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,gammaln
try:
    import phase4_comparison_math as scoring
except ModuleNotFoundError:
    from Scripts.data_platform.features import phase4_backtest as scoring

VERSION='phase4-probability-calibration.v1'
CS=(.01,.1,1.,10.)
TEMPERATURES=(.75,.9,1.,1.1,1.25,1.5)
PROBLEMS=('btts','goals','corners','sot')
FAMILIES={'btts':'strength_goals','goals':'strength_goals','corners':'dispersion_corners','sot':'dispersion_sot'}
END=datetime(2024,1,1,tzinfo=timezone.utc)


def utc(value):
    d=datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('Explicit timezone required')
    return d.astimezone(timezone.utc)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(encode(value).encode()).hexdigest()
def logit(p):
    if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
    p=min(1-1e-12,max(1e-12,p));return math.log(p)-math.log1p(-p)


def preceding(rows,cutoff):
    boundary=utc(cutoff)
    if boundary>END:raise ValueError('Reserved fitting cutoff')
    return sorted([r for r in rows if utc(r['available_at'])<boundary],key=lambda r:(utc(r['kickoff']),r['fixture_id']))


def validate_training(rows,cutoff,problem,base_id):
    boundary=utc(cutoff);seen=set()
    if boundary>END or problem not in PROBLEMS:raise ValueError('Unsupported fitting scope')
    for r in rows:
        if r['problem']!=problem or r['period']!='regulation_time' or r['fixture_id'] in seen:raise ValueError('Mixed/duplicate target identity')
        seen.add(r['fixture_id'])
        if utc(r['available_at'])>=boundary or utc(r['available_at'])<utc(r['kickoff'])+timedelta(hours=3):raise ValueError('Label unavailable at fit cutoff')
        if utc(r['as_of'])>utc(r['kickoff']):raise ValueError('Post-kickoff feature')
        if r['y'] not in (0,1) or base_id not in r['bases']:raise ValueError('Missing qualified input')
    counts=[sum(r['y']==i for r in rows) for i in (0,1)]
    support=scoring.support(rows,True)
    support['class_counts']=counts;support['effective_sample_size']=len(rows)
    if not support['sufficient'] or min(counts)<50:raise ValueError('Insufficient fitting support')
    return support


def numeric(row,kind,base_id):
    base=row['bases'][base_id]
    if kind=='sigmoid':return [logit(base['p'])]
    control=row['bases']['control'];values=[logit(control['p']),control['mean']]
    if row['problem'] in ('goals','btts'):values.append(control['goal_means'][0]-control['goal_means'][1])
    if not all(math.isfinite(x) for x in values):raise ValueError('Missing/nonfinite comparator feature')
    return values


def fit_binary(rows,*,cutoff,problem,kind,C,base_id='control',maxiter=2000):
    if kind not in ('comparator','sigmoid') or C not in CS or (kind=='comparator' and base_id!='control'):raise ValueError('Undeclared binary model')
    rows=sorted(rows,key=lambda r:(utc(r['kickoff']),r['fixture_id']))
    support=validate_training(rows,cutoff,problem,base_id)
    x=np.array([numeric(r,kind,base_id) for r in rows]);means=x.mean(axis=0);scales=x.std(axis=0);scales[scales==0]=1.
    x=(x-means)/scales;leagues=sorted({r['league'] for r in rows}) if kind=='comparator' else []
    if leagues:x=np.column_stack([x,[[float(r['league']==league) for league in leagues] for r in rows]])
    y=np.array([r['y'] for r in rows]);n=len(y)
    def loss(beta):
        z=beta[0]+x@beta[1:];p=expit(z)
        value=np.mean(np.logaddexp(0,z)-y*z)+np.dot(beta[1:],beta[1:])/(2*C*n)
        gradient=np.r_[np.mean(p-y),x.T@(p-y)/n+beta[1:]/(C*n)]
        return float(value),gradient
    initial=np.zeros(x.shape[1]+1);initial[0]=logit(float(y.mean()))
    fitted=minimize(loss,initial,jac=True,method='L-BFGS-B',bounds=[(None,None)]+([(0,None)] if kind=='sigmoid' else [(None,None)]*x.shape[1]),
                    options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-8})
    if not fitted.success or not np.all(np.isfinite(fitted.x)):raise ValueError('Binary fit did not converge')
    bundle={'version':VERSION,'kind':kind,'problem':problem,'base_id':base_id,'C':C,'cutoff':cutoff,
            'means':means.tolist(),'scales':scales.tolist(),'leagues':leagues,'coefficients':fitted.x[1:].tolist(),
            'intercept':float(fitted.x[0]),'support':support,'fixture_ids':[r['fixture_id'] for r in rows],
            'training_hash':digest(rows),'converged':True,'publication_enabled':False,'period':'regulation_time'}
    return {**bundle,'id':digest(bundle)}


def predict_binary(bundle,row,*,end=END):
    if bundle.get('id')!=digest({k:v for k,v in bundle.items() if k!='id'}) or bundle['version']!=VERSION or not bundle['converged']:raise ValueError('Invalid binary bundle')
    if row['problem']!=bundle['problem'] or row['period']!=bundle['period']:raise ValueError('Wrong target scope')
    if utc(row['as_of'])>utc(row['kickoff']):raise ValueError('Post-kickoff forecast refused')
    if utc(row['as_of'])<utc(bundle['cutoff']) or utc(row['as_of'])>=end:raise ValueError('Future bundle or reserved forecast')
    values=(np.array(numeric(row,bundle['kind'],bundle['base_id']))-bundle['means'])/bundle['scales']
    if bundle['leagues']:values=np.r_[values,[float(row['league']==lg) for lg in bundle['leagues']]]
    z=float(bundle['intercept']+values@np.array(bundle['coefficients']))
    return {'p':float(expit(z)),'logit':z,'unknown_league':bool(bundle['leagues'] and row['league'] not in bundle['leagues']),
            'fit_id':bundle['id'],'support':bundle['support'],'calibration_version':VERSION,
            'interpretation':'event_probability_not_value_or_confidence','publication_enabled':False}


def binary_score(p,y,z=None):
    z=logit(p) if z is None else z
    return {'p':p,'y':int(y),'nll':float(np.logaddexp(0,z)-y*z),'brier':float((p-y)**2)}


def choose_base(rows,problem,cutoff,*,earlier_fitting=False):
    if earlier_fitting and not utc('2022-01-01T00:00:00Z')<=utc(cutoff)<utc('2023-01-01T00:00:00Z'):
        raise ValueError('Earlier fitting is for nested 2022 folds only')
    years=range(2019,2023) if earlier_fitting else (2022,)
    rows=[r for r in preceding(rows,cutoff) if r['problem']==problem and utc(r['kickoff']).year in years]
    if not scoring.support(rows,True)['sufficient']:raise ValueError('Insufficient earlier base-selection support')
    ids=sorted(k for k in rows[0]['bases'] if k.startswith(FAMILIES[problem]+':'))
    if not ids:raise ValueError('No base candidates')
    for r in rows:
        if sorted(k for k in r['bases'] if k.startswith(FAMILIES[problem]+':'))!=ids:
            raise ValueError('Incomplete base candidate slate')
        if any(not isinstance(r['bases'][k]['primary_nll'],(int,float)) or not math.isfinite(r['bases'][k]['primary_nll']) for k in ids):
            raise ValueError('Invalid base selection loss')
    losses={k:float(np.mean([r['bases'][k]['primary_nll'] for r in rows])) for k in ids}
    best=min(losses.values());tied=[k for k,v in losses.items() if v<=best+1e-8]
    selected=min(tied,key=lambda k:-float(k.split(':')[1]) if FAMILIES[problem].startswith('strength') else float(k.split(':')[1]))
    return {'base_id':selected,'cutoff':cutoff,'fixture_ids':[r['fixture_id'] for r in rows],'losses':losses,'input_hash':digest(rows)}


def powered_goals(home,away,temperature,probability,rho=-.1):
    if temperature not in TEMPERATURES:raise ValueError('Undeclared temperature')
    probability._validate_dixon_coles(home,away,rho)
    if max(home,away)>100:raise ValueError('Mean exceeds bounded calibration grid')
    power=1/temperature
    def marginal(mu,n):
        k=np.arange(n+1)
        logs=k*math.log(mu)-mu-gammaln(k+1) if mu>0 else np.r_[0.,np.full(n,-np.inf)]
        values=np.exp(logs*power)
        if mu==0:return values,0.
        ratio=(mu/(n+2))**power
        tail=math.exp(((n+1)*math.log(mu)-mu-math.lgamma(n+2))*power)/(1-ratio) if ratio<1 else math.inf
        return values,tail
    n=max(8,math.ceil(max(home,away))+2)
    while n<=500:
        h,htail=marginal(home,n);a,atail=marginal(away,n);matrix=np.outer(h,a)
        for i in (0,1):
            for j in (0,1):matrix[i,j]*=probability._tau(i,j,home,away,rho)**power
        normalizer=float(matrix.sum());tail=htail*float(a.sum())+atail*float(h.sum())+htail*atail
        if normalizer>0 and tail/normalizer<=1e-10:
            matrix/=normalizer
            return {'matrix':matrix,'normalizer':normalizer,'tail_bound':tail/normalizer,'extent':n,
                    'temperature':temperature,'goal_means':[home,away],'rho':rho}
        n=max(n+1,int(n*1.4))
    raise ValueError('Transformed goal tail exceeds bound')


def score_joint(distribution,team_target,primary_only=False):
    matrix=distribution['matrix'];h,a=distribution['goal_means'];T=distribution['temperature'];rho=distribution['rho']
    yh,ya=map(int,team_target);y=yh+ya
    tau=(1-h*a*rho if (yh,ya)==(0,0) else 1+a*rho if (yh,ya)==(1,0) else 1+h*rho if (yh,ya)==(0,1) else 1-rho if (yh,ya)==(1,1) else 1.)
    def logpmf(mu,k):return k*math.log(mu)-mu-math.lgamma(k+1) if mu else (0. if k==0 else -math.inf)
    lp=(logpmf(h,yh)+logpmf(a,ya)+math.log(tau))/T-math.log(distribution['normalizer']) if tau>0 else -math.inf
    if not math.isfinite(lp):raise ValueError('Invalid calibrated score probability')
    if primary_only:return {'nll':-lp,'distribution':{k:v for k,v in distribution.items() if k!='matrix'}}
    ii,jj=np.indices(matrix.shape);pmf=np.bincount((ii+jj).ravel(),matrix.ravel());mean=float(np.dot(np.arange(len(pmf)),pmf))
    cdf=pmf.cumsum();quantiles=[int(np.searchsorted(cdf,q)) for q in (.025,.1,.9,.975)]
    winner=[float(matrix[ii>jj].sum()),float(matrix[ii==jj].sum()),float(matrix[ii<jj].sum())];actual=0 if yh>ya else 1 if yh==ya else 2
    derived={'btts':scoring.binary(float(matrix[1:,1:].sum()),yh>0 and ya>0),
             'winner':{'p':winner,'y':actual,'nll':-math.log(max(winner[actual],1e-15)),'brier':sum((p-(i==actual))**2 for i,p in enumerate(winner))},'handicaps':{}}
    for line in (-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.):
        p=scoring.profile((ii-jj).ravel(),matrix.ravel(),-line);actual=scoring.outcome_class(yh-ya,-line)
        derived['handicaps'][str(line)]={'p':p,'y':actual,'nll':-math.log(max(p[actual],1e-15))}
    totals={}
    for line in scoring.LINES['goals']:
        p=scoring.profile(range(len(pmf)),pmf,line);actual=scoring.outcome_class(y,line)
        totals[str(line)]={'p':p,'y':actual,'nll':-math.log(max(p[actual],1e-15))}
        if line%1==.5:totals[str(line)]['binary']=scoring.binary(p[0],y>line)
    return {'nll':-lp,'error':mean-y,'absolute_error':abs(mean-y),'squared_error':(mean-y)**2,'target':y,
            'rps':float(np.sum((cdf-(np.arange(len(pmf))>=y))**2)+max(0,y-len(pmf))),
            'interval80':quantiles[1:3],'interval95':[quantiles[0],quantiles[3]],
            'coverage80':int(quantiles[1]<=y<=quantiles[2]),'coverage95':int(quantiles[0]<=y<=quantiles[3]),
            'totals':totals,'derived':derived,'distribution':{k:v for k,v in distribution.items() if k!='matrix'}}


def binary_metrics(rows):
    result=scoring.support(rows)
    result.update(nll=float(np.mean([r['scores']['nll'] for r in rows])),brier=float(np.mean([r['scores']['brier'] for r in rows])),
                  reliability=scoring.reliability([r['scores'] for r in rows]))
    return result


def compare_binary(reference,candidate):
    if [r['fixture_id'] for r in reference]!=[r['fixture_id'] for r in candidate]:raise ValueError('Unpaired binary cohorts')
    a,b=binary_metrics(reference),binary_metrics(candidate)
    interval=scoring.week_interval(reference,[y['scores']['nll']-x['scores']['nll'] for x,y in zip(reference,candidate)])
    slices={};regressions=[]
    for key in ('league','season','season_stage','forecast_stage','missingness'):
        slices[key]={}
        for value in sorted({str(r[key]) for r in reference}):
            indices=[i for i,r in enumerate(reference) if str(r[key])==value]
            aa=float(np.mean([reference[i]['scores']['nll'] for i in indices]));bb=float(np.mean([candidate[i]['scores']['nll'] for i in indices]))
            support=scoring.support([reference[i] for i in indices]);change=bb/aa-1
            slices[key][value]={'support':support,'reference_nll':aa,'candidate_nll':bb,'relative_change':change}
            if support['sufficient'] and change>.02:regressions.append(key+':'+value)
    relative=b['nll']/a['nll']-1
    return {'reference':a,'candidate':b,'relative_change':relative,'paired_week_delta95':interval,
            'supported_slice_regressions':regressions,'slices':slices,
            'passes':a['sufficient'] and relative<=-.005 and interval[1]<0 and not regressions}
