"""Focused development assessment and isolated dated player effects."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from Scripts.ops import phase4_weight_comparison as old
from Scripts.data_platform.features import phase4_player_context as player
from Scripts.data_platform.features.benchmarks.artifacts import read_json, write_json, sha, complete, verify_complete
from Scripts.data_platform.features.benchmarks.isolation import offline_guard

ROOT = old.ROOT
WEIGHTS = ROOT/'Research/phase4-weight-comparison-2026-10-02-v4'
PACKAGE = ROOT/'Index/prediction_experiments/player-history-prepared-2026-10-02'
PROTOCOL = 'docs/phase4-assessment-player-protocol-2026-10-02.md'
OWN = [PROTOCOL, 'Scripts/ops/phase4_assessment.py', 'Scripts/data_platform/features/phase4_player_context.py',
       'Scripts/tests/test_phase4_player_context.py']
END = old.w.cm.END


def jsonl(path, records):
    with gzip.open(path, 'wt') as h:
        for r in records: h.write(old.w.cm.canonical(r)+'\n')


def prepare(out):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    verify_complete(WEIGHTS); verify_complete(old.STAT); verify_complete(old.CARD)
    seal = read_json(PACKAGE/'PREPARED.json')
    for p in ('platform.db', 'archives.json', 'fixture-identities.json'):
        if sha(PACKAGE/p) != seal[p]: raise ValueError('Prepared source changed:'+p)
    preservation = {p: sha(ROOT/p) for p in read_json(WEIGHTS/'PRESERVATION_BEFORE.json')}
    write_json(out/'PRESERVATION_BEFORE.json', preservation)
    for name in sorted(set(OWN+old.SOURCES)):
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
    write_json(out/'PROTOCOL_LOCK.json', {'sources': {p:sha(ROOT/p) for p in sorted(set(OWN+old.SOURCES))},
               'weight_experiment':sha(WEIGHTS/'COMPLETE.json'), 'player_source_db':seal['platform.db'],
               'protocol':sha(ROOT/PROTOCOL), 'no_public_activation':True,'end_exclusive':END.isoformat()})
    # Certified regulation fixture identities constrain SQL before any player body is decoded.
    historical = {r['fixture_id']: r for r in old.rows(old.STAT/'history.jsonl')
                  if r['competition'] in old.w.scope.LEAGUES}
    archives = {r['id']: r for r in read_json(PACKAGE/'archives.json')}
    checkpoints = {}
    with sqlite3.connect((PACKAGE/'platform.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for fid,ep,state,aid in db.execute("""SELECT f.api_football_id,c.endpoint,c.state,c.archive_id
            FROM fixture_endpoint_collection c JOIN fixtures f ON f.id=c.fixture_id
            WHERE julianday(f.kickoff_utc,'+3 hours')<julianday('2024-01-01')
            AND c.endpoint IN ('/fixtures/players','/fixtures/lineups')"""):
            if fid in historical: checkpoints[fid,ep] = (state,aid)
    extracted=[]; coverage=Counter(); groups=defaultdict(Counter); references={}
    for fid,f in sorted(historical.items()):
        payloads={}; refs={}
        for ep in ('/fixtures/players','/fixtures/lineups'):
            checkpoint=checkpoints.get((fid,ep))
            if not checkpoint or checkpoint[0]!='ready': payloads[ep]=None; continue
            archive=archives[checkpoint[1]]; path=PACKAGE/archive['path']
            if path.is_symlink() or not path.resolve().is_relative_to(PACKAGE/'raw_archive'): raise ValueError('Unsafe player archive')
            if sha(path)!=archive['sha256']: raise ValueError('Changed raw archive')
            body=gzip.decompress(path.read_bytes())
            if hashlib.sha256(body).hexdigest()!=archive['payload_sha256']: raise ValueError('Changed raw payload')
            obj=json.loads(body)
            if obj['get']!=ep.lstrip('/') or str(obj['parameters']['fixture'])!=str(fid) or obj.get('errors'): raise ValueError('Envelope mismatch')
            payloads[ep]=obj; refs[ep]={'archive_id':checkpoint[1],'sha256':archive['sha256'],'payload_sha256':archive['payload_sha256']}
        r=player.normalize(f,payloads['/fixtures/players'],payloads['/fixtures/lineups'])
        extracted.append(r); references[str(fid)]=refs
        for name,good in (('player_history',r['players'] is not None),('actual_xi',r['lineups'] is not None)):
            key=name+(':available' if good else ':unavailable'); coverage[key]+=1; groups[f['competition']+':'+str(f['season'])][key]+=1
        coverage.update(r['reasons'])
    if sha(PACKAGE/'platform.db')!=seal['platform.db']: raise ValueError('Player database changed during extraction')
    jsonl(out/'player-history.jsonl.gz',extracted)
    write_json(out/'player-source-references.json', references)
    write_json(out/'coverage.json',{'counts':dict(coverage),'by_league_season':{k:dict(v) for k,v in groups.items()},
        'original_announcement_timestamps':0,'api_calls':0,'reserved_outcomes_decoded':0,'availability':'assumed_final'})
    write_json(out/'PREPARED.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(read_json(out/'coverage.json')['counts'],indent=2),flush=True)


def check(out):
    for p,h in read_json(out/'PREPARED.json').items():
        if sha(out/p)!=h: raise ValueError('Prepared input changed:'+p)
    for p,h in read_json(out/'PROTOCOL_LOCK.json')['sources'].items():
        if sha(ROOT/p)!=h: raise ValueError('Source changed:'+p)


def load(year,market):
    meta=list(old.rows(WEIGHTS/str(year)/(market+'-membership.jsonl')))
    with np.load(WEIGHTS/str(year)/(market+'-predictions.npz'),allow_pickle=False) as h: arrays=dict(h)
    return meta,arrays


def forecasts(meta, data, market, means):
    array=np.zeros((len(meta),1,2)); v=np.asarray(means)
    if v.ndim==1: array[:,0,0]=v
    else: array[:,0,:v.shape[1]]=v
    d={**data,'means':array,'alpha':data['alpha'][:,:1]}
    return old.detailed(meta,d,0,market,old.frozen_probability())


def compare(reference,candidate,market):
    if market=='cards':
        paired=[{**a,'scores':{'control':a['scores']['value'],'candidate':b['scores']['value']}} for a,b in zip(reference,candidate)]
        return old.w.cards.compare(paired,'candidate','control')
    return old.scoring.compare(reference,candidate,diagnostics=True)


def simple(f,market,history):
    league,season=f['competition'],f['season']; cutoff=old.w.cm.utc(f['as_of'])
    past=[r for r in history[league] if r['season'] in (season-1,season)
          and old.w.cm.utc(r['kickoff']).date()<cutoff.date() and old.w.cm.utc(r['kickoff'])+old.w.timedelta(hours=3)<cutoff]
    byside={s:[r[s].get(market) for r in past if r[s].get(market) is not None] for s in ('home','away')}
    if min(map(len,byside.values()))<20: raise ValueError('Simple league mean lacks support')
    lm={s:float(np.mean(v)) for s,v in byside.items()}; team={}
    for side,other in (('home','away'),('away','home')):
        tid=f[side+'_team_id']; values=[]; allowed=[]
        for r in past:
            s=next((s for s in ('home','away') if r[s+'_team_id']==tid),None)
            if s:
                v=r['away' if s=='home' else 'home'].get(market)
                if v is not None: allowed.append(v)
                v=r[s].get(market)
                if v is not None: values.append(v)
        team[side]={'own':float(np.mean(values)) if len(values)>=8 else lm[side],
                    'against':float(np.mean(allowed)) if len(allowed)>=8 else lm[other]}
    tm=[.5*(team[s]['own']+team[o]['against']) for s,o in (('home','away'),('away','home'))]
    lines={'goals':2.5,'corners':9.5,'sot':8.5}; known=[r for r in past if all(r[s].get(market) is not None for s in ('home','away'))]
    empirical={'over':(1+sum(sum(r[s][market] for s in ('home','away'))>lines[market] for r in known))/(len(known)+2)}
    if market=='goals': empirical['btts']=(1+sum(r['home']['goals']>0 and r['away']['goals']>0 for r in known))/(len(known)+2)
    return {'league':list(lm.values()) if market=='goals' else sum(lm.values()),
            'team':tm if market=='goals' else sum(tm),'empirical':empirical,'source_fixture_ids':[r['fixture_id'] for r in past]}


def assessment(out):
    hist=defaultdict(list)
    for r in old.rows(old.STAT/'history.jsonl'): hist[r['competition']].append(r)
    fixtures={r['fixture']['fixture_id']:r['fixture']|{'as_of':r['as_of']} for r in old.rows(WEIGHTS/'inputs-2023.jsonl.gz')}
    cards={r['fixture_id']:r for r in old.rows(old.CARD/'features.jsonl')}
    selections=read_json(WEIGHTS/'selection.json'); results={}; saved=[]
    for market in old.w.MARKETS:
        meta,d=load(2023,market); baseline=[]
        for r in meta:
            if market!='cards': b=simple(fixtures[r['fixture_id']],market,hist)
            else:
                f=cards[r['fixture_id']]; past=[cards[fid] for fid in f['inputs']['source_fixture_ids']]
                if len(past)<20: raise ValueError('Card league mean lacks support')
                means={s:{k:float(np.mean([t[k] for t in f['inputs']['teams'][s]])) for k in ('own','induced')} for s in ('home','away')}
                b={'league':float(np.mean([v['target'] for v in past])),
                   'team':sum(.5*(means[s]['own']+means[o]['induced']) for s,o in (('home','away'),('away','home'))),
                   'empirical':{'over':(1+sum(v['target']>4.5 for v in past))/(len(past)+2)},
                   'source_fixture_ids':[v['fixture_id'] for v in past]}
            baseline.append(b)
        methods={'reference':forecasts(meta,d,market,d['means'][:,0,:]),
                 'frozen_weight_setup':forecasts(meta,d,market,d['means'][:,selections[market]['index'],:]),
                 **{kind:forecasts(meta,d,market,[b[kind] for b in baseline]) for kind in ('league','team')}}
        results[market]={'vs_reference':{k:compare(methods['reference'],v,market) for k,v in methods.items() if k!='reference'},
                         'reference_vs_league':compare(methods['league'],methods['reference'],market),
                         'reference_vs_team':compare(methods['team'],methods['reference'],market)}
        binary={}; lines={'goals':'2.5','corners':'9.5','sot':'8.5','cards':'4.5'}
        for event in baseline[0]['empirical']:
            metrics={}
            for method,rr in methods.items():
                vals=[]
                for i,r in enumerate(rr):
                    if market=='cards': v=r['scores']['value']['diagnostics'][lines[market]]; value={'p':v['p'],'y':v['y']}
                    elif event=='btts': value=r['scores']['derived']['btts']
                    else: value=r['scores']['totals'][lines[market]]['binary']
                    vals.append(old.scoring.binary(value['p'],value['y']))
                metrics[method]={'logloss':float(np.mean([v['logloss'] for v in vals])), 'brier':float(np.mean([v['brier'] for v in vals])),
                                 'reliability':old.scoring.reliability(vals)}
                if method=='reference':
                    naive=[old.scoring.binary(baseline[i]['empirical'][event],v['y']) for i,v in enumerate(vals)]
                    metrics['empirical_league']={'logloss':float(np.mean([v['logloss'] for v in naive])), 'brier':float(np.mean([v['brier'] for v in naive])),
                                                'reliability':old.scoring.reliability(naive)}
            binary[event]=metrics
        results[market]['binary']=binary
        for i,r in enumerate(meta): saved.append({**r,'market':market,'baseline_evidence':baseline[i],
            'scores':{k:v[i]['scores'] for k,v in methods.items()}})
        print(market+': simple-baseline assessment complete.',flush=True)
    jsonl(out/'assessment-predictions.jsonl.gz',saved)
    write_json(out/'assessment.json',results)


def player_experiment(out):
    history=player.History(list(old.rows(out/'player-history.jsonl.gz')))
    bundles={}; results={}; fits={}; prediction_rows=[]; availability={}
    for year in (2022,2023):
        fixtures={r['fixture']['fixture_id']:r for r in old.rows(WEIGHTS/f'inputs-{year}.jsonl.gz')}
        features={stage:{} for stage in player.STAGES}
        for stage in player.STAGES:
            for fid,f in fixtures.items(): features[stage][fid]=history.build(f['fixture'],f['as_of'],stage)
            jsonl(out/f'player-features-{year}-{stage}.jsonl.gz',features[stage].values())
            availability[str(year)+':'+stage]={'total':len(fixtures),'available':sum(f['status']=='available' for f in features[stage].values()),
                'by_market':{m:sum(player.available(f,m) for f in features[stage].values()) for m in player.MARKETS},
                'fallbacks':dict(Counter(v for f in features[stage].values() for v in f['fallbacks']))}
        for market in player.MARKETS:
            meta,d=load(year,market)
            for stage in player.STAGES:
                key=market+':'+stage
                if year==2022:
                    records=[]
                    for i,r in enumerate(meta):
                        f=features[stage][r['fixture_id']]
                        records.append({**r,'available':player.available(f,market),'feature_id':f['id'],'x':player.vector(f,market),
                            'base':d['means'][i,0,:].tolist() if market=='goals' else [float(d['means'][i,0,0])],
                            'target':d['team_targets'][i].tolist() if market=='goals' else [float(d['targets'][i])]})
                    bundles[key]=player.fit(records,market,stage); fits[key]=records
                else:
                    means=[]
                    for i,r in enumerate(meta):
                        f=features[stage][r['fixture_id']]; base=d['means'][i,0,:].tolist() if market=='goals' else [float(d['means'][i,0,0])]
                        means.append(player.apply(bundles[key],f,base))
                    reference=forecasts(meta,d,market,d['means'][:,0,:]); candidate=forecasts(meta,d,market,means)
                    result=compare(reference,candidate,market)
                    delta=np.array([b['scores']['nll']-a['scores']['nll'] for a,b in zip(reference,candidate)])
                    result.update(paired=old.w.paired(meta,delta),coefficient=bundles[key]['coefficient'],fit_status=bundles[key]['status'],
                                  conditional_only=stage=='conditional_actual_xi',production_qualified=False)
                    supported=[i for i,r in enumerate(meta) if player.available(features[stage][r['fixture_id']],market)]
                    result['available_only']=compare([reference[i] for i in supported],[candidate[i] for i in supported],market) if supported else None
                    results[key]=result
                    for i,r in enumerate(meta): prediction_rows.append({**r,'market':market,'stage':stage,'feature_id':features[stage][r['fixture_id']]['id'],
                        'status':'available' if player.available(features[stage][r['fixture_id']],market) else 'control_fallback','base_means':d['means'][i,0,:].tolist(),'adjusted_means':means[i],
                        'scores':candidate[i]['scores'],'control_scores':reference[i]['scores']})
                    print(key+': fitted effect comparison complete.',flush=True)
        if year==2022:
            write_json(out/'PLAYER_METHOD_LOCK.json',bundles); jsonl(out/'player-fit-records.jsonl.gz',[{'method':k,**r} for k,rs in fits.items() for r in rs])
            print('Four player-effect bundles frozen before 2023 inference.',flush=True)
    correction=old.w.holm({m:results[m+':expected_players']['paired']['p'] for m in player.MARKETS})
    for m in player.MARKETS:
        r=results[m+':expected_players']; r['holm_adjusted_p']=correction[m]
        r['development_screen_passed']=bool(r['passes'] and r['paired']['interval95'][1]<0 and correction[m]<=.05)
    jsonl(out/'player-predictions.jsonl.gz',prediction_rows)
    write_json(out/'player-comparison.json',results); write_json(out/'player-availability.json',availability)


def run(out):
    check(out)
    with offline_guard(root=ROOT,output=out):
        assessment(out)
        player_experiment(out)
    print('Assessment and player comparisons complete; no production change.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','verify'));p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); out=a.output.resolve()
    if a.action=='prepare':prepare(out)
    elif a.action=='run':run(out)
    else:verify_complete(out);print('Complete artifact hashes verified.')


if __name__=='__main__':main()
