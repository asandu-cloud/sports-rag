"""Independent composite identity, numerical parity, scope and reserve-gate audit."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys
from Scripts.data_platform.features import phase4_finalists as f
from Scripts.ops.phase4_baseline import sha,write


def audit(output):
    output=output.resolve();(output/'validation').mkdir(exist_ok=True)
    counts=Counter();maximum=0.;references={};lookups={};seen=set();configurations={}
    for q in range(1,5):
        key=f'2023-Q{q}';folder=output/'folds'/key;run=output/'runs'/key
        refs={(r['fixture_id'],r['market'],r['candidate_id']):r for r in map(json.loads,(folder/'reference.jsonl').open())}
        with gzip.open(run/'fixture-configurations.jsonl.gz','rt') as stream:
            for r in map(json.loads,stream):
                assert r['cross_market_joint_probability'] is None
                identity=f.digest({'snapshot_id':r['snapshot_id'],'configuration':r['configuration'],'version':f.VERSION})
                assert r['id']==identity and identity not in configurations
                configurations[identity]=r
                if r['configuration']!='control':assert r['rates']['candidate']['family']=='composite'
        with gzip.open(run/'forecasts.jsonl.gz','rt') as stream:
            for r in map(json.loads,stream):
                fid=r['fixture_id'];m=r['market'];name=r['configuration'];key=(fid,m,name)
                assert key not in seen;seen.add(key);counts[name+':'+m]+=1
                assert r['active_scope']==f.supported(r['league'],r['season_stage'])
                config=configurations[r['configuration_id']]
                assert config['fixture_id']==fid and config['snapshot_id']==r['snapshot_id'] and config['configuration']==name
                base={'goals':'strength_goals:10.0','corners':'dispersion_corners:0.025','sot':'dispersion_sot:0.01'}[m] if name=='component_stack' or (name=='supported_stack' and r['active_scope']) else 'control'
                reference=refs[(fid,m,base)]
                # Independent exact comparison with the prior sealed run's complete score payload.
                assert r['scores']==reference['scores'];assert r['fit_id']==(reference['fit_id'] if m=='goals' else None)
                if m=='goals':assert config['goal_fit_id']==r['fit_id']
                if base=='control' and name=='supported_stack':assert r['fallbacks']==['unsupported_league_or_season_stage']
                elif name!='control' and m!='goals':assert r['fallbacks']==[]
                scores=r['scores'];totals=scores['totals']
                for v in [*totals.values(),*scores['derived'].get('handicaps',{}).values()]:
                    assert min(v['p'])>=0 and abs(sum(v['p'])-1)<1e-10
                if 'winner' in scores['derived']:assert abs(sum(scores['derived']['winner']['p'])-1)<1e-10
                ps=[totals[str(line)]['binary']['p'] for line in sorted(float(x) for x in totals) if line%1==.5]
                assert ps==sorted(ps,reverse=True)
                if name=='control':references[(fid,m)]=scores
                lookups[key]=r
    for name in ('component_stack','supported_stack'):
        assert {(fid,m) for fid,m,n in seen if n==name}==set(references)
    for fid,m in references:
        r=lookups[(fid,m,'supported_stack')]
        if not r['active_scope']:assert r['scores']==references[(fid,m)]
    finalist=json.loads((output/'finalists.json').read_text());assert finalist['id']==f.digest({k:v for k,v in finalist.items() if k!='id'})
    comparisons=json.loads((output/'comparison.json').read_text())
    for m,v in finalist['components'].items():
        assert v['development_pass']==comparisons['supported_stack:'+m+':active_scope']['passes']
        assert v['leagues']==list(f.LEAGUES) and v['minimum_current_matches_both_teams']==8
    for n,h in json.loads((output/'FINALIST_LOCK.json').read_text()).items():assert sha(output/n)==h
    preflight=json.loads((output/'qualification-preflight.json').read_text())
    expected=f.reserve_preflight(json.loads((output/'split-metadata.json').read_text())['memberships'])
    assert all(preflight[k]==v for k,v in expected.items())
    assert not preflight['reserved_outcomes_read'] and not preflight['final_system_test_opened']
    assert preflight['status']=='blocked_insufficient_metadata_support'
    result={'configuration_identities':len(configurations),'forecasts_replayed_exactly':sum(counts.values()),'counts':dict(counts),
        'maximum_score_difference':0.,'all_common_memberships_identical':True,'all_inactive_scopes_exact_control':True,
        'Asian_normalization_and_line_monotonicity_verified':True,'finalist_lock_verified':True,
        'qualification_preflight_independently_reproduced':True,'reserved_outcomes_read':False}
    write(output/'validation/independent-audit.json',result);print(json.dumps(result))


if __name__=='__main__':audit(Path(sys.argv[1]))
