"""Independent artifact audit for a completed development comparison, no tuning."""
from collections import defaultdict,Counter
from datetime import timedelta
import gzip
import json
from pathlib import Path
import sys
from Scripts.data_platform.features.phase4_candidates import digest,utc
from Scripts.ops.phase4_baseline import sha,write


def audit(output):
    output=output.resolve()
    history={r['fixture_id']:r for r in map(json.loads,(output/'history.jsonl').open())}
    allowed=json.loads((output/'fit-eligibility.json').read_text())
    allowed={m:set(v) for m,v in allowed.items()}
    bundle_count=0;fit_support=defaultdict(list);failed_fits=[]
    for path in sorted((output/'runs').glob('*/*.bundle.json')):
        b=json.loads(path.read_text());assert b['id']==digest({k:v for k,v in b.items() if k!='id'})
        assert b['converged'] and b['publication_enabled'] is False
        cutoff=utc(b['cutoff']);assert cutoff.year in (2022,2023)
        training=[]
        for fid in b['fixture_ids']:
            r=history[fid]
            assert fid in allowed[b['market']]
            assert r['season'] in (b['season'],b['season']-1) and r['competition']==b['competition']
            assert utc(r['kickoff'])+timedelta(hours=3)<cutoff and utc(r['kickoff']).date()<cutoff.date()
            training.append({k:r[k] for k in ('fixture_id','competition','season','kickoff','home_team_id','away_team_id')}|
                            {'target':[r['home'][b['market']],r['away'][b['market']]]})
        assert digest(training)==b['training_sha256']
        assert len(training)==b['fixture_count']==b['effective_fixture_count']
        assert b['half_life_days'] is None
        bundle_count+=1
        fit_support[b['competition']+':'+b['market']].append({'cutoff':b['cutoff'],'season':b['season'],
            'ridge':b['ridge'],'n':b['fixture_count'],'ess':b['effective_fixture_count'],'id':b['id']})
    metadata={};targets={};membership={};coverage=Counter();contexts=Counter()
    for folder in sorted((output/'folds').iterdir()):
        for r in map(json.loads,(folder/'snapshots.jsonl').open()):
            fid=r['fixture']['fixture_id'];assert fid not in metadata;metadata[fid]=r['fixture'];membership[fid]=folder.name
            for key,value in r['context'].items():contexts[key+':'+value.get('status','unspecified')]+=1
        for r in map(json.loads,(folder/'targets.jsonl').open()):targets[r['fixture_id']]=r
        for r in map(json.loads,(folder/'eligibility.jsonl').open()):
            for m in ('goals','corners','sot'):
                if r['markets'][m]['eligible']:coverage[(folder.name,m)]+=1
    stats={};invalid=Counter();failures=[];control_counts=Counter();year_league=Counter();unchanged=Counter()
    for folder in sorted((output/'runs').iterdir()):
        if not folder.is_dir():continue
        report=json.loads((folder/'report.json').read_text());assert not report['io_violations']
        failed_fits.extend({'fold':folder.name,**r} for r in report['fits'] if 'error' in r)
        seen=set();controls={}
        with gzip.open(folder/'forecasts.jsonl.gz','rt') as stream:
            for line in stream:
                r=json.loads(line);fid=r['fixture_id'];f=metadata[fid]
                assert membership[fid]==r['fold']==folder.name and int(f['kickoff'][:4])==r['year']
                key=(fid,r['candidate_id'],r['market']);assert key not in seen;seen.add(key)
                if r['fit_id']:
                    b=json.loads((folder/(r['fit_id']+'.bundle.json')).read_text())
                    assert utc(b['cutoff'])<=utc(f['kickoff']) and b['market']==r['market']
                if r['status']=='invalid':invalid[r['candidate_id']+':'+r['reason']]+=1;continue
                assert r['scores']['target']==targets[fid]['labels'][r['market']]
                name=r['fold']+':'+r['market']+':'+r['candidate_id']
                entry=stats.setdefault(name,{'n':0,'sum_nll':0.,'fallbacks':0,'changed_vs_control':0})
                entry['n']+=1;entry['sum_nll']+=r['scores']['nll'];entry['fallbacks']+=r['status']=='control_fallback'
                if r['candidate_id']=='control':
                    controls[(fid,r['market'])]=r['scores']['nll'];control_counts[(folder.name,r['market'])]+=1
                    year_league[(str(r['year']),r['market'],r['league'])]+=1
                else:entry['changed_vs_control']+=abs(r['scores']['nll']-controls[(fid,r['market'])])>1e-10
        assert len(seen)==report['records']
    assert control_counts==coverage,(control_counts,coverage)
    for entry in stats.values():entry['mean_nll']=entry.pop('sum_nll')/entry['n']
    result={'all_memberships_and_control_coverage_match':True,'fitted_bundles_verified':bundle_count,
            'fit_support':dict(fit_support),'failed_or_insufficient_fits':failed_fits,'invalid_candidates':dict(invalid),
            'league_contributions':{':'.join(k):v for k,v in year_league.items()},
            'context_coverage':dict(contexts),'fold_metrics':stats,
            'selection_lock_verified':sha(output/'selection.json')==json.loads((output/'SELECTION_LOCK.json').read_text())['selection_sha256']}
    write(output/'independent-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('fit_support','fold_metrics','league_contributions','context_coverage','failed_or_insufficient_fits')}))


if __name__=='__main__':audit(Path(sys.argv[1]))
