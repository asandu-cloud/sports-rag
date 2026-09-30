"""Independent chronology, preprocessing and prediction replay audit; no tuning."""
import gzip
import json
from pathlib import Path
import sys
from itertools import zip_longest
import numpy as np
from Scripts.data_platform.features import phase4_calibration as c
from Scripts.ops.phase4_baseline import sha,write


def audit(output):
    output=output.resolve();read=lambda name:[json.loads(l) for l in (output/name).open()]
    early=read('early.jsonl');dev=c.preceding(read('2022.jsonl'),'2023-01-01T00:00:00Z');evaluation=read('2023.jsonl')
    warmup=json.loads((output/'warmup-policy.json').read_text())['earlier_fitting'] if (output/'warmup-policy.json').exists() else False
    lookup={(r['fixture_id'],r['problem']):r for r in early+dev+evaluation}
    assert len(lookup)==len(early)+len(dev)+len(evaluation)
    fits=json.loads((output/'tuning/fit-audit.json').read_text());bundles={}
    for entry in fits:
        b=json.loads((output/'tuning'/(entry['id']+'.json')).read_text());problem=b['problem']
        assert b['id']==c.digest({k:v for k,v in b.items() if k!='id'})
        assert c.utc(b['cutoff'])<=c.utc('2023-01-01T00:00:00Z')
        kind,base,_=entry['name'].split(':')
        pool=dev if (base=='challenger' and not warmup) or (entry['fold']=='final_2023' and kind=='sigmoid') else early+dev
        train=c.preceding([r for r in pool if r['problem']==problem],b['cutoff'])
        assert b['fixture_ids']==[r['fixture_id'] for r in train]
        assert b['training_hash']==c.digest(train)
        assert all(c.utc(r['available_at'])<c.utc(b['cutoff']) for r in train)
        x=np.array([c.numeric(r,b['kind'],b['base_id']) for r in train]);means=x.mean(axis=0);scales=x.std(axis=0);scales[scales==0]=1
        assert np.array_equal(means,b['means']) and np.array_equal(scales,b['scales'])
        if kind=='sigmoid':assert len(b['coefficients'])==1 and b['coefficients'][0]>=0 and not b['leagues']
        assert len(train)==b['support']['n']==b['support']['effective_sample_size']
        bundles[b['id']]=b
    nested=json.loads((output/'tuning/nested-base-selections.json').read_text());nested_checks=0
    for problem,entries in nested.items():
        rows=[r for r in dev if r['problem']==problem]
        for v in [entries['final'],*entries['folds'].values()]:
            if v.get('status')=='unavailable':continue
            use_early=warmup and c.utc(v['cutoff'])<c.utc('2023-01-01T00:00:00Z')
            pool=[r for r in early if r['problem']==problem]+rows if use_early else rows
            assert c.choose_base(pool,problem,v['cutoff'],earlier_fitting=use_early)==v;nested_checks+=1
    count=0;maximum=0.;memberships={}
    for filename in ('tuning/binary-oof.jsonl.gz','evaluation/binary-predictions.jsonl.gz'):
        seen=set()
        with gzip.open(output/filename,'rt') as stream:
            for line in stream:
                r=json.loads(line);source=lookup[(r['fixture_id'],r['problem'])]
                assert c.utc(source['kickoff']).year==(2022 if filename.startswith('tuning') else 2023)
                key=(r['family'],r.get('C'),r['fixture_id']);assert key not in seen;seen.add(key)
                assert r['scores']['y']==source['y'] and r['raw_probability']==source['bases'][r['base_id']]['p']
                if r['fit_id']:
                    bundle=bundles[r['fit_id']]
                    assert r['fixture_id'] not in bundle['fixture_ids']
                    assert c.utc(bundle['cutoff'])<=c.utc(source['as_of'])
                    if filename.startswith('evaluation'):assert c.utc(bundle['cutoff'])==c.utc('2023-01-01T00:00:00Z')
                    prediction=c.predict_binary(bundle,source);p=prediction['p'];z=prediction['logit']
                else:p=r['raw_probability'];z=c.logit(p)
                delta=abs(p-r['calibrated_probability']);maximum=max(maximum,delta);assert delta<1e-14
                nll=c.binary_score(p,source['y'],z)['nll'];assert abs(nll-r['scores']['nll'])<1e-12
                count+=1
                if filename.startswith('evaluation'):memberships.setdefault(r['family'],set()).add(r['fixture_id'])
    for family,ids in memberships.items():
        problem=family.split(':')[2];assert ids=={r['fixture_id'] for r in evaluation if r['problem']==problem}
    for name,h in json.loads((output/'tuning/LOCK.json').read_text()).items():assert sha(output/'tuning'/name)==h
    joint=0;max_tail=0.
    for filename in ('tuning/temperature-oof.jsonl.gz','evaluation/joint-predictions.jsonl.gz'):
        with gzip.open(output/filename,'rt') as stream:
            for line in stream:
                r=json.loads(line);d=r['scores']['distribution'];assert d['tail_bound']<=1e-10;max_tail=max(max_tail,d['tail_bound']);joint+=1
                if 'derived' in r['scores']:
                    scores=r['scores'];assert abs(sum(scores['derived']['winner']['p'])-1)<1e-10
                    for v in [*scores['totals'].values(),*scores['derived']['handicaps'].values()]:assert abs(sum(v['p'])-1)<1e-10
    result={'verified_fits':len(fits),'nested_choices_replayed':nested_checks,'binary_predictions_replayed':count,
            'maximum_probability_replay_difference':maximum,'joint_records_checked':joint,'maximum_tail_bound':max_tail,
            'all_2023_memberships_match':True,'selection_and_bundle_lock_verified':True,'future_label_contamination':False}
    if warmup:
        previous=Path(__file__).resolve().parents[2]/'Research/phase4-step3-calibration-2026-09-29'
        for shard in ('2022.jsonl','2023.jsonl'):assert sha(output/shard)==sha(previous/shard)
        def controls(path):
            with gzip.open(path,'rt') as stream:
                for r in map(json.loads,stream):
                    if r['family'].split(':')[1]=='control':
                        yield {k:r.get(k) for k in ('fixture_id','family','C','scores','raw_probability','calibrated_probability')}
        unchanged=0
        for filename in ('tuning/binary-oof.jsonl.gz','evaluation/binary-predictions.jsonl.gz'):
            for old,new in zip_longest(controls(previous/filename),controls(output/filename)):
                assert old==new and old is not None;unchanged+=1
        plans=json.loads((output/'tuning/selection.json').read_text())
        result['extension']={'control_predictions_unchanged':unchanged,'2022_and_2023_inputs_byte_identical':True,
            'challenger_support':{k:v['oof_support']|{'tuning_available':v['tuning_available']} for k,v in plans['binary'].items() if v['base']=='challenger'},
            'joint_challenger_support':plans['temperature']['challenger']['oof_support']}
    write(output/'validation/independent-audit.json',result);print(json.dumps(result))


if __name__=='__main__':audit(Path(sys.argv[1]))
