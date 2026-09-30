"""Fresh-process calibration tuning/evaluation with separated development readers."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys


def run(prepared,stage):
    prepared=prepared.resolve();output=prepared/('tuning' if stage=='tune' else 'evaluation');output.mkdir(exist_ok=False)
    helper=Path(__file__).with_name('phase4_candidate_worker.py');namespace={'__name__':'helpers'}
    exec(compile(helper.read_bytes(),str(helper),'exec'),namespace);load=namespace['load']
    source=prepared/'implementation';readable={prepared/'frozen_prob_models.py',prepared/'warmup-policy.json'}
    if stage=='tune':readable|={prepared/'early.jsonl',prepared/'2022.jsonl',prepared/'step2-selection.json'}
    else:readable|={prepared/'2023.jsonl'}|set((prepared/'tuning').glob('*.json'))
    violations=namespace['guard'](source,output,readable)
    scoring=load('phase4_comparison_math',source/'Scripts/data_platform/features/phase4_backtest.py')
    cal=load('phase4_calibration_math',source/'Scripts/data_platform/features/phase4_calibration.py')
    probability=load('frozen_probability',prepared/'frozen_prob_models.py')
    def write(name,value):(output/name).write_text(cal.encode(value)+'\n')
    def read(name):return [json.loads(line) for line in (prepared/name).open()]
    def save(bundle):write(bundle['id']+'.json',bundle);return bundle['id']
    warmup=json.loads((prepared/'warmup-policy.json').read_text())['earlier_fitting'] if (prepared/'warmup-policy.json').exists() else False
    def metadata(r):return {k:r[k] for k in ('fixture_id','kickoff','week','year','league','season','season_stage','forecast_stage','missingness','problem','snapshot_id')}
    final_cutoff='2023-01-01T00:00:00+00:00'
    if stage=='tune':
        early=read('early.jsonl');all_dev=read('2022.jsonl');dev=cal.preceding(all_dev,final_cutoff)
        if any(cal.utc(r['kickoff']).year>=2022 for r in early) or any(cal.utc(r['kickoff']).year!=2022 for r in all_dev):raise ValueError('Wrong fitting shard')
        previous_selection=json.loads((prepared/'step2-selection.json').read_text())['selected']
        plans={};nested={};audit_fits=[];oof_counts={}
        with gzip.open(output/'binary-oof.jsonl.gz','wt') as stream:
            for problem in cal.PROBLEMS:
                dp=[r for r in dev if r['problem']==problem];ep=[r for r in early if r['problem']==problem]
                final_base=cal.choose_base(dp,problem,final_cutoff)
                expected=previous_selection[('goals' if problem=='btts' else problem)+':'+cal.FAMILIES[problem]]
                if final_base['base_id']!=expected:raise ValueError('Step 2 selection replay mismatch')
                nested[problem]={'final':final_base,'folds':{}}
                for quarter in range(1,5):
                    key=f'2022-Q{quarter}';cutoff=scoring.quarter_cutoff(key)
                    try:nested[problem]['folds'][key]=cal.choose_base(ep+dp if warmup else dp,problem,cutoff,earlier_fitting=warmup)
                    except ValueError as exc:
                        if str(exc)!='Insufficient earlier base-selection support':raise
                        nested[problem]['folds'][key]={'status':'unavailable','reason':str(exc),'cutoff':cutoff}
                for kind,base in [('comparator','control'),('sigmoid','control'),('sigmoid','challenger')]:
                    name=kind+':'+base+':'+problem;predictions={C:[] for C in cal.CS};raw=[];folds=[]
                    for quarter in range(1,5):
                        key=f'2022-Q{quarter}';cutoff=scoring.quarter_cutoff(key)
                        validation=[r for r in dp if scoring.quarter(r['kickoff'])==key]
                        choice=nested[problem]['folds'][key]
                        if base=='challenger' and choice.get('status')=='unavailable':
                            folds.append({'fold':key,'status':'unavailable','reason':choice['reason']});continue
                        bid=choice['base_id'] if base=='challenger' else 'control'
                        training=cal.preceding(dp if base=='challenger' and not warmup else ep+dp,cutoff)
                        try:cal.validate_training(training,cutoff,problem,bid)
                        except ValueError as exc:
                            if str(exc)!='Insufficient fitting support':raise
                            folds.append({'fold':key,'status':'unavailable','reason':str(exc)});continue
                        folds.append({'fold':key,'status':'available','base_id':bid,'fit_n':len(training),'evaluation_n':len(validation)})
                        for r in validation:raw.append({**metadata(r),'scores':cal.binary_score(r['bases'][bid]['p'],r['y'])})
                        for C in cal.CS:
                            bundle=cal.fit_binary(training,cutoff=cutoff,problem=problem,kind=kind,C=C,base_id=bid);save(bundle)
                            audit_fits.append({'id':bundle['id'],'name':name,'fold':key,'support':bundle['support'],'cutoff':cutoff,'base_id':bid})
                            for r in validation:
                                predicted=cal.predict_binary(bundle,r)
                                item={**metadata(r),'family':name,'C':C,'base_id':bid,'fit_id':bundle['id'],
                                      'scores':cal.binary_score(predicted['p'],r['y'],predicted['logit']),
                                      'raw_probability':r['bases'][bid]['p'],'calibrated_probability':predicted['p']}
                                predictions[C].append(item);stream.write(cal.encode(item)+'\n')
                    available=scoring.support(raw)['sufficient'] if raw else False
                    grid={str(C):float(sum(r['scores']['nll'] for r in values)/len(values)) for C,values in predictions.items()} if available else {}
                    raw_loss=float(sum(r['scores']['nll'] for r in raw)/len(raw)) if raw else None
                    if available:
                        best=min(grid.values());C=min(C for C in cal.CS if grid[str(C)]<=best+1e-8)
                        method='identity' if kind=='sigmoid' and raw_loss<=best+1e-8 else kind
                    else:C=None;method='identity'
                    bid=final_base['base_id'] if base=='challenger' else 'control';bundle_id=None
                    if method!='identity':
                        training=cal.preceding(ep+dp if kind=='comparator' else dp,final_cutoff)
                        bundle=cal.fit_binary(training,cutoff=final_cutoff,problem=problem,kind=kind,C=C,base_id=bid);bundle_id=save(bundle)
                        audit_fits.append({'id':bundle_id,'name':name,'fold':'final_2023','support':bundle['support'],'cutoff':final_cutoff,'base_id':bid})
                    plans[name]={'kind':kind,'base':base,'problem':problem,'base_id':bid,'C':C,'method':method,'bundle_id':bundle_id,
                        'oof_grid':grid,'oof_identity_nll':raw_loss,'oof_support':scoring.support(raw),'folds':folds,'tuning_available':available,
                        'scope':'binary_diagnostic_only','publication_enabled':False}
                    oof_counts[name]={str(C):len(values) for C,values in predictions.items()}
                    print('Tuned '+name+' -> '+method+' C='+str(C),flush=True)
        temperature_plans={}
        goals=[r for r in dev if r['problem']=='goals']
        with gzip.open(output/'temperature-oof.jsonl.gz','wt') as stream:
            for base in ('control','challenger'):
                cohort=[];totals={T:0. for T in cal.TEMPERATURES};counts={T:0 for T in cal.TEMPERATURES}
                for r in goals:
                    choice=nested['goals']['folds'][scoring.quarter(r['kickoff'])]
                    if base=='challenger' and choice.get('status')=='unavailable':continue
                    bid=choice['base_id'] if base=='challenger' else 'control';cohort.append(r)
                    for T in cal.TEMPERATURES:
                        distribution=cal.powered_goals(*r['bases'][bid]['goal_means'],T,probability)
                        scores=cal.score_joint(distribution,r['team_target'],primary_only=True)
                        totals[T]+=scores['nll'];counts[T]+=1
                        stream.write(cal.encode({**metadata(r),'base':base,'base_id':bid,'temperature':T,'scores':scores})+'\n')
                sufficient=scoring.support(cohort)['sufficient'] if cohort else False
                losses={str(T):totals[T]/counts[T] for T in cal.TEMPERATURES} if sufficient else {}
                if sufficient:
                    best=min(losses.values());tied=[T for T in cal.TEMPERATURES if losses[str(T)]<=best+1e-8]
                    selected=min(tied,key=lambda T:(T!=1.,abs(T-1),T))
                else:selected=1.
                temperature_plans[base]={'temperature':selected,'base_id':nested['goals']['final']['base_id'] if base=='challenger' else 'control',
                    'oof_grid':losses,'oof_support':scoring.support(cohort),'tuning_available':sufficient,'version':cal.VERSION,'scope':'coherent_joint_goals'}
                print('Tuned joint temperature '+base+' -> '+str(selected),flush=True)
        write('selection.json',{'binary':plans,'temperature':temperature_plans,'version':cal.VERSION,'selection_year':2022,'evaluation_year':2023})
        write('nested-base-selections.json',nested);write('fit-audit.json',audit_fits)
        write('audit.json',{'io_violations':violations,'oof_counts':oof_counts,'late_2022_labels_excluded':len(all_dev)-len(dev),
                            'read_2023':False,'earlier_challenger_fitting':warmup,'publication_enabled':False})
        lock={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
        write('LOCK.json',lock)
        for p in output.iterdir():
            if p.is_file():p.chmod(0o444)
    else:
        plans=json.loads((prepared/'tuning/selection.json').read_text());rows=read('2023.jsonl')
        if any(cal.utc(r['kickoff']).year!=2023 for r in rows):raise ValueError('Wrong evaluation shard')
        comparisons={};lineage=[]
        with gzip.open(output/'binary-predictions.jsonl.gz','wt') as stream:
            for name,plan in plans['binary'].items():
                cohort=[r for r in rows if r['problem']==plan['problem']];reference=[];control=[];predictions=[]
                bundle=json.loads((prepared/'tuning'/(plan['bundle_id']+'.json')).read_text()) if plan['bundle_id'] else None
                for r in cohort:
                    base=r['bases'][plan['base_id']];raw=base['p']
                    predicted=cal.predict_binary(bundle,r) if bundle else {'p':raw,'logit':cal.logit(raw),'fit_id':None,'unknown_league':False}
                    ref={**metadata(r),'scores':cal.binary_score(raw,r['y'])};reference.append(ref)
                    control.append({**metadata(r),'scores':cal.binary_score(r['bases']['control']['p'],r['y'])})
                    item={**metadata(r),'family':name,'base_id':plan['base_id'],'base_fit_id':base['base_fit_id'],'base_fallbacks':base['fallbacks'],
                        'fit_id':predicted['fit_id'],'raw_probability':raw,'calibrated_probability':predicted['p'],'calibration_version':cal.VERSION,
                        'scores':cal.binary_score(predicted['p'],r['y'],predicted['logit']),'unknown_league':predicted['unknown_league'],
                        'scope':'binary_diagnostic_only','support':bundle['support'] if bundle else None,
                        'uncertainty':'paired-week evaluation; no per-forecast confidence asserted','publication_enabled':False}
                    predictions.append(item);stream.write(cal.encode(item)+'\n')
                comparisons[name]={'plan':plan,'against_identity':cal.compare_binary(reference,predictions),'against_frozen_control':cal.compare_binary(control,predictions),
                                   'unknown_league_n':sum(r['unknown_league'] for r in predictions),'joint_distribution_qualified':False}
                print('Evaluated '+name,flush=True)
        write('binary-comparison.json',comparisons)
        joint={}
        with gzip.open(output/'joint-predictions.jsonl.gz','wt') as stream:
            for base,plan in plans['temperature'].items():
                reference=[];control=[];predictions=[];tails=[]
                for r in rows:
                    if r['problem']!='goals':continue
                    bid=plan['base_id'];h,a=r['bases'][bid]['goal_means']
                    raw=cal.score_joint(cal.powered_goals(h,a,1.,probability),r['team_target'])
                    frozen=cal.score_joint(cal.powered_goals(*r['bases']['control']['goal_means'],1.,probability),r['team_target'])
                    calibrated=cal.score_joint(cal.powered_goals(h,a,plan['temperature'],probability),r['team_target'])
                    tails.append(calibrated['distribution']['tail_bound'])
                    reference.append({**metadata(r),'scores':raw});control.append({**metadata(r),'scores':frozen})
                    item={**metadata(r),'base':base,'base_id':bid,'base_fit_id':r['bases'][bid]['base_fit_id'],
                          'scores':calibrated,'raw_scores':raw,'calibration_version':cal.VERSION,'publication_enabled':False,
                          'scope':'coherent_joint_goals','selection_support':plan['oof_support']}
                    predictions.append(item);stream.write(cal.encode(item)+'\n')
                joint[base]={'plan':plan,'against_identity':scoring.compare(reference,predictions,diagnostics=True),
                             'against_frozen_control':scoring.compare(control,predictions,diagnostics=True),'max_transformed_tail_bound':max(tails)}
                print('Evaluated coherent goals '+base,flush=True)
        write('joint-comparison.json',joint)
        write('audit.json',{'io_violations':violations,'evaluation_rows':len(rows),'read_2023':True,'reserved_outcomes_read':False,'publication_enabled':False})
        write('summary.json',{'version':cal.VERSION,
            'binary_development_passes':[k for k,v in comparisons.items() if v['against_identity']['passes']],
            'coherent_temperature_incremental_passes':[k for k,v in joint.items() if v['against_identity']['passes']],
            'coherent_system_passes_vs_control':[k for k,v in joint.items() if v['against_frozen_control']['passes']],
            'production_changed':False,'later_qualification_complete':False})
    if violations:raise ValueError('Forbidden IO')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--stage',choices=('tune','evaluate'),required=True)
    args=parser.parse_args();run(args.prepared,args.stage)
