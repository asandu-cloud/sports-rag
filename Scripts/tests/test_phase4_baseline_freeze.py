"""Baseline freeze boundaries and independent replay invariants."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from Scripts.ops import phase4_baseline as freeze
from Scripts.ops.phase4_baseline_replay import scenarios

ROOT = Path(__file__).resolve().parents[2]


def test_existing_baseline_directory_refused_before_any_write(tmp_path):
    marker=tmp_path/'keep';marker.write_text('preserved')
    with pytest.raises(FileExistsError): freeze.capture(ROOT,tmp_path)
    assert marker.read_text()=='preserved' and list(tmp_path.iterdir())==[marker]


def test_copy_rejects_escaping_source_and_retains_dirty_bytes(tmp_path):
    root=tmp_path/'root';root.mkdir()
    (root/'changed.py').write_text('dirty current source')
    dest=tmp_path/'copy'
    expected=freeze.copy_file(root,dest,'changed.py')
    assert (dest/'changed.py').read_text()=='dirty current source'
    assert expected==freeze.sha(root/'changed.py')
    (tmp_path/'outside').write_text('never copy')
    with pytest.raises(ValueError): freeze.copy_file(root,dest,'../outside')
    with pytest.raises(FileExistsError): freeze.copy_file(root,dest,'changed.py')


def test_manifest_detects_tamper_missing_and_extra_files(tmp_path):
    f=tmp_path/'input.json';f.write_text('{}')
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'input.json':freeze.sha(f)}))
    assert freeze.verify(tmp_path)==1
    f.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'): freeze.verify(tmp_path)
    f.write_text('{}');(tmp_path/'extra').write_text('new')
    with pytest.raises(ValueError,match='membership'): freeze.verify(tmp_path)
    (tmp_path/'extra').unlink();f.unlink()
    with pytest.raises(ValueError,match='membership'): freeze.verify(tmp_path)


def test_data_allowlist_contains_no_reserved_outcomes():
    assert set(freeze.DEVELOPMENT)=={'manifest.json','feature-schema.json','splits.json','lockbox-policy.json',
                                    'development/features.jsonl','development/labels.jsonl'}
    assert all(not n.startswith(('audit/','calibration/','final_system_test/','confirmation/')) for n in freeze.DEVELOPMENT)


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    folder=tmp_path_factory.mktemp('phase4-baseline-replay')
    inputs=folder/'inputs.json';inputs.write_text(json.dumps(scenarios(),sort_keys=True))
    outputs=[]
    for i in range(2):
        output=folder/f'output-{i}.json'
        proc=subprocess.run([sys.executable,'-B',str(ROOT/'Scripts/ops/phase4_baseline_replay.py'),
            '--source-root',str(ROOT),'--inputs',str(inputs),'--output',str(output)],cwd=folder,
            capture_output=True,text=True,timeout=90)
        assert proc.returncode==0,proc.stderr
        outputs.append(output.read_bytes())
    assert outputs[0]==outputs[1]
    return json.loads(outputs[0])


def test_real_projection_probability_replay_is_deterministic(recorded):
    assert recorded['io_violations']==[]
    assert set(recorded['ml_weights'].values())=={0.}
    assert len(recorded['scenarios'])==9
    for scenario in recorded['scenarios']:
        assert len(scenario['results'])==7
        if scenario['id']=='missing_profiles':
            assert all(r['projection']['value'] is None for r in scenario['results'])
        else:
            assert all(r['decision'].get('model_probability') is not None for r in scenario['results'])
        assert all(r['decision']['status']!='recommended' for r in scenario['results'])


def test_goal_coherence_and_full_asian_probabilities(recorded):
    for scenario in recorded['scenarios']:
        score=scenario['score_distribution']
        if not score: continue
        assert sum(p for h,a,p in score)==pytest.approx(1.,abs=1e-10)
        assert min(p for h,a,p in score)>=0.
        results={r['market']['group']:r for r in scenario['results']}
        assert results['btts']['projection']['value']==pytest.approx(sum(p for h,a,p in score if h>0 and a>0))
        for result in results.values():
            profile=result['decision'].get('settlement_profile')
            if profile:
                assert set(profile)=={'full_win','half_win','push','half_loss','full_loss'}
                assert min(profile.values())>=0.
                assert sum(profile.values())==pytest.approx(1.,abs=1e-10)


def test_variance_and_quality_inputs_reach_unmodified_engine(recorded):
    cases={s['id']:{r['market']['group']:r for r in s['results']} for s in recorded['scenarios']}
    assert cases['high_variance']['corners']['projection']['variance']>cases['standard']['corners']['projection']['variance']
    assert cases['high_variance']['sot']['decision']['model_probability']!=cases['standard']['sot']['decision']['model_probability']
    assert cases['sparse_support']['goals']['context']['data_quality']['recommendation_guardrail']['eligible'] is False
    assert cases['context_adjusted']['goals']['projection']['value']!=cases['standard']['goals']['projection']['value']
