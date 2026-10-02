"""Run the owner-authorized, development-only complete weight comparison."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from Scripts.data_platform.features import phase4_weight_setups as w
from Scripts.data_platform.features import phase4_backtest as scoring
from Scripts.data_platform.features.benchmarks.artifacts import complete, verify_complete, read_json, write_json, sha
from Scripts.data_platform.features.benchmarks.isolation import offline_guard
from Scripts.rag_ingest.core.prediction_guardrails import resolve_total_variance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'Research/phase4-baseline-2026-09-28/control'
STAT = ROOT/'Research/phase4-step2-backtest-2026-09-29'
CARD = ROOT/'Research/phase4-card-improvements-2026-10-02'
RETAINED = ROOT/'Research/phase4-retained-components-2026-10-02'
PROTOCOL = 'docs/phase4-weight-comparison-protocol-2026-10-02.md'
SOURCES = [PROTOCOL, 'Scripts/ops/phase4_weight_comparison.py', 'Scripts/data_platform/features/phase4_weight_setups.py',
           'Scripts/data_platform/features/phase4_candidates.py', 'Scripts/data_platform/features/phase4_candidate_adapter.py',
           'Scripts/data_platform/features/phase4_backtest.py', 'Scripts/data_platform/features/phase4_finalists.py',
           'Scripts/data_platform/features/phase4_card_candidates.py', 'Scripts/data_platform/features/phase4_card_reconstruction.py',
           'Scripts/data_platform/features/phase4_cards.py', 'Scripts/data_platform/features/count_calibration.py',
           'Scripts/rag_ingest/core/prediction_guardrails.py', 'Scripts/data_platform/features/benchmarks/artifacts.py',
           'Scripts/data_platform/features/benchmarks/isolation.py', 'Scripts/tests/test_phase4_weight_setups.py']


def rows(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        for line in stream:
            yield json.loads(line)


def put_rows(path, records):
    with path.open('x') as handle:
        for record in records:
            handle.write(w.cm.canonical(record)+'\n')


def engine(card=False):
    base = CARD/'inputs' if card else BASE/'workspace/Scripts/rag_ingest/core'
    return w.load_engine((base/'projections.py').read_text(), (base/'weights.py').read_text())


def check_prepared(output):
    for name, expected in read_json(output/'PREPARED.json').items():
        if sha(output/name) != expected:
            raise ValueError('Prepared file changed: '+name)
    for name, expected in read_json(output/'manifest.json')['sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Implementation changed: '+name)


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    preserved = {name: sha(ROOT/name) for name in read_json(BASE/'source-hashes.json')
                 if (name.startswith('Scripts/') or name.startswith('Index/ml_models/') or name.startswith('requirements'))
                 and (ROOT/name).is_file()}
    write_json(output/'PRESERVATION_BEFORE.json', preserved)
    with offline_guard(root=ROOT, output=output):
        seals = {}
        for source in (BASE, STAT, CARD, RETAINED):
            verify_complete(source)
            seals[str(source.relative_to(ROOT))] = sha(source/'COMPLETE.json')
        for name in SOURCES:
            target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        write_json(output/'registry.json', {m: w.registry(m) for m in w.MARKETS})
        write_json(output/'manifest.json', {'version': w.VERSION, 'sources': {p: sha(ROOT/p) for p in SOURCES},
                   'input_seals': seals, 'seed': w.SEED, 'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')},
                   'python': sys.version, 'later_period_access': False, 'production_enabled': False,
                   'implementation_decision': 'await_owner_review'})
        write_json(output/'PREPARED.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print('Registry and protocol frozen: 60 configurations per market.', flush=True)


def metadata(s):
    f = s['fixture']; date = w.cm.utc(f['kickoff'])
    return {'fixture_id': f['fixture_id'], 'snapshot_id': s['snapshot_id'], 'kickoff': f['kickoff'],
            'as_of': s['as_of'], 'week': date.strftime('%G-W%V'), 'year': date.year,
            'quarter': (date.month-1)//3+1, 'league': f['competition'], 'season': str(f['season']),
            'season_stage': w.scope.scope(s)['season_stage'], 'forecast_stage': s['forecast_stage'],
            'missingness': 'xg_partial_or_missing' if any(p.get('xg_home_pm') is None or p.get('xg_away_pm') is None
                                                       for p in s['profiles'].values()) else 'xg_both_venues'}


def statistical_year(output, year, registry):
    historical = w.Inputs(list(rows(STAT/'history.jsonl')))
    numerical = engine()
    result = {m: {'metadata': [], 'means': [], 'alpha': [], 'targets': [], 'team_targets': []} for m in w.ALPHAS.keys() | {'goals'}}
    coverage = Counter(); parity = 0; maximum = 0.
    with gzip.open(output/f'inputs-{year}.jsonl.gz', 'wt') as saved:
        for quarter in range(1, 5):
            folder = STAT/'folds'/f'{year}-Q{quarter}'
            allow = {r['fixture_id']: r['markets'] for r in rows(folder/'eligibility.jsonl')}
            controls = {r['fixture_id']: r for r in rows(folder/'control-rates.jsonl')}
            targets = {r['fixture_id']: r for r in rows(folder/'targets.jsonl')}
            for s in rows(folder/'snapshots.jsonl'):
                if w.cm.utc(s['fixture']['kickoff']).year != year:
                    raise ValueError('Fold/date mismatch')
                fid = s['fixture']['fixture_id']
                if not w.scope.scope(s)['active']:
                    coverage['outside_five_league_eight_match_scope'] += 1
                    continue
                if year == 2022 and w.cm.utc(s['fixture']['kickoff'])+w.timedelta(hours=3) >= w.cm.utc('2023-01-01T00:00:00Z'):
                    coverage['tuning_label_not_available_at_selection'] += 1
                    continue
                markets = [m for m in ('goals', 'corners', 'sot') if allow[fid][m]['eligible']]
                for m in ('goals', 'corners', 'sot'):
                    if not allow[fid][m]['eligible']:
                        coverage[m+':source_ineligible'] += 1
                if not markets:
                    continue
                f = historical.features(s); saved.write(w.cm.canonical(f)+'\n')
                meta = metadata(s); meta['input_id'] = f['id']
                for market in markets:
                    specs = registry[market]
                    predicted = [w.stat_mean(f, spec['options'], market, numerical) for spec in specs]
                    baseline = controls[fid]
                    expected = baseline['goal_means'] if market == 'goals' else [baseline['means'][market]]
                    delta = max(abs(x-y) for x, y in zip(predicted[0], expected))
                    maximum = max(maximum, delta)
                    if delta > 1e-10:
                        raise ValueError('Archived control parity failed '+str(fid)+' '+market+' '+str(delta))
                    parity += 1
                    mu = baseline['means'][market]
                    original_alpha = 0. if market == 'goals' else max(0., ((baseline['variances'][market] or mu)-mu)/mu**2)
                    alphas = [original_alpha if spec['kind'] == 'original_control' else w.ALPHAS.get(market, 0.) for spec in specs]
                    r = result[market]
                    r['metadata'].append(meta); r['means'].append([list(p) if len(p) == 2 else [p[0], 0.] for p in predicted])
                    r['alpha'].append(alphas); r['targets'].append(targets[fid]['labels'][market])
                    r['team_targets'].append([targets[fid]['team_labels'][side]['goals'] for side in ('home', 'away')])
                if len(result['goals']['metadata']) % 250 == 0:
                    print(f'{year}: {len(result["goals"]["metadata"])} goal fixtures predicted.', flush=True)
            print(f'{year}-Q{quarter}: statistical forecasts complete.', flush=True)
    write_json(output/f'coverage-{year}.json', {'exclusions': dict(coverage), 'control_market_parity': parity, 'maximum_difference': maximum,
                                              'eligible': {m: len(r['metadata']) for m, r in result.items()}})
    return result


def card_year(output, year, registry):
    numerical = engine(True)
    controls = {r['fixture_id']: r for r in rows(CARD/'candidate-means.jsonl')}
    previous = {r['fixture_id']: r for r in rows(CARD/'predictions.jsonl')} if year == 2023 else {}
    result = {'metadata': [], 'means': [], 'alpha': [], 'targets': [], 'team_targets': [], 'previous_nll': [], 'previous_expected': []}
    maximum = 0.; excluded = Counter()
    for f in rows(CARD/'features.jsonl'):
        date = w.cards.cards.utc(f['kickoff'])
        if date.year != year or (year == 2023 and date.month < 4):
            continue
        if not f['baseline']:
            excluded['below_fixed_eight_match_gate'] += 1
            continue
        if year == 2022 and date+w.timedelta(hours=3) >= w.cm.utc('2023-01-01T00:00:00Z'):
            excluded['tuning_label_not_available_at_selection'] += 1
            continue
        fid = f['fixture_id']; old = controls[fid]
        control_mu = old['means']['fuller_control']
        variance, _ = resolve_total_variance('cards', control_mu, *old['observed_variances'])
        alpha = max(0., (variance-control_mu)/control_mu**2)
        cache = {}; predicted = []; alphas = []
        for spec in registry:
            if spec['kind'] == 'previous_card_candidate':
                mu = previous[fid]['scores']['selected_calibrated']['raw_mean'] if year == 2023 else float('nan')
                aa = .05
            elif spec['kind'] == 'fixed_reference':
                mu = old['fixed_referee']; aa = 0.
            else:
                mu = w.card_mean(f, spec['options'], numerical, cache); aa = alpha
            predicted.append([mu, 0.]); alphas.append(aa)
        delta = abs(predicted[0][0]-control_mu); maximum = max(maximum, delta)
        if delta > 1e-10:
            raise ValueError('Fuller card control parity failed')
        result['metadata'].append({'fixture_id': fid, 'snapshot_id': f['input_id'], 'input_id': f['input_id'],
            'kickoff': date.isoformat(), 'as_of': f['as_of'], 'week': date.strftime('%G-W%V'), 'year': year, 'quarter': (date.month-1)//3+1,
            'league': f['competition'], 'season': str(f['season']), 'season_stage': f['stage'],
            'forecast_stage': 'reconstructed_immediately_before_kickoff', 'missingness': f['foul_state']})
        result['means'].append(predicted); result['alpha'].append(alphas); result['targets'].append(f['target'])
        result['team_targets'].append([0, 0])
        result['previous_nll'].append(previous[fid]['scores']['selected_calibrated']['nll'] if year == 2023 else float('nan'))
        result['previous_expected'].append(previous[fid]['scores']['selected_calibrated']['mean'] if year == 2023 else float('nan'))
        if len(result['metadata']) % 250 == 0:
            print(f'{year}: {len(result["metadata"])} card fixtures predicted.', flush=True)
    write_json(output/f'card-coverage-{year}.json', {'exclusions': dict(excluded), 'control_parity': len(result['metadata']), 'maximum_difference': maximum})
    return result


def save_predictions(output, year, market, result):
    folder = output/str(year); folder.mkdir(exist_ok=True)
    metadata_ = result.pop('metadata')
    if len({r['fixture_id'] for r in metadata_}) != len(metadata_):
        raise ValueError('Duplicate prediction fixture')
    put_rows(folder/(market+'-membership.jsonl'), metadata_)
    arrays = {k: np.asarray(v, dtype=float) for k, v in result.items()}
    np.savez_compressed(folder/(market+'-predictions.npz'), **arrays)
    return metadata_, arrays


def losses_for(market, specs, data):
    n = len(data['targets']); losses = np.zeros((n, len(specs)))
    for i, spec in enumerate(specs):
        if spec['kind'] == 'previous_card_candidate':
            losses[:, i] = data['previous_nll']
        else:
            losses[:, i] = w.primary_loss(data['means'][:, i, :], data['targets'], data['alpha'][:, i],
                                           data['team_targets'] if market == 'goals' else None)
            if not np.all(np.isfinite(losses[:, i])):
                raise ValueError('Invalid predictions: whole configuration cannot qualify')
    return losses


def run(output):
    check_prepared(output)
    if (output/'SELECTION_LOCK.json').exists():
        raise FileExistsError('Selection already frozen')
    with offline_guard(root=ROOT, output=output):
        registry = read_json(output/'registry.json')
        selection = {}
        for year in (2022, 2023):
            records = statistical_year(output, year, registry)
            records['cards'] = card_year(output, year, registry['cards'])
            saved = {}
            for market, result in records.items():
                saved[market] = save_predictions(output, year, market, result)
            write_json(output/f'PREDICTIONS-{year}.json', {str(p.relative_to(output)): sha(p) for p in sorted((output/str(year)).iterdir())})
            for market, (meta, data) in saved.items():
                loss = losses_for(market, registry[market], data)
                np.save(output/str(year)/(market+'-losses.npy'), loss, allow_pickle=False)
                if year == 2022:
                    selection[market] = w.choose(registry[market], loss, meta)
            if year == 2022:
                write_json(output/'selection.json', selection)
                write_json(output/'SELECTION_LOCK.json', {'selection_sha256': sha(output/'selection.json'), 'registry_sha256': sha(output/'registry.json'),
                                                        'selected_before_2023_inference': True, 'implementation_decision': 'await_owner_review'})
                print('2022 ranking locked before 2023 forecasts.', flush=True)
    print('All forecasts and primary losses saved; no implementation decision made.', flush=True)


def frozen_probability():
    path = BASE/'workspace/Scripts/rag_ingest/prob_models.py'
    spec = importlib.util.spec_from_file_location('weight_frozen_probability', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def detailed(meta, data, column, market, probability):
    result = []
    for i, row in enumerate(meta):
        if market == 'cards':
            score = w.cards.score(float(data['means'][i, column, 0]), float(data['alpha'][i, column]), float(data['targets'][i]))
            result.append({**row, 'competition': row['league'], 'stage': row['season_stage'], 'foul_state': row['missingness'],
                           'target': float(data['targets'][i]), 'scores': {'value': score}})
        else:
            means = data['means'][i, column]; mu = float(sum(means))
            rates = {'goal_means': means.tolist(), 'means': {market: mu},
                     'variances': {market: float(mu+data['alpha'][i, column]*mu*mu)}}
            target = {'labels': {market: data['targets'][i]}, 'team_labels': {s: {'goals': data['team_targets'][i, k]} for k, s in enumerate(('home', 'away'))}}
            score = scoring.score_forecast(rates, w.cm.registry()[0], market, target, w.cm, probability)
            result.append({**row, 'scores': score})
    return result


def summarize(output):
    check_prepared(output)
    if sha(output/'selection.json') != read_json(output/'SELECTION_LOCK.json')['selection_sha256']:
        raise ValueError('Changed selection')
    with offline_guard(root=ROOT, output=output):
        registry = read_json(output/'registry.json'); selections = read_json(output/'selection.json')
        comparisons = {}; tables = {}; diagnostics = {}; probability = frozen_probability()
        for market in w.MARKETS:
            meta = list(rows(output/'2023'/(market+'-membership.jsonl')))
            with np.load(output/'2023'/(market+'-predictions.npz'), allow_pickle=False) as a:
                data = dict(a)
            loss = np.load(output/'2023'/(market+'-losses.npy'), allow_pickle=False)
            if not np.allclose(loss, losses_for(market, registry[market], data), rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError('Primary score replay mismatch')
            table = []
            for j, spec in enumerate(registry[market]):
                mu = data['means'][:, j, :].sum(axis=1)
                if spec['kind'] == 'previous_card_candidate': mu = data['previous_expected']
                errors = mu-data['targets']
                comparison = w.paired(meta, loss[:, j]-loss[:, 0])
                quarters = {str(q): {'n': sum(r['quarter'] == q for r in meta),
                             'relative_improvement': 1-float(np.mean(loss[[i for i, r in enumerate(meta) if r['quarter'] == q], j]))/
                             float(np.mean(loss[[i for i, r in enumerate(meta) if r['quarter'] == q], 0]))}
                            for q in sorted({r['quarter'] for r in meta})}
                table.append({**spec, 'nll': float(loss[:, j].mean()), 'relative_improvement': 1-float(loss[:, j].mean()/loss[:, 0].mean()),
                              'mae': float(np.abs(errors).mean()), 'rmse': float(np.sqrt((errors**2).mean())), 'bias': float(errors.mean()),
                              'paired': comparison, 'quarters': quarters,
                              'preselected_in_2022': j == selections[market]['index'],
                              'interpretation': 'descriptive_development_scoreboard_not_implementation_selection'})
            tables[market] = sorted(table, key=lambda r: (r['nll'], r['id']))
            j = selections[market]['index']
            reference = detailed(meta, data, 0, market, probability)
            challenger = detailed(meta, data, j, market, probability)
            if market == 'cards':
                paired_rows = [{**a, 'scores': {'reference': a['scores']['value'], 'candidate': b['scores']['value']}} for a, b in zip(reference, challenger)]
                full = w.cards.compare(paired_rows, 'candidate', 'reference')
                related = [line for line in full['control']['lines'] if full['candidate']['lines'][line]['log_loss'] > 1.02*full['control']['lines'][line]['log_loss']]
                # Preserve the complete Asian-category proper-score checks as well.
                for line in ('4.0', '4.25', '4.75', '5.0'):
                    def asian_loss(key):
                        return float(np.mean([-np.log(max(r['scores'][key]['asian'][line]['over'][scoring.outcome_class(r['target'], float(line))], 1e-15)) for r in paired_rows]))
                    if asian_loss('candidate') > 1.02*asian_loss('reference'): related.append('asian:'+line)
                passed = full['development_gate_passed'] and not related
                detailed_nll = np.array([r['scores']['candidate']['nll'] for r in paired_rows])
                full['related_market_regressions'] = related
            else:
                full = scoring.compare(reference, challenger, diagnostics=True); passed = full['passes']
                detailed_nll = np.array([r['scores']['nll'] for r in challenger])
            if np.max(np.abs(detailed_nll-loss[:, j])) > 1e-10:
                raise ValueError('Independent/detailed likelihood mismatch')
            paired = w.paired(meta, loss[:, j]-loss[:, 0])
            comparisons[market] = {'selected_id': registry[market][j]['id'], 'selected_options': registry[market][j]['options'],
               'support': {'fixtures': len(meta), 'weeks': len({r['week'] for r in meta})},
               'relative_improvement': 1-float(loss[:, j].mean()/loss[:, 0].mean()), 'paired': paired,
               'pre_multiplicity_screen': bool(passed), 'details': full,
               'implementation_decision': 'await_owner_review', 'production_qualified': False}
            diagnostics[market] = {'primary': comparisons[market]['selected_id'], 'reference': reference, 'candidate': challenger}
            print(market+': detailed comparison complete.', flush=True)
        correction = w.holm({m: c['paired']['p'] for m, c in comparisons.items()})
        for m, c in comparisons.items():
            c['holm_adjusted_p'] = correction[m]
            c['development_screen_passed'] = bool(c['pre_multiplicity_screen'] and correction[m] <= .05)
        write_json(output/'comparison.json', comparisons); write_json(output/'all-setups.json', tables)
        with gzip.open(output/'primary-diagnostics.json.gz', 'wt') as h:
            json.dump(diagnostics, h, sort_keys=True, allow_nan=False)
        write_json(output/'summary.json', {'status': 'research_comparison_complete', 'markets': {m: {k: c[k] for k in
            ('selected_id', 'selected_options', 'support', 'relative_improvement', 'paired', 'holm_adjusted_p', 'development_screen_passed')} for m, c in comparisons.items()},
            'implementation_decision': 'await_owner_review', 'production_changed': False, 'retained_bundle_changed': False,
            'qualification_2025_accessed': False, 'final_system_test_opened': False, 'prospective_reserve_opened': False})
    print((output/'summary.json').read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('prepare', 'run', 'summarize'))
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    {'prepare': prepare, 'run': run, 'summarize': summarize}[a.action](a.output.resolve())


if __name__ == '__main__': main()
