"""Production communication, passive audit and baseline numerical parity."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag_ingest"))

from data_platform.services.decision_audit import run_selector
from data_platform.services.decision_communication import build_decision_explanation
from data_platform.services.match_read_cards import build_match_read_card
from data_platform.recommendation_identity import recommendation_identity
from Scripts.tests.data_platform.test_match_reads import _market_result


def test_persistence_keeps_all_markets_and_shared_identity(settings, engine, session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_reads import MatchReadService
    from Scripts.discord_bot.match_read_hubs import _selection_line
    service = MatchReadService(repo=MatchReadRepository(session_factory=session_factory))
    chosen = _market_result(odds=2.075)
    rejected = _market_result(group='cards', status='no_bet')
    chosen['context']['decision_audit'] = {'coverage': 'test', 'policy_version': 'fixed',
        'candidates': [{'option': {'odds': 2.075}}, {'option': {'odds': 1.8}}]}
    saved = service.create(canonical_results=[chosen, rejected], status='recommended',
                           thesis='Test', selections=[{'result_index': 0, 'role': 'core'}])
    loaded = service.get(saved['id'])
    rows = loaded['game_script']['decision_audit']['markets']
    assert len(rows) == 2 and rows[0]['selected'] and not rows[1]['selected']
    assert rows[1]['reason'] == rejected['decision']['reason']
    assert len(loaded['selections'][0]['data']['context']['decision_audit']['candidates']) == 2
    card = build_match_read_card(loaded)
    selection = card['selections'][0]
    assert selection['recommendation_id'] == recommendation_identity(chosen)
    assert '2.075' in selection['explanation']['reasoning']
    assert '2.075' in _selection_line(selection)
    assert selection['explanation']['checked_at'] is None
    assert 'unverified' in _selection_line(selection)
    assert 'not win probabilities' in selection['explanation']['uncertainty']


def test_identity_is_stable_across_copy_and_changes_with_quote_or_stage():
    result = _market_result()
    original = recommendation_identity(result)
    result['evidence'] = [{'key': 'prose', 'value': 'new wording'}]
    assert recommendation_identity(result) == original
    result['context']['forecast_stage'] = 'confirmed_lineups'
    assert recommendation_identity(result) != original
    result['context'].clear()
    result['decision']['quote']['odds'] = 2.075
    assert recommendation_identity(result) != original


def test_missing_metrics_do_not_become_zero_or_verified():
    result = _market_result()
    result['decision'].update(expected_value=None, model_probability=None)
    explanation = build_decision_explanation(result)
    assert explanation['estimated_ev'] is None
    assert 'Estimated return' not in explanation['reasoning']
    assert explanation['checked_at'] is None
    assert explanation['expires_at'] is None
    assert 'timestamp unavailable' in explanation['price_conditions']
    assert 'not been verified' in explanation['price_conditions']


@pytest.mark.parametrize('mean', [1.1, 2.1, 2.8, 3.8, 5.0])
def test_passive_audit_retains_input_filters_without_changing_selector(mean):
    from rag_ingest.core.line_selection import select_best_total_recommendation
    options = [{'side': side, 'point': line, 'odds': odds, 'bookmaker': 'Book', 'market_key': key}
               for side in ('over', 'under') for line in (1.5, 2.5, 3.5)
               for odds in (1.05, 1.8, 2.075) for key in ('totals', 'alternate_totals')]
    expected = select_best_total_recommendation(deepcopy(options), mean)
    context = {}
    actual = run_selector(context, select_best_total_recommendation, deepcopy(options), mean)
    assert actual == expected
    audit = context['decision_audit']
    assert audit['input_options'] == options
    assert audit['selector_output'] == actual
    assert len(audit['candidates']) == len(actual['all_lines'])
    assert all(row['reason'] for row in audit['candidates'])


@pytest.mark.parametrize('market', ['goals', 'corners', 'cards', 'sot', 'btts', 'moneyline', 'spreads'])
@pytest.mark.parametrize('available', [True, False])
def test_market_service_matches_unmodified_baseline(monkeypatch, market, available):
    from core import market_service as candidate
    from Scripts.tests.test_market_service import _event
    from prob_models import dixon_coles_scoreline_matrix
    baseline_path = Path(__file__).resolve().parents[1] / 'fixtures/communication_market_service_before.py'
    spec = importlib.util.spec_from_file_location('core.communication_baseline_market_service', baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    matrix = dixon_coles_scoreline_matrix(1.6, 1.2)
    scores = {(h, a): p for h, row in enumerate(matrix) for a, p in enumerate(row)}
    for module in (baseline, candidate):
        audit = {'temporal_status': 'fixture_rows_strictly_before_target_date',
                 'current_season_matches': 12, 'effective_sample_size': 12}
        monkeypatch.setattr(module, 'get_prediction_profile_context', lambda *a, **k: ({'goals_for_pm': 1.5}, audit))
        monkeypatch.setattr(module, '_total_variance', lambda *a, **k: (3.0, {'source': 'historical_observed'}))
        for name, value in [('projected_total_goals', 2.8), ('projected_total_corners', 10.0),
                            ('projected_total_sot', 9.0), ('projected_goal_difference', .4)]:
            monkeypatch.setattr(module, name, lambda *a, _v=value, **k: (_v, _v, _v) if available else (None, None, None))
        monkeypatch.setattr(module, '_card_statistics', lambda *a, **k: (4.0, 4.0, 4.0, None) if available else (None, None, None, None))
        monkeypatch.setattr(module, 'projected_btts_prob', lambda *a, **k: (.6, 1.6, 1.2, None, None) if available else (None, None, None, None, None))
        monkeypatch.setattr(module, 'projected_moneyline_probs', lambda *a, **k: (.5, .25, .25, 1.6, 1.2) if available else (None, None, None, None, None))
        monkeypatch.setattr(module, 'projected_correct_score_probs', lambda *a, **k: (scores, 1.6, 1.2) if available else (None, None, None))
        monkeypatch.setattr(module, 'apply_release_policy', lambda d, c: d)
    event = _event()
    for key, line in [('corners', 9.5), ('cards', 3.5), ('sot', 8.5)]:
        event['bookmakers'][0]['markets'].append({'key': key, 'outcomes': [
            {'name': side, 'point': line, 'price': 2.075} for side in ('Over', 'Under')]})
    kwargs = dict(generated_at='2026-04-30T12:00:00Z', input_snapshot_id='synthetic')
    expected = baseline.evaluate_market(deepcopy(event), 'EPL', market, **kwargs).to_dict()
    actual = candidate.evaluate_market(deepcopy(event), 'EPL', market, **kwargs).to_dict()
    audit = actual['context'].pop('decision_audit')
    assert actual == expected
    assert audit['final_decision'] == actual['decision']


def test_five_fixture_three_selection_discord_budget():
    from Scripts.discord_bot.match_read_hubs import build_match_read_hub_embeds
    reads = []
    for i in range(5):
        result = _market_result(fixture_id=str(i), odds=2.075)
        selections = []
        for j in range(3):
            data = deepcopy(result)
            data['market']['group'] = ['goals', 'corners', 'sot'][j]
            selections.append({'position': j+1, 'role': 'core' if j == 0 else 'supporting',
                               'market': data['market'], 'pick': 'Over 2.5',
                               'quote_odds': 2.075, 'bookmaker': 'William Hill', 'data': data})
        reads.append({'id': i, 'fixture': result['fixture'], 'stage': 'pre_match', 'version': 1,
                      'status': 'recommended', 'thesis': 'Fixture briefing.', 'game_script': {},
                      'selections': selections, 'packages': []})
    embeds = build_match_read_hub_embeds(reads, league='EPL', target_date='2026-09-06', color=1)
    fields = [f for e in embeds for f in e.to_dict()['fields']]
    assert len(fields) == 15
    assert all('2.075' in f['value'] for f in fields)


def test_local_website_job_is_loopback_only_and_keeps_separate_logs(tmp_path):
    import plistlib
    from Scripts.ops.match_read_launchd import build_paths
    from Scripts.ops.website_launchd import render
    paths = build_paths(root=tmp_path)
    config = plistlib.loads(render(paths).encode())
    assert config['Label'] == 'com.bettingrag.website'
    assert config['ProgramArguments'][-4:] == ['--host', '127.0.0.1', '--port', '8000']
    assert config['KeepAlive'] is True
    assert 'website.stderr.log' in config['StandardErrorPath']
