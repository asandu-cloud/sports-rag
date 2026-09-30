"""Research safety and semantics; no provider calls or production writes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features import count_calibration as calibration
from Scripts.ops import phase4_development as runner


def fixture():
    return dict(fixture_id=7, competition='EPL', season=2023, kickoff='2023-08-01T12:00:00+00:00',
                status='FT', home_team_id=1, away_team_id=2)


def players():
    return [dict(player_id=team * 100 + i, team_id=team, minutes=90, yellow_cards=0, red_cards=0)
            for team in (1, 2) for i in range(11)]


def qualified(rows, **kwargs):
    return cards.qualify(fixture(), rows, fixture(), raw_players=cards.normalized_payload(rows),
                         raw_reference={'sha256': 'test'}, **kwargs)


@pytest.mark.parametrize('yellow,red,total', [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3), (2, 1, 3)])
def test_target_matches_approved_participation_policy(yellow, red, total):
    rows = players()
    rows[0].update(yellow_cards=yellow, red_cards=red)
    result = qualified(rows)
    assert result['eligible'] and result['target'] == total
    assert result['settlement_policy'] == 'spix-participation-settlement.v1'


def test_arithmetic_is_distinct_from_qualified_evidence():
    result = cards.qualify(fixture(), players(), fixture())
    assert result['normalized_candidate_total'] == 0
    assert result['target'] is None
    assert result['exclusions'] == ['original_player_response_unavailable']


@pytest.mark.parametrize('update,reason', [({'red_cards': None}, 'missing_player_card_counts'),
    ({'yellow_cards': 2}, 'inconsistent_player_card_counts'),
    ({'yellow_cards': 1, 'minutes': None}, 'missing_carded_player_minutes')])
def test_missing_inconsistent_evidence_is_not_zero(update, reason):
    rows = players()
    rows[0].update(update)
    assert reason in qualified(rows)['exclusions']


def test_participation_and_period_boundaries():
    rows = players()
    rows[0].update(minutes=0, yellow_cards=2, red_cards=1)
    assert qualified(rows)['target'] == 0
    f = fixture() | {'status': 'AET'}
    result = cards.qualify(f, rows, None, raw_players=cards.normalized_payload(rows), raw_reference={})
    assert result['target'] is None
    assert 'regulation_player_statistics_unavailable' in result['exclusions']
    assert 'regulation_source_unverified' in result['exclusions']
    with pytest.raises(ValueError, match='Reserved'):
        cards.qualify(f | {'kickoff': '2024-01-01T00:00:00Z'}, rows, None)


def test_identity_conflicts_partial_teams_and_raw_conflict():
    rows = players()
    result = cards.qualify(fixture(), rows, fixture() | {'home_team_id': 999},
                           raw_players=cards.normalized_payload(rows), raw_reference={})
    assert 'regulation_source_identity_conflict' in result['exclusions']
    assert not qualified(rows[:-1])['eligible']
    raw = cards.normalized_payload(rows)
    rows[0]['yellow_cards'] = 1
    result = cards.qualify(fixture(), rows, fixture(), raw_players=raw, raw_reference={})
    assert result['target'] is None and 'normalized_raw_player_conflict' in result['exclusions']


def test_future_history_is_never_decoded(monkeypatch):
    old = fixture()
    future = old | {'fixture_id': 8, 'kickoff': '2025-01-01T12:00:00Z', 'outcome': 'SECRET_FUTURE'}
    original = json.loads
    decoded = []
    def guarded(text, *args, **kwargs):
        assert 'SECRET_FUTURE' not in text
        decoded.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(cards.json, 'loads', guarded)
    index, skipped = cards.period_index(json.dumps({'history': [old, future]}))
    assert list(index) == [7] and skipped == 1 and len(decoded) == 1
    assert cards.period_index(json.dumps({'history': [old, future | {'outcome': 999}]}))[0] == index


@pytest.mark.parametrize('text', ['{"history":[', '{"history":[{}]}', '{"history":[{"kickoff":"2023-01-01T00:00:00Z"}'])
def test_truncated_or_ambiguous_history_rejected(text):
    with pytest.raises((ValueError, KeyError)):
        cards.period_index(text)


def forecasts(year=2022):
    rng = np.random.default_rng(7)
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(700):
        dt = start + timedelta(hours=10 * i)
        result.append({'fixture_id': i, 'kickoff': dt.isoformat(), 'week': dt.strftime('%G-W%V'),
                       'reference': 3., 'target': int(rng.poisson(3)), 'league': 'EPL',
                       'season': year, 'stage': '16+', 'missing_fallback': False})
    return result


def test_fit_only_selection_and_determinism():
    rows = forecasts()
    fitted = calibration.fit(rows)
    assert fitted == calibration.fit(deepcopy(rows))
    with pytest.raises(ValueError, match='only use 2022'):
        calibration.fit(rows + forecasts(2023)[:1])
    with pytest.raises(ValueError, match='only use 2023'):
        calibration.evaluate(rows, fitted, 'goals')
    with pytest.raises(ValueError, match='Insufficient'):
        calibration.fit(rows[:50])


@pytest.mark.parametrize('kickoff', ['2022-12-31T23:00:00Z', '2023-12-31T23:00:00Z', '2024-01-01T12:00:00Z', '2025-01-01T12:00:00Z'])
def test_label_availability_and_reserved_periods(kickoff):
    with pytest.raises(ValueError):
        calibration.partition({'kickoff': kickoff})


@pytest.mark.parametrize('alpha', calibration.ALPHAS)
def test_distribution_normalization_and_intervals(alpha):
    d = calibration.distribution(5., alpha)
    assert np.isclose(d.pmf(np.arange(10000)).sum(), 1., atol=1e-12)
    assert d.ppf(.1) <= d.ppf(.9)
    assert np.isclose(d.cdf(4) + d.sf(4), 1.)


def test_week_bootstrap_paired_deterministic_and_reporting():
    fitting, evaluation = forecasts(), forecasts(2023)
    fit = calibration.fit(fitting)
    report, predictions = calibration.evaluate(evaluation, fit, 'goals')
    again, _ = calibration.evaluate(evaluation, fit, 'goals')
    assert report == again
    assert len(predictions) == 700
    assert report['reference_poisson']['n'] == 700
    assert sum(b['n'] for b in report['reference_poisson']['reliability']) == 700
    assert calibration.week_interval(evaluation, np.zeros(700)) == [0., 0.]


def test_reference_versions_deduplicate_or_fail():
    row = forecasts()[0] | {'market': 'goals', 'snapshot_id': 'same'}
    serialize = lambda rows: '\n'.join(json.dumps(r) for r in rows).encode()
    assert len(runner.forecast_rows(serialize([row, row | {'prediction': 999}])) ) == 1
    with pytest.raises(ValueError, match='Conflicting'):
        runner.forecast_rows(serialize([row, row | {'reference': 9}]))
    with pytest.raises(ValueError, match='Fixture versions'):
        runner.forecast_rows(serialize([row, row | {'market': 'corners', 'snapshot_id': 'different'}]))


def test_input_checksums_and_existing_output_never_overwritten(tmp_path):
    (tmp_path / 'input').write_text('original')
    (tmp_path / 'COMPLETE.json').write_text(json.dumps({'input': hashlib.sha256(b'original').hexdigest()}))
    assert runner.verified(tmp_path, 'input') == b'original'
    (tmp_path / 'input').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        runner.verified(tmp_path, 'input')
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, tmp_path)


def test_sql_snapshot_excludes_future_player_values_and_is_read_only(tmp_path):
    path = tmp_path / 'platform.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE competitions(id,code); INSERT INTO competitions VALUES(1,'EPL');
        CREATE TABLE seasons(id,year); INSERT INTO seasons VALUES(1,2023);
        CREATE TABLE teams(id,api_football_id); INSERT INTO teams VALUES(1,10),(2,20);
        CREATE TABLE players(id,api_football_id); INSERT INTO players VALUES(1,101);
        CREATE TABLE fixtures(id,api_football_id,competition_id,season_id,kickoff_utc,status,home_team_id,away_team_id,updated_at);
        INSERT INTO fixtures VALUES(1,7,1,1,'2023-08-01','FT',1,2,'now'),(2,8,1,1,'2025-08-01','FT',1,2,'now');
        CREATE TABLE fixture_player_stats(id,fixture_id,team_id,player_id,minutes,yellow_cards,red_cards,raw_payload_digest,updated_at);
        INSERT INTO fixture_player_stats VALUES(1,1,1,1,90,0,NULL,'old','now'),(2,2,1,1,90,999,999,'future','now');
        CREATE TABLE fixture_team_stats(id,fixture_id,team_id,yellow_cards,red_cards,raw_payload_digest,updated_at);
        CREATE TABLE raw_payload_archive(id,endpoint,params);
        ''')
    before = path.read_bytes()
    data = runner.snapshot(path)
    assert [f['fixture_id'] for f in data['fixtures']] == [7]
    assert len(data['players']) == 1 and data['players'][0]['red_cards'] is None
    assert path.read_bytes() == before


def test_raw_archive_reserved_fixture_refused_before_file_access(tmp_path):
    with pytest.raises(ValueError, match='reserved_archive'):
        runner.read_players(tmp_path, {}, fixture() | {'kickoff': '2025-01-01'})


def test_raw_player_archive_identity_and_integrity(tmp_path):
    import gzip
    path = tmp_path / 'Index/raw_archive/players.json.gz'
    path.parent.mkdir(parents=True)
    params = {'fixture': 7}
    payload = {'errors': [], 'parameters': params, 'paging': {'total': 1},
               'response': cards.normalized_payload(players())}
    body = json.dumps(payload).encode()
    path.write_bytes(gzip.compress(body))
    archive = {'provider': 'api_football', 'endpoint': '/fixtures/players', 'storage_backend': 'local',
               'params': json.dumps(params), 'params_digest': runner.sha(json.dumps(params, sort_keys=True, separators=(',', ':')).encode()),
               'storage_uri': path.as_uri(), 'payload_digest': runner.sha(body), 'id': 1, 'fetched_at': '2026-09-28'}
    raw, ref = runner.read_players(tmp_path, archive, fixture())
    result = cards.qualify(fixture(), players(), fixture(), raw_players=raw, raw_reference=ref)
    assert result['eligible'] and result['target'] == 0
    with pytest.raises(ValueError, match='scope_mismatch'):
        runner.read_players(tmp_path, archive, fixture() | {'fixture_id': 8})
    with pytest.raises(ValueError, match='payload_checksum'):
        runner.read_players(tmp_path, archive | {'payload_digest': 'wrong'}, fixture())
    payload['parameters']['fixture'] = 8
    body = json.dumps(payload).encode()
    path.write_bytes(gzip.compress(body))
    with pytest.raises(ValueError, match='envelope_invalid'):
        runner.read_players(tmp_path, archive | {'payload_digest': runner.sha(body)}, fixture())


def test_frozen_snapshot_audit_replay_and_zero_red_bias_separation(tmp_path):
    f = fixture()
    rows = [r | {'fixture_id': 7} for r in players()]
    data = {'fixtures': [f], 'players': rows,
            'teams': [{'fixture_id': 7, 'team_id': team, 'yellow_cards': 0, 'red_cards': 0} for team in (1, 2)],
            'archives': [], 'archive_inventory': []}
    first = runner.audit(tmp_path, data, {7: f})
    assert first == runner.audit(tmp_path, deepcopy(data), {7: f})
    assert first[1]['overall']['legacy_label_red_categories'] == {'zero': 1}
    data['teams'][0]['red_cards'] = None
    changed = runner.audit(tmp_path, data, {7: f})
    assert changed[1]['overall']['legacy_label_red_categories'] == {}
    assert changed[1]['overall']['all_fixture_red_categories'] == {'unknown': 1}
