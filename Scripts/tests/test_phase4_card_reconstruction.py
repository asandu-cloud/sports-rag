"""Chronology, evidence and isolation tests for the local card reconstruction."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from Scripts.data_platform.features import phase4_card_reconstruction as m
from Scripts.data_platform.features import phase4_cards as cards
from Scripts.ops import phase4_card_reconstruction as cli


def fixture(fid=1, kickoff='2023-08-01T12:00:00Z'):
    return dict(fixture_id=fid, competition='EPL', season=2023, kickoff=kickoff,
                status='FT', home_team_id=10, away_team_id=20, home_name='Alpha',
                away_name='Beta', referee='A. Referee, England')


def players():
    return [dict(team_id=tid, player_id=tid * 100 + i, minutes=90, yellow_cards=0, red_cards=0)
            for tid in (10, 20) for i in range(11)]


def target(fid=1, kickoff='2023-08-01T12:00:00Z', total=3):
    f = fixture(fid, kickoff)
    f['season'] = int(kickoff[:4])
    return {**f, 'eligible': True, 'contract': cards.CONTRACT,
            'settlement_policy': 'spix-participation-settlement.v1', 'exclusions': [],
            'raw_reference': {'payload_sha256': 'synthetic-test-evidence'},
            'target': total, 'team_targets': {'home': total, 'away': 0}}


def weights():
    return cli.literal_weights(Path(__file__).resolve().parents[2])


def test_legacy_future_and_ambiguous_outcomes_never_decode(monkeypatch):
    identities = m.identity_map([fixture(), fixture(2, '2024-05-01T12:00:00Z') | {'home_name': 'Gamma'},
                                 fixture(3) | {'home_name': 'Repeated'}, fixture(4) | {'home_name': 'Repeated'}])
    rows = [{'fixture': 'Alpha vs Beta', 'team': 'Alpha', 'player_id': 101, 'minutes': 90,
             'yellow_cards': 0, 'red_cards': None},
            {'fixture': 'Gamma vs Beta', 'team': 'Gamma', 'player_id': 102, 'yellow_cards': 'SECRET_FUTURE'},
            {'fixture': 'Repeated vs Beta', 'team': 'Repeated', 'player_id': 103, 'yellow_cards': 'SECRET_AMBIGUOUS'},
            {'fixture': 'Unknown vs Beta', 'team': 'Unknown', 'player_id': 104, 'yellow_cards': 'SECRET_UNKNOWN'}]
    raw = json.dumps(rows)
    decode = json.loads
    def guarded(text, *args, **kwargs):
        assert 'SECRET_' not in text
        return decode(text, *args, **kwargs)
    monkeypatch.setattr(m.json, 'loads', guarded)
    admitted, excluded, counts = m.legacy_rows(raw, competition='EPL', season=2023, identities=identities, source='test')
    assert len(admitted) == 1 and admitted[0]['red_cards'] is None and admitted[0]['yellow_cards'] == 0
    assert {r['reason'] for r in excluded} == {'reserved_period_not_decoded', 'ambiguous_fixture_identity', 'unresolved_fixture_identity'}
    assert counts['rows'] == 4


def test_provider_identity_reader_does_not_decode_winner_or_score(monkeypatch):
    provider = [{'fixture': {'id': 1, 'date': '2025-01-01', 'status': {'short': 'FT'}, 'referee': 'A'},
                 'league': {'season': 2024}, 'teams': {
                     'home': {'id': 10, 'name': 'Alpha', 'winner': 'SECRET_WINNER'},
                     'away': {'id': 20, 'name': 'Beta', 'winner': 'SECRET_WINNER'}},
                 'score': 'SECRET_SCORE', 'goals': 'SECRET_GOALS'}]
    raw = json.dumps(provider)
    decode = json.loads
    def guarded(text, *args, **kwargs):
        assert 'SECRET_' not in text
        return decode(text, *args, **kwargs)
    monkeypatch.setattr(m.json, 'loads', guarded)
    assert cli.fixture_aliases(raw, 'EPL')[0]['fixture_id'] == 1


def test_aliases_require_same_provider_identity_and_do_not_fuzzy_merge():
    f = fixture()
    aliases = [f | {'home_name': 'Álpha'}]
    identities = m.identity_map([f], aliases)
    rows = [{'fixture': 'Álpha vs Beta', 'team': 'Alpha', 'player_id': 1001, 'minutes': 90, 'yellow_cards': 0, 'red_cards': 0},
            {'fixture': 'Alfa vs Beta', 'team': 'Alpha', 'player_id': 1002}]
    admitted, _, counts = m.legacy_rows(json.dumps(rows), competition='EPL', season=2023, identities=identities, source='test')
    assert admitted[0]['team_id'] == 10 and counts['unresolved_fixture_identity'] == 1
    assert m.referee_key('A. Taylor, England') != m.referee_key('Anthony Taylor, England')


def test_only_explicit_competition_registry_aliases_resolve_team_rows():
    f = fixture() | {'home_name': 'Hellas Verona', 'competition': 'SerieA'}
    index = m.identity_map([f], team_aliases={'SerieA': {'Verona': 'Hellas Verona'}})
    raw = json.dumps([{'fixture': 'Hellas Verona vs Beta', 'team': 'Verona', 'player_id': 1001,
                       'minutes': 90, 'yellow_cards': 0, 'red_cards': 0}])
    admitted, rejected, _ = m.legacy_rows(raw, competition='SerieA', season=2023, identities=index, source='test')
    assert admitted[0]['team_id'] == 10 and not rejected
    index = m.identity_map([f], team_aliases={'EPL': {'Verona': 'Hellas Verona'}})
    admitted, rejected, _ = m.legacy_rows(raw, competition='SerieA', season=2023, identities=index, source='test')
    assert not admitted and rejected[0]['reason'] == 'unresolved_team_identity'


def test_filtered_exports_cannot_certify_targets_even_when_arithmetic_passes():
    f = fixture()
    canonical = cards.qualify(f, [], f)
    canonical.update(team_red_category='unknown')
    legacy = [dict(r, fixture_id=1, source='legacy') for r in players()]
    legacy[0].update(yellow_cards=2, red_cards=1)
    result = m.reconcile([canonical], [], legacy)[0]
    assert result['local_provisional_total'] == 3
    assert not result['eligible'] and result['target'] is None
    assert 'legacy_participant_completeness_unverified' in result['evidence_sources'][0]['source_restrictions']
    assert m.backtest([result], weights())[0]['metrics'] is None


def test_equal_totals_do_not_hide_player_conflicts():
    f, rows = fixture(), players()
    rows[0]['yellow_cards'] = 1
    canonical = cards.qualify(f, rows, f)
    legacy = [dict(r, fixture_id=1, source='legacy') for r in rows]
    legacy[0]['yellow_cards'] = 0
    legacy[1]['yellow_cards'] = 1
    result = m.reconcile([canonical], [dict(r, fixture_id=1) for r in rows], legacy)[0]
    assert result['local_evidence_conflict'] and result['local_provisional_total'] is None


@pytest.mark.parametrize('field,value', [('minutes', None), ('red_cards', None)])
def test_unknown_cards_and_carded_minutes_are_not_repaired(field, value):
    f, rows = fixture(), players()
    rows[0]['yellow_cards'] = 1
    rows[0][field] = value
    canonical = cards.qualify(f, rows, f)
    result = m.reconcile([canonical], [dict(r, fixture_id=1) for r in rows], [])[0]
    assert result['local_provisional_total'] is None


def test_zero_participant_excluded_one_minute_participant_included():
    rows = players()
    rows[0].update(minutes=1, yellow_cards=0, red_cards=1)
    rows[1].update(minutes=0, yellow_cards=2, red_cards=1)
    assert m.evidence(fixture(), rows)['totals']['home'] == 2


@pytest.mark.parametrize('mutate', [lambda r: r.update(target=999), lambda r: r.update(raw_reference=None),
                                   lambda r: r.update(status='AET'), lambda r: r.update(exclusions=['missing'])])
def test_numeric_boundary_rejects_false_qualification(mutate):
    row = target()
    mutate(row)
    with pytest.raises(ValueError, match='contract'):
        m.qualified_rows([row])


def test_versions_and_reserved_outcomes_rejected():
    with pytest.raises(ValueError, match='versions'):
        m.qualified_rows([target(), target()])
    with pytest.raises(ValueError, match='Reserved'):
        m.qualified_rows([target(kickoff='2025-01-01T12:00:00Z')])


def test_shared_settlement_parser_float_counts_are_valid_integers():
    f, rows = fixture(), players()
    rows[0]['yellow_cards'] = 1
    qualified = cards.qualify(f, rows, f, raw_players=cards.normalized_payload(rows),
                              raw_reference={'payload_sha256': 'synthetic-test-evidence'})
    assert m.qualified_rows([qualified])[0]['target'] == 1


def test_process_guard_blocks_network_secrets_writable_databases_and_outside_writes(tmp_path):
    import subprocess
    import sys
    script = '''
from pathlib import Path
import socket,sqlite3,sys
from Scripts.ops.phase4_card_reconstruction import local_guard
out=Path(sys.argv[1]).resolve()
local_guard(out)
operations=[lambda:socket.socket(), lambda:sqlite3.connect(str(out/'forbidden.db')),
            lambda:(out.parent/'forbidden-write').write_text('x'),lambda:(out/'.env').read_text()]
for operation in operations:
    try: operation()
    except RuntimeError: pass
    else: raise AssertionError('Guard did not block operation')
(out/'allowed').write_text('ok')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'allowed').read_text() == 'ok'


def test_own_result_future_results_simultaneous_and_availability_cannot_enter_inputs():
    f = target(5, '2023-08-10T12:00:00Z')
    past = target(1, '2023-08-08T12:00:00Z')
    late_previous_day = target(2, '2023-08-09T23:00:00Z')
    same_day = target(3, '2023-08-10T01:00:00Z')
    simultaneous = target(4, f['kickoff'])
    later = target(6, '2023-08-12T12:00:00Z')
    history = [past, late_previous_day, same_day, simultaneous, f, later]
    inputs = m.dated_inputs(f, history)
    assert inputs['source_fixture_ids'] == [1]
    changed = [past, *[r | {'target': 100, 'team_targets': {'home': 50, 'away': 50}} for r in history[1:]]]
    assert m.dated_inputs(f, changed) == inputs
    assert m.dated_inputs(f, list(reversed(history))) == inputs


def test_league_and_season_contexts_do_not_cross_and_sparse_ref_is_neutral():
    rows = [target(i + 1, f'2023-08-{i+1:02}T12:00:00Z', 4) for i in range(9)]
    rows += [target(100, '2023-08-01T12:00:00Z') | {'competition': 'LaLiga'},
             target(101, '2021-08-01T12:00:00Z')]
    f = target(102, '2023-08-20T12:00:00Z') | {'referee': 'Another referee'}
    inputs = m.dated_inputs(f, rows)
    prediction, reason = m.fixed_prediction(f, inputs, weights())
    assert reason is None and len(inputs['source_fixture_ids']) == 9
    assert prediction['team_only_mean'] == prediction['team_referee_mean']
    assert prediction['referee']['reason'] == 'sparse_referee'
    assert prediction['team_only_mean'] == pytest.approx(4)


def test_referee_uses_prior_qualified_rates_and_existing_confidence_formula():
    rows = [target(i + 1, f'2023-08-{i+1:02}T12:00:00Z', 4 if i < 10 else 2) |
            {'referee': 'A. Referee' if i < 10 else 'B. Referee'} for i in range(20)]
    f = target(21, '2023-09-01T12:00:00Z')
    prediction, reason = m.fixed_prediction(f, m.dated_inputs(f, rows), weights())
    assert reason is None
    ref = prediction['referee']
    assert ref['matches'] == 10 and ref['mean'] == 4 and ref['league_mean'] == 3
    assert ref['confidence'] == pytest.approx((10 - 5) / (20 - 5))
    assert ref['multiplier'] == pytest.approx(1 + .35 * ref['confidence'] * (4 / 3 - 1))


def synthetic_history():
    rows = []
    for year, weeks in ((2022, 52), (2023, 30)):
        start = datetime(year, 1, 3, 12, tzinfo=timezone.utc)
        for week in range(weeks):
            for day in range(10):
                kickoff = start + timedelta(weeks=week, days=day // 2, hours=day % 2)
                if kickoff.year != year:
                    continue
                row = target(len(rows) + 1, kickoff.isoformat(), (week + day) % 8)
                row['team_targets'] = {'home': row['target'] // 2, 'away': row['target'] - row['target'] // 2}
                row['referee'] = 'A. Referee' if day % 2 else 'B. Referee'
                rows.append(row)
    return rows


def test_full_fixed_comparison_is_paired_reproducible_and_has_no_weight_search():
    rows = synthetic_history()
    report, predictions, profiles = m.backtest(rows, weights())
    assert report['status'] == 'development_diagnostic_complete'
    assert report['scored_fixtures'] == len(predictions) >= 200
    assert len(profiles) == len(predictions)
    assert report['metrics']['team_only']['fixtures'] == report['metrics']['team_plus_referee']['fixtures']
    assert report['weight_optimisation'] is False and report['production_qualification'] is False
    assert (report, predictions, profiles) == m.backtest(deepcopy(rows), weights())
    assert sum(b['n'] for b in report['metrics']['team_only']['reliability']) == len(predictions)


def test_no_earlier_fitting_cohort_is_an_explicit_block_not_a_tiny_test():
    rows = [r for r in synthetic_history() if r['season'] == 2023]
    report, predictions, profiles = m.backtest(rows, weights())
    assert report['status'] == 'blocked_evidence_or_support'
    assert 'insufficient_qualified_2022_fitting_history' in report['reasons']
    assert predictions == profiles == [] and report['metrics'] is None


def test_existing_experiment_never_overwritten(tmp_path):
    (tmp_path / 'sentinel').write_text('original')
    with pytest.raises(FileExistsError):
        cli.run(tmp_path, tmp_path)
    assert (tmp_path / 'sentinel').read_text() == 'original'


def test_sql_cutoff_readonly_transaction_and_metadata_only_later_period(tmp_path):
    path = tmp_path / 'platform.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE competitions(id,code); INSERT INTO competitions VALUES(1,'EPL');
        CREATE TABLE seasons(id,year); INSERT INTO seasons VALUES(1,2023);
        CREATE TABLE teams(id,api_football_id,name); INSERT INTO teams VALUES(1,10,'Alpha'),(2,20,'Beta');
        CREATE TABLE players(id,api_football_id); INSERT INTO players VALUES(1,101);
        CREATE TABLE fixtures(id,api_football_id,competition_id,season_id,kickoff_utc,status,home_team_id,away_team_id,updated_at,referee);
        INSERT INTO fixtures VALUES(1,7,1,1,'2023-08-01','FT',1,2,'now','A'),(2,8,1,1,'2025-08-01','FT',1,2,'now','B');
        CREATE TABLE fixture_player_stats(id,fixture_id,team_id,player_id,minutes,yellow_cards,red_cards,raw_payload_digest,updated_at);
        INSERT INTO fixture_player_stats VALUES(1,1,1,1,90,0,NULL,'old','now'),(2,2,1,1,90,999,999,'future','now');
        CREATE TABLE fixture_team_stats(id,fixture_id,team_id,yellow_cards,red_cards,raw_payload_digest,updated_at);
        CREATE TABLE raw_payload_archive(id,endpoint,params);
        ''')
    before = path.read_bytes()
    data = cli.snapshot(path)
    assert len(data['identity_metadata']) == 2
    assert len(data['fixtures']) == len(data['players']) == 1
    assert data['players'][0]['red_cards'] is None and '999' not in json.dumps(data)
    assert path.read_bytes() == before
