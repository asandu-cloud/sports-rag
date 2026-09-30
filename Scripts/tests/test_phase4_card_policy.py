"""Two-minute candidate, retained production policy and phase-scoped evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from Scripts.data_platform.participation_cards import parse_participation_cards
from Scripts.data_platform.settlement_policy import new_publication_policy, product_policy
from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features import phase4_card_policy as policy
from Scripts.data_platform.features import phase4_card_reconstruction as model
from Scripts.ops import phase4_card_policy as cli


def fixture():
    return {'fixture_id': 1, 'competition': 'EPL', 'season': 2023, 'kickoff': '2023-09-01T12:00:00Z',
            'home_team_id': 10, 'away_team_id': 20, 'status': 'FT', 'round': 'Regular Season - 4'}


def rows():
    return [{'player_id': tid * 100 + i, 'team_id': tid, 'minutes': 90, 'yellow_cards': 0, 'red_cards': 0}
            for tid in (10, 20) for i in range(11)]


def parse(players, minimum=2):
    result = {'status': 'FT', 'home_team_id': '10', 'away_team_id': '20'}
    return parse_participation_cards(result, cards.normalized_payload(players), minimum_recorded_minutes=minimum)


@pytest.mark.parametrize('minutes,expected', [(0, 0), (1, 0), (2, 2), (90, 2)])
def test_minute_boundary_retains_actual_recorded_evidence(minutes, expected):
    players = rows()
    players[0].update(minutes=minutes, red_cards=1)
    original = deepcopy(players)
    result = parse(players)
    assert result['totals']['home'] == expected and result['pending_reason'] is None
    assert result['players'][0]['minutes'] == minutes and players == original
    assert result['minimum_recorded_minutes'] == 2


@pytest.mark.parametrize('yellow,red,expected', [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3), (2, 1, 3)])
def test_approved_weighted_arithmetic_preserved_at_two_minutes(yellow, red, expected):
    players = rows()
    players[0].update(minutes=2, yellow_cards=yellow, red_cards=red)
    assert parse(players)['totals']['home'] == expected


def test_missing_red_remains_unknown_for_eligible_player():
    players = rows()
    players[0].update(minutes=2, yellow_cards=0, red_cards=None)
    result = parse(players)
    assert result['pending_reason'] == 'missing_player_card_counts' and result['totals'] == {}
    players[0]['red_cards'] = 0
    assert parse(players)['totals']['home'] == 0


def test_below_threshold_unknown_cards_cannot_affect_target_but_unknown_minutes_can():
    players = rows()
    players[0].update(minutes=1, yellow_cards=None, red_cards=None)
    assert parse(players)['totals']['home'] == 0
    assert parse(players, 1)['pending_reason'] == 'missing_player_card_counts'
    players[0].update(minutes=None, yellow_cards=1, red_cards=0)
    assert parse(players)['pending_reason'] == 'missing_carded_player_minutes'
    players[0]['yellow_cards'] = 0
    assert parse(players)['totals']['home'] == 0


def test_production_default_and_publication_policy_not_activated():
    players = rows()
    players[0].update(minutes=1, red_cards=1)
    result = {'status': 'FT', 'home_team_id': '10', 'away_team_id': '20'}
    assert parse_participation_cards(result, cards.normalized_payload(players)) == parse(players, 1)
    assert parse(players, 1)['totals']['home'] == 2
    live = new_publication_policy()
    assert live['version'] == 'spix-participation-settlement.v1'
    assert live['card_eligibility'] == 'recorded_minutes_greater_than_zero'
    assert product_policy({'settlement_policy': policy.POLICY}) is None


@pytest.mark.parametrize('minimum', [0, 3, True, 2.0, None])
def test_unsupported_or_ambiguous_minimum_rejected(minimum):
    with pytest.raises(ValueError, match='minimum'):
        parse(rows(), minimum)


@pytest.mark.parametrize('competition,season,round_name,allowed', [
    ('EPL', 2023, 'Regular Season - 1', True),
    ('Championship', 2023, 'Promotion Play-offs - Final', False),
    ('Bundesliga', 2023, 'Relegation Round', False),
    ('BelgianProLeague', 2023, 'Championship Round - 2', True),
    ('BelgianProLeague', 2023, 'Conference League Play-offs - Final', False),
    ('UCL', 2023, 'Group Stage - 1', True),
    ('UEL', 2023, 'Group A - 3', True),
    ('UECL', 2024, 'League Stage - 2', True),
    ('UECL', 2023, 'League Stage - 2', False),
    ('UCL', 2023, 'Quarter-finals', False),
    ('UEL', 2023, 'Knockout Round Play-offs', False),
    ('UCL', 2023, '1st Qualifying Round', False),
    ('EPL', 2023, None, False),
])
def test_known_regular_formats_only(competition, season, round_name, allowed):
    f = fixture() | {'competition': competition, 'season': season, 'round': round_name}
    assert policy.scope(f)['eligible'] is allowed


@pytest.mark.parametrize('status', ['AET', 'PEN', 'ABD', 'PST', 'NS'])
def test_non_regulation_completion_not_admitted(status):
    assert not policy.scope(fixture() | {'status': status})['eligible']


def qualify(f=None, players=None, metadata=None, raw=True):
    f = f or fixture()
    players = players or rows()
    return policy.qualify(f, players, metadata or f | {'metadata_reference': {'snapshot': 'unit-test'}},
                          raw_players=cards.normalized_payload(players) if raw else None,
                          raw_reference={'payload_sha256': 'test-evidence'} if raw else None)


def test_known_league_format_resolves_period_but_not_missing_player_response():
    result = qualify(raw=False)
    assert result['period_evidence']['established']
    assert result['exclusions'] == ['original_player_response_unavailable']
    assert result['normalized_candidate_total'] == 0 and result['target'] is None
    qualified = qualify()
    assert qualified['eligible'] and qualified['target'] == 0
    assert qualified['settlement_policy'] == policy.POLICY_VERSION
    assert model.qualified_rows([qualified], minimum_recorded_minutes=2) == [qualified]
    with pytest.raises(ValueError, match='contract'):
        model.qualified_rows([qualified])


def test_identity_conflicts_still_fail_and_knockout_never_enters_context():
    f = fixture()
    result = qualify(metadata=f | {'home_team_id': 999})
    assert not result['eligible'] and not result['period_evidence']['established']
    result = qualify(metadata=f | {'round': 'Final'})
    assert 'round_metadata_identity_conflict' in result['exclusions']
    result = qualify(f | {'competition': 'UCL', 'round': 'Quarter-finals'})
    assert 'special_round_deferred' in result['exclusions']
    assert model.qualified_rows([result], minimum_recorded_minutes=2) == []
    forged = result | {'eligible': True, 'exclusions': [], 'target': 0, 'team_targets': {'home': 0, 'away': 0}}
    with pytest.raises(ValueError, match='scope'):
        model.qualified_rows([forged], minimum_recorded_minutes=2)


def test_two_minute_targets_are_used_by_reconciliation_not_just_labelled():
    players = rows()
    players[0].update(minutes=1, red_cards=1)
    result = qualify(players=players)
    combined = model.reconcile([result], [r | {'fixture_id': 1} for r in players], [], minimum_recorded_minutes=2)[0]
    assert result['target'] == combined['local_provisional_total'] == 0
    assert combined['contract'] == policy.CONTRACT


def test_v2_backtest_profile_validation_and_cohort_gates():
    from Scripts.tests.test_phase4_card_reconstruction import synthetic_history, weights
    history = [r | {'round': 'Regular Season - 1', 'contract': policy.CONTRACT,
                    'settlement_policy': policy.POLICY_VERSION} for r in synthetic_history()]
    report, predictions, profiles = model.backtest(history, weights(), minimum_recorded_minutes=2)
    assert report['status'] == 'development_diagnostic_complete'
    assert report['minimum_recorded_minutes'] == 2 and report['weight_optimisation'] is False
    assert len(predictions) == len(profiles) >= 200
    assert report['settlement_policy'] == policy.POLICY_VERSION


def test_inventory_is_presence_only_and_does_not_certify_present_response():
    past = fixture() | {'normalized_player_rows': 22}
    future = fixture() | {'fixture_id': 2, 'season': 2025, 'kickoff': '2025-09-01', 'normalized_player_rows': 30}
    metadata = {'fixtures': [past, future], 'player_archive_metadata': [
        {'id': 7, 'provider': 'api_football', 'params': '{"fixture": 2}'}]}
    inventory, report = cli.inventory(metadata, [{'fixture_id': 1}])
    assert inventory[0]['missing_archived_player_response']
    assert inventory[0]['legacy_rows_in_permitted_artifact'] == 1
    assert inventory[1]['outcome_inspection'] == 'metadata_only'
    assert not inventory[1]['archive_contents_verified']
    assert not report['collection_authorized']


def test_v1_parser_matches_saved_original_outputs():
    root = Path(__file__).resolve().parents[2]
    path = root / 'Research/phase4-card-reconstruction-2026-09-30/reconstruction/source/Scripts/data_platform/participation_cards.py'
    namespace = {'__package__': 'Scripts.data_platform'}
    exec(compile(path.read_bytes(), str(path), 'exec'), namespace)
    original = namespace['parse_participation_cards']
    result = {'status': 'FT', 'home_team_id': '10', 'away_team_id': '20'}
    for minute in (None, 0, 1, 2, 90, 151):
        for yellow, red in ((0, 0), (1, 0), (0, 1), (2, 1), (2, 0), (0, None)):
            players = rows()
            players[0].update(minutes=minute, yellow_cards=yellow, red_cards=red)
            raw = cards.normalized_payload(players)
            assert parse_participation_cards(result, raw) == original(result, raw)


def test_metadata_snapshot_does_not_select_outcomes(tmp_path):
    path = tmp_path / 'test.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE competitions(id,code); INSERT INTO competitions VALUES(1,'EPL');
        CREATE TABLE seasons(id,year); INSERT INTO seasons VALUES(1,2025);
        CREATE TABLE teams(id,api_football_id,name); INSERT INTO teams VALUES(1,10,'A'),(2,20,'B');
        CREATE TABLE fixtures(id,api_football_id,competition_id,season_id,kickoff_utc,status,round,home_team_id,away_team_id,referee,home_goals);
        INSERT INTO fixtures VALUES(1,7,1,1,'2025-09-01','FT','Regular Season - 1',1,2,'Ref','SECRET_GOALS');
        CREATE TABLE fixture_player_stats(fixture_id,red_cards); INSERT INTO fixture_player_stats VALUES(1,'SECRET_REDS');
        CREATE TABLE raw_payload_archive(id,provider,endpoint,params,storage_uri,payload_digest,fetched_at);
        ''')
    before = path.read_bytes()
    result = cli.metadata_snapshot(path)
    assert result['fixtures'][0]['normalized_player_rows'] == 1
    assert 'SECRET' not in json.dumps(result) and path.read_bytes() == before
