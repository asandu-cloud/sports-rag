"""Source qualification and research-only card reconstruction; no model fitting."""
from collections import Counter
from copy import deepcopy

from Scripts.ops import player_history_download as collector
from . import phase4_cards as cards
from . import phase4_card_policy as policy
from Scripts.data_platform.participation_cards import parse_participation_cards
from Scripts.data_platform.settlement import provider_id

VERSION = 'player-history-reconciliation.v1'
RULES = {
    'version': VERSION,
    'negative_event_time': 'retain_raw_mark_partial_normalized_time_null',
    'identity': 'exact_provider_ids_no_name_matching_or_duplicate_resolution',
    'canonical_conflicts': 'preserve_existing_rows_and_store_incoming_observations',
    'missing_cards': 'unknown_never_zero',
    'card_policy': policy.POLICY,
    'development_end_exclusive': '2024-01-01T00:00:00+00:00',
    'later_periods': 'source_integrity_and_import_only_no_targets_or_performance',
    'card_evidence': 'explicit_player_counts_plus_lineup_and_card_event_reconciliation',
    'event_card_weights': 'yellow_1_straight_red_2_second_yellow_dismissal_3_total',
    'null_minutes_zero_cards': 'cannot_change_total_but_starters_need_recorded_minutes',
    'old_nonnull_card_or_minutes_conflicts': 'pending_no_silent_replacement',
    'profile_builds_backtests_optimization_promotion': False,
}


def response_blocks(payload, fixture, endpoint):
    """Accept a verified complete envelope or an older archived response list.

    The caller must verify archive hash, provider, endpoint and exact fixture
    parameters first, including for legacy lists that have no envelope identity.
    """
    if isinstance(payload, dict):
        collector.validate_identity(fixture, payload, endpoint)
        return payload['response']
    if isinstance(payload, list):
        return payload
    raise ValueError('Unsupported archived response shape')


def envelope(payload, fixture, endpoint):
    if isinstance(payload, dict):
        collector.validate_identity(fixture, payload, endpoint)
        return payload
    blocks = response_blocks(payload, fixture, endpoint)
    return {'get': endpoint.lstrip('/'), 'parameters': {'fixture': str(fixture['fixture_id'])},
            'errors': [], 'results': len(blocks), 'response': blocks,
            'paging': {'current': 1, 'total': 1}}


def classify(fixture, payload, endpoint):
    """Structural validation only; negative timing remains explicitly partial."""
    try:
        full = envelope(payload, fixture, endpoint)
        validator = {collector.ENDPOINT: collector.validate_players,
                     '/fixtures/lineups': collector.validate_lineups,
                     '/fixtures/events': collector.validate_events}[endpoint]
        state = validator(fixture, full)
        reasons = []
        if endpoint == '/fixtures/events':
            for e in full['response']:
                if any(type(v) is int and v < 0 for v in (e.get('time') or {}).values()):
                    reasons.append('negative_event_time_unknown')
            if state == 'partial':
                reasons.append('timeline_has_unresolved_identity_timing_or_duplicates')
        return {'state': state, 'error': None, 'reasons': sorted(set(reasons))}
    except (ValueError, TypeError, KeyError, AttributeError, collector.ApiFootballResponseError) as exc:
        return {'state': 'invalid', 'error': str(exc), 'reasons': [str(exc)]}


def fixture_metadata(f):
    return {'fixture_id': f['fixture_id'], 'competition': f['code'], 'season': f['season'],
            'kickoff': f['kickoff_utc'], 'status': f['status'], 'round': f['round'],
            'home_team_id': f['home_team_id'], 'away_team_id': f['away_team_id'],
            'referee': f.get('referee')}


def reconstruct(f, payloads, references, *, existing_rows=()):
    """Reconstruct permitted targets; reserved periods rejected before payload access.

    No red/yellow null repair. Explicit complete player counts supply the total;
    independently linked card events must agree with its weighted meaning.
    Unknown actor/timing, missing participation or source conflict keeps it pending.
    """
    meta = fixture_metadata(f)
    if not cards.permitted(meta['kickoff']):
        raise ValueError('Reserved outcome access refused')
    scoped = policy.scope(meta)
    result = {**meta, 'reconstruction_version': VERSION, 'policy': deepcopy(policy.POLICY),
              'availability': 'assumed_final', 'round_scope': scoped, 'eligible': False,
              'target': None, 'team_targets': None, 'exclusions': list(scoped['exclusions']),
              'source_references': references, 'bookmaker_equivalence': 'not_established'}
    if not scoped['eligible']:
        return result
    reasons = set()
    blocks = {}
    for ep in collector.ENDPOINTS:
        if ep not in payloads or ep not in references:
            reasons.add('missing_source:' + ep)
            continue
        outcome = classify(f, payloads[ep], ep)
        if outcome['state'] not in ('ready', 'partial'):
            reasons.add(outcome['state'] + ':' + ep)
        blocks[ep] = response_blocks(payloads[ep], f, ep)
    if reasons:
        result['exclusions'] = sorted(reasons)
        return result
    players, lineups, events = [blocks[e] for e in collector.ENDPOINTS]
    result_identity = {**meta, **{k: provider_id(meta[k]) for k in ('home_team_id', 'away_team_id')}}
    evidence = parse_participation_cards(result_identity, players, minimum_recorded_minutes=2)
    result['player_evidence'] = evidence
    if evidence['pending_reason']:
        reasons.add(evidence['pending_reason'])
    by_player = {}
    for team in players:
        for row in team['players']:
            s = row['statistics'][0]
            by_player[row['player']['id']] = (team['team']['id'], (s.get('games') or {}).get('minutes'), s.get('cards') or {})
    roster, starters, coaches = {}, set(), {}
    for team in lineups:
        tid = team['team']['id']
        coach = team.get('coach') or {}
        if collector.positive_id(coach.get('id')) and coach.get('name'):
            coaches[(tid, coach['id'])] = coach['name']
        for field in ('startXI', 'substitutes'):
            for row in team[field]:
                pid = row['player']['id']
                roster[pid] = tid
                if field == 'startXI':
                    starters.add(pid)
    for pid in starters:
        if pid not in by_player or by_player[pid][0] != roster[pid]:
            reasons.add('starting_player_statistics_missing_or_wrong_team')
        elif by_player[pid][1] is None:
            reasons.add('starting_player_minutes_unknown')
    for pid, (tid, minutes, counts) in by_player.items():
        if minutes is not None and minutes > 0 and roster.get(pid) != tid:
            reasons.add('participant_missing_from_lineup')
    for old in existing_rows:
        pid, tid = old['player_id'], old['team_id']
        if pid not in by_player:
            if old.get('minutes') is not None and old['minutes'] >= 2:
                reasons.add('existing_participant_missing_from_new_source')
            continue
        newtid, minutes, counts = by_player[pid]
        if tid != newtid:
            reasons.add('existing_player_team_conflict')
        for key, new in (('minutes', minutes), ('yellow_cards', counts.get('yellow')),
                         ('red_cards', counts.get('red'))):
            if old.get(key) is not None and old[key] != new:
                reasons.add('existing_nonnull_' + key + '_conflict')
    event_counts, seen, substitutions = {}, set(), set()
    event_evidence = []
    for e in events:
        tid, pid = (e.get('team') or {}).get('id'), (e.get('player') or {}).get('id')
        if e['type'] == 'subst':
            for role in ('player', 'assist'):
                who = (e.get(role) or {}).get('id')
                if who in roster and roster[who] == tid:
                    substitutions.add(who)
            continue
        if e['type'] != 'Card':
            continue
        key = collector.digest(e)
        if key in seen:
            reasons.add('duplicate_card_event')
        seen.add(key)
        if pid not in by_player:
            # Exact coach identity from the independently supplied lineup is
            # evidence of staff; a missing roster entry alone is not evidence.
            if ((tid, pid) in coaches and coaches[tid, pid] == (e.get('player') or {}).get('name')
                    and pid not in roster):
                event_evidence.append({'player_id': pid, 'reason': 'exact_lineup_coach_identity'})
                continue
            reasons.add('card_actor_without_player_participation_evidence')
            continue
        player_team, minutes, _ = by_player[pid]
        if player_team != tid:
            reasons.add('card_event_team_conflict')
        if minutes is not None and minutes < 2:
            event_evidence.append({'player_id': pid, 'reason': 'below_two_minutes'})
            continue
        if minutes is None:
            reasons.add('card_event_player_minutes_unknown')
        if collector.usable_event_time((e.get('time') or {}).get('elapsed')) is None:
            reasons.add('eligible_card_event_time_unknown')
        extra = (e.get('time') or {}).get('extra')
        if extra is not None and collector.usable_event_time(extra) is None:
            reasons.add('eligible_card_event_extra_time_unknown')
        counts = event_counts.setdefault(pid, Counter())
        counts[e.get('detail')] += 1
    for pid, (tid, minutes, counts) in by_player.items():
        if minutes is not None and minutes > 0 and pid not in starters and pid not in substitutions:
            reasons.add('playing_substitute_without_substitution_evidence')
        if minutes is None or minutes < 2:
            continue
        ec = event_counts.get(pid, Counter())
        if set(ec) - {'Yellow Card', 'Red Card', 'Yellow-Red Card'}:
            reasons.add('unknown_card_event_detail')
        y, r, yr = ec['Yellow Card'], ec['Red Card'], ec['Yellow-Red Card']
        if y > 2 or r > 1 or yr > 1 or (yr and (r or y == 0)) or (y == 2 and not (r or yr)):
            reasons.add('inconsistent_card_event_sequence')
        weighted = 3 if yr else min(y + 2 * r, 3)
        cy, cr = counts.get('yellow'), counts.get('red')
        if cy is not None and cr is not None and weighted != min(cy + 2 * cr, 3):
            reasons.add('player_and_event_weighted_cards_conflict')
    result['event_exclusions_evidence'] = event_evidence
    result['exclusions'] = sorted(reasons)
    if not reasons:
        result.update(eligible=True, target=sum(evidence['totals'].values()), team_targets=evidence['totals'])
    return result
