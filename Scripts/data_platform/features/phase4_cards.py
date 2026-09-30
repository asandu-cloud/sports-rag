"""Offline card-target qualification; never repairs canonical rows or fetches data."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import re

from Scripts.data_platform.participation_cards import parse_participation_cards
from Scripts.data_platform.settlement_policy import POLICY_VERSION
from Scripts.data_platform.settlement import provider_id

CONTRACT = 'phase4-participation-card-target.v1'
END = datetime(2024, 1, 1, tzinfo=timezone.utc)


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    # Canonical SQLite timestamps are stored as naive UTC.
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def permitted(kickoff):
    return utc(kickoff) + timedelta(hours=3) < END


def _object_end(text, start):
    depth, quoted, escaped = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '{[':
            depth += 1
        elif char in '}]':
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError('Truncated history')


def period_index(text):
    """Skip future JSON objects before decoding, as in the prior-transition audit.

    Caller verifies frozen artifact hash. Only retain period/identity metadata.
    """
    headers = list(re.finditer(r'"history"\s*:\s*\[', text))
    if len(headers) != 1:
        raise ValueError('Expected one history array')
    pos, result, skipped = headers[0].end(), {}, 0
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \r\n\t,':
            pos += 1
        if pos == len(text):
            break
        if text[pos] == ']':
            return result, skipped
        if text[pos] != '{':
            raise ValueError('Invalid history object')
        end = _object_end(text, pos)
        raw = text[pos:end]
        dates = re.findall(r'"kickoff"\s*:\s*"([^"\\]+)"', raw)
        if len(dates) != 1:
            raise ValueError('Ambiguous history kickoff')
        if permitted(dates[0]):
            row = json.loads(raw)
            if row['status'] != 'FT' or row['fixture_id'] in result:
                raise ValueError('Invalid or duplicate certified history')
            result[row['fixture_id']] = {k: row[k] for k in (
                'fixture_id', 'kickoff', 'competition', 'season', 'home_team_id', 'away_team_id', 'status')}
        else:
            skipped += 1
        pos = end
    raise ValueError('Unterminated history')


def normalized_payload(rows):
    teams = defaultdict(list)
    for row in rows:
        teams[row['team_id']].append({'player': {'id': row['player_id']}, 'statistics': [{
            'games': {'minutes': row['minutes']},
            'cards': {'yellow': row['yellow_cards'], 'red': row['red_cards']}}]})
    return [{'team': {'id': team}, 'players': values} for team, values in sorted(teams.items(), key=lambda x: str(x[0]))]


def qualify(fixture, rows, certified, *, raw_players=None, raw_reference=None, raw_error=None):
    """Raw evidence must already have checksum/endpoint/fixture-envelope verification.

    Normalized arithmetic is diagnostic. Only complete equivalent source evidence
    plus frozen regulation identity can produce a qualified target.
    """
    if not permitted(fixture['kickoff']):
        raise ValueError('Reserved outcome access refused')
    result = {'status': fixture['status'], **{k: provider_id(fixture[k]) for k in ('home_team_id', 'away_team_id')}}
    diagnostic = parse_participation_cards(result, normalized_payload(rows))
    reasons = []
    period_keys = ('fixture_id', 'competition', 'season', 'home_team_id', 'away_team_id', 'status')
    if certified is None:
        reasons.append('regulation_source_unverified')
    elif any(fixture[k] != certified[k] for k in period_keys) or utc(fixture['kickoff']) != utc(certified['kickoff']):
        reasons.append('regulation_source_identity_conflict')
    if raw_error:
        reasons.append(raw_error)
    if raw_players is None or raw_reference is None:
        reasons.append('original_player_response_unavailable')
        evidence = diagnostic
    else:
        evidence = parse_participation_cards(result, raw_players)
        # Missing normalized rows may be repaired in an artifact from complete raw
        # evidence; contradictory existing rows require review rather than overwrite.
        raw = {(x['team_id'], x['player_id']): (x['minutes'], x['yellow'], x['red'])
               for x in evidence['players']}
        if not evidence['pending_reason'] and any(
            raw.get((provider_id(r['team_id']), provider_id(r['player_id']))) != (r['minutes'], r['yellow_cards'], r['red_cards'])
            for r in rows):
            reasons.append('normalized_raw_player_conflict')
    if evidence['pending_reason']:
        reasons.append(evidence['pending_reason'])
    return {**fixture, 'contract': CONTRACT, 'settlement_policy': POLICY_VERSION,
            'availability': 'assumed_final', 'eligible': not reasons,
            'exclusions': sorted(set(reasons)),
            'target': sum(evidence['totals'].values()) if not reasons else None,
            'team_targets': evidence['totals'] if not reasons else None,
            'normalized_candidate_total': sum(diagnostic['totals'].values()) if not diagnostic['pending_reason'] else None,
            'normalized_pending_reason': diagnostic['pending_reason'],
            'player_row_count': len(rows), 'raw_reference': raw_reference,
            'bookmaker_equivalence': 'not_established'}
