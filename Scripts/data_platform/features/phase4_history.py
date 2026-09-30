"""Dated research inputs for the unchanged Phase 4 control; no live store IO."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import re

from Scripts.data_platform.features.phase4_cards import _object_end, permitted, utc

VERSION = 'phase4-historical-control-inputs.v1'
EUROPE = {'UCL', 'UEL', 'UECL'}
START = utc('2022-01-01T00:00:00Z')
END = utc('2024-01-01T00:00:00Z')


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def development_history(text):
    """Inspect kickoff metadata before decoding objects; never decode reserve rows."""
    headers = list(re.finditer(r'"history"\s*:\s*\[', text))
    if len(headers) != 1:
        raise ValueError('Expected one history array')
    pos, kept, seen, skipped = headers[0].end(), [], set(), 0
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \r\n\t,':
            pos += 1
        if pos >= len(text):
            break
        if text[pos] == ']':
            return sorted(kept, key=lambda r: (utc(r['kickoff']), r['fixture_id'])), skipped
        if text[pos] != '{':
            raise ValueError('Invalid history object')
        end = _object_end(text, pos)
        raw = text[pos:end]
        dates = re.findall(r'"kickoff"\s*:\s*"([^"\\]+)"', raw)
        if len(dates) != 1:
            raise ValueError('Ambiguous history kickoff')
        if permitted(dates[0]):
            row = json.loads(raw)
            if row['status'] != 'FT' or row['fixture_id'] in seen:
                raise ValueError('Invalid/duplicate regulation history')
            seen.add(row['fixture_id'])
            kept.append(row)
        else:
            skipped += 1
        pos = end
    raise ValueError('Unterminated history')


def fixture_metadata(row, side):
    """Map certified provider fields to the frozen engine's fixture-row contract.

    No derived aggression/control/form or card semantics are invented. Historical
    rows are shared in the artifact so every mapped value is independently auditable.
    """
    other = 'away' if side == 'home' else 'home'
    own, opp = row[side], row[other]
    hg, ag = row['home']['goals'], row['away']['goals']
    meta = dict(team=str(row[side+'_team_id']), opponent=str(row[other+'_team_id']),
                fixture=str(row['fixture_id']), fixture_date=row['kickoff'],
                season=str(row['season']), league=row['competition'], home_away=side,
                final_score=f'{int(hg)}-{int(ag)}' if hg is not None and ag is not None else None,
                xg_for=own.get('xg'), possession=own.get('possession'),
                fouls_per_90_team=own.get('fouls'), fouls_committed=own.get('fouls'))
    for stat in ('corners', 'shots', 'sot'):
        meta[stat+'_for'], meta[stat+'_against'] = own.get(stat), opp.get(stat)
    for key in ('cards_per_90_team', 'cards_total', 'yellow_cards', 'red_cards',
                'control_index', 'aggression_index_norm', 'form_index_team'):
        meta[key] = None
    return {'meta': meta}


class HistoricalInputs:
    """Exact-ID index with a conservative date boundary and availability check."""
    def __init__(self, history, *, end=END):
        self.history = {r['fixture_id']: r for r in history}
        if len(self.history) != len(history):
            raise ValueError('Duplicate history fixture')
        self.by_team = defaultdict(list)
        self.metas = {}
        for row in history:
            if row['status'] != 'FT' or utc(row['kickoff']) + timedelta(hours=3) >= end:
                raise ValueError('Non-development or non-regulation history')
            for side in ('home', 'away'):
                team = str(row[side+'_team_id'])
                self.by_team[(team, row['competition'])].append(row)
                self.metas[(team, str(row['fixture_id']))] = fixture_metadata(row, side)
        for rows in self.by_team.values():
            rows.sort(key=lambda r: (utc(r['kickoff']), r['fixture_id']), reverse=True)

    def before(self, team, league, cutoff):
        cutoff = utc(cutoff)
        return [r for r in self.by_team.get((str(team), league), ())
                if utc(r['kickoff']).date() < cutoff.date()
                and utc(r['kickoff']) + timedelta(hours=3) < cutoff]

    def domestic(self, team, cutoff):
        # Infer membership solely from the latest completed domestic fixture.
        # This is an explicit dated reconstruction, not live alias resolution.
        options = [(utc(rows[0]['kickoff']), league) for (t, league) in self.by_team
                   if t == str(team) and league not in EUROPE
                   and (rows := self.before(team, league, cutoff))]
        if not options:
            return None
        latest = max(time for time, _ in options)
        leagues = {league for time, league in options if time == latest}
        return next(iter(leagues)) if len(leagues) == 1 else None

    def rows(self, team, league, cutoff):
        return [self.metas[(str(team), str(r['fixture_id']))] for r in self.before(team, league, cutoff)]

    def evidence(self, team, league, cutoff, target_rank):
        rows = self.before(team, league, cutoff)
        prior_rank = max((r['season'] for r in rows if r['season'] < target_rank), default=None)
        groups = {'current': [r for r in rows if r['season'] == target_rank],
                  'prior': [r for r in rows if r['season'] == prior_rank]}
        groups['recent_six'] = groups['current'][:6]
        groups['recent_eight'] = groups['current'][:8]
        result = {}
        for name, items in groups.items():
            metas = [self.metas[(str(team), str(r['fixture_id']))]['meta'] for r in items]
            counts = {key: sum(m.get(key) is not None for m in metas)
                      for key in ('xg_for', 'corners_for', 'corners_against', 'sot_for',
                                  'sot_against', 'shots_for', 'fouls_committed', 'possession',
                                  'final_score', 'cards_per_90_team', 'control_index',
                                  'aggression_index_norm', 'form_index_team')}
            result[name] = {'fixture_ids': [r['fixture_id'] for r in items],
                            'field_counts': counts,
                            'home_fixture_ids': [r['fixture_id'] for r in items if str(r['home_team_id']) == str(team)],
                            'away_fixture_ids': [r['fixture_id'] for r in items if str(r['away_team_id']) == str(team)]}
        return {'competition': league, 'current_rank': target_rank, 'prior_rank': prior_rank, **result}


def request_rows(dataset, history, *, scope='development'):
    """Separate labels from inputs; keep every permitted development fixture."""
    if scope not in ('development', 'earlier_fitting'):
        raise ValueError('Only pre-2024 development/fitting scopes are permitted')
    by_id = {r['fixture_id']: r for r in history}
    requests, targets = [], []
    for row in dataset.rows:
        fixture = row['fixture']
        kickoff = utc(fixture['kickoff'])
        if not (START <= kickoff < END if scope == 'development' else kickoff < START):
            continue
        if not permitted(fixture['kickoff']):
            raise ValueError('Target label not available inside development boundary')
        certified = by_id.get(fixture['fixture_id'])
        if certified is None and (any(v is not None for v in row['labels'].values()) or any(
                e['eligible'] for e in row['market_eligibility'].values())):
            raise ValueError('Uncertified fixture has qualified targets')
        if certified is not None and any(certified[k] != fixture[k] for k in (
                'competition', 'season', 'home_team_id', 'away_team_id', 'kickoff', 'status')):
            raise ValueError('Fixture identity disagrees with certified history')
        for side in ('home', 'away'):
            for market in ('goals', 'corners', 'sot'):
                if certified is not None and certified[side][market] != row['team_labels'][side][market]:
                    raise ValueError('Target disagrees with certified history')
        safe_fixture = {key: fixture[key] for key in ('fixture_id', 'competition', 'season',
                                                      'home_team_id', 'away_team_id', 'kickoff')}
        requests.append({'fixture': safe_fixture, 'as_of': row['as_of'],
                         'forecast_stage': row['forecast_stage'], 'availability': 'assumed_final',
                         'source_snapshot_id': row['snapshot_id'],
                         'source_eligibility': row['market_eligibility']})
        if row['as_of'] != fixture['kickoff']:
            raise ValueError('Unexpected forecast cutoff')
        targets.append({'fixture_id': fixture['fixture_id'], 'labels': row['labels'],
                        'team_labels': row['team_labels'], 'label_available_at': row['label_available_at'],
                        'actual_observed_at': row['actual_observed_at'], 'period': 'regulation_time',
                        'availability': 'assumed_final'})
    return requests, targets
