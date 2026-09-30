"""Streaming integrity/coverage audit of a completed historical control run."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import timedelta
from itertools import zip_longest
import json

from Scripts.data_platform.features.phase4_history import digest, utc


def validate_history_membership(snapshot, history):
    cutoff = utc(snapshot['as_of'])
    for team, leagues in snapshot['history_evidence'].items():
        for league, evidence in leagues.items():
            for group in ('current', 'prior', 'recent_six', 'recent_eight'):
                ids = evidence[group]['fixture_ids']
                if len(ids) != len(set(ids)):
                    raise ValueError('Duplicate contributing fixture')
                for fid in ids:
                    row = history[fid]
                    if str(team) not in (str(row['home_team_id']), str(row['away_team_id'])) or row['competition'] != league:
                        raise ValueError('History team/competition identity mismatch')
                    kickoff = utc(row['kickoff'])
                    if kickoff.date() >= cutoff.date() or kickoff+timedelta(hours=3) >= cutoff:
                        raise ValueError('Future/same-day/unavailable result in snapshot')
                    rank = evidence['prior_rank'] if group == 'prior' else evidence['current_rank']
                    if row['season'] != rank:
                        raise ValueError('History season mismatch')
            if evidence['recent_six']['fixture_ids'] != evidence['current']['fixture_ids'][:6]:
                raise ValueError('Recent-six membership differs')
            if evidence['recent_eight']['fixture_ids'] != evidence['current']['fixture_ids'][:8]:
                raise ValueError('Recent-eight membership differs')


def audit(output):
    history = {r['fixture_id']: r for r in map(json.loads, (output/'history.jsonl').open())}
    report = json.loads((output/'reproduction/report.json').read_text())
    counters = defaultdict(Counter)
    ids, snapshots_seen = set(), set()
    names = ('requests.jsonl', 'targets.jsonl', 'reproduction/snapshots.jsonl',
             'reproduction/control.jsonl', 'reproduction/eligibility.jsonl')
    with ExitStack() as stack:
        streams = [stack.enter_context((output/name).open()) for name in names]
        for lines in zip_longest(*streams):
            if any(line is None for line in lines):
                raise ValueError('Artifact row counts differ')
            request, target, snapshot, control, eligibility = map(json.loads, lines)
            fid = request['fixture']['fixture_id']
            if fid in ids or any(r['fixture_id'] != fid for r in (target, control, eligibility)):
                raise ValueError('Duplicated or misaligned fixture')
            ids.add(fid)
            if snapshot['fixture'] != request['fixture'] or snapshot['as_of'] != request['as_of']:
                raise ValueError('Snapshot fixture/cutoff differs')
            snapshot_id = snapshot['snapshot_id']
            if snapshot_id in snapshots_seen or digest({k:v for k,v in snapshot.items() if k != 'snapshot_id'}) != snapshot_id:
                raise ValueError('Snapshot identity mismatch')
            snapshots_seen.add(snapshot_id)
            if any(r['snapshot_id'] != snapshot_id for r in (control, eligibility)):
                raise ValueError('Forecast input identity mismatch')
            if utc(snapshot['as_of']) >= utc('2024-01-01T00:00:00Z'):
                raise ValueError('Reserved forecast found')
            validate_history_membership(snapshot, history)
            year = snapshot['fixture']['kickoff'][:4]
            counters['fixtures_by_year'][year] += 1
            if snapshot['fixture']['season'] != next(iter(next(iter(snapshot['history_evidence'].values())).values()))['current_rank']:
                counters['limitations']['provider_season_differs_from_control_calendar_rank'] += 1
            for team, profile in snapshot['profile_quality'].items():
                counters['profile_modes'][profile['profile_mode']] += 1
                if snapshot['recent'][team].get('xg_for_avg') is None:
                    counters['limitations']['team_recent_xg_unavailable'] += 1
            for market, status in eligibility['markets'].items():
                for reason in status['reasons']:
                    counters[market+'_exclusions'][reason] += 1
                if status['eligible']:
                    if target['labels'][market] is None or market not in control['distributions']:
                        raise ValueError('Eligible fixture lacks target/control distribution')
                    counters['eligible_by_year'][year+':'+market] += 1
                if market in control['distributions']:
                    counters['distribution_available'][market] += 1
            if target['labels']['cards'] is not None or eligibility['markets']['cards']['eligible']:
                raise ValueError('Unqualified cards admitted')
            if len(control['results']) != 7:
                raise ValueError('Incomplete canonical result set')
            for result in control['results']:
                if result['decision']['status'] == 'recommended' or result['decision'].get('price'):
                    raise ValueError('Unpriced research output contains recommendation/price')
                group = result['market']['group']
                quality = result['context'].get('data_quality', {}).get('recommendation_guardrail', {})
                counters['public_quality_eligibility'][group+':'+str(quality.get('eligible'))] += 1
                source = result['projection']['components'].get('variance_source')
                if source:
                    counters['variance_sources'][group+':'+source] += 1
    if len(ids) != report['fixtures'] or len(ids) != report['exact_serialized_replays']:
        raise ValueError('Completion counters differ from actual files')
    return {'verified_fixtures': len(ids), 'verified_snapshot_hashes': len(snapshots_seen),
            'all_history_memberships_before_cutoff': True, 'all_seven_canonical_results_present': True,
            'no_recommendations_or_prices': True, **{k:dict(v) for k,v in sorted(counters.items())}}
