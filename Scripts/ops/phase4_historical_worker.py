"""Fresh-process frozen numerical engine, with dated external inputs only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
import json
import hashlib
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def install_guard(source, output, adapter_path=None):
    violations = []
    def audit(event, args):
        reason = None
        if event in ('socket.connect', 'socket.getaddrinfo', 'sqlite3.connect', 'subprocess.Popen', 'os.system'):
            reason = event
        if event == 'open' and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not path.is_relative_to(output):
                reason = 'write_outside_experiment:' + str(path)
            elif path.name == '.env' or path.suffix in ('.db', '.sqlite', '.sqlite3'):
                reason = 'protected_input:' + str(path)
            elif any(part in path.parts for part in ('Index', 'Output', 'Research')):
                if not path.is_relative_to(source) and not path.is_relative_to(output) and path != adapter_path:
                    reason = 'non_frozen_input:' + str(path)
                elif 'Index' in path.parts and not path.is_relative_to(source/'Index/ml_models'):
                    reason = 'non_model_index_input:' + str(path)
        if reason:
            violations.append(reason)
            raise RuntimeError(reason)
    sys.addaudithook(audit)
    return violations


def run(source, prepared, output, limit=None, qualification=False):
    source, prepared, output = (p.resolve() for p in (source, prepared, output))
    if qualification:
        if limit is not None:
            raise ValueError('Qualification cannot silently truncate its cohort')
        for name in ('METHOD_LOCK.json', 'INPUT_LOCK.json'):
            for relative, expected in json.loads((prepared/name).read_text()).items():
                path = prepared/relative
                if not path.resolve().is_relative_to(prepared) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Qualification input/method lock mismatch')
        if not (prepared/'QUALIFICATION_OPENED.json').exists():
            raise ValueError('Qualification opening must be recorded first')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    history = [json.loads(line) for line in (prepared/'history.jsonl').open()]
    requests = [json.loads(line) for line in (prepared/'requests.jsonl').open()]
    if qualification and any(not '2025-01-01' <= r['as_of'][:10] < '2025-07-01' for r in requests):
        raise ValueError('Qualification requests cross the frozen interval')
    if limit is not None:
        requests = requests[:limit]
    sys.dont_write_bytecode = True
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
    dotenv.dotenv_values = lambda *a, **k: {}
    os.environ['PREDICTION_RELEASE_MODE'] = 'shadow'
    sys.path[:0] = [str(source), str(source/'Scripts'), str(source/'Scripts/rag_ingest')]
    # Load the new adapter by path while all engine imports resolve to the archive.
    adapter_path = Path(__file__).parents[1]/'data_platform/features/phase4_history.py'
    violations = install_guard(source, output, adapter_path.resolve())
    adapter = ModuleType('historical_adapter')
    # Compile explicit source bytes: importlib otherwise probes an unapproved
    # bytecode cache even under -B. No cache is an input to this reconstruction.
    exec(compile(adapter_path.read_bytes(), str(adapter_path), 'exec'), adapter.__dict__)
    from core import projections as p, team_resolution as tr, market_service as ms
    from core import line_selection as ls
    from core.weights import SCORING_WEIGHTS
    from core.system_identity import prediction_system_manifest
    import prob_models as probability
    for module in (p, tr, ms, ls, probability):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError('Engine import escaped frozen source')
    weights = {m: p._ml_blend_weight(m) for m in ('goals', 'corners', 'sot', 'cards')}
    if any(weights.values()):
        raise ValueError('Nonzero control ML contribution')
    index = adapter.HistoricalInputs(history, end=datetime(2025, 7, 1, tzinfo=timezone.utc)) if qualification else adapter.HistoricalInputs(history)
    encode, digest = adapter.encode, adapter.digest
    counts, slices = Counter(), defaultdict(Counter)
    parity_count = 0
    context = {
        'league': {'status': 'unavailable_neutral', 'reason': 'no certified dated standings/regime snapshot'},
        'knockout': {'status': 'unavailable_neutral', 'reason': 'no certified round/first-leg context'},
        'referee': {'status': 'unavailable_neutral', 'reason': 'no dated regulation-qualified card profile'},
        'lineup': {'status': 'unavailable', 'reason': 'no dated player/confirmed-XI snapshot'},
        'engineered_profile_fields': {'status': 'unavailable', 'fields': ['control_index', 'aggression_index_norm', 'form_index_team']},
        'prices': {'status': 'unavailable', 'reason': 'no authentic timestamped equivalent quote slate'},
    }
    league_ctx = SimpleNamespace(adjustments={m: 1. for m in ('goals', 'corners', 'sot', 'cards')})
    knockout_ctx = SimpleNamespace(is_knockout=False, goals_modifier=1., corners_modifier=1., cards_modifier=1., sot_modifier=1.)
    referee = SimpleNamespace(source='unavailable', multiplier=1., confidence=0., sample_size=0,
        avg_cards_per_match=0., cards_per_foul=0., avg_fouls_per_match=0., strictness_ratio=1., referee_name=None)

    def numerical(request, snapshot_id):
        f = request['fixture']
        home, away, league = str(f['home_team_id']), str(f['away_team_id']), f['competition']
        kwargs = dict(fixture_date=request['as_of'], league_ctx=league_ctx, knockout_ctx=knockout_ctx)
        event = {'id': str(f['fixture_id']), 'home_team': home, 'away_team': away,
                 'commence_time': f['kickoff'], 'bookmakers': []}
        results = ms.evaluate_event(event, league, **kwargs, ref_mod=referee,
            generated_at=request['as_of'], input_snapshot_id=snapshot_id,
            model_version='phase4-repaired-statistical-control.v1')
        score, h, a = p.projected_correct_score_probs(home, away, league, **kwargs)
        return {'fixture_id': f['fixture_id'], 'snapshot_id': snapshot_id,
                'goal_means': [h, a], 'score_distribution': [[i, j, v] for (i, j), v in sorted((score or {}).items())],
                'results': [r.to_dict() for r in results]}

    with (output/'snapshots.jsonl').open('x') as snapshots, (output/'control.jsonl').open('x') as forecasts, \
            (output/'eligibility.jsonl').open('x') as eligibility:
        for number, request in enumerate(requests, 1):
            f, cutoff = request['fixture'], request['as_of']
            league = f['competition']
            teams = [str(f['home_team_id']), str(f['away_team_id'])]
            tr.clear_profile_caches()
            with ExitStack() as stack:
                patches = [
                    (tr, '_profile_data_revision', lambda: ('historical', f['fixture_id'], cutoff)),
                    (tr, 'get_team_profile_docs', lambda *a, **k: []),
                    (tr, '_get_all_team_fixture_rows', lambda team, lg: index.rows(team, lg, cutoff)),
                    (tr, '_prefetch_opponent_fixture_meta', lambda *a, **k: None),
                    (tr, '_get_team_fixture_meta', lambda team, lg, fixture, *a, **k: index.metas.get((str(team), str(fixture)), {}).get('meta')),
                    (tr, 'resolve_domestic_league', lambda team: index.domestic(team, cutoff)),
                    (p, 'get_card_risk_profiles', lambda *a, **k: (None, None)),
                ]
                for module, name, value in patches:
                    stack.enter_context(patch.object(module, name, new=value))
                profiles, quality, recent, variance, evidence, components = {}, {}, {}, {}, {}, {}
                for team in teams:
                    profiles[team], quality[team] = tr.get_prediction_profile_context(team, league, target_date=cutoff)
                    recent[team] = tr._recent_stats(team, league, target_date=cutoff)
                    variance[team] = tr.get_team_recent_variance(team, league, target_date=cutoff)
                    leagues = {league}
                    domestic = index.domestic(team, cutoff) if league in adapter.EUROPE else None
                    if domestic:
                        leagues.add(domestic)
                    evidence[team], components[team] = {}, {}
                    rank = tr._season_rank_for_target_date(cutoff)
                    for lg in sorted(leagues):
                        ev = index.evidence(team, lg, cutoff, rank)
                        evidence[team][lg] = ev
                        components[team][lg] = {}
                        for period in ('current', 'prior'):
                            rows = [index.metas[(team, str(fid))] for fid in ev[period]['fixture_ids']]
                            season = rank if period == 'current' else ev['prior_rank']
                            components[team][lg][period] = tr._profile_from_fixture_rows(team, lg, str(season), rows)
                snapshot = {**request, 'version': adapter.VERSION, 'profiles': profiles, 'profile_quality': quality,
                            'recent': recent, 'recent_variance': variance, 'profile_components': components,
                            'history_evidence': evidence, 'context': context,
                            'league_context': vars(league_ctx), 'knockout_context': vars(knockout_ctx),
                            'referee': vars(referee), 'lineup': None,
                            'domestic_resolution': 'latest_completed_domestic_fixture_by_exact_provider_team_id',
                            'season_resolution': 'unchanged_control_July_calendar_rule'}
                # Eligibility contains target-presence information; keep it out of prediction identity.
                snapshot.pop('source_eligibility')
                snapshot_id = digest(snapshot)
                direct = numerical(request, snapshot_id)
                # Re-run the numerical functions from serialized saved aggregates, not lookup caches.
                saved = json.loads(encode(snapshot))
                with ExitStack() as replay:
                    def profile(team, *a, **k): return deepcopy(saved['profiles'][team])
                    def recent_stats(team, *a, **k): return deepcopy(saved['recent'][team])
                    def recent_variance(team, *a, **k): return deepcopy(saved['recent_variance'][team])
                    def profile_quality(team, *a, **k): return profile(team), deepcopy(saved['profile_quality'][team])
                    for module, name, value in [(p, '_profile_meta', profile), (tr, '_profile_meta', profile),
                            (p, '_recent_stats', recent_stats), (tr, 'get_team_recent_variance', recent_variance),
                            (ms, 'get_prediction_profile_context', profile_quality)]:
                        replay.enter_context(patch.object(module, name, new=value))
                    repeated = numerical(request, snapshot_id)
                if encode(direct) != encode(repeated):
                    raise ValueError('Saved-input numerical replay differs: '+str(f['fixture_id']))
                parity_count += 1
                distributions, derived = distribution_output(direct, probability, ls)
                direct.update(distributions=distributions, diagnostics=derived, context=context,
                              availability='assumed_final', exact_historical_publication=False)
                allowed = {}
                for market in ('goals', 'corners', 'sot', 'cards'):
                    reasons = list(request['source_eligibility'][market]['reasons'])
                    if market not in distributions:
                        reasons.append('control_distribution_unavailable')
                    allowed[market] = {'eligible': not reasons, 'reasons': sorted(set(reasons)),
                                       'source_contract': 'frozen_phase3_eligibility_retained_conservatively'}
                    key = market+':'+('eligible' if not reasons else 'excluded')
                    counts[key] += 1
                    slices[league][key] += 1
                snapshots.write(encode({'snapshot_id': snapshot_id, **snapshot})+'\n')
                forecasts.write(encode(direct)+'\n')
                eligibility.write(encode({'fixture_id': f['fixture_id'], 'snapshot_id': snapshot_id, 'markets': allowed})+'\n')
            if number % 250 == 0:
                print(f'{number}/{len(requests)} historical controls replayed', flush=True)
    if violations:
        raise RuntimeError('Forbidden IO was attempted: '+repr(violations))
    report = {'fixtures': len(requests), 'exact_serialized_replays': parity_count, 'io_violations': violations,
              'coverage': dict(counts), 'league_coverage': {k: dict(v) for k, v in sorted(slices.items())},
              'ml_weights': weights, 'scoring_weights': SCORING_WEIGHTS, 'system_identity': prediction_system_manifest(),
              'publication_enabled': False, 'parameters_fitted': False, 'context': context,
              'parity_boundary': 'unchanged frozen lookup/profile arithmetic versus same frozen numerical functions with serialized aggregates'}
    (output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False,
        default=lambda value: sorted(value) if isinstance(value, set) else vars(value))+'\n')


def distribution_output(result, probability, line_selection):
    """Unpriced diagnostics; preserve real control variance and Asian arithmetic."""
    score = {(h, a): p for h, a, p in result['score_distribution']}
    distributions, diagnostics = {}, {}
    if score:
        check_probabilities(score.values())
        totals = defaultdict(float)
        for (h, a), value in score.items():
            totals[h+a] += value
        distributions['goals'] = {'pmf': [totals[k] for k in range(max(totals)+1)],
                                  'source': 'unchanged_shared_Dixon_Coles_score_distribution', 'omitted_mass_bound': 1e-10}
        diagnostics['btts'] = {'yes': sum(p for (h, a), p in score.items() if h > 0 and a > 0)}
        diagnostics['btts']['no'] = 1-diagnostics['btts']['yes']
        diagnostics['winner'] = {name: sum(p for (h, a), p in score.items() if predicate(h, a))
            for name, predicate in [('home', lambda h,a:h>a), ('draw', lambda h,a:h==a), ('away', lambda h,a:h<a)]}
        diagnostics['handicaps'] = {str(line): line_selection._score_matrix_spread_profile(
            '', '', '', True, line, score_probs=score) for line in (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.)}
    for r in result['results']:
        market = r['market']['group']
        if market not in ('corners', 'sot') or r['projection']['value'] is None:
            continue
        mean, variance = r['projection']['value'], r['projection']['variance']
        pmf, mass = [], 0.
        for k in range(10001):
            value = probability._count_pmf(k, mean, variance)
            pmf.append(value); mass += value
            if k >= 40 and mass >= 1-1e-12:
                break
        if abs(1-mass) > 1e-10 or min(pmf) < 0:
            raise ValueError('Count distribution tail/positivity failed')
        distributions[market] = {'pmf': pmf, 'residual_mass': max(0., 1-mass),
                                  'mean': mean, 'variance': variance, 'source': 'unchanged_control_count_pmf'}
    for market, centre in [('goals', 2.5), ('corners', 9.5), ('sot', 8.5)]:
        if market not in distributions:
            continue
        pmf = dict(enumerate(distributions[market]['pmf']))
        diagnostics[market] = {}
        for line in (centre-.5, centre-.25, centre, centre+.25, centre+.5):
            outcomes = {}
            for side in ('Over', 'Under'):
                if market == 'goals':
                    outcomes[side] = probability.asian_total_profile_from_counts(pmf, line, side)
                else:
                    d = distributions[market]
                    outcomes[side] = probability.asian_total_settlement_profile(d['mean'], line, side, d['variance'])
                check_probabilities(outcomes[side].values())
            diagnostics[market][str(line)] = outcomes
    diagnostics['price_status'] = 'fixed_diagnostic_lines_only_no_quotes_no_EV_no_ROI'
    return distributions, diagnostics


def check_probabilities(values):
    values = list(values)
    if not values or min(values) < 0 or any(not math.isfinite(v) for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-10):
        raise ValueError('Invalid normalized probability distribution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--qualification', action='store_true')
    args = parser.parse_args()
    run(args.source, args.prepared, args.output, args.limit, args.qualification)
