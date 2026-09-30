"""Local card evidence reconciliation and a fixed, dated research comparison.

No live services, API clients, database writes, weight selection or promotion.
Provisional arithmetic is deliberately unusable by the numerical runner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
import hashlib
import json
import math
import unicodedata

import numpy as np
from scipy.stats import poisson

from Scripts.data_platform.features import phase4_cards as cards
from Scripts.data_platform.features.benchmarks.confirmation_data import _end, _members, _white
from Scripts.data_platform.participation_cards import parse_participation_cards
from Scripts.data_platform.settlement import provider_id

VERSION = 'phase4-local-card-reconstruction.v1'
SEED = 20260930


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def name(value):
    # No fuzzy matching, accent stripping or initial/surname merging.
    return ' '.join(unicodedata.normalize('NFC', str(value or '')).casefold().split())


def referee_key(value):
    return name(str(value or '').split(',')[0])


def array_spans(text):
    pos = _white(text, 0)
    if text[pos] != '[':
        raise ValueError('Expected JSON array')
    pos = _white(text, pos + 1)
    while text[pos] != ']':
        stop = _end(text, pos)
        yield pos, stop
        pos = _white(text, stop)
        if text[pos] == ']':
            break
        if text[pos] != ',':
            raise ValueError('Invalid array separator')
        pos = _white(text, pos + 1)
        if text[pos] == ']':
            raise ValueError('Trailing comma')
    if _white(text, pos + 1) != len(text):
        raise ValueError('Trailing JSON data')


def fields(text, start, allowed):
    return {k: json.loads(text[a:b]) for k, a, b in _members(text, start) if k in allowed}


def identity_map(fixtures, aliases=(), team_aliases=None):
    result = defaultdict(dict)
    team_names = defaultdict(lambda: defaultdict(set))
    for f in [*fixtures, *aliases]:
        for side in ('home', 'away'):
            team_names[f['fixture_id']][name(f[side + '_name'])].add(f[side + '_team_id'])
    for f in [*fixtures, *aliases]:
        for raw, canonical in (team_aliases or {}).get(f['competition'], {}).items():
            matched = team_names[f['fixture_id']].get(name(canonical))
            if matched:
                team_names[f['fixture_id']][name(raw)].update(matched)
    for f in [*fixtures, *aliases]:
        key = (f['competition'], f['season'], name(f["home_name"] + ' vs ' + f["away_name"]))
        result[key][f['fixture_id']] = {**f, 'team_name_ids': {
            k: next(iter(v)) for k, v in team_names[f['fixture_id']].items() if len(v) == 1}}
    return {k: list(v.values()) for k, v in result.items()}


def legacy_rows(text, *, competition, season, identities, source):
    """Decode outcome fields only AFTER a unique permitted fixture is resolved.

    Files are season-spanning; no whole-file/row json.loads is used. All excluded
    rows stay opaque apart from identity metadata, including unknown identities.
    """
    admitted, ledger, counts = [], [], Counter()
    for number, (start, _) in enumerate(array_spans(text), 1):
        meta = fields(text, start, {'fixture', 'team', 'player_id'})
        matches = identities.get((competition, season, name(meta.get('fixture'))), [])
        reason, f = None, None
        if not matches:
            reason = 'unresolved_fixture_identity'
        elif len(matches) != 1:
            reason = 'ambiguous_fixture_identity'
        else:
            f = matches[0]
            if not cards.permitted(f['kickoff']):
                reason = 'reserved_period_not_decoded'
            elif f['status'] not in ('FT', 'AET', 'PEN'):
                reason = 'not_completed'
        teams = {} if f is None else f['team_name_ids']
        if reason is None and name(meta.get('team')) not in teams:
            reason = 'unresolved_team_identity'
        if reason is None and (type(meta.get('player_id')) is not int or meta['player_id'] <= 0):
            reason = 'invalid_player_identity'
        counts['rows'] += 1
        counts[reason or 'admitted_provisional_rows'] += 1
        if reason:
            ledger.append({'source': source, 'row': number, **meta,
                           'fixture_id': f['fixture_id'] if f else None, 'reason': reason})
            continue
        values = fields(text, start, {'minutes', 'yellow_cards', 'red_cards'})
        admitted.append({'fixture_id': f['fixture_id'], 'team_id': teams[name(meta['team'])],
                         'player_id': meta['player_id'], **{k: values.get(k) for k in ('minutes', 'yellow_cards', 'red_cards')},
                         'source': source, 'row': number,
                         'identity_method': 'unique_competition_season_ordered_exact_names'})
    return admitted, ledger, dict(counts)


def evidence(fixture, rows):
    return parse_participation_cards(
        {'status': fixture['status'], **{k: provider_id(fixture[k]) for k in ('home_team_id', 'away_team_id')}},
        cards.normalized_payload(rows))


def reconcile(canonical_results, canonical_players, legacy_players):
    """Do not merge partial rosters or allow a provisional source to certify itself."""
    sources = defaultdict(lambda: defaultdict(list))
    for r in canonical_players:
        sources[r['fixture_id']]['platform_normalized'].append(r)
    for r in legacy_players:
        sources[r['fixture_id']][r['source']].append(r)
    output = []
    for f in canonical_results:
        alternatives, canonical = [], sources[f['fixture_id']].get('platform_normalized', [])
        for source, rows in sorted(sources[f['fixture_id']].items()):
            parsed = evidence(f, rows)
            reasons = [] if source == 'platform_normalized' else [
                'legacy_original_response_unavailable', 'legacy_participant_completeness_unverified',
                'legacy_identity_reconstructed_from_names']
            alternatives.append({'source': source, 'rows': len(rows), 'pending_reason': parsed['pending_reason'],
                                 'provisional_total': sum(parsed['totals'].values()) if not parsed['pending_reason'] else None,
                                 'provisional_team_totals': parsed['totals'] or None,
                                 'source_restrictions': reasons,
                                 'rows_sha256': digest(rows)})
        complete = [a for a in alternatives if a['provisional_total'] is not None]
        conflict = len({json.dumps(a['provisional_team_totals'], sort_keys=True) for a in complete}) > 1
        # Equal totals alone cannot hide contradictory players/minutes.
        seen = {}
        for rows in sources[f['fixture_id']].values():
            for r in rows:
                key = r['team_id'], r['player_id']
                value = r['minutes'], r['yellow_cards'], r['red_cards']
                if key in seen and seen[key] != value:
                    conflict = True
                seen[key] = value
        exclusions = set(f['exclusions'])
        if alternatives and not canonical and not f['eligible']:
            exclusions.discard('missing_player_statistics')
            exclusions.update(x for a in alternatives for x in a['source_restrictions'])
            if not complete:
                exclusions.update(a['pending_reason'] for a in alternatives if a['pending_reason'])
        if conflict:
            exclusions.add('local_player_evidence_conflict')
        # Canonical missing rows do not negate a legacy diagnostic, but they never
        # establish a qualified target. Preserve both sets of pending reasons.
        candidate = complete[0] if complete and not conflict else None
        output.append({**f, 'eligible': f['eligible'] and not conflict,
                       'target': f['target'] if f['eligible'] and not conflict else None,
                       'team_targets': f['team_targets'] if f['eligible'] and not conflict else None,
                       'exclusions': sorted(exclusions), 'canonical_exclusions': f['exclusions'],
                       'evidence_sources': alternatives,
                       'local_provisional_total': candidate['provisional_total'] if candidate else None,
                       'local_provisional_team_totals': candidate['provisional_team_totals'] if candidate else None,
                       'provisional_source': candidate['source'] if candidate else None,
                       'local_evidence_conflict': conflict,
                       'canonical_player_rows': len(canonical)})
    return output


def coverage(rows):
    def summary(items):
        known = [r for r in items if r['local_provisional_total'] is not None]
        return {'fixtures': len(items), 'with_player_evidence': sum(bool(r['evidence_sources']) for r in items),
                'provisional_totals': len(known), 'provisional_zero': sum(r['local_provisional_total'] == 0 for r in known),
                'provisional_positive': sum(r['local_provisional_total'] > 0 for r in known),
                'qualified': sum(r['eligible'] for r in items),
                'conflicts': sum(r['local_evidence_conflict'] for r in items),
                'referee_assignments': sum(bool(referee_key(r.get('referee'))) for r in items),
                'source_exclusions': dict(sorted(Counter(x for r in items for x in r['exclusions']).items())),
                'provisional_source_restrictions': dict(sorted(Counter(x for r in items for a in r['evidence_sources']
                                                                        for x in a['source_restrictions']).items())),
                'all_team_red_categories': dict(Counter(r['team_red_category'] for r in items)),
                'provisional_team_red_categories': dict(Counter(r['team_red_category'] for r in known))}
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['competition']}:{r['season']}"].append(r)
    return {'overall': summary(rows), 'league_season': {k: summary(v) for k, v in sorted(groups.items())}}


def qualified_rows(rows):
    selected, seen = [], set()
    for r in rows:
        if r['fixture_id'] in seen:
            raise ValueError('Fixture versions must not be repeated')
        seen.add(r['fixture_id'])
        if not cards.permitted(r['kickoff']):
            raise ValueError('Reserved card outcome access refused')
        if not r['eligible']:
            continue
        totals = r.get('team_targets')
        if (r.get('exclusions') or r.get('contract') != cards.CONTRACT or not r.get('raw_reference')
                or r.get('settlement_policy') != 'spix-participation-settlement.v1'
                or r['status'] != 'FT' or not isinstance(totals, dict) or set(totals) != {'home', 'away'}
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or not float(x).is_integer()
                       for x in totals.values())
                or sum(totals.values()) != r.get('target')):
            raise ValueError('Inconsistent qualified target contract')
        selected.append(r)
    return sorted(selected, key=lambda r: (cards.utc(r['kickoff']), r['fixture_id']))


def support(rows, fitting=False):
    weeks = {cards.utc(r['kickoff']).strftime('%G-W%V') for r in rows}
    return {'fixtures': len(rows), 'weeks': len(weeks),
            'sufficient': len(rows) >= (500 if fitting else 200) and len(weeks) >= (26 if fitting else 20)}


def dated_inputs(fixture, history):
    cutoff = cards.utc(fixture['kickoff']).replace(hour=0, minute=0, second=0, microsecond=0)
    # Validate evidence at this public boundary too; synthetic tests supply the
    # same explicit contract as an actual source-certified target.
    past = [r for r in qualified_rows(history) if r['competition'] == fixture['competition']
            and r['season'] in (fixture['season'] - 1, fixture['season'])
            and cards.utc(r['kickoff']) + timedelta(hours=3) < cutoff
            and r['fixture_id'] != fixture['fixture_id']]
    teams = {}
    for side in ('home', 'away'):
        tid = fixture[side + '_team_id']
        team = []
        for r in past:
            old_side = 'home' if tid == r['home_team_id'] else 'away' if tid == r['away_team_id'] else None
            if old_side:
                team.append({'fixture_id': r['fixture_id'], 'kickoff': r['kickoff'], 'season': r['season'],
                             'venue': old_side, 'own': r['team_targets'][old_side],
                             'induced': r['team_targets']['away' if old_side == 'home' else 'home']})
        teams[side] = team
    key = referee_key(fixture.get('referee'))
    referee = [r for r in past if key and referee_key(r.get('referee')) == key]
    return {'cutoff': cutoff.isoformat(), 'teams': teams,
            'league': [{'fixture_id': r['fixture_id'], 'total': r['target']} for r in past],
            'referee': [{'fixture_id': r['fixture_id'], 'total': r['target']} for r in referee],
            'referee_key': key, 'source_fixture_ids': [r['fixture_id'] for r in past],
            'availability': 'assumed_final'}


def fixed_prediction(fixture, inputs, weights):
    """Aligned reference, not the full frozen public model. No fitted parameters."""
    profiles = {}
    for side in ('home', 'away'):
        rows = inputs['teams'][side]
        current = [r for r in rows if r['season'] == fixture['season']]
        if len(current) < 8:
            return None, 'insufficient_current_team_history'
        venue = [r for r in current if r['venue'] == side]
        overall = float(np.mean([r['own'] for r in current]))
        own = (weights['venue'] * float(np.mean([r['own'] for r in venue])) + (1 - weights['venue']) * overall) if venue else overall
        recent = list(reversed(rows))[:6]
        decay = np.power(weights['recency_alpha'], np.arange(len(recent)))
        profiles[side] = {'own': own, 'induced': float(np.mean([r['induced'] for r in current])),
                          'recent_own': float(np.average([r['own'] for r in recent], weights=decay)),
                          'recent_induced': float(np.average([r['induced'] for r in recent], weights=decay)),
                          'current_n': len(current), 'venue_n': len(venue), 'recent_n': len(recent)}
    h, a = profiles['home'], profiles['away']
    season = weights['own'] * (h['own'] + a['own']) + weights['opponent'] * (h['induced'] + a['induced'])
    recent = .6 * (h['recent_own'] + a['recent_own']) + .4 * (h['recent_induced'] + a['recent_induced'])
    refrows, league = inputs['referee'], inputs['league']
    refmean = float(np.mean([r['total'] for r in refrows])) if refrows else None
    league_mean = float(np.mean([r['total'] for r in league])) if league else None
    confidence, multiplier, reason = 0., 1., 'unknown_referee' if not inputs['referee_key'] else 'sparse_referee'
    if len(refrows) >= weights['referee_minimum'] and league_mean and refmean is not None:
        confidence = min(1., max(0., (len(refrows) - weights['referee_minimum']) /
                                    (weights['referee_full_confidence'] - weights['referee_minimum'])))
        multiplier = 1 + weights['referee_modifier'] * confidence * (refmean / league_mean - 1)
        reason = 'qualified_dated_profile'
    def combine(s):
        divergence = abs(recent - s) / max(s, 1.)
        shift = min(.30, max(0., divergence - .20))
        return (weights['season'] - shift) * s + (weights['recent'] + shift) * recent
    base = combine(season)
    with_ref = combine(season * multiplier)
    anchor = weights['referee_anchor'] * confidence if len(refrows) >= weights['referee_anchor_minimum'] else 0.
    if confidence and refmean is not None:
        with_ref = (1 - anchor) * with_ref + anchor * refmean
    if not all(math.isfinite(x) and x > 0 for x in (base, with_ref)):
        return None, 'nonpositive_or_invalid_mean'
    return {'team_only_mean': base, 'team_referee_mean': with_ref, 'profiles': profiles,
            'referee': {'name': inputs['referee_key'], 'matches': len(refrows), 'reason': reason,
                        'mean': refmean, 'league_mean': league_mean, 'confidence': confidence,
                        'multiplier': multiplier, 'anchor': anchor},
            'missing_context': ['foul_and_aggression_rates', 'lineup', 'knockout', 'league_regime', 'original_assignment_time'],
            'season_stage': '8-15' if min(h['current_n'], a['current_n']) < 16 else '16+'}, None


def metrics(rows, key):
    actual = np.array([r['target'] for r in rows], dtype=float)
    means = np.array([r[key] for r in rows])
    dist = poisson(means)
    probability = np.clip(dist.sf(4), 1e-15, 1 - 1e-15)
    binary = (actual > 4.5).astype(float)
    reliability = []
    for i in range(10):
        mask = (probability >= i / 10) & (probability < (i + 1) / 10)
        reliability.append({'lower': i / 10, 'n': int(mask.sum()),
                            'probability': float(probability[mask].mean()) if mask.any() else None,
                            'frequency': float(binary[mask].mean()) if mask.any() else None})
    return {**support(rows), 'nll': float(-dist.logpmf(actual).mean()),
            'mae': float(np.abs(means - actual).mean()), 'bias': float((means - actual).mean()),
            'rmse': float(np.sqrt(np.mean((means - actual) ** 2))),
            'brier_over45': float(np.mean((probability - binary) ** 2)),
            'log_loss_over45': float(-np.mean(binary * np.log(probability) + (1 - binary) * np.log1p(-probability))),
            'interval80_coverage': float(np.mean((actual >= dist.ppf(.1)) & (actual <= dist.ppf(.9)))),
            'interval80_width': float(np.mean(dist.ppf(.9) - dist.ppf(.1))), 'reliability': reliability}


def comparison(rows):
    base = metrics(rows, 'team_only_mean')
    candidate = metrics(rows, 'team_referee_mean')
    weeks = sorted({cards.utc(r['kickoff']).strftime('%G-W%V') for r in rows})
    deltas = defaultdict(list)
    for r in rows:
        deltas[cards.utc(r['kickoff']).strftime('%G-W%V')].append(
            float(poisson.logpmf(r['target'], r['team_only_mean']) - poisson.logpmf(r['target'], r['team_referee_mean'])))
    sums = np.array([sum(deltas[w]) for w in weeks])
    counts = np.array([len(deltas[w]) for w in weeks])
    draws = np.random.default_rng(SEED).integers(0, len(weeks), (1000, len(weeks)))
    interval = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975]).tolist()
    return {'team_only': base, 'team_plus_referee': candidate, 'nll_delta': candidate['nll'] - base['nll'],
            'paired_week_bootstrap95': interval}


def backtest(rows, weights):
    qualified = qualified_rows(rows)
    fitting = [r for r in qualified if cards.utc(r['kickoff']).year == 2022
               and (cards.utc(r['kickoff']) + timedelta(hours=3)).year == 2022]
    evaluation = [r for r in qualified if cards.utc(r['kickoff']).year == 2023]
    report = {'version': VERSION, 'fitting_support': support(fitting, fitting=True),
              'evaluation_target_support': support(evaluation), 'weights': weights,
              'weight_optimisation': False, 'production_qualification': False,
              'model': 'fixed_participation_aligned_team_and_referee_poisson_reference',
              'exact_frozen_control_replay': False, 'availability': 'assumed_final'}
    reasons = []
    if not report['fitting_support']['sufficient']:
        reasons.append('insufficient_qualified_2022_fitting_history')
    if not report['evaluation_target_support']['sufficient']:
        reasons.append('insufficient_qualified_2023_evaluation_targets')
    if reasons:
        return report | {'status': 'blocked_evidence_or_support', 'reasons': reasons,
                         'scored_fixtures': 0, 'metrics': None}, [], []
    predictions, snapshots, exclusions = [], [], Counter()
    for f in evaluation:
        inputs = dated_inputs(f, qualified)
        prediction, reason = fixed_prediction(f, inputs, weights)
        if reason:
            exclusions[reason] += 1
            continue
        snapshot_id = digest(inputs)
        snapshots.append({'fixture_id': f['fixture_id'], 'snapshot_id': snapshot_id, **inputs})
        predictions.append({k: f[k] for k in ('fixture_id', 'competition', 'season', 'kickoff', 'target')} |
                           prediction | {'snapshot_id': snapshot_id})
    report['forecast_exclusions'] = dict(exclusions)
    report['evaluation_forecast_support'] = support(predictions)
    if not report['evaluation_forecast_support']['sufficient']:
        return report | {'status': 'blocked_evidence_or_support', 'reasons': ['insufficient_dated_forecast_support'],
                         'scored_fixtures': 0, 'metrics': None}, predictions, snapshots
    slices = defaultdict(list)
    for r in predictions:
        for k, v in [('league', r['competition']), ('season_stage', r['season_stage']),
                     ('referee', r['referee']['reason'])]:
            slices[f'{k}:{v}'].append(r)
    report.update(status='development_diagnostic_complete', scored_fixtures=len(predictions),
                  metrics=comparison(predictions),
                  slices={k: {'supported': support(v)['sufficient'], **comparison(v)} for k, v in sorted(slices.items())})
    return report, predictions, snapshots
