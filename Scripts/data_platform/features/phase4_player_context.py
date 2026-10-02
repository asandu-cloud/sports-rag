"""Dated player composition research. No live lookups, odds or public wiring."""
from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import minimize_scalar
from . import phase4_candidates as cm

VERSION = 'phase4-dated-player-context.v2'
MARKETS = ('goals', 'sot')
STAGES = ('expected_players', 'conditional_actual_xi')
BOUND = math.log(1.25)


def allocate(values):
    """Normalize a nonnegative minute allocation to 990 with a 90-minute cap."""
    if len(values) < 11 or any(not cm.finite_nonnegative(v) for v in values.values()):
        raise ValueError('insufficient_or_invalid_minute_allocation')
    positive = {k: float(v) for k, v in values.items() if v > 0}
    if len(positive) < 11:
        raise ValueError('fewer_than_eleven_positive_allocations')
    result = {k: 0. for k in values}; remaining = 990.
    while positive:
        total = sum(positive.values())
        capped = [k for k, v in positive.items() if remaining*v/total >= 90.]
        if not capped:
            result.update({k: remaining*v/total for k, v in positive.items()}); break
        for k in capped:
            result[k] = 90.; remaining -= 90.; del positive[k]
    if not math.isclose(sum(result.values()), 990., abs_tol=1e-7):
        raise ValueError('minute_budget_not_conserved')
    return result


def normalize(fixture, player_payload, lineup_payload):
    """Original response qualification; current players and XI are separate."""
    f = {k: fixture[k] for k in ('fixture_id', 'competition', 'season', 'kickoff', 'home_team_id', 'away_team_id')}
    if cm.utc(f['kickoff']) + timedelta(hours=3) >= cm.END:
        raise ValueError('reserved_player_response')
    result = {'fixture': f, 'players': None, 'lineups': None, 'reasons': [], 'reconciliation': {}}
    teams = {f[s+'_team_id']: s for s in ('home', 'away')}
    if lineup_payload is not None:
        try:
            xi = {}; seen = set()
            for block in lineup_payload['response']:
                side = teams[block['team']['id']]
                if side in xi: raise ValueError('duplicate_lineup_team')
                xi[side] = {'starters': [], 'bench': [], 'positions': {}}
                for field, role in (('startXI', 'starters'), ('substitutes', 'bench')):
                    for entry in block[field]:
                        p = entry['player']; pid = p['id']
                        if type(pid) is not int or pid <= 0 or pid in seen: raise ValueError('invalid_lineup_identity')
                        seen.add(pid); xi[side][role].append(pid); xi[side]['positions'][str(pid)] = p.get('pos') or 'unknown'
                if len(xi[side]['starters']) != 11: raise ValueError('incomplete_xi')
            if set(xi) != {'home', 'away'}: raise ValueError('incomplete_lineup_teams')
            result['lineups'] = xi
        except (ValueError, KeyError, TypeError) as exc:
            result['reasons'].append('lineup:'+str(exc))
    else: result['reasons'].append('lineup_response_unavailable')
    if player_payload is not None and result['lineups']:
        try:
            players = {}; seen = set()
            for block in player_payload['response']:
                side = teams[block['team']['id']]
                if side in players: raise ValueError('duplicate_player_team')
                players[side] = []
                roster = set(result['lineups'][side]['starters'] + result['lineups'][side]['bench'])
                for entry in block['players']:
                    pid = entry['player']['id']; stats = entry['statistics']
                    if type(pid) is not int or pid <= 0 or pid in seen or len(stats) != 1: raise ValueError('invalid_player_identity')
                    seen.add(pid); s = stats[0]; minutes = (s.get('games') or {}).get('minutes')
                    if minutes is None or minutes == 0: continue
                    if type(minutes) is not int or not 0 < minutes <= 90 or pid not in roster: raise ValueError('invalid_participation')
                    goals = (s.get('goals') or {}).get('total'); sot = (s.get('shots') or {}).get('on')
                    if any(v is not None and (type(v) is not int or v < 0) for v in (goals, sot)): raise ValueError('invalid_count')
                    players[side].append({'player_id': pid, 'minutes': minutes, 'goals': goals, 'sot': sot,
                        'position': (s.get('games') or {}).get('position') or result['lineups'][side]['positions'].get(str(pid), 'unknown'),
                        'starter': pid in result['lineups'][side]['starters']})
                if not set(result['lineups'][side]['starters']) <= {p['player_id'] for p in players[side]}: raise ValueError('starter_minutes_missing')
                if not 720 <= sum(p['minutes'] for p in players[side]) <= 1020: raise ValueError('inconsistent_team_minutes')
                allocate({p['player_id']: float(p['minutes']) for p in players[side]})
                result['reconciliation'][side] = {}
                for market in MARKETS:
                    total = fixture.get(side, {}).get(market)
                    known = sum(p[market] for p in players[side] if p[market] is not None)
                    qualified = total is not None and total == known
                    result['reconciliation'][side][market] = {'team_total': total, 'sum_reported_players': known,
                        'qualified': qualified, 'zeroes_established_by_exhausted_team_total': sum(p[market] is None for p in players[side]) if qualified else 0,
                        'reason': 'exhaustive_nonnegative_team_total' if qualified else 'unknown_or_unallocated_team_total',
                        'source_fixture_id': f['fixture_id']}
                    for p in players[side]:
                        p['reported_'+market] = p[market]
                        # Once nonnegative known contributions exhaust the certified
                        # team total, all unallocated contributions are exactly zero.
                        # Otherwise exclude the entire team/field observation so
                        # positive-only reporting cannot bias rate denominators.
                        p[market] = (p[market] if p[market] is not None else 0) if qualified else None
            if set(players) != {'home', 'away'}: raise ValueError('incomplete_player_teams')
            result['players'] = players
        except (ValueError, KeyError, TypeError) as exc:
            result['reasons'].append('players:'+str(exc))
    else: result['reasons'].append('qualified_player_response_unavailable')
    result['id'] = cm.digest(result)
    return result


class History:
    def __init__(self, rows):
        self.rows = {}; self.league = defaultdict(list); self.team = defaultdict(list); self.player = defaultdict(list)
        for row in sorted(rows, key=lambda r: (cm.utc(r['fixture']['kickoff']), r['fixture']['fixture_id'])):
            f = row['fixture']; fid = f['fixture_id']
            if fid in self.rows or cm.utc(f['kickoff'])+timedelta(hours=3) >= cm.END: raise ValueError('duplicate_or_reserved_history')
            self.rows[fid] = row
            if not row['players']: continue
            self.league[f['competition']].append(row)
            for side in ('home', 'away'):
                self.team[(f['competition'], f[side+'_team_id'])].append((row, side))
                for p in row['players'][side]: self.player[(f['competition'], p['player_id'])].append((row, side, p))

    @staticmethod
    def before(row, season, cutoff):
        f = row['fixture']; t = cm.utc(f['kickoff']); c = cm.utc(cutoff)
        return f['season'] in (season-1, season) and t.date() < c.date() and t+timedelta(hours=3) < c

    @lru_cache(maxsize=4096)
    def priors(self, league, season, cutoff):
        aggregates = defaultdict(lambda: [0., 0.]); ids = []; starts = defaultdict(list); subs = defaultdict(list)
        for row in self.league[league]:
            if not self.before(row, season, cutoff): continue
            ids.append(row['fixture']['fixture_id'])
            for side in ('home', 'away'):
                for p in row['players'][side]:
                    (starts if p['starter'] else subs)[p['position']].append(p['minutes'])
                    for m in MARKETS:
                        if p[m] is not None:
                            for role in (p['position'], '*'):
                                a = aggregates[(role, m)]; a[0] += p[m]; a[1] += p['minutes']
        rates = {}
        for role, m in aggregates:
            a = aggregates[role, m] if aggregates[role, m][1] >= 450 else aggregates['*', m]
            rates[role+':'+m] = 90*a[0]/a[1] if a[1] else None
        return {'rates': rates, 'fixture_ids': ids,
                'starter_minutes': {r: float(np.mean(v)) for r, v in starts.items()},
                'sub_minutes': {r: float(np.mean(v)) for r, v in subs.items()}}

    def player_rate(self, league, season, cutoff, pid, position, prior):
        history = [(r, side, p) for r, side, p in self.player[league, pid] if self.before(r, season, cutoff)]
        result = {'player_id': pid, 'position': position, 'fixture_ids': [r['fixture']['fixture_id'] for r, _, _ in history]}
        for m in MARKETS:
            known = [p for _, _, p in history if p[m] is not None]
            minutes = sum(p['minutes'] for p in known); rate = prior['rates'].get(position+':'+m, prior['rates'].get('*:'+m))
            result[m+'_per90'] = (90*sum(p[m] for p in known) + 450*rate)/(minutes+450) if rate is not None else None
            result[m+'_known_minutes'] = minutes
        started = [p['minutes'] for _, _, p in history if p['starter']]
        subbed = [p['minutes'] for _, _, p in history if not p['starter']]
        result['starter_minutes'] = (sum(started)+5*prior['starter_minutes'].get(position, 75.))/(len(started)+5)
        result['sub_minutes'] = (sum(subbed)+5*prior['sub_minutes'].get(position, 15.))/(len(subbed)+5)
        result['last_team_id'] = history[-1][0]['fixture'][history[-1][1]+'_team_id'] if history else None
        return result

    def build(self, fixture, cutoff, stage):
        if stage not in STAGES or cm.utc(cutoff) >= cm.END or cm.utc(cutoff) > cm.utc(fixture['kickoff']): raise ValueError('invalid_forecast_stage_or_cutoff')
        result = {'version': VERSION, 'fixture': fixture, 'as_of': cutoff, 'stage': stage,
                  'availability': 'assumed_final', 'announcement_time_verified': False,
                  'public_eligible': False, 'teams': {}, 'status': 'available', 'fallbacks': []}
        league, season = fixture['competition'], fixture['season']
        prior = self.priors(league, season, cutoff)
        result['prior_fixture_ids'] = prior['fixture_ids']
        for side in ('home', 'away'):
            try:
                tid = fixture[side+'_team_id']
                past = [(r, s) for r, s in self.team[league, tid] if self.before(r, season, cutoff) and r['fixture']['season'] == season]
                if len(past) < 3: raise ValueError('fewer_than_three_current_team_responses')
                allocations = [allocate({p['player_id']: float(p['minutes']) for p in r['players'][s]}) for r, s in past]
                reference = defaultdict(float)
                for a in allocations:
                    for pid, value in a.items(): reference[pid] += value/len(allocations)
                positions = {p['player_id']: p['position'] for r, s in past for p in r['players'][s]}
                current_lineup = None
                if stage == 'conditional_actual_xi':
                    current = self.rows.get(fixture['fixture_id'])
                    if not current or not current['lineups']: raise ValueError('historical_actual_xi_unavailable')
                    if any(current['fixture'][k] != fixture[k] for k in ('home_team_id', 'away_team_id', 'competition', 'season', 'kickoff')): raise ValueError('target_lineup_identity_mismatch')
                    current_lineup = current['lineups'][side]
                    positions.update({int(k): v for k, v in current_lineup['positions'].items()})
                rates = {pid: self.player_rate(league, season, cutoff, pid, pos, prior) for pid, pos in positions.items()}
                if stage == 'expected_players':
                    forecast = defaultdict(float); weights = [.85**i for i in range(min(3, len(allocations)))]
                    for a, weight in zip(reversed(allocations[-3:]), weights):
                        for pid, minutes in a.items():
                            if rates[pid]['last_team_id'] == tid: forecast[pid] += weight*minutes/sum(weights)
                    forecast = allocate(dict(forecast))
                else:
                    forecast = {pid: rates[pid]['starter_minutes'] for pid in current_lineup['starters']}
                    remaining = 990-sum(forecast.values()); bench = {pid: rates[pid]['sub_minutes'] for pid in current_lineup['bench']}
                    if not bench and remaining > 1e-7: raise ValueError('bench_replacement_unavailable')
                    if bench: forecast.update({pid: remaining*v/sum(bench.values()) for pid, v in bench.items()})
                    forecast = allocate(forecast)
                players = []
                for pid in sorted(set(reference) | set(forecast)):
                    players.append({**rates[pid], 'reference_minutes': reference.get(pid, 0.), 'forecast_minutes': forecast.get(pid, 0.)})
                features = {}; support = {}
                for m in MARKETS:
                    rates_known = all(p[m+'_per90'] is not None for p in players)
                    ref = sum(p['reference_minutes']*p[m+'_per90']/90 for p in players) if rates_known else 0.
                    new = sum(p['forecast_minutes']*p[m+'_per90']/90 for p in players) if rates_known else 0.
                    known = sum(p['forecast_minutes'] for p in players if p[m+'_known_minutes'] > 0)/990
                    usable = min(ref, new) > 0 and known >= .8
                    features[m] = float(np.clip(math.log(new/ref), -BOUND, BOUND)) if usable else None
                    support[m] = {'reference_production': ref, 'forecast_production': new, 'known_minute_share': known,
                                  'raw_log_ratio': math.log(new/ref) if min(ref,new)>0 else None,
                                  'status': 'available' if usable else 'insufficient_player_rate_support'}
                result['teams'][side] = {'team_id': tid, 'players': players, 'features': features, 'support': support,
                    'team_fixture_ids': [r['fixture']['fixture_id'] for r, _ in past],
                    'lineup_source': 'earlier_completed_matches' if current_lineup is None else 'postmatch_download_of_actual_xi'}
            except ValueError as exc:
                result['status'] = 'control_fallback'; result['fallbacks'].append(side+':'+str(exc))
        result['id'] = cm.digest(result)
        return result


def available(feature, market):
    return feature['status'] == 'available' and all(feature['teams'][s]['features'][market] is not None for s in ('home','away'))


def vector(feature, market):
    if not available(feature, market): return [0., 0.]
    return [feature['teams'][s]['features'][market] for s in ('home', 'away')]


def fit(records, market, stage, cutoff='2023-01-01T00:00:00Z'):
    if market not in MARKETS or stage not in STAGES: raise ValueError('invalid_fit_contract')
    if any(cm.utc(r['kickoff'])+timedelta(hours=3) >= cm.utc(cutoff) or cm.utc(r['kickoff']).year != 2022 for r in records): raise ValueError('future_training_label')
    valid = [r for r in records if r['available']]
    result = {'version': VERSION, 'market': market, 'stage': stage, 'cutoff': cutoff, 'coefficient': 0.,
              'fixture_ids': [r['fixture_id'] for r in valid], 'training_hash': cm.digest(records),
              'weeks': len({r['week'] for r in valid}), 'n': len(valid), 'penalty': 10., 'publication_enabled': False}
    if len(valid) < 500 or result['weeks'] < 26:
        result['status'] = 'insufficient_training_support'
    else:
        x = np.asarray([r['x'] for r in valid]); base = np.asarray([r['base'] for r in valid]); y = np.asarray([r['target'] for r in valid])
        if market == 'sot': x = x.mean(axis=1, keepdims=True)
        if np.any(base <= 0) or not all(np.all(np.isfinite(a)) for a in (x, base, y)) or np.any(y < 0): raise ValueError('invalid_fit_values')
        def objective(beta):
            mu = base*np.exp(beta*x)
            return float(np.sum(mu-y*np.log(mu)) + 10*beta*beta)
        fitted = minimize_scalar(objective, bounds=(0., 1.), method='bounded', options={'xatol': 1e-10})
        if not fitted.success: raise ValueError('coefficient_fit_failed')
        beta = min((0., float(fitted.x), 1.), key=objective)
        result.update(status='fitted', coefficient=beta, training_objective=objective(beta), zero_objective=objective(0.))
    result['id'] = cm.digest(result)
    return result


def apply(bundle, feature, base, *, public=False):
    if public: raise ValueError('research_player_effects_not_publicly_qualified')
    if bundle['id'] != cm.digest({k: v for k, v in bundle.items() if k != 'id'}): raise ValueError('changed_coefficient_bundle')
    if feature['id'] != cm.digest({k: v for k, v in feature.items() if k != 'id'}): raise ValueError('changed_player_snapshot')
    if bundle['stage'] != feature['stage'] or cm.utc(bundle['cutoff']) > cm.utc(feature['as_of']): raise ValueError('future_or_wrong_stage_bundle')
    market = bundle['market']; x = np.asarray(vector(feature, market))
    if market == 'sot': x = np.asarray([x.mean()])
    mu = np.asarray(base)*np.exp(bundle['coefficient']*x)
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)): raise ValueError('invalid_player_adjusted_mean')
    return mu.tolist()
