"""Replay the preserved numerical engine from synthetic external-input snapshots.

No projection, probability, selector, quality or variance arithmetic is mocked.
Only profile/recent-history/context lookups are replaced by explicitly saved inputs.
Invoke in a fresh process against --source-root; no databases or networking.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def scenarios():
    profile = dict(cards_per_90_team=2., opp_cards_induced_pm=1.5, corners_pm=5.,
        goals_for_pm=1.5, goals_against_pm=1.2, sot_for_pm=4., sot_against_pm=3.5,
        control_index=.55, dominance_index=.5, possession=.52, aggression_index_norm=.3,
        fouls_per_90_team=11., cards_per_foul_team=.18, shots_for_pm=12., form_index_team=.55,
        goals_var=1.8, corners_var=7., cards_var=2.8, sot_var=5., yellows_pm=1.8, reds_pm=.2)
    recent = dict(n=6, cards_avg=2., cards_induced_avg=1.5, corners_for_avg=5.,
        corners_against_avg=4.5, sot_for_avg=4., sot_against_avg=3.5, xg_for_avg=1.5,
        shots_for_avg=12., fouls_avg=11., aggression_avg=.3, control_avg=.55, form_avg=.55,
        possession_avg=.52, corners_for_slope=0., sot_for_slope=0., cards_slope=0.,
        goals_slope=0., yellows_avg=None, reds_avg=None, yellows_induced_avg=None,
        reds_induced_avg=None, cards_per_foul_avg=None, fouls_drawn_avg=None)
    quality = dict(temporal_status='fixture_rows_strictly_before_target_date',
                   current_season_matches=20, effective_sample_size=20.)
    cases = []
    for name in ('standard', 'quarter_lines', 'missing_variance', 'one_sided_variance',
                 'sparse_support', 'missing_profiles', 'context_adjusted', 'continental', 'high_variance'):
        case = {'id': name, 'kind': 'synthetic_external_input_snapshot', 'league': 'EPL',
                'profiles': {'Home': deepcopy(profile), 'Away': deepcopy(profile)},
                'recent': {'Home': deepcopy(recent), 'Away': deepcopy(recent)},
                'recent_variance': {team: dict(goals_var=1.4, corners_for_var=6., cards_var=2.3, sot_for_var=4.) for team in ('Home','Away')},
                'profile_quality': {team: deepcopy(quality) for team in ('Home','Away')},
                'league_context': {'adjustments': {'goals':1.,'corners':1.,'cards':1.,'sot':1.}},
                'knockout_context': {'is_knockout':False,'goals_modifier':1.,'corners_modifier':1.,'cards_modifier':1.,'sot_modifier':1.},
                'referee': {'source':'unavailable','multiplier':1.,'confidence':0.,'sample_size':0,
                            'avg_cards_per_match':0.,'cards_per_foul':0.,'avg_fouls_per_match':0.,'strictness_ratio':1.,'referee_name':None},
                'lineup': None, 'card_risk_fallback': [None,None]}
        case['profiles']['Away'].update(goals_for_pm=1.1, goals_against_pm=1.6, corners_pm=4., sot_for_pm=3.5)
        case['recent']['Away']['xg_for_avg'] = .9
        quarter = .25 if name == 'quarter_lines' else 0.
        markets = []
        for key, line in [('totals',2.5), ('totals_corners',9.5), ('totals_cards_over_under',4.5), ('shots_on_target_over_under',8.5)]:
            markets.append({'key':key,'period':'regulation_time','outcomes':[
                {'name':side,'point':line+quarter,'price':2.} for side in ('Over','Under')]})
        markets += [{'key':'btts','outcomes':[{'name':s,'price':2.} for s in ('Yes','No')]},
                    {'key':'h2h','outcomes':[{'name':s,'price':p} for s,p in [('Home',2.3),('Draw',3.3),('Away',3.2)]]},
                    {'key':'spreads','outcomes':[{'name':'Home','point':-.25 if quarter else 0.,'price':2.},
                                               {'name':'Away','point':.25 if quarter else 0.,'price':2.}]}]
        case['event'] = {'id':'synthetic-phase4-'+name,'home_team':'Home','away_team':'Away',
                         'commence_time':'2023-10-01T15:00:00Z','bookmakers':[{'title':'Synthetic Book','markets':markets}]}
        if name in ('missing_variance','one_sided_variance'):
            for team in (('Home','Away') if name == 'missing_variance' else ('Home',)):
                for field in ('goals_var','corners_var','cards_var','sot_var'):
                    case['profiles'][team][field] = None
                case['recent_variance'][team] = {}
        if name == 'high_variance':
            for team in ('Home','Away'):
                for field in ('goals_var','corners_var','cards_var','sot_var'):
                    case['profiles'][team][field] = 20.
                case['recent_variance'][team] = {k:25. for k in case['recent_variance'][team]}
        if name == 'sparse_support':
            for team in ('Home','Away'):
                case['profile_quality'][team].update(current_season_matches=1,effective_sample_size=2.)
                case['recent'][team]['n'] = 1
        if name == 'missing_profiles':
            for team in ('Home','Away'):
                case['profiles'][team] = {}; case['recent'][team] = {'n':0}; case['recent_variance'][team] = {}
        if name == 'context_adjusted':
            case['league_context']['adjustments'].update(goals=1.05,corners=.95,cards=1.1,sot=1.03)
            case['knockout_context'].update(is_knockout=True,goals_modifier=.9,cards_modifier=1.1)
            case['referee'].update(source='profile',sample_size=30,confidence=.8,multiplier=1.1,
                                   avg_cards_per_match=4.8,cards_per_foul=.2,avg_fouls_per_match=24.)
        if name == 'continental': case['league'] = 'UCL'
        cases.append(case)
    return cases


def install_guard(source, output):
    """Trusted research IO guard; protect live data even when imported code catches errors."""
    violations = []
    def audit(event, args):
        reason = None
        if event in ('socket.connect','socket.getaddrinfo','sqlite3.connect','subprocess.Popen','os.system'):
            reason = event
        if event == 'open' and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1:3]
            writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and path != output:
                reason = 'write_outside_replay_output:' + str(path)
            elif path.name == '.env' or path.suffix in ('.db','.sqlite','.sqlite3'):
                reason = 'protected_input:' + str(path)
            elif 'Index' in path.parts or 'Output' in path.parts:
                if not path.is_relative_to(source/'Index/ml_models'):
                    reason = 'non_frozen_data:' + str(path)
        if reason:
            violations.append(reason)
            raise RuntimeError(reason)
    sys.addaudithook(audit)
    return violations


def replay(source, inputs, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    cases = json.loads(Path(inputs).read_text())
    sys.dont_write_bytecode = True
    # Never load a local .env. The process has an explicit, isolated shadow setting.
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
    dotenv.dotenv_values = lambda *a, **k: {}
    os.environ['PREDICTION_RELEASE_MODE'] = 'shadow'
    violations = install_guard(source, output)
    sys.path[:0] = [str(source), str(source/'Scripts'), str(source/'Scripts/rag_ingest')]
    from core import projections, team_resolution, market_service
    from core.weights import SCORING_WEIGHTS
    from core.system_identity import prediction_system_manifest
    weights = {market: projections._ml_blend_weight(market) for market in ('goals','corners','cards','sot')}
    if any(value != 0 for value in weights.values()):
        raise ValueError('Control no longer has zero ML contribution')
    recorded = []
    for case in cases:
        def profile(team, *args, **kwargs): return deepcopy(case['profiles'][team])
        def recent(team, *args, **kwargs): return deepcopy(case['recent'][team])
        def variance(team, *args, **kwargs): return deepcopy(case['recent_variance'][team])
        def quality(team, *args, **kwargs): return profile(team), deepcopy(case['profile_quality'][team])
        with ExitStack() as stack:
            for module, name, value in [(projections,'_profile_meta',profile), (team_resolution,'_profile_meta',profile),
                    (projections,'_recent_stats',recent), (team_resolution,'get_team_recent_variance',variance),
                    (market_service,'get_prediction_profile_context',quality),
                    (projections,'get_card_risk_profiles',lambda *a,**k:(None,None))]:
                stack.enter_context(patch.object(module,name,side_effect=value))
            kwargs = {'fixture_date':case['event']['commence_time'],
                      'league_ctx':SimpleNamespace(**case['league_context']),
                      'knockout_ctx':SimpleNamespace(**case['knockout_context'])}
            ref = SimpleNamespace(**case['referee'])
            results = market_service.evaluate_event(case['event'],case['league'], **kwargs, ref_mod=ref,
                input_snapshot_id='synthetic:'+case['id'], generated_at='2023-09-30T12:00:00Z',
                model_version='phase4-repaired-statistical-control.v1')
            score, home, away = projections.projected_correct_score_probs('Home','Away',case['league'],**kwargs)
            if score is not None:
                if not math.isclose(sum(score.values()),1.,abs_tol=1e-10) or min(score.values()) < 0:
                    raise ValueError('Invalid shared goal distribution')
            row = {'id':case['id'], 'results':[r.to_dict() for r in results],
                   'score_distribution':[[h,a,p] for (h,a),p in sorted((score or {}).items())],
                   'goal_means':[home,away]}
            recorded.append(row)
    if violations:
        raise RuntimeError('Caught forbidden IO during replay: '+repr(violations))
    result = {'version':'phase4-control-replay.v1','kind':'synthetic_external_input_snapshots',
              'ml_weights':weights,'scoring_weights':SCORING_WEIGHTS,
              'system_identity':prediction_system_manifest(),'scenarios':recorded,
              'io_violations':violations,'publication_enabled':False,
              'boundary':'saved profile/recent aggregates, recent variance, profile evidence, external contexts and synthetic quotes; numerical functions unchanged'}
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False,default=lambda x:sorted(x) if isinstance(x,set) else vars(x))+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    replay(args.source_root,args.inputs,args.output)


if __name__ == '__main__': main()
