"""Independent mathematical and contract regressions for the Phase 4 repair.

All forecasts and quotes are synthetic. No provider, archive or reserved
outcomes are required by these tests.
"""
from copy import deepcopy
from math import erf, exp, factorial, sqrt
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag_ingest'))
import prob_models as probabilities
from core import line_selection as lines, projections, team_resolution, market_service
from core.market_result import Decision, MarketResultError, decision_from_selector
from core.odds_extraction import extract_total_line_options


def scores(home=2.0, away=1.0):
    matrix = probabilities.dixon_coles_scoreline_matrix(home, away)
    return {(h, a): p for h, row in enumerate(matrix) for a, p in enumerate(row)}


@pytest.mark.parametrize('home,away,rho', [(2., 1., -.1), (1., 3., -.1), (0., 0., 0.), (.1, 4., .02)])
def test_dc_mass_marginals_and_home_away_symmetry(home, away, rho):
    matrix = probabilities.dixon_coles_scoreline_matrix(home, away, rho, max_goals=2)
    reverse = probabilities.dixon_coles_scoreline_matrix(away, home, rho, max_goals=2)
    assert sum(map(sum, matrix)) == pytest.approx(1, abs=1e-12)
    for h, row in enumerate(matrix):
        assert min(row) >= 0
        expected = exp(-home) * home ** h / factorial(h)
        assert sum(row) == pytest.approx(expected, abs=1e-10)
        for a, probability in enumerate(row):
            assert probability == pytest.approx(reverse[a][h], abs=1e-12)
    # Independent formula, deliberately asymmetric rates catch the swapped factors.
    assert probabilities.dixon_coles_scoreline_prob(1, 0, home, away, rho) == pytest.approx(
        exp(-home-away) * home * (1+away*rho))
    assert probabilities.dixon_coles_scoreline_prob(0, 1, home, away, rho) == pytest.approx(
        exp(-home-away) * away * (1+home*rho))


@pytest.mark.parametrize('home,away,rho', [(2, 1, -2), (3, 2, 1), (-1, 1, 0), (float('nan'), 1, 0)])
def test_dc_rejects_invalid_distributions(home, away, rho):
    with pytest.raises(ValueError):
        probabilities.dixon_coles_scoreline_matrix(home, away, rho)


def test_dc_grid_tail_is_bounded_and_consistent_across_old_cutoffs():
    six = probabilities.dixon_coles_scoreline_matrix(3, 1, max_goals=6)
    seven = probabilities.dixon_coles_scoreline_matrix(3, 1, max_goals=7)
    assert six == seven
    assert len(six) > 8
    captured = sum(exp(-3)*3**k/factorial(k) for k in range(len(six)))
    captured *= sum(exp(-1)/factorial(k) for k in range(len(six)))
    assert 1-captured <= 1e-10


def normal_reference(mean, std, line):
    """Enumerate integer goal margins and settle each half stake independently."""
    result = dict.fromkeys(('full_win', 'half_win', 'push', 'half_loss', 'full_loss'), 0.)
    components = [line] if line * 2 == int(line * 2) else [line - .25, line + .25]
    for margin in range(-80, 81):
        probability = (erf((margin+.5-mean)/(std*sqrt(2)))-erf((margin-.5-mean)/(std*sqrt(2))))/2
        outcome = sum(1 if margin+h > 0 else -1 if margin+h < 0 else 0 for h in components)/len(components)
        bucket = {1:'full_win', .5:'half_win', 0:'push', -.5:'half_loss', -1:'full_loss'}[outcome]
        result[bucket] += probability
    return result


@pytest.mark.parametrize('line', [i/4 for i in range(-12,13)])
@pytest.mark.parametrize('mean,std', [(0.,1.25), (1.8,1.85), (-2.,1.15)])
def test_normal_asian_profile_matches_independent_settlement(line, mean, std):
    profile = lines._normal_spread_profile(mean, std, line)
    assert min(profile.values()) >= 0
    assert sum(profile.values()) == pytest.approx(1, abs=1e-12)
    assert profile == pytest.approx(normal_reference(mean,std,line), abs=1e-12)
    for odds in (1.8, 2., 2.2):
        expected = sum(profile[k]*profit for k,profit in
                       [('full_win',odds-1),('half_win',(odds-1)/2),('push',0),('half_loss',-.5),('full_loss',-1)])
        assert lines._spread_expected_value(profile,odds) == pytest.approx(expected)


def handicap_quotes(odds=2.):
    return [dict(team='Home',point=0.,odds=odds,bookmaker='Book',market_key='spreads',period='regulation_time'),
            dict(team='Away',point=0.,odds=2.,bookmaker='Book',market_key='spreads',period='regulation_time')]


def test_zero_handicap_pairs_and_probability_is_price_independent(monkeypatch):
    monkeypatch.setattr(lines, 'compute_margin_std', None)
    matrix={(1,0):.4,(0,0):.3,(0,1):.3}
    results = [lines.select_best_spread_recommendation(handicap_quotes(price), .1, 'Home', away_team='Away', score_probs=matrix)
               for price in (2.,3.)]
    home = [next(o for o in r['all_lines'] if o['team']=='Home') for r in results]
    assert all(o['_vig_adjusted_pair'] for o in home)
    assert all(o['_model_prob'] == pytest.approx(4/7) for o in home)
    assert home[0]['_ev'] != home[1]['_ev']
    assert home[0]['_settlement_profile']['push'] == .3
    assert home[0]['_probability_basis'] == 'asian_equivalent_non_push'


@pytest.mark.parametrize('field,value', [('market_key','other_handicap'),('period','first_half'),('settlement_definition','other'),('fixture_id','other')])
def test_spreads_never_pair_different_contracts(field,value,monkeypatch):
    monkeypatch.setattr(lines, 'compute_margin_std', None)
    quotes=handicap_quotes(); quotes[1][field]=value
    result=lines.select_best_spread_recommendation(quotes,.5,'Home',away_team='Away',score_probs={(1,0):.7,(0,1):.3})
    assert all(not o['_vig_adjusted_pair'] for o in result['all_lines'])
    assert result['bet_recommendation'] is None


@pytest.mark.parametrize('field,value', [('market_key','totals_yellow_cards'),('period','first_half'),('settlement_definition','yellow_only'),('fixture_id','other')])
def test_totals_never_pair_different_contracts(field,value):
    quotes=[dict(side='over',point=4.5,odds=2.,bookmaker='Book',market_key='totals_cards',period='regulation_time'),
            dict(side='under',point=4.5,odds=2.,bookmaker='Book',market_key='totals_cards',period='regulation_time')]
    quotes[1][field]=value
    result=lines.select_best_total_recommendation(quotes,6.,9.6)
    assert all(not o['_vig_adjusted_pair'] for o in result['all_lines'])
    assert result['bet_recommendation'] is None


def test_yellow_only_and_booking_points_never_use_generic_card_target():
    event={'id':'fixture','bookmakers':[{'title':'Book','markets':[
        {'key':key,'outcomes':[{'name':'Over','point':4.5,'price':2.}]} for key in
        ['totals_yellow_cards','totals_red_cards','totals_booking_points','totals_cards_over_under']]}]}
    assert [o['market_key'] for o in extract_total_line_options(event,'cards')] == ['totals_cards_over_under']


def test_spread_propagates_cutoff_and_context_once(monkeypatch):
    matrix=Mock(return_value=(scores(),2.,1.)); std=Mock(return_value=1.25)
    monkeypatch.setattr(lines,'projected_correct_score_probs',matrix)
    monkeypatch.setattr(lines,'compute_margin_std',std)
    context=object(); knockout=object()
    lines.select_best_spread_recommendation(handicap_quotes(),1.,'Home','Away','EPL',
        fixture_date='2024-01-01',league_ctx=context,knockout_ctx=knockout)
    matrix.assert_called_once_with('Home','Away','EPL',fixture_date='2024-01-01',league_ctx=context,knockout_ctx=knockout)
    std.assert_called_once_with('Home','Away','EPL',fixture_date='2024-01-01')


def test_context_reaches_all_goal_derivatives(monkeypatch):
    total=Mock(return_value=(3.,3.,3.))
    monkeypatch.setattr(projections,'projected_total_goals',total)
    monkeypatch.setattr(projections,'_profile_as_of',Mock(return_value={}))
    monkeypatch.setattr(projections,'projected_goals',Mock(return_value=(2.,1.)))
    recent=Mock(return_value={'xg_for_avg':None})
    monkeypatch.setattr(projections,'_recent_stats',recent)
    league=object(); knockout=object()
    kwargs=dict(league_ctx=league,knockout_ctx=knockout,fixture_date='2024-01-01')
    for fn in (projections.projected_correct_score_probs,projections.projected_btts_prob,
               projections.projected_moneyline_probs,projections.projected_goal_difference):
        fn('Home','Away','EPL',**kwargs)
    assert all(c.kwargs==kwargs for c in total.call_args_list)
    assert all(c.kwargs['target_date']=='2024-01-01' for c in recent.call_args_list)


def test_recent_goal_variance_uses_score_not_xg_and_preserves_zero(monkeypatch):
    rows=[{'meta':{'final_score':score,'home_away':venue,'xg_for':99.}} for score,venue in
          [('0-2','home'),('0-1','away'),('3-1','home')]]
    monkeypatch.setattr(team_resolution,'get_recent_team_fixture_rows',Mock(return_value=rows))
    team_resolution._team_recent_var_cache.clear()
    result=team_resolution.get_team_recent_variance('Home','EPL',target_date='2024-01-01')
    assert result['goals_var'] == pytest.approx(7/3)
    for row in rows: row['meta'].pop('final_score')
    team_resolution._team_recent_var_cache.clear()
    assert team_resolution.get_team_recent_variance('Home','EPL')['goals_var'] is None


def test_settlement_profile_survives_decision_serialization_without_reinterpreting_legacy():
    result=lines.select_best_total_recommendation([
        dict(side='over',point=2.75,odds=2.,bookmaker='Book',market_key='totals'),
        dict(side='under',point=2.75,odds=2.,bookmaker='Book',market_key='totals')],3.)
    decision=decision_from_selector(result)
    payload=decision.to_dict()
    assert Decision.from_dict(payload).to_dict()==payload
    assert payload['probability_version']=='market-probability.v2'
    assert payload['settlement_profile']['half_win']>0
    legacy={k:v for k,v in payload.items() if k not in {'settlement_profile','probability_basis','probability_version'}}
    assert Decision.from_dict(legacy).to_dict()==legacy
    payload['settlement_profile']['push']=-.1
    with pytest.raises(MarketResultError): Decision.from_dict(payload)


def test_canonical_goals_prices_shared_score_distribution(monkeypatch):
    matrix=scores(2.,1.)
    monkeypatch.setattr(market_service,'projected_total_goals',Mock(return_value=(3.,3.,3.)))
    score_spy=Mock(return_value=(matrix,2.,1.))
    monkeypatch.setattr(market_service,'projected_correct_score_probs',score_spy)
    monkeypatch.setattr(market_service,'_input_quality',Mock(return_value={}))
    monkeypatch.setattr(market_service,'_total_variance',Mock(return_value=(3.75,{'source':'conservative_fallback_floor'})))
    monkeypatch.setattr(market_service,'_apply_quality_guardrails',lambda decision,*a,**k:decision)
    event={'id':'synthetic','home_team':'Home','away_team':'Away','commence_time':'2024-01-01T12:00:00Z',
           'bookmakers':[{'title':'Book','markets':[{'key':'totals','outcomes':[
               {'name':'Over','point':2.5,'price':2.}, {'name':'Under','point':2.5,'price':2.}]}]}]}
    result=market_service.evaluate_market(event,'EPL','goals')
    assert result.decision.quote.side=='over'
    assert result.decision.model_probability==pytest.approx(sum(p for (h,a),p in matrix.items() if h+a>2.5))
    assert result.projection.components['distribution_version']=='dixon_coles.v2'
    assert score_spy.call_args.kwargs['fixture_date']==event['commence_time']
    counts={}
    for (h,a),p in matrix.items(): counts[h+a]=counts.get(h+a,0)+p
    for line in (0.,.25,.5,1.,2.25,2.75,3.,4.5):
        over=probabilities.asian_total_profile_from_counts(counts,line,'over')
        under=probabilities.asian_total_profile_from_counts(counts,line,'under')
        assert over['full_win']==pytest.approx(under['full_loss'])
        assert over['half_win']==pytest.approx(under['half_loss'])
        assert over['push']==pytest.approx(under['push'])


def test_all_goal_derivatives_share_one_serializable_cached_snapshot(tmp_path,monkeypatch):
    from core.projection_cache import ProjectionStore, statistical_reuse
    from core import projection_cache
    team_split=Mock(return_value=(2.,1.,1.8,1.2))
    monkeypatch.setattr(projections,'_goal_market_team_projections',team_split)
    monkeypatch.setattr(projections,'_recent_stats',Mock(return_value={'xg_for_avg':1.}))
    store=ProjectionStore(tmp_path/'cache.sqlite')
    def evaluate():
        matrix,h,a=projections.projected_correct_score_probs('Home','Away','EPL')
        btts=projections.projected_btts_prob('Home','Away','EPL')[0]
        winner=projections.projected_moneyline_probs('Home','Away','EPL')[:3]
        margin=projections.projected_goal_difference('Home','Away','EPL')[0]
        return matrix,btts,winner,margin
    with statistical_reuse('synthetic',enabled=True,store=store,revision_provider=lambda:'fixed'):
        first=evaluate()
    assert team_split.call_count==1
    with statistical_reuse('synthetic',enabled=True,store=store,revision_provider=lambda:'fixed'):
        assert evaluate()==first
    assert team_split.call_count==1
    matrix,btts,winner,margin=first
    assert btts==pytest.approx(sum(p for (h,a),p in matrix.items() if h and a))
    assert sum(winner)==pytest.approx(1.)
    assert margin==pytest.approx(1.,abs=1e-8)


def test_contract_rejects_inconsistent_probability_and_preserves_quote_rules():
    from core.market_result import PriceQuote
    quote=PriceQuote(side='over',line=3.,odds=2.,period='regulation_time',settlement_definition='synthetic_rule')
    assert PriceQuote.from_dict(quote.to_dict())==quote
    with pytest.raises(MarketResultError,match='disagrees'):
        Decision(status='recommended',quote=quote,model_probability=.4,
                 probability_basis='asian_equivalent_non_push',probability_version='market-probability.v2',
                 settlement_profile=dict(full_win=.4,half_win=0.,push=.3,half_loss=0.,full_loss=.3))


@pytest.mark.parametrize('mean,var', [(1.,None),(3.,None),(6.,None),(10.,None),(1.,1.6),(3.,4.8),(6.,9.6),(10.,16.)])
@pytest.mark.parametrize('line', [.5,1.,1.25,1.75,2.5,3.,3.25,3.75,9.5,10.25])
@pytest.mark.parametrize('side', ['over','under'])
def test_existing_total_arithmetic_against_independent_count_enumeration(mean,var,line,side):
    # Count probabilities from scipy, half-stake settlement independent of production helpers.
    from scipy.stats import poisson, nbinom
    distribution=poisson(mean) if var is None else nbinom(mean*mean/(var-mean),mean/var)
    components=[line] if line*2==int(line*2) else [line-.25,line+.25]
    profile=dict.fromkeys(('full_win','half_win','push','half_loss','full_loss'),0.)
    direction=1 if side=='over' else -1
    for count,probability in enumerate(distribution.pmf(range(250))):
        outcome=sum(direction*(1 if count>h else -1 if count<h else 0) for h in components)/len(components)
        profile[{1:'full_win',.5:'half_win',0:'push',-.5:'half_loss',-1:'full_loss'}[outcome]]+=probability
    assert probabilities.asian_total_settlement_profile(mean,line,side,var)==pytest.approx(profile,abs=1e-11)
