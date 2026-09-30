"""Quote-definition, canonical specialist parity and no-bet regression tests."""
import copy
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag_ingest'))
from core import odds_extraction as extraction, line_selection, market_service, quote_assessment, parlay
from core.market_contract import compatible
from prob_models import dixon_coles_scoreline_matrix
import rag_cli_v2 as rag
from odds_provider import _transform_apifootball_odds


def event(key='totals', outcomes=None, **metadata):
    return {'id':'10','_league':'EPL','home_team':'Arsenal','away_team':'Chelsea',
            'commence_time':'2026-09-28T18:00:00Z','bookmakers':[{'title':'Book','markets':[
                {'key':key, **metadata, 'outcomes': outcomes or [
                    {'name':'Over','point':2.,'price':2.1},
                    {'name':'Under','point':2.,'price':1.85}]}]}]}


@pytest.fixture
def numerical_engine():
    matrix=dixon_coles_scoreline_matrix(2.,1.1)
    scores={(h,a):p for h,row in enumerate(matrix) for a,p in enumerate(row)}
    audit={'temporal_status':'fixture_rows_strictly_before_target_date','current_season_matches':20,'effective_sample_size':20}
    with patch.object(market_service,'projected_total_goals',return_value=(3.1,3.,3.2)), \
         patch.object(market_service,'projected_correct_score_probs',return_value=(scores,2.,1.1)), \
         patch.object(line_selection,'projected_correct_score_probs',return_value=(scores,2.,1.1)), \
         patch.object(market_service,'projected_goal_difference',return_value=(.9,.8,1.)), \
         patch.object(market_service,'get_prediction_profile_context',return_value=({'goals_for_pm':1.6},audit)), \
         patch.object(market_service,'get_blended_variance',return_value={'goals_var':2.}), \
         patch.dict('os.environ',{'PREDICTION_RELEASE_MODE':'live'}):
        yield

@pytest.mark.parametrize('key,market', [('totals','goals'),('totals_corners_over_under','corners'),('totals_cards_over_under','cards'),('shots_on_target_over_under','sot'),('h2h','moneyline'),('btts','btts'),('spreads','spreads')])
@pytest.mark.parametrize('metadata',[{'period':'first_half'},{'period':'extra_time'},{'market_period':'second_half'},{'includes_extra_time':True},{'settlement_definition':'yellow_only'}])
def test_explicit_incompatibility_never_reaches_forecast(key,market,metadata):
    ev=event(key,**metadata)
    extractor={'moneyline':extraction.extract_moneyline_odds,'btts':extraction.extract_btts_odds,'spreads':extraction.extract_spread_line_options}.get(market)
    assert (extractor(ev) if extractor else extraction.extract_total_line_options(ev,market)) == []
    assert parlay.build_candidates([ev]) == []
    assert rag.build_candidates([ev]) == []

@pytest.mark.parametrize('key,group',[('total_shots','sot'),('totals_shots','goals'),('corners_handicap','spreads'),('european_handicap','spreads'),('team_totals_home_goals','goals'),('totals_yellow_cards','cards'),('totals_booking_points','cards'),('corners_1x2','moneyline')])
def test_statistic_and_scope_must_match(key,group):
    assert not compatible({'key':key},group)


def test_provider_preserves_declared_rules_and_source():
    books=_transform_apifootball_odds([{'name':'Book','bets':[{'id':5,'name':'Goals Over/Under','period':'first_half','settlement_definition':'test-definition','values':[{'value':'Over 2.5','odd':'2.0'}]}]}],'Arsenal','Chelsea')
    mk=books[0]['markets'][0]
    assert mk['provider_bet_id']==5 and mk['provider_bet_name']=='Goals Over/Under'
    assert mk['period']=='first_half' and mk['settlement_definition']=='test-definition'
    assert not extraction.extract_total_line_options({'bookmakers':books},'goals')


def test_btts_unknown_side_and_mixed_contract_pair_are_not_recommendations():
    ev=event('btts',[{'name':'Maybe','price':2.},{'name':'Yes','price':2.}])
    assert [x['side'] for x in extraction.extract_btts_odds(ev)]==['Yes']
    result=line_selection.select_best_btts_recommendation([
        {'side':'Yes','odds':2.,'bookmaker':'Book','market_key':'btts','period':'regulation_time'},
        {'side':'No','odds':2.,'bookmaker':'Book','market_key':'btts','period':'first_half'}],.7)
    assert result['bet_recommendation'] is None
    assert all(not row['_vig_adjusted_pair'] for row in result['all_sides'])


def test_moneyline_mixed_periods_do_not_form_complete_book():
    rows=[{'side':side,'odds':odd,'bookmaker':'Book','market_key':'h2h','period':period} for side,odd,period in [('home',2.,'regulation_time'),('draw',3.5,'first_half'),('away',4.,'regulation_time')]]
    result=line_selection.choose_best_moneyline_side(.7,.15,.15,rows,'Arsenal','Chelsea')
    assert result['bet_recommendation'] is None
    assert all(not row['_vig_adjusted_pair'] for row in result['all_sides'])


def test_asian_leg_uses_identical_canonical_profile_ev_and_cutoff(numerical_engine):
    ev=event()
    leg=next(x for x in rag.build_candidates([ev]) if x.outcome=='Over')
    canonical=market_service.evaluate_market(ev,'EPL','goals')
    actual=quote_assessment.assess_leg(leg,'EPL')
    assert actual['model_prob']==canonical.decision.model_probability
    assert actual['settlement_profile']==canonical.decision.settlement_profile
    assert actual['settlement_profile']['push']>0
    assert actual['expected_value']==canonical.decision.expected_value
    assert actual['probability_basis']=='asian_equivalent_non_push'
    assert actual['eligible']==canonical.decision.is_recommended
    assert market_service.projected_total_goals.call_args.kwargs['fixture_date']==ev['commence_time']
    assert rag._leg_standalone_confidence(leg,'EPL')==actual
    assert parlay._combo_leg_model_probability(leg,'EPL')==actual['model_prob']


def test_price_change_does_not_change_asian_probability(numerical_engine):
    ev=event(); leg=next(x for x in rag.build_candidates([ev]) if x.outcome=='Over')
    first=quote_assessment.assess_leg(leg,'EPL')
    ev2=copy.deepcopy(ev);ev2['bookmakers'][0]['markets'][0]['outcomes'][0]['price']=2.3
    other=replace(leg,odds=2.3,event=ev2)
    second=quote_assessment.assess_leg(other,'EPL')
    assert first['model_prob']==second['model_prob']
    assert first['settlement_profile']==second['settlement_profile']
    assert first['expected_value']!=second['expected_value']


def test_missing_quote_cannot_inherit_another_price(numerical_engine):
    leg=rag.build_candidates([event()])[0]
    assert quote_assessment.assess_leg(replace(leg,odds=3.),'EPL')['model_prob'] is None
    assert quote_assessment.assess_leg(replace(leg,event=None),'EPL')['model_prob'] is None


def test_no_bet_opposite_quote_and_missing_cutoff_stay_unqualified(numerical_engine):
    leg=next(x for x in rag.build_candidates([event()]) if x.outcome=='Under')
    assessment=quote_assessment.assess_leg(leg,'EPL')
    assert not assessment['eligible'] and assessment['model_prob'] is not None
    assert rag.kb_leg_quality(leg,'EPL')==0
    ev=copy.deepcopy(leg.event);del ev['commence_time']
    assert quote_assessment.assess_leg(replace(leg,event=ev),'EPL')['model_prob'] is None


def test_cache_lives_only_within_request(numerical_engine):
    leg=rag.build_candidates([event()])[0]
    with patch.object(quote_assessment,'evaluate_market_quote',wraps=market_service.evaluate_market_quote) as call:
        @quote_assessment.assessment_scope
        def run():
            quote_assessment.assess_leg(leg,'EPL');quote_assessment.assess_leg(leg,'EPL')
        run();assert call.call_count==1
        run();assert call.call_count==2


def test_combo_does_not_invent_joint_ev_or_missing_probabilities():
    legs=rag.build_candidates([event()])
    c=rag.ConstraintSpec(requested_markets=set(),hard_include_groups=set(),hard_exclude_groups=set(),soft_prefer_groups=set(),required_group_counts={},forbid_spread_keys=False,require_total_corner_keys=False,target_multiplier=None,target_mode="none",leg_count=2,per_match_mode=False,require_unique_events=False,time_window="upcoming",league="EPL")
    for module in (rag,parlay):
        with patch.object(quote_assessment,'assess_leg',side_effect=AssertionError('No joint probability calculation')):
            assert isinstance(module.score_combo(legs,c,{}),float)


def test_ambiguous_market_definition_is_unavailable(numerical_engine):
    ev=event();ev['bookmakers'][0]['markets'].append(copy.deepcopy(ev['bookmakers'][0]['markets'][0]))
    leg=rag.build_candidates([ev])[0]
    assert not quote_assessment.assess_leg(leg,'EPL')['eligible']

@pytest.mark.parametrize('market,key,outcomes,projection_name,projection',[
    ('corners','totals_corners_over_under',[{'name':'Over','point':9.5,'price':2.1},{'name':'Under','point':9.5,'price':1.8}],'projected_total_corners',(11.,10.,12.)),
    ('cards','totals_cards_over_under',[{'name':'Over','point':4.5,'price':2.1},{'name':'Under','point':4.5,'price':1.8}],'projected_total_cards',(5.5,5.,6.,None)),
    ('sot','shots_on_target_over_under',[{'name':'Over','point':8.5,'price':2.1},{'name':'Under','point':8.5,'price':1.8}],'projected_total_sot',(10.,9.,11.)),
    ('btts','btts',[{'name':'Yes','price':2.1},{'name':'No','price':1.8}],'projected_btts_prob',(.7,2.,1.1,2.,1.1)),
    ('moneyline','h2h',[{'name':'Arsenal','price':2.1},{'name':'Draw','price':3.5},{'name':'Chelsea','price':4.}],'projected_moneyline_probs',(.7,.15,.15,2.,1.1)),
    ('spreads','spreads',[{'name':'Arsenal','point':-.25,'price':2.1},{'name':'Chelsea','point':.25,'price':1.8}],'projected_goal_difference',(.9,.8,1.)),
])
def test_all_supported_specialist_assessments_preserve_canonical_metrics(numerical_engine,market,key,outcomes,projection_name,projection):
    ev=event(key,outcomes)
    with patch.object(market_service,projection_name,return_value=projection):
        reference=market_service.evaluate_market(ev,'EPL',market)
        decision=reference.decision
        assert decision.quote is not None
        leg=next(x for x in rag.build_candidates([ev]) if x.odds==decision.quote.odds)
        result=quote_assessment.assess_leg(leg,'EPL')
    assert result['model_prob']==decision.model_probability
    assert result['expected_value']==decision.expected_value
    assert result['settlement_profile']==decision.settlement_profile
    assert result['probability_basis']==decision.probability_basis
    assert result['eligible']==decision.is_recommended


def test_shadow_release_cannot_be_bypassed_by_parlay(numerical_engine):
    leg=rag.build_candidates([event()])[0]
    with patch.dict('os.environ',{'PREDICTION_RELEASE_MODE':'shadow'}):
        assessed=quote_assessment.assess_leg(leg,'EPL')
    assert not assessed['eligible'] and assessed['status']=='no_bet'
    assert 'Shadow' in assessed['warning']


def test_same_game_and_cross_book_products_are_not_automatic_unit_stakes():
    # Load only pure guard functions; importing the cog is unnecessary for this check.
    import ast
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'discord_bot/cogs/auto_push.py'
    names={'_unit_leg_market_group','_is_integer_line','_is_unit_integer_under','_unit_bet_rejected_legs','_unit_leg_tracker_payload','_tracker_market_for_unit_leg','_tracker_side_for_unit_leg'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace={'_UNIT_BET_MAX_CARD_LEGS':1,'rag':rag}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    def leg(event_id,book):
        return SimpleNamespace(event_id=event_id,bookmaker=book,market_group='moneyline',outcome='Arsenal',probability_basis='outcome_probability',settlement_profile=None,expected_value=.1)
    reject=namespace['_unit_bet_rejected_legs']
    assert len(reject(SimpleNamespace(selected_legs=[leg('1','A'),leg('1','A')]),0))==2
    assert len(reject(SimpleNamespace(selected_legs=[leg('1','A'),leg('2','B')]),0))==2
    assert reject(SimpleNamespace(selected_legs=[leg('1','A'),leg('2','A')]),0)==[]
    payload=namespace['_unit_leg_tracker_payload'](leg('1','A'))
    assert payload['probability_basis']=='outcome_probability'
    assert payload['expected_value']==.1


def test_legacy_leg_records_do_not_acquire_new_probability_versions():
    from core.parlay_models import ParlayLegResult
    record=dict(event_id='1',league='EPL',fixture='A vs B',home_team='A',away_team='B',market_key='totals',market_group='totals',outcome='Over',point=2.5,pick_display='Over 2.5',odds=2.,bookmaker='Book',quality_score=.1,confidence='medium',model_prob=.6,implied_prob=.5,value_edge=.1,projected_total=3.)
    old=ParlayLegResult.from_dict(record)
    assert old.probability_version is None and old.settlement_profile is None
    record.update(probability_basis='asian_equivalent_non_push',probability_version='market-probability.v2',settlement_profile={'full_win':.6,'half_win':0.,'push':.1,'half_loss':0.,'full_loss':.3},expected_value=.3)
    fresh=ParlayLegResult.from_dict(record)
    assert ParlayLegResult.from_dict(fresh.to_dict()).settlement_profile==record['settlement_profile']
