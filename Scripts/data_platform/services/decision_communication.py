"""One numerical explanation and identity for both delivery surfaces.

Prose is derived after selection. Neither formatting nor a text generator can
change a pick, a probability, a price, an evidence claim, or its identity.
"""
from __future__ import annotations
from copy import deepcopy
import math
from datetime import datetime, timezone

from ..publication_identity import digest
from ..recommendation_identity import recommendation_identity

VERSION = 'decision-communication.production.v1'



def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def build_decision_explanation(result):
    decision = result.get('decision') or {}
    context = result.get('context') or {}
    quote = decision.get('quote') or {}
    chosen = (context.get('bet_justification') or {}).get('selected') or {}
    price = chosen.get('price') or {}
    probability, ev = _finite(decision.get('model_probability')), _finite(decision.get('expected_value'))
    conservative = _finite(chosen.get('conservative_ev'))
    odds, floor = _finite(quote.get('odds')), _finite(chosen.get('minimum_acceptable_odds'))
    market = (result.get('market') or {}).get('group') or 'market'
    market_label = {'sot': 'Shots on target', 'moneyline': 'Match winner', 'spreads': 'Handicap', 'btts': 'BTTS'}.get(market, market.title())
    side, line = str(quote.get('side') or ''), quote.get('line')
    pick = side.title() + (f' {float(line):g}' if line is not None else '')
    if market == 'moneyline':
        pick = (result.get('fixture') or {}).get(side.lower() + '_team') or pick
    facts = []
    if decision.get('status') == 'recommended' and odds is not None:
        facts.append(f'{market_label}: {pick} @ {odds} ({quote.get("bookmaker") or "bookmaker unavailable"}).')
        if probability is not None:
            label = 'Asian price-comparison probability' if decision.get('probability_basis') == 'asian_equivalent_non_push' else 'model probability'
            facts.append(f'{probability:.1%} {label}.')
        if ev is not None:
            facts.append(f'Estimated return {ev:+.1%} per unit staked.')
        if conservative is not None:
            facts.append(f'Conservative scenario return {conservative:+.1%}; it clears the policy threshold.')
    else:
        facts.append(decision.get('reason') or 'No supported selection is available.')
    uncertainty = ('Scenario estimates reflect model uncertainty; they are not a guaranteed return or a stated confidence interval.'
                   if conservative is not None else 'Estimation uncertainty has not been quantified. Confidence labels are heuristic ratings, not win probabilities.')
    if chosen.get('support_grade'):
        uncertainty += f' Evidence support: {chosen["support_grade"]}.'
    timing = []
    audit = context.get('decision_audit') or {}
    observed = audit.get('source_updated_at')
    if observed:
        timing.append(f'Provider update {observed}.')
    elif not price.get('execution_checked_at'):
        timing.append('Quote timestamp unavailable.')
    if not price.get('execution_checked_at'):
        timing.append('Bookmaker availability has not been verified.')
    if floor is not None:
        displayed = (math.floor(floor * 100 + 1e-8) + 1) / 100
        timing.append(f'Minimum odds {displayed:.2f}.')
    if price.get('execution_checked_at'):
        timing.append(f'Price checked {price["execution_checked_at"]}.')
    if price.get('expires_at'):
        timing.append(f'Valid before {price["expires_at"]}, subject to continued availability.')
    compact = f'{market.upper() if market == "btts" else market.title()}: {pick} @ {odds} ({quote.get("bookmaker") or "bookmaker unavailable"}).'
    if probability is not None:
        compact += f' {probability:.1%} ' + ('Asian price-comparison probability.' if decision.get('probability_basis') == 'asian_equivalent_non_push' else 'model probability.')
    if ev is not None:
        compact += f' EV {ev:+.1%}.'
    if conservative is not None:
        compact += f' Conservative scenario {conservative:+.1%}.'
    if conservative is not None:
        compact += f'\nEvidence support: {chosen.get("support_grade", "unquantified")}; scenarios are not guarantees or confidence intervals.'
    else:
        compact += '\nUncertainty unquantified; availability unverified.'
    if floor is not None:
        compact += f'\nMinimum odds {displayed:.2f}.'
    if price.get('execution_checked_at') and price.get('expires_at'):
        checked = datetime.fromisoformat(price['execution_checked_at'].replace('Z', '+00:00')).astimezone(timezone.utc)
        expires = datetime.fromisoformat(price['expires_at'].replace('Z', '+00:00')).astimezone(timezone.utc)
        if checked.date() == expires.date():
            compact += f' Price checked {checked:%H:%M:%S}; expires {expires:%H:%M:%S} ({checked:%Y-%m-%d} UTC).'
        else:
            compact += f' Price checked {checked.isoformat()}; expires {expires.isoformat()}.'
        compact += ' Availability may change.'
    return {'schema_version': VERSION, 'recommendation_id': recommendation_identity(result),
            'status': decision.get('status'), 'selection': pick, 'market': market, 'line': line,
            'quoted_price': odds, 'bookmaker': quote.get('bookmaker'),
            'compact': compact, 'reasoning': ' '.join(facts), 'uncertainty': uncertainty, 'price_conditions': ' '.join(timing),
            'minimum_odds_display': displayed if floor is not None else None,
            'probability_basis': decision.get('probability_basis'), 'model_probability': probability,
            'estimated_ev': ev, 'conservative_ev': conservative,
            'uncertainty_method': chosen.get('uncertainty_method'),
            'scenario_ev_range': chosen.get('scenario_ev_range'),
            'source_updated_at': (quote.get('price_evidence') or {}).get('source_updated_at') or observed,
            'checked_at': price.get('execution_checked_at'), 'expires_at': price.get('expires_at'),
            'policy_version': (context.get('selection_policy') or {}).get('version') or audit.get('policy_version'),
            'prediction_version': (result.get('provenance') or {}).get('system_version')}


def decision_briefing(draft, generator=None):
    """Optional generator orders verified sentences; arbitrary prose is ignored.

    Always include all numerical explanations and their uncertainty. Requiring a
    complete permutation prevents the generator from omitting an awkward pick.
    """
    selected = [draft.canonical_results[item['result_index']] for item in draft.selections]
    explanations = [build_decision_explanation(result) for result in selected]
    if not explanations:
        explanations = [build_decision_explanation(result) for result in draft.canonical_results]
    fragments = {f'decision-{i}': item['reasoning'] for i, item in enumerate(explanations)}
    order, source = list(fragments), 'deterministic_decision_facts'
    if generator is not None:
        try:
            response = generator(deepcopy({'sentences': fragments, 'instruction': 'Return sentence_order containing every ID exactly once; do not add or change facts.'}))
            proposed = response.get('sentence_order')
            if (isinstance(proposed, list) and len(proposed) == len(order)
                    and all(isinstance(key, str) for key in proposed) and set(proposed) == set(order)
                    and set(response) == {'sentence_order'}):
                order, source = proposed, 'verified_sentence_order'
        except Exception:
            pass
    return {'schema_version': VERSION, 'source': source, 'model': None,
            'summary': ' '.join(fragments[key] for key in order),
            'bullets': [item['uncertainty'] for item in explanations[:3]],
            'recommendation_ids': [item['recommendation_id'] for item in explanations],
            'fact_key': digest(explanations)}
