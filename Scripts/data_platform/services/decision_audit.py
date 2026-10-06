"""Observe the existing selector without choosing, rescoring or publishing bets.

The normal immutable Match Read transaction persists this audit with every
canonical result. Old records remain explicitly unaudited; there is no backfill.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from ..publication_identity import digest

VERSION = 'production-decision-audit.v1'


@lru_cache(maxsize=1)
def policy_identity():
    root = Path(__file__).resolve().parents[3]
    paths = ('Scripts/rag_ingest/core/line_selection.py',
             'Scripts/rag_ingest/core/weights.py',
             'Scripts/data_platform/services/match_read_compiler.py',
             'Scripts/web_app/routers/match_reads.py')
    files = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    return 'legacy-confidence-edge.v1:' + digest(files)


def _snapshot(value):
    if isinstance(value, dict):
        return {str(k): _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Preserve invalidity, never silently impute zero.
    return deepcopy(value)


def run_selector(context, selector, *args, **kwargs):
    """Call the original selector once, returning its untouched result."""
    option_position = 3 if selector.__name__ == 'choose_best_moneyline_side' else 0
    inputs = _snapshot(args[option_position])
    output = selector(*args, **kwargs)
    evaluated = output.get('all_lines', output.get('all_sides', []))
    chosen = output.get('bet_recommendation')
    candidates = []
    for position, row in enumerate(evaluated):
        selected = row is chosen or (chosen is not None and row == chosen)
        reason = ('selector_selected' if selected else
                  'eligible_but_not_highest_ranked' if row.get('_eligible') else
                  'existing_selector_eligibility_failed' if '_eligible' in row else
                  'not_scored_by_existing_selector')
        candidates.append({'occurrence': position, 'option': _snapshot(row),
                           'selector_selected': selected, 'reason': reason})
    context['decision_audit'] = {
        'schema_version': VERSION, 'policy_version': policy_identity(),
        'selector': selector.__name__, 'selector_ran': True,
        'input_options': inputs, 'selector_output': _snapshot(output),
        'candidates': candidates,
        'coverage': 'all_extracted_inputs_and_all_returned_evaluated_candidates',
        'coverage_limit': 'Original provider parsing is upstream; excluded input options remain in input_options. Eligibility reasons preserve existing gates rather than invent finer attribution.',
    }
    return output


def complete_market_audit(context, decision, provenance):
    from .communication_release import communication_release_manifest
    result = deepcopy(dict(context or {}))
    audit = result.setdefault('decision_audit', {
        'schema_version': VERSION, 'policy_version': policy_identity(),
        'selector_ran': False, 'coverage': 'selector_not_run', 'candidates': [],
    })
    audit['final_decision'] = decision.to_dict()
    audit['prediction_version'] = provenance.system_version
    audit['input_snapshot_id'] = provenance.input_snapshot_id
    audit['probability_version'] = decision.probability_version
    audit['release_identity'] = communication_release_manifest()['identity']
    audit['post_selector_reason'] = decision.reason
    return result


def fixture_decision_audit(results, selections):
    selected = {s['result_index']: s['role'] for s in selections}
    rows = []
    for index, result in enumerate(results):
        decision = result.get('decision') or {}
        audit = (result.get('context') or {}).get('decision_audit') or {}
        rows.append({'result_index': index, 'market': result.get('market'),
                     'selected': index in selected, 'role': selected.get(index),
                     'reason': ('selected' if index in selected else decision.get('reason')
                                if decision.get('status') != 'recommended' else
                                'not_selected_by_existing_compatibility_and_three_selection_limit'),
                     'candidate_coverage': audit.get('coverage', 'legacy_not_recorded'),
                     'policy_version': audit.get('policy_version')})
    return {'schema_version': VERSION, 'markets': rows,
            'joint_probability': None, 'joint_expected_value': None,
            'publication_claim': False}
