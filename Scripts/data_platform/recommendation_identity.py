"""Canonical recommendation identity, independent of explanation formatting."""
from .publication_identity import digest

def recommendation_identity(result):
    context = result.get('context') or {}
    provenance = result.get('provenance') or {}
    return 'recommendation.v1:' + digest({
        'fixture': result.get('fixture'), 'market': result.get('market'),
        'quote': (result.get('decision') or {}).get('quote'),
        'input_snapshot_id': provenance.get('input_snapshot_id'),
        'prediction_version': provenance.get('system_version'),
        'probability_version': (result.get('decision') or {}).get('probability_version'),
        'policy_version': (context.get('selection_policy') or {}).get('version') or (context.get('decision_audit') or {}).get('policy_version') or 'legacy_unversioned',
        'stage': context.get('match_read_stage', context.get('forecast_stage', 'pre_match'))})
