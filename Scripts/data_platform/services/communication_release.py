"""Capture communication/audit source identity when a process imports it."""
from copy import deepcopy
import hashlib
from pathlib import Path
from ..publication_identity import digest

VERSION = 'communication-audit.2026-10-06.v1'
_ROOT = Path(__file__).resolve().parents[3]
_FILES = (
    'Scripts/data_platform/recommendation_identity.py',
    'Scripts/data_platform/services/communication_release.py',
    'Scripts/data_platform/services/decision_communication.py',
    'Scripts/data_platform/services/decision_audit.py',
    'Scripts/data_platform/services/match_read_cards.py',
    'Scripts/data_platform/services/match_reads.py',
    'Scripts/discord_bot/match_read_hubs.py',
    'Scripts/rag_ingest/core/market_service.py',
    'Scripts/web_app/api.py',
    'Scripts/web_app/routers/match_reads.py',
    'Scripts/data_platform/repositories/fixture_schedule.py',
    'Scripts/web_app/static/app.js',
    'Scripts/web_app/static/style.css',
    'Scripts/web_app/static/index.html',
    'Scripts/web_app/static/league-logos/sources.json',
) + tuple(
    f'Scripts/web_app/static/league-logos/{provider_id}.png'
    for provider_id in (39, 140, 135, 78, 61, 40, 203, 88, 94, 144, 2, 3, 848)
)
_HASHES = {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in _FILES}
_MANIFEST = {'version': VERSION, 'identity': VERSION + ':' + digest(_HASHES),
             'files': _HASHES, 'new_selection_policy_active': False}


def communication_release_manifest():
    return deepcopy(_MANIFEST)
