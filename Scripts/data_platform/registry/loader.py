"""Load + validate the competition registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from ..config import SETTINGS
from .models import CompetitionEntry, CompetitionRegistry, ProviderMapping, SeasonRules


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "competitions.yaml"


class RegistryError(Exception):
    pass


def _coerce_season_rules(raw: Optional[Mapping[str, Any]]) -> SeasonRules:
    if not raw:
        return SeasonRules()
    return SeasonRules(
        style=str(raw.get("style", "europe_crossyear")),
        start_month=int(raw.get("start_month", 7)),
        end_month=int(raw.get("end_month", 6)),
        first_matchday_offset_days=int(raw.get("first_matchday_offset_days", 7)),
    )


def _coerce_provider(name: str, raw: Mapping[str, Any]) -> ProviderMapping:
    if "provider_id" not in raw:
        raise RegistryError(f"provider {name!r} missing 'provider_id'")
    return ProviderMapping(
        provider=name,
        provider_id=int(raw["provider_id"]),
        team_aliases=dict(raw.get("team_aliases") or {}),
        player_aliases=dict(raw.get("player_aliases") or {}),
        endpoint_flags=dict(raw.get("endpoint_flags") or {}),
        notes=raw.get("notes"),
    )


def _coerce_entry(code: str, raw: Mapping[str, Any]) -> CompetitionEntry:
    providers_raw = raw.get("providers") or {}
    if not providers_raw:
        raise RegistryError(f"competition {code!r} has no providers")
    providers: Dict[str, ProviderMapping] = {
        name: _coerce_provider(name, cfg) for name, cfg in providers_raw.items()
    }
    default_markets = frozenset(raw.get("default_markets") or
                                ["goals", "corners", "cards", "sot", "btts", "moneyline", "spreads"])
    kb_entities = frozenset(raw.get("kb_entity_types") or ["team", "player", "fixture"])
    return CompetitionEntry(
        code=code,
        name=str(raw.get("name") or code),
        country=raw.get("country"),
        competition_type=str(raw.get("competition_type") or "domestic_league"),
        tier=int(raw.get("tier", 1)),
        providers=providers,
        season_rules=_coerce_season_rules(raw.get("season_rules")),
        default_markets=default_markets,
        update_cadence_minutes=int(raw.get("update_cadence_minutes", 30)),
        kb_entity_types=kb_entities,
        feature_module=raw.get("feature_module"),
        enabled=bool(raw.get("enabled", True)),
        labels=tuple(raw.get("labels") or ()),
    )


def load_registry(path: Optional[Path] = None) -> CompetitionRegistry:
    """Parse + validate the YAML registry. Raises ``RegistryError`` on problems."""
    target = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not target.exists():
        raise RegistryError(f"Competition registry not found: {target}")
    raw = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise RegistryError(f"Registry must be a mapping at the top level; got {type(raw).__name__}")
    entries_raw = raw.get("competitions") or {}
    if not entries_raw:
        raise RegistryError("Registry has no competitions.")
    entries: Dict[str, CompetitionEntry] = {}
    for code, cfg in entries_raw.items():
        if not isinstance(cfg, Mapping):
            raise RegistryError(f"Competition {code!r} is not a mapping")
        entries[str(code)] = _coerce_entry(str(code), cfg)

    # Uniqueness check on provider_id per provider — catches copy/paste mistakes
    seen: Dict[str, Dict[int, str]] = {}
    for entry in entries.values():
        for provider_name, mapping in entry.providers.items():
            by_provider = seen.setdefault(provider_name, {})
            if mapping.provider_id in by_provider:
                raise RegistryError(
                    f"Duplicate provider_id {mapping.provider_id} on {provider_name}: "
                    f"{entry.code} vs {by_provider[mapping.provider_id]}"
                )
            by_provider[mapping.provider_id] = entry.code

    return CompetitionRegistry(entries=entries, version=str(raw.get("version", "1")))
