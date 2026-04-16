"""Thin API-Football HTTP client used by the sync pipeline.

Scope is deliberately small: fetch fixtures, fixture statistics, fixture
players, team metadata, and player metadata. Rate limiting is handled by
a simple polite delay between calls; the existing pull scripts use a
similar approach.

Keeping this separate from ``Scripts/rag_ingest/odds_provider.py`` on
purpose — this client targets the *historical* endpoints for sync; the
odds layer targets the live odds endpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)

API_BASE = "https://v3.football.api-sports.io"


@dataclass(frozen=True)
class CompetitionSpec:
    code: str
    name: str
    api_football_id: int
    country: Optional[str]
    competition_type: str  # 'domestic_league' | 'continental_cup'


def _load_competitions_from_registry() -> Dict[str, CompetitionSpec]:
    """Derive the legacy ``COMPETITIONS`` dict from the V3 registry.

    Falls back to a hardcoded baseline if the registry fails to load so
    sync keeps working in degraded environments (e.g. broken YAML). The
    baseline matches the pre-V3 hardcoded values.
    """
    fallback = {
        "EPL":        CompetitionSpec("EPL", "Premier League", 39, "England", "domestic_league"),
        "LaLiga":     CompetitionSpec("LaLiga", "La Liga", 140, "Spain", "domestic_league"),
        "SerieA":     CompetitionSpec("SerieA", "Serie A", 135, "Italy", "domestic_league"),
        "Bundesliga": CompetitionSpec("Bundesliga", "Bundesliga", 78, "Germany", "domestic_league"),
        "Ligue1":     CompetitionSpec("Ligue1", "Ligue 1", 61, "France", "domestic_league"),
        "UCL":        CompetitionSpec("UCL", "UEFA Champions League", 2, None, "continental_cup"),
        "UEL":        CompetitionSpec("UEL", "UEFA Europa League", 3, None, "continental_cup"),
        "UECL":       CompetitionSpec("UECL", "UEFA Europa Conference League", 848, None, "continental_cup"),
    }
    try:
        from ..registry import load_registry
        registry = load_registry()
    except Exception as exc:
        logger.warning("Registry load failed (%s); using baseline competitions.", exc)
        return fallback
    out: Dict[str, CompetitionSpec] = {}
    for entry in registry.entries.values():
        mapping = entry.provider("api_football")
        if mapping is None:
            continue
        out[entry.code] = CompetitionSpec(
            code=entry.code, name=entry.name, api_football_id=mapping.provider_id,
            country=entry.country, competition_type=entry.competition_type,
        )
    if not out:
        return fallback
    return out


COMPETITIONS: Dict[str, CompetitionSpec] = _load_competitions_from_registry()


class ApiFootballClient:
    """Tiny wrapper around ``requests`` with retries + polite pacing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        request_pause_s: float = 0.35,
        session: Optional[requests.Session] = None,
        settings: Optional[Settings] = None,
    ):
        s = settings or SETTINGS
        self.api_key = api_key or s.api_football_key
        if not self.api_key:
            raise RuntimeError(
                "API-Football key not configured. Set API_FOOTBALL_KEY or API-FOOTBALL-KEY in the environment."
            )
        self.session = session or requests.Session()
        self.session.headers.update({"x-apisports-key": self.api_key})
        self.request_pause_s = request_pause_s

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{API_BASE}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params or {}, timeout=30)
                if resp.status_code == 429:
                    backoff = 2 ** attempt
                    logger.warning("API-Football 429 (attempt %s); sleeping %ss", attempt + 1, backoff)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if self.request_pause_s:
                    time.sleep(self.request_pause_s)
                return data
            except Exception as exc:  # pragma: no cover - network path
                last_err = exc
                time.sleep(0.5 + attempt)
        raise RuntimeError(f"API-Football GET {path} failed after retries: {last_err}")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    def fixtures(self, *, league: int, season: int, from_date: Optional[str] = None,
                 to_date: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"league": league, "season": season}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status
        data = self._get("/fixtures", params=params)
        return data.get("response", []) or []

    def fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        data = self._get("/fixtures/statistics", params={"fixture": fixture_id})
        return data.get("response", []) or []

    def fixture_players(self, fixture_id: int) -> List[Dict[str, Any]]:
        data = self._get("/fixtures/players", params={"fixture": fixture_id})
        return data.get("response", []) or []

    def teams_for_league(self, *, league: int, season: int) -> List[Dict[str, Any]]:
        data = self._get("/teams", params={"league": league, "season": season})
        return data.get("response", []) or []


def iter_competitions(codes: Optional[Iterable[str]] = None) -> Iterable[CompetitionSpec]:
    if codes is None:
        yield from COMPETITIONS.values()
        return
    for code in codes:
        spec = COMPETITIONS.get(code)
        if spec is None:
            raise KeyError(f"Unknown competition code: {code}. Valid: {', '.join(COMPETITIONS)}")
        yield spec
