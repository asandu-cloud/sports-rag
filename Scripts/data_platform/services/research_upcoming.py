"""A team's or a referee's next matches for the Research Area (tracker D110, the overhaul's B1, D107).

Unplayed matches in every competition the database holds, in kickoff order. Each carries the state of its Match
Read and, once the read is out, Spix's expected goals, corners and cards from the read's own figures
(``match_read_metrics.match_metrics``, the figures the match page shows); nothing is worked out afresh. A referee is
named with the key his research profile uses, also under a spelling the alias table has not seen yet: the key is
computed from the name the way the rank build computes it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .match_read_metrics import match_metrics
from .research_profiles import _normalize_ref_name, iso, raw_referee_key

SCHEMA_VERSION = "research-upcoming.v1"
EXPECTED = ("goals", "corners", "cards")

Read = Tuple[Mapping[str, Any], Any]  # the match page's card and the read's canonical results


def one_decimal(value) -> Optional[float]:
    return round(value, 1) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _figure(metrics: Mapping[str, Any], label: str) -> Optional[float]:
    return one_decimal((metrics.get(label) or {}).get("projected"))


def _time(value) -> Optional[str]:
    return iso(value) if isinstance(value, datetime) else value


def read_summary(event_id: str, kickoff: datetime, *, load_read: Callable[[str], Optional[Read]], now: datetime,
                 outlook_hours: int) -> dict:
    """The ``read`` of one match: ``ready`` or ``preliminary`` with Spix's expected figures while the board shows a
    read, else ``waiting`` with the time the read can first come out (null once that has passed)."""
    read = load_read(event_id)
    if read is None:
        opens = kickoff - timedelta(hours=outlook_hours)  # the Match Read worker's outlook window
        return {"status": "waiting", "read_at": None, "opens_at": iso(opens) if opens > now else None,
                "expected": None}
    card, canonical_results = read
    results = canonical_results if isinstance(canonical_results, Sequence) and not isinstance(
        canonical_results, (str, bytes)) else []
    metrics = match_metrics([r for r in results if isinstance(r, Mapping)])
    return {"status": "preliminary" if (card.get("timing") or {}).get("is_preliminary") else "ready",
            "read_at": _time(card.get("created_at")), "opens_at": None,
            "expected": {label: _figure(metrics, label) for label in EXPECTED}}


def referee_refs(raw_names: Iterable[str], alias_rows: Sequence[Mapping[str, Any]]) -> Dict[str, dict]:
    """``{raw name: {"name", "referee_key"}}``: the alias table's key and name, else the key the build would give."""
    table = {row["raw_name"]: row["referee_key"] for row in alias_rows}
    names: Dict[str, str] = {}
    for row in sorted(alias_rows, key=lambda r: r["raw_name"]):
        names.setdefault(row["referee_key"], row["name"])
    out = {}
    for raw in sorted({n for n in raw_names if n}):
        key = table.get(raw) or raw_referee_key(raw)
        if key is not None:
            out[raw] = {"name": names.get(key) or _normalize_ref_name(raw).title(), "referee_key": key}
    return out


def refs_for(repo, session, raw_names) -> Dict[str, dict]:
    """``referee_refs`` for raw names, with the alias rows of those names and of their computed keys."""
    raw_names = sorted({n for n in raw_names if n})
    keys = sorted({key for key in map(raw_referee_key, raw_names) if key})
    return referee_refs(raw_names, repo.alias_rows(session, raw_names=raw_names, keys=keys))


def kickoff_utc(row) -> datetime:
    kickoff = row["kickoff_utc"]
    return kickoff.replace(tzinfo=kickoff.tzinfo or timezone.utc)


def match_entry(row, refs, *, load_read, now, outlook_hours, team_id=None) -> dict:
    """One match of the schema from a ``ResearchRepository._schedule_query`` row; ``venue`` is ``team_id``'s side."""
    kickoff, event_id = kickoff_utc(row), str(row["api_football_id"])
    side = {k: {"id": row[f"{k}_team_id"], "name": row[f"{k}_name"], "short_code": row[f"{k}_short_code"],
                "logo_url": row[f"{k}_logo_url"]} for k in ("home", "away")}
    return {"event_id": event_id, "fixture_id": row["fixture_id"], "kickoff": iso(kickoff),
            "competition": {"code": row["competition_code"], "name": row["competition_name"]},
            **side, "venue": None if team_id is None else "home" if row["home_team_id"] == team_id else "away",
            "referee": refs.get(row["referee"]),
            "read": read_summary(event_id, kickoff, load_read=load_read, now=now, outlook_hours=outlook_hours)}


class ResearchUpcomingService:
    def __init__(self, repo, *, load_read: Callable[[str], Optional[Read]], now: Callable[[], datetime],
                 outlook_hours: int):
        self.repo, self.load_read, self.now, self.outlook_hours = repo, load_read, now, outlook_hours

    def _refs(self, session, raw_names) -> Dict[str, dict]:
        return refs_for(self.repo, session, raw_names)

    def _match(self, row, refs, now, team_id=None) -> dict:
        return match_entry(row, refs, load_read=self.load_read, now=now, outlook_hours=self.outlook_hours,
                           team_id=team_id)

    def team(self, team_id: int, *, limit: int) -> dict:
        now = self.now()
        with self.repo.session_factory() as session:
            subject = self.repo.subject(session, "team", team_id)
            if not subject:
                raise LookupError("Unknown team")
            rows = self.repo.upcoming(session, now=now, limit=limit, team_id=team_id)
            refs = self._refs(session, [row["referee"] for row in rows])
        return {"schema_version": SCHEMA_VERSION, "subject": {"type": "team", "id": team_id, "name": subject["name"]},
                "matches": [self._match(row, refs, now, team_id) for row in rows]}

    def referee(self, referee_key: str, *, limit: int) -> dict:
        now = self.now()
        with self.repo.session_factory() as session:
            known = self.repo.alias_rows(session, keys=[referee_key])
            refs = self._refs(session, self.repo.upcoming_referee_names(session, now=now))
            raws = sorted(raw for raw, ref in refs.items() if ref["referee_key"] == referee_key)
            if not known and not raws:
                raise LookupError("Unknown referee")
            rows = self.repo.upcoming(session, now=now, limit=limit, referees=raws) if raws else []
        name = min(known, key=lambda r: r["raw_name"])["name"] if known else refs[raws[0]]["name"]
        return {"schema_version": SCHEMA_VERSION,
                "subject": {"type": "referee", "referee_key": referee_key, "name": name},
                "matches": [self._match(row, refs, now) for row in rows]}
