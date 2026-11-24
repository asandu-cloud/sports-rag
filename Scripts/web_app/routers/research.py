"""Read-only research profiles from the canonical database and built peer ranks."""
from collections import OrderedDict
from threading import Lock
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query

from data_platform.services.refresh_coordination import RefreshBusy, read_gate
from data_platform.services.research_home import ResearchHomeService
from data_platform.services.research_profiles import ResearchProfileService

router = APIRouter(prefix="/api/research", tags=["research"])
SeasonYear = Annotated[int | None, Query(ge=1900, le=2100)]
CompetitionCode = Annotated[str | None, Query(min_length=1, max_length=32)]
SubjectID = Annotated[int, Path(gt=0)]
RefereeKey = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[\w-]+$")]
Limit = Annotated[int, Query(ge=1, le=10)]
BUSY = "Research is updating; try again shortly."
_gate = read_gate  # the new routes never wait for the refresh, nor make it wait (D82, D110)
_SHOWN_AT_KICKOFF = OrderedDict()  # event id -> the board's before-kickoff entry, or None (D122)
SHOWN_AT_KICKOFF_KEPT = 5000  # about two seasons of the public leagues' matches
_SHOWN_LOCK = Lock()


def _kickoff_picks(event_ids):
    """Each finished match's lists as the board showed them at kickoff (D70), by event id. A match with no read kept
    is left out. A started match's answer never changes, so each is read once, inside the refresh gate
    (``RefreshBusy`` while the refresh writes), and remembered."""
    from web_app.routers import match_reads
    with _SHOWN_LOCK:
        todo = [event_id for event_id in event_ids if event_id not in _SHOWN_AT_KICKOFF]
    found = {}
    if todo:
        service, now = match_reads._get_match_read_service(), match_reads._utc_now()
        with _gate():
            for event_id in todo:
                reads = service.list_delivered_for_fixture(event_id, surface="website", limit=100)
                found[event_id] = next(iter(match_reads._before_kickoff(reads, now=now)), None)
    with _SHOWN_LOCK:  # the website answers on several threads
        for event_id in event_ids:
            if event_id in found:
                _SHOWN_AT_KICKOFF[event_id] = found[event_id]
            elif event_id in _SHOWN_AT_KICKOFF:
                found[event_id] = _SHOWN_AT_KICKOFF[event_id]
            else:
                continue
            _SHOWN_AT_KICKOFF.move_to_end(event_id)
        while len(_SHOWN_AT_KICKOFF) > SHOWN_AT_KICKOFF_KEPT:
            _SHOWN_AT_KICKOFF.popitem(last=False)
    return {event_id: entry for event_id, entry in found.items() if entry is not None}


def _get_research_service():
    from data_platform.services import before_kickoff
    # A team's match log carries what Spix showed before kickoff (D122) while the board shows it (D70).
    return ResearchProfileService(kickoff_picks=_kickoff_picks if before_kickoff.enabled() else None)


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _gated(method, *args, **kwargs):
    """``_call`` inside the refresh gate: 503 at once while the daily refresh writes, nothing read."""
    try:
        with _gate():
            return _call(method, *args, **kwargs)
    except RefreshBusy as exc:
        raise HTTPException(status_code=503, detail=BUSY, headers={"Retry-After": "600"}) from exc


def _load_read(event_id):
    """The card the match page shows and the read's canonical results, or None while the board shows no read for the
    fixture: none yet, or one it withholds (stale, or kicked off)."""
    from data_platform.services.match_read_cards import build_match_read_card
    from web_app.routers import match_reads
    try:
        _, read, _ = match_reads._actionable_reads(event_id)
    except HTTPException as exc:
        if exc.status_code in (404, 410):
            return None
        raise
    return build_match_read_card(read), (read.get("data") or {}).get("canonical_results")


def _get_upcoming_service():
    from data_platform.config import load_match_read_worker_settings
    from data_platform.repositories.research import ResearchRepository
    from data_platform.services.research_upcoming import ResearchUpcomingService
    from web_app.routers import match_reads
    return ResearchUpcomingService(ResearchRepository(), load_read=_load_read, now=match_reads._utc_now,
                                   outlook_hours=load_match_read_worker_settings().outlook_hours)


def _get_match_service():
    from data_platform.config import load_match_read_worker_settings
    from data_platform.repositories.research import ResearchRepository
    from data_platform.services.research_match import ResearchMatchService
    from web_app.routers import match_reads
    return ResearchMatchService(ResearchRepository(), load_read=_load_read, now=match_reads._utc_now,
                                outlook_hours=load_match_read_worker_settings().outlook_hours)


def _get_home_service():
    return ResearchHomeService()


@router.get("/home")
def home(type: Literal["players", "teams", "referees"],
         league: Annotated[str | None, Query(min_length=1, max_length=32, pattern=r"^\w+$")] = None):
    return _call(_get_home_service().home, type, league=league)


@router.get("/search")
def search(type: Literal["players", "teams", "referees"],
           q: Annotated[str, Query(min_length=1, max_length=100)],
           competition: CompetitionCode = None,
           limit: Annotated[int, Query(ge=1, le=100)] = 20):
    q = q.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Search query must not be blank")
    return _call(_get_research_service().search, type, q, competition=competition, limit=limit)


@router.get("/players/{player_id}")
def player(player_id: SubjectID, competition: CompetitionCode = None, season: SeasonYear = None):
    return _call(_get_research_service().profile, "player", player_id, competition=competition, season=season)


@router.get("/teams/{team_id}")
def team(team_id: SubjectID, competition: CompetitionCode = None, season: SeasonYear = None):
    return _call(_get_research_service().profile, "team", team_id, competition=competition, season=season)


@router.get("/referees/{referee_key}")
def referee(referee_key: RefereeKey, season: SeasonYear = None):
    return _call(_get_research_service().profile, "referee", referee_key, season=season)


@router.get("/teams/{team_id}/upcoming")
def team_upcoming(team_id: SubjectID, limit: Limit = 3):
    """The team's next matches in every competition, each with its Match Read's state (D110, B1)."""
    return _gated(lambda: _get_upcoming_service().team(team_id, limit=limit))


@router.get("/referees/{referee_key}/upcoming")
def referee_upcoming(referee_key: RefereeKey, limit: Limit = 1):
    """The referee's next appointments, once named (D110, B1)."""
    return _gated(lambda: _get_upcoming_service().referee(referee_key, limit=limit))


_LINEUPS = None


def _get_lineups():
    """The website's one in-memory line-ups store (D135)."""
    global _LINEUPS
    if _LINEUPS is None:
        from data_platform.services.lineups import Lineups
        _LINEUPS = Lineups()
    return _LINEUPS


def _lineups(event_id):
    """The match's line-ups block, or None while ``SPIX_LINEUPS`` is off (D135). The provider is asked between two
    short gated reads, never inside the gate, so a slow answer cannot hold up the refresh."""
    from data_platform.services import lineups
    from web_app.routers import match_reads
    if not lineups.enabled():
        return None
    service = _get_match_service()
    facts = _gated(service.lineup_facts, event_id)
    known = _get_lineups().lookup(event_id, facts["kickoff"], match_reads._utc_now())
    return _gated(service.lineup_block, facts, known)


@router.get("/matches/{event_id}")
def match(event_id: Annotated[str, Path(pattern=r"^\d{1,12}$")]):
    """One match for the research view: the fixture, Spix's read and its lines, and the past numbers (D110, B2)."""
    body = _gated(lambda: _get_match_service().match(event_id))
    body["lineups"] = _lineups(event_id)
    return body


@router.get("/matches/{event_id}/lineups")
def match_lineups(event_id: Annotated[str, Path(pattern=r"^\d{1,12}$")]):
    """The match page's line-ups alone (D135): null while the switch is off."""
    return {"schema_version": "research-lineups.v1", "event_id": event_id, "lineups": _lineups(event_id)}
