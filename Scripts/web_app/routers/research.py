"""Read-only research profiles from the canonical database and built peer ranks."""
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query

from data_platform.services.research_home import ResearchHomeService
from data_platform.services.research_profiles import ResearchProfileService

router = APIRouter(prefix="/api/research", tags=["research"])
SeasonYear = Annotated[int | None, Query(ge=1900, le=2100)]
CompetitionCode = Annotated[str | None, Query(min_length=1, max_length=32)]
SubjectID = Annotated[int, Path(gt=0)]


def _get_research_service():
    return ResearchProfileService()


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
def referee(referee_key: Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[\w-]+$")],
            season: SeasonYear = None):
    return _call(_get_research_service().profile, "referee", referee_key, season=season)
