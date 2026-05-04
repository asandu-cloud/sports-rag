"""
Structured models for deterministic Discord parlay workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParlayLegRef:
    event_id: str
    market_key: str
    outcome: str
    point: Optional[float]
    league: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParlayLegRef":
        return cls(
            event_id=str(data.get("event_id") or ""),
            market_key=str(data.get("market_key") or ""),
            outcome=str(data.get("outcome") or ""),
            point=_safe_float(data.get("point")),
            league=str(data.get("league") or ""),
        )


@dataclass
class ParlayBuildRequest:
    league_scope: str
    leagues: List[str]
    default_league: str
    display_league: str
    target_date: str
    date_phrase: str
    legs_requested: int
    target_odds: Optional[float]
    target_mode: str = "max"
    flexible_leg_count: bool = False
    min_legs_requested: int = 1
    max_legs_requested: int = 8
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    market_filter: str = "mixed"
    match_filter: Optional[str] = None
    same_game_only: bool = False
    require_unique_events: bool = True
    min_model_prob: Optional[float] = None
    allowed_confidences: List[str] = field(default_factory=list)
    allowed_groups: List[str] = field(default_factory=list)
    soft_prefer_groups: List[str] = field(default_factory=list)
    required_group_counts: Dict[str, int] = field(default_factory=dict)
    constraint_query: Optional[str] = None
    source_command: str = "parlay"
    risk_profile: str = "standard"
    locked_legs: List[ParlayLegRef] = field(default_factory=list)
    excluded_legs: List[ParlayLegRef] = field(default_factory=list)
    excluded_event_groups: List[Tuple[str, str]] = field(default_factory=list)  # [(event_id, market_group), ...]
    allowed_event_ids: List[str] = field(default_factory=list)
    prefer_new_fixture: bool = False
    action_type: str = "initial"
    parent_session_id: Optional[int] = None
    discord_user_id: Optional[int] = None
    guild_id: Optional[int] = None
    channel_id: Optional[int] = None
    thread_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["locked_legs"] = [leg.to_dict() for leg in self.locked_legs]
        data["excluded_legs"] = [leg.to_dict() for leg in self.excluded_legs]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParlayBuildRequest":
        payload = dict(data or {})
        payload["leagues"] = [str(x) for x in payload.get("leagues") or []]
        payload["locked_legs"] = [ParlayLegRef.from_dict(x) for x in payload.get("locked_legs") or []]
        payload["excluded_legs"] = [ParlayLegRef.from_dict(x) for x in payload.get("excluded_legs") or []]
        payload["allowed_event_ids"] = [str(x) for x in payload.get("allowed_event_ids") or []]
        payload["target_odds"] = _safe_float(payload.get("target_odds"))
        payload["target_min"] = _safe_float(payload.get("target_min"))
        payload["target_max"] = _safe_float(payload.get("target_max"))
        payload["min_model_prob"] = _safe_float(payload.get("min_model_prob"))
        payload["allowed_confidences"] = [str(x) for x in payload.get("allowed_confidences") or []]
        payload["allowed_groups"] = [str(x) for x in payload.get("allowed_groups") or []]
        payload["soft_prefer_groups"] = [str(x) for x in payload.get("soft_prefer_groups") or []]
        payload["required_group_counts"] = {
            str(k): int(v) for k, v in (payload.get("required_group_counts") or {}).items()
        }
        return cls(**payload)


@dataclass
class ParlayLegResult:
    event_id: str
    league: str
    fixture: str
    home_team: str
    away_team: str
    market_key: str
    market_group: str
    outcome: str
    point: Optional[float]
    pick_display: str
    odds: float
    bookmaker: Optional[str]
    quality_score: float
    confidence: Optional[str]
    model_prob: Optional[float]
    implied_prob: Optional[float]
    value_edge: Optional[float]
    projected_total: Optional[float]
    side_agrees: bool = True
    warning: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def ref(self) -> ParlayLegRef:
        return ParlayLegRef(
            event_id=self.event_id,
            market_key=self.market_key,
            outcome=self.outcome,
            point=self.point,
            league=self.league,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParlayLegResult":
        payload = dict(data or {})
        payload["point"] = _safe_float(payload.get("point"))
        payload["odds"] = float(payload.get("odds") or 0.0)
        payload["quality_score"] = float(payload.get("quality_score") or 0.0)
        payload["model_prob"] = _safe_float(payload.get("model_prob"))
        payload["implied_prob"] = _safe_float(payload.get("implied_prob"))
        payload["value_edge"] = _safe_float(payload.get("value_edge"))
        payload["projected_total"] = _safe_float(payload.get("projected_total"))
        payload["reasons"] = [str(x) for x in payload.get("reasons") or []]
        return cls(**payload)


@dataclass
class ParlayBuildResult:
    request: ParlayBuildRequest
    selected_legs: List[ParlayLegResult]
    combined_odds: float
    notes: List[str] = field(default_factory=list)
    cross_warnings: List[str] = field(default_factory=list)
    confidence_breakdown: str = ""
    league_distribution: str = ""
    deterministic_summary: str = ""
    llm_summary: Optional[str] = None
    session_id: Optional[int] = None

    @property
    def summary_text(self) -> str:
        return self.llm_summary or self.deterministic_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "selected_legs": [leg.to_dict() for leg in self.selected_legs],
            "combined_odds": self.combined_odds,
            "notes": list(self.notes),
            "cross_warnings": list(self.cross_warnings),
            "confidence_breakdown": self.confidence_breakdown,
            "league_distribution": self.league_distribution,
            "deterministic_summary": self.deterministic_summary,
            "llm_summary": self.llm_summary,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParlayBuildResult":
        payload = dict(data or {})
        return cls(
            request=ParlayBuildRequest.from_dict(payload.get("request") or {}),
            selected_legs=[ParlayLegResult.from_dict(x) for x in payload.get("selected_legs") or []],
            combined_odds=float(payload.get("combined_odds") or 0.0),
            notes=[str(x) for x in payload.get("notes") or []],
            cross_warnings=[str(x) for x in payload.get("cross_warnings") or []],
            confidence_breakdown=str(payload.get("confidence_breakdown") or ""),
            league_distribution=str(payload.get("league_distribution") or ""),
            deterministic_summary=str(payload.get("deterministic_summary") or ""),
            llm_summary=payload.get("llm_summary"),
            session_id=payload.get("session_id"),
        )


@dataclass
class ParlaySessionRecord:
    id: int
    parent_session_id: Optional[int]
    discord_user_id: Optional[int]
    guild_id: Optional[int]
    channel_id: Optional[int]
    thread_id: Optional[int]
    message_id: Optional[int]
    source_command: str
    action_type: str
    created_at: str
    request: ParlayBuildRequest
    result: ParlayBuildResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_session_id": self.parent_session_id,
            "discord_user_id": self.discord_user_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "source_command": self.source_command,
            "action_type": self.action_type,
            "created_at": self.created_at,
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ParlaySessionRecord":
        return cls(
            id=int(row["id"]),
            parent_session_id=row["parent_session_id"],
            discord_user_id=row["discord_user_id"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            thread_id=row["thread_id"],
            message_id=row["message_id"],
            source_command=str(row["source_command"] or ""),
            action_type=str(row["action_type"] or ""),
            created_at=str(row["created_at"] or ""),
            request=ParlayBuildRequest.from_dict(row["request_json"]),
            result=ParlayBuildResult.from_dict(row["result_json"]),
        )


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
