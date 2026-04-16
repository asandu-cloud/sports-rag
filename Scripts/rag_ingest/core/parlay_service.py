"""
Structured deterministic parlay workflow for Discord surfaces.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import rag_cli_v2 as rag
except ImportError:
    from Scripts.rag_ingest import rag_cli_v2 as rag  # type: ignore[import]

try:
    from .parlay_evidence import leg_evidence, leg_standalone_confidence
    from .parlay_models import (
        ParlayBuildRequest,
        ParlayBuildResult,
        ParlayLegRef,
        ParlayLegResult,
        ParlaySessionRecord,
    )
except ImportError:
    try:
        from core.parlay_evidence import leg_evidence, leg_standalone_confidence
        from core.parlay_models import (
            ParlayBuildRequest,
            ParlayBuildResult,
            ParlayLegRef,
            ParlayLegResult,
            ParlaySessionRecord,
        )
    except ImportError:
        from Scripts.rag_ingest.core.parlay_evidence import leg_evidence, leg_standalone_confidence  # type: ignore[import]
        from Scripts.rag_ingest.core.parlay_models import (  # type: ignore[import]
            ParlayBuildRequest,
            ParlayBuildResult,
            ParlayLegRef,
            ParlayLegResult,
            ParlaySessionRecord,
        )

try:
    from prob_models import implied_prob, value_edge
except ImportError:
    from Scripts.rag_ingest.prob_models import implied_prob, value_edge  # type: ignore[import]

log = logging.getLogger("parlay_service")

TOP5_LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "Index" / "predictions.db"

# ---------------------------------------------------------------------------
# V2 platform shim — lazy import so SQLAlchemy isn't a hard runtime dep.
# ---------------------------------------------------------------------------

import sys as _sys

_SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_ROOT))


def _platform_parlay_store():
    """Return a PlatformParlayStore when PLATFORM_ENABLED=1, else None."""
    try:
        from data_platform.compat import PlatformParlayStore
        from data_platform.compat.parlay_shim import is_platform_enabled
    except Exception:
        return None
    if not is_platform_enabled():
        return None
    try:
        return PlatformParlayStore()
    except Exception:  # pragma: no cover - DB unreachable
        log.exception("Platform parlay store unavailable; falling back to SQLite")
        return None

MARKET_FILTER_TO_GROUP = {
    "mixed": None,
    "totals": "totals",
    "corners": "corners",
    "cards": "cards",
    "sot": "sot",
    "btts": "btts",
    "moneyline": "moneyline",
}

MARKET_FILTER_TO_ENRICH = {
    "mixed": {"corners", "cards", "sot", "btts"},
    "totals": set(),
    "corners": {"corners"},
    "cards": {"cards"},
    "sot": {"sot"},
    "btts": {"btts"},
    "moneyline": set(),
}

CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


class ParlayBuildError(RuntimeError):
    def __init__(self, message: str, notes: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.message = message
        self.notes = list(notes or [])


class ParlaySessionStore:
    """Session store.

    When ``PLATFORM_ENABLED=1`` *and* no custom ``db_path`` is supplied the
    store routes every call through the canonical PostgreSQL repository
    via :class:`PlatformParlayStore`. Otherwise it keeps the original
    SQLite implementation. Tests pinning a temp file via ``db_path=``
    continue to use SQLite.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._platform = _platform_parlay_store() if Path(db_path) == Path(DEFAULT_DB_PATH) else None

    def _get_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parlay_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_session_id INTEGER,
                discord_user_id INTEGER,
                guild_id INTEGER,
                channel_id INTEGER,
                thread_id INTEGER,
                message_id INTEGER,
                source_command TEXT NOT NULL,
                action_type TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_parlay_sessions_user ON parlay_sessions(discord_user_id);",
            "CREATE INDEX IF NOT EXISTS idx_parlay_sessions_thread ON parlay_sessions(thread_id);",
            "CREATE INDEX IF NOT EXISTS idx_parlay_sessions_parent ON parlay_sessions(parent_session_id);",
            "CREATE INDEX IF NOT EXISTS idx_parlay_sessions_message ON parlay_sessions(message_id);",
        ):
            conn.execute(sql)
        conn.commit()
        return conn

    def create_session(self, request: ParlayBuildRequest, result: ParlayBuildResult) -> int:
        if self._platform is not None:
            return self._platform.create_session(request, result)
        conn = self._get_db()
        cur = conn.execute(
            """
            INSERT INTO parlay_sessions (
                parent_session_id, discord_user_id, guild_id, channel_id, thread_id,
                source_command, action_type, request_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.parent_session_id,
                request.discord_user_id,
                request.guild_id,
                request.channel_id,
                request.thread_id,
                request.source_command,
                request.action_type,
                json.dumps(request.to_dict(), ensure_ascii=True),
                json.dumps(result.to_dict(), ensure_ascii=True),
            ),
        )
        session_id = int(cur.lastrowid)
        result.session_id = session_id
        conn.execute(
            "UPDATE parlay_sessions SET result_json = ? WHERE id = ?",
            (json.dumps(result.to_dict(), ensure_ascii=True), session_id),
        )
        conn.commit()
        conn.close()
        return session_id

    def update_message_context(
        self,
        session_id: int,
        *,
        message_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        thread_id: Optional[int] = None,
    ) -> None:
        if self._platform is not None:
            return self._platform.update_message_context(
                session_id,
                message_id=message_id,
                channel_id=channel_id,
                thread_id=thread_id,
            )
        conn = self._get_db()
        conn.execute(
            """
            UPDATE parlay_sessions
            SET message_id = COALESCE(?, message_id),
                channel_id = COALESCE(?, channel_id),
                thread_id = COALESCE(?, thread_id)
            WHERE id = ?
            """,
            (message_id, channel_id, thread_id, session_id),
        )
        conn.commit()
        conn.close()

    def get_session(self, session_id: int) -> Optional[ParlaySessionRecord]:
        if self._platform is not None:
            return self._platform.get_session(session_id)
        conn = self._get_db()
        row = conn.execute("SELECT * FROM parlay_sessions WHERE id = ?", (session_id,)).fetchone()
        conn.close()
        return self._row_to_record(row) if row else None

    def list_persistent_sessions(self, limit: int = 250) -> List[ParlaySessionRecord]:
        if self._platform is not None:
            return self._platform.list_persistent_sessions(limit=limit)
        conn = self._get_db()
        rows = conn.execute(
            """
            SELECT *
            FROM parlay_sessions
            WHERE message_id IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows if row]

    def _row_to_record(self, row: sqlite3.Row) -> ParlaySessionRecord:
        payload = dict(row)
        payload["request_json"] = json.loads(payload["request_json"])
        payload["result_json"] = json.loads(payload["result_json"])
        return ParlaySessionRecord.from_row(payload)


def build_parlay_request(
    *,
    league_value: str,
    target_date: date,
    legs: int,
    odds: Optional[float],
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
    market: Optional[str] = None,
    match: Optional[str] = None,
    source_command: str = "parlay",
    target_mode: str = "max",
    leagues_override: Optional[Sequence[str]] = None,
    display_league: Optional[str] = None,
    min_model_prob: Optional[float] = None,
    allowed_confidences: Optional[Sequence[str]] = None,
    allowed_groups: Optional[Sequence[str]] = None,
    soft_prefer_groups: Optional[Sequence[str]] = None,
    required_group_counts: Optional[Dict[str, int]] = None,
    constraint_query: Optional[str] = None,
    risk_profile: str = "standard",
    discord_user_id: Optional[int] = None,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    thread_id: Optional[int] = None,
) -> ParlayBuildRequest:
    is_cross_league = league_value == "cross-league"
    leagues = list(leagues_override) if leagues_override else (TOP5_LEAGUES if is_cross_league else [league_value])
    default_league = leagues[0]
    display_name = display_league or ("Cross-League" if is_cross_league or len(leagues) > 1 else league_value)
    same_game_only = bool(match)
    # General parlays should optimize for the best available legs on the slate,
    # even if that means taking multiple non-contradictory legs from one fixture.
    require_unique_events = False
    return ParlayBuildRequest(
        league_scope=league_value,
        leagues=leagues,
        default_league=default_league,
        display_league=display_name,
        target_date=target_date.isoformat(),
        date_phrase=humanize_date(target_date),
        legs_requested=max(1, int(legs)),
        target_odds=float(odds) if odds is not None else None,
        target_mode=target_mode,
        target_min=float(target_min) if target_min is not None else None,
        target_max=float(target_max) if target_max is not None else None,
        market_filter=(market or "mixed"),
        match_filter=match,
        same_game_only=same_game_only,
        require_unique_events=require_unique_events,
        min_model_prob=min_model_prob,
        allowed_confidences=[str(x) for x in (allowed_confidences or [])],
        allowed_groups=[str(x) for x in (allowed_groups or [])],
        soft_prefer_groups=[str(x) for x in (soft_prefer_groups or [])],
        required_group_counts={str(k): int(v) for k, v in (required_group_counts or {}).items()},
        constraint_query=constraint_query,
        source_command=source_command,
        risk_profile=risk_profile,
        discord_user_id=discord_user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
    )


def derive_followup_request(record: ParlaySessionRecord, action: str) -> ParlayBuildRequest:
    prior = record.result
    base = prior.request
    selected_refs = [leg.ref for leg in prior.selected_legs]
    action = action.strip().lower()

    if action == "add_leg":
        return replace(
            base,
            legs_requested=base.legs_requested + 1,
            locked_legs=selected_refs,
            excluded_legs=[],
            allowed_event_ids=[],
            prefer_new_fixture=False,
            action_type="add_leg",
            risk_profile=base.risk_profile,
            parent_session_id=record.id,
        )

    if action == "swap_weakest":
        weakest = min(prior.selected_legs, key=lambda leg: (leg.quality_score, leg.model_prob or 0.0))
        locked = [leg.ref for leg in prior.selected_legs if leg.event_id != weakest.event_id or leg.market_key != weakest.market_key or leg.outcome != weakest.outcome or _point_key(leg.point) != _point_key(weakest.point)]
        return replace(
            base,
            locked_legs=locked,
            excluded_legs=[weakest.ref],
            allowed_event_ids=[],
            prefer_new_fixture=False,
            action_type="swap_weakest",
            parent_session_id=record.id,
        )

    if action in {"safer", "riskier"}:
        factor = 0.75 if action == "safer" else 1.35
        target = prior.combined_odds * factor
        if base.target_odds:
            target = min(target, base.target_odds) if action == "safer" else max(target, base.target_odds)
        if action == "safer":
            target = max(1.6, target)
        allowed_event_ids = sorted({leg.event_id for leg in prior.selected_legs})
        return replace(
            base,
            target_odds=target,
            target_mode="around",
            locked_legs=[],
            excluded_legs=[],
            allowed_event_ids=allowed_event_ids,
            prefer_new_fixture=False,
            action_type=action,
            risk_profile=action,
            parent_session_id=record.id,
        )

    raise ValueError(f"Unsupported follow-up action: {action}")


def build_and_store_parlay(
    request: ParlayBuildRequest,
    store: Optional[ParlaySessionStore] = None,
) -> ParlayBuildResult:
    result = build_parlay(request)
    if store is None:
        return result
    session_id = store.create_session(request, result)
    result.session_id = session_id
    return result


def build_parlay(request: ParlayBuildRequest) -> ParlayBuildResult:
    notes: List[str] = []
    events, fetch_notes = _fetch_events_for_request(request)
    notes.extend(fetch_notes)

    events = _filter_events_for_request(events, request, notes)
    if not events:
        raise ParlayBuildError("No fixtures found for the requested league/date combination.", notes)

    enriched_events, enrich_notes = _enrich_events(events, request)
    notes.extend(enrich_notes)

    candidates = rag.build_candidates(enriched_events)
    if not candidates:
        raise ParlayBuildError("No valid betting candidates were available from the current odds feed.", notes)

    constraint = _build_constraint_spec(request)
    locked_legs = _resolve_locked_legs(candidates, request.locked_legs)
    filtered_candidates = _filter_candidate_pool(candidates, request, constraint, locked_legs)

    remaining_needed = max(0, request.legs_requested - len(locked_legs))
    adjusted_constraint = _adjust_constraint_for_locked(constraint, locked_legs, remaining_needed)

    selected_new: List[rag.CandidateLeg] = []
    if remaining_needed > 0:
        feasible, warnings = rag.check_feasibility(filtered_candidates, remaining_needed, adjusted_constraint)
        notes.extend(warnings)
        if not feasible:
            raise ParlayBuildError("Could not construct a parlay with the current constraints.", notes)

        selected_new, selection_notes = rag.select_parlay(filtered_candidates, remaining_needed, adjusted_constraint)
        notes.extend(selection_notes)
        if not selected_new:
            raise ParlayBuildError("Could not construct a valid parlay from the available markets.", notes)

    selected = locked_legs + selected_new
    if not selected:
        raise ParlayBuildError("No parlay legs were selected.", notes)
    if not rag.combo_valid(selected, constraint.require_unique_events):
        raise ParlayBuildError("The selected legs conflict with each other after follow-up adjustments.", notes)

    leg_results, cross_warnings = _build_leg_results(selected, request)
    result = ParlayBuildResult(
        request=request,
        selected_legs=leg_results,
        combined_odds=rag.combined_odds(selected),
        notes=notes,
        cross_warnings=cross_warnings,
        confidence_breakdown=_confidence_breakdown(leg_results),
        league_distribution=_league_distribution(leg_results),
        deterministic_summary="",
        llm_summary=None,
    )
    result.deterministic_summary = _build_deterministic_summary(result)
    llm_summary, llm_note = _generate_llm_summary(result)
    if llm_note:
        result.notes.append(llm_note)
    result.llm_summary = llm_summary
    return result


def humanize_date(target_date: date) -> str:
    suffix = "th" if 11 <= target_date.day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(target_date.day % 10, "th")
    return f"{target_date.day}{suffix} {target_date.strftime('%B')}"


def _fetch_events_for_request(request: ParlayBuildRequest) -> Tuple[List[Dict], List[str]]:
    target_date = date.fromisoformat(request.target_date)
    all_events: List[Dict] = []
    notes: List[str] = []
    for league in request.leagues:
        events, fetch_notes = rag.fetch_events(league, set(rag.DEFAULT_MARKETS), target_date=target_date)
        for event in events:
            event["_league"] = league
        all_events.extend(events)
        notes.extend(fetch_notes)
    # Provider fetches can occasionally leak adjacent fixtures; normalize on exact date.
    all_events = rag.filter_events_by_exact_date(all_events, target_date)
    return all_events, notes


def _filter_events_for_request(events: List[Dict], request: ParlayBuildRequest, notes: List[str]) -> List[Dict]:
    filtered = list(events)
    if request.allowed_event_ids:
        allowed = set(request.allowed_event_ids)
        filtered = [event for event in filtered if str(event.get("id") or "") in allowed]

    if request.match_filter:
        filtered = [event for event in filtered if _event_matches_filter(event, request.match_filter)]
        if not filtered:
            notes.append(f"No fixture matched '{request.match_filter}' on {request.date_phrase}.")
    return filtered


def _enrich_events(events: List[Dict], request: ParlayBuildRequest) -> Tuple[List[Dict], List[str]]:
    desired_groups = set(MARKET_FILTER_TO_ENRICH.get(request.market_filter or "mixed", set()))
    if not desired_groups:
        return events, []

    grouped: Dict[str, List[Dict]] = {}
    for event in events:
        grouped.setdefault(str(event.get("_league") or request.default_league), []).append(event)

    notes: List[str] = []
    enriched: List[Dict] = []
    for league, league_events in grouped.items():
        league_enriched, league_notes = rag.enrich_events_for_groups(league_events, league, desired_groups)
        enriched.extend(league_enriched)
        notes.extend(league_notes)
    return enriched, notes


def _build_constraint_spec(request: ParlayBuildRequest) -> rag.ConstraintSpec:
    parsed = None
    if request.constraint_query:
        try:
            parsed = rag.parse_constraints(request.constraint_query, default_league=request.default_league)
        except Exception:
            parsed = None

    market_group = MARKET_FILTER_TO_GROUP.get(request.market_filter or "mixed")
    hard_include = set(parsed.hard_include_groups) if parsed else set()
    hard_exclude = set(parsed.hard_exclude_groups) if parsed else set()
    soft_prefer = set(parsed.soft_prefer_groups) if parsed else set()
    required_group_counts = dict(parsed.required_group_counts) if parsed else {}

    if market_group:
        hard_include.add(market_group)
        hard_exclude |= (set(rag.KNOWN_GROUPS) - {market_group})
        soft_prefer.add(market_group)

    if request.allowed_groups:
        allowed = set(request.allowed_groups)
        hard_exclude |= (set(rag.KNOWN_GROUPS) - allowed)
        hard_include |= (allowed & set(request.required_group_counts.keys()))

    if request.soft_prefer_groups:
        soft_prefer |= set(request.soft_prefer_groups)

    if request.required_group_counts:
        for group, count in request.required_group_counts.items():
            required_group_counts[group] = max(required_group_counts.get(group, 0), count)

    return rag.ConstraintSpec(
        requested_markets=set(parsed.requested_markets) if parsed else set(rag.DEFAULT_MARKETS),
        hard_include_groups=hard_include,
        hard_exclude_groups=hard_exclude,
        soft_prefer_groups=soft_prefer,
        required_group_counts=required_group_counts,
        forbid_spread_keys=bool(parsed.forbid_spread_keys) if parsed else False,
        require_total_corner_keys=bool(parsed.require_total_corner_keys) if parsed else False,
        target_multiplier=request.target_odds,
        target_mode=request.target_mode,
        leg_count=request.legs_requested,
        per_match_mode=request.same_game_only,
        require_unique_events=request.require_unique_events,
        time_window="upcoming",
        league=request.default_league,
        leagues=request.leagues if len(request.leagues) > 1 else None,
        target_min=request.target_min,
        target_max=request.target_max,
    )


def _resolve_locked_legs(candidates: List[rag.CandidateLeg], locked_refs: Sequence[ParlayLegRef]) -> List[rag.CandidateLeg]:
    resolved: List[rag.CandidateLeg] = []
    for ref in locked_refs:
        match = next((candidate for candidate in candidates if _candidate_matches_ref(candidate, ref)), None)
        if match is None:
            raise ParlayBuildError(
                "One or more original legs are no longer available at the current prices.",
                [f"Unavailable locked leg: {ref.market_key} {ref.outcome} on event {ref.event_id}."],
            )
        resolved.append(match)
    return resolved


def _filter_candidate_pool(
    candidates: List[rag.CandidateLeg],
    request: ParlayBuildRequest,
    constraint: rag.ConstraintSpec,
    locked_legs: Sequence[rag.CandidateLeg],
) -> List[rag.CandidateLeg]:
    locked_refs = [ParlayLegRef(event_id=leg.event_id, market_key=leg.market_key, outcome=leg.outcome, point=leg.point, league=leg.league or request.default_league) for leg in locked_legs]
    excluded_refs = list(request.excluded_legs)
    blocked_event_ids = {leg.event_id for leg in locked_legs} if (constraint.require_unique_events and not request.same_game_only) else set()
    if request.prefer_new_fixture and not request.same_game_only:
        blocked_event_ids |= {leg.event_id for leg in locked_legs}

    filtered: List[rag.CandidateLeg] = []
    allowed_event_ids = set(request.allowed_event_ids)
    allowed_groups = set(request.allowed_groups)
    allowed_confidences = {str(x).lower() for x in request.allowed_confidences}
    for candidate in candidates:
        if any(_candidate_matches_ref(candidate, ref) for ref in locked_refs):
            continue
        if any(_candidate_matches_ref(candidate, ref) for ref in excluded_refs):
            continue
        if allowed_event_ids and candidate.event_id not in allowed_event_ids:
            continue
        if candidate.event_id in blocked_event_ids:
            continue
        if any(rag.contradictory(candidate, locked) for locked in locked_legs):
            continue
        group = rag.market_group_from_key(candidate.market_key)
        if allowed_groups and group not in allowed_groups:
            continue
        if request.min_model_prob is not None or allowed_confidences:
            sc = leg_standalone_confidence(candidate, candidate.league or request.default_league)
            model_prob = sc.get("model_prob")
            confidence = str(sc.get("confidence") or "").lower()
            if request.min_model_prob is not None and (model_prob is None or model_prob < request.min_model_prob):
                continue
            if allowed_confidences and confidence not in allowed_confidences:
                continue
        filtered.append(candidate)
    return filtered


def _adjust_constraint_for_locked(
    constraint: rag.ConstraintSpec,
    locked_legs: Sequence[rag.CandidateLeg],
    remaining_needed: int,
) -> rag.ConstraintSpec:
    if not locked_legs:
        return constraint

    locked_groups = Counter(rag.market_group_from_key(leg.market_key) for leg in locked_legs)
    target_multiplier = constraint.target_multiplier
    target_min = constraint.target_min
    target_max = constraint.target_max
    locked_odds = rag.combined_odds(list(locked_legs))
    if target_multiplier:
        target_multiplier = max(1.01, target_multiplier / locked_odds)
    if target_min:
        target_min = max(1.01, target_min / locked_odds)
    if target_max:
        target_max = max(1.01, target_max / locked_odds)

    required_group_counts = dict(constraint.required_group_counts)
    for group, count in list(required_group_counts.items()):
        remaining = count - locked_groups.get(group, 0)
        if remaining > 0:
            required_group_counts[group] = remaining
        else:
            required_group_counts.pop(group, None)

    hard_include = {
        group for group in constraint.hard_include_groups
        if locked_groups.get(group, 0) <= 0
    }

    return rag.ConstraintSpec(
        requested_markets=set(constraint.requested_markets),
        hard_include_groups=hard_include,
        hard_exclude_groups=set(constraint.hard_exclude_groups),
        soft_prefer_groups=set(constraint.soft_prefer_groups),
        required_group_counts=required_group_counts,
        forbid_spread_keys=constraint.forbid_spread_keys,
        require_total_corner_keys=constraint.require_total_corner_keys,
        target_multiplier=target_multiplier,
        target_mode=constraint.target_mode,
        leg_count=remaining_needed,
        per_match_mode=constraint.per_match_mode,
        require_unique_events=constraint.require_unique_events,
        time_window=constraint.time_window,
        league=constraint.league,
        leagues=list(constraint.leagues) if constraint.leagues else None,
        target_min=target_min,
        target_max=target_max,
    )


def _build_leg_results(
    selected: Sequence[rag.CandidateLeg],
    request: ParlayBuildRequest,
) -> Tuple[List[ParlayLegResult], List[str]]:
    results: List[ParlayLegResult] = []
    warnings: List[str] = []
    for leg in selected:
        league = leg.league or request.default_league
        standalone = leg_standalone_confidence(leg, league)
        model_prob = standalone.get("model_prob")
        implied = implied_prob(float(leg.odds)) if getattr(leg, "odds", None) else None
        edge = value_edge(model_prob, implied) if model_prob is not None and implied is not None else None
        quality = rag.kb_leg_quality(leg, league)
        reasons = leg_evidence(leg, league)[:3]
        if not standalone.get("side_agrees", True):
            warning = standalone.get("warning")
            if warning:
                warnings.append(warning)
            elif standalone.get("standalone_side"):
                warnings.append(
                    f"{leg.fixture}: standalone {rag.market_group_from_key(leg.market_key)} model favors {standalone['standalone_side']}."
                )
        results.append(
            ParlayLegResult(
                event_id=leg.event_id,
                league=league,
                fixture=leg.fixture,
                home_team=leg.home_team,
                away_team=leg.away_team,
                market_key=leg.market_key,
                market_group=rag.market_group_from_key(leg.market_key),
                outcome=leg.outcome,
                point=leg.point,
                pick_display=_format_pick_display(leg),
                odds=float(leg.odds),
                bookmaker=leg.bookmaker,
                quality_score=float(quality),
                confidence=standalone.get("confidence"),
                model_prob=model_prob,
                implied_prob=implied,
                value_edge=edge,
                projected_total=standalone.get("projected"),
                side_agrees=bool(standalone.get("side_agrees", True)),
                warning=standalone.get("warning"),
                reasons=reasons,
            )
        )
    return results, warnings


def _build_deterministic_summary(result: ParlayBuildResult) -> str:
    legs = result.selected_legs
    conf_counts = Counter(leg.confidence for leg in legs if leg.confidence)
    group_counts = Counter(leg.market_group for leg in legs)
    avg_edge = None
    edge_values = [leg.value_edge for leg in legs if leg.value_edge is not None]
    if edge_values:
        avg_edge = sum(edge_values) / len(edge_values)

    parts: List[str] = []
    if conf_counts:
        ordered = sorted(conf_counts.items(), key=lambda item: CONFIDENCE_ORDER.get(str(item[0]), 99))
        parts.append("Confidence mix: " + ", ".join(f"{count} {label}" for label, count in ordered) + ".")

    if group_counts or avg_edge is not None:
        detail_bits: List[str] = []
        if group_counts:
            focus = ", ".join(f"{count} {group}" for group, count in group_counts.most_common(2))
            detail_bits.append(f"Market mix stays concentrated around {focus}")
        if avg_edge is not None:
            direction = "positive" if avg_edge >= 0 else "negative"
            detail_bits.append(f"model-versus-market pricing is {direction} on average ({avg_edge:+.1%})")
        parts.append(" and ".join(detail_bits).rstrip(".") + ".")

    if result.request.target_odds:
        parts.append(
            f"Combined odds landed at {result.combined_odds:.2f}x against a {result.request.target_odds:.2f}x target."
        )
    else:
        parts.append(f"Combined odds landed at {result.combined_odds:.2f}x.")

    if result.cross_warnings:
        parts.append(f"{len(result.cross_warnings)} leg(s) conflict with standalone projections.")
    else:
        parts.append("No standalone projection conflicts were flagged.")

    return " ".join(parts[:3]).strip()


def _generate_llm_summary(result: ParlayBuildResult) -> Tuple[Optional[str], Optional[str]]:
    if not getattr(rag, "client", None):
        return None, "OPENAI_API_KEY not set; using deterministic summary."

    leg_lines = []
    for index, leg in enumerate(result.selected_legs, start=1):
        reasons = " | ".join(leg.reasons[:2]) or "No extra evidence."
        model_prob = f"{leg.model_prob:.1%}" if leg.model_prob is not None else "n/a"
        edge = f"{leg.value_edge:+.1%}" if leg.value_edge is not None else "n/a"
        leg_lines.append(
            f"LEG {index}: {leg.market_group} | confidence={leg.confidence or 'n/a'} | "
            f"model_prob={model_prob} | value_edge={edge} | reasons={reasons}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You summarize a fixed football parlay after selection is already complete. "
                "Do not recommend changes. Do not mention specific line numbers, odds, or leg numbers. "
                "Write 2 concise sentences about the shared logic only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Parlay target: {result.request.target_odds or 'none'}x\n"
                f"Combined odds: {result.combined_odds:.2f}x\n"
                + "\n".join(leg_lines)
            ),
        },
    ]

    try:
        response = rag.client.chat.completions.create(
            model=getattr(rag, "CHAT_MODEL", "gpt-5.4"),
            messages=messages,
            temperature=0.2,
            max_tokens=120,
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return None, f"LLM summary failed: {exc}"

    if not _summary_is_safe(summary):
        return None, "LLM summary rejected; using deterministic summary."
    return summary, None


def _summary_is_safe(summary: str) -> bool:
    if not summary:
        return False
    if len(summary) > 500:
        return False
    bad_patterns = [
        r"leg\s+\d+",
        r"@\s*\d",
        r"\b(?:replace|swap|instead|consider|alternative)\b",
        r"\b(?:over|under)\s+\d",
    ]
    return not any(re.search(pattern, summary, flags=re.IGNORECASE) for pattern in bad_patterns)


def _confidence_breakdown(legs: Sequence[ParlayLegResult]) -> str:
    counts = Counter(leg.confidence for leg in legs if leg.confidence)
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: CONFIDENCE_ORDER.get(str(item[0]), 99))
    return ", ".join(f"{count} {label}" for label, count in ordered)


def _league_distribution(legs: Sequence[ParlayLegResult]) -> str:
    counts = Counter(leg.league for leg in legs if leg.league)
    if len(counts) <= 1:
        return ""
    return ", ".join(f"{count}x {league}" for league, count in counts.most_common())


def _format_pick_display(leg: rag.CandidateLeg) -> str:
    group = rag.market_group_from_key(leg.market_key)
    if leg.point is None:
        return f"{leg.outcome} ({leg.market_key}/{group})"
    if group == "spreads":
        return f"{leg.outcome} {leg.point:+g} ({leg.market_key}/{group})"
    return f"{leg.outcome} {leg.point:g} ({leg.market_key}/{group})"


def _event_matches_filter(event: Dict, match_filter: str) -> bool:
    event_home = str(event.get("home_team") or "")
    event_away = str(event.get("away_team") or "")
    fixture = f"{event_home} vs {event_away}"
    if _normalize_text(fixture) == _normalize_text(match_filter):
        return True

    home, away = _split_fixture(match_filter)
    if not home or not away:
        return _normalize_text(match_filter) in _normalize_text(fixture)

    event_home_aliases = rag.team_name_aliases(event_home)
    event_away_aliases = rag.team_name_aliases(event_away)
    home_aliases = rag.team_name_aliases(home)
    away_aliases = rag.team_name_aliases(away)
    return bool(event_home_aliases & home_aliases and event_away_aliases & away_aliases)


def _candidate_matches_ref(candidate: rag.CandidateLeg, ref: ParlayLegRef) -> bool:
    if str(candidate.event_id) != str(ref.event_id):
        return False
    if candidate.market_key.lower() != ref.market_key.lower():
        return False
    if candidate.outcome.strip().lower() != ref.outcome.strip().lower():
        return False
    if _point_key(candidate.point) != _point_key(ref.point):
        return False
    if ref.league and (candidate.league or "").lower() != ref.league.lower():
        return False
    return True


def _split_fixture(text: str) -> Tuple[Optional[str], Optional[str]]:
    for separator in (" vs ", " v ", " @ "):
        if separator in text.lower():
            pattern = re.compile(re.escape(separator), re.IGNORECASE)
            parts = pattern.split(text, maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    return None, None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())).strip()


def _point_key(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"
