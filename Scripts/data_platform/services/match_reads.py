"""Phase 0 Match Read contract and orchestration service.

The service turns a set of canonical ``MarketResult`` snapshots into one
fixture-level decision without making a UI choice.  It deliberately keeps
individual official recommendations and same-game packages separate:

* a Match Read may be recommended, no-bet, or unavailable;
* only explicitly selected, priced leaf results may later be published;
* a same-game package requires one real bookmaker quote, but cannot carry a
  joint probability or EV until correlation modelling is introduced.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from ..repositories.match_reads import MatchReadRepository


MATCH_READ_SCHEMA_VERSION = "match_read.v1"
VALID_MATCH_READ_STATUSES = frozenset({"recommended", "no_bet", "unavailable"})
VALID_MATCH_READ_STAGES = frozenset({"pre_match", "confirmed_lineups"})
VALID_SELECTION_ROLES = frozenset({"core", "supporting", "alternative"})
VALID_DELIVERY_SURFACES = frozenset({"discord", "website"})
SAME_GAME_BUILD_KIND = "same_game_build"
COHERENT_BUILD_STATUS = "coherent_build"
MAX_ACTIONABLE_SELECTIONS = 3


class MatchReadValidationError(ValueError):
    """Raised when a proposed fixture-level read violates the Phase 0 rules."""


def _snapshot(value: Any, *, field: str) -> Dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    candidate = to_dict() if callable(to_dict) else value
    if not isinstance(candidate, Mapping):
        raise MatchReadValidationError(f"{field} must be a mapping or expose to_dict().")
    try:
        return json.loads(json.dumps(dict(candidate), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MatchReadValidationError(f"{field} must be JSON serializable.") from exc


def _text(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise MatchReadValidationError(f"{field} is required.")
    return result


def _number(value: Any, field: str, *, minimum: Optional[float] = None) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MatchReadValidationError(f"{field} must be numeric.") from exc
    if minimum is not None and result < minimum:
        raise MatchReadValidationError(f"{field} must be at least {minimum}.")
    return result


def _as_utc(value: Optional[Any]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MatchReadValidationError("evaluated_at must be an ISO-8601 timestamp.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _selection_pick(quote: Mapping[str, Any]) -> str:
    side = _text(quote.get("side"), "selection.quote.side")
    line = _number(quote.get("line"), "selection.quote.line")
    return f"{side.title()} {line:g}" if line is not None else side.title()


class MatchReadService:
    """Validate and store versioned fixture-level Match Reads."""

    def __init__(self, repo: Optional[MatchReadRepository] = None):
        self._repo = repo or MatchReadRepository()

    def create(
        self,
        *,
        canonical_results: Sequence[Any],
        thesis: str,
        status: str,
        selections: Sequence[Mapping[str, Any]] = (),
        packages: Sequence[Mapping[str, Any]] = (),
        stage: str = "pre_match",
        game_script: Optional[Mapping[str, Any]] = None,
        evaluated_at: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Record a new immutable read or return the exact existing snapshot.

        ``selections`` reference a canonical result by its zero-based
        ``result_index`` and declare one of ``core``, ``supporting`` or
        ``alternative``.  This explicit declaration prevents ranking logic
        from accidentally deciding product presentation in the data layer.
        """
        result_snapshots = [
            _snapshot(result, field=f"canonical_results[{index}]")
            for index, result in enumerate(canonical_results)
        ]
        if not result_snapshots:
            raise MatchReadValidationError("A Match Read requires canonical results for its fixture.")

        fixture = _fixture_from_results(result_snapshots)
        normalised_status = str(status or "").strip().lower()
        if normalised_status not in VALID_MATCH_READ_STATUSES:
            raise MatchReadValidationError(
                f"status must be one of: {', '.join(sorted(VALID_MATCH_READ_STATUSES))}."
            )
        normalised_stage = str(stage or "").strip().lower()
        if normalised_stage not in VALID_MATCH_READ_STAGES:
            raise MatchReadValidationError(
                f"stage must be one of: {', '.join(sorted(VALID_MATCH_READ_STAGES))}."
            )
        normalised_thesis = _text(thesis, "thesis")
        normalised_script = _snapshot(game_script, field="game_script") if game_script is not None else None
        selection_fields, selected_positions = _normalise_selections(
            result_snapshots,
            selections,
            status=normalised_status,
        )
        package_fields = _normalise_packages(
            packages,
            selected_positions=selected_positions,
        )
        if normalised_status != "recommended" and package_fields:
            raise MatchReadValidationError("Only a recommended Match Read may contain a same-game build.")

        provenance = _provenance_from_results(result_snapshots)
        evaluated = _as_utc(evaluated_at or provenance.get("generated_at"))
        payload = {
            "schema_version": MATCH_READ_SCHEMA_VERSION,
            "fixture": fixture,
            "stage": normalised_stage,
            "status": normalised_status,
            "thesis": normalised_thesis,
            "game_script": normalised_script,
            "canonical_results": result_snapshots,
            "selections": [
                {
                    "position": item["position"],
                    "role": item["role"],
                    "result_index": item["result_index"],
                }
                for item in selection_fields
            ],
            "packages": [item["package_json"] for item in package_fields],
            "provenance": provenance,
            "evaluated_at": evaluated.isoformat(),
        }
        read_key = _hash(payload)
        # A package may recur unchanged in a later read revision. Its identity
        # must therefore include the parent immutable read, not just its own
        # quote/leg payload.
        for package in package_fields:
            package["package_key"] = _hash({
                "read_key": read_key,
                "package": package["package_json"],
            })
        read_fields = {
            "read_key": read_key,
            "fixture_api_id": fixture["event_id"],
            "league": fixture["league"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "kickoff": fixture.get("kickoff"),
            "stage": normalised_stage,
            "status": normalised_status,
            "thesis": normalised_thesis,
            "game_script_json": normalised_script,
            "input_snapshot_id": provenance["input_snapshot_id"],
            "pipeline_version": provenance["pipeline_version"],
            "model_version": provenance.get("model_version"),
            "evaluated_at": evaluated,
            "read_json": payload,
        }
        return self._repo.record(
            read_fields=read_fields,
            selections=[_storage_selection(item) for item in selection_fields],
            packages=[_storage_package(item) for item in package_fields],
        )

    def get(self, match_read_id: int) -> Optional[Dict[str, Any]]:
        return self._repo.get(match_read_id)

    def list_for_fixture(
        self,
        fixture_api_id: str,
        *,
        stage: Optional[str] = None,
        limit: int = 20,
    ) -> list[Dict[str, Any]]:
        return self._repo.list_for_fixture(fixture_api_id, stage=stage, limit=limit)

    def record_delivery(
        self,
        match_read_id: int,
        *,
        surface: str,
        external_reference: Optional[str],
        delivered_at: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a visible match card, including no-bet/unavailable cards."""
        normalised_surface = str(surface or "").strip().lower()
        if normalised_surface not in VALID_DELIVERY_SURFACES:
            raise MatchReadValidationError(
                f"surface must be one of: {', '.join(sorted(VALID_DELIVERY_SURFACES))}."
            )
        read = self._repo.get(match_read_id)
        if read is None:
            raise MatchReadValidationError(f"Unknown Match Read {match_read_id}.")
        delivery_time = _as_utc(delivered_at)
        delivery_key = _hash({
            "read_key": read["read_key"],
            "surface": normalised_surface,
            "external_reference": str(external_reference or "default"),
        })
        return self._repo.record_delivery(
            match_read_id,
            delivery_key=delivery_key,
            surface=normalised_surface,
            external_reference=str(external_reference) if external_reference is not None else None,
            delivered_at=delivery_time,
            metadata=metadata,
        )

    def link_published_recommendation(
        self,
        match_read_id: int,
        *,
        selection_position: int,
        published_recommendation_id: int,
    ) -> bool:
        return self._repo.link_published_recommendation(
            match_read_id,
            selection_position=selection_position,
            published_recommendation_id=published_recommendation_id,
        )

    def settle_package(
        self,
        package_id: int,
        *,
        outcome: str,
        settled_at: Optional[Any] = None,
        profit_units: Optional[float] = None,
    ) -> bool:
        return self._repo.mark_package_outcome(
            package_id,
            outcome=outcome,
            settled_at=_as_utc(settled_at) if settled_at is not None else None,
            profit_units=profit_units,
        )


def _fixture_from_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    first = results[0].get("fixture")
    if not isinstance(first, Mapping):
        raise MatchReadValidationError("canonical_results[0].fixture is required.")
    fixture = {
        "event_id": _text(first.get("event_id"), "fixture.event_id"),
        "league": _text(first.get("league"), "fixture.league"),
        "home_team": _text(first.get("home_team"), "fixture.home_team"),
        "away_team": _text(first.get("away_team"), "fixture.away_team"),
        "kickoff": str(first.get("kickoff") or "").strip() or None,
    }
    for index, result in enumerate(results[1:], start=1):
        other = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
        for field in ("event_id", "league", "home_team", "away_team"):
            if str(other.get(field) or "").strip() != fixture[field]:
                raise MatchReadValidationError(
                    f"canonical_results[{index}] belongs to a different fixture ({field})."
                )
    return fixture


def _provenance_from_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    snapshots = []
    pipeline_versions = []
    model_versions = []
    generated_at = []
    for index, result in enumerate(results):
        provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
        snapshots.append(_text(provenance.get("input_snapshot_id"), f"canonical_results[{index}].provenance.input_snapshot_id"))
        pipeline_versions.append(_text(provenance.get("pipeline_version"), f"canonical_results[{index}].provenance.pipeline_version"))
        model_value = str(provenance.get("model_version") or "").strip()
        if model_value:
            model_versions.append(model_value)
        timestamp = str(provenance.get("generated_at") or provenance.get("as_of") or "").strip()
        if timestamp:
            generated_at.append(timestamp)
    return {
        # Full leaf snapshots are stored in read_json. This deterministic
        # digest identifies the complete card input even if markets were
        # evaluated from separate underlying snapshot IDs.
        "input_snapshot_id": f"match-read:{_hash({'snapshots': snapshots})}",
        "pipeline_version": pipeline_versions[0] if len(set(pipeline_versions)) == 1 else "mixed-canonical-pipeline",
        "model_version": model_versions[0] if len(set(model_versions)) == 1 and model_versions else None,
        "generated_at": generated_at[0] if generated_at else None,
        "source_snapshot_ids": snapshots,
    }


def _normalise_selections(
    results: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    status: str,
) -> tuple[list[Dict[str, Any]], set[int]]:
    if status != "recommended":
        if specs:
            raise MatchReadValidationError(f"A {status} Match Read cannot contain actionable selections.")
        return [], set()
    if not specs:
        raise MatchReadValidationError("A recommended Match Read requires at least one selected result.")
    if len(specs) > MAX_ACTIONABLE_SELECTIONS:
        raise MatchReadValidationError(
            f"A Match Read may contain at most {MAX_ACTIONABLE_SELECTIONS} actionable selections."
        )

    normalised = []
    seen_indexes: set[int] = set()
    roles = []
    for position, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            raise MatchReadValidationError(f"selections[{position - 1}] must be a mapping.")
        try:
            result_index = int(spec.get("result_index"))
        except (TypeError, ValueError) as exc:
            raise MatchReadValidationError(f"selections[{position - 1}].result_index must be an integer.") from exc
        if result_index < 0 or result_index >= len(results):
            raise MatchReadValidationError(f"selections[{position - 1}].result_index is out of range.")
        if result_index in seen_indexes:
            raise MatchReadValidationError("A canonical result may appear only once in a Match Read.")
        seen_indexes.add(result_index)
        role = str(spec.get("role") or "").strip().lower()
        if role not in VALID_SELECTION_ROLES:
            raise MatchReadValidationError(
                f"selections[{position - 1}].role must be one of: {', '.join(sorted(VALID_SELECTION_ROLES))}."
            )
        market_result = results[result_index]
        decision = market_result.get("decision") if isinstance(market_result.get("decision"), Mapping) else {}
        quote = decision.get("quote") if isinstance(decision.get("quote"), Mapping) else {}
        market = market_result.get("market") if isinstance(market_result.get("market"), Mapping) else {}
        if decision.get("status") != "recommended":
            raise MatchReadValidationError("Only a priced canonical recommendation can be selected.")
        odds = _number(quote.get("odds"), "selection.quote.odds", minimum=1.000001)
        model_probability = _number(decision.get("model_probability"), "selection.model_probability", minimum=0.0)
        if model_probability is None or model_probability > 1.0:
            raise MatchReadValidationError("A selected result requires model_probability between 0 and 1.")
        normalised.append({
            "result_index": result_index,
            "role": role,
            "position": position,
            "market_group": _text(market.get("group"), "selection.market.group"),
            "market_key": _text(market.get("key"), "selection.market.key"),
            "pick": _selection_pick(quote),
            "quote_odds": odds,
            "bookmaker": str(quote.get("bookmaker") or "").strip() or None,
            "model_probability": model_probability,
            "value_edge": _number(decision.get("value_edge"), "selection.value_edge"),
            "selection_json": market_result,
        })
        roles.append(role)
    if roles.count("core") != 1:
        raise MatchReadValidationError("A recommended Match Read requires exactly one core selection.")
    return normalised, {int(item["position"]) for item in normalised}


def _normalise_packages(
    specs: Sequence[Mapping[str, Any]],
    *,
    selected_positions: set[int],
) -> list[Dict[str, Any]]:
    normalised = []
    for position, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            raise MatchReadValidationError(f"packages[{position - 1}] must be a mapping.")
        kind = str(spec.get("kind") or "").strip().lower()
        if kind != SAME_GAME_BUILD_KIND:
            raise MatchReadValidationError(f"packages[{position - 1}].kind must be {SAME_GAME_BUILD_KIND!r}.")
        selected = spec.get("selection_positions")
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
            raise MatchReadValidationError("package.selection_positions must be a sequence of Match Read positions.")
        selection_positions = [int(item) for item in selected]
        if len(selection_positions) < 2 or len(set(selection_positions)) != len(selection_positions):
            raise MatchReadValidationError("A same-game build needs at least two distinct selections.")
        if not set(selection_positions).issubset(selected_positions):
            raise MatchReadValidationError("A package may only include selections in its Match Read.")
        quote = spec.get("quote") if isinstance(spec.get("quote"), Mapping) else {}
        quoted_odds = _number(quote.get("odds"), "package.quote.odds", minimum=1.000001)
        bookmaker = _text(quote.get("bookmaker"), "package.quote.bookmaker")
        if spec.get("joint_model_probability") is not None or spec.get("expected_value") is not None:
            raise MatchReadValidationError(
                "Phase 0 same-game builds cannot claim joint probability or expected value."
            )
        package_json = _snapshot(spec, field=f"packages[{position - 1}]")
        normalised.append({
            "package_key": _hash({"package": package_json}),
            "kind": SAME_GAME_BUILD_KIND,
            "status": COHERENT_BUILD_STATUS,
            "label": _text(spec.get("label") or "Same-game build", "package.label"),
            "rationale": str(spec.get("rationale") or "").strip() or None,
            "position": position,
            "quoted_odds": quoted_odds,
            "bookmaker": bookmaker,
            "joint_model_probability": None,
            "expected_value": None,
            "stake_units": _number(spec.get("stake_units", 1.0), "package.stake_units", minimum=0.000001),
            "profit_units": None,
            "outcome": None,
            "settled_at": None,
            "package_json": package_json,
            "selection_positions": selection_positions,
        })
    return normalised


def _storage_selection(selection: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "result_index"}


def _storage_package(package: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(package)
