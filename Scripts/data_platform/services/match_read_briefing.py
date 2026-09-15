"""Evidence-grounded editorial briefings for persisted Match Reads.

The Match Read compiler is intentionally deterministic: it decides whether a
fixture has compatible selections, but it is not a copywriter.  This module is
the narrow presentation enrichment boundary above that compiler.  It reduces a
canonical result set to a small, auditable fact pack and may ask GPT-5.6 to
turn *only those facts* into a short editorial briefing.

The numerical model and selection policy are never changed here.  If OpenAI is
unavailable, or the response does not satisfy the contract, a deterministic
briefing is used instead.  A fact-key lets the scheduled worker reuse a prior
briefing when the underlying model inputs have not changed.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .match_read_compiler import MatchReadDraft


MATCH_READ_BRIEFING_SCHEMA_VERSION = "match-read-briefing.v1"
MATCH_READ_BRIEFING_MODEL = "gpt-5.6"
MATCH_READ_GAME_SCRIPT_SCHEMA_VERSION = "match-read-game-script.v2"
_MAX_SUMMARY_WORDS = 72
_MIN_SUMMARY_WORDS = 30
_MAX_BULLET_WORDS = 16
_MAX_BULLET_CHARACTERS = 108
_BANNED_COPY_TERMS = re.compile(
    r"\b(bet|bets|betting|pick|selection|odds|bookmaker|line|edge|value|stake|parlay)\b",
    re.IGNORECASE,
)

_BRIEFING_INSTRUCTIONS = """You are Spix's football data editor. Write a concise,
genuinely informative football fixture briefing using only the verified JSON
facts supplied by the user. Do not infer causes, injuries, tactics, form,
players, or historical facts that are not explicitly present.

Return exactly one short paragraph and exactly three scannable card bullets.
The paragraph must be 30-72 words. Each bullet must be 5-16 words and under
108 characters. Explain the likely character of the match from the projected
goals, result balance, and event-volume figures. You may say "the model
projects" and cite supplied figures, but do not mention bets, picks,
selections, odds, lines, bookmakers, value, edge, staking, or parlays. Do not
give an instruction to the reader. Keep the language direct and specific."""


def enrich_match_read_draft(
    draft: MatchReadDraft,
    event: Mapping[str, Any],
    *,
    cached_briefing: Optional[Mapping[str, Any]] = None,
    generator: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> Tuple[MatchReadDraft, Optional[str]]:
    """Return a draft with a briefing and visual identity ready for delivery.

    ``cached_briefing`` is a prior immutable Match Read's briefing with the
    same fact-key.  It is deliberately reused byte-for-byte, which prevents a
    ten-minute scheduler from paying for or inventing a new version of prose
    when nothing factual changed.

    The returned note is operational context suitable for a worker log.  A
    fallback is still a valid persisted Match Read: editorial availability must
    never block a numerically valid, auditable fixture decision.
    """
    facts = build_match_read_facts(draft)
    fact_key = _fact_key(facts)
    briefing = _valid_cached_briefing(cached_briefing, fact_key)
    note: Optional[str] = None
    if briefing is None:
        candidate: Optional[Mapping[str, Any]] = None
        try:
            candidate = (generator or generate_gpt_briefing)(facts)
        except Exception as exc:  # The Match Read must survive an editorial outage.
            note = f"AI briefing unavailable ({type(exc).__name__}: {exc}); using deterministic briefing."
        briefing = _normalise_generated_briefing(candidate, fact_key=fact_key)
        if briefing is None:
            briefing = _deterministic_briefing(facts, fact_key=fact_key)
            if note is None:
                note = "AI briefing was unavailable or invalid; using deterministic briefing."

    game_script = dict(draft.game_script)
    game_script["schema_version"] = MATCH_READ_GAME_SCRIPT_SCHEMA_VERSION
    game_script["briefing"] = briefing
    visuals = extract_fixture_visuals(event)
    if visuals:
        game_script["visuals"] = visuals
    return replace(draft, thesis=str(briefing["summary"]), game_script=game_script), note


def build_match_read_facts(draft: MatchReadDraft) -> Dict[str, Any]:
    """Build the small numerical source-of-truth supplied to the copywriter.

    Prices, edges and recommendation decisions are intentionally excluded.
    They are not required to describe a match and must not leak into the
    editorial paragraph.  The full canonical snapshots remain attached to the
    Match Read itself for audit.
    """
    fixture = _fixture_from_results(draft.canonical_results)
    markets = {
        _text(_mapping(result.get("market")).get("group")).lower(): result
        for result in draft.canonical_results
        if _text(_mapping(result.get("market")).get("group"))
    }
    facts: Dict[str, Any] = {
        "fixture": fixture,
        "stage": draft.stage,
        "metrics": {},
        "coverage": {
            "evaluated_markets": len(draft.canonical_results),
            "lineup_state": _lineup_state(draft.canonical_results, draft.stage),
        },
    }
    metrics: Dict[str, Any] = facts["metrics"]

    for group, label in (("goals", "goals"), ("corners", "corners"), ("cards", "cards"), ("sot", "shots_on_target")):
        projection = _mapping(markets.get(group, {}).get("projection"))
        value = _finite(projection.get("value"))
        if value is None:
            continue
        metric: Dict[str, Any] = {"projected": _round(value)}
        season = _finite(projection.get("season_component"))
        recent = _finite(projection.get("recent_component"))
        if season is not None:
            metric["season_baseline"] = _round(season)
        if recent is not None:
            metric["recent_baseline"] = _round(recent)
        metrics[label] = metric

    for group in ("btts", "moneyline"):
        components = _mapping(_mapping(markets.get(group, {}).get("projection")).get("components"))
        if group == "btts":
            home_goals = _finite(components.get("home_goals"))
            away_goals = _finite(components.get("away_goals"))
            both_score = _finite(components.get("yes_probability"))
            if any(value is not None for value in (home_goals, away_goals, both_score)):
                metrics["team_goals"] = {
                    key: _round(value)
                    for key, value in (
                        ("home", home_goals),
                        ("away", away_goals),
                        ("both_teams_score_probability", both_score),
                    )
                    if value is not None
                }
        else:
            home = _finite(components.get("home_probability"))
            draw = _finite(components.get("draw_probability"))
            away = _finite(components.get("away_probability"))
            if any(value is not None for value in (home, draw, away)):
                metrics["result_probabilities"] = {
                    key: _round(value)
                    for key, value in (("home", home), ("draw", draw), ("away", away))
                    if value is not None
                }
    return facts


def match_read_fact_key(draft: MatchReadDraft) -> str:
    """Return the stable editorial cache key for one compiled fixture read."""
    return _fact_key(build_match_read_facts(draft))


def generate_gpt_briefing(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    """Ask GPT-5.6 for structured editorial copy from an auditable fact pack.

    Importing and creating the client lazily means no OpenAI dependency or API
    key is required for the numerical pipeline, offline tests, or local UI.
    Responses are not stored by OpenAI, and the model is asked for a tiny,
    strict JSON object only.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI

    response = OpenAI(api_key=api_key).responses.create(
        model=os.environ.get("MATCH_READ_BRIEFING_MODEL", MATCH_READ_BRIEFING_MODEL).strip() or MATCH_READ_BRIEFING_MODEL,
        instructions=_BRIEFING_INSTRUCTIONS,
        input=json.dumps(facts, sort_keys=True, separators=(",", ":")),
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "match_read_briefing",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "required": ["summary", "bullets"],
                    "additionalProperties": False,
                },
            },
        },
        max_output_tokens=260,
        store=False,
    )
    output = str(getattr(response, "output_text", "") or "").strip()
    if not output:
        raise RuntimeError("GPT-5.6 returned no briefing text")
    value = json.loads(output)
    if not isinstance(value, Mapping):
        raise RuntimeError("GPT-5.6 returned a non-object briefing")
    return value


def find_cached_briefing(records: Iterable[Mapping[str, Any]], fact_key: str) -> Optional[Dict[str, Any]]:
    """Find a reusable valid briefing in immutable fixture/stage history."""
    for record in records:
        script = _mapping(record.get("game_script"))
        cached = _valid_cached_briefing(script.get("briefing"), fact_key)
        if cached is not None:
            return cached
    return None


def extract_fixture_visuals(event: Mapping[str, Any]) -> Dict[str, str]:
    """Extract optional provider logos without ever requiring a second call."""
    source = _mapping(event.get("visuals"))
    aliases = {
        "home_team_logo": ("home_team_logo", "home_logo"),
        "away_team_logo": ("away_team_logo", "away_logo"),
        "league_logo": ("league_logo", "competition_logo"),
    }
    visuals: Dict[str, str] = {}
    for target, keys in aliases.items():
        value = next((source.get(key) for key in keys if _safe_url(source.get(key))), None)
        if value is None:
            value = next((event.get(key) for key in keys if _safe_url(event.get(key))), None)
        if _safe_url(value):
            visuals[target] = str(value).strip()
    return visuals


def _normalise_generated_briefing(value: Optional[Mapping[str, Any]], *, fact_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    summary = _clean_copy(value.get("summary"))
    bullets_raw = value.get("bullets")
    bullets = [_clean_copy(item) for item in bullets_raw] if isinstance(bullets_raw, Sequence) and not isinstance(bullets_raw, (str, bytes)) else []
    if not _valid_copy(summary, bullets):
        return None
    return {
        "schema_version": MATCH_READ_BRIEFING_SCHEMA_VERSION,
        "fact_key": fact_key,
        "source": "gpt-5.6",
        "model": os.environ.get("MATCH_READ_BRIEFING_MODEL", MATCH_READ_BRIEFING_MODEL).strip() or MATCH_READ_BRIEFING_MODEL,
        "summary": summary,
        "bullets": bullets,
    }


def _valid_cached_briefing(value: Optional[Mapping[str, Any]], fact_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping) or str(value.get("fact_key") or "") != fact_key:
        return None
    summary = _clean_copy(value.get("summary"))
    raw_bullets = value.get("bullets")
    bullets = [_clean_copy(item) for item in raw_bullets] if isinstance(raw_bullets, Sequence) and not isinstance(raw_bullets, (str, bytes)) else []
    if not _valid_copy(summary, bullets):
        return None
    return {
        "schema_version": MATCH_READ_BRIEFING_SCHEMA_VERSION,
        "fact_key": fact_key,
        "source": str(value.get("source") or "cached"),
        "model": _text(value.get("model")) or None,
        "summary": summary,
        "bullets": bullets,
    }


def _deterministic_briefing(facts: Mapping[str, Any], *, fact_key: str) -> Dict[str, Any]:
    fixture = _mapping(facts.get("fixture"))
    home = _text(fixture.get("home_team")) or "The home side"
    away = _text(fixture.get("away_team")) or "the away side"
    metrics = _mapping(facts.get("metrics"))
    goals = _mapping(metrics.get("goals"))
    team_goals = _mapping(metrics.get("team_goals"))
    result = _mapping(metrics.get("result_probabilities"))
    corners = _mapping(metrics.get("corners"))

    summary_bits = []
    total_goals = _finite(goals.get("projected"))
    home_goals = _finite(team_goals.get("home"))
    away_goals = _finite(team_goals.get("away"))
    if total_goals is not None:
        summary_bits.append(f"The model projects {total_goals:.2f} total goals")
        if home_goals is not None and away_goals is not None:
            summary_bits[-1] += f", split {home_goals:.2f} for {home} and {away_goals:.2f} for {away}"
    home_probability = _finite(result.get("home"))
    draw_probability = _finite(result.get("draw"))
    away_probability = _finite(result.get("away"))
    if home_probability is not None and draw_probability is not None and away_probability is not None:
        summary_bits.append(
            f"The result model gives {home} a {home_probability * 100:.0f}% win chance, "
            f"with the draw at {draw_probability * 100:.0f}% and {away} at {away_probability * 100:.0f}%"
        )
    projected_corners = _finite(corners.get("projected"))
    if projected_corners is not None:
        summary_bits.append(f"Projected corner volume is {projected_corners:.1f}")
    if not summary_bits:
        summary_bits.append("The model has assessed the fixture across its core match-event profiles")
    summary = ". ".join(summary_bits[:3]).rstrip(".") + "."

    bullets = _fallback_bullets(facts)
    return {
        "schema_version": MATCH_READ_BRIEFING_SCHEMA_VERSION,
        "fact_key": fact_key,
        "source": "deterministic_fallback",
        "model": None,
        "summary": summary,
        "bullets": bullets,
    }


def _fallback_bullets(facts: Mapping[str, Any]) -> list[str]:
    fixture = _mapping(facts.get("fixture"))
    home = _text(fixture.get("home_team")) or "Home side"
    away = _text(fixture.get("away_team")) or "away side"
    metrics = _mapping(facts.get("metrics"))
    candidates: list[str] = []
    goals = _mapping(metrics.get("goals"))
    team_goals = _mapping(metrics.get("team_goals"))
    total = _finite(goals.get("projected"))
    home_goals = _finite(team_goals.get("home"))
    away_goals = _finite(team_goals.get("away"))
    if total is not None:
        if home_goals is not None and away_goals is not None:
            candidates.append(f"Goals: {total:.2f} projected — {home} {home_goals:.2f}, {away} {away_goals:.2f}.")
        else:
            candidates.append(f"Goals: {total:.2f} total goals projected.")
    probabilities = _mapping(metrics.get("result_probabilities"))
    home_probability = _finite(probabilities.get("home"))
    draw_probability = _finite(probabilities.get("draw"))
    away_probability = _finite(probabilities.get("away"))
    if home_probability is not None and draw_probability is not None and away_probability is not None:
        candidates.append(
            f"Result balance: {home} {home_probability * 100:.0f}%, draw {draw_probability * 100:.0f}%, {away} {away_probability * 100:.0f}%.")
    for key, label in (("corners", "Corners"), ("shots_on_target", "Shots on target"), ("cards", "Cards")):
        metric = _mapping(metrics.get(key))
        projected = _finite(metric.get("projected"))
        if projected is None:
            continue
        recent = _finite(metric.get("recent_baseline"))
        if recent is not None:
            candidates.append(f"{label}: {projected:.1f} projected versus a {recent:.1f} recent baseline.")
        else:
            candidates.append(f"{label}: {projected:.1f} projected across the match.")
    lineup_state = _text(_mapping(facts.get("coverage")).get("lineup_state"))
    if lineup_state == "confirmed":
        candidates.append("Inputs include the confirmed lineups.")
    elif lineup_state:
        candidates.append("Assessment currently uses pre-match inputs.")
    while len(candidates) < 3:
        candidates.append("Core match-event profiles have been assessed.")
    return candidates[:3]


def _valid_copy(summary: str, bullets: Sequence[str]) -> bool:
    if not summary or len(summary.split()) < _MIN_SUMMARY_WORDS or len(summary.split()) > _MAX_SUMMARY_WORDS:
        return False
    if _BANNED_COPY_TERMS.search(summary):
        return False
    if len(bullets) != 3:
        return False
    for bullet in bullets:
        if not bullet or len(bullet.split()) < 3 or len(bullet.split()) > _MAX_BULLET_WORDS:
            return False
        if len(bullet) > _MAX_BULLET_CHARACTERS or _BANNED_COPY_TERMS.search(bullet):
            return False
    return True


def _fixture_from_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    first = _mapping(results[0].get("fixture")) if results else {}
    return {
        "event_id": _text(first.get("event_id")),
        "league": _text(first.get("league")),
        "home_team": _text(first.get("home_team")),
        "away_team": _text(first.get("away_team")),
    }


def _lineup_state(results: Sequence[Mapping[str, Any]], stage: str) -> str:
    for result in results:
        context = _mapping(result.get("context"))
        quality = _mapping(context.get("data_quality"))
        lineup = _text(_mapping(quality.get("lineup")).get("state")).lower()
        if lineup:
            return lineup
    return "confirmed" if stage == "confirmed_lineups" else "pre_match"


def _fact_key(facts: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(facts), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float) -> float:
    return round(value, 4)


def _clean_copy(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_url(value: Any) -> bool:
    return bool(re.match(r"^https?://", str(value or "").strip(), re.IGNORECASE))
