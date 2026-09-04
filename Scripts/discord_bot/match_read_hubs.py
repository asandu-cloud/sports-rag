"""Discord presentation helpers for fixture-level Match Read hubs.

This module is deliberately a delivery adapter, not a second prediction
engine.  It accepts persisted Match Read records, chooses the effective
version for each fixture, and renders compact Discord embeds.  It never
fetches odds, recompiles a read, or records a tracked recommendation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import discord

from data_platform.services.match_read_cards import (
    MatchReadCardError,
    build_match_read_card,
)


# Discord permits at most ten embeds per message.  Three fixture cards per
# embed gives a domestic-league hub plenty of room while keeping each card
# legible and comfortably below the 6,000-character embed limit.
_FIXTURES_PER_EMBED = 3
_MAX_EMBEDS_PER_MESSAGE = 10
_MAX_VISIBLE_FIXTURES = _FIXTURES_PER_EMBED * _MAX_EMBEDS_PER_MESSAGE
_MAX_FIELD_VALUE = 950

_STATUS_LABELS = {
    "recommended": ("✅ Recommended", "green"),
    "no_bet": ("⏸ No bet", "yellow"),
    "unavailable": ("⚪ Unavailable", "grey"),
}

_ROLE_LABELS = {
    "core": "Core",
    "supporting": "Supporting",
    "alternative": "Alternative",
}


def select_effective_match_reads(reads: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Choose one persisted read per fixture for a delivery surface.

    The explicit product policy is: latest ``confirmed_lineups`` version if
    one exists, otherwise latest ``pre_match`` version.  The immutable source
    rows remain untouched, so the caller can still retain a full audit trail
    of the earlier pre-match card.
    """
    by_fixture: Dict[str, list[Dict[str, Any]]] = {}
    for raw in reads:
        if not isinstance(raw, Mapping):
            continue
        read = dict(raw)
        fixture = read.get("fixture")
        if not isinstance(fixture, Mapping):
            continue
        event_id = str(fixture.get("event_id") or "").strip()
        if not event_id:
            continue
        by_fixture.setdefault(event_id, []).append(read)

    selected: list[Dict[str, Any]] = []
    for fixture_reads in by_fixture.values():
        confirmed = [
            read for read in fixture_reads
            if str(read.get("stage") or "").strip().lower() == "confirmed_lineups"
        ]
        pre_match = [
            read for read in fixture_reads
            if str(read.get("stage") or "").strip().lower() == "pre_match"
        ]
        candidates = confirmed or pre_match
        if not candidates:
            # The persistence contract currently permits only the two stages,
            # but retaining a malformed/legacy row is safer than silently
            # publishing it as an effective Match Read.
            continue
        selected.append(max(candidates, key=_revision_key))
    return sorted(selected, key=_fixture_sort_key)


def select_current_match_read_hub_reads(
    reads: Iterable[Mapping[str, Any]],
    *,
    now: Optional[Any] = None,
    max_age_minutes: int = 120,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Return only fixture cards still safe to show in a public hub.

    This is intentionally separate from rendering.  A Discord message can be
    edited or fetched long after it was first sent, so stored deliveries alone
    must never be treated as permission to publish a started fixture or a
    stale price.  The shared platform policy also keeps the website and hub
    on the same amendment precedence rules.
    """
    from data_platform.services.match_read_release import select_releasable_match_reads

    return select_releasable_match_reads(
        reads,
        now=now,
        max_age_minutes=max_age_minutes,
    )


def visible_read_snapshot(reads: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Return the stable, small identity set stored beside a hub delivery."""
    rows = []
    for read in reads:
        fixture = read.get("fixture") if isinstance(read, Mapping) else None
        if not isinstance(fixture, Mapping):
            continue
        try:
            read_id = int(read.get("id"))
            version = int(read.get("version"))
        except (TypeError, ValueError):
            continue
        event_id = str(fixture.get("event_id") or "").strip()
        stage = str(read.get("stage") or "").strip().lower()
        if not event_id or not stage:
            continue
        rows.append({
            "event_id": event_id,
            "id": read_id,
            "stage": stage,
            "version": version,
        })
    return sorted(rows, key=lambda item: (item["event_id"], item["stage"], item["version"], item["id"]))


def hub_has_changed(previous_metadata: Optional[Mapping[str, Any]], reads: Iterable[Mapping[str, Any]]) -> bool:
    """Whether a recovered hub message represents a different visible slate."""
    if not isinstance(previous_metadata, Mapping):
        return False
    previous = previous_metadata.get("visible_match_reads")
    if not isinstance(previous, Sequence) or isinstance(previous, (str, bytes)):
        return False
    normalised_previous = []
    for row in previous:
        if not isinstance(row, Mapping):
            continue
        try:
            normalised_previous.append({
                "event_id": str(row.get("event_id") or "").strip(),
                "id": int(row.get("id")),
                "stage": str(row.get("stage") or "").strip().lower(),
                "version": int(row.get("version")),
            })
        except (TypeError, ValueError):
            continue
    normalised_previous = [row for row in normalised_previous if row["event_id"] and row["stage"]]
    normalised_previous.sort(key=lambda item: (item["event_id"], item["stage"], item["version"], item["id"]))
    return normalised_previous != visible_read_snapshot(reads)


def has_confirmed_lineup_read(reads: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        str(read.get("stage") or "").strip().lower() == "confirmed_lineups"
        for read in reads
        if isinstance(read, Mapping)
    )


def build_match_read_hub_embeds(
    reads: Iterable[Mapping[str, Any]],
    *,
    league: str,
    target_date: str,
    color: int,
    updated: bool = False,
    rendered_at: Optional[datetime] = None,
) -> list[discord.Embed]:
    """Build one Discord message's worth of compact fixture cards.

    A hub makes every effective fixture visible, including intentional no-bet
    and unavailable states.  Alternatives remain in the persisted read and
    website/detail surface, rather than being promoted to extra public picks
    in the compact community hub.
    """
    effective = list(reads)
    if len(effective) > _MAX_VISIBLE_FIXTURES:
        raise ValueError(
            f"A Discord Match Read hub can show at most {_MAX_VISIBLE_FIXTURES} fixtures in one message."
        )
    now = _as_utc(rendered_at or datetime.now(timezone.utc))
    confirmed = has_confirmed_lineup_read(effective)
    fields = [_fixture_field(read) for read in effective]
    if not fields:
        fields = [(
            "No Match Reads available",
            "No persisted fixture read is available for this league and matchday yet.",
        )]

    pages = [fields[index:index + _FIXTURES_PER_EMBED]
             for index in range(0, len(fields), _FIXTURES_PER_EMBED)]
    embeds: list[discord.Embed] = []
    for page_index, fields_on_page in enumerate(pages, start=1):
        first_page = page_index == 1
        title = (
            f"📋 Matchday Hub — {league}"
            if first_page
            else f"📋 {league} Matchday Hub — continued"
        )
        description = _hub_description(
            target_date=target_date,
            updated=updated,
            confirmed=confirmed,
            rendered_at=now,
        ) if first_page else "One fixture-level read per card."
        embed = discord.Embed(title=title, description=description, color=color)
        for name, value in fields_on_page:
            embed.add_field(name=_truncate(name, 256), value=_truncate(value, _MAX_FIELD_VALUE), inline=False)
        embed.set_footer(
            text=(
                f"Spix's Picks · {len(effective)} fixture read(s) · "
                f"page {page_index}/{len(pages)}"
            )
        )
        embeds.append(embed)
    return embeds


def discord_message_reference(message: Any, channel: Any) -> str:
    """Build the stable external reference shared by each card in one hub."""
    guild = getattr(channel, "guild", None)
    guild_id = getattr(guild, "id", None) or "unknown"
    channel_id = getattr(channel, "id", None) or "unknown"
    message_id = getattr(message, "id", None)
    if message_id is None:
        raise ValueError("A delivered Discord hub needs a message id.")
    return f"discord:{guild_id}:{channel_id}:{message_id}"


def discord_message_id_from_reference(reference: Any, *, channel_id: Any = None) -> Optional[int]:
    """Extract a message ID from this project's Discord delivery reference."""
    parts = str(reference or "").split(":")
    if len(parts) != 4 or parts[0] != "discord":
        return None
    if channel_id is not None and str(parts[2]) != str(channel_id):
        return None
    try:
        return int(parts[3])
    except (TypeError, ValueError):
        return None


def _revision_key(read: Mapping[str, Any]) -> tuple[int, datetime, int]:
    try:
        version = int(read.get("version"))
    except (TypeError, ValueError):
        version = -1
    provenance = read.get("provenance") if isinstance(read.get("provenance"), Mapping) else {}
    evaluated = _parse_datetime(provenance.get("evaluated_at")) or datetime.min.replace(tzinfo=timezone.utc)
    try:
        read_id = int(read.get("id"))
    except (TypeError, ValueError):
        read_id = -1
    return (version, evaluated, read_id)


def _fixture_sort_key(read: Mapping[str, Any]) -> tuple[datetime, str]:
    fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
    kickoff = _parse_datetime(fixture.get("kickoff")) or datetime.max.replace(tzinfo=timezone.utc)
    return (kickoff, str(fixture.get("event_id") or ""))


def _fixture_field(read: Mapping[str, Any]) -> tuple[str, str]:
    try:
        card = build_match_read_card(read)
    except MatchReadCardError as exc:
        fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
        name = f"⚪ {fixture.get('home_team') or 'Unknown'} vs {fixture.get('away_team') or 'Unknown'}"
        return name, f"This persisted read could not be rendered safely: {exc}"

    fixture = card["fixture"]
    status = str(card["status"]).strip().lower()
    status_label, _color = _STATUS_LABELS.get(status, ("⚪ Unavailable", "grey"))
    kickoff = _kickoff_label(fixture.get("kickoff"))
    name = f"{status_label} · {fixture['home_team']} vs {fixture['away_team']}"
    if kickoff:
        name += f" · {kickoff}"

    stage = str(card.get("stage") or "").strip().lower()
    stage_label = "🔄 Confirmed lineups" if stage == "confirmed_lineups" else "🕒 Pre-match"
    lines = [stage_label, card["thesis"]]
    selections = card.get("selections") or []
    if selections:
        lines.extend(["", "**Selections**"])
        for selection in selections:
            lines.append(_selection_line(selection))
        relationship = card.get("game_script", {}).get("selection_relationship")
        if relationship and len(selections) > 1:
            lines.extend(["", "_Individual angles only — not a combined-bet recommendation._"])
    elif status == "no_bet":
        lines.extend(["", "_Assessed, but no current price qualified for release._"])
    else:
        lines.extend(["", "_Held back until the required data and current prices are available._"])
    return name, "\n".join(lines)


def _selection_line(selection: Mapping[str, Any]) -> str:
    role = _ROLE_LABELS.get(str(selection.get("role") or "").lower(), "Selection")
    pick = str(selection.get("pick") or "Selection")
    odds = _number(selection.get("odds"))
    odds_text = f" @ **{odds:.2f}**" if odds is not None else ""
    details = []
    confidence = str(selection.get("confidence") or "").strip()
    if confidence:
        details.append(confidence.title())
    probability = _number(selection.get("model_probability"))
    if probability is not None:
        details.append(f"{probability:.0%} model")
    edge = _number(selection.get("value_edge"))
    if edge is not None:
        details.append(f"{edge:+.1%} edge")
    suffix = f" · {' · '.join(details)}" if details else ""
    return f"• **{role}** — **{pick}**{odds_text}{suffix}"


def _hub_description(
    *,
    target_date: str,
    updated: bool,
    confirmed: bool,
    rendered_at: datetime,
) -> str:
    timestamp = f"<t:{int(rendered_at.timestamp())}:R>"
    if updated and confirmed:
        state = "🔄 **Updated** — confirmed-lineup reads now replace pre-match reads where available."
    elif updated:
        state = "🔄 **Updated** — the visible fixture reads changed."
    elif confirmed:
        state = "✅ Confirmed-lineup reads are shown where available."
    else:
        state = "🕒 Pre-match reads. A future confirmed-lineup update will edit this hub."
    return (
        f"**{target_date}** · {timestamp}\n{state}\n\n"
        "Each selection is an individual fixture angle; no combined odds, "
        "probability, or expected value is implied."
    )


def _kickoff_label(value: Any) -> Optional[str]:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return f"<t:{int(parsed.timestamp())}:t>"
    text = str(value or "").strip()
    return text or None


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
