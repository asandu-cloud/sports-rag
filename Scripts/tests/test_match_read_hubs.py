"""Focused behaviour tests for public Discord Match Read hubs."""

from __future__ import annotations

from Scripts.discord_bot.match_read_hubs import (
    build_match_read_hub_embeds,
    discord_message_id_from_reference,
    hub_has_changed,
    select_effective_match_reads,
    visible_read_snapshot,
)


def _read(
    read_id: int,
    *,
    fixture_id: str = "fixture-1",
    stage: str = "pre_match",
    version: int = 1,
    status: str = "recommended",
    kickoff: str = "2026-09-06T15:00:00Z",
) -> dict:
    selections = []
    if status == "recommended":
        selections = [{
            "position": 1,
            "role": "core",
            "market": {"group": "goals", "key": "totals"},
            "pick": "Over 2.5",
            "quote_odds": 1.95,
            "bookmaker": "Book",
            "model_probability": 0.62,
            "value_edge": 0.107,
            "data": {
                "decision": {"confidence": "high"},
                "evidence": [],
            },
        }]
    return {
        "id": read_id,
        "fixture": {
            "event_id": fixture_id,
            "league": "EPL",
            "home_team": f"Home {fixture_id}",
            "away_team": f"Away {fixture_id}",
            "kickoff": kickoff,
        },
        "stage": stage,
        "version": version,
        "status": status,
        "thesis": f"{fixture_id}: a deliberately concise fixture thesis.",
        "game_script": {
            "tags": ["open_game"] if status == "recommended" else [status],
            "selection_relationship": (
                "These are compatible individual selections from the same fixture. "
                "They are not a combined-bet recommendation."
            ) if status == "recommended" else None,
            "alternative_candidates": [],
        },
        "selections": selections,
        "packages": [],
        "provenance": {"evaluated_at": f"2026-09-05T0{version}:00:00Z"},
    }


def test_effective_selector_prefers_confirmed_lineups_over_any_pre_match_version():
    pre_v1 = _read(1, version=1)
    pre_v2 = _read(2, version=2)
    confirmed_v1 = _read(3, stage="confirmed_lineups", version=1)
    other = _read(4, fixture_id="fixture-2", status="no_bet", kickoff="2026-09-06T12:00:00Z")

    selected = select_effective_match_reads([pre_v1, pre_v2, confirmed_v1, other])

    # Sorted by kickoff, then the fixture policy itself chooses the confirmed
    # lineup card even though its separate stage stream has a lower version.
    assert [item["id"] for item in selected] == [4, 3]


def test_hub_snapshot_detects_a_lineup_amendment_but_not_a_scheduler_restart():
    initial = [_read(1), _read(2, fixture_id="fixture-2")]
    prior_metadata = {"visible_match_reads": visible_read_snapshot(initial)}

    assert hub_has_changed(prior_metadata, initial) is False

    amended = [
        _read(3, stage="confirmed_lineups", version=1),
        _read(2, fixture_id="fixture-2"),
    ]
    assert hub_has_changed(prior_metadata, amended) is True


def test_renderer_keeps_no_bets_visible_and_disclaims_same_game_independence():
    recommended = _read(1)
    recommended["selections"].append({
        **recommended["selections"][0],
        "position": 2,
        "role": "supporting",
        "pick": "BTTS Yes",
        "market": {"group": "btts", "key": "btts"},
    })
    no_bet = _read(2, fixture_id="fixture-2", status="no_bet", kickoff="2026-09-06T19:00:00Z")

    embeds = build_match_read_hub_embeds(
        [recommended, no_bet],
        league="EPL",
        target_date="2026-09-06",
        color=0x2ECC71,
        updated=True,
    )

    assert len(embeds) == 1
    payload = embeds[0].to_dict()
    assert payload["title"] == "📋 Matchday Hub — EPL"
    assert "Updated" in payload["description"]
    fields = payload["fields"]
    assert len(fields) == 2
    assert "Recommended" in fields[0]["name"]
    assert "Individual angles only" in fields[0]["value"]
    assert "No bet" in fields[1]["name"]
    assert "no current price qualified" in fields[1]["value"]


def test_discord_reference_parser_rejects_the_wrong_channel_or_bad_ids():
    assert discord_message_id_from_reference("discord:10:20:30", channel_id=20) == 30
    assert discord_message_id_from_reference("discord:10:20:30", channel_id=21) is None
    assert discord_message_id_from_reference("discord:10:20:not-a-message") is None
    assert discord_message_id_from_reference("not-a-reference") is None
