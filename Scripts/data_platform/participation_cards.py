"""Strict player-total adapter; no substitution/lineup timeline reconstruction."""
from collections.abc import Mapping

from .settlement import count, provider_id


def parse_participation_cards(result, players):
    """Return separate weighted totals and compact auditable player evidence.

    Only FT player totals are usable: AET/PEN player statistics include extra time.
    Nulls, partial teams and contradictory card combinations never become zeroes.
    An explicit zero-card row cannot affect the total even if minutes are unknown.
    """
    evidence = {"period": "regulation_time" if result.get("status") == "FT" else "unknown",
                "source": "api_football_fixture_players", "players": [], "totals": {}}

    def pending(reason):
        evidence["pending_reason"] = reason
        evidence["totals"] = {}  # Never expose a partial total as settled evidence.
        return evidence

    if result.get("status") != "FT":
        return pending("regulation_player_statistics_unavailable")
    if not isinstance(players, list) or not players:
        return pending("missing_player_statistics")
    ids = [result.get("home_team_id"), result.get("away_team_id")]
    received = [provider_id((p.get("team") or {}).get("id"))
                if isinstance(p, Mapping) else None for p in players]
    if (None in ids or ids[0] == ids[1] or len(players) != 2
            or len(set(received)) != 2 or set(received) != set(ids)):
        return pending("player_statistics_team_identity_mismatch")
    seen = set()
    for team in players:
        team_id = provider_id(team["team"]["id"])
        side = "home" if team_id == ids[0] else "away"
        rows = team.get("players")
        if not isinstance(rows, list) or len(rows) < 11:
            return pending("incomplete_player_statistics")
        total, participants = 0, 0
        for row in rows:
            if not isinstance(row, Mapping):
                return pending("malformed_player_statistics")
            player_id = provider_id((row.get("player") or {}).get("id"))
            if player_id is None or player_id in seen:
                return pending("player_statistics_identity_mismatch")
            seen.add(player_id)
            stats = row.get("statistics")
            if not isinstance(stats, list) or len(stats) != 1 or not isinstance(stats[0], Mapping):
                return pending("malformed_player_statistics")
            games, cards = stats[0].get("games") or {}, stats[0].get("cards") or {}
            if not isinstance(games, Mapping) or not isinstance(cards, Mapping):
                return pending("malformed_player_statistics")
            minutes = count(games.get("minutes"))
            yellow, red = count(cards.get("yellow")), count(cards.get("red"))
            item = {"team_id": team_id, "player_id": player_id, "minutes": minutes,
                    "yellow": yellow, "red": red, "weighted_cards": None}
            evidence["players"].append(item)
            if minutes == 0:
                item.update(weighted_cards=0, reason="zero_minutes")
                continue
            if yellow is None or red is None:
                return pending("missing_player_card_counts")
            if yellow not in {0, 1, 2} or red not in {0, 1} or (yellow == 2 and red != 1):
                return pending("inconsistent_player_card_counts")
            if minutes is None:
                if yellow == red == 0:
                    item.update(weighted_cards=0, reason="explicit_zero_cards_minutes_unknown")
                    continue
                return pending("missing_carded_player_minutes")
            if minutes > 150:
                return pending("invalid_player_minutes")
            participants += 1
            # Both yellow + straight red and two yellows + dismissal total 3.
            weighted = min(yellow + 2 * red, 3)
            item.update(weighted_cards=weighted, reason="positive_minutes")
            total += weighted
        if participants < 7:
            return pending("incomplete_player_participation")
        evidence["totals"][side] = total
    evidence["pending_reason"] = None
    return evidence
