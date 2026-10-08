"""Research-only interpretation of count nulls; canonical rows stay unchanged."""

COUNT_NULL_POLICY = "observed-match-count-zero.v2"
# v3: comparable-round scope, team match minimum, position-aware headlines.
RANK_BUILD_PREFIX = "profiles-v3:"

PLAYER_COUNT_FIELDS = (
    "goals", "assists", "shots_total", "shots_on", "passes_total", "passes_accurate",
    "tackles", "interceptions", "duels_won", "duels_total", "yellow_cards", "red_cards",
    "fouls_committed", "fouls_drawn",
)
PLAYER_MEASUREMENT_FIELDS = ("rating", "pass_accuracy")
TEAM_COUNT_FIELDS = (
    "goals", "shots_total", "shots_on", "shots_off", "shots_blocked", "shots_inside_box",
    "shots_outside_box", "fouls_committed", "corners", "offsides", "yellow_cards",
    "red_cards", "goalkeeper_saves", "passes_total", "passes_accurate",
)
TEAM_MEASUREMENT_FIELDS = ("possession", "pass_accuracy", "expected_goals", "goals_prevented")
TEAM_OBSERVATION_FIELDS = ("shots_total", "shots_on", "corners", "fouls_committed",
                           "yellow_cards", "red_cards", "possession", "passes_total")


def normalize_research_row(kind, row):
    """Played players are observed; team stats require BOTH non-empty sides.

    Only the eight declared observation fields establish team availability.
    Fixture results stay separate from this gate. Missing status in pure
    calculation inputs means the caller already filtered finished fixtures.
    """
    if row.get("_count_null_policy") == COUNT_NULL_POLICY:
        return row
    result = dict(row)
    if kind == "player":
        counts, measurements = PLAYER_COUNT_FIELDS, PLAYER_MEASUREMENT_FIELDS
        sides = (("subject", ""),)
        eligible = (row.get("minutes") or 0) > 0
    else:
        counts, measurements = TEAM_COUNT_FIELDS, TEAM_MEASUREMENT_FIELDS
        sides = (("subject", ""), ("opponent", "opponent_")) if kind == "team" else (("home", "home_stats_"), ("away", "away_stats_"))
        eligible = True
    eligible = eligible and row.get("status", "FT") in ("FT", "AET", "PEN")
    availability, filled = {}, {}
    for side, prefix in sides:
        availability[side] = eligible and (kind == "player" or any(
            row.get(prefix + field) is not None for field in TEAM_OBSERVATION_FIELDS))
    observed = all(availability.values())
    for side, prefix in sides:
        filled[side] = []
        for field in counts:
            key = prefix + field
            if observed and row.get(key) is None:
                result[key] = 0
                filled[side].append(field)
            elif not observed:
                result[key] = None
        if not observed and kind != "player":
            for field in measurements:
                result[prefix + field] = None
    if kind == "team":
        is_home = row.get("team_id") == row.get("home_team_id") if "team_id" in row else row.get("is_home", True)
        result["result_goals_for"] = row.get("home_goals" if is_home else "away_goals")
        result["result_goals_against"] = row.get("away_goals" if is_home else "home_goals")
    result["_stats_available"] = availability
    result["_stats_observed"] = observed
    result["_zero_filled_counts"] = filled
    result["_count_null_policy"] = COUNT_NULL_POLICY
    return result
