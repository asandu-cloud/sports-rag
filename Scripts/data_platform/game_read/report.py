"""Game-read bench (D77): write a run's ``summary.json`` and a plain ``report.md`` for the owner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

TITLES = {"tuning": "tuning seasons (2019/20–2022/23)", "locked": "locked seasons (2023/24 on)"}
RULES = {"edge": "Ranked by edge", "blend": "Ranked by blended chance"}
MARKETS = {"result": "result", "handicap": "handicap", "over25": "over 2.5"}
LEADS_SHOWN = 15


def _plain(value):
    return float(value) if hasattr(value, "__float__") else str(value)


def _num(value, digits=4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _points(value) -> str:
    """A chance or a share as percentage points."""
    return "—" if value is None else f"{100 * value:+.2f}"


def _per_100(value) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _count(value) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _interval(found: Mapping, show=_num) -> str:
    return "—" if found.get("low") is None else f"[{show(found['low'])}, {show(found['high'])}]"


def _table(header, rows) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows), ""]


def _yes(passed) -> str:
    return "pass" if passed else "fail"


def _zero_line(results) -> list:
    zero = results["zero_line"]
    lines = ["## The zero line", "",
             "A model that copies the market must score zero on every score. If it does not, the bench is wrong and "
             "the run stops before writing anything.", ""]
    lines += _table(["Check", "Largest gap", "Result"],
                    [(name, f"{check['value']:g}", _yes(check["passed"])) for name, check in zero["checks"].items()])
    return lines + [f"The zero line {'passed' if zero['passed'] else 'failed'}.", ""]


def _accuracy(models) -> list:
    rows = [(name, MARKETS[market], _num(found["model_log_loss"]), _num(found["market_log_loss"]),
             _num(found["estimate"]), _interval(found), _count(found["n"]))
            for name, scored in models.items() for market, found in scored["accuracy"].items()]
    return ["## Accuracy", "", "Log loss of the model minus the market's, on the lines history prices. Below zero, "
            "the model was more accurate than the market.", "",
            *_table(["Model", "Market", "Model's log loss", "Market's", "Difference", "Interval", "Matches"], rows)]


def _blend(models) -> list:
    rows = [(name, MARKETS[market], _num(found["c"], 3), _num(found["estimate"], 5), _interval(found, lambda v:
             _num(v, 5)), _count(found["n"])) for name, scored in models.items()
            for market, found in scored["blend"].items()]
    return ["## Adds to the market (blend)", "",
            "logit p = a + b·logit(market) + c·logit(model), fitted on the tuning seasons. The model adds to the "
            "market when c is above zero and the blend's log loss is below the market-only refit's, with the whole "
            "interval below zero.", "", *_table(["Model", "Market", "c", "Difference", "Interval", "Matches"], rows)]


def _picks(models) -> list:
    lines = ["## Most likely picks", "",
             "Wins minus the market's expected wins, per 100 picks. Today's lists use the live rule. The engine's put "
             "the options the model backs first. The difference is the engine's minus today's, on the same matches; "
             "real and derived lines are shown apart.", ""]
    for rule, title in RULES.items():
        rows = [(name, _count(found["engine"]["picks"]), _per_100(found["today"]["per_100"]),
                 _per_100(found["engine"]["per_100"]), _per_100(found["estimate"]), _interval(found, _per_100),
                 _per_100(found["real"]["estimate"]), _per_100(found["derived"]["estimate"]))
                for name, scored in models.items() for found in [scored["picks"][rule]]]
        lines += [f"### {title}", "", *_table(["Model", "Picks", "Today", "Engine", "Difference", "Interval",
                                               "Real lines", "Derived lines"], rows)]
    return lines


def _value(models) -> list:
    rows = []
    for name, scored in models.items():
        for price, found in scored["value"].items():
            today, engine = found["today"], found["engine"]
            rows.append((name, {"avg": "average", "max": "best"}[price], _count(today["picks"]),
                         _points(today["return"]), _points(today["clv"]), _count(engine["picks"]),
                         _points(engine["return"]), _points(engine["clv"]), _interval(found, _points)))
    return ["## Value", "",
            "A pick needs chance × price − 1 of at least 2%; at most 2 a match. Return is per unit staked and closing "
            "line value is price × the closing fair chance − 1, both in %. The interval is the engine's closing "
            "line value.", "",
            *_table(["Model", "Price", "Today: picks", "Return", "CLV", "Engine: picks", "Return", "CLV",
                     "Interval"], rows)]


def _closing(models) -> list:
    rows = [(name, RULES[rule].split(" ", 2)[-1], _points(found["today"]), _points(found["engine"]),
             _points(found["estimate"]), _interval(found, _points))
            for name, scored in models.items() for rule, found in scored["closing"].items()]
    return ["## The closing check", "",
            "The mean move of the fair chance from pick time to the close, in points: the engine's Most likely "
            "picks minus today's. Above zero, closing prices moved towards the engine's picks.", "",
            *_table(["Model", "Ranked by", "Today", "Engine", "Difference", "Interval"], rows)]


def _leagues(models) -> list:
    leagues = sorted({league for scored in models.values() for league in scored["picks"]["edge"]["by_league"]})
    rows = [(league, *(_per_100((scored["picks"]["edge"]["by_league"].get(league) or {}).get("estimate"))
                       for scored in models.values())) for league in leagues]
    return ["## Leagues", "",
            "Most likely difference per league (ranked by edge), engine minus today, per 100 picks. A model fails if "
            "any league is worse than −2.", "", *_table(["League", *models], rows)]


def _verdicts(verdicts) -> list:
    rows = []
    for rule, found in verdicts.items():
        for name, verdict in found.items():
            blend = verdict.get("blend") or {}
            rows.append((name, RULES[rule].split(" ", 2)[-1], *(_yes(blend.get(m)) for m in MARKETS),
                         _yes(verdict.get("picks")), _yes(verdict.get("value"))))
    return ["## Verdicts", "",
            "Pass or fail by the spec's rules; they count on the locked seasons. The zero and control lines are "
            "references and are expected to fail. Value is judged at the average price, on at least 300 picks.", "",
            *_table(["Model", "Ranked by", "Blend: result", "Blend: handicap", "Blend: over 2.5", "Most likely",
                     "Value"], rows)]


def _control(models) -> list:
    control = models.get("control")
    if not control:
        return ["## The control line", "", "Not run.", ""]
    lines = ["## The control line", "",
             "M1's goal means with Dixon–Coles's low-score correction. M1 is a plain Poisson model and is known to be "
             "off on low scores, so on the derived lines a model must beat the control line, not zero.", ""]
    for rule, title in RULES.items():
        picks = control["picks"][rule]
        lines.append(f"- {title}: Most likely on derived lines, engine minus today, "
                     f"{_per_100(picks['derived']['estimate'])} per 100 picks (the bar on derived lines); on all "
                     f"lines {_per_100(picks['estimate'])} {_interval(picks, _per_100)}.")
    return lines + [f"- Result log loss against the market's: {_num(control['accuracy']['result']['estimate'])} "
                    f"{_interval(control['accuracy']['result'])}.", ""]


def _average(models) -> list:
    weights = models.get("average", {}).get("weights")
    if not weights:
        return []
    names = [name for name in ("elo", "pi", "dc", "drift") if any(name in part for part in weights.values())]
    titles = {"result": "Result", "handicap": "Goal difference (handicaps)", "goals": "Score matrix (goal lines, both "
              "teams to score)"}
    rows = [[titles[part], *(f"{weights[part][name]:.2f}" if name in weights[part] else "—" for name in names)]
            for part in ("result", "handicap", "goals")]
    return ["## The average's weights", "",
            "A weighted average of the members on the log scale; each part's weights are never negative, sum to 1 "
            "and are fitted on the tuning seasons only. A member without a score matrix takes no part in it. Its scores "
            "on the tuning seasons are therefore in-sample; the locked seasons are its fair test.", "",
            *_table(["Part", *names], rows)]


def _reference(reference) -> list:
    rows = [(season, _points(found["mean"]).lstrip("+"), _points(found["p95"]).lstrip("+"), _count(found["n"]))
            for season, found in reference.items()]
    return ["## The market against the named books", "",
            "The largest gap, over the three result outcomes, between the average price's fair chance and the median "
            "of the named books' (matches with 3 books or more), in points.", "",
            *_table(["Season", "Mean", "95th percentile", "Matches"], rows)]


def _opportunities(found) -> list:
    chance_by = max(1, round(found["groups"] / 20))
    lines = ["## Opportunities", "",
             "Found on the tuning seasons only; nothing here changes a pick. Anything worth following becomes a "
             f"candidate for the bench. {found['groups']} groups were measured: about {chance_by} would leave out zero "
             "by chance alone.", "", "### Leads", ""]
    lines += [f"- {lead['what']}: {_points(lead['estimate'])} [{_points(lead['low'])}, {_points(lead['high'])}], "
              f"{lead['n']:,} matches" for lead in found["leads"][:LEADS_SHOWN]] or ["- None."]
    rows = [(market, group, _points(stat["estimate"]), _interval(stat, _points), _count(stat["n"]))
            for market, node in found["drift"].items()
            for group, stat in [("all", node["all"]), *node["by_league"].items(), *node["by_weekday"].items()]]
    lines += ["", "### Drift to the close", "", "The fair chance at the close minus at pick time, in points. Above "
              "zero, the side was underpriced at pick time.", "",
              *_table(["Side", "Group", "Move", "Interval", "Matches"], rows)]
    rows = [(band, _points(stat["said"]).lstrip("+"), _points(stat["won"]).lstrip("+"), _points(stat["estimate"]),
             _interval(stat, _points), _points(stat["return"]["estimate"]), _interval(stat["return"], _points),
             _count(stat["n"])) for band, stat in found["calibration"].items()]
    lines += ["### Price bands", "", "Each result outcome by its average price: the chance the market said, how often "
              "it won, the gap, and the return of one unit at the average price (%).", "",
              *_table(["Price", "Said", "Won", "Gap", "Interval", "Return", "Interval", "Outcomes"], rows)]
    promoted = found["promoted"]
    lines += ["### Promoted teams", "",
              f"{promoted['teams']} teams new to their league; their first 10 matches: won minus the market's chance "
              f"{_points(promoted['estimate'])} {_interval(promoted, _points)} points, return at the average price "
              f"{_points(promoted['return']['estimate'])}% {_interval(promoted['return'], _points)}, "
              f"{promoted['n']:,} matches.", ""]
    return lines


def _report(results) -> list:
    run, models = results["run"], results["models"]
    leagues = "all" if run["leagues"] == "all" else ", ".join(run["leagues"])
    return [f"# Game-read bench: {TITLES[run['block']]}", "",
            f"Run {run['time']}, commit {run['commit'][:7]}. {run['matches_scored']:,} matches scored; leagues: "
            f"{leagues}. Models: {', '.join(run['models'])}.", "",
            "Every interval is 95%, from 2,000 resamples of whole weeks.", "",
            *_zero_line(results), *_accuracy(models), *_blend(models), *_picks(models), *_value(models),
            *_closing(models), *_leagues(models), *_verdicts(results["verdicts"]), *_control(models),
            *_average(models), *_reference(results["reference"]), *_opportunities(results["opportunities"])]


def write(out_dir, results: Mapping) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=_plain) + "\n")
    (out / "report.md").write_text("\n".join(_report(results)) + "\n")

