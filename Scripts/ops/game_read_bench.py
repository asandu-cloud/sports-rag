"""Game-read bench (D77): score models day by day on football-data's history and write the report.

    python -m Scripts.ops.game_read_bench run --models market,control --block tuning \\
        --out Research/game-read-bench-<date>
    python -m Scripts.ops.game_read_bench export-stats   # once, outside refresh hours (phase 2)

Every run scores the zero line (``market``) and the control line (``control``) first. When the zero line does not
score zero, the bench is wrong: the run stops with exit code 2 and writes nothing.

The locked seasons (2023/24 to 2025/26) and the fresh block (2026/27 on) are scored only when the run is named
(``--confirm-locked``) and the bench code is committed. A locked run never sees the fresh block. Every locked or fresh
run is appended to ``Research/game-read-bench/locked-runs.jsonl`` and goes into the tracker: each model version runs
on each of them once, and is never rerun after its result is seen.
Nothing here touches the live site or ``Index/platform.db``; football-data's file is opened read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from Scripts.data_platform.game_read import history, opportunities, report, rules, scores, stats, walkforward
from Scripts.data_platform.game_read.models import average
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes
from Scripts.data_platform.game_read.models.drifting import Drifting
from Scripts.data_platform.game_read.models.elo import Elo
from Scripts.data_platform.game_read.features import Features
from Scripts.data_platform.game_read.models.lean import LeanAway
from Scripts.data_platform.game_read.models.trees import GRID, Trees
from Scripts.data_platform.game_read.models.market import MarketControl, MarketCopy
from Scripts.data_platform.game_read.models.pi_ratings import PiRatings
from Scripts.data_platform.game_read.models.rest import RestVariable
from Scripts.data_platform.game_read.models.shots import KINDS, ShotVariable
from Scripts.data_platform.services.refresh_coordination import RefreshBusy

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "Research" / "game-read-bench" / "locked-runs.jsonl"
BENCH_CODE = ("Scripts/data_platform/game_read", "Scripts/ops/game_read_bench.py")
REFERENCES = ("market", "control")
MODELS = {"market": MarketCopy, "control": MarketControl, "elo": Elo, "pi": PiRatings, "dc": DixonColes,
          "drift": Drifting}
# Phase 2 variants: models 3–4 with variables, named "<model>+<variable>+…" ("dc+box+rest"); each is gated against
# the same name without its last variable.
WITH_VARIABLES = {"dc": DixonColes, "drift": Drifting}
VARIABLES = {**{kind: (lambda kind=kind: ShotVariable(kind, _stats())) for kind in KINDS},
             "rest": lambda: RestVariable(_stats())}
ABLATED = ("result", "over25")
# Model 5b: "lean-<member>" blends the market with one member ("lean-dc+xg"), or with the members' average
# ("lean-average" pools elo, pi, dc and drift; "lean-average+xg" gives dc and drift the variable).
LEAN = "lean-"
# Model 6: "trees-error[-nolead]@<leaves>-<minimum leaf>-<L2>[+xg]" ("trees-error@7-200-1+xg"); a name without "@"
# stands for the whole tuning grid. "+xg" gives dc and drift the derived xG and adds it to the inputs, and is gated
# against the same name without it. Every trees model of a run with the same inputs reads one shared features pass.
TREES = "trees-"
TREES_VARIABLES = ("box", "target", "rest")
_PASSES = {}
# Variables kept by the owner's choice although they missed the gate; every report that uses one says so.
CARRIED = {"xg": "the derived shot-location xG missed the 0.5% gate in phase 2 task 3 (−0.39% on the result, "
                 "[−0.51, −0.26]); the owner carried it forward on 17 Nov"}
AVERAGE = "average"  # built from its members' rows in the run, with weights fitted on the tuning seasons only
ZERO_BLEND = 1e-6


def _load(leagues):
    return history.load_matches(competitions=leagues or None)


@lru_cache(maxsize=1)
def _stats():
    return stats.load()


def _lean(target: str):
    model, *variables = target.split("+")
    if model == "average":
        if any(v not in VARIABLES for v in variables):
            return None
        members = ["+".join([m, *variables]) if m in WITH_VARIABLES else m for m in average.MEMBERS]
    elif target in REFERENCES or target.startswith(LEAN) or _factory(target) is None:
        return None
    else:
        members = [target]
    return lambda: LeanAway(target, {member: _factory(member)() for member in members})


def _expand(names) -> list:
    """Each trees name without its settings becomes the whole grid, in order."""
    found = []
    for name in names:
        if name.startswith(TREES) and "@" not in name:
            model, *variables = name.split("+")
            found += ["+".join([f"{model}@{'-'.join(map(str, values))}", *variables])
                      for values in itertools.product(*GRID.values())]
        else:
            found.append(name)
    return found


def _features(carried: bool) -> Features:
    """The run's one features pass for these inputs: the four members (dc and drift with the derived xG when
    ``carried``) and the shot and rest variables."""
    if carried not in _PASSES:
        names = [f"{m}+xg" if carried and m in WITH_VARIABLES else m for m in average.MEMBERS]
        variables = [*TREES_VARIABLES, *(("xg",) if carried else ())]
        _PASSES[carried] = Features({m: _factory(m)() for m in names}, [VARIABLES[v]() for v in variables])
    return _PASSES[carried]


def _trees(name: str):
    model, _, rest = name.partition("@")
    settings, *variables = rest.split("+")
    values = settings.split("-")
    on = model[len(TREES):].removesuffix("-nolead")
    if on not in ("error", "outcome") or variables not in ([], ["xg"]) or len(values) != 3:
        return None
    try:
        params = dict(zip(GRID, map(int, values)))
    except ValueError:
        return None
    if any(params[key] not in GRID[key] for key in GRID):
        return None
    return lambda: Trees(on, _features(bool(variables)), params, lead=not model.endswith("-nolead"), name=name)


def _factory(name: str):
    """What builds the model of this name, or None for an unknown name."""
    if name in MODELS:
        return MODELS[name]
    if name.startswith(LEAN):
        return _lean(name[len(LEAN):])
    if name.startswith(TREES):
        return _trees(name)
    model, *variables = name.split("+")
    if model not in WITH_VARIABLES or not variables or any(v not in VARIABLES for v in variables):
        return None
    return lambda: WITH_VARIABLES[model](variables=tuple(VARIABLES[v]() for v in variables))


def _with_bases(names) -> list:
    """Each variant preceded by the model it is gated against ("dc+box+rest" after "dc+box", after "dc")."""
    found = []
    for name in names:
        parts = name.split("+")
        for size in range(1, len(parts) + 1):
            if (base := "+".join(parts[:size])) not in found:
                found.append(base)
    return found


def _ablation(base: str, base_rows, rows, blocks) -> dict:
    """The variant against its base on each block, per market, and whether it passes on the tuning seasons: the
    gate on the result or over 2.5, without the other's interval above zero."""
    found = {"base": base}
    for block in blocks:
        pick = lambda pairs: [row for row in pairs if history.block(row[0].season) == block]
        found[block] = {market: {**(a := scores.ablation(pick(base_rows), pick(rows), market)), "gate": scores.gate(a)}
                        for market in ABLATED}
    tuning = found["tuning"]
    found["passed"] = any(tuning[market]["gate"] and not ((tuning[other]["low"] or 0) > 0)
                          for market, other in (ABLATED, ABLATED[::-1]))
    return found


def _named():
    return history.named_result_prices()


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _bench_code_changes() -> str:
    return _git("status", "--porcelain", "--", *BENCH_CODE)


def _commit() -> str:
    return _git("rev-parse", "HEAD")


def _score(rows, tuning_rows) -> dict:
    fitted = scores.option_blend(tuning_rows)
    strengths = {"edge": rules.by_edge, "blend": rules.by_blend(fitted) if fitted else rules.by_edge}
    return {"accuracy": {market: scores.accuracy(rows, market) for market in scores.MARKETS},
            "blend": {market: scores.blend_score(tuning_rows, rows, market) for market in scores.MARKETS},
            "picks": {rule: scores.picks_score(rows, strength=s) for rule, s in strengths.items()},
            "value": {price: scores.value_score(rows, price=price) for price in ("avg", "max")},
            "closing": {rule: scores.closing_check(rows, strength=s) for rule, s in strengths.items()},
            "option_blend": None if fitted is None else {k: fitted[k] for k in ("b", "c", "n")}}


def _largest(values) -> float:
    return max((abs(v) for v in values if v is not None), default=0.0)


def zero_line(found) -> dict:
    """The zero line's checks (Task 6's tolerances): accuracy and Most likely exactly zero, the blend within 1e-6,
    and no Value pick at the average price."""
    checks = {
        "accuracy": _largest(f["estimate"] for f in found["accuracy"].values()),
        "blend": _largest(f["estimate"] for f in found["blend"].values()),
        "most likely": _largest(v for f in found["picks"].values()
                                for v in (f["estimate"], f["real"]["estimate"], f["derived"]["estimate"])),
        "value at the average price (engine picks)": found["value"]["avg"]["engine"]["picks"],
    }
    limits = {"blend": ZERO_BLEND}
    checked = {name: {"value": value, "passed": value <= limits.get(name, 0)} for name, value in checks.items()}
    return {"checks": checked, "passed": all(c["passed"] for c in checked.values())}


def _headline(found) -> dict:
    picks, value = found["picks"]["edge"], found["value"]["avg"]
    return {"most_likely_edge": {k: picks[k] for k in ("estimate", "low", "high")},
            "derived_edge": picks["derived"]["estimate"],
            "value_avg": {"picks": value["engine"]["picks"], "return": value["engine"]["return"],
                          "clv": value["estimate"], "low": value["low"], "high": value["high"]},
            "blend_result": {k: found["blend"]["result"][k] for k in ("c", "estimate", "low", "high")},
            "accuracy_result": found["accuracy"]["result"]["estimate"]}


def run(args) -> int:
    locked = args.block in history.GUARDED
    if locked and not args.confirm_locked:
        print(f"A {args.block} run must be named: add --confirm-locked. Each model version runs on the {args.block} "
              "seasons once.", file=sys.stderr)
        return 2
    if locked and (changes := _bench_code_changes()):
        print(f"The bench code has uncommitted changes; a {args.block} run needs committed code:\n{changes}",
              file=sys.stderr)
        return 2
    _PASSES.clear()
    names = [*REFERENCES, *_with_bases(_expand(name for name in args.models if name not in (*REFERENCES, AVERAGE)))]
    if AVERAGE in args.models:
        names += [*(name for name in average.MEMBERS if name not in names), AVERAGE]
    unknown = [name for name in names if _factory(name) is None and name != AVERAGE]
    if unknown:
        print(f"Unknown models: {', '.join(unknown)} (known: {', '.join(MODELS)}, and {' or '.join(WITH_VARIABLES)} "
              f"with any of +{', +'.join(VARIABLES)}; {LEAN}<model> or {LEAN}average[+<variable>]; "
              f"{TREES}error|outcome[-nolead][@<leaves>-<minimum leaf>-<L2>][+xg])", file=sys.stderr)
        return 2

    started, clock = datetime.now(timezone.utc), time.perf_counter()
    matches = _load(args.leagues)
    until = {"tuning": history.LOCKED_FROM - 1, "locked": history.FRESH_FROM - 1}.get(args.block)
    results, zero, members, all_rows = {}, None, {}, {}
    for name in names:
        model = None
        if name == AVERAGE:
            weights = average.fit_all({member: [row for row in members[member]
                                                if history.block(row[0].season) == "tuning"]
                                       for member in average.MEMBERS})
            rows = average.combine_rows({member: members[member] for member in average.MEMBERS}, weights)
        else:
            model = _factory(name)()
            rows = walkforward.run(model, matches, score_from=history.TUNING_FROM, until=until)
            if name in average.MEMBERS:
                members[name] = rows
        all_rows[name] = rows
        tuning = [row for row in rows if history.block(row[0].season) == "tuning"]
        scored = [row for row in rows if history.block(row[0].season) == args.block]
        results[name] = _score(scored, tuning)
        if name == AVERAGE:
            results[name]["weights"] = weights
        if isinstance(model, Trees):  # each season's fit: its training seasons, and the rounds each part kept
            results[name]["fits"] = {str(season): fitted for season, fitted in sorted(model.fits.items())
                                     if history.block(season) in ("tuning", args.block)}
        if isinstance(model, LeanAway):  # each season's blend: how much it leans on the member (c), and on what
            results[name]["fits"] = {str(season): {"n": fitted["n"], **{part: None if fitted[part] is None else {
                key: fitted[part][key] for key in ("b", "c", "intercepts")} for part in ("result", "over25")}}
                for season, fitted in sorted(model.fits.items()) if history.block(season) in ("tuning", args.block)}
        if "+" in name:
            base = name.rsplit("+", 1)[0]
            results[name]["ablation"] = _ablation(base, all_rows[base], rows, dict.fromkeys(("tuning", args.block)))
        if name == "market":
            zero = zero_line(results[name])
            if not zero["passed"]:
                print(f"The zero line did not score zero, so the bench is wrong; nothing written. {zero['checks']}",
                      file=sys.stderr)
                return 2
        print(f"{name}: {len(scored):,} matches scored ({time.perf_counter() - clock:.0f} s)", flush=True)

    scored_matches = [m for m in matches if history.block(m.season) == args.block]
    summary = {
        "run": {"block": args.block, "time": started.strftime("%Y-%m-%d %H:%M UTC"), "commit": _commit(),
                "models": names, "leagues": args.leagues or "all", "matches_scored": len(scored_matches)},
        "zero_line": zero,
        "models": results,
        "verdicts": {rule: scores.verdicts({name: {"blend": found["blend"], "picks": found["picks"][rule],
                                                   "value": found["value"]["avg"]}
                                            for name, found in results.items()})
                     for rule in ("edge", "blend")},
        "reference": scores.reference_check(scored_matches, _named()),
        "carried": {variable: note for variable, note in CARRIED.items()
                    if any(variable in name.split("+")[1:] for name in names)},
        "opportunities": opportunities.scan([m for m in matches if history.block(m.season) not in history.GUARDED]),
    }
    summary["run"]["seconds"] = round(time.perf_counter() - clock)
    report.write(args.out, summary)
    if locked:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        entry = {"time": summary["run"]["time"], "block": args.block, "models": names, "commit": summary["run"]["commit"],
                 "leagues": summary["run"]["leagues"], "out": str(args.out),
                 "headline": {name: _headline(found) for name, found in results.items()}}
        with LEDGER.open("a") as ledger:
            ledger.write(json.dumps(entry) + "\n")
    print(f"Wrote {args.out} in {summary['run']['seconds']} s")
    return 0


def export_stats(args) -> int:
    try:
        found = stats.export(args.out)
    except RefreshBusy as busy:
        print(f"The data refresh holds the database ({busy}); nothing written. Try again after the refresh.",
              file=sys.stderr)
        return 2
    print(json.dumps(found, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("run", help="score models and write the report")
    bench.add_argument("--models", type=lambda text: [n for n in text.split(",") if n], default=list(REFERENCES),
                       help="comma-separated; market and control always run first")
    bench.add_argument("--block", choices=("tuning", "locked", "fresh"), required=True)
    bench.add_argument("--confirm-locked", action="store_true", help="name this run as the one locked or fresh run")
    bench.add_argument("--leagues", nargs="*", default=None, help="competition names, e.g. EPL LaLiga")
    bench.add_argument("--out", type=Path, required=True)
    bench.set_defaults(command=run)
    export = commands.add_parser("export-stats", help="export the matches' statistics from our database, once")
    export.add_argument("--out", type=Path, default=stats.STATS_DB)
    export.set_defaults(command=export_stats)
    args = parser.parse_args(argv)
    return args.command(args)


if __name__ == "__main__":
    sys.exit(main())
