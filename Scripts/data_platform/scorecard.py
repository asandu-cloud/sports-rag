"""Weekly scorecard: Spix's published picks against the market (rework A1 step 5).

Two cohorts of published picks:

* ``selections``: each distinct selection once, at its first publication
  (re-reads republish the same pick, often several times);
* ``initial``: the official track-record cohort, picks on each fixture's first
  delivered Match Read.

Per cohort, overall and by market, confidence label and both: settled count,
Asian-weighted win rate (half results count half, pushes and voids not at
all), Spix's stated probability, the market's fair probability at publication,
log loss of each and their gap, calibration (slope, intercept, binned error),
closing line value in price (same bookmaker) and in fair probability
(consensus at close over consensus at publication), and flat-stake return.
Intervals are 95% percentile bootstraps over matchweeks, or over fixtures
while fewer than ``MIN_WEEKS`` matchweeks exist.

The market probabilities come from ``reference_price`` (proportional margin
removal, median across books). For quarter and whole lines this approximates
the win probability excluding pushes, the basis of Spix's stated probability.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import math
import random
import statistics
from typing import Dict, List, Optional

from .outcomes import outcome_accounting

MIN_SETTLED = 30      # below this a slice shows counts only
MIN_WEEKS = 8         # matchweek bootstrap needs at least this many weeks
EPS = 1e-6
INTERVAL_METRICS = ("win_rate_minus_stated", "log_loss_gap", "price_clv", "fair_clv", "roi")


def outcome_weight(outcome):
    """(win share, weight): a half win is a win on half the stake; pushes and voids carry no weight."""
    accounting = outcome_accounting(outcome)
    if accounting is None or accounting.resolved_units <= 0:
        return None, 0.0
    return accounting.win_units / accounting.resolved_units, accounting.resolved_units


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _week(kickoff):
    try:
        day = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00")).date()
    except ValueError:
        return None
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def record(row) -> dict:
    """The fields the scorecard needs from one published prediction row."""
    y, w = outcome_weight(row.get("outcome"))
    accounting = outcome_accounting(row.get("outcome"), row.get("odds"))
    reference = row.get("publication_reference") or {}
    market_p = _number(reference.get("fair_probability")) if reference.get("status") == "available" else None
    capture = row.get("closing_capture") or {}
    close_p = _number((capture.get("consensus") or {}).get("fair_probability")) if capture.get("status") == "final" else None
    return {
        "fixture": row.get("fixture_id"), "week": _week(row.get("kickoff")),
        "market": row.get("market"), "confidence": row.get("confidence") or "none",
        "y": y, "w": w, "p": _number(row.get("model_prob")), "market_p": market_p,
        "stake": accounting.stake_units if accounting and row.get("odds") else 0.0,
        "returned": accounting.returned_units if accounting and row.get("odds") else 0.0,
        "price_clv": _number(row.get("clv")),
        "fair_clv": close_p / market_p - 1 if close_p and market_p else None,
        "closing_minutes": _number(capture.get("minutes_before_kickoff")),
    }


def cohort(rows, kind):
    if kind == "initial":
        return [r for r in rows if r.get("publication_role") == "initial"]
    first = {}
    for r in sorted(rows, key=lambda r: r.get("recommendation_id") or 0):
        first.setdefault(r.get("selection_key") or ("id", r.get("id")), r)
    return list(first.values())


def _log_loss(y, p):
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def calibration_fit(points):
    """Weighted logistic regression of the outcome on logit(stated probability): (intercept, slope)."""
    a, b = 0.0, 1.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for y, w, p in points:
            x = _logit(p)
            q = 1 / (1 + math.exp(-(a + b * x)))
            ga += w * (y - q)
            gb += w * (y - q) * x
            v = w * q * (1 - q)
            haa, hab, hbb = haa + v, hab + v * x, hbb + v * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            return None, None
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def _calibration_error(points, bins=10):
    total = sum(w for _, w, _ in points)
    buckets = defaultdict(lambda: [0.0, 0.0, 0.0])
    for y, w, p in points:
        b = buckets[min(int(p * bins), bins - 1)]
        b[0] += w * y
        b[1] += w * p
        b[2] += w
    return sum(abs(s_y - s_p) for s_y, s_p, _ in buckets.values()) / total if total else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _core(records):
    """The quantities bootstrapped for intervals (cheap: no calibration fit)."""
    settled = [r for r in records if r["w"] > 0]
    weight = sum(r["w"] for r in settled)
    win = sum(r["w"] * r["y"] for r in settled) / weight if weight else None
    stated_rows = [r for r in settled if r["p"] is not None]
    stated_w = sum(r["w"] for r in stated_rows)
    stated = sum(r["w"] * r["p"] for r in stated_rows) / stated_w if stated_w else None
    both = [r for r in settled if r["p"] is not None and r["market_p"] is not None]
    both_w = sum(r["w"] for r in both)
    gap = (sum(r["w"] * (_log_loss(r["y"], r["p"]) - _log_loss(r["y"], r["market_p"])) for r in both) / both_w
           if both_w else None)
    stake = sum(r["stake"] for r in records)
    return {
        "win_rate_minus_stated": win - stated if win is not None and stated is not None else None,
        "log_loss_gap": gap,
        "price_clv": _mean(r["price_clv"] for r in records),
        "fair_clv": _mean(r["fair_clv"] for r in records),
        "roi": (sum(r["returned"] for r in records) - stake) / stake if stake else None,
    }


def metrics(records) -> dict:
    settled = [r for r in records if r["w"] > 0]
    weight = sum(r["w"] for r in settled)
    with_p = [r for r in settled if r["p"] is not None]
    with_market = [r for r in settled if r["market_p"] is not None]
    both = [r for r in with_p if r["market_p"] is not None]
    both_w = sum(r["w"] for r in both)

    def wmean(rows, field):
        w = sum(r["w"] for r in rows)
        return sum(r["w"] * r[field] for r in rows) / w if w else None

    points = [(r["y"], r["w"], r["p"]) for r in with_p]
    intercept, slope = calibration_fit(points) if len(points) >= MIN_SETTLED else (None, None)
    price = [r["price_clv"] for r in records if r["price_clv"] is not None]
    fair = [r["fair_clv"] for r in records if r["fair_clv"] is not None]
    minutes = [r["closing_minutes"] for r in records if r["closing_minutes"] is not None]
    core = _core(records)
    return {
        "published": len(records), "settled": len(settled), "with_market": len(with_market),
        "win_rate": wmean(settled, "y"), "stated": wmean(with_p, "p"), "market": wmean(with_market, "market_p"),
        "log_loss_spix": sum(r["w"] * _log_loss(r["y"], r["p"]) for r in both) / both_w if both_w else None,
        "log_loss_market": sum(r["w"] * _log_loss(r["y"], r["market_p"]) for r in both) / both_w if both_w else None,
        "log_loss_gap": core["log_loss_gap"],
        "calibration_slope": slope, "calibration_intercept": intercept,
        "calibration_error": _calibration_error(points) if points else None,
        "price_clv": core["price_clv"], "price_clv_positive": sum(v > 0 for v in price) / len(price) if price else None,
        "fair_clv": core["fair_clv"], "fair_clv_positive": sum(v > 0 for v in fair) / len(fair) if fair else None,
        "closing_minutes_median": statistics.median(minutes) if minutes else None,
        "roi": core["roi"], "stake_units": sum(r["stake"] for r in records),
    }


def intervals(records, cluster_field, resamples, seed):
    clusters = defaultdict(list)
    for r in records:
        clusters[r[cluster_field]].append(r)
    groups = list(clusters.values())
    if len(groups) < 2:
        return {k: None for k in INTERVAL_METRICS}
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(resamples):
        sample = [r for g in rng.choices(groups, k=len(groups)) for r in g]
        for key, value in _core(sample).items():
            if value is not None:
                draws[key].append(value)
    out = {}
    for key in INTERVAL_METRICS:
        values = sorted(draws[key])
        out[key] = ([values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
                    if len(values) >= resamples // 2 else None)
    return out


def _slice(records, cluster_field, resamples, seed):
    settled = sum(1 for r in records if r["w"] > 0)
    if settled < MIN_SETTLED:
        return {"published": len(records), "settled": settled, "insufficient_sample": True}
    return {**metrics(records), "intervals": intervals(records, cluster_field, resamples, seed)}


def build(rows, *, as_of=None, resamples=2000, seed=0, after_recommendation_id=None) -> dict:
    as_of = as_of or date.today().isoformat()
    card = {"as_of": as_of, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "after_recommendation_id": after_recommendation_id, "cohorts": {}}
    weeks = {record(r)["week"] for r in cohort(rows, "selections")} - {None}
    cluster_field = "week" if len(weeks) >= MIN_WEEKS else "fixture"
    card["interval_method"] = "matchweek_bootstrap" if cluster_field == "week" else "fixture_bootstrap"
    card["matchweeks"] = sorted(weeks)
    for kind in ("selections", "initial"):
        records = [record(r) for r in cohort(rows, kind)]
        groups = {"by_market": defaultdict(list), "by_confidence": defaultdict(list),
                  "by_market_confidence": defaultdict(list), "by_week": defaultdict(list)}
        for r in records:
            groups["by_market"][r["market"]].append(r)
            groups["by_confidence"][r["confidence"]].append(r)
            groups["by_market_confidence"][f"{r['market']} · {r['confidence']}"].append(r)
            groups["by_week"][r["week"]].append(r)
        card["cohorts"][kind] = {
            "overall": _slice(records, cluster_field, resamples, seed),
            **{name: {key: _slice(items, "fixture", resamples, seed) for key, items in sorted(g.items(), key=lambda kv: str(kv[0]))}
               for name, g in groups.items()},
        }
    return card


# --- report -----------------------------------------------------------------------------------

COHORT_TITLES = {"selections": "Each selection once (first publication)",
                 "initial": "Official cohort (first match read)"}
COLUMNS = [("Published", "published", "n"), ("Settled", "settled", "n"), ("Win rate", "win_rate", "pct"),
           ("Spix said", "stated", "pct"), ("Market said", "market", "pct"),
           ("Log-loss gap", "log_loss_gap", "num"), ("Calib. slope", "calibration_slope", "num"),
           ("Price CLV", "price_clv", "pct"), ("Fair CLV", "fair_clv", "pct"), ("Return", "roi", "pct")]


def _fmt(value, kind):
    if value is None:
        return "–"
    if kind == "n":
        return f"{value:,}"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    return f"{value:+.3f}"


def _minutes(value):
    return "–" if value is None else f"{value:.0f} min"


def _interval(values, kind):
    if not values:
        return ""
    low, high = values
    return f" ({low * 100:+.1f} to {high * 100:+.1f} pts)" if kind == "pct" else f" ({low:+.3f} to {high:+.3f})"


def _table(slices):
    lines = ["| | " + " | ".join(title for title, _, _ in COLUMNS) + " |",
             "|---|" + "---:|" * len(COLUMNS)]
    for name, m in slices.items():
        if m.get("insufficient_sample"):
            cells = [_fmt(m["published"], "n"), _fmt(m["settled"], "n")] + ["too few"] + [""] * (len(COLUMNS) - 3)
        else:
            cells = [_fmt(m.get(key), kind) for _, key, kind in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def markdown(card) -> str:
    lines = [f"# Spix scorecard — {card['as_of']}", "",
             f"Published picks since recommendation {card.get('after_recommendation_id')}; matchweeks "
             f"{', '.join(card['matchweeks']) or 'none'}. Intervals: 95% {card['interval_method'].replace('_', ' ')}.",
             "",
             "- **Win rate** is Asian-weighted (half results count half; pushes and voids are left out).",
             "- **Log-loss gap** = Spix's log loss minus the market's at publication: **negative means Spix beat the market**.",
             "- **Calib. slope** 1.0 is ideal; below 1 means Spix is overconfident.",
             "- **Price CLV**: published odds over the same bookmaker's final pre-kickoff odds, minus 1. "
             "**Fair CLV**: market fair probability at close over at publication, minus 1 (positive means the market moved toward the pick).",
             f"- Closing = API-Football's final pre-kickoff snapshot (median "
             f"{_minutes(card['cohorts']['selections']['overall'].get('closing_minutes_median'))} before kickoff).",
             ""]
    for kind, title in COHORT_TITLES.items():
        c = card["cohorts"][kind]
        o = c["overall"]
        lines += [f"## {title}", ""]
        if not o.get("insufficient_sample"):
            iv = o["intervals"]
            lines += [f"- Win rate {_fmt(o['win_rate'], 'pct')} vs Spix said {_fmt(o['stated'], 'pct')}: "
                      f"{_fmt(o['win_rate'] - o['stated'], 'pct')} pts{_interval(iv['win_rate_minus_stated'], 'pct')}"
                      if o["win_rate"] is not None and o["stated"] is not None else "- Win rate: –",
                      f"- Log-loss gap {_fmt(o['log_loss_gap'], 'num')}{_interval(iv['log_loss_gap'], 'num')}",
                      f"- Fair CLV {_fmt(o['fair_clv'], 'pct')}{_interval(iv['fair_clv'], 'pct')}; "
                      f"price CLV {_fmt(o['price_clv'], 'pct')}{_interval(iv['price_clv'], 'pct')}",
                      f"- Return {_fmt(o['roi'], 'pct')}{_interval(iv['roi'], 'pct')} on {o['stake_units']:.0f} units", ""]
        lines += ["### Overall and by market", ""] + _table({"All": o, **c["by_market"]}) + [""]
        lines += ["### By confidence label", ""] + _table(c["by_confidence"]) + [""]
        lines += ["### By market and confidence", ""] + _table(c["by_market_confidence"]) + [""]
        lines += ["### By matchweek", ""] + _table(c["by_week"]) + [""]
    return "\n".join(lines)
