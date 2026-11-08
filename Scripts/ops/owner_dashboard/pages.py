"""The dashboard's two pages, rendered on the server from its own file. Every value is escaped; the one inline
script (the chart's readout) writes with ``textContent`` only."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from html import escape
from typing import List, Optional
from urllib.parse import urlencode

from Scripts.data_platform import fair_prices as fp
from Scripts.ops.owner_dashboard import report
from Scripts.ops.owner_dashboard.report import KINDS, LEAGUES, MARKETS, local

THEME = """<script>try{var t=localStorage.getItem("spix-owner-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>"""
THEME_TOGGLE = """<script>document.querySelector(".theme").addEventListener("click",function(){var r=document.documentElement,
d=r.dataset.theme?r.dataset.theme==="dark":matchMedia("(prefers-color-scheme: dark)").matches,n=d?"light":"dark";
r.dataset.theme=n;try{localStorage.setItem("spix-owner-theme",n)}catch(e){}});</script>"""
READOUT = """<script>(function(){var c=document.querySelector(".chart"),o=document.querySelector(".readout"),
x=c.querySelector(".cross");function show(t){var d=t.dataset;x.setAttribute("x1",d.x);x.setAttribute("x2",d.x);
x.style.visibility="visible";o.textContent=d.say}function hide(){x.style.visibility="hidden";o.textContent=o.dataset.idle}
c.querySelectorAll(".week").forEach(function(t){t.addEventListener("pointerenter",function(){show(t)});
t.addEventListener("focus",function(){show(t)});t.addEventListener("blur",hide)});c.addEventListener("pointerleave",hide)})();</script>"""


def pct(value) -> str:
    return "–" if value is None else f"{round(value * 100)}%"


def _count(value) -> str:
    return f"{value:,g}"


def _tally(row) -> str:
    """"1,542 settled", with the pending count only when there is one."""
    return f"{_count(row['picks'])} settled" + (f" · {row['pending']:,} pending" if row["pending"] else "")


def _day(moment) -> str:
    return f"{moment:%a} {moment.day} {moment:%b}"


def _short(day) -> str:
    return f"{day.day} {day:%b}"


def _updated(updated: Optional[str]) -> str:
    moment = local(updated) if updated else None
    return f"Updated {_day(moment)}, {moment:%H:%M}" if moment else "Not updated yet"


def _current(on: bool, value: str = "page") -> str:
    return f' aria-current="{value}"' if on else ""


def layout(title: str, active: str, body: str, *, updated=None, notice=None, back="/", script="") -> str:
    nav = "".join(f'<a href="{href}"{_current(name == active)}>{name}</a>'
                  for name, href in (("Report", "/"), ("History", "/history")))
    note = f'<p class="notice" role="status">{escape(notice)}</p>' if notice else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · Spix owner</title>
<link rel="stylesheet" href="/dashboard.css">
{THEME}
</head>
<body>
<header class="top"><div class="wrap">
<span class="brand">Spix <span>owner</span></span>
<nav>{nav}</nav>
<form class="update" method="post" action="/update"><input type="hidden" name="back" value="{escape(back)}">
<span>{escape(_updated(updated))}</span><button type="submit">Update now</button></form>
<button class="theme" type="button" aria-label="Switch light or dark">◐</button>
</div></header>
<main class="wrap">
{note}{body}
</main>
{THEME_TOGGLE}{script}
</body>
</html>
"""


def _href(kind: str, market: Optional[str] = None, league: Optional[str] = None) -> str:
    return "/?" + urlencode({"kind": kind, **({"market": market} if market else {}),
                             **({"league": league} if league else {})})


def _link(kind: str, market: Optional[str] = None, league: Optional[str] = None) -> str:
    return escape(_href(kind, market, league))


W, H, LEFT, RIGHT, TOP, BOTTOM = 960, 280, 48, 110, 14, 30


def _chart(weeks: List[dict]) -> str:
    """Hit (solid) and said (dashed) per week on one axis, fitted to the data in steps of 10% (0–100% in quarters
    when the values spread wider than 50 points); a week with no settled pick has no point."""
    plot_w, plot_h, n = W - LEFT - RIGHT, H - TOP - BOTTOM, len(weeks)
    xs = [LEFT + (plot_w / 2 if n == 1 else i * plot_w / (n - 1)) for i in range(n)]
    values = [w[k] for w in weeks for k in ("hit", "said") if w[k] is not None]
    low = max(0.0, math.floor((min(values) - 0.05) * 10) / 10)
    high = min(1.0, math.ceil((max(values) + 0.05) * 10) / 10)
    if high - low > 0.5:
        low, high = 0.0, 1.0
    step = 0.1 if high - low <= 0.5 else 0.25
    y = lambda value: TOP + (high - value) / (high - low) * plot_h
    parts = []
    for level in [low + i * step for i in range(round((high - low) / step) + 1)]:
        parts.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
                     f'<text class="tick" x="{LEFT - 8}" y="{y(level) + 4:.1f}" text-anchor="end">{pct(level)}</text>')
    step = max(1, -(-n // 7))
    for i, week in enumerate(weeks):
        if i % step == 0 or i == n - 1:
            day = local(week["week"] + "T12:00:00") or None
            parts.append(f'<text class="tick" x="{xs[i]:.1f}" y="{H - 8}" text-anchor="middle">'
                         f'{escape(_short(day) if day else week["week"])}</text>')
    ends = []
    for key, name in (("hit", "Hit"), ("said", "Said")):
        path, pen = [], "M"
        for i, week in enumerate(weeks):
            if week[key] is None:
                pen = "M"
                continue
            path.append(f"{pen}{xs[i]:.1f},{y(week[key]):.1f}")
            pen = "L"
        parts.append(f'<path class="line {key}" d="{" ".join(path)}"/>')
        parts.extend(f'<circle class="dot {key}" cx="{xs[i]:.1f}" cy="{y(w[key]):.1f}" r="4"/>'
                     for i, w in enumerate(weeks) if w[key] is not None)
        last = next((i for i in range(n - 1, -1, -1) if weeks[i][key] is not None), None)
        if last is not None:
            ends.append([y(weeks[last][key]), xs[last], f"{name} {pct(weeks[last][key])}"])
    if len(ends) == 2 and abs(ends[0][0] - ends[1][0]) < 16:
        high, low = sorted(ends, key=lambda e: e[0])
        middle = (high[0] + low[0]) / 2
        high[0], low[0] = middle - 8, middle + 8
    parts.extend(f'<text class="end" x="{x + 10:.1f}" y="{at + 4:.1f}">{escape(text)}</text>' for at, x, text in ends)
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{TOP}" y2="{TOP + plot_h}"/>')
    for i, week in enumerate(weeks):
        left = LEFT if i == 0 else (xs[i - 1] + xs[i]) / 2
        right = W - RIGHT if i == n - 1 else (xs[i] + xs[i + 1]) / 2
        day = local(week["week"] + "T12:00:00")
        say = (f"Week of {_short(day)}: hit {pct(week['hit'])}, said {pct(week['said'])}, "
               f"{week['picks']:,} settled, {week['pending']} pending")
        parts.append(f'<rect class="week" tabindex="0" x="{left:.1f}" y="{TOP}" width="{max(right - left, 1):.1f}" '
                     f'height="{plot_h}" data-x="{xs[i]:.1f}" data-say="{escape(say)}" aria-label="{escape(say)}"/>')
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Hit and said, week by week. '
            f'The table below has the numbers.">{"".join(parts)}</svg>')


def _gap(row) -> str:
    """Hit minus said in points ("+3", "−2", "±0"); empty before anything settles."""
    if row["hit"] is None:
        return ""
    gap = round((row["hit"] - row["said"]) * 100)
    return f'{"+" if gap > 0 else "−" if gap < 0 else "±"}{abs(gap)}'


def _units(value) -> str:
    """A profit in units, signed ("+1.50", "−1.00", "0.00"); "–" before anything is staked."""
    if value is None:
        return "–"
    text = f"{abs(value):.2f}"
    return text if text == "0.00" else ("+" if value > 0 else "−") + text


def _roi(value) -> str:
    """The profit over the units staked, in whole percent, signed."""
    if value is None:
        return "–"
    points = round(value * 100)
    return f'{"+" if points > 0 else "−" if points < 0 else ""}{abs(points)}%'


def _tone(row) -> str:
    """Colours a profit and its ROI by their sign."""
    if _units(row["profit"]) in ("–", "0.00"):
        return ""
    return ' class="up"' if row["profit"] > 0 else ' class="down"'


def _money(row) -> str:
    return f'<td{_tone(row)}>{_units(row["profit"])}</td><td{_tone(row)}>{_roi(row["roi"])}</td>'


def _returns(row, more: str = "") -> str:
    """"+0.50 units · ROI +17% on 3 units staked", for Value picks."""
    staked = row["staked"]
    return (f'<b{_tone(row)}>{_units(row["profit"])}</b> units <span>· ROI {_roi(row["roi"])} on '
            f'{_count(staked)} {"unit" if staked == 1 else "units"} staked{more}</span>')


MONEY = "<th>Profit (units)</th><th>ROI</th>"
FLAT_STAKE = ("One unit on each settled pick, at the price it was published at. A push or a void gives the unit "
              "back; a half result settles half of it.")


def _tile(kind: str, market: dict, chosen: Optional[str], league: Optional[str]) -> str:
    gap_text = f'<span class="gap">{_gap(market)} vs said</span>' if _gap(market) else ""
    on = _current(market["market"] == chosen, "true")
    return (f'<a class="tile" href="{_link(kind, market["market"], league)}"{on}>'
            f'<span class="name">{escape(MARKETS.get(market["market"], market["market"]))}</span>'
            f'<b>{pct(market["hit"])}</b><span class="sub">said {pct(market["said"])} · '
            f'{_tally(market)}</span>{gap_text}</a>')


def _league_row(kind: str, row: dict, market: Optional[str], chosen: Optional[str]) -> str:
    """One league's line in the table: its name opens it, keeping the market."""
    on = row.get("league") == chosen
    name = LEAGUES.get(row["league"], row["league"]) if row.get("league") else "All leagues"
    return (f'<tr{' class="on"' if on else ""}><td><a href="{_link(kind, market, row.get("league"))}"'
            f'{_current(on, "true")}>{escape(name)}</a></td><td>{row["picks"]:,}</td><td>{row["pending"]:,}</td>'
            f'<td>{pct(row["said"])}</td><td>{pct(row["hit"])}</td><td>{_gap(row) or "–"}</td></tr>')


def report_page(conn, *, kind: str = "most_likely", market: Optional[str] = None, league: Optional[str] = None,
                updated=None, notice=None) -> str:
    kind = kind if kind in KINDS else "most_likely"
    data = report.summary(conn, kind=kind, market=market, league=league)
    if league is not None and league not in {lg["league"] for lg in data["leagues"]}:
        league, data = None, report.summary(conn, kind=kind, market=market)
    if market is not None and market not in {m["market"] for m in data["markets"]}:
        market, data = None, report.summary(conn, kind=kind, league=league)
    switch = "".join(f'<a href="{_link(name, market, league)}"{_current(name == kind)}>{label}</a>'
                     for name, label in KINDS.items())
    scope = " · ".join([MARKETS.get(market, market) if market else "All markets"]
                       + ([LEAGUES.get(league, league)] if league else []))
    overall = data["overall"]
    head = f'<div class="switch">{switch}</div><h1>{escape(KINDS[kind])} <span>· {escape(scope)}</span></h1>'
    if not overall["picks"]:
        waiting = f" {overall['pending']} waiting for their results." if overall["pending"] else ""
        body = (head + f'<p class="empty">No settled {escape(KINDS[kind])} picks yet.{waiting} '
                f'The report fills as matches finish.</p>')
        return layout("Report", "Report", body, updated=updated, notice=notice, back=_href(kind, market, league))
    idle = "Hover or tab through the weeks to read each one."
    money = kind == "value"  # Value picks carry a price; Most likely shows said against hit only
    rows = "".join(
        f'<tr><td>Week of {escape(_short(local(w["week"] + "T12:00:00")))}</td><td>{w["picks"]:,}</td>'
        f'<td>{w["pending"]}</td><td>{pct(w["said"])}</td><td>{pct(w["hit"])}</td>{_money(w) if money else ""}</tr>'
        for w in data["weeks"])
    daily = "".join(
        f'<tr><td><a href="/history?day={d["day"]}">{escape(_day(date.fromisoformat(d["day"])))}</a></td>'
        f'<td>{d["picks"]:,}</td><td>{d["pending"]:,}</td><td>{pct(d["said"])}</td><td>{pct(d["hit"])}</td>'
        f'{_money(d)}</tr>' for d in data["days"])
    days = f"""
<h2>Day by day</h2>
<div class="panel table daily"><table>
<thead><tr><th>Day</th><th>Settled</th><th>Pending</th><th>Said</th><th>Hit</th>{MONEY}</tr></thead>
<tbody>{daily}</tbody>
</table></div>
<p class="note">{FLAT_STAKE}</p>""" if money else ""
    returns = f'\n<p class="headline money">{_returns(overall)}</p>' if money else ""
    every = report.summary(conn, kind=kind, league=league)["overall"] if market else overall
    every_league = report.summary(conn, kind=kind, market=market)["overall"] if league else overall
    leagues = _league_row(kind, every_league, market, league) + "".join(
        _league_row(kind, row, market, league) for row in data["leagues"])
    all_tile = (f'<a class="tile all" href="{_link(kind, None, league)}"{_current(not market, "true")}>'
                f'<span class="name">All markets</span><b>{pct(every["hit"])}</b><span class="sub">said '
                f'{pct(every["said"])} · {_tally(every)}</span></a>')
    body = f"""{head}
<p class="headline"><b>{pct(overall["hit"])}</b> hit <span>· said {pct(overall["said"])} · {_tally(overall)}</span></p>{returns}
<section class="panel">
<div class="legend"><span><i class="swatch hit"></i>Hit</span><span><i class="swatch said"></i>Said</span></div>
<p class="readout" aria-live="polite" data-idle="{idle}">{idle}</p>
<div class="chart-scroll">{_chart(data["weeks"])}</div>
</section>
<h2>Markets</h2>
<div class="tiles">{all_tile}{"".join(_tile(kind, m, market, league) for m in data["markets"])}</div>
<h2>Leagues</h2>
<div class="panel table leagues"><table>
<thead><tr><th>League</th><th>Settled</th><th>Pending</th><th>Said</th><th>Hit</th><th>Hit vs said</th></tr></thead>
<tbody>{leagues}</tbody>
</table></div>
<p class="note">A league with few settled picks can land far from what was said by chance alone.</p>
<h2>Week by week</h2>
<div class="panel table"><table>
<thead><tr><th>Week</th><th>Settled</th><th>Pending</th><th>Said</th><th>Hit</th>{MONEY if money else ""}</tr></thead>
<tbody>{rows}</tbody>
</table></div>{days}"""
    return layout("Report", "Report", body, updated=updated, notice=notice, back=_href(kind, market, league),
                  script=READOUT)


RESULTS = {"hit": ("Hit", "hit"), "half_hit": ("Half hit", "hit"), "miss": ("Miss", "miss"),
           "half_miss": ("Half miss", "miss"), "push": ("Push", "void"), "void": ("Void", "void")}
WAITING = {"corners not in yet": "waiting for the corners", "score not in yet": "waiting for the score",
           "not final": "waiting for the final result"}
COUNTS = {"goals": ("goal", "goals"), "corners": ("corner", "corners"), "cards": ("card", "cards"),
          "sot": ("shot on target", "shots on target")}


def _pair(facts, key):
    home, away = facts.get(f"{key}_home"), facts.get(f"{key}_away")
    return None if home is None or away is None else (home, away)


def _cards(facts):
    counts = [facts.get(k) for k in ("yellow_home", "red_home", "yellow_away", "red_away")]
    return None if None in counts else counts


def _number(value, market) -> str:
    one, many = COUNTS[market]
    return f"{float(value):g} {one if float(value) == 1 else many}"


def _team_side(side, home, away) -> Optional[str]:
    if side in ("home", "away"):
        return side
    return {fp.norm(home or ""): "home", fp.norm(away or ""): "away"}.get(fp.norm(side or ""))


def what_happened(market: str, facts: dict, pick: dict) -> str:
    """The number the pick settled on: a total for over/under markets, the score or the margin otherwise."""
    home_goals, away_goals = facts.get("home_goals"), facts.get("away_goals")
    score = None if home_goals is None or away_goals is None else f"{home_goals}–{away_goals}"
    if facts.get("status") in ("AET", "PEN") and score and market in ("goals", "moneyline", "btts", "spreads"):
        return f"{score} after extra time"
    if market in COUNTS:
        if pick.get("actual_result") is not None:
            return _number(pick["actual_result"], market)
        if market == "goals":
            total = None if score is None else home_goals + away_goals
        elif market == "cards":
            total = None if _cards(facts) is None else sum(_cards(facts))
        else:
            pair = _pair(facts, "corners" if market == "corners" else "sot")
            total = None if pair is None else sum(pair)
        return f"{'score' if market == 'goals' else COUNTS[market][1]} not in yet" if total is None else (
            _number(total, market))
    if score is None:
        return "score not in yet"
    side = _team_side(pick.get("side"), facts.get("home"), facts.get("away")) if market == "spreads" else None
    if side is None:
        return score
    margin = home_goals - away_goals if side == "home" else away_goals - home_goals
    team = facts.get("home") if side == "home" else facts.get("away")
    return f"{team} drew" if margin == 0 else f"{team} {'won' if margin > 0 else 'lost'} by {abs(margin)}"


def _facts_line(match: dict) -> str:
    parts = []
    if match.get("status") in ("AET", "PEN"):
        parts.append("After extra time" + (" and penalties" if match["status"] == "PEN" else ""))
    corners = _pair(match, "corners")
    parts.append(f"Corners {sum(corners)} ({corners[0]}–{corners[1]})" if corners else "Corners not in yet")
    cards = _cards(match)
    if cards:
        reds = cards[1] + cards[3]
        parts.append(f"Cards {sum(cards)}" + (f" ({reds} red)" if reds else ""))
    sot = _pair(match, "sot")
    if sot:
        parts.append(f"Shots on target {sum(sot)} ({sot[0]}–{sot[1]})")
    return " · ".join(parts)


def _played(conn, now: datetime) -> List[dict]:
    """Every match with a logged pick that has kicked off, with its picks (Most likely first, in list order)."""
    matches = {row["fixture_api_id"]: dict(row, picks=[]) for row in conn.execute(
        "SELECT * FROM matches WHERE fixture_api_id IN (SELECT fixture_api_id FROM most_likely "
        "UNION SELECT fixture_api_id FROM value_picks)")}
    for row in conn.execute("SELECT * FROM most_likely ORDER BY fixture_api_id, position"):
        matches[row["fixture_api_id"]]["picks"].append(
            dict(row, list="Most likely", said=row["chance"],
                 price=None if row["typical_price"] is None else f"{row['typical_price']:.2f}"))
    for row in conn.execute("SELECT * FROM value_picks ORDER BY published_at, selection_key"):
        match = matches.setdefault(row["fixture_api_id"], {
            "fixture_api_id": row["fixture_api_id"], "league": row["league"], "home": row["home"],
            "away": row["away"], "kickoff": row["kickoff"], "status": None, "picks": []})
        price = None if row["odds"] is None else f"{row['odds']:.2f}" + (
            f" at {row['bookmaker']}" if row["bookmaker"] else "")
        match["picks"].append(dict(row, list="Value", price=price, pending_reason=None))
    played = []
    for match in matches.values():
        moment = local(match.get("kickoff"))
        if moment and moment <= now.astimezone():
            played.append(dict(match, moment=moment, day=moment.date()))
    return played


def _pick_line(match: dict, pick: dict) -> str:
    what = what_happened(pick["market"], match, pick)
    if pick["outcome"] is None:
        result, style = "Pending", "pending"
        reason = pick.get("pending_reason")
        what = f"{what} · not graded" if reason == "extra time" else WAITING.get(reason, what)
    else:
        result, style = RESULTS.get(pick["outcome"], (pick["outcome"].replace("_", " ").capitalize(), "void"))
    meta = " · ".join(x for x in (f"said {pct(pick['said'])}", pick.get("price")) if x)
    return (f'<div class="pick"><span class="list">{escape(pick["list"])}</span>'
            f'<span><b>{escape(pick["label"] or "")}</b> <span class="meta">· {escape(meta)}</span><br>'
            f'<span class="what">{escape(what)}</span></span><span class="result {style}">{escape(result)}</span></div>')


def _match_card(match: dict) -> str:
    home_goals, away_goals = match.get("home_goals"), match.get("away_goals")
    score = "" if home_goals is None or away_goals is None else f"{home_goals}–{away_goals}"
    if match.get("status") and match["status"] != "FT":
        score = " · ".join(x for x in (score, match["status"]) if x)
    return (f'<article class="match"><div class="head"><span class="teams">{escape(match["home"] or "")} v '
            f'{escape(match["away"] or "")}<small>{match["moment"]:%H:%M}</small></span>'
            f'<span class="score">{escape(score)}</span></div><div class="facts">{escape(_facts_line(match))}</div>'
            f'{"".join(_pick_line(match, p) for p in match["picks"])}</article>')


def _day_link(day, label: str, *, on=False, step=False) -> str:
    return (f'<a{' class="step"' if step else ""} href="/history?day={day.isoformat()}"{_current(on)}>'
            f'{escape(label)}</a>')


def history_page(conn, *, day=None, updated=None, notice=None, now: Optional[datetime] = None) -> str:
    """One played day: its matches by league, one line per pick. Opens on the latest day with a settled pick;
    the strip shows that day's week (days with picks only) and steps to the nearest week with picks."""
    played = _played(conn, now or datetime.now(timezone.utc))
    days = sorted({m["day"] for m in played})
    back = "/history" + (f"?day={day}" if day else "")
    if not days:
        body = '<h1>History</h1><p class="empty">No played matches with picks yet. They appear here once a match is over.</p>'
        return layout("History", "History", body, updated=updated, notice=notice, back=back)
    try:
        chosen = date.fromisoformat(str(day)) if day else None
    except ValueError:
        chosen = None
    if chosen not in days:
        settled = [m["day"] for m in played if any(p["outcome"] is not None for p in m["picks"])]
        chosen = max(settled or days)
    monday = chosen - timedelta(days=chosen.weekday())
    sunday = monday + timedelta(days=6)
    earlier = [d for d in days if d < monday]
    later = [d for d in days if d > sunday]
    strip = ((_day_link(max(earlier), "‹ Earlier week", step=True) if earlier else "")
             + "".join(_day_link(d, _day(d), on=d == chosen) for d in days if monday <= d <= sunday)
             + (_day_link(min(later), "Later week ›", step=True) if later else ""))
    today = sorted((m for m in played if m["day"] == chosen), key=lambda m: (
        list(LEAGUES).index(m["league"]) if m["league"] in LEAGUES else len(LEAGUES), m["league"] or "",
        m["moment"], m["home"] or ""))
    groups, league = [], object()
    for match in today:
        if match["league"] != league:
            league = match["league"]
            groups.append(f'<h2 class="league">{escape(LEAGUES.get(league, league or "Other"))}</h2>')
        groups.append(_match_card(match))
    picks = sum(len(m["picks"]) for m in today)
    value = report.tally([p for m in today for p in m["picks"] if p["list"] == "Value"])
    waiting = f' · {value["pending"]:,} pending' if value["pending"] else ""
    if value["staked"]:
        returns = f'<p class="headline">Value picks {_returns(value, waiting)}</p>'
    elif value["pending"]:
        returns = f'<p class="headline">Value picks <span>· {value["pending"]:,} pending, no result yet</span></p>'
    else:
        returns = ""
    body = (f'<nav class="days" aria-label="Days">{strip}</nav><h1>{escape(_day(chosen))} <span>· {len(today)} '
            f'{"match" if len(today) == 1 else "matches"} · {picks} {"pick" if picks == 1 else "picks"}</span></h1>'
            + returns + "".join(groups))
    return layout("History", "History", body, updated=updated, notice=notice, back=back)
