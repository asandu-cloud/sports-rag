"""The dashboard's two pages, rendered on the server from its own file. Every value is escaped; the one inline
script (the chart's readout) writes with ``textContent`` only."""
from __future__ import annotations

from html import escape
from typing import List, Optional
from urllib.parse import urlencode

from Scripts.ops.owner_dashboard import report
from Scripts.ops.owner_dashboard.report import KINDS, MARKETS, local

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
    return f"{value:g}"


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


def _href(kind: str, market: Optional[str] = None) -> str:
    return "/?" + urlencode({"kind": kind, **({"market": market} if market else {})})


def _link(kind: str, market: Optional[str] = None) -> str:
    return escape(_href(kind, market))


W, H, LEFT, RIGHT, TOP, BOTTOM = 720, 240, 44, 104, 14, 30


def _chart(weeks: List[dict]) -> str:
    """Hit (solid) and said (dashed) per week on one 0–100% axis; a week with no settled pick has no point."""
    plot_w, plot_h, n = W - LEFT - RIGHT, H - TOP - BOTTOM, len(weeks)
    xs = [LEFT + (plot_w / 2 if n == 1 else i * plot_w / (n - 1)) for i in range(n)]
    y = lambda value: TOP + (1 - value) * plot_h
    parts = []
    for level in (0, 0.25, 0.5, 0.75, 1):
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
               f"{_count(week['settled'])} settled, {week['pending']} pending")
        parts.append(f'<rect class="week" tabindex="0" x="{left:.1f}" y="{TOP}" width="{max(right - left, 1):.1f}" '
                     f'height="{plot_h}" data-x="{xs[i]:.1f}" data-say="{escape(say)}" aria-label="{escape(say)}"/>')
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Hit and said, week by week. '
            f'The table below has the numbers.">{"".join(parts)}</svg>')


def _tile(kind: str, market: dict, chosen: Optional[str]) -> str:
    gap = "" if market["hit"] is None else round((market["hit"] - market["said"]) * 100)
    gap_text = "" if gap == "" else f'<span class="gap">{"+" if gap > 0 else "−" if gap < 0 else "±"}{abs(gap)} vs said</span>'
    on = _current(market["market"] == chosen, "true")
    return (f'<a class="tile" href="{_link(kind, market["market"])}"{on}>'
            f'<span class="name">{escape(MARKETS.get(market["market"], market["market"]))}</span>'
            f'<b>{pct(market["hit"])}</b><span class="sub">said {pct(market["said"])} · '
            f'{market["picks"]} settled · {market["pending"]} pending</span>{gap_text}</a>')


def report_page(conn, *, kind: str = "most_likely", market: Optional[str] = None, updated=None, notice=None) -> str:
    kind = kind if kind in KINDS else "most_likely"
    data = report.summary(conn, kind=kind, market=market)
    if market not in {m["market"] for m in data["markets"]}:
        market, data = None, report.summary(conn, kind=kind)
    switch = "".join(f'<a href="{_link(name, market)}"{_current(name == kind)}>{label}</a>'
                     for name, label in KINDS.items())
    scope = MARKETS.get(market, market) if market else "All markets"
    overall = data["overall"]
    head = f'<div class="switch">{switch}</div><h1>{escape(KINDS[kind])} <span>· {escape(scope)}</span></h1>'
    if not overall["picks"]:
        waiting = f" {overall['pending']} waiting for their results." if overall["pending"] else ""
        body = (head + f'<p class="empty">No settled {escape(KINDS[kind])} picks yet.{waiting} '
                f'The report fills as matches finish.</p>')
        return layout("Report", "Report", body, updated=updated, notice=notice, back=_href(kind, market))
    idle = "Hover or tab through the weeks to read each one."
    rows = "".join(
        f'<tr><td>Week of {escape(_short(local(w["week"] + "T12:00:00")))}</td><td>{_count(w["settled"])}</td>'
        f'<td>{w["pending"]}</td><td>{pct(w["said"])}</td><td>{pct(w["hit"])}</td></tr>' for w in data["weeks"])
    every = report.summary(conn, kind=kind)["overall"] if market else overall
    all_tile = (f'<a class="tile all" href="{_link(kind)}"{_current(not market, "true")}>'
                f'<span class="name">All markets</span><b>{pct(every["hit"])}</b><span class="sub">said '
                f'{pct(every["said"])} · {every["picks"]} settled · {every["pending"]} pending</span></a>')
    body = f"""{head}
<p class="headline"><b>{pct(overall["hit"])}</b> hit <span>· said {pct(overall["said"])} · {overall["picks"]} settled · {overall["pending"]} pending</span></p>
<section class="panel">
<div class="legend"><span><i class="swatch hit"></i>Hit</span><span><i class="swatch said"></i>Said</span></div>
<p class="readout" aria-live="polite" data-idle="{idle}">{idle}</p>
{_chart(data["weeks"])}
</section>
<h2>Markets</h2>
<div class="tiles">{all_tile}{"".join(_tile(kind, m, market) for m in data["markets"])}</div>
<h2>Week by week</h2>
<div class="panel table"><table>
<thead><tr><th>Week</th><th>Settled</th><th>Pending</th><th>Said</th><th>Hit</th></tr></thead>
<tbody>{rows}</tbody>
</table></div>"""
    return layout("Report", "Report", body, updated=updated, notice=notice, back=_href(kind, market), script=READOUT)
