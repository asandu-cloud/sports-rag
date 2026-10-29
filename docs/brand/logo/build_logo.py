"""Build the Spix wordmark SVGs (D48): lowercase "spix" in Fraunces, "Landing" layout, a classic football for the dot.

Not part of the app or the test suite. Needs fontTools (`pip install fonttools`) and the static Fraunces instance
"Fraunces 144pt SuperSoft Bold" (Google Fonts, SIL OFL), e.g. fetched from
https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@144,700,100 with a browser that receives WOFF/TTF.

    python docs/brand/logo/build_logo.py path/to/fraunces.woff

Writes, next to this file: spix-wordmark-white.svg (for turf green or dark backgrounds), spix-wordmark-ink.svg
(for light backgrounds), spix-discord-avatar.svg (full name on a turf tile, 512 px) and spix-header.svg (the
website header version, with the class names the page-change animation uses). The layout and the ball follow the
concepts page https://claude.ai/artifact/Nj3bB4vjm9pdEFmThZdVxZ exactly.
"""
import math
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
B = 100  # font size of a k=1 letter, in SVG units
BALL_SCALE = 1.75  # ball diameter as a multiple of the font's own i dot
# "Landing": the last frame of the drop. k = size, dy = down (em), r = tilt (deg), sx/sy = squash
LANDING = [
    dict(ch="s", k=1.05, dy=-0.03, r=-2),
    dict(ch="p", k=0.97, dy=0, r=-4),
    dict(ch="dotlessi", k=1, dy=0.03, sx=1.05, sy=0.92),
    dict(ch="x", k=1.05, dy=-0.02, r=5),
]
INK, TURF, WHITE, PAPER = "#111E17", "#0E6336", "#FFFFFF", "#FAFDFA"
CLASSIC = dict(patch=INK, front=INK, shade=INK, shade_op=0.16, hi=WHITE, hi_op=0.85, outline=INK)


def f(n):
    s = f"{n:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def ball_geometry():
    """A truncated icosahedron on the unit sphere, turned towards the viewer, projected (y down)."""
    phi = (1 + 5 ** 0.5) / 2
    V = []
    for a in (-1, 1):
        for b in (-phi, phi):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    sub = lambda p, q: tuple(x - y for x, y in zip(p, q))
    dot = lambda p, q: sum(x * y for x, y in zip(p, q))
    cross = lambda p, q: (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])
    norm = lambda p: tuple(x / math.sqrt(dot(p, p)) for x in p)
    lerp = lambda p, q, t: tuple(x + (y - x) * t for x, y in zip(p, q))
    nb = [[j for j in range(len(V)) if j != i and abs(dot(sub(V[i], V[j]), sub(V[i], V[j])) - 4) < 1e-6]
          for i in range(len(V))]
    A, Bt = norm(V[0]), norm((-0.2, 0.26, 0.94))
    k = cross(A, Bt)
    sn, c = math.sqrt(dot(k, k)), dot(A, Bt)
    K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
    K2 = [[sum(K[i][m] * K[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    R = [[(1 if i == j else 0) + K[i][j] + K2[i][j] * (1 - c) / (sn * sn) for j in range(3)] for i in range(3)]
    rot = lambda p: tuple(sum(R[i][m] * p[m] for m in range(3)) for i in range(3))
    third = lambda i, j: norm(rot(lerp(V[i], V[j], 1 / 3)))

    def proj(p):
        if p[2] >= 0:
            return (p[0], -p[1])
        l = math.hypot(p[0], p[1]) or 1
        return (p[0] / l, -p[1] / l)

    arc = lambda p, q, n=8: [norm(lerp(p, q, t / n)) for t in range(n + 1)]
    patches, seams = [], []
    for i, v in enumerate(V):
        cN = norm(rot(v))
        u = cross(cN, (0, 0, 1))
        u = norm(u) if dot(u, u) > 1e-12 else (1, 0, 0)
        w = cross(cN, u)
        pts = sorted((third(i, j) for j in nb[i]), key=lambda p: math.atan2(dot(p, w), dot(p, u)))
        outline = []
        for m, p in enumerate(pts):
            outline += arc(p, pts[(m + 1) % 5])[:-1]
        if cN[2] > -0.35:
            patches.append(dict(z=cN[2], pts=[proj(p) for p in outline]))
    for i in range(len(V)):
        for j in nb[i]:
            if i < j:
                seg = arc(third(i, j), third(j, i))
                if all(p[2] > 0.02 for p in seg):
                    seams.append([proj(p) for p in seg])
    zmax = max(p["z"] for p in patches)
    for p in patches:
        p["front"] = p["z"] == zmax
    return patches, seams


BALL_PATCHES, BALL_SEAMS = ball_geometry()


def ball_svg(x, y, r, base, style=CLASSIC, clip_id="spix-ball", live=False):
    sw = 0.05 * r
    path = lambda pts: "M" + " L".join(f"{f(px * r)} {f(py * r)}" for px, py in pts)
    patches = "".join(
        f'<path d="{path(p["pts"])}Z" fill="{style["front"] if p["front"] else style["patch"]}"/>' for p in BALL_PATCHES)
    seams = " ".join(path(s) for s in BALL_SEAMS)
    R1, cx, cy = r * 1.04, -0.22 * r, -0.26 * r
    shade = (f'<path d="M{f(-r)} 0A{f(r)} {f(r)} 0 1 0 {f(r)} 0A{f(r)} {f(r)} 0 1 0 {f(-r)} 0Z'
             f'M{f(cx - R1)} {f(cy)}A{f(R1)} {f(R1)} 0 1 1 {f(cx + R1)} {f(cy)}A{f(R1)} {f(R1)} 0 1 1 {f(cx - R1)} {f(cy)}Z" '
             f'fill-rule="evenodd" fill="{style["shade"]}" fill-opacity="{style["shade_op"]}"/>')
    hx, hy = -0.42 * r, -0.5 * r
    hi = (f'<ellipse cx="{f(hx)}" cy="{f(hy)}" rx="{f(0.2 * r)}" ry="{f(0.1 * r)}" '
          f'transform="rotate(-40 {f(hx)} {f(hy)})" fill="{style["hi"]}" fill-opacity="{style["hi_op"]}"/>')
    drop = ' class="spx-drop"' if live else ""
    spin = ' class="spx-spin"' if live else ""
    return (f'<g transform="translate({f(x)} {f(y)})"><g{drop}>'
            f'<clipPath id="{clip_id}"><circle r="{f(r)}"/></clipPath>'
            f'<circle r="{f(r)}" fill="{base}"/>'
            f'<g clip-path="url(#{clip_id})"><g{spin} stroke="{style["patch"]}" stroke-linejoin="round" stroke-linecap="round">'
            f'<g stroke-width="{f(sw * 0.6)}">{patches}</g>'
            f'<path d="{seams}" fill="none" stroke-width="{f(sw * 0.9)}"/></g>{shade}{hi}</g>'
            f'<circle r="{f(r - sw / 2)}" fill="none" stroke="{style["outline"]}" stroke-width="{f(sw)}"/>'
            f'</g></g>')


class Font:
    def __init__(self, path):
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.upm = self.tt["head"].unitsPerEm
        self.hmtx = self.tt["hmtx"]

    def bounds(self, g):
        pen = BoundsPen(self.gs)
        self.gs[g].draw(pen)
        return pen.bounds  # xMin, yMin, xMax, yMax in font units

    def path(self, g, scale, x, y):
        pen = SVGPathPen(self.gs, ntos=f)
        self.gs[g].draw(TransformPen(pen, (scale, 0, 0, -scale, x, y)))
        return pen.getCommands()

    def dot(self):
        """Centre x, size and bottom of the i's own dot, in em."""
        stem_top = self.bounds("dotlessi")[3]
        rec = DecomposingRecordingPen(self.gs)
        self.gs["i"].draw(rec)
        contours, cur = [], []
        for op, args in rec.value:
            cur += [p for p in args if p is not None]
            if op in ("closePath", "endPath"):
                contours.append(cur)
                cur = []
        pts = [p for c in contours if min(y for _, y in c) > stem_top for p in c]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        u = self.upm
        return dict(cx=(min(xs) + max(xs)) / 2 / u, w=(max(xs) - min(xs)) / u, h=(max(ys) - min(ys)) / u,
                    bottom=min(ys) / u)


def wordmark(font, fg, ball_base, live=False, clip_id="spix-ball", pad=6, label="spix"):
    x, top, bot, parts = 0.0, 1e9, -1e9, []
    dot = font.dot()
    for L in LANDING:
        g, k = L["ch"], L["k"]
        size = k * B
        scale = size / font.upm
        w = font.hmtx[g][0] * scale
        bl = L.get("dy", 0) * B
        cx = x + w / 2
        sx, sy = L.get("sx", 1), L.get("sy", 1)
        glyph = f'<path d="{font.path(g, scale, x, bl)}"/>'
        if sx != 1 or sy != 1:
            glyph = f'<g transform="translate({f(cx)} {f(bl)}) scale({sx} {sy}) translate({f(-cx)} {f(-bl)})">{glyph}</g>'
        extra = ""
        if g == "dotlessi":
            d = max(dot["w"], dot["h"]) * size * BALL_SCALE
            bx, by = x + dot["cx"] * size, bl - (dot["bottom"] * size * sy + d / 2)
            extra = ball_svg(bx, by, d / 2, ball_base, clip_id=clip_id, live=live)
            top = min(top, by - d / 2)
            if live:
                glyph = f'<g class="spx-stem">{glyph}</g>'
        rot = L.get("r", 0)
        parts.append(f'<g transform="rotate({rot} {f(cx)} {f(bl)})">{glyph}{extra}</g>' if rot else f"<g>{glyph}{extra}</g>")
        b = font.bounds(g)
        top = min(top, bl - b[3] * scale * sy)
        bot = max(bot, bl - b[1] * scale)
        x += w
    vb = (-pad, top - pad, x + 2 * pad, bot - top + 2 * pad)
    view = " ".join(f(v) for v in vb)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" aria-label="{label}">'
            f'<g fill="{fg}">{"".join(parts)}</g></svg>'), vb


def avatar(font):
    """Full name on a turf tile, sized to sit inside Discord's round crop."""
    svg, (vx, vy, vw, vh) = wordmark(font, PAPER, PAPER, clip_id="spix-avatar-ball", pad=0)
    inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
    width = 512 * 0.74
    s = width / vw
    tx, ty = (512 - vw * s) / 2 - vx * s, (512 - vh * s) / 2 - vy * s
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img" aria-label="spix">'
            f'<rect width="512" height="512" fill="{TURF}"/><g transform="translate({f(tx)} {f(ty)}) scale({s:.4f})">{inner}</g></svg>')


def main(font_path):
    font = Font(font_path)
    files = {
        "spix-wordmark-white.svg": wordmark(font, PAPER, PAPER, clip_id="spix-white-ball")[0],
        "spix-wordmark-ink.svg": wordmark(font, INK, WHITE, clip_id="spix-ink-ball")[0],
        "spix-discord-avatar.svg": avatar(font),
        # the header colours its letters with currentColor; the ball keeps its own colours
        "spix-header.svg": wordmark(font, "currentColor", PAPER, live=True, clip_id="spix-header-ball")[0],
    }
    for name, svg in files.items():
        (OUT / name).write_text(svg + "\n")
        print(f"{name}: {len(svg)} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
