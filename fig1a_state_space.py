#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conceptual panel (a): piecewise-linear AL-RNN state space.

Canonical toy example (shared verbatim with fig1b/c/d): P = 4, seven
visited symbols sigma^(1..7) (FP: 2,4,6), plus gray unvisited regions.
One continuous deterministic trajectory realizes the canonical symbol
sequence 1-2-3-5-4-2-3-7-6-7-3 (revisits sigma^(2,3,7); leaves sigma^(3)
through different boundaries on different visits — no literal branching).
Each polygon carries its own affine field f_sigma(x) = A_sigma x +
b_sigma; the three stars are exact equilibria of their local fields and
sigma^(4) is saddle-like. All region membership claims are asserted.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from pathlib import Path
import time
import warnings

# ============================================================
# COMMON CANONICAL TOY EXAMPLE (copied verbatim into fig1a-d)
# ============================================================
P = 4
SYMBOLS = {1: (0, 0, 1, 1), 2: (0, 0, 0, 1), 3: (1, 0, 0, 1),
           4: (0, 1, 0, 1), 5: (1, 1, 0, 1), 6: (1, 0, 1, 0),
           7: (1, 0, 0, 0)}
FP_IDS = {2, 4, 6}
EDGES = [(1, 2), (2, 3), (3, 5), (5, 4), (4, 2),
         (3, 7), (7, 6), (6, 7), (7, 3)]
CANONICAL_SEQ = [1, 2, 3, 5, 4, 2, 3, 7, 6, 7, 3]

C_FP, C_FREE, C_UNVIS = "#cfe3f5", "#fbe0c4", "#e8e8e8"
C_EDGE = "0.2"

SHOW_TITLE = False
SHOW_PANEL_LABEL = False
OUT_DIR = Path("figures/paper")
FIG_BASENAME = "fig1a_state_space"

# common consistency: every edge flips exactly one ReLU bit
for a, b in EDGES:
    assert sum(x != y for x, y in zip(SYMBOLS[a], SYMBOLS[b])) == 1, (a, b)
assert FP_IDS == {2, 4, 6} and len(SYMBOLS) == 7

# ============================================================
# Geometry (manual, reproducible)
# ============================================================
POLY = {
    1: [(0.0, 0.0), (2.3, 0.0), (2.6, 2.6), (0.0, 2.2)],
    2: [(2.3, 0.0), (4.6, 0.0), (4.8, 2.8), (2.6, 2.6)],
    3: [(4.6, 0.0), (7.0, 0.0), (7.2, 3.0), (4.8, 2.8)],
    7: [(7.0, 0.0), (9.2, 0.0), (9.4, 2.7), (7.2, 3.0)],
    6: [(7.2, 3.0), (9.4, 2.7), (9.6, 5.2), (7.4, 5.0)],
    4: [(2.6, 2.6), (4.8, 2.8), (4.9, 5.0), (2.7, 4.9)],
    5: [(4.8, 2.8), (7.2, 3.0), (7.4, 5.0), (4.9, 5.0)],
}
UNVIS_POLY = [
    [(0.0, 2.2), (2.6, 2.6), (2.7, 4.9), (0.0, 4.6)],
    [(0.0, 4.6), (2.7, 4.9), (4.9, 5.0), (7.4, 5.0), (9.6, 5.2),
     (10.0, 5.35), (10.0, 7.0), (0.0, 7.0)],
    [(9.2, 0.0), (10.0, 0.0), (10.0, 5.35), (9.6, 5.2), (9.4, 2.7)],
]
XMAX, YMAX = 10.0, 7.0

LABEL_POS = {1: (0.85, 1.85), 2: (4.0, 0.5), 3: (5.25, 0.5),
             4: (4.35, 4.55), 5: (5.45, 4.55), 6: (7.7, 4.35),
             7: (8.6, 0.55)}

# fixed points (strictly inside their FP polygons, off the trajectory)
FP_POS = {2: np.array([3.15, 0.85]), 4: np.array([3.35, 4.15]),
          6: np.array([8.75, 4.35])}

# trajectory waypoints realizing CANONICAL_SEQ. The route is designed so
# that ONE affine field per region is consistent with every pass:
#   sigma^(3): three passes separated by the shear line y ~ 2.05
#              (rightward below it, leftward above it)
#   sigma^(7): outbound and return passes separated across the diagonal
#              band u = 0.83 x - 0.55 y (up-right vs down-left shear)
#   sigma^(6): counterclockwise loop around its star (spiral field)
WAYPOINTS = np.array([
    (0.9, 0.9), (1.6, 1.15), (2.45, 1.3),                    # s1 -> s2
    (3.4, 1.2), (4.3, 1.35),                                  # s2 pass 1
    (5.0, 1.5), (5.45, 1.6), (5.78, 2.0), (5.84, 2.5),        # s3 pass 1
    (5.8, 3.1), (5.9, 3.65), (5.55, 4.15), (4.95, 4.12),      # s5 arc
    (4.15, 4.2), (3.8, 3.7), (3.66, 2.9),                     # s4 descent
    (3.74, 2.2), (4.1, 1.95), (4.6, 1.88),                    # s2 pass 2
    (5.5, 1.8), (6.3, 1.6), (7.06, 1.37),                     # s3 pass 2
    (8.0, 1.55), (8.6, 2.0), (8.95, 2.72),                    # s7 outbound
    (9.05, 3.3), (9.2, 4.1), (8.85, 4.85), (8.2, 4.75),       # s6 CCW loop
    (7.95, 4.0), (8.08, 3.25),
    (8.0, 2.8), (7.6, 2.5), (7.24, 2.28),                     # s7 return
    (6.85, 2.2), (6.45, 2.13),                                # s3 pass 3
])


def trajectory_spline():
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(WAYPOINTS, axis=0),
                                          axis=1))]
    return make_interp_spline(d / d[-1], WAYPOINTS, k=2)


# ============================================================
# Local affine fields
# ============================================================
def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def make_fields():
    """sigma id (or ('u', k)) -> dict(A, b). Every visited region's affine
    field is LEAST-SQUARES FITTED to the trajectory's unit tangents inside
    that region, so the drawn trajectory is (approximately) an orbit of
    the drawn piecewise field. FP regions fit under the hard constraint
    A x* + b = 0 (star = exact equilibrium). If a fitted FP-free field
    acquires an equilibrium inside its own polygon, b is nudged along the
    mean flow direction until the equilibrium leaves the polygon."""
    spl = trajectory_spline()
    t = np.linspace(0, 1, 3000)
    pts, tans = spl(t), spl.derivative()(t)
    tans = tans / np.linalg.norm(tans, axis=1, keepdims=True)
    regs = region_of(pts)

    f = {}
    for sid in POLY:
        mask = np.array([r == sid for r in regs])
        X, T = pts[mask], tans[mask]
        x_star = FP_POS.get(sid)
        if sid == 7:
            # rank-1 shear: v = q (u - u0) d + delta e, u = n.x. The
            # outbound pass lies at u > u0 (flow along +d, up-right), the
            # return pass at u < u0 (flow along -d, down-left). A is
            # singular and b is chosen off its column space, so this
            # FP-free field has NO equilibrium anywhere by construction.
            n = np.array([0.83, -0.55])
            n = n / np.linalg.norm(n)
            d = np.array([0.8, 0.6])
            d = d / np.linalg.norm(d)
            u = X @ n
            fwd = T @ d > 0
            u0 = 0.5 * (u[fwd].min() + u[~fwd].max())
            e = np.array([-d[1], d[0]])
            A = np.outer(d, n)
            b = -u0 * d + 0.006 * e
            f[sid] = dict(A=A, b=b, x_star=None)
            continue
        if x_star is not None:
            sol, *_ = np.linalg.lstsq(X - x_star, T, rcond=None)
            A = sol.T
            b = -A @ x_star
        else:
            M = np.c_[X, np.ones(len(X))]
            sol, *_ = np.linalg.lstsq(M, T, rcond=None)
            A, b = sol[:2].T, sol[2]
            # keep FP-free regions genuinely FP-free
            pa = MplPath(POLY[sid])
            vbar = T.mean(axis=0)
            vbar = vbar / max(np.linalg.norm(vbar), 1e-9)
            for _ in range(200):
                try:
                    x_eq = np.linalg.solve(A, -b)
                except np.linalg.LinAlgError:
                    break
                if not pa.contains_point(x_eq, radius=0.15):
                    break
                b = b + 0.02 * vbar
        f[sid] = dict(A=A, b=b, x_star=x_star)
    # unvisited regions: mild drift (no trajectory constraint)
    rng = np.random.default_rng(5)
    for k in range(len(UNVIS_POLY)):
        ang = rng.uniform(0, 2 * np.pi)
        A = np.array([[0.04, 0.02], [-0.02, 0.04]])
        f[("u", k)] = dict(A=A, b=0.7 * np.array([np.cos(ang),
                                                  np.sin(ang)]),
                           x_star=None)
    return f


# ============================================================
# Validation
# ============================================================
def region_of(points):
    """region id (or ('u',k) or None) for each point, via polygon paths."""
    paths = {sid: MplPath(POLY[sid]) for sid in POLY}
    upaths = {("u", k): MplPath(p) for k, p in enumerate(UNVIS_POLY)}
    out = []
    for pt in points:
        hit = None
        for sid, pa in paths.items():
            if pa.contains_point(pt):
                hit = sid
                break
        if hit is None:
            for sid, pa in upaths.items():
                if pa.contains_point(pt):
                    hit = sid
                    break
        out.append(hit)
    return out


def validate(fields):
    spl = trajectory_spline()
    t = np.linspace(0, 1, 2500)
    pts, tans = spl(t), spl.derivative()(t)
    tans = tans / np.linalg.norm(tans, axis=1, keepdims=True)
    regs = region_of(pts)
    assert all(r in POLY for r in regs), \
        "trajectory leaves the visited corridor"
    seq = [regs[0]]
    for r in regs[1:]:
        if r != seq[-1]:
            seq.append(r)
    assert seq == CANONICAL_SEQ, f"symbol sequence {seq} != canonical"
    for sid, x in FP_POS.items():
        assert MplPath(POLY[sid]).contains_point(x), \
            f"star {sid} outside its polygon"
        r = np.linalg.norm(fields[sid]["A"] @ x + fields[sid]["b"])
        assert r < 1e-9, f"star {sid} is not an equilibrium"
    ev = np.linalg.eigvals(fields[4]["A"])
    assert ev.real.min() < 0 < ev.real.max(), "sigma^(4) is not a saddle"
    # realizability: along the trajectory the local field must point in
    # the direction of motion (region-wise affine consistency)
    report = []
    for sid in POLY:
        mask = np.array([r == sid for r in regs])
        X, T = pts[mask], tans[mask]
        V = X @ fields[sid]["A"].T + fields[sid]["b"]
        cos = (V * T).sum(1) / np.maximum(np.linalg.norm(V, axis=1), 1e-9)
        report.append((sid, float(np.median(cos)),
                       float((cos > 0.25).mean())))
        assert np.median(cos) >= 0.8, \
            f"sigma^({sid}): field/trajectory median cos {np.median(cos):.2f}"
        assert (cos > 0.25).mean() >= 0.9, \
            f"sigma^({sid}): field opposes motion on " \
            f"{100 * (cos <= 0.25).mean():.0f}% of the pass"
        if FP_POS.get(sid) is None:
            try:
                x_eq = np.linalg.solve(fields[sid]["A"], -fields[sid]["b"])
                assert not MplPath(POLY[sid]).contains_point(x_eq), \
                    f"FP-free sigma^({sid}) acquired an interior equilibrium"
            except np.linalg.LinAlgError:
                pass
    print("checks OK: canonical sequence realized, stars exact "
          "equilibria, sigma^(4) saddle; field/trajectory alignment "
          + " ".join(f"s{sid}:med={m:.2f}/frac={fr:.2f}"
                     for sid, m, fr in report))


# ============================================================
# Plot
# ============================================================
def make_figure(fields):
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        19.5,
        "axes.linewidth":   0.9,
        "figure.dpi":       120,
    })
    FS_SIG, FS_AXIS, FS_LEG = 26, 25.5, 19
    fig = plt.figure(figsize=(10.7, 6.8))
    ax = fig.add_axes([0.042, 0.055, 0.625, 0.93])

    for sid, poly in POLY.items():
        ax.add_patch(Polygon(poly, closed=True, lw=1.0, zorder=1,
                             facecolor=C_FP if sid in FP_IDS else C_FREE,
                             edgecolor=C_EDGE))
    for poly in UNVIS_POLY:
        ax.add_patch(Polygon(poly, closed=True, facecolor=C_UNVIS,
                             edgecolor=C_EDGE, lw=1.0, zorder=1))

    # local affine vector fields
    gx, gy = np.meshgrid(np.arange(0.35, XMAX, 0.62),
                         np.arange(0.35, YMAX, 0.62))
    grid = np.c_[gx.ravel(), gy.ravel()]
    all_regions = [(sid, MplPath(POLY[sid])) for sid in POLY] + \
                  [(("u", k), MplPath(p)) for k, p in enumerate(UNVIS_POLY)]
    for sid, pa in all_regions:
        inside = np.array([pa.contains_point(p, radius=-0.55)
                           for p in grid])
        if not inside.any():
            inside = np.array([pa.contains_point(p, radius=-0.3)
                               for p in grid])
        p = grid[inside]
        f = fields[sid]
        v = p @ f["A"].T + f["b"]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(n, 1e-9) * (0.24 + 0.08 * np.tanh(n))
        faint = 0.45 if isinstance(sid, tuple) else 0.65
        ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="0.55",
                  angles="xy", scale_units="xy", scale=1.0, width=0.0035,
                  headwidth=4.2, headlength=4.8, alpha=faint, zorder=2)

    # trajectory
    t = np.linspace(0, 1, 700)
    pts = trajectory_spline()(t)
    ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.2, zorder=5,
            solid_capstyle="round")
    for frac in (0.06, 0.22, 0.40, 0.58, 0.74, 0.92):
        k = int(frac * (len(t) - 1))
        ax.add_patch(FancyArrowPatch(pts[k], pts[k + 2], arrowstyle="-|>",
                                     mutation_scale=24, color="black",
                                     lw=0, zorder=6))

    # fixed points + symbol labels
    for sid, x in FP_POS.items():
        ax.plot(*x, marker="*", ms=20, color="black", mec="white",
                mew=0.7, ls="", zorder=7)
    for sid, (lx, ly) in LABEL_POS.items():
        ax.text(lx, ly, rf"$\mathbf{{\sigma^{{({sid})}}}}$",
                fontsize=FS_SIG,
                ha="center", va="center", zorder=8, color="0.1")

    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    ax.set_xticks([]); ax.set_yticks([])
    if SHOW_PANEL_LABEL:
        ax.text(0.01, 0.985, "(a)", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    if SHOW_TITLE:
        ax.set_title("AL-RNN state space and symbol types")

    from matplotlib.legend_handler import HandlerBase

    class HandlerArrow(HandlerBase):
        def create_artists(self, legend, h, xd, yd, width, height,
                           fontsize, trans):
            return [FancyArrowPatch((0, height / 2), (width, height / 2),
                                    arrowstyle="-|>",
                                    mutation_scale=h.get_mutation_scale(),
                                    color=h.get_edgecolor(),
                                    lw=h.get_linewidth(),
                                    alpha=h.get_alpha(), transform=trans)]

    traj_arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>",
                                 mutation_scale=26, color="black", lw=2.4)
    vf_arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>",
                               mutation_scale=15, color="0.55", lw=1.3,
                               alpha=0.9)
    handles = [
        Patch(facecolor=C_FP, edgecolor=C_EDGE, lw=0.9),
        Patch(facecolor=C_FREE, edgecolor=C_EDGE, lw=0.9),
        Patch(facecolor=C_UNVIS, edgecolor=C_EDGE, lw=0.9),
        Line2D([], [], marker="*", color="black", ls="", ms=24,
               mec="white", mew=0.7),
        traj_arrow,
        vf_arrow,
        Line2D([], [], color=C_EDGE, lw=1.4),
    ]
    labels = ["Visited FP symbol", "Visited FP-free\nsymbol",
              "Unvisited region", "Fixed point", "Trajectory",
              "Local affine\nvector field", "Region boundary"]
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(0.672, 0.52), frameon=False,
               fontsize=FS_LEG, handlelength=1.9, labelspacing=0.95,
               handletextpad=0.55,
               handler_map={FancyArrowPatch: HandlerArrow()})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{FIG_BASENAME}.{ext}"
        for _ in range(3):
            try:
                fig.savefig(path, **kw)
                break
            except OSError:
                time.sleep(1.0)
        else:
            alt = path.with_name(path.stem + "_new" + path.suffix)
            warnings.warn(f"{path} locked — saved as {alt}")
            fig.savefig(alt, **kw)
    plt.close(fig)
    print(f"saved: {OUT_DIR / FIG_BASENAME}.pdf / .svg / .png")


def main():
    fields = make_fields()
    validate(fields)
    make_figure(fields)


if __name__ == "__main__":
    main()
