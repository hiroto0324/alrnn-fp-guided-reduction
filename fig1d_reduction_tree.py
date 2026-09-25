#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conceptual panel (d): fixed-point-guided hierarchical reduction tree.

Same canonical toy example as fig1a/b/c. Root = the seven-symbol parent
graph. Main valid path: D={4} (merges sigma^(3),sigma^(7); |Sigma_D|=6,
P_D=3) -> D={1,4} (exactly Q=3 quotient symbols, one FP each: certified
symbol-minimal; highlighted). Secondary valid path (drawn smaller):
D={1} (5 symbols) -> D={1,3} (also 3 symbols, FPs distinct). Rejected
branch: D={2} merges the DISTINCT FP symbols sigma^(2),sigma^(4) — red X.
Every quotient graph (nodes, colors, edges) is computed from the common
symbol table and edge list by projection; nothing is drawn by hand.
All structural claims are asserted before drawing.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
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
Q = 3

C_FP, C_FREE = "#cfe3f5", "#fbe0c4"
C_EDGE = "0.2"
C_REJ = "#c22525"

SHOW_TITLE = False
SHOW_PANEL_LABEL = False
OUT_DIR = Path("figures/paper")
FIG_BASENAME = "fig1d_reduction_tree"

for a, b in EDGES:
    assert sum(x != y for x, y in zip(SYMBOLS[a], SYMBOLS[b])) == 1, (a, b)
assert FP_IDS == {2, 4, 6} and len(SYMBOLS) == 7

# base node positions: compact version of the panel-(b) STRICT GRID
# (bottom row 1,2,3,7; top row 4 above 2, 5 above 3, 6 above 7)
BASE_POS = {1: (0.0, 0.0), 2: (1.2, 0.0), 3: (2.4, 0.0), 7: (3.6, 0.0),
            4: (1.2, 1.2), 5: (2.4, 1.2), 6: (3.6, 1.2)}
# top-row root labels go beside the node (the grid's vertical edges
# would run straight through a below-node label)
ROOT_LABEL_SIDE = {4: "left", 5: "right", 6: "right"}


# ============================================================
# Quotient machinery (projection of the common definitions)
# ============================================================
def quotient(D):
    """Linearize the (1-indexed) ReLUs in D. Returns classes (frozensets
    of original ids), quotient edges, and validity (no two distinct FP
    symbols merged)."""
    keep = [k for k in range(P) if (k + 1) not in D]
    proj = {i: tuple(SYMBOLS[i][k] for k in keep) for i in SYMBOLS}
    groups = {}
    for i, key in proj.items():
        groups.setdefault(key, set()).add(i)
    classes = [frozenset(g) for g in groups.values()]
    cls_of = {i: c for c in classes for i in c}
    qedges = sorted({(tuple(sorted(cls_of[a])), tuple(sorted(cls_of[b])))
                     for a, b in EDGES if cls_of[a] != cls_of[b]})
    valid = all(len(c & FP_IDS) <= 1 for c in classes)
    return classes, qedges, valid


# ============================================================
# Structural assertions (do not silently draw an inconsistent tree)
# ============================================================
def check_all():
    c4, _, v4 = quotient({4})
    assert v4 and len(c4) == 6
    assert frozenset({3, 7}) in c4, "D={4} must merge sigma^(3),sigma^(7)"

    c2, _, v2 = quotient({2})
    assert not v2, "D={2} must be invalid"
    assert any({2, 4} <= c for c in c2), \
        "D={2} must merge the FP symbols sigma^(2),sigma^(4)"

    c14, _, v14 = quotient({1, 4})
    assert v14 and len(c14) == Q, "D={1,4} must give exactly Q=3 symbols"
    assert all(len(c & FP_IDS) == 1 for c in c14), \
        "each final class must contain exactly one FP symbol"

    c1, _, v1 = quotient({1})
    assert v1 and len(c1) == 5

    c13, _, v13 = quotient({1, 3})
    assert v13 and len(c13) == Q
    assert all(len(c & FP_IDS) == 1 for c in c13)
    print("checks OK: D={4} merges (3,7) -> 6 symbols; D={2} rejected "
          "(merges FP 2,4); D={1,4} and D={1,3} reach |Sigma_D| = Q = 3 "
          "with distinct FPs")


# ============================================================
# Drawing helpers
# ============================================================
def draw_quotient(ax, D, cx, cy, scale=1.0, reject=False,
                  node_r=0.17, fs=12.75, min_sep=3.2, manual_pos=None,
                  node_scale=None):
    """Draw the quotient graph for D centered at (cx, cy). Node position
    = mean of member base positions (visually trackable merging)."""
    classes, qedges, valid = quotient(D)
    base = np.array(list(BASE_POS.values()))
    center = base.mean(axis=0)
    pos = {}
    for c in classes:
        pts = np.array([BASE_POS[i] for i in c])
        pos[tuple(sorted(c))] = (pts.mean(axis=0) - center) * scale \
            + np.array([cx, cy])
    # mean positions of merged classes can nearly coincide — push
    # overlapping nodes apart (deterministic pairwise repulsion)
    if manual_pos is not None:
        for key, off in manual_pos.items():
            pos[key] = np.array([cx, cy]) + np.array(off)
    keys = list(pos)
    min_d = min_sep * node_r * scale
    for _ in range(80):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = pos[keys[j]] - pos[keys[i]]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    push = (d / max(dist, 1e-6)) * (min_d - dist) / 2
                    pos[keys[i]] = pos[keys[i]] - push
                    pos[keys[j]] = pos[keys[j]] + push
                    moved = True
        if not moved:
            break
    for a, b in qedges:
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], arrowstyle="-|>", mutation_scale=10,
            color="0.35", lw=0.9, shrinkA=12, shrinkB=12, zorder=2,
            connectionstyle="arc3,rad=0.15"))
    # node radius decoupled from the layout scale so all graphs in one
    # row can share the same node size (node_scale); merged classes stay
    # 1.25x
    ns = node_scale if node_scale is not None else scale
    for c in classes:
        key = tuple(sorted(c))
        x, y = pos[key]
        n_fp = len(c & FP_IDS)
        fc = C_FP if n_fp >= 1 else C_FREE
        ec = C_REJ if n_fp >= 2 else C_EDGE
        r = node_r * ns * (1.25 if len(c) > 1 else 1.0)
        ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec,
                            lw=2.0 if n_fp >= 2 else 1.1, zorder=3))
        if n_fp >= 1:
            ax.plot(x, y + 0.02 * ns, marker="*", ms=10.5 * ns,
                    color="black", mec="white", mew=0.4, ls="", zorder=5)
        if len(c) > 1:
            lab_tex = (r"$\{\sigma^{(" + r")},\sigma^{(".join(
                str(i) for i in sorted(c)) + r")}\}$")
            col = C_REJ if n_fp >= 2 else "0.2"
            if c == frozenset({4, 5}):
                # the {4,5} node always sits above a vertical reciprocal
                # edge pair — put its label on the LEFT of the node so
                # the edges never run through the text
                ax.text(x - r - 0.08 * scale, y, lab_tex, ha="right",
                        va="center", fontsize=fs * scale, zorder=6,
                        color=col)
            elif c == frozenset({2, 4}):
                # rejected merge: label above the node, clear of the
                # neighbouring {3,5} label at the same height
                ax.text(x, y + r + 0.08 * scale, lab_tex, ha="center",
                        va="bottom", fontsize=fs * scale, zorder=6,
                        color=col)
            else:
                ax.text(x, y - r - 0.10 * scale, lab_tex, ha="center",
                        va="top", fontsize=fs * scale, zorder=6,
                        color=col)
        elif not D:
            i = next(iter(c))
            side = ROOT_LABEL_SIDE.get(i)
            if side == "left":
                ax.text(x - r - 0.09, y, rf"$\sigma^{{({i})}}$",
                        ha="right", va="center", fontsize=fs, zorder=6,
                        color="0.2")
            elif side == "right":
                ax.text(x + r + 0.09, y, rf"$\sigma^{{({i})}}$",
                        ha="left", va="center", fontsize=fs, zorder=6,
                        color="0.2")
            else:
                ax.text(x, y - r - 0.08, rf"$\sigma^{{({i})}}$",
                        ha="center", va="top", fontsize=fs, zorder=6,
                        color="0.2")
    return pos


def tree_arrow(ax, p0, p1, label, reject=False, fs=16.5):
    col = C_REJ if reject else "0.15"
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=16, color=col, lw=1.5,
                                 zorder=1))
    mid = 0.5 * (np.array(p0) + np.array(p1))
    ax.text(mid[0] + 0.12, mid[1], label, fontsize=fs, ha="left",
            va="center", color=col)
    if reject:
        ax.text(*mid, r"$\times$", fontsize=26, ha="center", va="center",
                color=C_REJ, fontweight="bold", zorder=4)


# ============================================================
def make_figure():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        19.5,
        "figure.dpi":       120,
    })
    FS_CAP = 18
    fig = plt.figure(figsize=(9.9, 8.6))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_axis_off()

    # ── row labels: P / P_D is a property of the reduction LEVEL, not
    # of the individual graphs — one label per row on the left margin ──
    for y, lab in ((9.35, r"$P=4$"), (5.55, r"$P_{\mathcal{D}}=3$"),
                   (2.0, r"$P_{\mathcal{D}}=2$")):
        ax.text(-1.0, y, lab, ha="center", va="center", rotation=90,
                fontsize=FS_CAP + 4, color="0.15",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.35", lw=1.2))

    # ── root ──
    draw_quotient(ax, set(), 4.6, 9.35, scale=1.0)
    ax.text(4.6, 10.4,
            r"$\mathcal{D}=\varnothing$:  $|\Sigma|=7$",
            ha="center", fontsize=FS_CAP)

    # ── level 1 (left to right: D={1}, D={2} rejected, D={4}) ──
    draw_quotient(ax, {1}, 1.25, 5.55, scale=0.78, node_scale=0.85,
                  fs=12.75 / 0.78)
    ax.text(1.25, 6.5, r"$\mathcal{D}=\{1\}$:  $|\Sigma_{\mathcal{D}}|=5$",
            ha="center", fontsize=FS_CAP, color="0.3")
    draw_quotient(ax, {2}, 4.55, 5.7, scale=0.78, reject=True,
                  node_scale=0.85, fs=12.75 / 0.78)
    ax.text(4.55, 6.6, r"$\mathcal{D}=\{2\}$", ha="center",
            fontsize=FS_CAP, color=C_REJ)
    ax.text(4.55, 4.5, "reject: merges distinct\nFP symbols",
            ha="center", fontsize=15.75, color=C_REJ, style="italic",
            fontweight="bold")
    draw_quotient(ax, {4}, 8.0, 5.4, scale=0.9, node_scale=0.85,
                  fs=12.75 / 0.9)
    ax.text(8.0, 6.55,
            r"$\mathcal{D}=\{4\}$:  $|\Sigma_{\mathcal{D}}|=6$",
            ha="center", fontsize=FS_CAP)

    # ── level 2 (children under their parents) ──
    # D={1,3} reaches |Sigma_D| = Q with distinct FPs as well — it is an
    # equally certified minimal leaf (drawn smaller as the secondary
    # path, but boxed like the main one)
    # fs compensates the smaller scale so both bottom-row graphs use the
    # same effective class-label font size (12.75 pt)
    draw_quotient(ax, {1, 3}, 1.7, 2.0, scale=0.72, min_sep=4.5,
                  fs=12.75 / 0.72, node_scale=1.0)
    ax.text(1.7, 3.1,
            r"$\mathcal{D}=\{1,3\}$:  $|\Sigma_{\mathcal{D}}|=3=Q$",
            ha="center", fontsize=FS_CAP, color="0.3")
    ax.text(1.7, 0.68, "certified minimal", ha="center",
            fontsize=FS_CAP, style="italic", color="0.3",
            fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.1, 0.35), 3.3, 3.05,
                                boxstyle="round,pad=0.12",
                                facecolor="none", edgecolor="0.35",
                                lw=1.4, ls=(0, (4, 2)), zorder=0))
    draw_quotient(ax, {1, 4}, 7.35, 1.7, scale=1.0, min_sep=0.0,
                  manual_pos={(1, 6): (-1.05, 0.05),
                              (2, 3, 7): (0.85, -0.35),
                              (4, 5): (0.0, 0.95)})
    ax.text(7.35, 3.15,
            r"$\mathcal{D}=\{1,4\}$:  $|\Sigma_{\mathcal{D}}|=3=Q$",
            ha="center", fontsize=FS_CAP, fontweight="bold")
    ax.text(7.35, 0.5, "certified minimal", ha="center",
            fontsize=FS_CAP, style="italic", color="0.2",
            fontweight="bold")
    # box hugging the content (title + graph + labels)
    ax.add_patch(FancyBboxPatch((5.4, 0.3), 3.95, 3.15,
                                boxstyle="round,pad=0.12",
                                facecolor="none", edgecolor="0.35",
                                lw=1.4, ls=(0, (4, 2)), zorder=0))

    # ── tree arrows ──
    # edge labels hug their arrows (as close as possible without
    # touching the graphs or crossing the arrow line)
    tree_arrow(ax, (3.4, 8.25), (1.7, 6.75), "", reject=False)
    ax.text(1.9, 7.25, "linearize\nReLU 1", fontsize=15.75, ha="center",
            color="0.3")
    tree_arrow(ax, (4.6, 8.1), (4.55, 6.85), "", reject=True)
    ax.text(5.25, 7.2, "linearize\nReLU 2", fontsize=15.75, ha="center",
            color=C_REJ)
    tree_arrow(ax, (5.8, 8.3), (7.45, 6.7), "", reject=False)
    ax.text(7.38, 7.42, "linearize\nReLU 4", fontsize=15.75, ha="center",
            color="0.15")
    tree_arrow(ax, (1.3, 4.75), (1.6, 3.45), "", reject=False)
    ax.text(2.11, 4.0, "linearize\nReLU 3", fontsize=15.75, ha="center",
            color="0.3")
    tree_arrow(ax, (8.0, 4.45), (7.45, 3.55), "", reject=False)
    ax.text(8.5, 3.8, "linearize\nReLU 1", fontsize=15.75, ha="center",
            color="0.15")

    if SHOW_PANEL_LABEL:
        ax.text(0.0, 0.99, "(d)", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    if SHOW_TITLE:
        ax.set_title("Fixed-point-guided reduction tree")

    ax.set_xlim(-1.55, 9.95)
    ax.set_ylim(0.0, 10.75)
    ax.set_aspect("equal")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {"transparent": True}),
                    ("svg", {"transparent": True}),
                    ("png", {"dpi": 300, "transparent": True})):
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


if __name__ == "__main__":
    check_all()
    make_figure()
