#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conceptual panel (b): symbolic transition graph of the canonical toy
example (same seven symbols and directed edge list as fig1a/c/d).

Nodes = visited activation patterns (blue = FP symbol, orange = FP-free),
edges = observed transitions. The layout is manual so cycles (2-3-5-4-2)
and the reciprocal pairs 6<->7, 3<->7 (via 3->7->3) stay visible, and
sigma^(3)'s two outgoing transitions (->5, ->7) are apparent.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
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

C_FP, C_FREE = "#cfe3f5", "#fbe0c4"
C_EDGE = "0.2"

SHOW_TITLE = False
SHOW_PANEL_LABEL = False
OUT_DIR = Path("figures/paper")
FIG_BASENAME = "fig1b_transition_graph"

for a, b in EDGES:
    assert sum(x != y for x, y in zip(SYMBOLS[a], SYMBOLS[b])) == 1, (a, b)
assert FP_IDS == {2, 4, 6} and len(SYMBOLS) == 7
assert {e for e in EDGES if e[0] == 3} == {(3, 5), (3, 7)}, \
    "sigma^(3) must have exactly the two outgoing edges ->5, ->7"

# manual layout (no dynamical meaning; chosen for clarity):
# strict grid, columns x = 0, 1.5, 3.0, 4.5 and rows y = 0, 1.5, so the
# cycle 2-3-5-4-2 is a rectangle and the reciprocal pairs 3<->7 / 7<->6
# run exactly horizontally / vertically
POS = {1: (0.0, 0.0), 2: (1.5, 0.0), 3: (3.0, 0.0), 7: (4.5, 0.0),
       4: (1.5, 1.5), 5: (3.0, 1.5), 6: (4.5, 1.5)}
NODE_R = 0.34
# curvature per edge (reciprocal pairs curve to opposite sides)
CURVE = {(3, 7): 0.22, (7, 3): 0.22, (7, 6): 0.22, (6, 7): 0.22}


def make_figure():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        19.5,
        "figure.dpi":       120,
    })
    FS_SIG, FS_LEG = 21, 16.5
    fig = plt.figure(figsize=(9.9, 4.4))
    ax = fig.add_axes([0.01, 0.02, 0.755, 0.96])
    ax.set_axis_off()

    for (a, b) in EDGES:
        pa, pb = np.array(POS[a]), np.array(POS[b])
        ax.add_patch(FancyArrowPatch(
            pa, pb, arrowstyle="-|>", mutation_scale=17, color="0.25",
            lw=1.5, shrinkA=29, shrinkB=29, zorder=2,
            connectionstyle=f"arc3,rad={CURVE.get((a, b), 0.0)}"))

    for sid, (x, y) in POS.items():
        ax.add_patch(Circle((x, y), NODE_R, zorder=3,
                            facecolor=C_FP if sid in FP_IDS else C_FREE,
                            edgecolor=C_EDGE, lw=1.3))
        if sid in FP_IDS:
            # FP symbol: black star (same mark as fig1a's fixed points),
            # in the upper part of the node; label shifted down a bit
            ax.plot(x, y + 0.175, marker="*", ms=17, color="black",
                    mec="white", mew=0.5, ls="", zorder=5)
            ax.text(x, y - 0.07, rf"$\sigma^{{({sid})}}$", ha="center",
                    va="center", fontsize=FS_SIG, zorder=4)
        else:
            ax.text(x, y, rf"$\sigma^{{({sid})}}$", ha="center",
                    va="center", fontsize=FS_SIG, zorder=4)

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase, HandlerTuple

    class HandlerArrow(HandlerBase):
        def create_artists(self, legend, h, xd, yd, width, height,
                           fontsize, trans):
            return [FancyArrowPatch((0, height / 2), (width, height / 2),
                                    arrowstyle="-|>",
                                    mutation_scale=h.get_mutation_scale(),
                                    color=h.get_edgecolor(),
                                    lw=h.get_linewidth(), transform=trans)]

    edge_arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>",
                                 mutation_scale=18, color="0.25", lw=1.6)
    class HandlerCircleStar(HandlerBase):
        """FP-symbol legend entry: node circle with the star overlaid
        at the SAME center."""
        def create_artists(self, legend, h, xd, yd, width, height,
                           fontsize, trans):
            cx, cy = width / 2, height / 2
            c = Line2D([cx], [cy], marker="o", ls="", ms=24, mfc=C_FP,
                       mec=C_EDGE, mew=1.3, transform=trans)
            s = Line2D([cx], [cy], marker="*", ls="", ms=13,
                       color="black", mec="white", mew=0.5,
                       transform=trans)
            return [c, s]

    fp_handle = (Line2D([], [], marker="o", ls="", ms=24, mfc=C_FP,
                        mec=C_EDGE, mew=1.3),
                 Line2D([], [], marker="*", ls="", ms=13, color="black",
                        mec="white", mew=0.5))
    handles = [
        fp_handle,
        Line2D([], [], marker="o", ls="", ms=24, mfc=C_FREE, mec=C_EDGE,
               mew=1.3),
        edge_arrow,
    ]
    labels = ["FP symbol\n(visited)", "FP-free symbol\n(visited)",
              "Observed\ntransition"]
    fig.legend(handles, labels, loc="center left",
               bbox_to_anchor=(0.77, 0.52), frameon=False,
               fontsize=FS_LEG, handlelength=1.6, labelspacing=1.05,
               handletextpad=0.55,
               handler_map={FancyArrowPatch: HandlerArrow(),
                            tuple: HandlerCircleStar()})
    if SHOW_PANEL_LABEL:
        ax.text(0.0, 0.98, "(b)", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    if SHOW_TITLE:
        ax.set_title("Symbolic transition graph")

    ax.set_xlim(-0.6, 5.3)
    ax.set_ylim(-0.55, 2.05)
    ax.set_aspect("equal")

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


if __name__ == "__main__":
    make_figure()
