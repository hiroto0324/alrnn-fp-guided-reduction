#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conceptual panel (c): linearizing one ReLU merges symbols.

Canonical pair (same toy example as fig1a/b/d, = the {2,3} merge of the
D={1} reduction in fig1d): the FP symbol sigma^(2) = (0,0,0,1) and the
FP-free symbol sigma^(3) = (1,0,0,1) differ ONLY in ReLU activation
bit 1 (asserted). Top: FP region + FP-free region with their switching
boundary (dashed) -> "Linearize ReLU 1" -> one region with the boundary
removed; the merged symbol INHERITS the fixed point (whole region takes
the FP color and keeps the star). Bottom: the corresponding graph
operation. State-space dimensionality is unchanged throughout — only
the switching boundary disappears.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
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
FIG_BASENAME = "fig1c_relu_linearization"

for a, b in EDGES:
    assert sum(x != y for x, y in zip(SYMBOLS[a], SYMBOLS[b])) == 1, (a, b)
diff = [k for k in range(P) if SYMBOLS[2][k] != SYMBOLS[3][k]]
assert diff == [0], "sigma^(2), sigma^(3) must differ only in ReLU bit 1"
assert 2 in FP_IDS and 3 not in FP_IDS, \
    "the pair must be FP + FP-free (valid merge keeping one FP)"
assert (2, 3) in EDGES


def make_figure():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        19.5,
        "figure.dpi":       120,
    })
    FS_SIG, FS_ANN = 21, 18.75
    fig = plt.figure(figsize=(9.9, 6.8))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_axis_off()

    # ── row titles ──
    ax.text(0.05, 5.62, "State space:", fontsize=FS_SIG + 4,
            fontweight="bold", color="0.2")
    ax.text(0.05, 2.15, "Transition graph:", fontsize=FS_SIG + 4,
            fontweight="bold", color="0.2")

    # ── top left: BEFORE — FP region + FP-free region. Fills carry no
    # edges; the shared switching boundary is a RED DASHED line only,
    # the outer contour a solid line ──
    p2 = [(0.0, 3.1), (1.55, 2.9), (1.7, 5.0), (0.15, 5.2)]
    p3 = [(1.55, 2.9), (3.15, 3.15), (3.05, 5.15), (1.7, 5.0)]
    outline = [(0.0, 3.1), (1.55, 2.9), (3.15, 3.15), (3.05, 5.15),
               (1.7, 5.0), (0.15, 5.2)]
    ax.add_patch(Polygon(p2, closed=True, facecolor=C_FP,
                         edgecolor="none", zorder=1))
    ax.add_patch(Polygon(p3, closed=True, facecolor=C_FREE,
                         edgecolor="none", zorder=1))
    ax.add_patch(Polygon(outline, closed=True, facecolor="none",
                         edgecolor=C_EDGE, lw=1.2, zorder=2))
    ax.plot([1.55, 1.7], [2.9, 5.0], color="#c22525", lw=2.0,
            ls=(0, (5, 3)), zorder=3)
    # the fixed point lives in sigma^(2)
    ax.plot(0.8, 4.8, marker="*", ms=17, color="black", mec="white",
            mew=0.6, ls="", zorder=4)
    C_BIT = "#c22525"

    def bits_label(cx, y, first, fs_delta=0.0):
        """=(X,0,0,1) with the single differing FIRST bit emphasized in
        red bold and the shared suffix in black."""
        ax.text(cx - 0.42, y, r"$=($", ha="right", va="center",
                fontsize=FS_ANN + fs_delta)
        ax.text(cx - 0.42, y, rf"$\mathbf{{{first}}}$", ha="left",
                va="center", fontsize=FS_ANN + fs_delta, color=C_BIT)
        ax.text(cx - 0.27, y, r"$,0,0,1)$", ha="left", va="center",
                fontsize=FS_ANN + fs_delta)

    ax.text(0.8, 4.35, r"$\sigma^{(2)}$", ha="center", fontsize=FS_SIG)
    bits_label(0.8, 3.85, 0)
    ax.text(2.4, 4.35, r"$\sigma^{(3)}$", ha="center", fontsize=FS_SIG)
    bits_label(2.4, 3.85, 1)

    # ── center arrow ──
    ax.add_patch(FancyArrowPatch((3.55, 4.15), (4.85, 4.15),
                                 arrowstyle="-|>", mutation_scale=24,
                                 color="0.1", lw=2.0))
    ax.text(4.2, 4.5, "Linearize ReLU 1", ha="center", fontsize=FS_ANN,
            fontweight="bold")

    # ── inset: the activation itself is replaced by a linear function ──
    def mini_axes(cx, cy, w=0.56, h=0.5):
        ax.plot([cx - w, cx + w], [cy, cy], color="0.55", lw=0.9,
                zorder=2)
        ax.plot([cx, cx], [cy - h, cy + h], color="0.55", lw=0.9,
                zorder=2)

    ins_y = 2.5
    mini_axes(3.6, ins_y)                       # ReLU(z1)
    ax.plot([3.6 - 0.5, 3.6, 3.6 + 0.45],
            [ins_y, ins_y, ins_y + 0.45], color="0.1", lw=2.6, zorder=3,
            solid_capstyle="round")
    mini_axes(4.85, ins_y)                      # identity z1
    ax.plot([4.85 - 0.45, 4.85 + 0.45], [ins_y - 0.45, ins_y + 0.45],
            color="0.1", lw=2.6, zorder=3, solid_capstyle="round")
    ax.text(4.22, ins_y, r"$\rightarrow$", ha="center", va="center",
            fontsize=FS_ANN + 3, color="0.1")
    ax.text(3.6, 1.76, r"$\mathrm{ReLU}(z_1)$", ha="center",
            fontsize=FS_ANN - 1.75)
    ax.text(4.85, 1.76, r"$z_1$", ha="center",
            fontsize=FS_ANN - 1.75)

    # ── top right: AFTER — one FP-colored region, boundary removed,
    # the merged symbol inherits the fixed point ──
    merged = [(5.25, 3.1), (6.8, 2.9), (8.4, 3.15), (8.3, 5.15),
              (6.95, 5.0), (5.4, 5.2)]
    ax.add_patch(Polygon(merged, closed=True, facecolor=C_FP,
                         edgecolor=C_EDGE, lw=1.2, zorder=1))
    # star at the SAME location as before the merge (before-panel star
    # at (0.8, 4.8) + the panel translation of +5.25 in x): linearizing
    # a ReLU never moves the fixed point, only the boundary vanishes
    ax.plot(0.8 + 5.25, 4.8, marker="*", ms=17, color="black",
            mec="white", mew=0.6, ls="", zorder=4)
    ax.text(6.82, 4.55,
            r"$\pi_{\{1\}}\!\left(\sigma^{(2)}\right)"
            r"=\pi_{\{1\}}\!\left(\sigma^{(3)}\right)$",
            ha="center", fontsize=FS_ANN + 0.5)
    # concrete binary of the quotient symbol: first bit is a wildcard
    ax.text(6.44, 3.95, r"$=($", ha="right", va="center",
            fontsize=FS_ANN + 0.5)
    ax.text(6.44, 3.95, r"$\mathbf{*}$", ha="left", va="center",
            fontsize=FS_ANN + 0.5, color=C_BIT)
    ax.text(6.6, 3.95, r"$,0,0,1)$", ha="left", va="center",
            fontsize=FS_ANN + 0.5)
    ax.text(6.82, 3.4, "boundary removed", ha="center",
            fontsize=FS_ANN + 1.5, color="0.3", style="italic",
            fontweight="bold")

    # ── bottom: graph interpretation ──
    r = 0.42
    # sigma^(2): FP node (star, fig1b convention); sigma^(3): FP-free
    ax.add_patch(Circle((0.9, 1.05), r, facecolor=C_FP,
                        edgecolor=C_EDGE, lw=1.3, zorder=3))
    ax.plot(0.9, 1.05 + 0.2, marker="*", ms=15, color="black",
            mec="white", mew=0.5, ls="", zorder=5)
    ax.text(0.9, 1.05 - 0.1, r"$\sigma^{(2)}$", ha="center",
            va="center", fontsize=FS_SIG, zorder=4)
    ax.add_patch(Circle((2.4, 1.05), r, facecolor=C_FREE,
                        edgecolor=C_EDGE, lw=1.3, zorder=3))
    ax.text(2.4, 1.05, r"$\sigma^{(3)}$", ha="center", va="center",
            fontsize=FS_SIG, zorder=4)
    # the single observed transition 2 -> 3: red dashed, matching the
    # switching boundary it crosses
    ax.add_patch(FancyArrowPatch((0.9, 1.05), (2.4, 1.05),
                                 arrowstyle="-|>", mutation_scale=22,
                                 color="#c22525", lw=2.0, shrinkA=29,
                                 shrinkB=29, zorder=2,
                                 linestyle=(0, (4, 2.5)),
                                 connectionstyle="arc3,rad=0.15"))
    ax.add_patch(FancyArrowPatch((3.55, 1.05), (4.85, 1.05),
                                 arrowstyle="-|>", mutation_scale=24,
                                 color="0.1", lw=2.0))
    ax.add_patch(Circle((6.3, 1.05), r * 1.25, facecolor=C_FP,
                        edgecolor=C_EDGE, lw=1.6, zorder=3))
    ax.plot(6.3, 1.05 + 0.26, marker="*", ms=15, color="black",
            mec="white", mew=0.5, ls="", zorder=5)
    ax.text(6.3, 1.05 - 0.12, r"$\{\sigma^{(2)}\!,\sigma^{(3)}\}$",
            ha="center", va="center", fontsize=FS_ANN - 1, zorder=4)
    ax.text(6.3, 0.28, "merged quotient node", ha="center",
            fontsize=FS_ANN + 1.5, color="0.3", style="italic",
            fontweight="bold")

    if SHOW_PANEL_LABEL:
        ax.text(0.0, 0.98, "(c)", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
    if SHOW_TITLE:
        ax.set_title("Linearizing one ReLU merges symbols")

    ax.set_xlim(-0.1, 8.6)
    ax.set_ylim(0.0, 6.0)
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
