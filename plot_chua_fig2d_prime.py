#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 2(d'): Chua direct P=3 vs reduction-guided retrained
P_eff=3 vs parent P=10, Fig. 2(c) inherited style.

Left  — structural success rates (FP recovery / minimal realization /
        + E_stsp < 2), SEED-MACRO convention: every parent seed carries
        one vote, split equally over its retrained candidates
        (candidate-less parents count as failures, denominator = 30
        parents). Direct is per-seed as usual. Parent P=10 is omitted
        from this panel (user, 2026-09-23).
Right — E_stsp distribution over ALL models, PARENT-BALANCED weighting
        for the retrained condition (weighted-quantile box, jitter
        marker area ~ candidate weight), all three conditions.

Data/conventions come from plot_chua_direct_retrained_dist (baseline
retained + readout_retained sync + parent teacher + alpha=0.1;
capacity-matched P_effective=3 subset; official freerun metrics).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_retraining_success_hierarchy as panel_c
import plot_chua_direct_retrained_dist as base

OUT_DIR = panel_c.OUT_DIR
FIG_BASENAME = "chua_fig2d_prime"
N_PARENTS = 30
ESTSP_THRESHOLD = 2.0

CRITERIA = [("fp", "FP recovery"),
            ("mn", "Minimal realization"),
            ("fid", r"+ $E_{\mathrm{stsp}} < 2.0$")]


def crit_flags(d):
    e = d["E"]
    fid = d["ok"] & np.isfinite(e) & (e < ESTSP_THRESHOLD)
    return dict(fp=d["fp"], mn=d["ok"], fid=fid)


def macro_rate(d, flag):
    """Seed-macro success rate over the 30-parent universe."""
    return 100.0 * float(np.sum(d["w"] * flag)) / N_PARENTS


def main():
    FS_LABEL, FS_XTICK, FS_PCT, FS_LEGEND = 20.6, 18.75, 15.4, 16.5
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        12.5,
        "axes.labelsize":   FS_LABEL,
        "xtick.labelsize":  FS_XTICK,
        "ytick.labelsize":  12.5,
        "axes.linewidth":   0.8,
        "figure.dpi":       120,
    })

    d3 = base.load_direct(3)
    # MAIN protocol (2026-09-24): full guidance + coord-equal fixed
    # lambda + full-state forcing
    rt = base.load_retrained(p_eff=3, condition="full_fixed")
    p10 = base.load_direct(10)
    conds3 = [(r"Direct $P{=}3$", "white", d3),
              (r"Retrained $P_{\mathcal{D}}{=}3$", "#dbe7f5", rt),
              (r"Parent $P{=}10$", "#e8e8e8", p10)]

    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ── left: seed-macro success rates (P=10 omitted) ──
    bar_conds = conds3[:2]
    n_m = len(bar_conds)
    width = 0.30
    x0 = np.arange(len(CRITERIA))
    for i, (label, color, d) in enumerate(bar_conds):
        flags = crit_flags(d)
        vals = [macro_rate(d, flags[c]) for c, _ in CRITERIA]
        print(f"{label}: " + "  ".join(
            f"{lab}={v:.1f}%" for (_, lab), v in zip(CRITERIA, vals)))
        xs = x0 + (i - (n_m - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=color, edgecolor="0.15",
               lw=0.8, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=FS_PCT,
                        color="0.15", zorder=4)
    ax.set_xticks(x0)
    ax.set_xticklabels([lab for _, lab in CRITERIA], fontsize=12.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_LEGEND * 0.8, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.1, handlelength=1.1, handletextpad=0.45)

    # ── right: parent-balanced E_stsp distribution (all 3 conds) ──
    rng = np.random.default_rng(0)
    for j, (label, color, d) in enumerate(conds3):
        v, w = d["E"], d["w"]
        fin = np.isfinite(v)
        vf, wf = v[fin], w[fin]
        q1, med, q3 = base.wquantile(vf, wf, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        wlo = vf[vf >= q1 - 1.5 * iqr].min()
        whi = vf[vf <= q3 + 1.5 * iqr].max()
        print(f"{label}: E med {med:.3f} IQR [{q1:.3f}, {q3:.3f}]  "
              f"n={len(vf)}")
        axd.bxp([dict(med=med, q1=q1, q3=q3, whislo=wlo, whishi=whi,
                      label="")],
                positions=[j], widths=0.5, showfliers=False,
                patch_artist=True,
                medianprops=dict(color="black", lw=1.6),
                whiskerprops=dict(color="0.3"),
                capprops=dict(color="0.3"),
                boxprops=dict(facecolor=color, alpha=0.9,
                              edgecolor="0.15", lw=0.8))
        x = j + rng.uniform(-0.16, 0.16, len(vf))
        ms = 3.4 * np.sqrt(wf / wf.max())
        axd.scatter(x, vf, s=ms ** 2, c="0.25", edgecolors="none",
                    alpha=0.4, zorder=3)
    axd.axhline(ESTSP_THRESHOLD, color="0.35", lw=1.0, ls=(0, (4, 3)),
                zorder=2)
    axd.text(2.42, ESTSP_THRESHOLD * 1.07,
             "high-fidelity threshold "
             r"($E_{\mathrm{stsp}}{=}2.0$)",
             ha="right", va="bottom", fontsize=10.5, color="0.35",
             zorder=2)
    axd.set_title("All models", fontsize=16.5, color="0.15")
    axd.set_yscale("log")
    axd.set_xticks(range(3))
    axd.set_xticklabels([r"Direct" "\n" r"$P{=}3$",
                         "Retrained\n" r"$P_{\mathcal{D}}{=}3$",
                         r"Parent" "\n" r"$P{=}10$"], fontsize=12.5)
    axd.set_ylabel(r"$E_{\mathrm{stsp}}$", fontsize=FS_LABEL)
    axd.grid(axis="y", which="both", color="0.9", lw=0.6)
    axd.set_axisbelow(True)
    axd.spines[["top", "right"]].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {"bbox_inches": "tight"}),
                    ("svg", {"bbox_inches": "tight"}),
                    ("png", {"dpi": 300, "bbox_inches": "tight"})):
        panel_c.savefig_robust(fig, OUT_DIR / f"{FIG_BASENAME}.{ext}",
                               **kw)
    plt.close(fig)
    print(f"saved: {OUT_DIR / FIG_BASENAME}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
