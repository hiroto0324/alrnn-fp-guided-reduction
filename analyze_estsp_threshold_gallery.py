#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blinded gallery of free-run attractors around the E_stsp threshold.

Pools MINIMAL-success runs (Q_vis=5 & |Sigma_hat|=5) from every condition
(sync + weighting aggregates), picks the run whose E_stsp is closest to
each target in E_TARGETS, and shows — labelled ONLY by its E value —
  row 1: x-y projection of the 10k free run (1k transient cut)
  row 2: x-z projection
  row 3: power spectrum of x (model, colored) vs raw data (gray)
Leftmost column: the raw Chua data itself. The condition identities are
printed to the console (the figure stays blind) so the visual judgement
of "where does it stop looking like Chua" cannot anchor on method names.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freerun_cache as fc
import plot_retraining_success_hierarchy as panel_c
from metrics import compute_and_smooth_power_spectrum

SYNC_AGG = Path("results/aug_only/"
                "summary_retrain_sync_scope_ablation_p10_v1_aggregate.csv")
WEIGHT_AGG = Path("results/aug_only/"
                  "summary_retrain_weighting_ablation_p10_sync_v1_"
                  "aggregate.csv")
OUT_DIR = Path("figures/analysis")
FIG_BASENAME = "estsp_threshold_gallery"
Q = 5
E_TARGETS = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
CUT = 1000
C_MODEL = "#1f5fa8"


def pool_minimal_runs():
    """Minimal-success runs with POST-TRANSIENT E_stsp (official
    fidelity convention 2026-09-25); reuses the appendix-ablation
    per-model cache so no rollout is recomputed."""
    import plot_chua_appendix_ablations as abl
    runs = []
    for r in csv.DictReader(open(SYNC_AGG, newline="")):
        if (r.get("run_status") == "completed"
                and int(r["Q_vis"]) == Q
                and int(r["n_posttransient_symbols"]) == Q):
            e_cut = abl.model_metrics(r["ckpt"])[0]
            runs.append((e_cut, r["ckpt"],
                         f"{r.get('guidance')}/{r.get('sync_scope')}"
                         f"/{r.get('teacher')} seed{r['source_seed']}"
                         f" cand{r['candidate_id']}"))
    for r in csv.DictReader(open(WEIGHT_AGG, newline="")):
        if (r.get("run_status") == "completed"
                and int(r["Q_vis"]) == Q
                and int(r["n_posttransient_symbols"]) == Q):
            e_cut = abl.model_metrics(r["ckpt"])[0]
            runs.append((e_cut, r["ckpt"],
                         f"{r['condition']} seed{r['source_seed']}"
                         f" cand{r['candidate_id']}"))
    abl._save_fid_cache()
    return [x for x in runs if np.isfinite(x[0])]


def main():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        12,
        "figure.dpi":       120,
    })
    runs = pool_minimal_runs()
    chosen, used = [], set()
    for t in E_TARGETS:
        best = min((x for x in runs if x[1] not in used),
                   key=lambda x: abs(x[0] - t))
        chosen.append(best)
        used.add(best[1])

    raw = np.load(fc.RAW_DATA_PATH).astype(np.float32)
    n_cols = 1 + len(chosen)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.4 * n_cols, 7.4),
                             constrained_layout=True)

    raw_bg = raw[CUT:40000]

    def draw_col(j, xyz, title, color, n_steps=9000):
        # model columns overlay the 10k free run (9k post-transient)
        # on the raw trajectory (40k, light gray); raw column: data only
        seg = xyz[:n_steps]
        if color != "0.3":
            for i, (a, b) in enumerate(((0, 1), (0, 2))):
                axes[i, j].plot(raw_bg[:, a], raw_bg[:, b], lw=0.15,
                                color="0.78", alpha=0.6, zorder=1)
        axes[0, j].plot(seg[:, 0], seg[:, 1], lw=0.25, color=color,
                        alpha=0.8, zorder=2)
        axes[1, j].plot(seg[:, 0], seg[:, 2], lw=0.25, color=color,
                        alpha=0.8, zorder=2)
        ps_raw = compute_and_smooth_power_spectrum(raw[:10000, 0], 20)
        axes[2, j].semilogy(ps_raw, color="0.65", lw=1.0)
        if color != "0.3":
            ps = compute_and_smooth_power_spectrum(xyz[:10000, 0], 20)
            axes[2, j].semilogy(ps, color=color, lw=1.0, alpha=0.9)
        axes[2, j].set_xlim(0, 800)
        axes[2, j].set_ylim(1e-9, 1e-1)
        axes[0, j].set_title(title, fontsize=12)
        for i in range(3):
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_color("0.7")

    draw_col(0, raw[CUT:], "raw data (40k)", "0.3", n_steps=39000)
    key = []
    for j, (e, ck, ident) in enumerate(chosen, start=1):
        # for display, build a 40k free run (same length as the raw
        # data) in a separate cache (E labels keep the official 10k values)
        rec = fc.freerun_record(ck, steps=40000)
        draw_col(j, rec["readout"][CUT:].astype(float),
                 f"$E_{{\\mathrm{{stsp}}}}={e:.2f}$", C_MODEL,
                 n_steps=39000)
        key.append(f"  E={e:.3f}: {ident}")

    axes[0, 0].set_ylabel("$x$–$y$", fontsize=13)
    axes[1, 0].set_ylabel("$x$–$z$", fontsize=13)
    axes[2, 0].set_ylabel("PS($x$)", fontsize=13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 250})):
        panel_c.savefig_robust(fig, OUT_DIR / f"{FIG_BASENAME}.{ext}",
                               bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"saved: {OUT_DIR / FIG_BASENAME}.pdf / .svg / .png")
    print("\n[BLIND KEY] (identities, figure shows only E):")
    for k in key:
        print(k)

    # ── multi-sample x-y gallery: rows = E levels (incl. clear
    # failures E>3, >4), 3 samples per level as columns ──
    targets2 = [1.5, 2.0, 3.0, 4.0]
    per_bin = 4
    chosen2, used2 = {}, set()
    for t in targets2:
        picks = []
        for _ in range(per_bin):
            pool = [x for x in runs if x[1] not in used2]
            if not pool:
                break
            best = min(pool, key=lambda x: abs(x[0] - t))
            picks.append(best)
            used2.add(best[1])
        chosen2[t] = picks
    fig2, axes2 = plt.subplots(len(targets2), per_bin,
                               figsize=(2.4 * per_bin,
                                        2.45 * len(targets2)),
                               constrained_layout=True)
    key2 = []
    for rrow, t in enumerate(targets2):
        for c, (e, ck, ident) in enumerate(chosen2[t]):
            ax = axes2[rrow, c]
            rec = fc.freerun_record(ck, steps=40000)
            seg = rec["readout"][CUT:40000].astype(float)
            ax.plot(raw_bg[:, 0], raw_bg[:, 1], lw=0.12, color="0.78",
                    alpha=0.6, zorder=1)
            ax.plot(seg[:, 0], seg[:, 1], lw=0.12, color=C_MODEL,
                    alpha=0.75, zorder=2)
            ax.set_title(f"$E_{{\\mathrm{{stsp}}}}={e:.2f}$",
                         fontsize=15.5)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("0.7")
            key2.append(f"  E={e:.3f}: {ident}")
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 250})):
        panel_c.savefig_robust(fig2,
                               OUT_DIR / f"{FIG_BASENAME}_multi.{ext}",
                               bbox_inches="tight", **kw)
    plt.close(fig2)
    print(f"saved: {OUT_DIR / FIG_BASENAME}_multi.pdf / .svg / .png")
    print("\n[BLIND KEY 2]:")
    for k in key2:
        print(k)


if __name__ == "__main__":
    main()
