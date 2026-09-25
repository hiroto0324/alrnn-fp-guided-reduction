#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chua: E_stsp (left) and E_H (right) distributions over ALL models
for Direct P=3 / Reduction-guided retrained P_eff=3 / Parent P=10,
with the parent-balanced statistical convention (2026-09-23).
E_stsp/E_H use the POST-TRANSIENT rollout (first 1000 steps dropped;
official fidelity convention 2026-09-25, freerun_cache
.posttransient_fidelity):

- main figure: parent-balanced candidate distribution (each source
  parent totals weight 1, 1/|C_s| per candidate; weighted-quantile
  boxes, jitter marker area ~ weight); *_raw = unweighted appendix
- summary: per-seed means over matched seeds
- test: parent-seed cluster bootstrap of the parent-balanced median
  difference (retrained - direct)
- parent-balanced H(tau) = P(S_min & E<tau) with S_min = (Q_vis=5 and
  |Sigma_hat|=5), failures at +inf, candidate-less parents contributing
  0 (denominator = all 30 parents); plus the S_min-conditioned E
  distribution (also parent-balanced)

Retrained condition = the baseline protocol (retained guidance +
readout_retained sync + parent free-run teacher, auto alpha=0.1) from
the sync-scope aggregate CSV. The headline uses the capacity-matched
P_effective=3 subset; all-candidate numbers are printed alongside.
Metrics: official freerun protocol via freerun_cache (Chua defaults).
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freerun_cache as fc
import plot_retraining_success_hierarchy as panel_c

OUT_DIR = panel_c.OUT_DIR
FIG_BASENAME = "chua_direct_retrained_dist"
AGG = Path("results/aug_only/"
           "summary_retrain_sync_scope_ablation_p10_v1_aggregate.csv")
CHUA_TAG = ("chua_orig_nint128_bs16_sig0.0_lr1e-03-1e-05"
            "_tfp0_ramp200_hLR1_m20_p{P}_seed{seed}")
Q = 5                    # target FPs = published minimal symbols
N_PARENTS = 30
N_BOOT = 10000
TAUS = [1.0, 1.5, 2.0, 3.0, 5.0, np.inf]


def wquantile(v, w, q):
    v, w = np.asarray(v, float), np.asarray(w, float)
    i = np.argsort(v)
    v, w = v[i], w[i]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


def load_direct(p):
    e, eh, ok, fp = [], [], [], []
    for s in range(30):
        rec = fc.freerun_record(
            "models/" + CHUA_TAG.format(P=p, seed=s) + ".pth")
        # post-transient fidelity convention (user, 2026-09-25)
        e_c, eh_c = fc.posttransient_fidelity(rec)
        e.append(e_c)
        eh.append(eh_c)
        fp.append(int(rec["Q_vis"]) == Q)
        ok.append(int(rec["Q_vis"]) == Q
                  and int(rec["n_posttransient_symbols"]) == Q)
    return dict(E=np.array(e), EH=np.array(eh), fp=np.array(fp),
                ok=np.array(ok), w=np.ones(30), seed=np.arange(30))


SELECTION_JSON = Path("results/relu_hierarchy/retraining_specs/"
                      "global_min_retrain_selection_original_nint128"
                      "_m20_p10.json")
# the weighting-ablation aggregate holds the fixed-lambda cells,
# including the paper's MAIN protocol (full + coord-equal fixed lambda
# + full-state forcing, decided 2026-09-24)
WAGG = Path("results/aug_only/"
            "summary_retrain_weighting_ablation_p10_sync_v1"
            "_aggregate.csv")


def _row_match(condition, r):
    if r.get("run_status") != "completed":
        return False
    if condition == "retained_auto":
        return (r.get("guidance") == "retained"
                and r.get("sync_scope") == "readout_retained"
                and r.get("teacher") == "parent_free_run")
    if condition == "full_fixed":
        return (r.get("distill_mode") == "full_preactivation"
                and r.get("sync_scope") == "full"
                and r.get("weighting_scheme") == "coord_equal_fixed")
    raise ValueError(condition)


def load_retrained(p_eff=None, condition="retained_auto"):
    src = WAGG if condition == "full_fixed" else AGG
    e, eh, ok, fp, seed = [], [], [], [], []
    for r in csv.DictReader(open(src, newline="")):
        if not _row_match(condition, r):
            continue
        if p_eff is not None and int(float(r["P_effective"])) != p_eff:
            continue
        rec = fc.freerun_record(r["ckpt"])
        # post-transient fidelity convention (user, 2026-09-25)
        e_c, eh_c = fc.posttransient_fidelity(rec)
        e.append(e_c)
        eh.append(eh_c)
        fp.append(int(rec["Q_vis"]) == Q)
        ok.append(int(rec["Q_vis"]) == Q
                  and int(rec["n_posttransient_symbols"]) == Q)
        seed.append(int(float(r["source_seed"])))
    seed = np.array(seed)
    # parent-balanced weights over the SELECTED candidate counts (full
    # 141-run selection), so candidates that failed before producing a
    # row (calibration NaN) count as zero contribution instead of
    # silently upweighting their siblings
    import json
    sel = json.load(open(SELECTION_JSON))
    cands = sel.get("selected_candidates") or sel.get("candidates")
    n_sel = {}
    for c in cands:
        if p_eff is not None and int(c["P_effective"]) != p_eff:
            continue
        n_sel[int(c["source_seed"])] = \
            n_sel.get(int(c["source_seed"]), 0) + 1
    w = np.array([1.0 / n_sel[s] for s in seed])
    return dict(E=np.array(e), EH=np.array(eh), ok=np.array(ok),
                fp=np.array(fp), w=w, seed=seed)


def cluster_bootstrap(rt, dd, rng):
    seeds = np.unique(rt["seed"])
    by_seed = {s: rt["E"][rt["seed"] == s] for s in seeds}
    d_by_seed = {s: dd["E"][dd["seed"] == s][0] for s in seeds}
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        rv = np.concatenate([by_seed[s] for s in pick])
        rw = np.concatenate([np.full(len(by_seed[s]),
                                     1.0 / len(by_seed[s]))
                             for s in pick])
        dv = np.array([d_by_seed[s] for s in pick])
        diffs[b] = wquantile(rv, rw, 0.5) - np.median(dv)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    obs = (wquantile(rt["E"], rt["w"], 0.5)
           - np.median([d_by_seed[s] for s in seeds]))
    return obs, lo, hi, p


def h_tau(cond, tau, parent_balanced):
    """P(S_min & E<tau) over the 30-parent universe."""
    e, ok = cond["E"], cond["ok"]
    hit = ok & np.isfinite(e) & (e < tau)
    if not parent_balanced:
        return hit.mean() * len(e) / N_PARENTS if len(e) == 30 \
            else hit.mean()
    return float(np.sum(cond["w"] * hit)) / N_PARENTS


def draw(data, conds, weighted, suffix):
    FS_LABEL = 20.6
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4),
                             constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, key, ylab in ((axes[0], "E", r"$E_{\mathrm{stsp}}$"),
                          (axes[1], "EH", r"$E_H$")):
        for j, ((label, color), d) in enumerate(zip(conds, data)):
            v, w = d[key], d["w"]
            fin = np.isfinite(v)
            vf = v[fin]
            wf = (w[fin] if weighted else np.ones(fin.sum()))
            q1, med, q3 = wquantile(vf, wf, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            wlo = vf[vf >= q1 - 1.5 * iqr].min()
            whi = vf[vf <= q3 + 1.5 * iqr].max()
            print(f"[{suffix or 'main'}] {label.splitlines()[0]} {key}: "
                  f"total={len(v)} finite={len(vf)} "
                  f"median={med:.4f} IQR=[{q1:.4f}, {q3:.4f}]")
            ax.bxp([dict(med=med, q1=q1, q3=q3, whislo=wlo,
                         whishi=whi, label="")],
                   positions=[j], widths=0.5, showfliers=False,
                   patch_artist=True,
                   medianprops=dict(color="black", lw=1.6),
                   whiskerprops=dict(color="0.3"),
                   capprops=dict(color="0.3"),
                   boxprops=dict(facecolor=color, alpha=0.9,
                                 edgecolor="0.15", lw=0.8))
            x = j + rng.uniform(-0.16, 0.16, len(vf))
            ms = (3.4 * np.sqrt(wf / wf.max()) if weighted
                  else np.full(len(vf), 2.2))
            ax.scatter(x, vf, s=ms ** 2, c="0.25", edgecolors="none",
                       alpha=0.4, zorder=3)
            ne = len(v) - len(vf)
            if ne:
                ax.annotate(f"{ne}/{len(v)} excluded",
                            (j, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -46), textcoords="offset points",
                            ha="center", va="top", fontsize=9,
                            color="0.4", annotation_clip=False)
        ax.set_yscale("log")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[0] for c in conds], fontsize=13.5)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.grid(axis="y", which="both", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("All models", fontsize=13.5, color="0.15",
                      loc="left")
    base = FIG_BASENAME + suffix
    for ext, kw in (("pdf", {"bbox_inches": "tight"}),
                    ("svg", {"bbox_inches": "tight"}),
                    ("png", {"dpi": 300, "bbox_inches": "tight"})):
        panel_c.savefig_robust(fig, OUT_DIR / f"{base}.{ext}", **kw)
    plt.close(fig)
    print(f"saved: {OUT_DIR / base}.pdf / .svg / .png")


def main():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        12.5,
        "axes.labelsize":   20.6,
        "xtick.labelsize":  18.75,
        "ytick.labelsize":  12.5,
        "axes.linewidth":   0.8,
        "figure.dpi":       120,
    })
    conds = [(r"Direct" "\n" r"$P{=}3$", "white"),
             ("Retrained\n" r"$P_{\mathcal{D}}{=}3$", "#dbe7f5"),
             (r"Parent" "\n" r"$P{=}10$", "#e8e8e8")]
    d3 = load_direct(3)
    rt3 = load_retrained(p_eff=3)
    rt_all = load_retrained(p_eff=None)
    p10 = load_direct(10)
    print(f"retrained candidates: P_eff=3 n={len(rt3['E'])} "
          f"({len(np.unique(rt3['seed']))} parents), all "
          f"n={len(rt_all['E'])} "
          f"({len(np.unique(rt_all['seed']))} parents)")

    draw([d3, rt3, p10], conds, weighted=True, suffix="")
    draw([d3, rt3, p10], conds, weighted=False, suffix="_raw")

    seeds = np.unique(rt3["seed"])
    per_seed_mean = np.array([rt3["E"][rt3["seed"] == s].mean()
                              for s in seeds])
    dm = np.array([d3["E"][d3["seed"] == s][0] for s in seeds])
    print(f"\nseed-summary ({len(seeds)} matched seeds): retrained "
          f"per-seed mean {per_seed_mean.mean():.3f}+-"
          f"{per_seed_mean.std(ddof=1):.3f} vs direct "
          f"{dm.mean():.3f}+-{dm.std(ddof=1):.3f}")
    rng = np.random.default_rng(1)
    obs, lo, hi, p = cluster_bootstrap(rt3, d3, rng)
    print(f"cluster bootstrap (parent-balanced median diff, "
          f"retrained - direct): {obs:.3f}  CI95 [{lo:.3f}, {hi:.3f}]  "
          f"p={p:.4f}  (B={N_BOOT})")

    print("\nparent-balanced H(tau), S_min = (Q_vis=5 & |S|=5), "
          f"denominator {N_PARENTS} parents:")
    print("tau     H_direct  H_retr(pb,Peff3)  H_retr(pb,all)")
    for t in TAUS:
        print(f"{('inf' if np.isinf(t) else t):<6}  "
              f"{h_tau(d3, t, False):.3f}     "
              f"{h_tau(rt3, t, True):.3f}             "
              f"{h_tau(rt_all, t, True):.3f}")

    for name, c in (("direct", d3), ("retrained(pb, Peff3)", rt3)):
        m = c["ok"] & np.isfinite(c["E"])
        v, w = c["E"][m], c["w"][m]
        q1, med, q3 = wquantile(v, w, [0.25, 0.5, 0.75])
        print(f"S_min-conditioned E [{name}]: n={m.sum()} "
              f"med {med:.2f} [{q1:.2f}, {q3:.2f}]")


if __name__ == "__main__":
    main()
