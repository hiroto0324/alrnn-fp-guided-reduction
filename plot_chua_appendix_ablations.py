#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Appendix figures: the two Chua P=10 retraining ablations.

Figure 1 (chua_appendix_forcing_ablation):
  guidance x parent-forcing ablation under the auto-calibrated lambda
  (alpha=0.1): No guidance (raw-data readout TF) and symbol / retained /
  full guidance with sync scope none -> readout -> matched
  (readout+retained for retained, full state for full).
Figure 2 (chua_appendix_fixed_guidance):
  guidance-mode comparison under FIXED, interpretable auxiliary
  weights and matched forcing: retained lambda=1, symbol lambda=1,
  full lambda=(M-N)/N=5.67 (the paper's main protocol). The
  gradient-based alpha-calibration cells were exploratory and are
  deliberately NOT reported (user decision 2026-09-25).

Statistical conventions (paper-wide):
  success rates = parent-balanced seed-macro over the 30-parent
  universe (weights 1/n_selected(s) from the full 141-candidate
  selection; failed / missing runs = 0 votes);
  E_stsp = post-transient fidelity (first 1000 rollout steps dropped,
  freerun_cache.posttransient_fidelity, convention 2026-09-25);
  Q_vis / |Sigma_hat| from the official free-run records.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import freerun_cache as fc
import plot_retraining_success_hierarchy as panel_c

OUT_DIR = panel_c.OUT_DIR
AGG = Path("results/aug_only/"
           "summary_retrain_sync_scope_ablation_p10_v1_aggregate.csv")
WAGG = Path("results/aug_only/"
            "summary_retrain_weighting_ablation_p10_sync_v1"
            "_aggregate.csv")
SELECTION_JSON = Path("results/relu_hierarchy/retraining_specs/"
                      "global_min_retrain_selection_original_nint128"
                      "_m20_p10.json")
Q = 5
N_PARENTS = 30
ESTSP_THRESHOLD = 2.0

COLOR = dict(none="#9a9a9a", symbol="#b79bd4", retained="#1f5fa8",
             full="#a8cdf0")
CRITERIA = [("fp", "FP recovery"),
            ("mn", "Minimal realization"),
            ("fid", r"+ $E_{\mathrm{stsp}} < 2.0$")]

_FID_CACHE = Path("results/aug_only/chua_appendix_fid_cache.json")
_fid_memo = {}
if _FID_CACHE.exists():
    _fid_memo = {
        k: tuple(float("nan") if x is None else x for x in v)
        for k, v in json.load(open(_FID_CACHE)).items()}


def n_selected_by_seed():
    sel = json.load(open(SELECTION_JSON))
    n = {}
    for c in sel["selected_candidates"]:
        s = int(c["source_seed"])
        n[s] = n.get(s, 0) + 1
    return n


N_SEL = n_selected_by_seed()


def model_metrics(ckpt):
    """(E_cut, EH_cut, fp, mn) of one completed run, memoized.

    Entries written by the first script version lack EH (3-tuple) and
    are recomputed on demand."""
    v = _fid_memo.get(ckpt)
    if v is None or len(v) < 4:
        rec = fc.freerun_record(ckpt)
        e, eh = fc.posttransient_fidelity(rec)
        fp = int(rec["Q_vis"]) == Q
        mn = fp and int(rec["n_posttransient_symbols"]) == Q
        _fid_memo[ckpt] = (e, eh, bool(fp), bool(mn))
    return _fid_memo[ckpt]


def cell(rows):
    """Seed-macro rates + per-run arrays for one condition cell."""
    e, eh, fp, mn, w, lam = [], [], [], [], [], []
    for r in rows:
        ec, ehc, f, m = model_metrics(r["ckpt"])
        e.append(ec)
        eh.append(ehc)
        fp.append(f)
        mn.append(m)
        w.append(1.0 / N_SEL[int(float(r["source_seed"]))])
        lv = r.get("distill_lambda_effective") or r.get("distill_lambda")
        try:
            lam.append(float(lv))
        except (TypeError, ValueError):
            lam.append(np.nan)
    e = np.array(e)
    eh = np.array(eh)
    fp = np.array(fp)
    mn = np.array(mn)
    w = np.array(w)
    fid = mn & np.isfinite(e) & (e < ESTSP_THRESHOLD)
    rates = {k: 100.0 * float(np.sum(w * v)) / N_PARENTS
             for k, v in (("fp", fp), ("mn", mn), ("fid", fid))}
    return dict(rates=rates, n=len(e), E=e, EH=eh, mn=mn, fid=fid,
                lam=np.array(lam), w=w)


def agg_rows(guidance, scope, teacher):
    return [r for r in csv.DictReader(open(AGG, newline=""))
            if r.get("guidance") == guidance
            and r.get("sync_scope") == scope
            and r.get("teacher") == teacher
            and r.get("run_status") == "completed"]


def wagg_rows(mode, scheme):
    return [r for r in csv.DictReader(open(WAGG, newline=""))
            if r.get("distill_mode") == mode
            and r.get("weighting_scheme") == scheme
            and r.get("run_status") == "completed"]


def style():
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        11.5,
        "axes.labelsize":   16.5,
        "xtick.labelsize":  13.5,
        "ytick.labelsize":  11.5,
        "axes.linewidth":   0.8,
        "figure.dpi":       120,
        "hatch.linewidth":  0.7,
    })


def draw_grouped(ax, conds, fs_pct=9.0):
    """conds: list of (label, color, hatch, cell)."""
    n_m = len(conds)
    width = 0.86 / n_m
    x0 = np.arange(len(CRITERIA))
    for i, (label, color, hatch, c) in enumerate(conds):
        vals = [c["rates"][k] for k, _ in CRITERIA]
        xs = x0 + (i - (n_m - 1) / 2) * width
        ax.bar(xs, vals, width * 0.9, color=color, edgecolor="0.15",
               lw=0.7, hatch=hatch, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", (x, v),
                        textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=fs_pct,
                        color="0.15", zorder=4)
    ax.set_xticks(x0)
    ax.set_xticklabels([lab for _, lab in CRITERIA])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def wquantile(v, w, q):
    v, w = np.asarray(v, float), np.asarray(w, float)
    i = np.argsort(v)
    v, w = v[i], w[i]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cw, v)


def draw_dist(ax, conds, key, ylabel, log=False, tick_labels=None):
    """Parent-balanced weighted-quantile boxes + weight-scaled jitter
    (fig. 2d' right-panel conventions)."""
    rng = np.random.default_rng(0)
    for j, (label, color, _, c) in enumerate(conds):
        v, w = c[key], c["w"]
        fin = np.isfinite(v)
        vf, wf = v[fin], w[fin]
        q1, med, q3 = wquantile(vf, wf, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        wlo = vf[vf >= q1 - 1.5 * iqr].min()
        whi = vf[vf <= q3 + 1.5 * iqr].max()
        print(f"  {key} {label}: med {med:.3f} IQR [{q1:.3f}, "
              f"{q3:.3f}]  n_fin={len(vf)}/{len(v)}")
        ax.bxp([dict(med=med, q1=q1, q3=q3, whislo=wlo, whishi=whi,
                     label="")],
               positions=[j], widths=0.5, showfliers=False,
               patch_artist=True,
               medianprops=dict(color="black", lw=1.6),
               whiskerprops=dict(color="0.3"),
               capprops=dict(color="0.3"),
               boxprops=dict(facecolor=color, alpha=0.9,
                             edgecolor="0.15", lw=0.8))
        x = j + rng.uniform(-0.16, 0.16, len(vf))
        ms = 3.2 * np.sqrt(wf / wf.max())
        ax.scatter(x, vf, s=ms ** 2, c="0.25", edgecolors="none",
                   alpha=0.4, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(tick_labels or
                       [l for l, *_ in conds], fontsize=11.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", which="both", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, basename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {"bbox_inches": "tight"}),
                    ("svg", {"bbox_inches": "tight"}),
                    ("png", {"dpi": 300, "bbox_inches": "tight"})):
        panel_c.savefig_robust(fig, OUT_DIR / f"{basename}.{ext}", **kw)
    plt.close(fig)
    print(f"saved: {OUT_DIR / basename}.pdf / .svg / .png")


def report(name, conds):
    print(f"\n== {name} (seed-macro over {N_PARENTS} parents) ==")
    for label, _, _, c in conds:
        r = c["rates"]
        print(f"  {label:<38} n={c['n']:>3}  FP {r['fp']:5.1f}%  "
              f"minimal {r['mn']:5.1f}%  +E<2 {r['fid']:5.1f}%")


# ============================================================
def figure_forcing():
    # sync-scope ablation, auto lambda alpha=0.1 throughout
    H_NONE, H_RO, H_MATCH = "///", "...", None
    conds = [
        ("No guidance (raw-data TF)", COLOR["none"], None,
         cell(agg_rows("none", "readout", "raw_data"))),
        ("Symbol, no forcing", COLOR["symbol"], H_NONE,
         cell(agg_rows("symbol", "none", "parent_free_run"))),
        ("Retained, no forcing", COLOR["retained"], H_NONE,
         cell(agg_rows("retained", "none", "parent_free_run"))),
        ("Retained, readout forcing", COLOR["retained"], H_RO,
         cell(agg_rows("retained", "readout", "parent_free_run"))),
        ("Retained, readout+retained forcing", COLOR["retained"],
         H_MATCH,
         cell(agg_rows("retained", "readout_retained",
                       "parent_free_run"))),
        ("Full, no forcing", COLOR["full"], H_NONE,
         cell(agg_rows("full", "none", "parent_free_run"))),
        ("Full, full-state forcing", COLOR["full"], H_MATCH,
         cell(agg_rows("full", "full", "parent_free_run"))),
    ]
    report("forcing ablation", conds)

    style()
    fig, ax = plt.subplots(figsize=(9.4, 4.1), constrained_layout=True)
    draw_grouped(ax, conds)
    guid = [Patch(facecolor=COLOR["none"], edgecolor="0.15",
                  label="No guidance"),
            Patch(facecolor=COLOR["symbol"], edgecolor="0.15",
                  label="Symbol"),
            Patch(facecolor=COLOR["retained"], edgecolor="0.15",
                  label="Retained"),
            Patch(facecolor=COLOR["full"], edgecolor="0.15",
                  label="Full")]
    sync = [Patch(facecolor="white", edgecolor="0.15", hatch="///",
                  label="no forcing"),
            Patch(facecolor="white", edgecolor="0.15", hatch="...",
                  label="readout forcing"),
            Patch(facecolor="white", edgecolor="0.15",
                  label="matched forcing")]
    leg1 = ax.legend(handles=guid, frameon=False, fontsize=11.5,
                     title="Guidance", title_fontsize=12, ncol=4,
                     loc="lower left", bbox_to_anchor=(0.0, 1.015),
                     handlelength=1.2, columnspacing=1.0,
                     alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=sync, frameon=False, fontsize=11.5,
              title="Parent-state sync", title_fontsize=12, ncol=3,
              loc="lower right", bbox_to_anchor=(1.0, 1.015),
              handlelength=1.2, columnspacing=1.0, alignment="left")
    save(fig, "chua_appendix_forcing_ablation")


def figure_fixed_guidance():
    conds = [
        ("No guidance", COLOR["none"], None,
         cell(agg_rows("none", "readout", "raw_data"))),
        (r"Symbol, $\lambda_{\mathrm{sym}}{=}1$", COLOR["symbol"],
         None, cell(wagg_rows("symbol", "fixed_lambda1"))),
        (r"Retained, $\lambda_{\mathrm{ret}}{=}1$", COLOR["retained"],
         None,
         cell(wagg_rows("retained_preactivation", "coord_equal_fixed"))),
        (r"Full, $\lambda_{\mathrm{full}}{=}(M{-}N)/N$", COLOR["full"],
         None,
         cell(wagg_rows("full_preactivation", "coord_equal_fixed"))),
    ]
    report("fixed-lambda guidance comparison (matched forcing)", conds)

    style()
    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax = fig.add_subplot(gs[0, :])
    axe = fig.add_subplot(gs[1, 0])
    axh = fig.add_subplot(gs[1, 1])
    draw_grouped(ax, conds, fs_pct=11.5)
    guid = [Patch(facecolor=c, edgecolor="0.15", label=l)
            for l, c, _, _ in conds]
    ax.legend(handles=guid, frameon=False, fontsize=11.5,
              ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.015),
              handlelength=1.2, columnspacing=1.1, alignment="left")

    ticks = ["No\nguidance", "Symbol", "Retained", "Full"]
    draw_dist(axe, conds, "E", r"$E_{\mathrm{stsp}}$", log=True,
              tick_labels=ticks)
    axe.axhline(ESTSP_THRESHOLD, color="0.35", lw=1.0,
                ls=(0, (4, 3)), zorder=2)
    axe.text(3.42, ESTSP_THRESHOLD * 1.09,
             r"$E_{\mathrm{stsp}}{=}2.0$", ha="right", va="bottom",
             fontsize=10.5, color="0.35", zorder=2)
    draw_dist(axh, conds, "EH", r"$E_{\mathrm{H}}$", log=True,
              tick_labels=ticks)
    save(fig, "chua_appendix_fixed_guidance")


def figure_htau():
    """High-fidelity success curves H(tau) = P(S_min & E_stsp < tau),
    parent-balanced over the 30-parent universe (failures at +inf,
    candidate-less parents 0). Fixed-lambda conditions only + the
    no-guidance baseline; supersedes the old candidate-level
    plot_high_fidelity_curves.py design (auto-alpha cells dropped)."""
    import plot_chua_direct_retrained_dist as base

    def direct_cell(p):
        d = base.load_direct(p)
        return dict(E=d["E"], mn=d["ok"], w=d["w"])

    conds = [
        (r"Direct $P{=}3$", "black", (0, (4, 2.5)), direct_cell(3)),
        ("No guidance", COLOR["none"], "-",
         cell(agg_rows("none", "readout", "raw_data"))),
        (r"Symbol, $\lambda_{\mathrm{sym}}{=}1$", COLOR["symbol"], "-",
         cell(wagg_rows("symbol", "fixed_lambda1"))),
        (r"Retained, $\lambda_{\mathrm{ret}}{=}1$", COLOR["retained"],
         "-",
         cell(wagg_rows("retained_preactivation", "coord_equal_fixed"))),
        (r"Full, $\lambda_{\mathrm{full}}{=}(M{-}N)/N$", "#4886c4", "-",
         cell(wagg_rows("full_preactivation", "coord_equal_fixed"))),
    ]
    taus = np.linspace(0.9, 4.0, 400)
    style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    print("\n== H(tau) curves (parent-balanced /30) ==")
    for label, color, ls, c in conds:
        e, mn, w = c["E"], c["mn"], c["w"]
        base_ok = mn & np.isfinite(e)
        h = [100.0 * float(np.sum(w[base_ok & (e < t)])) / N_PARENTS
             for t in taus]
        ax.plot(taus, h, color=color, ls=ls, lw=2.0, label=label,
                zorder=3)
        asym = 100.0 * float(np.sum(w * mn)) / N_PARENTS
        print(f"  {label}: H(2.0)="
              f"{100.0*float(np.sum(w[base_ok & (e < 2.0)]))/N_PARENTS:.1f}%"
              f"  ->{asym:.1f}% (minimal rate)")
    ax.axvline(2.0, color="0.35", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(2.03, 0.96, r"$E_{\mathrm{stsp}}{=}2.0$", ha="left",
            va="top", fontsize=14.5, color="0.35",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"Fidelity threshold $\tau$")
    ax.set_ylabel(r"$H(\tau)=\Pr\!\left[S_{\min}\wedge "
                  r"E_{\mathrm{stsp}}<\tau\right]$ (%)")
    ax.set_xlim(0.9, 4.0)
    ax.set_ylim(0, 100)
    ax.grid(color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # legend outside the axes (user, 2026-09-25)
    ax.legend(frameon=False, fontsize=14.5, loc="center left",
              bbox_to_anchor=(1.02, 0.5), handlelength=1.5,
              labelspacing=0.55)
    save(fig, "chua_high_fidelity_curves")


def _save_fid_cache():
    with open(_FID_CACHE, "w") as f:
        json.dump({k: [None if (isinstance(x, float) and not
                                np.isfinite(x)) else x for x in v]
                   for k, v in _fid_memo.items()}, f)


if __name__ == "__main__":
    figure_forcing()
    figure_fixed_guidance()
    figure_htau()
    _save_fid_cache()
    print(f"fidelity cache -> {_FID_CACHE} ({len(_fid_memo)} models)")
