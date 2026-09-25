#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panel (c): success hierarchy of four retraining strategies for reduced
AL-RNNs (P_parent = 10 Chua, five-region reduction candidates).

Grouped bar chart over three nested success criteria
    S_FP  = 1[Q_vis == 5]
    S_min = 1[Q_vis == 5  and  |Sigma_hat| == 5]
    S_fid = 1[Q_vis == 5  and  |Sigma_hat| == 5  and  D_stsp < 2]
for the four retraining strategies (distill_mode metadata column):
    none / full_preactivation / retained_preactivation / symbol.

Data sources
------------
1. results/aug_only/summary_retrain.csv
   one row per retrained model; columns used: tag, ckpt, P_original,
   num_clusters, P_effective, source_seed, candidate_id, distill_mode.
   Population filter: P_original == 10 and num_clusters == 5 (the same
   five-region candidate set used for the main retraining comparison).
2. results/aug_only/{tag}_train.npz + {tag}_fp_snapshots.npz
   training-time deterministic evaluation records. Q_vis and D_stsp are
   taken at the best-score epoch (score = Dstsp + 5*|Q_vis-5|, first
   argmin) — exactly the epoch whose weights were saved as the
   checkpoint, so these equal re-evaluating the saved model.
3. |Sigma_hat|: 10,000-step autonomous rollout of the saved checkpoint,
   first 1,000 steps discarded, distinct visited bit patterns on the
   retained ReLU slots. Cached per model in NVISITED_CACHE; models
   missing from the cache are rolled out here and appended.
"""

from pathlib import Path
import csv
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration (edit here)
# ============================================================
SUMMARY_CSV    = Path("results/aug_only/summary_retrain.csv")
RECORDS_DIR    = Path("results/aug_only")
MODELS_DIR     = Path("models")
NVISITED_CACHE = Path("results/aug_only/p10_bestepoch_nvisited.csv")
RAW_DATA_PATH  = Path("data/chua_3-scroll_train.npy")
OUT_DIR        = Path("figures/paper")
FIG_BASENAME   = "chua_retraining_success_hierarchy"

P_PARENT        = 10      # parent model P
NUM_CLUSTERS    = 5       # five-region reduction candidates only
Q               = 5       # target fixed points
DSTSP_THRESHOLD = 2.0
FP_COUNT_WEIGHT = 5.0     # checkpoint-selection score (repo convention)
ROLLOUT_STEPS   = 10000
TRANSIENT_CUT   = 1000
M, N            = 20, 3   # hidden size / readout dims

SHOW_TITLE      = False
SHOW_PANEL_TAG  = False   # draw "(c)" in the corner
TITLE = "Retained-preactivation guidance enables faithful minimal realization"

# repository method name (distill_mode metadata) -> display label
METHOD_ORDER = ["none", "full_preactivation", "retained_preactivation",
                "symbol"]
METHOD_LABEL = {
    "none":                   "None",
    "full_preactivation":     "Full preactivation",
    "retained_preactivation": "Retained preactivation",
    "symbol":                 "Symbol",
}
METHOD_COLOR = {
    "none":                   "#9a9a9a",
    "full_preactivation":     "#a8cdf0",
    "retained_preactivation": "#1f5fa8",
    "symbol":                 "#b79bd4",
}

# validation-only reference values (paper draft); NOT a data source
# switch: use the teacher-forcing (readout_retained parent-forcing)
# runs for the retained bars; False restores the no-forcing retained runs
USE_TF_RETAINED = True
RETAINED_TF_CSV = Path("results/aug_only/"
                       "summary_retrain_sync_scope_ablation_p10_v1.csv")
RETAINED_TF_NVIS = Path("results/aug_only/p10_ablation_nvisited.csv")

REFERENCE = {   # method -> (fp %, minimal %, fidelity %)
    "none":                   (23, 15, 15),
    "full_preactivation":     (62, 48, 47),
    # retained = the TF (readout_retained parent-forcing) values (2026-09-15)
    "retained_preactivation": (91, 85, 83),
    "symbol":                 (63, 53, 1),
}
REFERENCE_TOL = 3.0   # warn if recomputed rate deviates more than this (%)


def savefig_robust(fig, path: Path, **kw):
    """savefig with retry + fallback name. On Windows an image viewer or
    PowerPoint holding the target file locks it (OSError errno 22/13);
    don't lose the run over that."""
    import time
    for _ in range(3):
        try:
            fig.savefig(path, **kw)
            return
        except OSError:
            time.sleep(1.0)
    alt = path.with_name(path.stem + "_new" + path.suffix)
    warnings.warn(f"{path} is locked by another program (close the viewer/"
                  f"PowerPoint) — saved as {alt} instead")
    fig.savefig(alt, **kw)


# ============================================================
# 1. Data loading
# ============================================================
def load_population():
    """Rows of summary_retrain.csv for the evaluated population, one final
    row per retrained model (unique (source_seed, candidate_id, mode))."""
    rows = {}
    dup = 0
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if int(float(r["P_original"])) != P_PARENT:
                    continue
                if int(float(r["num_clusters"])) != NUM_CLUSTERS:
                    continue
            except (KeyError, ValueError):
                continue
            mode = (r.get("distill_mode") or "none").strip() or "none"
            key = (int(float(r["source_seed"])),
                   int(float(r["candidate_id"])), mode)
            if key in rows:
                dup += 1
            # the metrics are recomputed from the per-tag record files, and
            # tag/ckpt are identical across duplicate CSV rows of one model,
            # so keeping the last row is only a bookkeeping choice
            rows[key] = r
    if dup:
        print(f"note: {dup} duplicate summary rows collapsed "
              "(metrics come from per-model record files, not CSV rows)")
    # ── retained bars: swap in the teacher-forcing runs (2026-09-15) ──
    # replace the retained_preactivation rows with the sync-ablation
    # retained + readout_retained parent-forcing runs.
    # legend label stays "Retained preactivation".
    if USE_TF_RETAINED and RETAINED_TF_CSV.exists():
        rows = {k: v for k, v in rows.items()
                if k[2] != "retained_preactivation"}
        n_tf = 0
        with open(RETAINED_TF_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    if (r["distill_mode"] != "retained_preactivation"
                            or r.get("sync_scope") != "readout_retained"
                            or float(r["distill_lambda"]) == 0.0
                            or int(float(r["num_clusters"]))
                            != NUM_CLUSTERS):
                        continue
                except (KeyError, ValueError):
                    continue
                key = (int(float(r["source_seed"])),
                       int(float(r["candidate_id"])),
                       "retained_preactivation")
                rows[key] = r
                n_tf += 1
        print(f"note: retained bars use the readout_retained "
              f"parent-forcing runs ({n_tf} models, "
              f"{RETAINED_TF_CSV.name})")
    return rows


# ============================================================
# 2. Final-evaluation selection (best-score epoch from records)
# ============================================================
def best_epoch_metrics(tag: str):
    f_train = RECORDS_DIR / f"{tag}_train.npz"
    f_fp    = RECORDS_DIR / f"{tag}_fp_snapshots.npz"
    if not (f_train.exists() and f_fp.exists()):
        return None
    with np.load(f_train) as z:
        dstsp = np.asarray(z["Dstsp_history"], dtype=float)
        eps   = np.asarray(z["eval_epochs"], dtype=int)
    with np.load(f_fp, allow_pickle=True) as z:
        vis = np.array([int(z[f"ep{e}_is_visited"].sum())
                        if f"ep{e}_is_visited" in z.files else 0
                        for e in eps], dtype=int)
    n = min(len(dstsp), len(vis))
    dstsp, vis = dstsp[:n], vis[:n]
    if np.isnan(dstsp).any():
        warnings.warn(f"{tag}: NaN in Dstsp_history (excluded from argmin)")
    score = np.where(np.isnan(dstsp), np.inf,
                     dstsp + FP_COUNT_WEIGHT * np.abs(vis - Q))
    b = int(np.argmin(score))
    return float(dstsp[b]), int(vis[b])


def load_nvisited_cache():
    cache = {}
    if NVISITED_CACHE.exists():
        with open(NVISITED_CACHE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (int(float(r["source_seed"])),
                       int(float(r["candidate_id"])), r["mode"])
                cache[key] = int(float(r["n_visited_cut"]))
    return cache


def rollout_n_visited(ckpt_path: Path):
    """|Sigma_hat| of one saved checkpoint (fallback when not cached).

    Goes through freerun_cache so the rollout is done once and the full
    record (Dstsp, DH, symbol sequence, visited FPs) is persisted with it.
    """
    import freerun_cache
    rec = freerun_cache.freerun_record(ckpt_path, steps=ROLLOUT_STEPS,
                                       cut=TRANSIENT_CUT)
    return int(rec["n_posttransient_symbols"])


# ============================================================
# 3. Metric computation
# ============================================================
def evaluate_all(rows):
    cache = load_nvisited_cache()
    # nvis cache keyed by tag (sync-ablation runs; made by make_sync_scope_...)
    tag_cache = {}
    if RETAINED_TF_NVIS.exists():
        with open(RETAINED_TF_NVIS, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tag_cache[r["tag"]] = int(float(r["n_visited_cut"]))
    per_model = []          # dict(key, mode, dstsp, qvis, nvis)
    n_rolled = 0
    for key, r in sorted(rows.items()):
        mode = key[2]
        m = best_epoch_metrics(r["tag"])
        if m is None:
            warnings.warn(f"{r['tag']}: record files missing — model skipped")
            continue
        dstsp, qvis = m
        ck = Path(r.get("ckpt") or "")
        if not ck.exists():
            ck = MODELS_DIR / f"{r['tag']}.pth"
        if not ck.exists():
            warnings.warn(f"{r['tag']}: checkpoint missing — skipped")
            continue
        if r["tag"] in tag_cache:
            nvis = tag_cache[r["tag"]]
        elif key in cache:
            nvis = cache[key]
        else:
            nvis = rollout_n_visited(ck)
            n_rolled += 1
        # E_stsp: post-transient fidelity convention (user, 2026-09-25);
        # overrides the best-epoch full-rollout history value
        import freerun_cache
        rec = freerun_cache.freerun_record(ck, steps=ROLLOUT_STEPS,
                                           cut=TRANSIENT_CUT)
        dstsp = freerun_cache.posttransient_fidelity(rec)[0]
        per_model.append(dict(key=key, mode=mode, dstsp=dstsp, qvis=qvis,
                              nvis=nvis))
    if n_rolled:
        print(f"note: |Sigma_hat| freshly rolled out for {n_rolled} models "
              "not in the cache")
    return per_model


def summarize(per_model):
    out = []
    for mode in METHOD_ORDER:
        g = [m for m in per_model if m["mode"] == mode]
        n = len(g)
        fp  = [m for m in g if m["qvis"] == Q]
        mn  = [m for m in fp if m["nvis"] == Q]
        fid = [m for m in mn if m["dstsp"] < DSTSP_THRESHOLD]
        out.append(dict(
            method=mode, n_models=n,
            n_fp_success=len(fp),
            fp_recovery_rate=100.0 * len(fp) / n if n else float("nan"),
            n_minimal_success=len(mn),
            minimal_realization_rate=100.0 * len(mn) / n if n else float("nan"),
            n_fidelity_success=len(fid),
            high_fidelity_minimal_rate=100.0 * len(fid) / n if n else float("nan"),
        ))
    return out


# ============================================================
# 4. Validation
# ============================================================
def validate(summary):
    ns = {s["method"]: s["n_models"] for s in summary}
    if len(set(ns.values())) > 1:
        print(f"NOTE: evaluated model counts differ between methods: {ns} "
              "(failed/diverged runs have no final model)")
    for s in summary:
        # nested criteria are subsets by construction; verify anyway
        if not (s["high_fidelity_minimal_rate"]
                <= s["minimal_realization_rate"] + 1e-9
                <= s["fp_recovery_rate"] + 1e-9):
            warnings.warn(f"{s['method']}: nested-criteria relation violated")
        ref = REFERENCE.get(s["method"])
        if ref:
            got = (s["fp_recovery_rate"], s["minimal_realization_rate"],
                   s["high_fidelity_minimal_rate"])
            for name, g, r in zip(("FP", "minimal", "fidelity"), got, ref):
                if abs(g - r) > REFERENCE_TOL:
                    warnings.warn(
                        f"{s['method']} {name}: recomputed {g:.1f}% deviates "
                        f"from paper-draft reference {r}%")


# ============================================================
# 5. Plotting
# ============================================================
def make_figure(summary):
    # fonts: 1.5x base, then everything (y ticks included) scaled a further
    # 1.25x per user request
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
    criteria = [("fp_recovery_rate", "FP recovery"),
                ("minimal_realization_rate", "Minimal realization"),
                ("high_fidelity_minimal_rate",
                 r"+ $E_{\mathrm{stsp}} < 2.0$")]

    # flat layout: wider than tall; the one-row legend keeps height down
    fig, ax = plt.subplots(figsize=(8.0, 3.4), constrained_layout=True)
    n_m = len(METHOD_ORDER)
    width = 0.19
    x0 = np.arange(len(criteria))
    for i, mode in enumerate(METHOD_ORDER):
        s = next(s for s in summary if s["method"] == mode)
        vals = [s[c] for c, _ in criteria]
        xs = x0 + (i - (n_m - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=METHOD_COLOR[mode],
               edgecolor="white", lw=0.5, label=METHOD_LABEL[mode], zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=FS_PCT,
                        color="0.15", zorder=4)

    ax.set_xticks(x0)
    ax.set_xticklabels([lab for _, lab in criteria])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.1,
              handlelength=1.1, handletextpad=0.45)
    if SHOW_TITLE:
        fig.suptitle(TITLE, fontsize=11.5, y=1.09)
    if SHOW_PANEL_TAG:
        ax.text(-0.10, 1.06, "(c)", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {"bbox_inches": "tight"}),
                    ("svg", {"bbox_inches": "tight"}),
                    ("png", {"dpi": 300, "bbox_inches": "tight"})):
        savefig_robust(fig, OUT_DIR / f"{FIG_BASENAME}.{ext}", **kw)
    plt.close(fig)


# ============================================================
def main():
    rows = load_population()
    print(f"population: P_original={P_PARENT}, num_clusters={NUM_CLUSTERS} "
          f"-> {len(rows)} retrained models")
    per_model = evaluate_all(rows)
    summary = summarize(per_model)
    validate(summary)

    print(f"\n{'method':>24} {'n':>4} {'FP recovery':>13} "
          f"{'minimal':>13} {'+Dstsp<2':>13}")
    for s in summary:
        print(f"{s['method']:>24} {s['n_models']:>4} "
              f"{s['n_fp_success']:>4}/{s['n_models']:<3}={s['fp_recovery_rate']:>4.0f}% "
              f"{s['n_minimal_success']:>4}/{s['n_models']:<3}={s['minimal_realization_rate']:>4.0f}% "
              f"{s['n_fidelity_success']:>4}/{s['n_models']:<3}={s['high_fidelity_minimal_rate']:>4.0f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{FIG_BASENAME}_summary.csv"
    cols = ["method", "n_models", "n_fp_success", "fp_recovery_rate",
            "n_minimal_success", "minimal_realization_rate",
            "n_fidelity_success", "high_fidelity_minimal_rate"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in summary:
            w.writerow(s)

    make_figure(summary)
    print(f"\nsaved: {OUT_DIR / FIG_BASENAME}.pdf / .svg / .png / _summary.csv")
    print("\nSources: population + method identity from "
          f"{SUMMARY_CSV} (columns P_original, num_clusters, distill_mode); "
          "Q_vis & D_stsp at the best-score epoch (saved checkpoint) from "
          "{tag}_train.npz / {tag}_fp_snapshots.npz; |Sigma_hat| from the "
          f"10k-step autonomous rollout (first {TRANSIENT_CUT} discarded), "
          f"cached in {NVISITED_CACHE}. Candidate set: all five-region "
          "(num_clusters==5) reduction candidates of the global-minimum "
          "selection, the same set as the main retraining comparison.")


if __name__ == "__main__":
    main()
