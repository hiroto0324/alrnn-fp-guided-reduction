#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panel (a): direct AL-RNN training becomes more reliable with larger P.

Two vertically stacked axes sharing the x-axis (P = number of ReLU units):
  (top)    FP recovery rate (%)  = fraction of seeds with Q_vis == 5
  (bottom) E_stsp median with interquartile band (all seeds, unconditional)

Data source (per P, per seed) — the official free-run record of the saved
checkpoint (freerun_cache, 10k-step deterministic autonomous rollout):
  Q_vis  : visited admissible FPs (full-rollout visited set, no cut)
  E_stsp : POST-TRANSIENT fidelity (first 1000 rollout steps dropped),
           freerun_cache.posttransient_fidelity — the official fidelity
           convention since 2026-09-25, shared with every other figure.
(The previous version read the training-time Dstsp_history at the
best-score epoch; that equals the full-rollout value of the saved
checkpoint and is superseded by the post-transient convention.)

Outputs: figures/paper/chua_direct_training_vs_P.{pdf,svg,png} and
         figures/paper/chua_direct_training_vs_P_summary.csv
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration (edit here)
# ============================================================
RESULTS_DIR = Path("results/aug_only")
OUT_DIR     = Path("figures/paper")

# Direct-training condition (matches the P=10 parent-model condition)
TAG_TEMPLATE = ("chua_orig_nint128_bs16_sig0.0_lr1e-03-1e-05"
                "_tfp0_ramp200_hLR1_m20_p{P}_seed{seed}")

P_VALUES = list(range(1, 11))     # P = 1 .. 10 (3 = theoretical lower bound)
SEEDS    = list(range(30))

# Checkpoint-selection score (must match train_alrnn.py)
TARGET_FP_COUNT = 5
FP_COUNT_WEIGHT = 5.0

FIG_BASENAME = "chua_direct_training_vs_P"


# ============================================================
# Data loading
# ============================================================
def freerun_metrics(tag: str):
    """Return (E_stsp, Q_vis) of the saved checkpoint's official
    free-run record, or None if the checkpoint is missing.

    E_stsp follows the post-transient fidelity convention (2026-09-25);
    Q_vis keeps the full-rollout visited-set convention."""
    import freerun_cache as fc
    ckpt = Path("models") / f"{tag}.pth"
    if not ckpt.exists():
        return None
    rec = fc.freerun_record(ckpt)
    e = fc.posttransient_fidelity(rec)[0]
    return float(e), int(rec["Q_vis"])


def load_all():
    """Collect per-seed metrics; returns (records, missing_P)."""
    records = []                       # dict(P, seed, estsp, qvis)
    missing_p = []
    for P in P_VALUES:
        n_found = 0
        for seed in SEEDS:
            tag = TAG_TEMPLATE.format(P=P, seed=seed)
            m = freerun_metrics(tag)
            if m is None:
                continue
            d, v = m
            records.append(dict(P=P, seed=seed, estsp=d, qvis=v))
            n_found += 1
        if n_found == 0:
            missing_p.append(P)
        elif n_found != len(SEEDS):
            warnings.warn(f"P={P}: only {n_found}/{len(SEEDS)} seeds found")
    return records, missing_p


def summarize(records):
    """Per-P aggregation for the plot and the summary CSV."""
    rows = []
    for P in P_VALUES:
        d = np.array([r["estsp"] for r in records if r["P"] == P])
        v = np.array([r["qvis"] for r in records if r["P"] == P])
        if len(d) == 0:
            continue
        n_nan = int(np.isnan(d).sum())
        if n_nan:
            warnings.warn(f"P={P}: {n_nan} NaN E_stsp values excluded from "
                          "quantiles")
        dq = d[~np.isnan(d)]
        rows.append(dict(
            P=P,
            n_seeds=len(d),
            # strict 30-seed convention (user, 2026-09-25): missing /
            # failed runs count as failures, denominator = all seeds
            fp_recovery_rate=100.0 * float((v == TARGET_FP_COUNT).sum())
            / len(SEEDS),
            estsp_median=float(np.median(dq)),
            estsp_q25=float(np.percentile(dq, 25)),
            estsp_q75=float(np.percentile(dq, 75)),
        ))
    return rows


# ============================================================
# Plotting
# ============================================================
def make_figure(rows, out_dir: Path, figsize=(7.0, 4.6),
                basename=FIG_BASENAME, layout="stacked", fs_pct=None):
    # font sizes: axis labels / % labels / legend / annotation are 1.5x
    FS_LABEL, FS_PCT, FS_LEGEND, FS_ANNOT = 16.5, 12.75, 12.75, 12.75
    if fs_pct is not None:
        FS_PCT = fs_pct
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        10,
        "axes.labelsize":   FS_LABEL,
        "axes.titlesize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.linewidth":   0.8,
        "figure.dpi":       120,
    })

    P   = [r["P"] for r in rows]
    rec = [r["fp_recovery_rate"] for r in rows]
    med = [r["estsp_median"] for r in rows]
    q25 = [r["estsp_q25"] for r in rows]
    q75 = [r["estsp_q75"] for r in rows]

    # stacked 2-row layout, kept as flat as the wrapped y-labels allow
    if layout == "row":
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True)
    else:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]})

    color = "#1f5fa8"                  # single restrained accent

    # ── top: FP recovery rate ──────────────────────────────
    ax1.plot(P, rec, "-o", color=color, lw=1.8, ms=5.5, zorder=3)
    for x, y in zip(P, rec):
        # on the rising part labels shift left to clear the trend line;
        # on the flat 0% stretch (P=1,2) they sit centered above the point
        off = (0, 8) if y < 5 else (-10, 6)
        # on narrow panels the last two labels collide — push the final
        # one to the right of its point
        if x == P[-1] and (layout == "row" or figsize[0] < 6):
            off = (10, 6)
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                     xytext=off, ha="center", fontsize=FS_PCT,
                     color="0.2")
    ax1.set_ylabel("FP recovery\nrate (%)")
    ax1.set_ylim(-4, 112)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_xticks(P)
    ax1.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    if layout == "row":
        ax1.set_xlabel(r"Number of ReLU units $P$")


    # ── bottom: E_stsp median + IQR ────────────────────────
    ax2.fill_between(P, q25, q75, color=color, alpha=0.16, lw=0, zorder=1,
                     label="IQR (25–75%)")
    ax2.plot(P, med, "-o", color=color, lw=1.8, ms=5.5, zorder=3,
             label="median")
    ax2.set_ylabel("State-space\n" + r"error $E_{\mathrm{stsp}}$")
    ax2.set_xlabel(r"Number of ReLU units $P$")
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(P)
    ax2.grid(axis="y", color="0.88", lw=0.7, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right",
               handlelength=1.6, borderaxespad=0.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = out_dir / f"{basename}.{ext}"
        # retry + fallback name: on Windows a viewer/PowerPoint holding the
        # file locks it (OSError) — don't lose the run over that
        import time
        for _ in range(3):
            try:
                fig.savefig(path, **kw)
                break
            except OSError:
                time.sleep(1.0)
        else:
            alt = path.with_name(path.stem + "_new" + path.suffix)
            warnings.warn(f"{path} is locked (close the viewer/PowerPoint) "
                          f"— saved as {alt} instead")
            fig.savefig(alt, **kw)
    plt.close(fig)


# ============================================================
def main():
    records, missing_p = load_all()
    if missing_p:
        print(f"WARNING: no data found for P = {missing_p}; "
              "plotting available P only.", file=sys.stderr)
    rows = summarize(records)
    if not rows:
        sys.exit("ERROR: no data found at all — check TAG_TEMPLATE/paths.")

    print(f"{'P':>3} {'n_seeds':>8} {'FP rec.':>8} {'med E_stsp':>10} "
          f"{'IQR':>15}")
    for r in rows:
        print(f"{r['P']:>3} {r['n_seeds']:>8} {r['fp_recovery_rate']:>7.1f}% "
              f"{r['estsp_median']:>10.3f} "
              f"[{r['estsp_q25']:.3f}, {r['estsp_q75']:.3f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{FIG_BASENAME}_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["P", "n_seeds", "fp_recovery_rate",
                "estsp_median", "estsp_q25", "estsp_q75"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    make_figure(rows, OUT_DIR)
    # flatter companion version (wider, shorter)
    make_figure(rows, OUT_DIR, figsize=(8.8, 3.3),
                basename=FIG_BASENAME + "_flat")
    # side-by-side companion version (1 x 2)
    make_figure(rows, OUT_DIR, figsize=(9.8, 2.9),
                basename=FIG_BASENAME + "_row", layout="row")
    # slim companion version (stacked, narrower; smaller % labels)
    make_figure(rows, OUT_DIR, figsize=(4.2, 4.6),
                basename=FIG_BASENAME + "_slim", fs_pct=10)
    print(f"\nsaved: {OUT_DIR / FIG_BASENAME}.pdf / .svg / .png "
          f"(+ _flat) / _summary.csv")


if __name__ == "__main__":
    main()
