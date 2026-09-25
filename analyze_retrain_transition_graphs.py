#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare predicted quotient graphs with the transition graphs actually
realized by retrained candidates (success vs failure case studies).

Per example row:
  LEFT  = predicted quotient graph: the parent's autonomous trajectory
          (grp protocol: TF warmup 500 + 40k free run, 1k cut) projected
          onto the candidate's retained ReLU bits. Blue+star nodes are the
          projections of the parent's real-FP regions.
  RIGHT = realized graph: the RETRAINED model's own 10k free run (1k cut),
          visited retained-bit patterns and observed transitions. Nodes
          shared with the prediction are drawn at the SAME positions;
          predicted-FP patterns that the retrained model never visits are
          shown as dashed ghost nodes.

Examples (retained + readout_retained forcing condition, ga0p1):
  success       seed7 cand1 (Q_vis=5, |Sigma_hat|=5)
  collapse fail seed7 cand0 (Q_vis=3, |Sigma_hat|=3: two scrolls lost)
  surplus fail  seed4 cand0 (Q_vis=3, |Sigma_hat|=6: extra regions)
Parent trajectories/FPs are cached per seed in the panel-(b) cache files.
"""

from pathlib import Path
import csv
import json
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import networkx as nx
import torch

import graph_reduction_with_relu_pruning as grp
from tutorial import AL_RNN, predict_free_sequence
# retrained-model free runs and FP verdicts go through freerun_cache
# (in load_example): mask-aware training-side analysis, records persisted

M, P, N = 20, 10, 3
T_GEN, T_R = 40000, 1000
STUDENT_STEPS, STUDENT_CUT = 10000, 1000
DATA_PATH = "data/chua_3-scroll_train.npy"
OUT_DIR = Path("figures/analysis")
FIG_BASENAME = "retrain_transition_graphs_comparison"

COND_CSV = Path("results/aug_only/"
                "summary_retrain_sync_scope_ablation_p10_v1.csv")
AGG_CSV = Path("results/aug_only/"
               "summary_retrain_sync_scope_ablation_p10_v1_aggregate.csv")
SPEC_TPL = ("results/relu_hierarchy/retraining_specs/"
            "chua_orig_nint128_bs16_sig0.0_lr1e-03-1e-05_tfp0_ramp200"
            "_hLR1_m20_p10_seed{s}_minimal_symbol_reductions.json")
SRC_TPL = ("models/chua_orig_nint128_bs16_sig0.0_lr1e-03-1e-05_tfp0"
           "_ramp200_hLR1_m20_p10_seed{s}.pth")

EXAMPLES = [  # (seed, cand, row label)
    (7, 1, "Success"),
    (7, 0, "Failure (scroll collapse)"),
    (4, 0, "Failure (surplus regions)"),
]

# non-chain-predicted candidates whose retrained model nonetheless
# realizes a chain (reorganization evidence); separate figure with the
# realized chain laid out on a line and the prediction following it
NONCHAIN_EXAMPLES = [
    (16, 0, "Minimal candidate (2 cycles, no leaf)"),
    (10, 5, "Minimal candidate (1 cycle + tail)"),
    (2, 0, "Minimal candidate (1 cycle + tail)"),
]
NONCHAIN_BASENAME = "retrain_transition_graphs_nonchain_success"

C_FP, C_FREE = "#cfe3f5", "#fbe0c4"
C_EDGE = "0.2"


def parent_cache(seed):
    """orbit (T, M) + real-FP info for one parent seed (panel-(b) cache)."""
    ckpt = SRC_TPL.format(s=seed)
    cache = Path(f"results/relu_hierarchy/panel_b_traj_cache_seed{seed}.npz")
    stamp = f"{ckpt}|{Path(ckpt).stat().st_mtime_ns}|{T_GEN}|{T_R}"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            if str(z["stamp"]) == stamp:
                return (z["orbit"],
                        [str(s) for s in z["fp_symbols"]], z["fp_coords"])
    model = AL_RNN(M=M, P=P, N=N)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    raw = np.load(DATA_PATH).astype(np.float32)
    x = torch.tensor(raw).unsqueeze(0)
    with torch.no_grad():
        z0 = grp._warmup_latent(model, x[:, :grp.TF_WARMUP_STEPS, :],
                                alpha=1.0, n_interleave=1)
        orbit = grp.predict_free_from_latent(
            model, z0, T_GEN + T_R).detach().numpy()[0][T_R:, :]
    bits = np.asarray(grp.lrf.convert_to_bits(orbit[:, -P:]), dtype=int)
    fpd = grp.analyze_fixed_points_continuous(model, np.unique(bits, axis=0),
                                              0.01)
    fp_syms = sorted(str(int(s, 2)) for s, v in fpd.items()
                     if v["type"] == "real")
    fp_coords = np.array([fpd[format(int(s), f"0{P}b")]["location"][:N]
                          for s in fp_syms])
    np.savez_compressed(cache, stamp=stamp,
                        orbit=orbit.astype(np.float32),
                        fp_symbols=np.array(fp_syms), fp_coords=fp_coords)
    return orbit.astype(np.float32), fp_syms, fp_coords


def seq_graph(patterns):
    """visited patterns + deduped observed transitions (no self loops)."""
    G = nx.DiGraph()
    G.add_nodes_from(sorted(set(patterns)))
    for a, b in zip(patterns[:-1], patterns[1:]):
        if a != b:
            G.add_edge(a, b)
    return G


def load_example(seed, cand):
    spec = json.load(open(SPEC_TPL.format(s=seed)))
    c = next(x for x in spec["candidates"]
             if int(x["candidate_id"]) == cand)
    deleted = set(int(j) for j in c["deleted_relu_local_indices"])
    retained_local = [j for j in range(P) if j not in deleted]

    orbit, fp_syms, _ = parent_cache(seed)
    pbits = np.asarray(grp.lrf.convert_to_bits(orbit[:, -P:]), dtype=int)
    proj = [tuple(b[j] for j in retained_local) for b in pbits]
    G_pred = seq_graph(proj)
    fp_patterns = {tuple(grp.symbol_to_bits(s, P)[j]
                         for j in retained_local) for s in fp_syms}
    assert fp_patterns <= set(G_pred.nodes()), \
        "projected FP pattern missing from parent quotient"

    # retrained model free run
    row = next(r for r in csv.DictReader(open(COND_CSV, newline=""))
               if r["sync_scope"] == "readout_retained"
               and float(r["distill_lambda"]) != 0
               and int(r["source_seed"]) == seed
               and int(r["candidate_id"]) == cand)
    # via freerun_cache: compute the rollout and the official FP verdict
    # (collect_fp_snapshot, mask-aware) once and reuse the persisted record
    import freerun_cache
    rec = freerun_cache.freerun_record(row["ckpt"], steps=STUDENT_STEPS,
                                       cut=STUDENT_CUT)
    sbits = rec["bits"][STUDENT_CUT:]
    sproj = [tuple(int(b[j]) for j in retained_local) for b in sbits]
    G_real = seq_graph(sproj)
    real_fp_patterns = freerun_cache.visited_fp_patterns(rec)

    # post-transient fidelity convention (2026-09-25)
    e_cut = freerun_cache.posttransient_fidelity(rec)[0]
    metrics = (f"$Q_{{vis}}={int(rec['Q_vis'])}$,  "
               f"$|\\hat\\Sigma|="
               f"{int(rec['n_posttransient_symbols'])}$,  "
               f"$E_{{\\mathrm{{stsp}}}}={e_cut:.2f}$")
    return (G_pred, G_real, fp_patterns, real_fp_patterns,
            retained_local, metrics)


def draw_graph(ax, G, pos, fp_patterns, star_patterns=None,
               ghost_nodes=(), r=0.10, fs=12, linear=False,
               fixed_xlim=None, fixed_ylim=None,
               label_weight="normal"):
    if star_patterns is None:
        star_patterns = fp_patterns
    ax.set_axis_off()
    max_sag = 0.0
    for u, v in G.edges():
        if linear:
            # nodes sit on a line: adjacent pairs get the standard
            # small arc, longer edges bow out proportionally so no
            # edge crosses another or passes through a node
            span = abs(pos[u][0] - pos[v][0])
            rad = 0.12 if span <= 1.01 else 0.14 * span
            max_sag = max(max_sag, rad * span / 2)
        else:
            rad = 0.12
        ax.add_patch(FancyArrowPatch(
            pos[u], pos[v], arrowstyle="-|>", mutation_scale=12,
            color="0.4", lw=1.0, shrinkA=14, shrinkB=14, zorder=2,
            connectionstyle=f"arc3,rad={rad}"))
    for n in G.nodes():
        x, y = pos[n]
        is_fp = n in fp_patterns
        ax.add_patch(Circle((x, y), r, facecolor=C_FP if is_fp else C_FREE,
                            edgecolor=C_EDGE, lw=1.2, zorder=3))
        if n in star_patterns:
            ax.plot(x, y + 0.035, marker="*", ms=9, color="black",
                    mec="white", mew=0.4, ls="", zorder=5)
        ax.text(x, y - r - 0.05, "".join(map(str, n)), ha="center",
                va="top", fontsize=fs, zorder=6, color="0.15",
                fontweight=label_weight)
    for n in ghost_nodes:      # predicted-FP patterns never visited
        x, y = pos[n]
        ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor="#c22525",
                            lw=1.4, ls=(0, (4, 2)), zorder=3))
        ax.text(x, y - r - 0.05, "".join(map(str, n)), ha="center",
                va="top", fontsize=fs, zorder=6, color="#c22525",
                fontweight=label_weight)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    m = 0.32
    my = max(m, max_sag + 0.18)
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    else:
        ax.set_xlim(min(xs) - m, max(xs) + m)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    else:
        ax.set_ylim(min(ys) - my - 0.08, max(ys) + my)
    ax.set_aspect("equal")


def chain_order(G):
    """Node order along an undirected path graph, or None."""
    U = G.to_undirected()
    if not nx.is_connected(U):
        return None
    deg = dict(U.degree())
    ends = [n for n, d in deg.items() if d == 1]
    if len(ends) != 2 or any(d > 2 for d in deg.values()):
        return None
    order = nx.shortest_path(U, min(ends), max(ends))
    return order if len(order) == U.number_of_nodes() else None


def render(examples, basename, chain_layout=False):
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        14,
        "figure.dpi":       120,
    })
    n_rows = len(examples)
    figsize = ((11.5, 2.9 * n_rows) if chain_layout
               else (10.5, 3.6 * n_rows))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize,
                             constrained_layout=True)
    for i, (seed, cand, label) in enumerate(examples):
        (G_pred, G_real, fp_patterns, real_fp_patterns,
         retained_local, metrics) = load_example(seed, cand)
        order = chain_order(G_real) if chain_layout else None
        if order is not None:
            # realized graph is a chain: lay its nodes on a gently
            # zig-zagged line (no edge crossings); the prediction
            # inherits the same positions
            pos_real = {n: (0.9 * j, 0.16 * (-1) ** j)
                        for j, n in enumerate(order)}
            leftover = [n for n in G_pred.nodes() if n not in pos_real]
            for k, n in enumerate(leftover):
                pos_real[n] = (0.9 * (len(order) + k), 0.0)
            pos_pred = pos_real
            linear = True
        else:
            pos_pred = nx.spring_layout(G_pred.to_undirected(),
                                        seed=1, k=1.2)
            shared = [n for n in G_real.nodes() if n in pos_pred]
            extra = [n for n in G_real.nodes() if n not in pos_pred]
            pos_real = dict(pos_pred)
            if extra:
                pos_real = nx.spring_layout(
                    G_real.to_undirected(),
                    pos={**{n: pos_pred[n] for n in shared},
                         **{n: np.random.default_rng(2).uniform(-1, 1, 2)
                            for n in extra}},
                    fixed=shared if shared else None, seed=2, k=1.2)
            linear = False
        ghost = sorted(fp_patterns - set(G_real.nodes()))

        draw_graph(axes[i, 0], G_pred, pos_pred, fp_patterns,
                   linear=linear)
        draw_graph(axes[i, 1], G_real, {**pos_pred, **pos_real},
                   fp_patterns, star_patterns=real_fp_patterns,
                   ghost_nodes=ghost, linear=linear)
        axes[i, 0].set_title(
            f"{label} — seed{seed} cand{cand} "
            f"(retained ReLU {retained_local})\n"
            f"Predicted quotient: {G_pred.number_of_nodes()} symbols",
            fontsize=13, loc="left")
        axes[i, 1].set_title(
            "Retrained model free run\n" + metrics,
            fontsize=13, loc="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{basename}.{ext}"
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
    print(f"saved: {OUT_DIR / basename}.pdf / .svg / .png")


def _linear_maxsag(G, pos):
    """(upward, downward) max arc sagitta: arc3 with rad>0 bows to the
    left of the travel direction, i.e. up for left-to-right edges and
    down for right-to-left ones."""
    up = down = 0.0
    for u, v in G.edges():
        span = abs(pos[u][0] - pos[v][0])
        rad = 0.12 if span <= 1.01 else 0.14 * span
        sag = rad * span / 2
        if pos[v][0] > pos[u][0]:
            up = max(up, sag)
        else:
            down = max(down, sag)
    return up, down


def render_nonchain_split():
    """The nonchain-success comparison as TWO separate figures
    (predicted quotients / realized chains), all panels sharing the
    same data limits so node sizes match across every graph."""
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        14,
        "figure.dpi":       120,
    })
    rows = []
    for seed, cand, label in NONCHAIN_EXAMPLES:
        (G_pred, G_real, fp_patterns, real_fp_patterns,
         retained_local, metrics) = load_example(seed, cand)
        order = chain_order(G_real)
        assert order is not None, "realized graph is not a chain"
        pos = {n: (0.9 * j, 0.16 * (-1) ** j)
               for j, n in enumerate(order)}
        rows.append(dict(seed=seed, cand=cand, label=label,
                         G_pred=G_pred, G_real=G_real, pos=pos,
                         fp=fp_patterns, rfp=real_fp_patterns,
                         ret=retained_local, metrics=metrics))
    # shared limits -> identical rendered node size everywhere
    xmax = max(max(p[0] for p in r["pos"].values()) for r in rows)
    xlim = (-0.32, xmax + 0.32)
    # node size is fixed by the shared xlim + figure width (equal
    # aspect, width-bound), so each side may use its own tight ylim
    # without changing the rendered node size

    # per-column tight y-limits; node size stays identical everywhere
    # because both columns share xlim and equal panel widths
    ylims = {}
    for key in ("G_pred", "G_real"):
        sags = [_linear_maxsag(r[key], r["pos"]) for r in rows]
        up = max(s[0] for s in sags)
        down = max(s[1] for s in sags)
        ylims[key] = (-0.16 - max(0.32, down + 0.18) - 0.08,
                      0.16 + max(0.32, up + 0.18))

    n = len(rows)
    xspan = xlim[1] - xlim[0]
    panel_w = 5.9
    row_h = ((ylims["G_pred"][1] - ylims["G_pred"][0]) / xspan
             * panel_w + 0.85)
    fig, axes = plt.subplots(n, 2,
                             figsize=(2 * panel_w + 1.0, row_h * n),
                             constrained_layout=True)
    # visible gap between the two columns
    fig.get_layout_engine().set(wspace=0.14)
    for i, r in enumerate(rows):
        draw_graph(axes[i, 0], r["G_pred"], r["pos"], r["fp"],
                   linear=True, fixed_xlim=xlim,
                   fixed_ylim=ylims["G_pred"], fs=19,
                   label_weight="bold")
        axes[i, 0].set_title(r["label"], fontsize=22, loc="left",
                             fontweight="bold")
        ghost = sorted(r["fp"] - set(r["G_real"].nodes()))
        draw_graph(axes[i, 1], r["G_real"], r["pos"], r["fp"],
                   star_patterns=r["rfp"], ghost_nodes=ghost,
                   linear=True, fixed_xlim=xlim,
                   fixed_ylim=ylims["G_real"], fs=19,
                   label_weight="bold")
        axes[i, 1].set_title(
            "Retrained model free run\n" + r["metrics"],
            fontsize=22, loc="left", fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{NONCHAIN_BASENAME}.{ext}"
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
    print(f"saved: {OUT_DIR / NONCHAIN_BASENAME}.pdf / .svg / .png")


def main():
    render(EXAMPLES, FIG_BASENAME, chain_layout=False)
    render_nonchain_split()   # single 3x2 figure with column gap


if __name__ == "__main__":
    main()
