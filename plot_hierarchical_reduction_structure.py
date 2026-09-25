#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panel (b): a certified minimal symbolic representation is embedded in the
trained overparameterized parent (structural result, no retraining).

Left-to-right: parent symbolic graph (P_parent = 10) -> two representative
intermediate quotient graphs along one ACTUAL valid reduction path -> the
minimal quotient graph (|Sigma_D| = 5, P_D = 3, both lower bounds attained)
-> the parent's autonomous Chua attractor colored by the SAME five minimal
quotient symbols pi_D(sigma(z_t)), with the five recovered parent fixed
points marked.

Everything displayed is reconstructed from the actual experiment:
- parent checkpoint  models/{tag}.pth  (representative seed chosen by a
  deterministic rule: smallest seed whose reduction spec contains a
  candidate with num_clusters=5 and P_effective=3)
- the reduction path from results/relu_hierarchy/.../relu_prune_hierarchy_
  *.csv (parent_set_str links), all levels valid and FP-preserving
- trajectory/symbols/graphs recomputed with graph_reduction_with_relu_
  pruning.py's own protocol (TF warmup 500 + 40k free run, 1k transient
  cutoff) and its own projection/quotient functions

Notation mapping (repository -> paper): deleted set D -> mathcal{D},
P_effective -> P_{mathcal{D}}, num_clusters -> |Sigma_{mathcal{D}}|.
"""

from pathlib import Path
import csv
import json
import re
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import torch

import graph_reduction_with_relu_pruning as grp
from tutorial import AL_RNN

# ============================================================
# Configuration
# ============================================================
Q                = 5
P_PARENT         = 10
TARGET_P_REDUCED = 3
TARGET_N_SYMBOLS = 5
M, N             = 20, 3

SHOW_TITLE       = False
SHOW_PANEL_LABEL = False

SPEC_DIR   = Path("results/relu_hierarchy/retraining_specs")
HIER_DIR   = Path("results/relu_hierarchy")
DATA_PATH  = Path("data/chua_3-scroll_train.npy")
OUT_DIR    = Path("figures/paper")
FIG_BASENAME = "chua_hierarchical_reduction_structure"

SPEC_GLOB  = ("chua_orig_nint128_bs16_sig0.0_lr1e-03-1e-05_tfp0_ramp200"
              "_hLR1_m20_p10_seed{s}_minimal_symbol_reductions.json")
HIER_CSV   = ("m20_p10_original_nint128_seed{s}/"
              "relu_prune_hierarchy_m20_p10_original_nint128_seed{s}.csv")

T_GEN, T_R, DELTA_T = 40000, 1000, 0.01   # grp.main() protocol
ATTRACTOR_STEPS = 20000                   # prefix of the SAME trajectory

# displayed |mathcal{D}| levels along the path (0 = parent, last = final)
DISPLAY_DEPTHS = [0, 3, 5, 7]

# five-category colorblind-friendly palette (Okabe-Ito) for the final
# quotient symbols — deliberately different from panels (c)/(d)
SYMBOL_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]


# ============================================================
# 1. Representative parent (deterministic rule)
# ============================================================
def find_representative():
    for s in range(30):
        f = SPEC_DIR / SPEC_GLOB.format(s=s)
        if not f.exists():
            continue
        spec = json.load(open(f))
        cands = [c for c in spec["candidates"]
                 if c["num_clusters"] == TARGET_N_SYMBOLS
                 and c["P_effective"] == TARGET_P_REDUCED]
        if cands:
            cands.sort(key=lambda c: sorted(c["deleted_relu_local_indices"]))
            return s, spec, cands[0]
    sys.exit("no qualifying parent (|Sigma_D|=5, P_D=3) found")


def aggregate_counts():
    """population summary: parents reaching |Sigma_D|=5 (and also P_D=3)."""
    n5 = n53 = 0
    for s in range(30):
        f = SPEC_DIR / SPEC_GLOB.format(s=s)
        if not f.exists():
            continue
        spec = json.load(open(f))
        c5 = [c for c in spec["candidates"]
              if c["num_clusters"] == TARGET_N_SYMBOLS]
        if c5:
            n5 += 1
            if any(c["P_effective"] == TARGET_P_REDUCED for c in c5):
                n53 += 1
    return n5, n53


# ============================================================
# 2-3. Trajectory + parent symbols (grp.main() protocol)
# ============================================================
def parent_trajectory(model, raw):
    x = torch.tensor(raw).unsqueeze(0)
    with torch.no_grad():
        warm = min(grp.TF_WARMUP_STEPS, x.size(1))
        z0 = grp._warmup_latent(model, x[:, :warm, :], alpha=1.0,
                                n_interleave=1)
        orbit = grp.predict_free_from_latent(
            model, z0, T_GEN + T_R).detach().numpy()[0][T_R:, :]
    bits = grp.lrf.convert_to_bits(orbit[:, -P_PARENT:])
    sym_dec = [str(int(''.join(map(str, map(int, b))), 2)) for b in bits]
    return orbit, np.asarray(bits, dtype=int), sym_dec


def build_parent_graph(sym_dec):
    G = nx.DiGraph()
    G.add_nodes_from(sorted(set(sym_dec), key=int))
    for a, b in zip(sym_dec[:-1], sym_dec[1:]):
        if a != b:
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


# ============================================================
# 4. Reduction path from the hierarchy CSV
# ============================================================
def load_reduction_path(seed, final_set):
    rows = list(csv.DictReader(open(HIER_DIR / HIER_CSV.format(s=seed))))
    bykey = {}
    for r in rows:
        bykey.setdefault(r["deleted_set_str"], r)
    key = ";".join(map(str, sorted(final_set)))
    if key not in bykey:
        sys.exit(f"final set {key} not in hierarchy CSV")
    cur, chain = bykey[key], []
    while True:
        chain.append(cur)
        if not cur["parent_set_str"]:
            break
        cur = bykey[cur["parent_set_str"]]
    chain.reverse()
    for r in chain:
        assert r["is_rejected"] in ("", "False"), \
            f"path level {r['deleted_set_str']} was rejected"
    return chain     # depth 1..7 rows (parent level 0 handled separately)


# ============================================================
# 5-6. Quotients along the path
# ============================================================
def quotient_level(G_parent, deleted_set, fp_nodes):
    clusters, assign, keys = grp.build_prune_clusters(
        list(G_parent.nodes()), P_PARENT, set(deleted_set))
    Qg = grp.build_quotient_graph_from_assign(G_parent, assign)
    fp_cids = [assign[n] for n in fp_nodes]
    return dict(clusters=clusters, assign=assign, keys=keys, graph=Qg,
                fp_cids=fp_cids, deleted=sorted(deleted_set))


# ============================================================
# 7. Recovered fixed points
# ============================================================
def recovered_fixed_points(model, bits):
    uniq = np.unique(bits, axis=0)
    fpd = grp.analyze_fixed_points_continuous(model, uniq, DELTA_T)
    real = {s: v for s, v in fpd.items() if v["type"] == "real"}
    out = {}
    for s, v in sorted(real.items()):
        out[str(int(s, 2))] = np.asarray(v["location"][:N], dtype=float)
    return out       # decimal symbol -> readout coords


# ============================================================
# 11. Consistency checks
# ============================================================
def validate(levels, fp_syms, sym_dec, final):
    assert len(fp_syms) == Q, f"parent has {len(fp_syms)} real FPs, not {Q}"
    assert len(final["keys"]) == TARGET_N_SYMBOLS
    assert P_PARENT - len(final["deleted"]) == TARGET_P_REDUCED
    assert len(set(final["fp_cids"])) == Q, "FP collision at final level"
    # all five final symbols are FP-containing
    assert set(final["fp_cids"]) == set(final["graph"].nodes()), \
        "final level contains an FP-free symbol"
    # every displayed quotient edge is an observed projected transition
    for lv in levels[1:]:
        proj = [lv["assign"][s] for s in sym_dec]
        seen = {(a, b) for a, b in zip(proj[:-1], proj[1:]) if a != b}
        assert set(lv["graph"].edges()) <= seen, \
            f"unobserved edge in quotient |D|={len(lv['deleted'])}"
    for lv in levels:
        assert len(set(lv["fp_cids"])) == Q, \
            f"FP collision at |D|={len(lv['deleted'])}"


# ============================================================
# 8-10. Plotting
# ============================================================
def graph_positions(G_parent):
    return nx.spring_layout(G_parent, seed=0, k=0.9)


def quotient_positions(level, parent_pos):
    """Quotient nodes inherit the mean position of their member parent
    symbols (visually trackable merging), lightly relaxed by a seeded
    spring step."""
    init = {}
    for key, members in level["clusters"].items():
        cid = level["keys"].index(key)
        pts = np.array([parent_pos[m] for m in members])
        init[cid] = pts.mean(axis=0)
    return nx.spring_layout(level["graph"], pos=init, iterations=25,
                            seed=0, k=0.9)


def draw_graph(ax, G, pos, fp_colors, emph=False):
    """fp_colors: {node -> color} for FP-containing symbols (same five
    colors as the minimal graph at every level); other nodes stay open."""
    ax.set_axis_off()
    # arrowheads scale with the panel's sparsity (fewer nodes -> longer
    # edges -> bigger heads) and the emphasized graph's larger nodes get
    # the biggest heads, pulled clear of the node circles
    n_nodes = max(G.number_of_nodes(), 1)
    s = min((31 / n_nodes) ** 0.5, 1.8)          # 31 nodes -> 1.0
    if emph:
        e_lw, e_ms, e_shrink = 1.1, 13, 10
    else:
        e_lw = min(0.7 * s, 1.0)
        e_ms = min(7.5 * s, 12)
        e_shrink = 4 + 2 * (s - 1)
    for u, v in G.edges():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="0.65",
                                    lw=e_lw, shrinkA=e_shrink,
                                    shrinkB=e_shrink,
                                    mutation_scale=e_ms), zorder=1)
    for n in G.nodes():
        x, y = pos[n]
        if n in fp_colors:
            # FP symbols sit ABOVE the plain nodes (zorder) so they are
            # never occluded in the dense parent graph
            fc, ec, r, z = (fp_colors[n], "black",
                            0.085 if emph else 0.062, 5)
        else:
            fc, ec, r, z = "white", "0.45", 0.05, 3
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                                lw=1.3 if emph else 0.8, zorder=z))
        if n in fp_colors:
            # FP symbol: black star (same mark as the fixed points in
            # the attractor panel), centered in the node
            ax.plot(x, y, marker="*", ms=10.5 if emph else 7.5,
                    color="black", mec="white", mew=0.4, ls="",
                    zorder=6)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    m = 0.22
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")


def make_figure(levels, fp_syms, orbit, sym_dec, final, n5, n53):
    # all fonts 1.5x
    plt.rcParams.update({
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size":        16.5,
        "axes.linewidth":   0.8,
        "figure.dpi":       120,
    })
    FS_LVL, FS_TRAJ, FS_LEG = 20.25, 18, 17

    # near-square layout: 2x2 graph grid (Z-shaped reduction order) on the
    # left (cells 1.3x the earlier size, rows pulled together), attractor
    # spanning both rows on the right
    fig = plt.figure(figsize=(11.6, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 2.1],
                          left=0.005, right=0.99, top=0.90, bottom=0.02,
                          wspace=0.10, hspace=0.30)
    gaxes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
             fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax3d = fig.add_subplot(gs[:, 2], projection="3d")

    parent_pos = graph_positions(levels[0]["graph"])
    poss = [parent_pos] + [quotient_positions(lv, parent_pos)
                           for lv in levels[1:]]
    final_colors = {cid: SYMBOL_COLORS[i]
                    for i, cid in enumerate(sorted(final["graph"].nodes()))}
    # every level colors its FP-containing symbols with the color of the
    # final quotient symbol that fixed point ends up in
    fp_final_color = {s: final_colors[final["assign"][s]]
                      for s in fp_syms.keys()}
    level_fp_colors = []
    for lv in levels:
        level_fp_colors.append({lv["assign"][s]: c
                                for s, c in fp_final_color.items()})

    titles = []
    for lv in levels:
        nD = len(lv["deleted"])
        nS = lv["graph"].number_of_nodes()
        if nD == 0:
            titles.append(f"$P_{{\\mathrm{{parent}}}}={P_PARENT}$,"
                          f"  $|\\Sigma|={nS}$")
        else:
            titles.append(f"$P_{{\\mathcal{{D}}}}={P_PARENT-nD}$,"
                          f"  $|\\Sigma_{{\\mathcal{{D}}}}|={nS}$")
    for i, (ax, lv, pos) in enumerate(zip(gaxes, levels, poss)):
        is_final = (i == len(levels) - 1)
        draw_graph(ax, lv["graph"], pos, level_fp_colors[i], emph=is_final)
        ax.set_title(titles[i], fontsize=FS_LVL,
                     fontweight="bold" if is_final else "normal", pad=4)

    # ── attractor colored by the final quotient symbol ──
    traj = orbit[:ATTRACTOR_STEPS:2, :N]        # stride 2: same line,
    cids = np.array([final["assign"][s]          # half the segments
                     for s in sym_dec[:ATTRACTOR_STEPS:2]])
    segs = np.stack([traj[:-1], traj[1:]], axis=1)
    seg_colors = [final_colors[c] for c in cids[:-1]]
    lc = Line3DCollection(segs, colors=seg_colors, linewidths=0.5, alpha=0.85)
    lc.set_rasterized(True)   # 20k 3D segments: raster-embed in PDF/SVG
    ax3d.add_collection3d(lc)
    for sym, xyz in fp_syms.items():
        ax3d.scatter(*xyz, marker="*", s=170, color="black", zorder=10,
                     edgecolors="white", linewidths=0.6, depthshade=False)
    ax3d.set_xlim(traj[:, 0].min(), traj[:, 0].max())
    ax3d.set_ylim(traj[:, 1].min(), traj[:, 1].max())
    ax3d.set_zlim(traj[:, 2].min(), traj[:, 2].max())
    ax3d.view_init(elev=22, azim=-60)
    ax3d.set_axis_off()          # trajectory + fixed points only
    ax3d.set_title("Trajectory partitioned by the final symbols",
                   fontsize=FS_TRAJ, pad=0, fontweight="bold")
    # fixed-point legend inside the plot (kept large)
    from matplotlib.lines import Line2D
    ax3d.legend([Line2D([], [], marker="*", color="black", ls="",
                        markersize=17, markeredgecolor="white",
                        markeredgewidth=0.6)],
                ["fixed point"], loc="lower right", frameon=False,
                fontsize=FS_LEG, handletextpad=0.25, borderaxespad=0.1)

    if SHOW_TITLE:
        fig.suptitle("Hierarchical reductions are embedded in the parent")
    if SHOW_PANEL_LABEL:
        fig.text(0.005, 0.97, "(b)", fontsize=13, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {"dpi": 250, "transparent": True}),
                    ("svg", {"dpi": 250, "transparent": True}),
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


# ============================================================
# 12. Audit outputs
# ============================================================
def save_audit(seed, levels, fp_syms, orbit, sym_dec, final):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{FIG_BASENAME}_metadata.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_seed", "reduction_level", "linearized_relu_set",
                    "P_mathcalD", "n_symbols", "n_fp_symbols"])
        for i, lv in enumerate(levels):
            w.writerow([seed, i, ";".join(map(str, lv["deleted"])),
                        P_PARENT - len(lv["deleted"]),
                        lv["graph"].number_of_nodes(),
                        len(set(lv["fp_cids"]))])
    with open(OUT_DIR / "chua_minimal_symbol_trajectory.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "x1", "x2", "x3", "parent_symbol",
                    "projected_symbol"])
        for t, (row, s) in enumerate(zip(orbit[:, :N], sym_dec)):
            w.writerow([t, f"{row[0]:.6f}", f"{row[1]:.6f}",
                        f"{row[2]:.6f}", s, final["assign"][s]])


def main():
    seed, spec, cand = find_representative()
    final_set = sorted(cand["deleted_relu_local_indices"])
    print(f"representative parent seed: {seed}")
    print(f"final mathcal{{D}} = {final_set}  "
          f"(P_D={cand['P_effective']}, |Sigma_D|={cand['num_clusters']})")

    chain = load_reduction_path(seed, final_set)
    print("reduction path (|D|: D):")
    for r in chain:
        print(f"  {r['num_deleted']}: {{{r['deleted_set_str']}}}  "
              f"clusters={float(r['num_clusters']):.0f} "
              f"fp={float(r['num_fixed_nodes']):.0f}")

    # trajectory + FP cache (rollout and eigen-analysis are the slow parts;
    # invalidated automatically when the checkpoint file changes)
    ckpt = spec["source_model_path"]
    cache = Path(f"results/relu_hierarchy/panel_b_traj_cache_seed{seed}.npz")
    stamp = f"{ckpt}|{Path(ckpt).stat().st_mtime_ns}|{T_GEN}|{T_R}"
    orbit = fp_cached = None
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            if str(z["stamp"]) == stamp:
                orbit = z["orbit"]
                fp_cached = dict(zip([str(s) for s in z["fp_symbols"]],
                                     z["fp_coords"]))
                print(f"trajectory cache HIT: {cache}")
    if orbit is None:
        model = AL_RNN(M=M, P=P_PARENT, N=N)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        raw = np.load(DATA_PATH).astype(np.float32)
        orbit, bits, sym_dec = parent_trajectory(model, raw)
        fp_cached = recovered_fixed_points(model, bits)
        np.savez_compressed(
            cache, stamp=stamp, orbit=orbit.astype(np.float32),
            fp_symbols=np.array(list(fp_cached.keys())),
            fp_coords=np.array(list(fp_cached.values())))
        print(f"trajectory cache saved: {cache}")
    bits = np.asarray(grp.lrf.convert_to_bits(orbit[:, -P_PARENT:]),
                      dtype=int)
    sym_dec = [str(int(''.join(map(str, b)), 2)) for b in bits]
    G_parent = build_parent_graph(sym_dec)
    print(f"parent graph: {G_parent.number_of_nodes()} symbols, "
          f"{G_parent.number_of_edges()} transitions")

    fp_syms = fp_cached
    print("recovered parent fixed points (symbol -> readout coords):")
    for s, xyz in fp_syms.items():
        print(f"  {s}: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    fp_nodes = list(fp_syms.keys())

    # displayed levels: parent + path rows at DISPLAY_DEPTHS
    by_depth = {int(r["num_deleted"]): r for r in chain}
    levels = []
    for d in DISPLAY_DEPTHS:
        if d == 0:
            # parent level: identity "quotient" on the original symbol ids
            levels.append(dict(clusters={n: [n] for n in G_parent.nodes()},
                               assign={n: n for n in G_parent.nodes()},
                               keys=sorted(G_parent.nodes(), key=int),
                               graph=G_parent,
                               fp_cids=list(fp_nodes),
                               deleted=[]))
        else:
            D = [int(x) for x in by_depth[d]["deleted_set_str"].split(";")]
            lv = quotient_level(G_parent, D, fp_nodes)
            # cross-check against the hierarchy CSV numbers
            assert lv["graph"].number_of_nodes() == int(
                float(by_depth[d]["num_clusters"])), \
                f"recomputed clusters mismatch at |D|={d}"
            levels.append(lv)
    final = levels[-1]

    # parent level uses relabeled graph; rebuild parent level positions on
    # the relabeled ids but fp set stays consistent
    fp_readout_syms = {s: xyz for s, xyz in fp_syms.items()}
    # symbol assignment map for the trajectory at the final level
    validate(levels, fp_syms, sym_dec, final)
    print("all consistency checks passed "
          "(5 FPs, 5 distinct final symbols, all FP-containing, "
          "edges observed, path valid)")
    print("FP symbol -> final quotient symbol:",
          {s: final["assign"][s] for s in fp_nodes})

    n5, n53 = aggregate_counts()
    print(f"population: {n5}/30 parents reach |Sigma_D|=5, "
          f"{n53}/30 also P_D=3")

    make_figure(levels, fp_readout_syms, orbit, sym_dec, final, n5, n53)
    save_audit(seed, levels, fp_syms, orbit, sym_dec, final)
    print(f"saved: {OUT_DIR / FIG_BASENAME}.pdf/.svg/.png, _metadata.csv, "
          "chua_minimal_symbol_trajectory.csv")


if __name__ == "__main__":
    main()
