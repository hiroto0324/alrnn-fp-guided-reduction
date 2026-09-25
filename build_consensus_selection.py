#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consensus symbolic-complexity candidate selection from the reduction
hierarchy (replaces min-node-count selection).

Principle (2026-09-22): first fix the target nonlinear capacity
P_D = P_min, then infer the SUPPORTED symbolic complexity at that
capacity from the reduction tree by a parent-balanced vote:

    A vote = one DISTINCT admissible ReLU-linearization configuration
    (mask) at P_eff = P_D — i.e. duplicate tree nodes reaching the same
    mask via different pruning paths count once per parent. This is
    support by reduction-hierarchy structure, NOT a vote of independent
    models. Two support statistics are computed and stored:
      pooled          H_pool(k)  = #{masks with K_q=k} / #{all masks}
      parent-balanced h_s(k) = n_s(k)/sum_j n_s(j),
                      H_macro(k) = mean_s h_s(k)
    K_hat = argmax_k H_macro(k) (parent-balanced is the decision rule so
    tree-rich parents do not dominate); consensus C = H_macro(K_hat).

Kept layers: every k with H(k) >= keep_ratio * C (so a weak/bimodal mode
keeps the runner-up too). Admissible = not rejected and no fixed-point
collision, i.e. the FP-identity preservation constraint of the tree.

Per parent, up to --per_parent_cap candidates per kept layer are chosen
DETERMINISTICALLY (sorted by (K_q, deleted_set_str)); they are appended
to the parent's spec JSON (idempotent: dedup by deleted set, tagged
selection_source='consensus') so the existing retraining machinery can
address them by candidate_id. A benchsel-style selection JSON with the
full consensus statistics is written for the retraining launcher
(--selection_json).
"""

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

import system_config as sc
import run_benchmark_training as bt

SPEC_DIR = Path("results/relu_hierarchy/retraining_specs")


def hierarchy_nodes(system, seed, p_parent):
    # chua uses the legacy dir naming (m20_p10_original_nint128_seedN);
    # m20_p10{system}_seedN
    stem = (f"m20_p{p_parent}_original_nint128_seed{seed}"
            if system == "chua"
            else f"m20_p{p_parent}{system}_seed{seed}")
    f = (f"results/relu_hierarchy/{stem}/"
         f"relu_prune_hierarchy_{stem}.csv")
    if not Path(f).exists():
        return None
    out = []
    for r in csv.DictReader(open(f, newline="")):
        try:
            rej = str(r["is_rejected"]).lower() in ("true", "1")
            coll = str(r["fixed_collision"]).lower() in ("true", "1")
            if rej or coll:
                continue
            out.append(dict(
                p_eff=int(float(r["num_remaining_relu"])),
                k=int(float(r["num_clusters"])),
                deleted=sorted(int(x) for x in
                               r["deleted_set_str"].replace(";", ",")
                               .split(",") if x.strip() != "")))
        except (KeyError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["rossler", "lorenz63", "chua"])
    ap.add_argument("--p_eff", type=int, default=None,
                    help="target capacity (default: system P_direct)")
    ap.add_argument("--per_parent_cap", type=int, default=3,
                    help="max candidates per parent per kept layer")
    ap.add_argument("--keep_ratio", type=float, default=0.5,
                    help="keep layer k if H(k) >= keep_ratio * H(K_hat)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = sc.get_system(args.system)
    p_d = args.p_eff if args.p_eff is not None else cfg["P_direct"]
    p_parent = cfg["P_parent"]

    scr_name = ("source_screening_original_nint128_m20_p10.json"
                if args.system == "chua"
                else f"source_screening_{args.system}"
                     f"_nint128_m20_p{p_parent}.json")
    scr = json.load(open(SPEC_DIR / scr_name))
    seeds = scr["successful_seeds"]
    if args.system == "chua" and not args.dry_run:
        raise SystemExit("[guard] chua is --dry_run only here: do not "
                         "append consensus candidates to existing specs")

    # ── consensus vote (votes = distinct masks per parent) ──
    per_parent = {}
    for s in seeds:
        nodes = hierarchy_nodes(args.system, s, p_parent)
        if nodes is None:
            print(f"  [WARN] seed {s}: hierarchy CSV missing")
            continue
        seen, at_pd = set(), []
        for n in nodes:
            if n["p_eff"] != p_d:
                continue
            t = tuple(n["deleted"])
            if t not in seen:              # distinct configurations only
                seen.add(t)
                at_pd.append(n)
        if at_pd:
            per_parent[s] = at_pd
    all_k = sorted({n["k"] for v in per_parent.values() for n in v})
    pool = [n["k"] for v in per_parent.values() for n in v]
    H_pool = {k: pool.count(k) / len(pool) for k in all_k}
    H = {k: float(np.mean([
            sum(1 for n in v if n["k"] == k) / len(v)
            for v in per_parent.values()])) for k in all_k}
    k_hat = max(H, key=H.get)
    C = H[k_hat]
    kept = sorted(k for k in all_k if H[k] >= args.keep_ratio * C)
    runner = max((H[k] for k in all_k if k != k_hat), default=0.0)
    print(f"[consensus] {args.system} P_D={p_d}: parents with nodes = "
          f"{len(per_parent)}/{len(seeds)}  (votes = distinct masks)")
    for k in all_k:
        print(f"  k={k}: H_macro={H[k]:.3f}  H_pool={H_pool[k]:.3f}"
              + ("   <- K_hat" if k == k_hat else ""))
    print(f"  C = {C:.3f}  gap to runner-up = {C - runner:.3f}  "
          f"kept layers = {kept}  "
          f"pooled/balanced agree: {max(H_pool, key=H_pool.get) == k_hat}")

    # ── deterministic per-parent candidate pick + spec augmentation ──
    selected = []
    for s, nodes in sorted(per_parent.items()):
        spec_path = (SPEC_DIR / (bt.make_tag(cfg, p_parent, s)
                                 + "_minimal_symbol_reductions.json"))
        spec = json.load(open(spec_path))
        existing = {tuple(sorted(c["deleted_relu_local_indices"])): c
                    for c in spec["candidates"]}
        next_id = 1 + max((int(c["candidate_id"])
                           for c in spec["candidates"]), default=-1)
        changed = False
        for k in kept:
            picks = sorted((n for n in nodes if n["k"] == k),
                           key=lambda n: (n["k"], n["deleted"]))
            # dedup within tree (same mask may appear on several paths)
            seen = set()
            uniq = []
            for n in picks:
                t = tuple(n["deleted"])
                if t not in seen:
                    seen.add(t)
                    uniq.append(n)
            for n in uniq[:args.per_parent_cap]:
                t = tuple(n["deleted"])
                if t in existing:
                    c = existing[t]
                else:
                    c = dict(candidate_id=next_id,
                             num_clusters=n["k"],
                             P_effective=p_d,
                             deleted_relu_local_indices=list(n["deleted"]),
                             remaining_relu_local_indices=[
                                 j for j in range(p_parent)
                                 if j not in n["deleted"]],
                             edge_types=None, is_leaf=None,
                             selection_source="consensus")
                    next_id += 1
                    spec["candidates"].append(c)
                    existing[t] = c
                    changed = True
                selected.append(dict(
                    source_seed=int(s),
                    candidate_id=int(c["candidate_id"]),
                    num_clusters=int(n["k"]), P_original=p_parent,
                    P_effective=p_d,
                    deleted_relu_local_indices=list(n["deleted"]),
                    retained_relu_local_indices=[
                        j for j in range(p_parent)
                        if j not in n["deleted"]],
                    spec_path=str(spec_path)))
        if changed and not args.dry_run:
            spec["num_candidates"] = len(spec["candidates"])
            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=2)

    out = dict(system=args.system, P_effective_target=p_d,
               vote_unit="distinct admissible ReLU-linearization "
                         "configurations (masks) per parent",
               consensus_H_macro={str(k): H[k] for k in all_k},
               consensus_H_pooled={str(k): H_pool[k] for k in all_k},
               K_hat=k_hat, consensus_strength=C,
               runner_up_gap=C - runner, keep_ratio=args.keep_ratio,
               k_layers_kept=kept, per_parent_cap=args.per_parent_cap,
               n_selected=len(selected),
               parents_with_nodes=sorted(per_parent),
               selected_candidates=selected)
    out_path = (SPEC_DIR / f"benchsel_consensus_{args.system}"
                           f"_p{p_d}.json")
    if not args.dry_run:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    per_k = collections.Counter(c["num_clusters"] for c in selected)
    print(f"[selection] total {len(selected)} candidates "
          f"({dict(per_k)}) -> {out_path}"
          + ("  [DRY: nothing written]" if args.dry_run else ""))


if __name__ == "__main__":
    main()
