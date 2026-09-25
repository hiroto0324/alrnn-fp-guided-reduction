"""
Graph-based clustering with compactness penalty for AL-RNN symbol analysis.

This version extends the base clustering algorithm with an efficient compactness
penalty term to prevent clusters from becoming too elongated. The objective
function for the heuristic solver is:

    objective = edge_types + lambda * max_mean_dist_to_seed

where:
- edge_types: number of distinct edge types in contracted graph
- max_mean_dist_to_seed: maximum of mean distances to seed across all clusters
- lambda: penalty weight (configurable via COMPACTNESS_PENALTY_LAMBDA)

The mean distance to seed is the average shortest path distance from the seed
node (real fixed point) to all other nodes in the cluster. This is much faster
to compute than diameter (O(n) vs O(n²) per cluster) while still encouraging
compact, round-shaped clusters.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataset import TimeSeriesDataset
from linear_region_functions import *
import linear_region_functions as lrf
# from linear_region_functions_K_regions import *
# import linear_region_functions_K_regions as lrf_k
from metrics import state_space_divergence_binning, power_spectrum_error

from matplotlib.colors import Normalize
from collections import Counter
import seaborn as sns
import copy

# from functions import *
from tutorial import *

import networkx as nx


@torch.no_grad()
def _warmup_latent(model, x_warmup, alpha=1.0, n_interleave=1):
    """Teacher-force the RNN for T_warmup steps and return the final latent state.

    x_warmup : (batch, T_warmup, N)
    Returns z : (batch, M)  — the hidden state after the last warmup step.
    """
    x_ = x_warmup.permute(1, 0, 2)   # (T_warmup, batch, N)
    T_w = x_.size(0)
    z = x_[0] @ model.B
    z = teacher_force(z, x_[0], alpha=1.0)
    for t in range(T_w):
        if t > 0 and (t % n_interleave == 0):
            z = teacher_force(z, x_[t], alpha)
        z = model(z)
    return z


@torch.no_grad()
def predict_free_from_latent(model, z0, T):
    """Free-run the model starting from a given latent state z0.

    z0 : (batch, M)
    Returns Z : (batch, T, M)
    """
    b = z0.size(0)
    Z = torch.empty(size=(T, b, model.M), device=z0.device)
    z = z0.clone()
    for t in range(T):
        z = model(z)
        Z[t] = z
    return Z.permute(1, 0, 2)
import matplotlib.patches as mpatches

# ============================================================
# Graph contraction clustering (minimize #edge types in contracted graph)
#  - #clusters = #real nodes
#  - each real node is fixed to its own cluster
#  - clusters are connected in an undirected "adjacency" graph (no enclaves)
#  - objective: minimize |{(i,j): exists u->v with c(u)=i, c(v)=j, i!=j}|
#  - NEW: compactness penalty (mean distance to seed) added to prevent elongated clusters
#        This is much faster than diameter penalty: O(n) vs O(n²) per cluster
# ============================================================

def _contracted_edge_types_from_counts(pair_cnt):
    return len(pair_cnt)


def _build_pair_counts(G_dir, assign):
    """
    pair_cnt[(i,j)] = number of edges u->v with assign[u]=i, assign[v]=j, i!=j.
    Stored only for pairs with count>0.
    """
    pair_cnt = {}
    for u, v in G_dir.edges():
        cu, cv = assign[u], assign[v]
        if cu != cv:
            key = (cu, cv)
            pair_cnt[key] = pair_cnt.get(key, 0) + 1
    return pair_cnt


def _delta_move_edge_types(G_dir, assign, pair_cnt, v, new_c):
    old_c = assign[v]
    if old_c == new_c:
        return 0

    changes = {}

    # outgoing v->u
    for u in G_dir.successors(v):
        cu = assign[u]
        if old_c != cu:
            changes[(old_c, cu)] = changes.get((old_c, cu), 0) - 1
        if new_c != cu:
            changes[(new_c, cu)] = changes.get((new_c, cu), 0) + 1

    # incoming u->v
    for u in G_dir.predecessors(v):
        cu = assign[u]
        if cu != old_c:
            changes[(cu, old_c)] = changes.get((cu, old_c), 0) - 1
        if cu != new_c:
            changes[(cu, new_c)] = changes.get((cu, new_c), 0) + 1

    delta = 0
    for key, d in changes.items():
        before = pair_cnt.get(key, 0)
        after = before + d
        if before <= 0 and after > 0:
            delta += 1
        elif before > 0 and after <= 0:
            delta -= 1
    return delta


def _apply_move_update_counts(G_dir, assign, pair_cnt, v, new_c):
    old_c = assign[v]
    if old_c == new_c:
        return

    # outgoing v->u
    for u in G_dir.successors(v):
        cu = assign[u]
        if old_c != cu:
            key = (old_c, cu)
            pair_cnt[key] = pair_cnt.get(key, 0) - 1
            if pair_cnt[key] <= 0:
                pair_cnt.pop(key, None)
        if new_c != cu:
            key = (new_c, cu)
            pair_cnt[key] = pair_cnt.get(key, 0) + 1

    # incoming u->v
    for u in G_dir.predecessors(v):
        cu = assign[u]
        if cu != old_c:
            key = (cu, old_c)
            pair_cnt[key] = pair_cnt.get(key, 0) - 1
            if pair_cnt[key] <= 0:
                pair_cnt.pop(key, None)
        if cu != new_c:
            key = (cu, new_c)
            pair_cnt[key] = pair_cnt.get(key, 0) + 1

    assign[v] = new_c


def _is_cluster_connected_after_removal(G_und, cluster_nodes, v):
    """Check if induced subgraph on cluster_nodes - {v} is connected (undirected)."""
    if v not in cluster_nodes:
        return True
    remaining = cluster_nodes - {v}
    if len(remaining) <= 1:
        return True
    start = next(iter(remaining))
    visited = {start}
    stack = [start]
    while stack:
        a = stack.pop()
        for b in G_und.neighbors(a):
            if b in remaining and b not in visited:
                visited.add(b)
                stack.append(b)
    return len(visited) == len(remaining)


def _compute_cluster_compactness(G_und, cluster_nodes, seed_node):
    """
    Compute the mean distance from seed to all nodes in a cluster.
    This is much faster than diameter (O(n) vs O(n²)).
    Returns 0 for single-node clusters, infinity if disconnected.
    """
    if len(cluster_nodes) <= 1:
        return 0.0

    subgraph = G_und.subgraph(cluster_nodes)

    # Check if seed is in the cluster
    if seed_node not in subgraph:
        return float('inf')

    # Compute shortest path lengths from seed to all nodes
    try:
        lengths = nx.single_source_shortest_path_length(subgraph, seed_node)
    except:
        return float('inf')  # disconnected

    # Check if all nodes are reachable
    if len(lengths) < len(cluster_nodes):
        return float('inf')  # disconnected

    # Compute mean distance
    mean_dist = sum(lengths.values()) / len(lengths)
    return mean_dist


def _compute_max_compactness(G_und, cluster_nodes_list, seed_nodes):
    """
    Compute the maximum mean distance to seed across all clusters.
    """
    max_compact = 0.0
    for i, cluster in enumerate(cluster_nodes_list):
        if len(cluster) > 0:
            compact = _compute_cluster_compactness(G_und, cluster, seed_nodes[i])
            if compact == float('inf'):
                return float('inf')
            max_compact = max(max_compact, compact)
    return max_compact


def _initial_region_growing(G_und, real_nodes, rng):
    """
    Multi-source region growing that guarantees each cluster connected in G_und.
    """
    from collections import deque

    k = len(real_nodes)
    assign = {}
    clusters = [set() for _ in range(k)]

    for i, r in enumerate(real_nodes):
        assign[r] = i
        clusters[i].add(r)

    unassigned = set(G_und.nodes()) - set(real_nodes)

    # frontier per cluster
    frontiers = [set() for _ in range(k)]
    for i, r in enumerate(real_nodes):
        for nb in G_und.neighbors(r):
            if nb in unassigned:
                frontiers[i].add(nb)

    while unassigned:
        candidates = [(i, v) for i in range(k) for v in frontiers[i]]
        if not candidates:
            raise RuntimeError("Cannot assign all nodes while keeping clusters connected (adjacency graph disconnected).")
        i, v = rng.choice(candidates)
        assign[v] = i
        clusters[i].add(v)
        unassigned.remove(v)

        for j in range(k):
            frontiers[j].discard(v)
        for nb in G_und.neighbors(v):
            if nb in unassigned:
                frontiers[i].add(nb)

    return assign


def solve_clusters_heuristic(G_dir, G_und, real_nodes, max_iters=20000, seed=42, anneal=True, compactness_penalty_lambda=0.0):
    """
    Heuristic: region growing init + boundary-node moves preserving connectivity.
    Minimizes contracted edge types + lambda * max_mean_dist_to_seed.
    """
    rng = random.Random(seed)

    # initial assignment
    assign = _initial_region_growing(G_und, real_nodes, rng)

    k = len(real_nodes)
    cluster_nodes = [set() for _ in range(k)]
    for v, c in assign.items():
        cluster_nodes[c].add(v)

    fixed = set(real_nodes)

    pair_cnt = _build_pair_counts(G_dir, assign)
    edge_types = _contracted_edge_types_from_counts(pair_cnt)

    # Compute initial cluster compactness (mean distance to seed)
    cluster_compactness = [_compute_cluster_compactness(G_und, cluster_nodes[i], real_nodes[i]) for i in range(k)]
    max_compactness = max(cluster_compactness) if cluster_compactness else 0.0

    cur_obj = edge_types + compactness_penalty_lambda * max_compactness
    best_assign = dict(assign)
    best_obj = cur_obj

    und_nbrs = {v: list(G_und.neighbors(v)) for v in G_und.nodes()}

    def is_boundary(v):
        c = assign[v]
        for nb in und_nbrs[v]:
            if assign[nb] != c:
                return True
        return False

    boundary = {v for v in G_und.nodes() if v not in fixed and is_boundary(v)}

    # annealing schedule
    def temperature(t):
        if not anneal:
            return 0.0
        T0, Tend = 1.0, 1e-3
        if max_iters <= 1:
            return Tend
        ratio = (Tend / T0) ** (t / (max_iters - 1))
        return T0 * ratio

    for it in range(max_iters):
        if not boundary:
            break

        v = rng.choice(tuple(boundary))
        old_c = assign[v]

        # candidate target clusters adjacent in G_und
        cand_clusters = {assign[nb] for nb in und_nbrs[v] if assign[nb] != old_c}
        if not cand_clusters:
            boundary.discard(v)
            continue

        # connectivity check for leaving cluster
        if not _is_cluster_connected_after_removal(G_und, cluster_nodes[old_c], v):
            boundary.discard(v)
            continue

        # choose best target by delta (edge types + compactness penalty)
        best_delta = None
        best_targets = []
        for new_c in cand_clusters:
            # Delta for edge types
            d_edge = _delta_move_edge_types(G_dir, assign, pair_cnt, v, new_c)

            # Delta for compactness penalty
            d_compact = 0.0
            if compactness_penalty_lambda > 0:
                # Compute new compactness for affected clusters
                old_cluster_after = cluster_nodes[old_c] - {v}
                new_cluster_after = cluster_nodes[new_c] | {v}

                compact_old_after = _compute_cluster_compactness(G_und, old_cluster_after, real_nodes[old_c])
                compact_new_after = _compute_cluster_compactness(G_und, new_cluster_after, real_nodes[new_c])

                # Compute max compactness after move
                new_compactness = list(cluster_compactness)
                new_compactness[old_c] = compact_old_after
                new_compactness[new_c] = compact_new_after
                new_max_compactness = max(new_compactness)

                d_compact = compactness_penalty_lambda * (new_max_compactness - max_compactness)

            d_total = d_edge + d_compact

            if best_delta is None or d_total < best_delta:
                best_delta = d_total
                best_targets = [new_c]
            elif d_total == best_delta:
                best_targets.append(new_c)

        new_c = rng.choice(best_targets)

        # Recompute delta for chosen move
        d_edge = _delta_move_edge_types(G_dir, assign, pair_cnt, v, new_c)
        d_compact = 0.0
        if compactness_penalty_lambda > 0:
            old_cluster_after = cluster_nodes[old_c] - {v}
            new_cluster_after = cluster_nodes[new_c] | {v}
            compact_old_after = _compute_cluster_compactness(G_und, old_cluster_after, real_nodes[old_c])
            compact_new_after = _compute_cluster_compactness(G_und, new_cluster_after, real_nodes[new_c])
            new_compactness = list(cluster_compactness)
            new_compactness[old_c] = compact_old_after
            new_compactness[new_c] = compact_new_after
            new_max_compactness = max(new_compactness)
            d_compact = compactness_penalty_lambda * (new_max_compactness - max_compactness)

        d = d_edge + d_compact

        accept = False
        if d <= 0:
            accept = True
        else:
            T = temperature(it)
            if anneal and T > 0:
                p = math.exp(-d / T)
                if rng.random() < p:
                    accept = True

        if not accept:
            continue

        # apply move
        cluster_nodes[old_c].remove(v)
        cluster_nodes[new_c].add(v)

        _apply_move_update_counts(G_dir, assign, pair_cnt, v, new_c)

        # Update compactness for affected clusters
        cluster_compactness[old_c] = _compute_cluster_compactness(G_und, cluster_nodes[old_c], real_nodes[old_c])
        cluster_compactness[new_c] = _compute_cluster_compactness(G_und, cluster_nodes[new_c], real_nodes[new_c])
        max_compactness = max(cluster_compactness)

        edge_types = _contracted_edge_types_from_counts(pair_cnt)
        cur_obj = edge_types + compactness_penalty_lambda * max_compactness

        # update boundary locally
        affected = {v}
        affected.update(und_nbrs[v])
        for u in affected:
            if u in fixed:
                continue
            if is_boundary(u):
                boundary.add(u)
            else:
                boundary.discard(u)

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = dict(assign)
            if best_obj == 0:
                break

    return best_assign, best_obj


def solve_clusters_milp(G_dir, G_und, real_nodes, time_limit_sec=300, msg=False):
    """
    Exact MILP (requires PuLP):
      - x[v,i] binary assignment
      - y[i,j] binary contracted edge existence
      - single-commodity flow per cluster to enforce connectivity in G_und
      - minimize sum_{i!=j} y[i,j]
    """
    try:
        import pulp
    except Exception as e:
        raise ImportError("MILP solver requires 'pulp'. Install with: pip install pulp") from e

    V = list(G_dir.nodes())
    k = len(real_nodes)
    n = len(V)
    M = n  # big-M

    # build arcs from undirected edges
    arcs = []
    for a, b in G_und.edges():
        arcs.append((a, b))
        arcs.append((b, a))

    out_arcs = {v: [] for v in V}
    in_arcs = {v: [] for v in V}
    for a, b in arcs:
        out_arcs[a].append((a, b))
        in_arcs[b].append((a, b))

    prob = pulp.LpProblem("MinContractedEdgeTypes", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (V, range(k)), lowBound=0, upBound=1, cat="Binary")
    y = pulp.LpVariable.dicts("y", (range(k), range(k)), lowBound=0, upBound=1, cat="Binary")
    f = pulp.LpVariable.dicts("f", (range(k), arcs), lowBound=0, upBound=M, cat="Integer")

    prob += pulp.lpSum(y[i][j] for i in range(k) for j in range(k) if i != j)

    # assignment
    for v in V:
        prob += pulp.lpSum(x[v][i] for i in range(k)) == 1

    # fix real nodes
    for i, r in enumerate(real_nodes):
        prob += x[r][i] == 1
        for j in range(k):
            if j != i:
                prob += x[r][j] == 0

    # define y by edges
    for u, v in G_dir.edges():
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                prob += y[i][j] >= x[u][i] + x[v][j] - 1

    # connectivity via flow in G_und
    for i, r in enumerate(real_nodes):
        # flow allowed only if both endpoints in cluster
        for a, b in arcs:
            prob += f[i][(a, b)] <= M * x[a][i]
            prob += f[i][(a, b)] <= M * x[b][i]

        # root balance: out - in = sum_{v != r} x[v,i]
        prob += (
            pulp.lpSum(f[i][arc] for arc in out_arcs[r]) - pulp.lpSum(f[i][arc] for arc in in_arcs[r])
            == pulp.lpSum(x[v][i] for v in V if v != r)
        )

        # other nodes: in - out = x[v,i]
        for v in V:
            if v == r:
                continue
            prob += (
                pulp.lpSum(f[i][arc] for arc in in_arcs[v]) - pulp.lpSum(f[i][arc] for arc in out_arcs[v])
                == x[v][i]
            )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec) if time_limit_sec else pulp.PULP_CBC_CMD(msg=msg)
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"MILP failed: status={pulp.LpStatus[status]}")

    assign = {}
    for v in V:
        chosen = None
        for i in range(k):
            val = pulp.value(x[v][i])
            if val is not None and val > 0.5:
                chosen = i
                break
        if chosen is None:
            raise RuntimeError(f"Could not decode assignment for {v}")
        assign[v] = chosen

    # objective value (edge types)
    obj = int(round(pulp.value(prob.objective)))
    return assign, obj


def solve_symbol_clustering(G_dir, G_und, real_nodes, method="heuristic", compactness_penalty_lambda=0.0):
    """
    Wrapper: returns (assign_idx, obj_value).
      assign_idx: node -> cluster_id (0..k-1)
      obj_value: minimized objective (edge types + compactness penalty for heuristic)
    """
    method = method.lower()
    if method == "milp":
        return solve_clusters_milp(
            G_dir, G_und, real_nodes,
            time_limit_sec=MILP_TIME_LIMIT_SEC,
            msg=False
        )
    elif method == "heuristic":
        return solve_clusters_heuristic(
            G_dir, G_und, real_nodes,
            max_iters=HEURISTIC_MAX_ITERS,
            seed=HEURISTIC_SEED,
            anneal=HEURISTIC_ANNEAL,
            compactness_penalty_lambda=compactness_penalty_lambda
        )
    else:
        raise ValueError(f"Unknown CLUSTER_SOLVER: {method}. Use 'milp' or 'heuristic'.")



# ============================================================
# Secondary metrics for cluster "beauty"/simplicity (tie-breakers)
# ============================================================

def compute_cluster_secondary_metrics(G_dir, G_und, assign, seed_nodes, cluster_id_seq=None, cluster_path_ids=None):
    """
    Compute secondary metrics for a clustering result.

    Parameters
    ----------
    G_dir : nx.DiGraph
        Directed graph used for objective (observed symbol transitions).
    G_und : nx.Graph
        Undirected adjacency graph used for connectivity (no enclaves).
    assign : dict[node -> cluster_id]
        Assignment from symbols (decimal string) to cluster id (0..k-1).
    seed_nodes : list[str]
        Real seed symbol (decimal string) for each cluster id i (seed_nodes[i]).
    cluster_id_seq : list[int] | None
        Cluster id sequence along the trajectory (length T). Optional.
    cluster_path_ids : list[int] | None
        Compressed cluster id path (consecutive duplicates removed). Optional.

    Returns
    -------
    metrics : dict[str, float|int]
    """
    import numpy as _np
    import networkx as _nx

    V = list(G_dir.nodes())
    k = len(seed_nodes)

    # --- objective-aligned quantities ---
    edge_types = set()
    cut_edges_total = 0
    for u, v in G_dir.edges():
        cu, cv = assign[u], assign[v]
        if cu != cv:
            edge_types.add((cu, cv))
            cut_edges_total += 1  # counts each crossing directed edge once

    # --- boundary complexity (undirected) ---
    boundary_nodes = 0
    for v in V:
        c = assign[v]
        for nb in G_und.neighbors(v):
            if assign[nb] != c:
                boundary_nodes += 1
                break

    # --- cluster sizes ---
    sizes = _np.zeros(k, dtype=int)
    for v in V:
        sizes[assign[v]] += 1

    # --- compactness: distances to seed within induced subgraph ---
    mean_dist_to_seed = []
    max_dist_to_seed = []
    for cid in range(k):
        nodes_c = [v for v in V if assign[v] == cid]
        if len(nodes_c) <= 1:
            mean_dist_to_seed.append(0.0)
            max_dist_to_seed.append(0.0)
            continue
        sub = G_und.subgraph(nodes_c)
        seed = seed_nodes[cid]
        # seed must be inside its cluster by construction, but be defensive
        if seed not in sub:
            seed = nodes_c[0]
        dist = _nx.single_source_shortest_path_length(sub, seed)
        # if something went wrong (should not, due to connectivity constraints)
        if len(dist) < len(nodes_c):
            # treat unreachable nodes as large penalty
            unreachable = len(nodes_c) - len(dist)
            mean_dist_to_seed.append(float("inf"))
            max_dist_to_seed.append(float("inf"))
        else:
            dvals = _np.array(list(dist.values()), dtype=float)
            mean_dist_to_seed.append(float(dvals.mean()))
            max_dist_to_seed.append(float(dvals.max()))

    # --- trajectory-derived secondary metrics (optional) ---
    traj_unique_edge_types = None
    traj_transitions = None
    traj_switch_rate = None
    if cluster_path_ids is not None and len(cluster_path_ids) >= 2:
        traj_types = set()
        for i in range(len(cluster_path_ids) - 1):
            a, b = cluster_path_ids[i], cluster_path_ids[i + 1]
            if a != b:
                traj_types.add((a, b))
        traj_unique_edge_types = len(traj_types)
        traj_transitions = len(cluster_path_ids) - 1

    if cluster_id_seq is not None and len(cluster_id_seq) >= 2:
        switches = sum(1 for i in range(len(cluster_id_seq) - 1) if cluster_id_seq[i] != cluster_id_seq[i + 1])
        traj_switch_rate = switches / (len(cluster_id_seq) - 1)

    metrics = {
        "k": k,
        "edge_types_min_obj": len(edge_types),
        "cut_edges_total": int(cut_edges_total),
        "boundary_nodes": int(boundary_nodes),
        "size_min": int(sizes.min()) if k > 0 else 0,
        "size_max": int(sizes.max()) if k > 0 else 0,
        "size_mean": float(sizes.mean()) if k > 0 else 0.0,
        "size_std": float(sizes.std()) if k > 0 else 0.0,
        "mean_dist_to_seed_avg": float(_np.mean(mean_dist_to_seed)) if k > 0 else 0.0,
        "mean_dist_to_seed_max": float(_np.max(mean_dist_to_seed)) if k > 0 else 0.0,
        "radius_to_seed_avg": float(_np.mean(max_dist_to_seed)) if k > 0 else 0.0,
        "radius_to_seed_max": float(_np.max(max_dist_to_seed)) if k > 0 else 0.0,
        "traj_unique_edge_types": traj_unique_edge_types,
        "traj_transitions": traj_transitions,
        "traj_switch_rate": traj_switch_rate,
    }
    return metrics


# ============================================================
# ReLU pruning-induced quotient graph hierarchy
# (independent from seed-based clustering above)
# ============================================================

def symbol_to_bits(symbol, P):
    """Convert decimal string node (e.g. '13') to length-P binary numpy array.
    Index convention: index 0 = MSB (leftmost bit in the binary string).
    """
    return np.array(list(format(int(symbol), f'0{P}b')), dtype=int)


def _node_label(dec_str, P, fmt):
    """Return display label for an original symbol (decimal string).
    fmt='decimal' keeps the decimal string; fmt='binary' returns P-bit binary string.
    """
    if fmt == 'binary':
        return format(int(dec_str), f'0{P}b')
    return dec_str


def _cluster_label(cluster_key, fmt):
    """Return display label for a quotient-graph cluster node (projected bit tuple).
    fmt='decimal' → decimal value of projected bits; fmt='binary' → joined bit string.
    """
    bits = ''.join(str(b) for b in cluster_key)
    if not bits:
        return '∅'
    if fmt == 'decimal':
        return str(int(bits, 2))
    return bits


def _quotient_edge_switch_labels(Q, cluster_to_proj_bits, deleted_set, P):
    """Return a dict {(u,v): label_str} with one label per undirected pair.

    For each unordered pair {u, v} that has at least one directed edge in Q,
    exactly one entry is added (whichever direction appears first in Q.edges()).
    The label lists the original ReLU indices that differ between the two
    projected bit patterns — undirected switching information.
    """
    remaining = sorted(set(range(P)) - deleted_set)
    labels = {}
    seen = set()
    for u, v in Q.edges():
        pair = frozenset((u, v))
        if pair in seen:
            continue
        seen.add(pair)
        bu = cluster_to_proj_bits.get(u)
        bv = cluster_to_proj_bits.get(v)
        if bu is None or bv is None:
            labels[(u, v)] = '?'
            continue
        orig_indices = [remaining[i] for i, (a, b) in enumerate(zip(bu, bv)) if a != b]
        labels[(u, v)] = ','.join(str(i) for i in orig_indices) if orig_indices else '–'
    return labels


def project_bits(bits, deleted_set):
    """Remove bits at indices in deleted_set and return remaining as tuple."""
    return tuple(int(b) for i, b in enumerate(bits) if i not in deleted_set)


def build_prune_clusters(nodes, P, deleted_set):
    """
    Group nodes by projected bit pattern (bits NOT in deleted_set).

    Returns
    -------
    clusters   : dict[projected_pattern_tuple -> list[node]]
    assign     : dict[node -> cluster_id]
    cluster_keys: sorted list of projected_pattern_tuples (cluster_id = index)
    """
    groups = {}
    for node in nodes:
        bits = symbol_to_bits(node, P)
        key = project_bits(bits, deleted_set)
        if key not in groups:
            groups[key] = []
        groups[key].append(node)

    cluster_keys = sorted(groups.keys())
    clusters = {k: groups[k] for k in cluster_keys}
    assign = {}
    for cid, k in enumerate(cluster_keys):
        for node in groups[k]:
            assign[node] = cid
    return clusters, assign, cluster_keys


def fixed_point_collision(assign, fixed_nodes):
    """Return True if two or more fixed_nodes are mapped to the same cluster."""
    seen_clusters = set()
    for node in fixed_nodes:
        if node in assign:
            cid = assign[node]
            if cid in seen_clusters:
                return True
            seen_clusters.add(cid)
    return False


def build_quotient_graph_from_assign(G_dir, assign):
    """
    Build quotient DiGraph from G_dir using assign.
    Self-loops are excluded. Edge weight = number of original edges mapped to it.
    """
    Q = nx.DiGraph()
    all_cids = sorted(set(assign.values()))
    Q.add_nodes_from(all_cids)

    edge_counts = {}
    for u, v in G_dir.edges():
        cu = assign.get(u)
        cv = assign.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            continue
        key = (cu, cv)
        w = G_dir[u][v].get("weight", 1)
        edge_counts[key] = edge_counts.get(key, 0) + w

    for (cu, cv), cnt in edge_counts.items():
        Q.add_edge(cu, cv, weight=cnt)

    return Q


def compute_prune_metrics(G_dir, G_und, assign, deleted_set, fixed_nodes, P):
    """Compute metrics for a given deletion set and its induced clustering."""
    Q = build_quotient_graph_from_assign(G_dir, assign)

    num_clusters = len(set(assign.values()))
    edge_types = Q.number_of_edges()
    weighted_edges_total = sum(d['weight'] for _, _, d in Q.edges(data=True))

    num_self_absorbed = sum(
        1 for u, v in G_dir.edges()
        if assign.get(u) is not None and assign.get(u) == assign.get(v)
    )

    sccs = list(nx.strongly_connected_components(Q))
    sccs_ge3 = [s for s in sccs if len(s) >= 3]
    num_scc_ge3 = len(sccs_ge3)
    largest_scc_size = max((len(s) for s in sccs), default=0)
    cycle_penalty_ge3 = sum(len(s) for s in sccs_ge3)

    fixed_coll = fixed_point_collision(assign, fixed_nodes)
    deleted_set_str = ";".join(str(x) for x in sorted(deleted_set)) if deleted_set else ""

    return {
        "num_deleted": len(deleted_set),
        "num_remaining_relu": P - len(deleted_set),
        "num_clusters": num_clusters,
        "edge_types": edge_types,
        "weighted_edges_total": weighted_edges_total,
        "num_self_absorbed_edges": num_self_absorbed,
        "num_fixed_nodes": sum(1 for n in fixed_nodes if n in assign),
        "fixed_collision": fixed_coll,
        "deleted_set": deleted_set,
        "deleted_set_str": deleted_set_str,
        "num_scc_ge3": num_scc_ge3,
        "largest_scc_size": largest_scc_size,
        "cycle_penalty_ge3": cycle_penalty_ge3,
    }


def get_candidates_for_state(T, base_candidates, edge_diff_set):
    """
    Get candidate ReLUs to add to deletion set T.
    Includes base candidates plus any ReLU that would newly internalize an edge
    (i.e., diff - T becomes singleton when j is added).
    """
    candidates = set(base_candidates) - set(T)
    for diff in edge_diff_set.values():
        remaining = set(diff) - set(T)
        if len(remaining) == 1:
            candidates |= remaining
    return candidates - set(T)


def explore_relu_pruning_hierarchy(G_dir, G_und, fixed_nodes, P,
                                    candidate_mode="switch_edges", max_depth=None,
                                    on_leaf=None):
    """
    Explore the lattice of ReLU deletion sets T in a BFS-style hierarchy.

    For each valid T (no fixed-point collision), build the quotient graph induced
    by projecting away the bits in T and compute metrics.

    Lattice properties guaranteed:
    - Each unique deletion set is evaluated exactly once (visited set).
    - If T causes a collision, all supersets are skipped (monotonicity pruning).
    - Parallel paths to the same T (e.g., {i}→{i,j} and {j}→{j,i}) are deduplicated.

    This analysis is INDEPENDENT of the seed-based graph clustering.
    """
    nodes = list(G_dir.nodes())

    # Build edge diff-sets and identify switch edges
    edge_diff_set = {}
    switch_edges_by_relu = {}

    for u, v in G_dir.edges():
        b_u = symbol_to_bits(u, P)
        b_v = symbol_to_bits(v, P)
        diff = frozenset(i for i in range(P) if b_u[i] != b_v[i])
        edge_diff_set[(u, v)] = diff
        if len(diff) == 1:
            idx = next(iter(diff))
            if idx not in switch_edges_by_relu:
                switch_edges_by_relu[idx] = []
            switch_edges_by_relu[idx].append((u, v))

    if candidate_mode == "switch_edges":
        base_candidates = frozenset(switch_edges_by_relu.keys())
    else:
        base_candidates = frozenset(range(P))

    visited = {frozenset()}
    valid_sets = []     # list of (T, assign, metrics)
    leaf_sets = []      # T (frozenset) values that are leaves
    rejected_sets = []  # list[frozenset] for monotonicity subset checks
    rejected_T_set = set()  # set[frozenset] for O(1) membership

    # DAG tracking
    dag_edges = []      # (T_parent, T_child, j_added) for valid transitions
    rejected_info = []  # (T_parent, T_rejected, j_added, reason)
    metrics_map = {}    # T -> metrics (includes root)

    # Root (T=∅): identity quotient — each node is its own cluster
    _, root_assign, _ = build_prune_clusters(nodes, P, frozenset())
    root_metrics = compute_prune_metrics(G_dir, G_und, root_assign, frozenset(), fixed_nodes, P)
    root_metrics['depth'] = 0
    root_metrics['is_leaf'] = False
    metrics_map[frozenset()] = root_metrics

    level = {frozenset()}
    depth = 0
    max_depth_reached = 0

    while True:
        next_level = set()

        for T in level:
            expandable = False
            candidates = get_candidates_for_state(T, base_candidates, edge_diff_set)

            for j in sorted(candidates):
                T_new = frozenset(T | {j})

                if T_new in visited:
                    # DAG: record alternate path to already-visited valid node,
                    # and mark T as expandable so it is not misclassified as a leaf.
                    if T_new not in rejected_T_set and T_new in metrics_map:
                        dag_edges.append((T, T_new, j))
                        expandable = True
                    continue

                visited.add(T_new)

                # Monotonicity pruning: collision is monotone upward
                if any(R.issubset(T_new) for R in rejected_sets):
                    rejected_sets.append(T_new)
                    rejected_T_set.add(T_new)
                    rejected_info.append((T, T_new, j, "monotone"))
                    continue

                # Build clusters and check fixed-point collision
                _, assign, _ = build_prune_clusters(nodes, P, T_new)

                if fixed_point_collision(assign, fixed_nodes):
                    rejected_sets.append(T_new)
                    rejected_T_set.add(T_new)
                    rejected_info.append((T, T_new, j, "collision"))
                    continue

                # Compute metrics
                metrics = compute_prune_metrics(G_dir, G_und, assign, T_new, fixed_nodes, P)
                metrics['depth'] = depth + 1
                metrics['is_leaf'] = False  # updated after full BFS

                valid_sets.append((T_new, assign, metrics))
                metrics_map[T_new] = metrics
                dag_edges.append((T, T_new, j))
                next_level.add(T_new)
                expandable = True

            # T is a leaf if it had no valid expansions and is non-empty
            if not expandable and T != frozenset():
                leaf_sets.append(T)
                if T in metrics_map:
                    metrics_map[T]['is_leaf'] = True
                if on_leaf is not None:
                    on_leaf(T, dag_edges, metrics_map, valid_sets)

        max_depth_reached = depth

        if max_depth is not None and depth + 1 >= max_depth:
            break

        if not next_level:
            break

        level = next_level
        depth += 1

    # Mark is_leaf in metrics
    leaf_set_frozen = set(leaf_sets)
    for T, assign, metrics in valid_sets:
        metrics['is_leaf'] = T in leaf_set_frozen
    if frozenset() in metrics_map:
        metrics_map[frozenset()]['is_leaf'] = (frozenset() in leaf_set_frozen)

    return {
        'valid_sets': valid_sets,
        'leaf_sets': leaf_sets,
        'rejected_sets': rejected_sets,
        'base_candidates': base_candidates,
        'switch_edges_by_relu': switch_edges_by_relu,
        'edge_diff_set': edge_diff_set,
        'max_depth_reached': max_depth_reached,
        'dag_edges': dag_edges,
        'rejected_info': rejected_info,
        'metrics_map': metrics_map,
    }


def select_minimal_symbol_candidates(valid_sets):
    """Return every distinct deletion set whose num_clusters (number of
    visited symbols after reduction = quotient-graph nodes) attains the global minimum.

    Parameters
    ----------
    valid_sets : list of (T, assign, metrics)
        Result of explore_relu_pruning_hierarchy. The root (T=empty) is not in
        valid_sets (it only exists in metrics_map), so every candidate returned
        here has at least one linearized ReLU.

    Returns
    -------
    (minimal_num_clusters, candidates)
        candidates is a list of (T, metrics), stably sorted by the
        lexicographic order of the deletion set (tuple(sorted(T))), so the
        candidate order (= candidate_id) is invariant across runs for the
        same reduction result. If valid_sets is empty: (None, []).

    NOTE (relation to leaves): num_clusters is monotone non-increasing under
    deletion-set inclusion (projecting out more bits can only merge clusters,
    never split them), so the global minimum is always attained at some leaf.
    A non-leaf valid set may still share the same minimum (extending to a
    child may leave the cluster count unchanged). Therefore is_leaf is NOT a
    selection criterion and is recorded as metadata only.
    No tie-breaking either: every candidate at the minimum num_clusters is
    kept regardless of num_deleted / edge_types etc.
    """
    if not valid_sets:
        return None, []
    minimal = min(m['num_clusters'] for _, _, m in valid_sets)
    # identical deletion sets are only evaluated once thanks to the explorer's
    # visited set, but dedupe defensively (same T -> one candidate)
    seen = {}
    for T, _, m in valid_sets:
        if m['num_clusters'] == minimal and T not in seen:
            seen[T] = m
    candidates = sorted(seen.items(), key=lambda x: tuple(sorted(x[0])))
    return minimal, candidates


def _build_best_parent_map(dag_edges, metrics_map):
    """
    For each child T_new, select the single best parent T_parent from all
    valid DAG edges.  Priority (all metrics refer to the PARENT node):
      1) edge_types small
      2) cycle_penalty_ge3 small
      3) num_clusters small
      4) deleted_set_str lexicographic
    Returns dict[T_child -> (T_parent, j_added)].
    """
    from collections import defaultdict
    children_to_parents = defaultdict(list)
    for T_parent, T_child, j in dag_edges:
        children_to_parents[T_child].append((T_parent, j))

    best_parent = {}
    for T_child, parents in children_to_parents.items():
        def _pk(pj, _mm=metrics_map):
            T_p, _ = pj
            m = _mm.get(T_p, {})
            return (
                m.get('edge_types', 0),
                m.get('cycle_penalty_ge3', 0),
                m.get('num_clusters', 0),
                m.get('deleted_set_str', ''),
            )
        best_parent[T_child] = min(parents, key=_pk)

    return best_parent


def _reconstruct_route(T_leaf, best_parent_map):
    """Follow best parent pointers from T_leaf back to root (frozenset())."""
    route = [T_leaf]
    T = T_leaf
    while T != frozenset():
        if T not in best_parent_map:
            break
        T_parent, _ = best_parent_map[T]
        route.append(T_parent)
        T = T_parent
    route.reverse()
    return route  # [frozenset(), T_1, ..., T_leaf]


def save_relu_prune_hierarchy_csv(hierarchy_results, M, P, option, save_leaves_only=False):
    """Save valid (and rejected) deletion sets with their metrics to CSV."""
    import pandas as pd

    valid_sets = hierarchy_results['valid_sets']
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])
    dag_edges = hierarchy_results.get('dag_edges', [])
    rejected_info = hierarchy_results.get('rejected_info', [])
    metrics_map = hierarchy_results.get('metrics_map', {})

    best_parent_map = _build_best_parent_map(dag_edges, metrics_map)

    # same minimum as the retraining selection (min num_clusters over valid_sets)
    minimal_num_clusters, _ = select_minimal_symbol_candidates(valid_sets)

    rows = []

    # --- valid node rows ---
    for T, assign, metrics in valid_sets:
        is_leaf = T in leaf_set_frozen
        if save_leaves_only and not is_leaf:
            continue

        if T in best_parent_map:
            T_parent, j_added = best_parent_map[T]
            parent_m = metrics_map.get(T_parent, {})
            parent_set_str = parent_m.get('deleted_set_str', '')
            added_relu = j_added
        else:
            parent_set_str = ''
            added_relu = ''

        rows.append({
            'M': M, 'P': P, 'option': option,
            'depth': metrics.get('depth', 0),
            'num_deleted': metrics['num_deleted'],
            'num_remaining_relu': metrics['num_remaining_relu'],
            'deleted_set_str': metrics['deleted_set_str'],
            'is_leaf': is_leaf,
            'is_rejected': False,
            'reject_reason': '',
            'parent_set_str': parent_set_str,
            'added_relu': added_relu,
            'num_clusters': metrics['num_clusters'],
            'edge_types': metrics['edge_types'],
            'weighted_edges_total': metrics['weighted_edges_total'],
            'num_self_absorbed_edges': metrics['num_self_absorbed_edges'],
            'num_fixed_nodes': metrics['num_fixed_nodes'],
            'fixed_collision': metrics['fixed_collision'],
            'num_scc_ge3': metrics['num_scc_ge3'],
            'largest_scc_size': metrics['largest_scc_size'],
            'cycle_penalty_ge3': metrics['cycle_penalty_ge3'],
            'is_minimal_num_clusters': (
                metrics['num_clusters'] == minimal_num_clusters),
        })

    # --- rejected node rows (only if not save_leaves_only) ---
    if not save_leaves_only and rejected_info:
        seen_rejected = {}
        for T_parent, T_rej, j, reason in rejected_info:
            if T_rej not in seen_rejected:
                seen_rejected[T_rej] = (T_parent, j, reason)

        for T_rej, (T_parent, j, reason) in seen_rejected.items():
            dstr = ";".join(str(x) for x in sorted(T_rej)) if T_rej else ""
            parent_m = metrics_map.get(T_parent, {})
            rows.append({
                'M': M, 'P': P, 'option': option,
                'depth': len(T_rej),
                'num_deleted': len(T_rej),
                'num_remaining_relu': P - len(T_rej),
                'deleted_set_str': dstr,
                'is_leaf': False,
                'is_rejected': True,
                'reject_reason': reason,
                'parent_set_str': parent_m.get('deleted_set_str', ''),
                'added_relu': j,
                'num_clusters': None,
                'edge_types': None,
                'weighted_edges_total': None,
                'num_self_absorbed_edges': None,
                'num_fixed_nodes': None,
                'fixed_collision': True,
                'num_scc_ge3': None,
                'largest_scc_size': None,
                'cycle_penalty_ge3': None,
                'is_minimal_num_clusters': None,
            })

    if not rows:
        print("[ReLU Prune] No results to save.")
        return None

    df = pd.DataFrame(rows)
    resdir = f"results/relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(resdir, exist_ok=True)
    csv_path = f"{resdir}/relu_prune_hierarchy_m{M}_p{P}{option}.csv"
    df.to_csv(csv_path, index=False)
    n_valid = sum(1 for r in rows if not r['is_rejected'])
    n_rej = sum(1 for r in rows if r['is_rejected'])
    print(f"[ReLU Prune] CSV saved: {csv_path} ({n_valid} valid, {n_rej} rejected rows)")
    return csv_path


def save_top_relu_prune_quotient_graphs(hierarchy_results, G_dir, M, P, option, top_k=10,
                                        real_fp_nodes=None, label_fmt="decimal",
                                        orbit=None, symbol_seq=None, N=3,
                                        real_fp_coords=None):
    """Save quotient graph figures for the top leaf deletion sets.

    If orbit (T,M) and symbol_seq (list of decimal symbol strings, length T) are
    provided, a trajectory subplot colored by cluster assignment is appended.
    real_fp_coords: list of (N,) arrays — real fixed-point readout coordinates to overlay.
    """
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])
    valid_sets = hierarchy_results['valid_sets']

    leaf_entries = [
        (T, assign, metrics)
        for T, assign, metrics in valid_sets
        if T in leaf_set_frozen
    ]

    if not leaf_entries:
        print("[ReLU Prune] No leaf entries to plot.")
        return

    # Sort: most deleted first, then fewest edge types, then smallest SCC penalty, then fewest clusters
    leaf_entries_sorted = sorted(leaf_entries, key=lambda x: (
        -x[2]['num_deleted'],
        x[2]['edge_types'],
        x[2]['cycle_penalty_ge3'],
        x[2]['num_clusters'],
    ))

    top_entries = leaf_entries_sorted[:top_k]
    figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(figdir, exist_ok=True)

    _real_fp = real_fp_nodes or set()

    for T, assign, metrics in top_entries:
        Q = build_quotient_graph_from_assign(G_dir, assign)

        # reverse-map: cluster key → original symbols, and cluster key → projected bits
        cluster_to_originals = {}
        cluster_to_proj_bits = {}
        for orig, ck in assign.items():
            cluster_to_originals.setdefault(ck, []).append(orig)
        for n in Q.nodes():
            origs = cluster_to_originals.get(n, [])
            if origs:
                cluster_to_proj_bits[n] = project_bits(symbol_to_bits(origs[0], P), T)

        deleted_set_str = metrics['deleted_set_str']
        fname_str = deleted_set_str.replace(";", "-") if deleted_set_str else "empty"

        has_traj = orbit is not None and symbol_seq is not None
        ncols_fig = 2 if has_traj else 1
        fig = plt.figure(figsize=(8 * ncols_fig, 6))
        ax = fig.add_subplot(1, ncols_fig, 1)

        pos_q = nx.spring_layout(Q, seed=42)

        edge_weights_q = [Q[u][v]['weight'] for u, v in Q.edges()]
        if edge_weights_q:
            wmin_q = min(edge_weights_q)
            wmax_q = max(edge_weights_q)
            if wmax_q > wmin_q:
                widths_q = [1.0 + 5.0 * (w - wmin_q) / (wmax_q - wmin_q) for w in edge_weights_q]
            else:
                widths_q = [3.0] * len(edge_weights_q)
        else:
            widths_q = []

        n_clusters = metrics['num_clusters']
        cmap_traj = plt.colormaps.get_cmap('tab10').resampled(max(n_clusters, 1))

        qnode_labels = {}
        qnode_colors = []
        qnode_ecs = []
        qnode_lws = []
        for n in Q.nodes():
            origs = cluster_to_originals.get(n, [])
            proj = cluster_to_proj_bits.get(n)
            qnode_labels[n] = _cluster_label(proj, label_fmt) if proj is not None else str(n)
            qnode_colors.append(cmap_traj(n % 10))
            if any(s in _real_fp for s in origs):
                qnode_ecs.append('black')
                qnode_lws.append(3.5)
            else:
                qnode_ecs.append('none')
                qnode_lws.append(0.0)

        nx.draw_networkx_nodes(Q, pos_q, ax=ax, node_size=800, node_color=qnode_colors,
                               edgecolors=qnode_ecs, linewidths=qnode_lws)
        nx.draw_networkx_labels(Q, pos_q, labels=qnode_labels, ax=ax, font_weight='bold')
        if Q.edges():
            nx.draw_networkx_edges(Q, pos_q, ax=ax, width=widths_q, edge_color='black',
                                   arrowsize=15, connectionstyle='arc3,rad=0.1')
            edge_switch = _quotient_edge_switch_labels(Q, cluster_to_proj_bits, T, P)
            nx.draw_networkx_edge_labels(Q, pos_q, edge_labels=edge_switch, ax=ax,
                                         font_size=7, label_pos=0.3)

        title = (
            f"deleted={{{deleted_set_str}}}, remaining ReLU={metrics['num_remaining_relu']}, "
            f"clusters={metrics['num_clusters']}, edge types={metrics['edge_types']}, "
            f"SCC>=3 penalty={metrics['cycle_penalty_ge3']}"
        )
        ax.set_title(title, fontsize=8)
        ax.axis('off')

        # ── trajectory subplot colored by cluster ──────────────────────────
        if has_traj:
            traj_colors = [
                cmap_traj(assign.get(sym, 0) % 10)
                for sym in symbol_seq
            ]
            ax_t = fig.add_subplot(1, 2, 2, projection='3d')
            ax_t.scatter(orbit[:, 0], orbit[:, 1], orbit[:, 2],
                         c=traj_colors, s=3, alpha=0.3, depthshade=False)
            # overlay real fixed points on top of trajectory
            if real_fp_coords:
                for coords in real_fp_coords:
                    ax_t.scatter(coords[0], coords[1], coords[2],
                                 color='red', s=150, edgecolors='black',
                                 linewidths=1.5, depthshade=False, zorder=10)
            ax_t.set_title("Trajectory by cluster", fontsize=8)
            ax_t.axis('off')

        plt.tight_layout()
        save_path = f"{figdir}/relu_prune_quotient_m{M}_p{P}_deleted-{fname_str}{option}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"[ReLU Prune] quotient graph saved: {save_path}")


def save_relu_prune_hierarchy_dag(hierarchy_results, M, P, option):
    """
    Save the full exploration DAG as a hierarchical PNG.
    Nodes are layered by deletion-set size (depth).
    Valid leaves have a thick border; rejected nodes are shown in red.
    """
    from matplotlib.patches import Patch as _Patch

    valid_sets = hierarchy_results['valid_sets']
    dag_edges = hierarchy_results.get('dag_edges', [])
    rejected_info = hierarchy_results.get('rejected_info', [])
    metrics_map = hierarchy_results.get('metrics_map', {})
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])

    valid_T_set = {T for T, _, _ in valid_sets} | {frozenset()}

    # Collect rejected nodes (one entry per unique T_rej)
    rejected_T_shown = {}
    for T_parent, T_rej, j, reason in rejected_info:
        if T_rej not in rejected_T_shown:
            rejected_T_shown[T_rej] = (T_parent, j, reason)

    all_nodes = valid_T_set | set(rejected_T_shown.keys())

    # Hierarchical layout: depth = len(T)
    by_depth = {}
    for n in all_nodes:
        d = len(n)
        by_depth.setdefault(d, []).append(n)
    for d in by_depth:
        by_depth[d].sort(key=lambda T: tuple(sorted(T)))

    pos = {}
    for d, ns in by_depth.items():
        n_at_d = len(ns)
        for i, n in enumerate(ns):
            pos[n] = ((i - (n_at_d - 1) / 2.0), -float(d))

    # Build DiGraph
    DAG = nx.DiGraph()
    DAG.add_nodes_from(all_nodes)

    edge_labels = {}
    for T_parent, T_child, j in dag_edges:
        if T_parent in all_nodes and T_child in all_nodes:
            DAG.add_edge(T_parent, T_child, etype='valid')
            key = (T_parent, T_child)
            edge_labels[key] = str(j) if key not in edge_labels else edge_labels[key] + f",{j}"

    rej_edge_labels = {}
    for T_rej, (T_parent, j, _) in rejected_T_shown.items():
        if T_parent in all_nodes:
            if not DAG.has_edge(T_parent, T_rej):
                DAG.add_edge(T_parent, T_rej, etype='rejected')
            rej_edge_labels[(T_parent, T_rej)] = str(j)

    # Node appearance
    node_list = list(all_nodes)
    node_colors, node_lws, node_ecs = [], [], []
    for n in node_list:
        if n in rejected_T_shown:
            node_colors.append('#ffcccc'); node_lws.append(1.5); node_ecs.append('#cc0000')
        elif n == frozenset():
            node_colors.append('#fffacd'); node_lws.append(1.5); node_ecs.append('#888800')
        elif n in leaf_set_frozen:
            node_colors.append('#cce5ff'); node_lws.append(3.5); node_ecs.append('#003399')
        else:
            node_colors.append('#d4edda'); node_lws.append(1.5); node_ecs.append('#155724')

    def _make_label(T):
        m = metrics_map.get(T)
        if T == frozenset():
            if m:
                return f"root\ncls={m.get('num_clusters','?')}\net={m.get('edge_types','?')}"
            return "root"
        if m:
            dstr = m.get('deleted_set_str') or '∅'
            return f"{dstr}\ndel={m['num_deleted']}\ncls={m['num_clusters']}\net={m['edge_types']}"
        dstr = ";".join(str(x) for x in sorted(T)) if T else '∅'
        return f"{dstr}\n[REJ]"

    labels = {n: _make_label(n) for n in node_list}

    # Figure size
    max_d = max(by_depth.keys()) if by_depth else 0
    max_w = max(len(v) for v in by_depth.values()) if by_depth else 1
    fig_h = min(max(6.0, (max_d + 1) * 2.5), 40.0)
    fig_w = min(max(10.0, max_w * 3.0), 60.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    nx.draw_networkx_nodes(DAG, pos, ax=ax, nodelist=node_list,
                           node_color=node_colors, node_size=1400,
                           edgecolors=node_ecs, linewidths=node_lws)
    nx.draw_networkx_labels(DAG, pos, labels=labels, ax=ax, font_size=6)

    valid_edges = [(u, v) for u, v, d in DAG.edges(data=True) if d.get('etype') == 'valid']
    rej_edges = [(u, v) for u, v, d in DAG.edges(data=True) if d.get('etype') == 'rejected']
    if valid_edges:
        nx.draw_networkx_edges(DAG, pos, edgelist=valid_edges, ax=ax,
                               edge_color='#333333', arrowsize=14, width=1.5,
                               connectionstyle='arc3,rad=0.05',
                               min_source_margin=25, min_target_margin=25)
    if rej_edges:
        nx.draw_networkx_edges(DAG, pos, edgelist=rej_edges, ax=ax,
                               edge_color='#cc0000', arrowsize=14, width=1.5,
                               style='dashed', connectionstyle='arc3,rad=0.05',
                               min_source_margin=25, min_target_margin=25)

    all_el = {**edge_labels, **rej_edge_labels}
    if all_el:
        nx.draw_networkx_edge_labels(DAG, pos, edge_labels=all_el, ax=ax,
                                     font_size=6, label_pos=0.4)

    legend_elements = [
        _Patch(facecolor='#fffacd', edgecolor='#888800', label='Root (T=∅)'),
        _Patch(facecolor='#d4edda', edgecolor='#155724', label='Valid (non-leaf)'),
        _Patch(facecolor='#cce5ff', edgecolor='#003399', linewidth=3, label='Valid leaf'),
        _Patch(facecolor='#ffcccc', edgecolor='#cc0000', label='Rejected'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(
        f"ReLU Pruning Hierarchy DAG  (M={M}, P={P})\n"
        f"valid={len(valid_T_set)}  leaves={len(leaf_set_frozen)}  "
        f"rejected={len(rejected_T_shown)}",
        fontsize=10
    )
    ax.axis('off')

    figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(figdir, exist_ok=True)
    save_path = f"{figdir}/relu_prune_hierarchy_dag_m{M}_p{P}{option}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"[ReLU Prune] DAG saved: {save_path}")


def save_relu_prune_route_plots(hierarchy_results, G_dir, M, P, option, num_plots=10,
                                real_fp_nodes=None, label_fmt="decimal"):
    """
    For each top leaf, reconstruct the best-parent route and save a multi-panel
    figure showing the quotient graph at each stage of the contraction.
    """
    valid_sets = hierarchy_results['valid_sets']
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])
    dag_edges = hierarchy_results.get('dag_edges', [])
    metrics_map = hierarchy_results.get('metrics_map', {})
    nodes = list(G_dir.nodes())

    leaf_entries = [
        (T, assign, metrics)
        for T, assign, metrics in valid_sets
        if T in leaf_set_frozen
    ]
    if not leaf_entries:
        print("[ReLU Prune] No leaf entries for route plots.")
        return

    leaf_entries_sorted = sorted(leaf_entries, key=lambda x: (
        -x[2]['num_deleted'], x[2]['edge_types'],
        x[2]['cycle_penalty_ge3'], x[2]['num_clusters'],
    ))
    top_leaves = leaf_entries_sorted[:num_plots]

    _real_fp = real_fp_nodes or set()

    best_parent_map = _build_best_parent_map(dag_edges, metrics_map)
    valid_lookup = {T: (a, m) for T, a, m in valid_sets}

    figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(figdir, exist_ok=True)

    for T_leaf, _, leaf_metrics in top_leaves:
        route = _reconstruct_route(T_leaf, best_parent_map)
        n_steps = len(route)

        # Pre-pass: build assign for each route step, collect all projected bit
        # patterns across the whole route for a globally consistent color map.
        route_assigns = {}
        for T in route:
            if T == frozenset():
                _, a, _ = build_prune_clusters(nodes, P, frozenset())
                route_assigns[T] = a
            elif T in valid_lookup:
                a, _ = valid_lookup[T]
                route_assigns[T] = a

        all_proj_patterns = set()
        for T, a in route_assigns.items():
            for orig in a:
                all_proj_patterns.add(project_bits(symbol_to_bits(orig, P), T))
        all_proj_sorted = sorted(all_proj_patterns)
        n_global = len(all_proj_sorted)
        cmap_route = plt.colormaps.get_cmap('tab10').resampled(max(n_global, 1))
        pattern_to_color = {pat: cmap_route(i % 10) for i, pat in enumerate(all_proj_sorted)}

        fig, axes = plt.subplots(1, n_steps, figsize=(max(5 * n_steps, 8), 5.5))
        if n_steps == 1:
            axes = [axes]

        for ax, T in zip(axes, route):
            if T not in route_assigns:
                ax.axis('off')
                continue

            assign = route_assigns[T]
            if T == frozenset():
                m = metrics_map.get(frozenset(), {})
            else:
                _, m = valid_lookup[T]

            Q = build_quotient_graph_from_assign(G_dir, assign)
            pos_q = nx.spring_layout(Q, seed=42)

            ew = [Q[u][v]['weight'] for u, v in Q.edges()]
            if ew:
                wmin_q, wmax_q = min(ew), max(ew)
                widths_q = (
                    [1.0 + 4.0 * (w - wmin_q) / (wmax_q - wmin_q) for w in ew]
                    if wmax_q > wmin_q else [2.5] * len(ew)
                )
            else:
                widths_q = []

            cluster_to_originals = {}
            cluster_to_proj_bits = {}
            for orig, ck in assign.items():
                cluster_to_originals.setdefault(ck, []).append(orig)
            for n in Q.nodes():
                origs = cluster_to_originals.get(n, [])
                if origs:
                    cluster_to_proj_bits[n] = project_bits(symbol_to_bits(origs[0], P), T)

            qnode_labels = {}
            qnode_colors = []
            qnode_ecs = []
            qnode_lws = []
            for n in Q.nodes():
                origs = cluster_to_originals.get(n, [])
                proj = cluster_to_proj_bits.get(n)
                qnode_labels[n] = _cluster_label(proj, label_fmt) if proj is not None else str(n)
                qnode_colors.append(pattern_to_color.get(proj, cmap_route(0)))
                if any(s in _real_fp for s in origs):
                    qnode_ecs.append('black')
                    qnode_lws.append(3.5)
                else:
                    qnode_ecs.append('none')
                    qnode_lws.append(0.0)

            nx.draw_networkx_nodes(Q, pos_q, ax=ax, node_size=500,
                                   node_color=qnode_colors, edgecolors=qnode_ecs, linewidths=qnode_lws)
            nx.draw_networkx_labels(Q, pos_q, labels=qnode_labels, ax=ax, font_size=7, font_weight='bold')
            if Q.edges():
                nx.draw_networkx_edges(Q, pos_q, ax=ax, width=widths_q,
                                       edge_color='black', arrowsize=12,
                                       connectionstyle='arc3,rad=0.1')
                edge_switch = _quotient_edge_switch_labels(Q, cluster_to_proj_bits, T, P)
                nx.draw_networkx_edge_labels(Q, pos_q, edge_labels=edge_switch, ax=ax,
                                             font_size=6, label_pos=0.3)

            dstr = m.get('deleted_set_str', '') or '∅'
            ax.set_title(
                f"T={{{dstr}}}\n"
                f"rem={m.get('num_remaining_relu', P)}, cls={m.get('num_clusters','?')}\n"
                f"et={m.get('edge_types','?')}, cyc={m.get('cycle_penalty_ge3','?')}",
                fontsize=7
            )
            ax.axis('off')

        leaf_dstr = leaf_metrics['deleted_set_str']
        fname_str = leaf_dstr.replace(";", "-") if leaf_dstr else "empty"
        fig.suptitle(
            f"ReLU Pruning Route → leaf T={{{leaf_dstr}}}  (M={M}, P={P})  "
            f"edge_types={leaf_metrics['edge_types']}, "
            f"cycle_penalty={leaf_metrics['cycle_penalty_ge3']}",
            fontsize=9
        )
        plt.tight_layout()
        save_path = f"{figdir}/relu_prune_route_m{M}_p{P}_leaf-{fname_str}{option}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"[ReLU Prune] route plot saved: {save_path}")


def save_relu_prune_route_subplots(hierarchy_results, G_dir, M, P, option, seed=0,
                                   num_routes=10, real_fp_nodes=None, label_fmt="binary"):
    """
    For each top-leaf route (root→leaf), save one figure with subplots showing the
    quotient graph at every step.

    Edge color coding in non-final panels:
      red    : will disappear at next step (clusters differ ONLY on the next-pruned unit)
      orange : may change (differ on next-pruned unit + others — edge won't vanish but
               the switch label will shift)
      gray   : unaffected by next pruning
    """
    from matplotlib.patches import Patch

    valid_sets      = hierarchy_results['valid_sets']
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])
    dag_edges       = hierarchy_results.get('dag_edges', [])
    metrics_map     = hierarchy_results.get('metrics_map', {})
    nodes           = list(G_dir.nodes())

    leaf_entries = [
        (T, assign, metrics)
        for T, assign, metrics in valid_sets
        if T in leaf_set_frozen
    ]
    if not leaf_entries:
        print("[ReLU Prune] No leaf entries for route subplots.")
        return

    leaf_entries_sorted = sorted(leaf_entries, key=lambda x: (
        -x[2]['num_deleted'], x[2]['edge_types'],
        x[2]['cycle_penalty_ge3'], x[2]['num_clusters'],
    ))
    top_leaves = leaf_entries_sorted[:num_routes]

    _real_fp     = real_fp_nodes or set()
    best_parent  = _build_best_parent_map(dag_edges, metrics_map)
    valid_lookup = {T: (a, m) for T, a, m in valid_sets}

    figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(figdir, exist_ok=True)

    for T_leaf, _, leaf_metrics in top_leaves:
        route   = _reconstruct_route(T_leaf, best_parent)
        n_steps = len(route)

        # Build assigns for every step in the route
        route_assigns = {}
        for T in route:
            if T == frozenset():
                _, a, _ = build_prune_clusters(nodes, P, frozenset())
                route_assigns[T] = a
            elif T in valid_lookup:
                a, _ = valid_lookup[T]
                route_assigns[T] = a

        # Global color map: one consistent color per projected bit pattern
        all_proj = set()
        for T, a in route_assigns.items():
            for orig in a:
                all_proj.add(project_bits(symbol_to_bits(orig, P), T))
        all_proj_sorted   = sorted(all_proj)
        n_global          = len(all_proj_sorted)
        cmap_route        = plt.colormaps.get_cmap('tab10').resampled(max(n_global, 1))
        pattern_to_color  = {pat: cmap_route(i % 10) for i, pat in enumerate(all_proj_sorted)}

        fig, axes = plt.subplots(1, n_steps, figsize=(max(5 * n_steps, 8), 6.8))
        if n_steps == 1:
            axes = [axes]

        for step_idx, (ax, T) in enumerate(zip(axes, route)):
            if T not in route_assigns:
                ax.axis('off')
                continue

            assign = route_assigns[T]
            if T == frozenset():
                m = metrics_map.get(frozenset(), {})
            elif T in valid_lookup:
                _, m = valid_lookup[T]
            else:
                m = {}

            Q     = build_quotient_graph_from_assign(G_dir, assign)
            pos_q = nx.spring_layout(Q, seed=42)

            # Map each cluster node → list of original symbols and projected bits
            cluster_to_originals = {}
            for orig, ck in assign.items():
                cluster_to_originals.setdefault(ck, []).append(orig)
            cluster_to_proj_bits = {}
            for n in Q.nodes():
                origs = cluster_to_originals.get(n, [])
                if origs:
                    cluster_to_proj_bits[n] = project_bits(symbol_to_bits(origs[0], P), T)

            # Determine which ReLU unit is removed in the NEXT step
            is_last = (step_idx == n_steps - 1)
            next_j  = None
            if not is_last:
                T_next = route[step_idx + 1]
                diff   = T_next - T
                if len(diff) == 1:
                    next_j = next(iter(diff))

            remaining = sorted(set(range(P)) - T)

            # Edge widths (weight-proportional) and color classification
            ew = [Q[u][v]['weight'] for u, v in Q.edges()]
            if ew:
                wmin_q, wmax_q = min(ew), max(ew)
                base_widths = (
                    [1.0 + 3.0 * (w - wmin_q) / (wmax_q - wmin_q) for w in ew]
                    if wmax_q > wmin_q else [2.0] * len(ew)
                )
            else:
                base_widths = []

            edge_colors = []
            edge_widths = []
            for (u, v), bw in zip(Q.edges(), base_widths):
                bu = cluster_to_proj_bits.get(u)
                bv = cluster_to_proj_bits.get(v)
                if is_last or next_j is None or bu is None or bv is None:
                    edge_colors.append('#555555')
                    edge_widths.append(bw)
                    continue
                differing = {remaining[i] for i, (a, b) in enumerate(zip(bu, bv)) if a != b}
                if differing == {next_j}:
                    # Clusters differ ONLY on the next-pruned bit → edge will vanish
                    edge_colors.append('#e63946')
                    edge_widths.append(bw + 0.8)
                elif next_j in differing:
                    # Next-pruned bit is one of several differing bits → edge persists
                    edge_colors.append('#f4a261')
                    edge_widths.append(bw + 0.3)
                else:
                    edge_colors.append('#555555')
                    edge_widths.append(bw)

            # Node drawing
            qnode_labels = {}
            qnode_colors = []
            qnode_ecs    = []
            qnode_lws    = []
            for n in Q.nodes():
                origs = cluster_to_originals.get(n, [])
                proj  = cluster_to_proj_bits.get(n)
                qnode_labels[n] = _cluster_label(proj, label_fmt) if proj is not None else str(n)
                qnode_colors.append(pattern_to_color.get(proj, cmap_route(0)))
                if any(s in _real_fp for s in origs):
                    qnode_ecs.append('black')
                    qnode_lws.append(3.0)
                else:
                    qnode_ecs.append('none')
                    qnode_lws.append(0.0)

            nx.draw_networkx_nodes(Q, pos_q, ax=ax, node_size=500,
                                   node_color=qnode_colors,
                                   edgecolors=qnode_ecs, linewidths=qnode_lws)
            nx.draw_networkx_labels(Q, pos_q, labels=qnode_labels, ax=ax,
                                    font_size=7, font_weight='bold')
            if Q.edges():
                nx.draw_networkx_edges(Q, pos_q, ax=ax, width=edge_widths,
                                       edge_color=edge_colors, arrowsize=12,
                                       connectionstyle='arc3,rad=0.1')
                edge_switch = _quotient_edge_switch_labels(Q, cluster_to_proj_bits, T, P)
                nx.draw_networkx_edge_labels(Q, pos_q, edge_labels=edge_switch, ax=ax,
                                             font_size=6, label_pos=0.3)

            # Per-panel subtitle
            dstr = ','.join(str(x) for x in sorted(T)) if T else '∅'
            if T == frozenset():
                step_label = "root  (no pruning)"
            else:
                prev_T  = route[step_idx - 1]
                added   = sorted(T - prev_T)
                step_label = f"+ prune ReLU {added[0]}" if added else f"del={{{dstr}}}"

            next_info = (f"\n→ next: prune ReLU {next_j}" if (not is_last and next_j is not None) else "")

            ax.set_title(
                f"{step_label}\n"
                f"del={{{dstr}}}  cls={m.get('num_clusters', '?')}\n"
                f"et={m.get('edge_types', '?')}  cyc={m.get('cycle_penalty_ge3', '?')}"
                f"{next_info}",
                fontsize=7
            )
            ax.axis('off')

        # Legend
        legend_elements = [
            Patch(facecolor='#555555', label='unaffected'),
            Patch(facecolor='#e63946', label='will disappear (only differs on next pruned unit)'),
            Patch(facecolor='#f4a261', label='may shift (also differs on other units)'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=7, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

        leaf_dstr = leaf_metrics['deleted_set_str']
        fname_str = leaf_dstr.replace(";", "-") if leaf_dstr else "empty"
        fig.suptitle(
            f"ReLU Pruning Route → leaf T={{{leaf_dstr}}}  (M={M}, P={P}, seed={seed})\n"
            f"edge_types={leaf_metrics['edge_types']}  "
            f"cycle_penalty={leaf_metrics['cycle_penalty_ge3']}  "
            f"clusters={leaf_metrics['num_clusters']}",
            fontsize=9
        )
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        save_path = f"{figdir}/relu_prune_route_subplot_leaf-{fname_str}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"[ReLU Prune] route subplot saved: {save_path}")


def save_relu_prune_min_cluster_comparison(hierarchy_results, G_dir, M, P, option,
                                           real_fp_nodes=None, label_fmt="decimal"):
    """Save a subplot comparison of all leaf deletion sets that achieve the minimum cluster count.

    Node colors are consistent across subplots: the same projected bit pattern always
    receives the same color (global tab10 assignment over the union of all patterns).
    """
    leaf_set_frozen = set(hierarchy_results['leaf_sets'])
    valid_sets = hierarchy_results['valid_sets']

    leaf_entries = [
        (T, assign, metrics)
        for T, assign, metrics in valid_sets
        if T in leaf_set_frozen
    ]
    if not leaf_entries:
        print("[ReLU Prune] No leaf entries for min-cluster comparison.")
        return

    min_clusters = min(m['num_clusters'] for _, _, m in leaf_entries)
    candidates = [e for e in leaf_entries if e[2]['num_clusters'] == min_clusters]

    candidates.sort(key=lambda x: (
        x[2]['edge_types'],
        x[2]['cycle_penalty_ge3'],
        x[2]['deleted_set_str'],
    ))

    n = len(candidates)
    if n == 0:
        return

    # Pre-pass: collect all projected bit patterns across every candidate to build
    # a global color map (same pattern → same color in every subplot).
    all_proj_patterns = set()
    for T, assign, _ in candidates:
        for orig in assign:
            all_proj_patterns.add(project_bits(symbol_to_bits(orig, P), T))
    all_proj_patterns_sorted = sorted(all_proj_patterns)
    n_global = len(all_proj_patterns_sorted)
    cmap_global = plt.colormaps.get_cmap('tab10').resampled(max(n_global, 1))
    pattern_to_color = {pat: cmap_global(i % 10) for i, pat in enumerate(all_proj_patterns_sorted)}

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows + 0.8),
                             squeeze=False)

    _real_fp = real_fp_nodes or set()

    for idx, (T, assign, metrics) in enumerate(candidates):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        Q = build_quotient_graph_from_assign(G_dir, assign)
        pos_q = nx.spring_layout(Q, seed=42)

        cluster_to_originals = {}
        cluster_to_proj_bits = {}
        for orig, ck in assign.items():
            cluster_to_originals.setdefault(ck, []).append(orig)
        for n_node in Q.nodes():
            origs = cluster_to_originals.get(n_node, [])
            if origs:
                cluster_to_proj_bits[n_node] = project_bits(symbol_to_bits(origs[0], P), T)

        ew = [Q[u][v]['weight'] for u, v in Q.edges()]
        if ew:
            wmin, wmax = min(ew), max(ew)
            widths_q = (
                [1.0 + 5.0 * (w - wmin) / (wmax - wmin) for w in ew]
                if wmax > wmin else [3.0] * len(ew)
            )
        else:
            widths_q = []

        qnode_labels = {}
        qnode_colors = []
        qnode_ecs = []
        qnode_lws = []
        for n_node in Q.nodes():
            origs = cluster_to_originals.get(n_node, [])
            proj = cluster_to_proj_bits.get(n_node)
            qnode_labels[n_node] = _cluster_label(proj, label_fmt) if proj is not None else str(n_node)
            qnode_colors.append(pattern_to_color.get(proj, cmap_global(0)))
            if any(s in _real_fp for s in origs):
                qnode_ecs.append('black')
                qnode_lws.append(3.5)
            else:
                qnode_ecs.append('none')
                qnode_lws.append(0.0)

        nx.draw_networkx_nodes(Q, pos_q, ax=ax, node_size=800, node_color=qnode_colors,
                               edgecolors=qnode_ecs, linewidths=qnode_lws)
        nx.draw_networkx_labels(Q, pos_q, labels=qnode_labels, ax=ax, font_weight='bold')
        if Q.edges():
            nx.draw_networkx_edges(Q, pos_q, ax=ax, width=widths_q, edge_color='black',
                                   arrowsize=15, connectionstyle='arc3,rad=0.1')
            edge_switch = _quotient_edge_switch_labels(Q, cluster_to_proj_bits, T, P)
            nx.draw_networkx_edge_labels(Q, pos_q, edge_labels=edge_switch, ax=ax,
                                         font_size=7, label_pos=0.3)

        dstr = metrics['deleted_set_str'] or '∅'
        ax.set_title(
            f"T={{{dstr}}}\n"
            f"et={metrics['edge_types']}, cyc={metrics['cycle_penalty_ge3']}",
            fontsize=8
        )
        ax.axis('off')

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    fig.suptitle(
        f"Min-cluster leaves  (clusters={min_clusters}, count={n})  M={M}, P={P}",
        fontsize=10
    )
    plt.tight_layout()

    figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(figdir, exist_ok=True)
    save_path = f"{figdir}/relu_prune_min_cluster_comparison_m{M}_p{P}{option}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"[ReLU Prune] min-cluster comparison saved: {save_path}")


# ============================================================
# CONFIG - fixed-point visualization options
# ============================================================
SHOW_VIRTUAL_FPS = False  # True: also show virtual fixed points
SHOW_FP_LABELS = False    # True: label fixed points with indices
FP_MARKER_SIZE = 100      # fixed-point marker size

# ============================================================
# CONFIG - Symbol clustering options (graph contraction)
# ============================================================
# Choose one: "milp" (exact, requires PuLP) or "heuristic" (fast)
CLUSTER_SOLVER = "heuristic"
# CLUSTER_SOLVER = "milp"
# MILP settings
MILP_TIME_LIMIT_SEC = 300  # None for no limit
# Heuristic settings
HEURISTIC_MAX_ITERS = 20000
HEURISTIC_ANNEAL = True
HEURISTIC_SEED = 42
# Compactness penalty (only for heuristic)
COMPACTNESS_PENALTY_LAMBDA = 0.5  # Weight for maximum mean distance to seed penalty
# ============================================================
# ============================================================

# ============================================================
# CONFIG - ReLU pruning-induced quotient hierarchy
# ============================================================
RUN_RELU_PRUNE_HIERARCHY = True
RELU_PRUNE_CANDIDATE_MODE = "switch_edges"  # "switch_edges" or "all"
RELU_PRUNE_MAX_DEPTH = None  # None means no explicit limit
RELU_PRUNE_SAVE_ALL_VALID = True
RELU_PRUNE_SAVE_LEAVES_ONLY = False
RELU_PRUNE_NUM_ROUTE_PLOTS    = 10  # number of top-leaf route diagrams to save; set 0 to skip
RELU_PRUNE_NUM_ROUTE_SUBPLOTS = 10  # number of root→leaf subplot figures to save; set 0 to skip
# SYMBOL_LABEL_FORMAT = "decimal"  # "decimal" or "binary" — applies to all graph figures
SYMBOL_LABEL_FORMAT = "binary"  # "decimal" or "binary" — applies to all graph figures
TF_WARMUP_STEPS = 500  # teacher-forcing steps used to initialize RNN hidden state (0 = use first obs only)
# ============================================================


# def analyze_fixed_points(model, unique_symbols_list):
#     """
#     Analyzes the fixed points for a given list of visited symbolic regions.
#     """
#     print("\nAnalyzing fixed points for visited symbolic regions...")
#     M, P = model.M, model.P
#     A = np.diag(model.A.detach().cpu().numpy())
#     W = model.W.detach().cpu().numpy()
#     h = model.h.detach().cpu().numpy()
#
#
#     results = {}
#
#     # loop only over the unique symbols actually visited
#     for symbol_vec in unique_symbols_list:
#         symbol_str = ''.join(map(str, symbol_vec))
#         # pad the linear-unit part (M-P entries) with '1'
#         # the trailing piecewise-linear units (P) take symbol_vec as-is
#         D_s = np.diag(np.pad(symbol_vec, (M - P, 0), 'constant', constant_values=1))
#         W_k = A + W @ D_s
#         # print(D_s)
#
#         try:
#             I = np.identity(M)
#             fixed_point = np.linalg.inv(I - W_k) @ h
#         except np.linalg.LinAlgError:
#             continue
#
#         pre_activations = fixed_point[-P:]
#         signs = (pre_activations > 0).astype(int)
#         is_real = np.array_equal(signs, symbol_vec)
#
#         eigenvalues = np.linalg.eigvals(W_k)
#         magnitudes = np.abs(eigenvalues)
#
#         if np.all(magnitudes < 1):
#             stability = 'stable'
#         elif np.all(magnitudes > 1):
#             stability = 'unstable'
#         else:
#             stability = 'saddle'
#
#         results[symbol_str] = {
#             'type': 'real' if is_real else 'virtual',
#             'stability': stability,
#             'location': fixed_point
#         }
#
#     return results

def analyze_fixed_points_continuous(model, unique_symbols_list, delta_t):
    """
    Analyzes the fixed points for a given list of visited symbolic regions
    using the continuous-time framework.
    """
    print(f"\nAnalyzing fixed points in continuous-time framework (Δt = {delta_t})...")
    M, P = model.M, model.P
    A = np.diag(model.A.detach().cpu().numpy())
    W = model.W.detach().cpu().numpy()
    h = model.h.detach().cpu().numpy()

    results = {}

    for symbol_vec in unique_symbols_list:
        symbol_str = ''.join(map(str, symbol_vec))

        # build the discrete Jacobian W_k
        padded_vec = np.pad(symbol_vec, (M - P, 0), 'constant', constant_values=1)
        D_s = np.diag(padded_vec)
        W_k = A + W @ D_s

        # --- continuous-time quantities ---
        # 1. continuous-time Jacobian A_c
        I = np.identity(M)
        A_c = (W_k - I) / delta_t

        # 2. eigenvalues of A_c
        eigenvalues = np.linalg.eigvals(A_c)

        # --- fixed point (still the discrete system) ---
        try:
            fixed_point = np.linalg.inv(I - W_k) @ h
        except np.linalg.LinAlgError:
            continue

        pre_activations = fixed_point[-P:]
        signs = (pre_activations > 0).astype(int)
        is_real = np.array_equal(signs, symbol_vec)

        # --- 3. classify stability from Re/Im of the eigenvalues ---
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag

        has_complex = np.any(np.abs(imag_parts) > 1e-9)  # numerical tolerance

        if np.all(real_parts < 0):
            stability = 'stable spiral' if has_complex else 'stable node'
        elif np.all(real_parts > 0):
            stability = 'unstable spiral' if has_complex else 'unstable node'
        else:
            stability = 'saddle'

        results[symbol_str] = {
            'type': 'real' if is_real else 'virtual',
            'stability': stability,
            'location': fixed_point,
            'eigenvalues_continuous': eigenvalues
        }

    return results

def main():
    # ============================================================
    # CONFIG - target model / data
    # defaults equal the legacy hard-coded values (no-arg run = legacy behavior).
    # CLI arguments allow run_graph_reduction_sweep.py to batch-process
    # multiple source models.
    # ============================================================
    import argparse
    parser = argparse.ArgumentParser(
        description='Graph reduction + ReLU pruning hierarchy for one AL-RNN checkpoint')
    parser.add_argument('--M', type=int, default=20,
        help='AL-RNN total units')
    parser.add_argument('--P', type=int, default=10,
        help='piecewise-linear (ReLU) units')  # known-good examples: 4, 6, 9, 12, 13, 14, 15, 17
    parser.add_argument('--option', default='',
        help='output filename suffix (make it unique per run in sweeps)')
    parser.add_argument('--model_path', default='',
        help='AL-RNN checkpoint path; defaults to the legacy hard-coded path '
             '(models/chua_3scroll_m{M}_p{P}_epoch2000of2000.pth)')
    parser.add_argument('--data_path', default='data/chua_3-scroll_train.npy')
    parser.add_argument('--delta_t', type=float, default=0.01,
        help='dt for the continuous-time fixed-point analysis (= data sample_dt).')
    _args = parser.parse_args()

    M          = _args.M
    P          = _args.P
    option     = _args.option
    num_epochs = 2000
    MODEL_PATH = _args.model_path or f"models/chua_3scroll_m{M}_p{P}_epoch{num_epochs}of2000.pth"
    DATA_PATH  = _args.data_path
    # ============================================================

    device = torch.device("cpu")

    X_train = np.load(DATA_PATH).astype(np.float32)
    X_test  = X_train

    N = X_train.shape[-1]

    model = AL_RNN(M=M, P=P, N=N)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f'Model loaded: {MODEL_PATH}  (M={M}, P={P}, N={N})')

    # Analysis
    print("\nRunning analysis...")
    X_test_torch = torch.tensor(X_test[:]).unsqueeze(0)

    # T_gen = 10000  # Sequence length
    T_gen = 40000  # Sequence length
    # T_gen = 190000
    T_r = 1000  # Transient cutoff length

    print(f"\nRunning analysis with TF warmup ({TF_WARMUP_STEPS} steps) + free run...")

    if TF_WARMUP_STEPS > 0:
        warmup_len = min(TF_WARMUP_STEPS, X_test_torch.size(1))
        z_init = _warmup_latent(model, X_test_torch[:, :warmup_len, :], alpha=1.0, n_interleave=1)
        orbit = predict_free_from_latent(model, z_init, T_gen + T_r).detach().numpy()[0][T_r:, :]
    else:
        orbit = predict_free_sequence(model, X_test_torch[:, 0, :], T_gen + T_r).detach().numpy()[0][T_r:, :]

    Dstsp = state_space_divergence_binning(torch.tensor(orbit[:, 0:model.N]), X_test_torch[0, :, :])
    DH = power_spectrum_error(torch.tensor(orbit[:, 0:model.N]), X_test_torch[0, 0:T_gen, :])
    print(f"State space distance (Dstsp): {Dstsp}")
    print(f"Hellinger Distance (DH): {DH}")

    # Plot attractor comparison
    Blues = plt.cm.Blues
    plt.rcParams["lines.linewidth"] = .35
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["lines.linewidth"] = 2.
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = orbit[:, 0]
    ys = orbit[:, 1]
    zs = orbit[:, 2]

    ax.plot(X_test[:T_gen, 0], X_test[:T_gen, 1], X_test[:T_gen, 2],
            color=Blues(0.9), label="Ground Truth")
    ax.plot(xs, ys, zs, color=Blues(0.6), alpha=1., label="Generated", linewidth=0.5)

    plt.title(r'$D_{stsp}=$'+f'{Dstsp:.3f}, ' + r'$D_H=$' + f'{DH:.3f}')

    plt.legend()
    plt.axis("off")
    _figdir = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(_figdir, exist_ok=True)
    plt.savefig(f'{_figdir}/trajectories_m{M}_p{P}_{num_epochs}epochs{option}.png', transparent=True)
    plt.savefig(f'{_figdir}/trajectories_m{M}_p{P}_{num_epochs}epochs{option}.svg', transparent=True)
    # plt.savefig(f'figures/trajectories_m{M}_p{P}_epoch{epoch}of{num_epochs}_reservoir-teacher_tf.png')
    # plt.savefig(f'figures/trajectories_m{M}_p{P}_epoch{epoch}of{num_epochs}_reservoir-teacher_PCA_tf.png')
    # plt.savefig(f'figures/trajectories_m{M}_p{P}_epoch{epoch}of{num_epochs}_only-reservoir_pca{PCA}_tf.png')
    plt.close()
    # plt.show()

    print("\nAnalyzing linear regions...")
    generated_latent = orbit[:, -P:]  # latent sequence in PWL units
    generated_observations = orbit[:, :N]  # predicted readout sequence

    bits = lrf.convert_to_bits(generated_latent)
    regions, unique_regions = lrf.unique_regions_crossed(bits, M)
    frequencies = lrf.frequency_of_regions(bits, unique_regions)

    # fixed_point_data = analyze_fixed_points(model, unique_regions)

    delta_t = _args.delta_t  # data sample_dt (default 0.01 = Chua)
    fixed_point_data = analyze_fixed_points_continuous(model, unique_regions, delta_t)

    # display the real fixed-point information
    print("\n=== Fixed Points Summary ===")
    real_fps = {k: v for k, v in fixed_point_data.items() if v['type'] == 'real'}
    virtual_fps = {k: v for k, v in fixed_point_data.items() if v['type'] == 'virtual'}

    print(f"Total unique symbols: {len(fixed_point_data)}")
    print(f"Real fixed points: {len(real_fps)}")
    print(f"Virtual fixed points: {len(virtual_fps)}")

    if real_fps:
        print("\nReal Fixed Points Details:")
        for symbol_str, fp_info in sorted(real_fps.items()):
            decimal_label = int(symbol_str, 2)
            readout_coords = fp_info['location'][:N]
            print(f"  Symbol {decimal_label} (binary: {symbol_str}): {fp_info['stability']}")
            print(f"    Readout coordinates: [{', '.join([f'{x:.4f}' for x in readout_coords])}]")

    if virtual_fps:
        print("\nVirtual Fixed Points Details:")
        for symbol_str, fp_info in sorted(virtual_fps.items()):
            decimal_label = int(symbol_str, 2)
            print(f"  Symbol {decimal_label} (binary: {symbol_str}): {fp_info['stability']}")

    print("=" * 50)

    # === Hamming-distance table ===
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    print("\nCalculating Hamming distance matrix...")
    if len(unique_regions) > 1:
    # symbols as strings for table labels
        string_labels = [''.join(map(str, region)) for region in unique_regions]


    # pairwise Hamming distances via pdist, then squareform to a square matrix
    # pdist('hamming') returns fractions, so multiply by P to get integer distances
        hamming_dist_matrix = squareform(pdist(unique_regions, 'hamming')) * P

    # display with a pandas DataFrame
        hamming_df = pd.DataFrame(hamming_dist_matrix, index=string_labels, columns=string_labels, dtype=int)
        print(hamming_df)

    # map symbol names to decimal
    symbol_to_decimal = {s: int(s, 2) for s in string_labels}
    # convert DataFrame index / columns to decimal
    hamming_df.rename(index=symbol_to_decimal, columns=symbol_to_decimal, inplace=True)

    # path_dist_df / diff_matrix are converted later (declared below)



    print("\nGenerating combined plot...")

    # 1. prepare data for the graph and plots
    # (same as the original code)
    string_labels = [''.join(map(str, region)) for region in unique_regions]
    bitcodes_str = [''.join(map(str, map(int, b))) for b in bits]
    bitcode_freq = Counter(bitcodes_str)
    most_frequent_regions = [item[0] for item in bitcode_freq.most_common()]

    # === shortest-path-length matrix of the transition graph ===
    print("\nCalculating shortest path matrix of the actual transition graph...")

    # convert string_labels to decimal
    string_labels_decimal = [str(int(s, 2)) for s in string_labels]
    
    # 1. build the directed graph from observed transitions - decimal nodes
    G_actual = nx.DiGraph()
    G_actual.add_nodes_from(string_labels_decimal)

    # add observed transitions from the time series as edges
    for i in range(len(bitcodes_str) - 1):
        source_node = str(int(bitcodes_str[i], 2))
        target_node = str(int(bitcodes_str[i + 1], 2))
        if source_node != target_node:  # ignore self-loops
            if G_actual.has_edge(source_node, target_node):
                G_actual[source_node][target_node]["weight"] += 1
            else:
                G_actual.add_edge(source_node, target_node, weight=1)

    # 2. all-pairs shortest path lengths (Floyd-Warshall)
    # computed in G_actual.nodes() order
    path_dist_matrix_np = nx.floyd_warshall_numpy(G_actual)

    # 3. arrange results into a pandas DataFrame
    # G_actual.nodes() order is not guaranteed; align to string_labels order
    node_order = list(G_actual.nodes())
    path_dist_df = pd.DataFrame(path_dist_matrix_np, index=node_order, columns=node_order)

    # reorder to match the Hamming matrix - use decimal labels
    path_dist_df = path_dist_df.reindex(index=string_labels_decimal, columns=string_labels_decimal)

    print("\nShortest Path Distance Matrix (D_path):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        # inf is hard to read; replace with -1 etc. for display
        print(path_dist_df.replace(np.inf, -1).astype(int))

    # convert path_dist_df and diff_matrix to decimal too
    path_dist_df.rename(index=symbol_to_decimal, columns=symbol_to_decimal, inplace=True)

    # 4. difference of the two matrices
    # already reindexed, so plain subtraction works
    # diff_matrix = path_dist_df - hamming_df

    # print("\nDifference Matrix (D_path - D_hamming):")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    #     replace NaN/inf first, then cast to int
        # diff_display = diff_matrix.replace([np.inf, -np.inf, np.nan], [-99, -98, -97])
        # print(diff_display.astype(int))  # use sentinel values like -99

    # color preparation (switched to PCA-based)
    from sklearn.decomposition import PCA

    # convert most_frequent_regions to decimal
    most_frequent_regions_decimal = [str(int(region, 2)) for region in most_frequent_regions]

    # vectorize the unique symbols
    unique_vectors = np.array([list(map(int, s)) for s in most_frequent_regions])

    # PCA down to one dimension
    pca = PCA(n_components=1)
    # vectorize all on-orbit symbols and apply the PCA transform
    all_vectors = np.array([list(map(int, s)) for s in bitcodes_str])
    projected_values = pca.fit_transform(all_vectors).flatten()

    # colors from the PCA value
    norm = Normalize(vmin=projected_values.min(), vmax=projected_values.max())
    # cmap = plt.get_cmap('cividis')  # similar Hamming distance -> similar color
    # === colormap customization ===
    # base colormap
    original_cmap = plt.get_cmap('cividis')
    # use the 0.2-1.0 range (avoid the darkest colors)
    start_point = 0.2
    # slice that range into a new colormap
    import matplotlib.colors as mcolors
    new_colors = original_cmap(np.linspace(start_point, 1.0, 256))
    cmap = mcolors.ListedColormap(new_colors)
    colors = cmap(norm(projected_values))

    # 2. figure with subplots
    fig = plt.figure(figsize=(32, 8))
    # fig.suptitle(f'AL-RNN Analysis (M={M}, P={P})', fontsize=16)

    # --- left plot: 3D trajectory ---
    # ax1 = fig.add_subplot(1, 7, 1, projection='3d')
    observations_plot = orbit[:, :3]

    # define marker styles per fixed-point type (defined first)
    fp_marker_styles = {
        'stable node': {'marker': 'o', 'color': 'blue', 'size': FP_MARKER_SIZE, 'label': 'Stable Node'},
        'stable spiral': {'marker': 'o', 'color': 'cyan', 'size': FP_MARKER_SIZE, 'label': 'Stable Spiral'},
        'unstable node': {'marker': '^', 'color': 'red', 'size': FP_MARKER_SIZE, 'label': 'Unstable Node'},
        'unstable spiral': {'marker': '^', 'color': 'orange', 'size': FP_MARKER_SIZE, 'label': 'Unstable Spiral'},
        'saddle': {'marker': 's', 'color': 'green', 'size': FP_MARKER_SIZE, 'label': 'Saddle'}
    }

    # collect real and virtual fixed-point data
    fp_groups = {key: {'coords': [], 'symbols': []} for key in fp_marker_styles.keys()}
    virtual_fp_groups = {key: {'coords': [], 'symbols': []} for key in fp_marker_styles.keys()}

    for symbol_str, fp_info in fixed_point_data.items():
        stability = fp_info['stability']
        fp_location = fp_info['location']
        # readout-neuron coordinates (first N dims, here 3)
        readout_coords = fp_location[:3]

        if stability in fp_groups:
            if fp_info['type'] == 'real':
                fp_groups[stability]['coords'].append(readout_coords)
                fp_groups[stability]['symbols'].append(str(int(symbol_str, 2)))
            else:  # virtual
                virtual_fp_groups[stability]['coords'].append(readout_coords)
                virtual_fp_groups[stability]['symbols'].append(str(int(symbol_str, 2)))

    # # draw the trajectory first (lower layer, more transparent)
    # ax1.scatter(observations_plot[:, 0], observations_plot[:, 1], observations_plot[:, 2],
    #             c=colors, s=5, alpha=0.2, depthshade=False)
    #
    # # plot real fixed points in output space (top layer)
    # for stability, data in fp_groups.items():
    #     if len(data['coords']) > 0:
    #         coords = np.array(data['coords'])
    #         style = fp_marker_styles[stability]
    #         ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
    #                    marker=style['marker'], c=style['color'], s=style['size'],
    #                    edgecolors='black', linewidths=2.5, alpha=1.0,
    #                    depthshade=False, label=style['label'])
    #
    #         # label fixed points with their indices
    #         if SHOW_FP_LABELS:
    #             for i, (coord, symbol) in enumerate(zip(coords, data['symbols'])):
    #                 ax1.text(coord[0], coord[1], coord[2], symbol, fontsize=8, weight='bold')
    #
    # # plot virtual fixed points (translucent, dotted edge) - optional
    # if SHOW_VIRTUAL_FPS:
    #     for stability, data in virtual_fp_groups.items():
    #         if len(data['coords']) > 0:
    #             coords = np.array(data['coords'])
    #             style = fp_marker_styles[stability]
    #             ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
    #                        marker=style['marker'], c=style['color'], s=style['size']*0.7,
    #                        edgecolors='gray', linewidths=1, alpha=0.4,
    #                        depthshade=False, label=f'Virtual {style["label"]}')
    #
    # # legend - decimal labels
    # legend_handles = []
    # num_legend_entries = len(most_frequent_regions)
    # unique_projected_map = {s: pca.transform(np.array([list(map(int, s))]))[0][0] for s in most_frequent_regions}
    #
    # for i in range(num_legend_entries):
    #     symbol_str = most_frequent_regions[i]
    #     decimal_label = str(int(symbol_str, 2))
    #     color = cmap(norm(unique_projected_map[symbol_str]))
    #     patch = mpatches.Patch(color=color, label=decimal_label)
    #     legend_handles.append(patch)
    #
    # # ax1.legend(handles=legend_handles, title="Symbols",
    # #            bbox_to_anchor=(1.15, 1), loc='upper right', fontsize='small')
    # ax1.set_title('Trajectory with Symbols and Real Fixed Points')
    # ax1.legend(loc='upper right', fontsize='small')
    # ax1.axis("off")


    # ax2 = fig.add_subplot(1, 7, 2)
    # G = nx.Graph()
    # G.add_nodes_from(string_labels_decimal)
    # for i in range(len(string_labels)):
    #     for j in range(i + 1, len(string_labels)):
    #         if hamming_df.iloc[i, j] == 1:
    #             G.add_edge(string_labels_decimal[i], string_labels_decimal[j])
    #
    # pos = nx.spring_layout(G, seed=42)
    #
    # # nx.draw_networkx(G, pos, ax=ax1, node_size=1500, node_color='skyblue', font_weight='bold')
    #
    # # color mapping for decimal nodes, matched to the legend
    # nodes_in_graph_order = list(G.nodes())
    # # convert decimal nodes back to binary for the PCA vectors
    # unique_vectors_for_color = np.array([list(map(int, format(int(node), f'0{P}b'))) for node in nodes_in_graph_order])
    # projected_for_color = pca.transform(unique_vectors_for_color).flatten()
    # symbol_to_pca_value = dict(zip(nodes_in_graph_order, projected_for_color))
    #
    # # node colors in graph-node order
    # node_colors = [cmap(norm(symbol_to_pca_value[node])) for node in G.nodes()]
    #
    # # draw the colored nodes
    # nx.draw_networkx(G, pos, ax=ax2, node_size=1500,
    #                  node_color=node_colors,  # changed from skyblue
    #                  font_weight='bold', width=2.0, edge_color='black')
    #
    # ax2.set_title('Hamming Distance-Based Symbol Connectivity')
    # ax2.axis('off')


    # --- right plot: Hamming-distance graph ---
    # ax3 = fig.add_subplot(1, 7, 3)
    # G_hamming = nx.Graph()
    # G_hamming.add_nodes_from(string_labels_decimal)
    # for i in range(len(string_labels)):
    #     for j in range(i + 1, len(string_labels)):
    #         if hamming_df.iloc[i, j] == 1:
    #             G_hamming.add_edge(string_labels_decimal[i], string_labels_decimal[j])
    #
    # pos = nx.spring_layout(G_hamming, seed=42)
    #
    # node_groups = {
    #     'real_stable_node': [], 'real_stable_spiral': [],
    #     'real_unstable_node': [], 'real_unstable_spiral': [],
    #     'real_saddle': [],
    #     'virtual_stable_node': [], 'virtual_stable_spiral': [],
    #     'virtual_unstable_node': [], 'virtual_unstable_spiral': [],
    #     'virtual_saddle': [],
    # }
    #
    # for node in G_hamming.nodes():
    #     # nodes are decimal; convert back to binary to match fixed_point_data keys
    #     node_binary = format(int(node), f'0{P}b')
    #     if node_binary in fixed_point_data:
    #         fp_info = fixed_point_data[node_binary]
    #         stability_parts = fp_info['stability'].split()
    #         key = f"{fp_info['type']}_{stability_parts[0]}"
    #         if len(stability_parts) > 1:
    #             key += f"_{stability_parts[1]}"
    #
    #         if key in node_groups:
    #             node_groups[key].append(node)
    #
    # style_map = {
    #     'real_stable_node': {'shape': 'o', 'color': 'skyblue', 'edge': 'black'},
    #     'real_stable_spiral': {'shape': 'o', 'color': 'lightcoral', 'edge': 'black'},
    #     'real_unstable_node': {'shape': '^', 'color': 'skyblue', 'edge': 'black'},
    #     'real_unstable_spiral': {'shape': '^', 'color': 'lightcoral', 'edge': 'black'},
    #     'real_saddle': {'shape': 's', 'color': 'lightgreen', 'edge': 'black'},
    #     'virtual_stable_node': {'shape': 'o', 'color': 'skyblue', 'edge': None},
    #     'virtual_stable_spiral': {'shape': 'o', 'color': 'lightcoral', 'edge': None},
    #     'virtual_unstable_node': {'shape': '^', 'color': 'skyblue', 'edge': None},
    #     'virtual_unstable_spiral': {'shape': '^', 'color': 'lightcoral', 'edge': None},
    #     'virtual_saddle': {'shape': 's', 'color': 'lightgreen', 'edge': None},
    # }
    #
    # for key, nodes in node_groups.items():
    #     if nodes:  # draw only if the node list is non-empty
    #         style = style_map[key]
    #         nx.draw_networkx_nodes(G_hamming, pos, ax=ax3, nodelist=nodes, node_shape=style['shape'],
    #                                node_color=style['color'], edgecolors=style['edge'],
    #                                linewidths=2 if style.get('edge') else 0, node_size=1500)
    #
    # # draw Hamming-distance-1 edges as dotted lines
    # nx.draw_networkx_edges(G_hamming, pos, ax=ax3, style='dotted', edge_color='gray')
    # # draw observed transitions as solid arrows
    # nx.draw_networkx_edges(G_actual, pos, ax=ax3, edge_color='black', arrowsize=20, node_size=1500)
    # # draw labels
    # nx.draw_networkx_labels(G_hamming, pos, ax=ax3, font_weight='bold')
    #
    # ax3.set_title('Symbol Connectivity (Dotted=Potential, Arrow=Actual)')
    # ax3.axis('off')
    #
    # # legend for the graph
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', label='Stable', markerfacecolor='gray', markersize=10),
    #     plt.Line2D([0], [0], marker='^', color='w', label='Unstable', markerfacecolor='gray', markersize=10),
    #     plt.Line2D([0], [0], marker='s', color='w', label='Saddle', markerfacecolor='gray', markersize=10),
    #     mpatches.Patch(color='skyblue', label='Node'),
    #     mpatches.Patch(color='lightcoral', label='Spiral'),
    #     mpatches.Patch(color='lightgreen', label='Saddle'),
    #     plt.Line2D([0], [0], marker='s', color='w', label='Real FP (Thick Black Border)', markerfacecolor='gray',
    #                markeredgecolor='black', markeredgewidth=2, markersize=10),
    #     plt.Line2D([0], [0], marker='s', color='w', label='Virtual FP (Gray Border)', markerfacecolor='gray',
    #                markeredgecolor='gray', markersize=10, alpha=0.6),
    # ]
    # ax3.legend(handles=legend_elements, loc='best', fontsize='small')


    # --- 2. build the new real-FP transition graph (G_meta) ---
    # real_fp_nodes = [node for node, data in fixed_point_data.items() if
    #                  data['type'] == 'real' and node in G_hamming.nodes()]
    # virtual_fp_nodes = [node for node, data in fixed_point_data.items() if
    #                     data['type'] == 'virtual' and node in G_hamming.nodes()]
    #
    # G_meta = nx.DiGraph()
    # G_meta.add_nodes_from(real_fp_nodes)
    #
    # # for each real-FP pair (A, B), search for a path
    # for A in real_fp_nodes:
    #     for B in real_fp_nodes:
    #         if A == B:
    #             continue
    #
    #         # subgraph containing A, B and all virtual nodes
    #         nodes_for_subgraph = [A, B] + virtual_fp_nodes
    #         subgraph = G_actual.subgraph(nodes_for_subgraph)
    #
    #         # check whether a path from A to B exists in the subgraph
    #         if nx.has_path(subgraph, source=A, target=B):
    #             G_meta.add_edge(A, B)
    #
    # # layout for G_meta
    # pos_meta = nx.spring_layout(G_meta, seed=42)

    # # --- 2. build the new essential real-FP transition graph (G_essential) ---
    # real_fp_nodes = {node for node, data in fixed_point_data.items() if
    #                  data['type'] == 'real' and node in G_hamming.nodes()}
    #
    # G_essential = nx.DiGraph()
    # G_essential.add_nodes_from(list(real_fp_nodes))
    #
    # last_real_fp = None
    # # scan the symbol time series
    # for symbol in bitcodes_str:
    #     if symbol in real_fp_nodes:
    #         # if there is a previously visited real FP different from the current one
    #         if last_real_fp is not None and symbol != last_real_fp:
    #             # add an edge from the last visited real FP to the current one
    #             G_essential.add_edge(last_real_fp, symbol)
    #         # update the last visited real FP
    #         last_real_fp = symbol
    #
    # # layout for G_essential
    # pos_essential = nx.spring_layout(G_essential, seed=42)

    # --- 2. build the essential real-FP transition graph (G_essential) with weights ---
    # real_fp_nodes: binary strings converted to decimal strings
    real_fp_nodes_binary = {node for node, data in fixed_point_data.items() if
                           data['type'] == 'real'}
    real_fp_nodes = {str(int(node, 2)) for node in real_fp_nodes_binary
                     if str(int(node, 2)) in string_labels_decimal}

    G_essential = nx.DiGraph()
    G_essential.add_nodes_from(list(real_fp_nodes))

    # sequence of real FPs visited by the orbit - binary converted to decimal
    real_fp_path = [str(int(s, 2)) for s in bitcodes_str if str(int(s, 2)) in real_fp_nodes]

    # count transitions and use the counts as edge weights
    for i in range(len(real_fp_path) - 1):
        source = real_fp_path[i]
        target = real_fp_path[i + 1]

        if source != target:
            if G_essential.has_edge(source, target):
                # increment the weight if the edge already exists
                G_essential[source][target]['weight'] += 1
            else:
                # add new edges with weight 1
                G_essential.add_edge(source, target, weight=1)

    # layout for G_essential
    pos_essential = nx.spring_layout(G_essential, seed=42)


    # plot 4: essential real-FP transition graph (ax4)
    # ax4 = fig.add_subplot(1, 7, 4)
    #
    # # --- scale edge widths by weight ---
    # edges = G_essential.edges()
    # scaled_widths = []  # initialize the list first
    # if edges:
    #     weights = [G_essential[u][v]['weight'] for u, v in edges]
    #     min_weight = min(weights)
    #     max_weight = max(weights)
    #     if max_weight > min_weight:
    #         scaled_widths = [1.0 + 9.0 * (w - min_weight) / (max_weight - min_weight) for w in weights]
    #     else:
    #         scaled_widths = [5.0] * len(weights)
    #
    # # --- draw the nodes (unchanged) ---
    # essential_node_groups = {key: [] for key in style_map.keys()}
    # for node in G_essential.nodes():
    #     # nodes are decimal; convert back to binary to match fixed_point_data keys
    #     node_binary = format(int(node), f'0{P}b')
    #     if node_binary in fixed_point_data:
    #         fp_info = fixed_point_data[node_binary]
    #         stability_parts = fp_info['stability'].split()
    #         key = f"real_{stability_parts[0]}"
    #         if len(stability_parts) > 1: key += f"_{stability_parts[1]}"
    #         if key in essential_node_groups: essential_node_groups[key].append(node)
    #
    # for key, nodes in essential_node_groups.items():
    #     if nodes and 'real' in key:
    #         style = style_map[key]
    #         nx.draw_networkx_nodes(G_essential, pos, ax=ax4, nodelist=nodes, node_shape=style['shape'],
    #                                node_color=style['color'], edgecolors='black', linewidths=2, node_size=1500)
    #
    # # --- draw edges and labels (run once) ---
    # nx.draw_networkx_edges(G_essential, pos, ax=ax4,
    #                        width=scaled_widths,  # set the widths
    #                        edge_color='black', arrowsize=20, node_size=1500,
    #                        connectionstyle='arc3,rad=0.1')
    # nx.draw_networkx_labels(G_essential, pos, ax=ax4, font_weight='bold')
    #
    # # --- graph title and legend ---
    # ax4.set_title('Essential Real FP Graph (Width = Transition Freq.)')
    # ax4.axis('off')
    # ax4.legend(handles=legend_elements, loc='best', fontsize='small')

    # plot 5: trajectory colored only in real-FP regions (ax5)
    ax5 = fig.add_subplot(1, 4, 1, projection='3d')

    # color preparation - real FPs get evenly spaced colors
    real_fp_nodes_sorted = sorted(list(real_fp_nodes), key=int)  # numeric sort

    # ============================================================
    # NEW: Cluster symbols into K=|Sigma_real| clusters (seeded by real symbols)
    # Objective: minimize the number of contracted edge TYPES in the cluster transition graph.
    # - Directed graph for objective: G_actual (observed symbol transitions)
    # - Undirected graph for connectivity: G_sym_adj (observed adjacency, no enclaves)
    # - Each cluster contains exactly one real symbol (seed), and must be connected in G_sym_adj
    # Solver is selected by CLUSTER_SOLVER ("milp" or "heuristic")
    # ============================================================

    # Directed graph (objective) and undirected graph (connectivity)
    G_dir = G_actual.copy()

    # Create undirected adjacency graph based on actual observed transitions
    G_sym_adj = G_actual.to_undirected()

    G_und = G_sym_adj.copy()

    # Seeds are real symbols (decimal strings) that appear in this graph
    seed_nodes = sorted([s for s in real_fp_nodes_sorted if s in G_und.nodes()], key=int)
    if len(seed_nodes) == 0:
        # fallback: at least 1 cluster
        seed_nodes = [sorted(list(G_und.nodes()), key=int)[0]]

    # Solve clustering (node -> cluster_id)
    sym_to_cluster_id, contracted_edge_types_min = solve_symbol_clustering(
        G_dir=G_dir,
        G_und=G_und,
        real_nodes=seed_nodes,
        method=CLUSTER_SOLVER,
        compactness_penalty_lambda=COMPACTNESS_PENALTY_LAMBDA
    )

    # Map cluster_id -> seed label (decimal string)
    cluster_id_to_seed = {i: seed_nodes[i] for i in range(len(seed_nodes))}

    # Map each timestep's symbol to its cluster seed (for trajectory coloring & cluster transitions)
    cluster_seq = []
    cluster_id_seq = []
    for sym_bin in bitcodes_str:
        sym_dec = str(int(sym_bin, 2))
        cid = sym_to_cluster_id.get(sym_dec, 0)  # should always exist
        cluster_id_seq.append(cid)
        cluster_seq.append(cluster_id_to_seed[cid])

    # Cluster colors: assign each seed a color position uniformly
    cluster_seeds_sorted = sorted(list(set(cluster_seq) & set(seed_nodes)), key=int)
    if len(cluster_seeds_sorted) == 0:
        cluster_seeds_sorted = sorted(seed_nodes, key=int)

    if len(cluster_seeds_sorted) > 1:
        cluster_color_positions = np.linspace(0, 1, len(cluster_seeds_sorted))
    else:
        cluster_color_positions = np.array([0.5])

    seed_to_color_pos = dict(zip(cluster_seeds_sorted, cluster_color_positions))
    cluster_colors = [cmap(seed_to_color_pos.get(c, 0.5)) for c in cluster_seq]

    # Build cluster transition graph (directed, weighted) from the cluster sequence
    G_cluster = nx.DiGraph()
    G_cluster.add_nodes_from(cluster_seeds_sorted)

    # compress consecutive duplicates
    cluster_path = []
    for c in cluster_seq:
        if len(cluster_path) == 0 or cluster_path[-1] != c:
            cluster_path.append(c)

    # same compressed path but in cluster-id space (for metrics)
    cluster_path_ids = []
    for cid in cluster_id_seq:
        if len(cluster_path_ids) == 0 or cluster_path_ids[-1] != cid:
            cluster_path_ids.append(cid)

    for i in range(len(cluster_path) - 1):
        a, b = cluster_path[i], cluster_path[i + 1]
        if a == b:
            continue
        if G_cluster.has_edge(a, b):
            G_cluster[a][b]['weight'] += 1
        else:
            G_cluster.add_edge(a, b, weight=1)

    # Precompute layout for cluster graph
    pos_cluster = nx.spring_layout(G_cluster, seed=42)

    # Compute cluster compactness for reporting
    V_all = list(G_dir.nodes())
    k = len(seed_nodes)
    cluster_nodes_list = [set() for _ in range(k)]
    for v in V_all:
        cluster_nodes_list[sym_to_cluster_id[v]].add(v)

    cluster_compactness_values = [_compute_cluster_compactness(G_und, cluster_nodes_list[i], seed_nodes[i]) for i in range(k)]
    max_cluster_compactness = max(cluster_compactness_values) if cluster_compactness_values else 0.0
    mean_cluster_compactness = float(np.mean(cluster_compactness_values)) if cluster_compactness_values else 0.0

    print(f"[Clustering] method={CLUSTER_SOLVER}, K={len(seed_nodes)}, contracted edge types={contracted_edge_types_min}, max_compactness={max_cluster_compactness:.2f}")
    # --- Secondary metrics (tie-breakers) and CSV logging ---
    sec = compute_cluster_secondary_metrics(
        G_dir=G_dir,
        G_und=G_und,
        assign=sym_to_cluster_id,
        seed_nodes=seed_nodes,
        cluster_id_seq=cluster_id_seq,
        cluster_path_ids=cluster_path_ids,
    )
    # enrich with run context
    sec.update({
        "M": M,
        "P": P,
        "option": option,
        "epoch": num_epochs,
        "T_gen": T_gen,
        "CLUSTER_SOLVER": CLUSTER_SOLVER,
        "COMPACTNESS_PENALTY_LAMBDA": COMPACTNESS_PENALTY_LAMBDA,
        "max_cluster_compactness": float(max_cluster_compactness),
        "mean_cluster_compactness": float(mean_cluster_compactness),
        "edge_types_solver_reported": int(contracted_edge_types_min),
        "edge_types_gap": int(sec["edge_types_min_obj"]) - int(contracted_edge_types_min),
        "n_symbols_nodes": int(len(G_dir.nodes())),
        "n_transitions_edges": int(len(G_dir.edges())),
    })

    _resdir = f"results/relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(_resdir, exist_ok=True)
    csv_path = f"{_resdir}/cluster_compactness-penalty_secondary_metrics_{CLUSTER_SOLVER}{option}.csv"
    df_row = pd.DataFrame([sec])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

    print(f"[Clustering] secondary metrics saved: {csv_path}")

    # ============================================================
    # ReLU pruning-induced quotient graph hierarchy
    # (independent from the seed-based clustering above)
    # ============================================================
    if RUN_RELU_PRUNE_HIERARCHY:
        print("\n[ReLU Prune] Starting pruning-induced quotient graph hierarchy...")

        # Build a per-leaf callback for incremental route-subplot saving
        if RELU_PRUNE_NUM_ROUTE_SUBPLOTS > 0:
            _subplot_G_dir   = G_dir
            _subplot_P       = P
            _subplot_M       = M
            _subplot_option  = option
            _subplot_seed    = 0
            _subplot_fp      = real_fp_nodes
            _subplot_fmt     = SYMBOL_LABEL_FORMAT

            def _on_leaf_subplot(T_leaf, dag_edges_partial, metrics_map_partial, valid_sets_partial):
                partial_results = {
                    'valid_sets': list(valid_sets_partial),
                    'leaf_sets':  [T_leaf],
                    'dag_edges':  list(dag_edges_partial),
                    'metrics_map': dict(metrics_map_partial),
                }
                save_relu_prune_route_subplots(
                    hierarchy_results=partial_results,
                    G_dir=_subplot_G_dir,
                    M=_subplot_M,
                    P=_subplot_P,
                    option=_subplot_option,
                    seed=_subplot_seed,
                    num_routes=1,
                    real_fp_nodes=_subplot_fp,
                    label_fmt=_subplot_fmt,
                )
        else:
            _on_leaf_subplot = None

        hierarchy_results = explore_relu_pruning_hierarchy(
            G_dir=G_dir,
            G_und=G_und,
            fixed_nodes=seed_nodes,
            P=P,
            candidate_mode=RELU_PRUNE_CANDIDATE_MODE,
            max_depth=RELU_PRUNE_MAX_DEPTH,
            on_leaf=_on_leaf_subplot,
        )

        _valid_sets = hierarchy_results['valid_sets']
        _leaf_sets = hierarchy_results['leaf_sets']
        _base_candidates = hierarchy_results['base_candidates']
        _max_depth_reached = hierarchy_results['max_depth_reached']

        print("\nReLU pruning hierarchy:")
        print(f"  base candidates: {sorted(_base_candidates)}")
        print(f"  valid deletion sets: {len(_valid_sets)}")
        print(f"  leaf deletion sets: {len(_leaf_sets)}")
        print(f"  max depth reached: {_max_depth_reached}")

        _leaf_set_frozen = set(_leaf_sets)
        _leaf_entries = [
            (T, assign, metrics)
            for T, assign, metrics in _valid_sets
            if T in _leaf_set_frozen
        ]
        if _leaf_entries:
            _leaf_sorted = sorted(_leaf_entries,
                                  key=lambda x: (-x[2]['num_deleted'], x[2]['edge_types']))
            _best = _leaf_sorted[0]
            print(f"  best leaves by num_deleted/edge_types: "
                  f"deleted={{{_best[2]['deleted_set_str']}}}, "
                  f"num_deleted={_best[2]['num_deleted']}, "
                  f"edge_types={_best[2]['edge_types']}")
        else:
            print("  best leaves by num_deleted/edge_types: (none)")

        # ============================================================
        # Save the retraining spec: store ALL distinct deletion sets whose
        # num_clusters (visited symbols after reduction) attains the global
        # minimum. Selection covers all valid_sets; is_leaf is not a criterion
        # (see the select_minimal_symbol_candidates docstring).
        # Index convention (shared with AL_RNN.relu_mask / retraining):
        #   deleted_relu_local_index j
        #       = j-th element among the last P activation slots
        #       → global hidden index = M - P + j
        # (symbol_to_bits: index 0 = MSB = local slot 0)
        # ============================================================
        _minimal_nc, _min_cands = select_minimal_symbol_candidates(_valid_sets)
        if _min_cands:
            _cand_entries = []
            for _cid, (_T, _m) in enumerate(_min_cands):
                _deleted_local = sorted(_T)
                _remaining_local = sorted(set(range(P)) - set(_deleted_local))
                _cand_entries.append({
                    "candidate_id": _cid,
                    "deleted_relu_local_indices": _deleted_local,
                    "remaining_relu_local_indices": _remaining_local,
                    "num_deleted": int(_m['num_deleted']),
                    "P_effective": len(_remaining_local),
                    "num_clusters": int(_m['num_clusters']),
                    "edge_types": int(_m['edge_types']),
                    "weighted_edges_total": int(_m['weighted_edges_total']),
                    "num_scc_ge3": int(_m['num_scc_ge3']),
                    "largest_scc_size": int(_m['largest_scc_size']),
                    "cycle_penalty_ge3": int(_m['cycle_penalty_ge3']),
                    "is_leaf": bool(_m.get('is_leaf', False)),
                })
            _multi_spec = {
                "source_model_path": MODEL_PATH,
                "M": M,
                "P_original": P,
                "selection_rule": "minimum_num_clusters",
                "minimal_num_clusters": int(_minimal_nc),
                # number of real-FP nodes present on the graph at reduction time
                # (= FPs protected by the fixed-point-collision constraint).
                # It can disagree with the training-time vis_fps because the
                # trajectory protocols differ; sources below TARGET_FP_COUNT
                # only admit reductions that cannot represent every fixed
                # point, so the retraining selection excludes them.
                "num_real_fp_nodes": len(seed_nodes),
                "num_candidates": len(_cand_entries),
                "candidates": _cand_entries,
            }
            _spec_dir = "results/relu_hierarchy/retraining_specs"
            os.makedirs(_spec_dir, exist_ok=True)
            _model_stem = os.path.splitext(os.path.basename(MODEL_PATH))[0]
            _spec_path = os.path.join(
                _spec_dir, f"{_model_stem}_minimal_symbol_reductions.json")
            with open(_spec_path, "w") as _f:
                json.dump(_multi_spec, _f, indent=2)

            print(f"\n  Source model: {MODEL_PATH}")
            print(f"  P_original: {P}")
            print(f"  Minimal visited-symbol count (num_clusters): {_minimal_nc}")
            print(f"  Number of minimal candidates: {len(_cand_entries)}")
            for _c in _cand_entries:
                print(f"\n  Candidate {_c['candidate_id']}:")
                print(f"    deleted = {{{','.join(map(str, _c['deleted_relu_local_indices']))}}}")
                print(f"    P_effective = {_c['P_effective']}")
                print(f"    num_clusters = {_c['num_clusters']}")
                print(f"    edge_types = {_c['edge_types']}")
                print(f"    is_leaf = {_c['is_leaf']}")
            print(f"\n  minimal symbol reduction spec saved: {_spec_path}")
        else:
            print("  No valid reduced candidates were found "
                  "(no spec saved — the source model is not retrained)")

        save_relu_prune_hierarchy_csv(
            hierarchy_results=hierarchy_results,
            M=M,
            P=P,
            option=option,
            save_leaves_only=RELU_PRUNE_SAVE_LEAVES_ONLY or (not RELU_PRUNE_SAVE_ALL_VALID),
        )
        real_fp_coords_list = [
            fixed_point_data[node]['location'][:N]
            for node in sorted(real_fp_nodes_binary)
            if str(int(node, 2)) in real_fp_nodes
        ]
        save_top_relu_prune_quotient_graphs(
            hierarchy_results=hierarchy_results,
            G_dir=G_dir,
            M=M,
            P=P,
            option=option,
            top_k=10,
            real_fp_nodes=real_fp_nodes,
            label_fmt=SYMBOL_LABEL_FORMAT,
            orbit=orbit,
            symbol_seq=[str(int(b, 2)) for b in bitcodes_str],
            N=N,
            real_fp_coords=real_fp_coords_list,
        )
        save_relu_prune_hierarchy_dag(
            hierarchy_results=hierarchy_results,
            M=M,
            P=P,
            option=option,
        )
        if RELU_PRUNE_NUM_ROUTE_PLOTS > 0:
            save_relu_prune_route_plots(
                hierarchy_results=hierarchy_results,
                G_dir=G_dir,
                M=M,
                P=P,
                option=option,
                num_plots=RELU_PRUNE_NUM_ROUTE_PLOTS,
                real_fp_nodes=real_fp_nodes,
                label_fmt=SYMBOL_LABEL_FORMAT,
            )
        save_relu_prune_min_cluster_comparison(
            hierarchy_results=hierarchy_results,
            G_dir=G_dir,
            M=M,
            P=P,
            option=option,
            real_fp_nodes=real_fp_nodes,
            label_fmt=SYMBOL_LABEL_FORMAT,
        )

    if len(real_fp_nodes_sorted) > 1:
        # divide the colormap range evenly
        color_positions = np.linspace(0, 1, len(real_fp_nodes_sorted))
        symbol_to_color_pos = dict(zip(real_fp_nodes_sorted, color_positions))
    else:
        # a single fixed point gets the middle color
        symbol_to_color_pos = {real_fp_nodes_sorted[0]: 0.5} if real_fp_nodes_sorted else {}

    real_fp_colors = []

    # modified block:
    # virtual-FP symbol colors as RGBA tuples
    # virtual_fp_color = (0.0, 0.0, 0.0, 1.0)  # Black in RGBA format
    virtual_fp_color = (1.0, 0.7, 0.7, 1.0)  # Light red in RGBA format

    for symbol in bitcodes_str:
        symbol_decimal = str(int(symbol, 2))
        if symbol_decimal in real_fp_nodes:
            color_pos = symbol_to_color_pos.get(symbol_decimal, 0.5)
            real_fp_colors.append(cmap(color_pos))
        else:
            # virtual fixed points in light red
            real_fp_colors.append(virtual_fp_color)
    # end of modified block

    # draw the trajectory first (lower layer, more transparent)
    ax5.scatter(observations_plot[:, 0], observations_plot[:, 1], observations_plot[:, 2],
                c=real_fp_colors, s=5, alpha=0.2, depthshade=False)

    # plot real fixed points in output space (top layer)
    for stability, data in fp_groups.items():
        if len(data['coords']) > 0:
            coords = np.array(data['coords'])
            style = fp_marker_styles[stability]
            ax5.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       marker=style['marker'], c=style['color'], s=style['size'],
                       edgecolors='black', linewidths=2.5, alpha=1.0,
                       depthshade=False, label=style['label'])

            # label fixed points with their indices
            if SHOW_FP_LABELS:
                for i, (coord, symbol) in enumerate(zip(coords, data['symbols'])):
                    ax5.text(coord[0], coord[1], coord[2], symbol, fontsize=8, weight='bold')

    # plot virtual fixed points (also on ax5) - optional
    if SHOW_VIRTUAL_FPS:
        for stability, data in virtual_fp_groups.items():
            if len(data['coords']) > 0:
                coords = np.array(data['coords'])
                style = fp_marker_styles[stability]
                ax5.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           marker=style['marker'], c=style['color'], s=style['size']*0.7,
                           edgecolors='gray', linewidths=1, alpha=0.4,
                           depthshade=False)

    # dedicated legend - decimal labels, evenly spaced colors
    real_fp_legend_handles = []
    # wrap with sorted() for numeric order
    for symbol_str in real_fp_nodes_sorted:
        color_pos = symbol_to_color_pos.get(symbol_str, 0.5)
        color = cmap(color_pos)
        patch = mpatches.Patch(color=color, label=symbol_str)
        real_fp_legend_handles.append(patch)
    # match the legend's black to the RGBA tuples
    real_fp_legend_handles.append(mpatches.Patch(color=virtual_fp_color, label='Virtual FP Paths'))

    # also add the fixed-point legend
    for stability, style in fp_marker_styles.items():
        if len(fp_groups[stability]['coords']) > 0:
            real_fp_legend_handles.append(
                plt.Line2D([0], [0], marker=style['marker'], color='w',
                          markerfacecolor=style['color'], markeredgecolor='black',
                          markeredgewidth=2, markersize=10, label=style['label'])
            )

    ax5.legend(handles=real_fp_legend_handles, title="Real FP Symbols & Types", fontsize='small')
    # ax5.set_title('Trajectory with Real FP Regions and Fixed Points')
    ax5.axis("off")

    # adjust the layout and save

    # ============================================================
    # NEW subplot (ax2): Actual transition graph (weighted directed graph)
    # ============================================================
    ax2 = fig.add_subplot(1, 4, 2)

    # weighted directed graph built from G_actual
    # node colors synchronized with the other plots
    nodes_in_actual_order = list(G_actual.nodes())
    unique_vectors_for_actual = np.array([list(map(int, format(int(node), f'0{P}b'))) for node in nodes_in_actual_order])
    projected_for_actual = pca.transform(unique_vectors_for_actual).flatten()
    symbol_to_pca_actual = dict(zip(nodes_in_actual_order, projected_for_actual))
    actual_node_colors = [cmap(norm(symbol_to_pca_actual[node])) for node in G_actual.nodes()]

    # layout for G_actual
    pos_actual = nx.spring_layout(G_actual, seed=42)

    # edge weights (transition counts)
    edge_weights = {}
    for i in range(len(bitcodes_str) - 1):
        source_node = str(int(bitcodes_str[i], 2))
        target_node = str(int(bitcodes_str[i + 1], 2))
        if source_node != target_node:
            edge = (source_node, target_node)
            edge_weights[edge] = edge_weights.get(edge, 0) + 1

    # scale the edge widths
    edges_actual = list(G_actual.edges())
    widths_actual = []
    if len(edges_actual) > 0:
        wts_actual = [edge_weights.get((u, v), 1) for u, v in edges_actual]
        wmin_actual, wmax_actual = min(wts_actual), max(wts_actual)
        if wmax_actual > wmin_actual:
            widths_actual = [0.5 + 4.5 * (w - wmin_actual) / (wmax_actual - wmin_actual) for w in wts_actual]
        else:
            widths_actual = [2.5] * len(wts_actual)

    # fixed-point border: real FP → black/thick, others → no border
    fp_edgecolors = []
    fp_lws = []
    actual_labels = {}
    for node in G_actual.nodes():
        actual_labels[node] = _node_label(node, P, SYMBOL_LABEL_FORMAT)
        if node in real_fp_nodes:
            fp_edgecolors.append('black')
            fp_lws.append(4.0)
        else:
            fp_edgecolors.append('none')
            fp_lws.append(0.0)

    # draw the graph
    nx.draw_networkx_nodes(G_actual, pos_actual, ax=ax2, node_size=1500,
                           node_color=actual_node_colors,
                           edgecolors=fp_edgecolors, linewidths=fp_lws)
    nx.draw_networkx_labels(G_actual, pos_actual, labels=actual_labels, ax=ax2, font_weight='bold')
    nx.draw_networkx_edges(G_actual, pos_actual, ax=ax2,
                           width=widths_actual, edge_color='black', arrowsize=20,
                           connectionstyle='arc3,rad=0.1')

    # ax2.set_title('Actual Transition Graph (Weighted by Frequency)')
    ax2.axis('off')

    # ============================================================
    # NEW subplot (ax6): Cluster transition graph (seeded by real symbols)
    # ============================================================
    ax6 = fig.add_subplot(1, 4, 3)

    # Node colors for cluster graph
    cluster_node_colors = [cmap(seed_to_color_pos.get(n, 0.5)) for n in G_cluster.nodes()]

    # Edge widths scaled by transition frequency
    edges_c = list(G_cluster.edges())
    widths_c = []
    if len(edges_c) > 0:
        wts = [G_cluster[u][v]['weight'] for u, v in edges_c]
        wmin, wmax = min(wts), max(wts)
        if wmax > wmin:
            widths_c = [1.0 + 7.0 * (w - wmin) / (wmax - wmin) for w in wts]
        else:
            widths_c = [4.0] * len(wts)

    cluster_labels = {n: _node_label(n, P, SYMBOL_LABEL_FORMAT) for n in G_cluster.nodes()}
    nx.draw_networkx_nodes(G_cluster, pos_cluster, ax=ax6, node_size=1500,
                           node_color=cluster_node_colors, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G_cluster, pos_cluster, labels=cluster_labels, ax=ax6, font_weight='bold')
    nx.draw_networkx_edges(G_cluster, pos_cluster, ax=ax6,
                           width=widths_c, edge_color='black', arrowsize=20,
                           connectionstyle='arc3,rad=0.12')

    # ax6.set_title('Cluster Transition Graph (Seeds = Real Symbols)')
    ax6.axis('off')

    # ============================================================
    # NEW subplot (ax7): Trajectory colored by cluster assignment
    # ============================================================
    ax7 = fig.add_subplot(1, 4, 4, projection='3d')

    ax7.scatter(observations_plot[:, 0], observations_plot[:, 1], observations_plot[:, 2],
                c=cluster_colors, s=5, alpha=0.2, depthshade=False)

    # overlay real fixed points (same markers as ax1/ax5)
    for stability, data in fp_groups.items():
        if len(data['coords']) > 0:
            coords = np.array(data['coords'])
            style = fp_marker_styles[stability]
            ax7.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        marker=style['marker'], c=style['color'], s=style['size'],
                        edgecolors='black', linewidths=2.5, alpha=1.0,
                        depthshade=False)

    # Cluster legend (show seeds only)
    cluster_legend_handles = []
    for seed in cluster_seeds_sorted:
        patch = mpatches.Patch(color=cmap(seed_to_color_pos.get(seed, 0.5)), label=f'Cluster {seed}')
        cluster_legend_handles.append(patch)

    ax7.legend(handles=cluster_legend_handles, title="Clusters (seed symbol)", fontsize='small', loc='upper right')
    # ax7.set_title('Trajectory Colored by Cluster')
    ax7.axis("off")

    plt.tight_layout()  # avoid overlap with the suptitle
    _figdir2 = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(_figdir2, exist_ok=True)
    save_path = f'{_figdir2}/trajectories_with_symbols_and_clusters_compactness-penalty_{CLUSTER_SOLVER}_m{M}_p{P}{option}.png'
    plt.savefig(save_path)
    plt.close()

    print(f"\nCombined analysis plot saved to {save_path}")



    # === residence-time distributions (total and consecutive) ===
    print("\nVisualizing symbol residence time distributions (total vs. continuous)...")

    # --- 1. consecutive residence times ---
    continuous_residence_times = {label: [] for label in bitcode_freq.keys()}
    if len(bitcodes_str) > 0:
        current_symbol = bitcodes_str[0]
        current_length = 1
        for i in range(1, len(bitcodes_str)):
            if bitcodes_str[i] == current_symbol:
                current_length += 1
            else:
                continuous_residence_times[current_symbol].append(current_length)
                current_symbol = bitcodes_str[i]
                current_length = 1
        continuous_residence_times[current_symbol].append(current_length)  # add the final stay

    # --- 2. two subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Symbol Residence Time Analysis', fontsize=16)

    # --- left: total residence time (bar plot) ---
    sorted_freqs = bitcode_freq.most_common()
    labels = [item[0] for item in sorted_freqs]
    labels_decimal = [str(int(label, 2)) for label in labels]  # decimal labels
    counts = np.array([item[1] for item in sorted_freqs])
    percentages = (counts / T_gen) * 100

    # colors synchronized with the other plots
    bar_vectors = np.array([list(map(int, s)) for s in labels])
    projected_for_bars = pca.transform(bar_vectors).flatten()
    bar_colors = cmap(norm(projected_for_bars))

    axes[0].bar(labels_decimal, percentages, color=bar_colors)
    axes[0].set_title('Total Residence Time Distribution')
    axes[0].set_ylabel('Residence Time (%)')
    axes[0].set_xlabel('Symbol')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0, max(percentages) * 1.1)

    # --- right: consecutive residence times (box plot) ---
    # data sorted by frequency
    plot_data = [continuous_residence_times[label] for label in labels]

    # patch_artist=True enables coloring
    bp = axes[1].boxplot(plot_data, labels=labels_decimal, patch_artist=True)

    # synchronize each box's color
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)

    axes[1].set_title('Continuous Residence Time Distribution')
    axes[1].set_ylabel('Continuous Residence Time (timesteps)')
    axes[1].set_xlabel('Symbol')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_yscale('log')  # time scales vary widely, use log

    # --- 3. save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _figdir3 = f"figures/graph_reduction_with_relu_hierarchy/m{M}_p{P}{option}"
    os.makedirs(_figdir3, exist_ok=True)
    dist_save_path = f'{_figdir3}/residence_analysis_m{M}_p{P}{option}.png'
    plt.savefig(dist_save_path)
    plt.close()

    print(f"Residence time analysis plot saved to {dist_save_path}")

    print("All tests completed!")

if __name__ == "__main__":
    main()

