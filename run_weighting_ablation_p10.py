#!/usr/bin/env python3
"""
Launcher for train_alrnn.py — data mode & ratio sweep.

DATA_MODE = "original"   -> train on the raw Chua time series (uses RAW_DATA_PATH)
DATA_MODE = "augmented"  -> train on each RATIO_GRID condition (bridge/long episodes)

Parallelism: SEEDS are launched simultaneously within each condition×nint combination.
Each subprocess covers all P in P_LIST.

Progress lines from subprocesses ([PROGRESS] prefix) are printed to the terminal.
"""

import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


# ============================================================
# User-editable settings
# ============================================================

TRAIN_SCRIPT = "train_alrnn.py"

# ── Data mode ──────────────────────────────────────────────
# "original"  : train on the raw Chua time series (uses RAW_DATA_PATH)
# "augmented" : train on each RATIO_GRID condition
# DATA_MODE = "augmented"
DATA_MODE = "original"

RAW_DATA_PATH = "data/chua_3-scroll_train.npy"  # only used when data_mode=original

# ── Episode ratio grid (used when DATA_MODE = "augmented") ─
# RATIO_GRID = [
#     # {"name": "b00_lo100", "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.00, "r_long": 1.00},
#     # {"name": "b10_lo90",  "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.10, "r_long": 0.90},
#     # {"name": "b20_lo80",  "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.20, "r_long": 0.80},
#     {"name": "b30_lo70",  "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.30, "r_long": 0.70},
#     # {"name": "b40_lo60",  "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.40, "r_long": 0.60},
#     # {"name": "b50_lo50",  "r_anchor": 0.0, "r_local": 0.0, "r_bridge": 0.50, "r_long": 0.50},
# ]

# ── Sweep axes ─────────────────────────────────────────────
N_INTERLEAVE_LIST = [128]
# P range split to run alongside the reverse launcher (overlap would double-
# train the same run when the fronts cross -> checkpoint corruption risk).
# P=16 is excluded because seed10 lacks a reduction spec
# (reaching it stops with FileNotFoundError; add it back once specs exist).
P_LIST            = [10]   # the weighting ablation only uses P_parent=10
# SEEDS             = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS             = list(range(30))   # training seeds for normal runs (RETRAIN_REDUCED=False)

# sweep axes when RETRAIN_REDUCED=True (SEEDS is unused):
#   SOURCE_SEEDS  : seed of the source over-parameterized model
#   RETRAIN_SEEDS : retraining seed (all combinations per source)
# e.g. 30 sources x 1 retrain seed -> SOURCE_SEEDS=range(30), RETRAIN_SEEDS=[0]
#     5×5 nested             → SOURCE_SEEDS=[0,5,10,15,20], RETRAIN_SEEDS=[0,1,2,3,4]
SOURCE_SEEDS  = list(range(30))
RETRAIN_SEEDS = [0]

# ── Training hyper-parameters ──────────────────────────────
NUM_EPOCHS      = 2000
STEPS_PER_EPOCH = 50
BATCH_EPISODES  = 16

SEGMENT_LEN    = 200
MAX_BRIDGE_LEN = 200

ALPHA = 1.0

INPUT_SIGMA = 0.0
STATE_SIGMA = 0.0

LR_START = 1e-3
LR_END   = 1e-5
LR_FIXED = False

SSI              = 10
NO_PERIODIC_EVAL = False

TARGET_FP_COUNT = 5
FP_COUNT_WEIGHT = 5.0

# ── Reduced-model retraining mode ──────────────────────────
# When RETRAIN_REDUCED = True, instead of normal scratch training, use the
# best reduction spec saved by graph_reduction_with_relu_pruning.py and
# retrain from the source over-parameterized checkpoint with ReLU->identity fixed.
# P_LIST then acts as a sweep axis over the source models' P_original
# (P_effective is determined by each reduction spec).
RETRAIN_REDUCED    = True
REDUCTION_SPEC_DIR = Path("results/relu_hierarchy/retraining_specs")

# ── global minimum symbol selection (retrain mode) ──────────
# When GLOBAL_MIN_ONLY = True, restrict retraining to the candidates that
#   attain the global minimum num_clusters across all eligible sources and
#   all candidates; do NOT simply run every per-source local
#   minimum as-is.
# TARGET_P_EFFS:
#   None        -> no P_eff filter (retrain every global-min candidate)
#   [3, 4] etc. -> among global-min candidates, retrain only those whose
#                  P_effective is in this list (error out if none match)
# NOTE: the global minimum itself is determined by the full candidate scan;
#   TARGET_P_EFFS is only a post-filter on what to retrain.
# False restores the old behavior: run every per-source local-min candidate.
GLOBAL_MIN_ONLY = True
TARGET_P_EFFS   = None

# ── teacher distillation (auxiliary loss in retrain mode) ─────────
# use the source over-parameterized model as a frozen teacher.
#   "none"                   : existing baseline (no auxiliary loss, default)
#   "full_preactivation"     : match all M preactivations to the teacher (MSE)
#   "retained_preactivation" : MSE only on the retained ReLU slots
#   "symbol"                 : match only the retained ReLUs' sign pattern
#                              (linear-region assignment) via margin-softplus
DISTILL_MODE = "none"

# retrain-mode execution structure: per P, run DISTILL_MODES sequentially.
#   P=10: none → full → retained → symbol → P=11: none → ...
# each (P, mode) block finishes its MAX_TOTAL_PARALLEL-wide run before the
# next starts, so concurrency never exceeds MAX_TOTAL_PARALLEL.
# to run a single mode, set DISTILL_MODES = ["symbol"] etc.
# (DISTILL_MODE above is internal state overwritten to the current mode at
#  runtime; it is also the default read by make_retrain_summary_table.py.)
# DISTILL_MODES = ["none", "full_preactivation", "retained_preactivation", "symbol"]
DISTILL_MODES = ["none"]   # (unused in this launcher; see ABLATION_CONDITIONS)

# ── the 5 weighting-ablation conditions (run serially in this order) ────
# unique tags that cannot collide with existing (_ga0p1) runs; everything but the lambda rule matches the existing setup.
# 2026-09-16 redesign: each guidance mode is paired with its matched
# parent-state forcing (compare lambda rules on top of the best forcing):
#   retained guidance → readout + retained sync
#   full guidance     → full-state sync
#   symbol guidance   → readout sync
# the baseline (weighting=default grad_auto alpha=0.1, same forcing) is
# taken from the sync ablation's retained_sync_readout_retained_ga0p1_v1 /
# full_sync_fullstate_ga0p1_v1 / symbol_sync_readout_ga0p1_v1。
ABLATION_CSV = "summary_retrain_weighting_ablation_p10_sync_v1.csv"
SYNC_INTERVAL = 128
ABLATION_CONDITIONS = [
    dict(name="retained_coordeq_fixed_syncrr_v1",
         mode="retained_preactivation",
         weighting="coord_equal_fixed", auto=False, alpha=None,
         sync_scope="readout_retained",
         lam_rule="lambda = P_eff / N"),
    dict(name="retained_ga1p0_syncrr_v1", mode="retained_preactivation",
         weighting="grad_auto_alpha1", auto=True, alpha=1.0,
         sync_scope="readout_retained",
         lam_rule="lambda = 1.0 * g_out / (g_aux + 1e-12)"),
    dict(name="symbol_lam1_fixed_syncro_v1", mode="symbol",
         weighting="fixed_lambda1", auto=False, alpha=None,
         sync_scope="readout",
         lam_rule="lambda = 1.0"),
    dict(name="full_coordeq_fixed_syncfull_v1", mode="full_preactivation",
         weighting="coord_equal_fixed", auto=False, alpha=None,
         sync_scope="full",
         lam_rule="lambda = (M - N) / N"),
    dict(name="full_ga1p0_syncfull_v1", mode="full_preactivation",
         weighting="grad_auto_alpha1", auto=True, alpha=1.0,
         sync_scope="full",
         lam_rule="lambda = 1.0 * g_out / (g_aux + 1e-12)"),
]
CURRENT_COND = None   # set per block by main()

DISTILL_LAMBDA_FULL     = 1.0
DISTILL_LAMBDA_RETAINED = 1.0
DISTILL_LAMBDA_SYMBOL   = 1.0
SYMBOL_MARGIN           = 1.0

# ── auto-lambda calibration (initial gradient-norm ratio) ──────────
# the auxiliary-loss scale differs wildly between modes (full ~O(10^2) ...
# symbol ~O(1)), so a raw lambda=1.0 would let gradient scale dominate.
# When DISTILL_AUTO_LAMBDA=True, at the start of training
#   λ = α · ‖∇θ L_out‖ / (‖∇θ L_aux‖ + ε)
# is determined once from calibration batches (median) and kept fixed.
# This aligns every mode's initial auxiliary gradient to alpha x output gradient.
# False uses the manual lambdas above (DISTILL_LAMBDA_*); ignored for none.
DISTILL_AUTO_LAMBDA        = True
DISTILL_GRAD_RATIO_ALPHA   = 0.1
DISTILL_CALIBRATION_BATCHES = 5
DISTILL_GRAD_EPS           = 1e-12


def distill_autolambda_tag() -> str:
    """tag suffix of auto-lambda runs (e.g. '_ga0p1'); '' for manual runs.
    Must match the function of the same name in train_alrnn.py."""
    if not DISTILL_AUTO_LAMBDA or DISTILL_MODE == 'none':
        return ''
    return '_ga' + f"{DISTILL_GRAD_RATIO_ALPHA:g}".replace('.', 'p').replace('-', 'm')

# teacher trajectory cache: precompute one global teacher-forced rollout per
# source checkpoint, shared by all candidates / distill modes / retrain seeds.
# the launcher prepares unique sources serially before retraining starts, so
# parallel workers never race to generate the same cache.
TEACHER_CACHE_DIR           = Path("results/teacher_cache")
FORCE_REBUILD_TEACHER_CACHE = False   # debug only (applies to the pre-build phase)

# mode -> checkpoint/log name suffix (must match DISTILL_TAGS in
# train_alrnn.py; 'none' is empty = legacy naming)
DISTILL_TAGS = {
    'none':                    '',
    'full_preactivation':      '_distfullpre',
    'retained_preactivation':  '_distretpre',
    'symbol':                  '_distsym',
}

# max number of candidates run concurrently within one source seed.
# e.g. 6: each seed runs cand0-5 together -> then cand6-11 -> ...
MAX_PARALLEL_CANDIDATES = 6

# global cap on concurrent processes.
# this machine has 18 physical cores; training processes are limited to one
# thread each (OMP_NUM_THREADS=1 below; BLAS parallelism hurts the tiny
# M=20 model), so process count = cores used; match physical cores.
# note: torch (CUDA build) reserves several GB of virtual memory (commit)
# per process; too many processes fail with WinError 1455 (pagefile
# exhaustion; happened at 125 concurrent, fine after enlarging the pagefile).
# MAX_TOTAL_PARALLEL = 18
MAX_TOTAL_PARALLEL = 28
# MAX_TOTAL_PARALLEL = 5

# ── Run control ────────────────────────────────────────────
LOG_DIR         = Path("logs/aug_only_ratio_sweep")
# FORCE_RERUN     = True
FORCE_RERUN     = False # skip already-trained models
STOP_ON_FAILURE = False
DRY_RUN         = False

# ============================================================


# ── Derived conditions ───────────────────────────────────────

_ORIGINAL_CONDITION = {
    "name":     "original",
    "r_anchor": 0.0,
    "r_local":  0.0,
    "r_bridge": 0.0,
    "r_long":   1.0,   # ignored by the training script (data_mode=original)
}

def get_conditions():
    if DATA_MODE == "original":
        return [_ORIGINAL_CONDITION]
    elif DATA_MODE == "augmented":
        return RATIO_GRID
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE!r}  (must be 'original' or 'augmented')")


# ── Helpers ──────────────────────────────────────────────────

def bool_flag(flag_name: str, enabled: bool) -> list[str]:
    return [flag_name] if enabled else []


def is_successful_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "# Return code: 0" in txt


def make_model_tag(ratio_cfg: dict, n_interleave: int, p: int, seed: int) -> str:
    """Must match the tag-generation logic of train_alrnn.py."""
    if DATA_MODE == "original":
        return (f"chua_orig_nint{n_interleave}_bs{BATCH_EPISODES}"
                f"_sig{INPUT_SIGMA}_lr{LR_START:.0e}-{LR_END:.0e}"
                f"_tfp0_ramp{200}_hLR1_m20_p{p}_seed{seed}")
    else:
        rb  = ratio_cfg["r_bridge"]
        rl  = ratio_cfg["r_long"]
        ra  = ratio_cfg["r_anchor"]
        rlo = ratio_cfg["r_local"]
        return (f"chua_aug_only_a{ra:.2f}_l{rlo:.2f}_b{rb:.2f}_lo{rl:.2f}"
                f"_nint{n_interleave}_bs{BATCH_EPISODES}"
                f"_sig{INPUT_SIGMA}_lr{LR_START:.0e}-{LR_END:.0e}"
                f"_tfp0_ramp{200}_hLR1_m20_p{p}_seed{seed}")


def resolve_reduction_inputs(tag: str) -> tuple[Path, Path, dict]:
    """Resolve the source checkpoint / reduction spec needed for retraining.

    If either is missing, raise a clear error instead of silently skipping.
    Returns (source_ckpt_path, spec_path, spec_dict).
    """
    src_path  = Path("models") / f"{tag}.pth"
    spec_path = REDUCTION_SPEC_DIR / f"{tag}_minimal_symbol_reductions.json"
    if not src_path.exists():
        raise FileNotFoundError(
            f"[RETRAIN_REDUCED] source over-parameterized checkpoint missing: "
            f"{src_path}\n  -> run normal training first (RETRAIN_REDUCED=False).")
    if not spec_path.exists():
        raise FileNotFoundError(
            f"[RETRAIN_REDUCED] reduction spec missing: {spec_path}\n"
            f"  -> run run_graph_reduction_sweep.py (or graph_reduction_with_relu_pruning.py "
            f"--model_path {src_path}) to save the minimal symbol reduction "
            f"spec.\n"
            f"  -> models whose graph reduction was [NO-REDUCTION] (no valid reduced "
            f"candidate) have no spec; remove them from SOURCE_SEEDS.")
    with open(spec_path) as f:
        spec = json.load(f)
    if 'candidates' not in spec or 'minimal_num_clusters' not in spec:
        raise ValueError(
            f"[RETRAIN_REDUCED] {spec_path} is the old single-candidate format or invalid. "
            f"Re-run graph reduction to generate "
            f"minimal_symbol_reductions.json.")
    return src_path, spec_path, spec


def retrained_model_path(tag: str, spec: dict, candidate: dict,
                         retrain_seed: int) -> Path:
    """Save path of the retrained reduced model (matches train_alrnn.py naming).

    Including minimal_num_clusters / candidate_id / P_effective / retraining
    seed in the filename keeps multiple candidates x multiple retraining
    seeds of the same source model from overwriting each other.
    """
    return Path("models") / (
        f"{tag}_minsym{spec['minimal_num_clusters']}"
        f"_cand{candidate['candidate_id']}"
        f"_reducedp{candidate['P_effective']}"
        f"_{CURRENT_COND['name']}"
        f"_rseed{retrain_seed}.pth")


def missing_p_list(ratio_cfg: dict, n_interleave: int, seed: int) -> list[int]:
    """(normal mode) return the list of P whose model files are missing."""
    missing = []
    for p in P_LIST:
        tag = make_model_tag(ratio_cfg, n_interleave, p, seed)
        if not (Path("models") / f"{tag}.pth").exists():
            missing.append(p)
    return missing


def compute_global_min_selection(source_entries: list, target_p_effs: list) -> dict:
    """Select the global-minimum-num_clusters candidates across eligible sources.

    source_entries: [(source_seed, spec_dict, spec_path), ...]
        NOTE: pass only eligible sources (vis_fps == TARGET_FP_COUNT).
        Each spec stores its source's local-minimum candidates, but since
        global min <= every source's local min, a global-min candidate is
        always contained in its source's spec, so scanning the specs finds
        the global minimum correctly (a source with local min > global min
        cannot own a global-min candidate).

    Returns dict:
      global_min, all_candidates, global_min_candidates, selected,
      counts_by_p_eff_at_min, excluded_by_num_clusters
    selected is stably sorted by (source_seed, P_effective, candidate_id)
    (deterministic order independent of input/JSON order).
    Multiple candidates of one source are all kept (no tie-breaking).
    """
    all_cands = []
    for seed, spec, spec_path in source_entries:
        for c in spec['candidates']:
            all_cands.append({
                'source_seed':   int(seed),
                'candidate_id':  int(c['candidate_id']),
                'num_clusters':  int(c['num_clusters']),
                'P_original':    int(spec['P_original']),
                'P_effective':   int(c['P_effective']),
                'deleted_relu_local_indices':   list(c['deleted_relu_local_indices']),
                'retained_relu_local_indices':  list(c['remaining_relu_local_indices']),
                'spec_path':     str(spec_path),
            })
    if not all_cands:
        return {'global_min': None, 'all_candidates': [], 'global_min_candidates': [],
                'selected': [], 'counts_by_p_eff_at_min': {},
                'excluded_by_num_clusters': {}}

    global_min = min(c['num_clusters'] for c in all_cands)
    gm_cands = [c for c in all_cands if c['num_clusters'] == global_min]
    # P_eff filter only (target_p_effs=None disables it = keep every
    # global-min candidate). Ties are NOT further pruned by num_deleted /
    # edge_types etc. Filtered-out ones stay in metadata (global_min_candidates).
    if target_p_effs is None:
        selected = list(gm_cands)
    else:
        selected = [c for c in gm_cands if c['P_effective'] in target_p_effs]
    selected.sort(key=lambda c: (c['source_seed'], c['P_effective'],
                                 c['candidate_id']))

    counts_at_min = {}
    for c in gm_cands:
        counts_at_min[c['P_effective']] = counts_at_min.get(c['P_effective'], 0) + 1
    excluded_nc = {}
    for c in all_cands:
        if c['num_clusters'] != global_min:
            excluded_nc[c['num_clusters']] = excluded_nc.get(c['num_clusters'], 0) + 1

    return {'global_min': global_min,
            'all_candidates': all_cands,
            'global_min_candidates': gm_cands,
            'selected': selected,
            'counts_by_p_eff_at_min': counts_at_min,
            'excluded_by_num_clusters': excluded_nc}


def source_num_fp_nodes(ratio_cfg: dict, n_interleave: int, p: int,
                        seed: int, spec: dict) -> tuple[int | None, str]:
    """Number of real fixed-point nodes present on the graph at reduction time.

    The training-time vis_fps (screening) and graph reduction use different
    trajectory protocols (TF warmup + 40000 steps + transient removal vs
    free-run 10000 steps), so a source that passed screening may not visit
    TARGET_FP_COUNT fixed-point regions on the reduction trajectory. Its
    candidates cannot represent every fixed point and must be excluded.

    Priority:
      1. 'num_real_fp_nodes' in the spec (new-format specs only)
      2. num_fixed_nodes of a valid hierarchy-CSV row (fallback for old
         specs; the CSV path follows run_graph_reduction_sweep.py --option naming)
      3. unknown -> (None, 'unknown')
    """
    if 'num_real_fp_nodes' in spec:
        return int(spec['num_real_fp_nodes']), 'spec'
    stem = f"m20_p{p}_{ratio_cfg['name']}_nint{n_interleave}_seed{seed}"
    csv_path = Path(f"results/relu_hierarchy/{stem}/relu_prune_hierarchy_{stem}.csv")
    if csv_path.exists():
        import csv as _csv
        vals = set()
        with open(csv_path, newline='') as f:
            for row in _csv.DictReader(f):
                if str(row.get('is_rejected', '')).lower() in ('false', '0', ''):
                    v = row.get('num_fixed_nodes')
                    if v not in (None, ''):
                        vals.add(int(float(v)))
        if vals:
            return max(vals), 'csv'
    return None, 'unknown'


def gather_eligible_source_entries(ratio_cfg: dict, n_interleave: int,
                                   p: int) -> tuple[list, list, list, list]:
    """Collect eligible sources' specs based on the screening metadata.

    Eligibility always consults the source screening JSON (vis_fps ==
    TARGET_FP_COUNT). Mere existence of a spec is NOT enough — this keeps
    candidates from stale specs of unsuccessful sources out of retraining.

    Sources whose reduction-time real-FP node count (source_num_fp_nodes)
    differs from TARGET_FP_COUNT are excluded too (fp_deficient).
    Heuristic when the node count is unavailable:
    num_clusters >= num_fixed_nodes always holds, so
    minimal_num_clusters < TARGET_FP_COUNT is sure proof of missing FP nodes.

    Returns (source_entries, ineligible_seeds, missing_spec_seeds,
             fp_deficient)  — fp_deficient is a list of (seed, n_fp, verdict source)
    """
    screening_path = (REDUCTION_SPEC_DIR /
                      f"source_screening_{ratio_cfg['name']}_nint{n_interleave}"
                      f"_m20_p{p}.json")
    if not screening_path.exists():
        raise FileNotFoundError(
            f"[GLOBAL_MIN_ONLY] source screening metadata missing: "
            f"{screening_path}\n  -> run run_graph_reduction_sweep.py first "
            f"(FILTER_SOURCE_BY_VIS_FPS=True).")
    with open(screening_path) as f:
        successful = set(json.load(f)['successful_seeds'])

    entries, ineligible, missing_spec, fp_deficient = [], [], [], []
    for seed in SOURCE_SEEDS:
        tag = make_model_tag(ratio_cfg, n_interleave, p, seed)
        spec_path = REDUCTION_SPEC_DIR / f"{tag}_minimal_symbol_reductions.json"
        if seed not in successful:
            ineligible.append(seed)          # never adopt stale specs
            continue
        if not spec_path.exists():
            missing_spec.append(seed)
            continue
        with open(spec_path) as f:
            spec = json.load(f)
        n_fp, fp_src = source_num_fp_nodes(ratio_cfg, n_interleave, p, seed, spec)
        if n_fp is not None:
            if n_fp != TARGET_FP_COUNT:
                fp_deficient.append((seed, n_fp, fp_src))
                continue
        elif int(spec['minimal_num_clusters']) < TARGET_FP_COUNT:
            fp_deficient.append((seed, int(spec['minimal_num_clusters']),
                                 'heuristic(minsym<target)'))
            continue
        entries.append((seed, spec, spec_path))
    return entries, ineligible, missing_spec, fp_deficient


def build_global_min_selection(ratio_cfg: dict, n_interleave: int) -> dict:
    """Run the global-min selection for each P in P_LIST; print and save results.

    Returns allowed_by_p: {p: {source_seed: set(candidate_id)}}
    """
    allowed_by_p = {}
    for p in P_LIST:
        entries, ineligible, missing_spec, fp_deficient = \
            gather_eligible_source_entries(ratio_cfg, n_interleave, p)
        if missing_spec:
            raise FileNotFoundError(
                f"[GLOBAL_MIN_ONLY] eligible sources without a reduction spec "
                f"exist (P={p}): seeds={missing_spec}\n"
                f"  -> run run_graph_reduction_sweep.py, or drop the "
                f"[NO-REDUCTION] seeds from SOURCE_SEEDS.")
        sel = compute_global_min_selection(entries, TARGET_P_EFFS)

        print("\n" + "=" * 60)
        print("GLOBAL MINIMAL SYMBOL RETRAINING SELECTION")
        print("=" * 60)
        print(f"condition / nint / P_original : {ratio_cfg['name']} / "
              f"{n_interleave} / {p}")
        print(f"Eligible source models        : {len(entries)}"
              + (f"  (ineligible: {ineligible})" if ineligible else ""))
        if fp_deficient:
            print(f"FP-deficient sources excluded : "
                  + ", ".join(f"seed {s} (fp_nodes={n}, via {src})"
                              for s, n, src in fp_deficient))
        print(f"Total source-local candidates : {len(sel['all_candidates'])}")
        print()
        print(f"Global minimum num_clusters   : {sel['global_min']}")
        print(f"Target P_effective            : "
              + ("ALL (no P_eff filter)" if TARGET_P_EFFS is None
                 else str(TARGET_P_EFFS)))
        print()
        print("Candidates with global min:")
        for pe in sorted(sel['counts_by_p_eff_at_min']):
            print(f"  P_eff={pe} : {sel['counts_by_p_eff_at_min'][pe]}")
        print()
        print("Selected for retraining:")
        _sel_p_effs = (sorted(sel['counts_by_p_eff_at_min'])
                       if TARGET_P_EFFS is None else TARGET_P_EFFS)
        for pe in _sel_p_effs:
            print(f"  P_eff={pe} : "
                  f"{sum(1 for c in sel['selected'] if c['P_effective'] == pe)}")
        print(f"  TOTAL   : {len(sel['selected'])}")
        if sel['excluded_by_num_clusters']:
            print()
            print("Excluded larger-symbol candidates:")
            for nc in sorted(sel['excluded_by_num_clusters']):
                print(f"  num_clusters={nc} : {sel['excluded_by_num_clusters'][nc]}")
        print("=" * 60)
        print("Selected candidates:")
        for c in sel['selected']:
            print(f"  src={c['source_seed']:>2} cand={c['candidate_id']:>2} "
                  f"nc={c['num_clusters']} P_eff={c['P_effective']} "
                  f"deleted={c['deleted_relu_local_indices']}")

        if not sel['selected']:
            raise SystemExit(
                f"[GLOBAL_MIN_ONLY] No candidates satisfy "
                f"TARGET_P_EFFS={TARGET_P_EFFS} "
                f"(global_min={sel['global_min']}, "
                f"P_eff at min: {sel['counts_by_p_eff_at_min']}). "
                f"No automatic fallback to other P_eff values.")

        # save a machine-readable selection JSON
        out = {
            'condition': ratio_cfg['name'],
            'n_interleave': n_interleave,
            'P_original': p,
            'target_fp_count': TARGET_FP_COUNT,
            'global_min_num_clusters': sel['global_min'],
            'target_p_effective': ('all' if TARGET_P_EFFS is None
                                   else TARGET_P_EFFS),
            'num_eligible_sources': len(entries),
            'ineligible_sources': ineligible,
            'fp_deficient_sources': [
                {'seed': s, 'num_fp_nodes': n, 'detected_via': src}
                for s, n, src in fp_deficient],
            'num_all_candidates': len(sel['all_candidates']),
            'num_global_min_candidates': len(sel['global_min_candidates']),
            'num_selected_candidates': len(sel['selected']),
            'counts_by_p_effective': {str(k): v for k, v
                                      in sorted(sel['counts_by_p_eff_at_min'].items())},
            'excluded_by_num_clusters': {str(k): v for k, v
                                         in sorted(sel['excluded_by_num_clusters'].items())},
            'selected_candidates': sel['selected'],
        }
        out_path = (REDUCTION_SPEC_DIR /
                    f"global_min_retrain_selection_{ratio_cfg['name']}"
                    f"_nint{n_interleave}_m20_p{p}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Selection JSON → {out_path}")

        allowed = {}
        for c in sel['selected']:
            allowed.setdefault(c['source_seed'], set()).add(c['candidate_id'])
        allowed_by_p[p] = allowed
    return allowed_by_p


def plan_retrain_candidates(ratio_cfg: dict, n_interleave: int, seed: int,
                            retrain_seed: int,
                            allowed_by_p: dict | None = None) -> list[dict]:
    """(retrain mode) build the per-candidate run/skip plan.

    For each P, inspect the spec's candidates and run only those whose
    retrained checkpoint does not exist yet (all candidates if FORCE_RERUN).
    resolve_reduction_inputs errors out on missing checkpoints/specs.

    allowed_by_p: {p: {source_seed: set(candidate_id)}} (for GLOBAL_MIN_ONLY).
      When given, only candidates in the selection are planned.
      A P whose seed is absent from the selection is left out of the plan
      (larger-symbol candidates outside the selection are not run, but
       their specs / old checkpoints are kept, not deleted).

    Returns: list of per-P dicts
      {p, tag, src, spec_path, spec, all_ids, run_ids, skip_ids}
    """
    plan = []
    for p in P_LIST:
        if allowed_by_p is not None:
            allowed = allowed_by_p.get(p, {}).get(seed)
            if not allowed:
                continue    # this source is outside the global-min selection
        else:
            allowed = None
        tag = make_model_tag(ratio_cfg, n_interleave, p, seed)
        src, spec_path, spec = resolve_reduction_inputs(tag)
        cands = [c for c in spec['candidates']
                 if allowed is None or int(c['candidate_id']) in allowed]
        all_ids = [int(c['candidate_id']) for c in cands]
        if FORCE_RERUN:
            run_ids = list(all_ids)
        else:
            run_ids = [int(c['candidate_id']) for c in cands
                       if not retrained_model_path(tag, spec, c, retrain_seed).exists()]
        plan.append({
            'p': p, 'tag': tag, 'src': src, 'spec_path': spec_path, 'spec': spec,
            'all_ids': all_ids, 'run_ids': run_ids,
            'skip_ids': [i for i in all_ids if i not in run_ids],
        })
    return plan


def build_command(ratio_cfg: dict, n_interleave: int, seed: int, p_list: list[int],
                  retrain_seed: int | None = None,
                  retrain_plan: list[dict] | None = None) -> list[str]:
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,

        "--data_mode", DATA_MODE,

        "--P_list", *[str(p) for p in p_list],

        "--n_interleave",    str(n_interleave),

        "--num_epochs",      str(NUM_EPOCHS),
        "--steps_per_epoch", str(STEPS_PER_EPOCH),
        "--batch_episodes",  str(BATCH_EPISODES),

        "--segment_len",    str(SEGMENT_LEN),
        "--max_bridge_len", str(MAX_BRIDGE_LEN),

        "--alpha", str(ALPHA),

        "--r_anchor", str(ratio_cfg["r_anchor"]),
        "--r_local",  str(ratio_cfg["r_local"]),
        "--r_bridge", str(ratio_cfg["r_bridge"]),
        "--r_long",   str(ratio_cfg["r_long"]),

        "--input_sigma", str(INPUT_SIGMA),
        "--state_sigma", str(STATE_SIGMA),

        "--lr_start", str(LR_START),
        "--lr_end",   str(LR_END),

        "--ssi", str(SSI),

        "--target_fp_count", str(TARGET_FP_COUNT),
        "--fp_count_weight", str(FP_COUNT_WEIGHT),

        "--seed", str(seed),
    ]

    if DATA_MODE == "original":
        cmd += ["--raw_data_path", RAW_DATA_PATH]

    if RETRAIN_REDUCED:
        # pass, in p_list order, the source checkpoint / reduction spec /
        # candidate ids to run (comma-separated).
        # --seed is the source seed (tag base); --retrain_seed sets the training RNG
        assert retrain_plan is not None
        items = [it for it in retrain_plan if it['p'] in p_list]
        assert [it['p'] for it in items] == list(p_list)
        cmd += ["--retrain_reduced",
                "--retrain_seed", str(retrain_seed),
                "--source_checkpoints", *[str(it['src']) for it in items],
                "--reduction_specs",    *[str(it['spec_path']) for it in items],
                "--reduction_candidate_ids",
                *[",".join(map(str, it['run_ids'])) for it in items],
                "--distill_mode", CURRENT_COND["mode"],
                "--distill_lambda_full",     str(DISTILL_LAMBDA_FULL),
                "--distill_lambda_retained", str(DISTILL_LAMBDA_RETAINED),
                "--distill_lambda_symbol",   str(DISTILL_LAMBDA_SYMBOL),
                "--symbol_margin",           str(SYMBOL_MARGIN),
                "--teacher_cache_dir",       str(TEACHER_CACHE_DIR),
                "--distill_grad_ratio_alpha",
                str(CURRENT_COND["alpha"] if CURRENT_COND["auto"] else 0.1),
                "--distill_calibration_batches", str(DISTILL_CALIBRATION_BATCHES),
                "--distill_grad_eps",            str(DISTILL_GRAD_EPS),
                "--distill_weighting", CURRENT_COND["weighting"],
                "--distill_sync_scope",    CURRENT_COND["sync_scope"],
                "--distill_sync_interval", str(SYNC_INTERVAL),
                "--distill_name_tag",  CURRENT_COND["name"],
                "--ablation_csv",      ABLATION_CSV,
                "--ablation_sync_cols"]
        if CURRENT_COND["auto"]:
            cmd += ["--distill_auto_lambda"]
        # --force_rebuild_teacher_cache applies to the pre-build phase only;
        # do not pass it to subprocesses (prevents mutual regeneration)

    cmd += bool_flag("--lr_fixed",         LR_FIXED)
    cmd += bool_flag("--no_periodic_eval", NO_PERIODIC_EVAL)

    return cmd


def make_log_path(ratio_name: str, n_interleave: int, seed: int,
                  retrain_seed: int | None = None,
                  p: int | None = None,
                  candidate_id: int | None = None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if RETRAIN_REDUCED:
        return (LOG_DIR / f"{ratio_name}_retrain__nint{n_interleave}"
                          f"__srcseed{seed}_rseed{retrain_seed}"
                          f"_p{p}_cand{candidate_id}"
                          f"_{CURRENT_COND['name']}.log")
    return LOG_DIR / f"{ratio_name}__nint{n_interleave}__seed{seed}.log"


def _write_text_as_bytes(f, text: str) -> None:
    f.write(text.encode("utf-8", errors="replace"))
    f.flush()


def _reader_thread(proc: subprocess.Popen, log_file, prefix: str) -> None:
    """Thread that tees stdout to the log while showing [PROGRESS] lines only."""
    assert proc.stdout is not None
    buf = b''
    while True:
        chunk = proc.stdout.read(256)
        if not chunk:
            break
        log_file.write(chunk)
        log_file.flush()
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            txt = line.decode('utf-8', errors='replace')
            if '[PROGRESS]' in txt:
                print(f"{prefix} {txt}", flush=True)
    if buf:
        txt = buf.decode('utf-8', errors='replace')
        if '[PROGRESS]' in txt:
            print(f"{prefix} {txt}", flush=True)


def spawn_one(ratio_cfg: dict, n_interleave: int, seed: int,
              retrain_seed: int | None = None,
              retrain_item: dict | None = None,
              candidate_id: int | None = None):
    """Spawn the process and return immediately (non-blocking); None if skipped.

    Normal mode: one call = one seed (covers all of P_LIST).
    RETRAIN_REDUCED=True: one call = one (source seed, candidate, retrain seed).
      Candidate expansion and skip decisions happen in main()
      (plan_retrain_candidates); this receives one candidate to run
      (retrain_item + candidate_id). All seed x candidate runs launch in parallel.
    """
    ratio_name = ratio_cfg["name"]

    if RETRAIN_REDUCED:
        assert retrain_item is not None and candidate_id is not None
        p = retrain_item['p']
        log_path = make_log_path(ratio_name, n_interleave, seed, retrain_seed,
                                 p, candidate_id)
        seed_label = (f"srcseed={seed} rseed={retrain_seed} "
                      f"P={p} cand={candidate_id}")
        plan_one = [{**retrain_item, 'run_ids': [candidate_id]}]
        cmd = build_command(ratio_cfg, n_interleave, seed, [p], retrain_seed,
                            plan_one)
        print(f"[LAUNCH] {seed_label}  log={log_path}")
        if DRY_RUN:
            print(f"  cmd: {' '.join(cmd)}")
            return None
    else:
        log_path   = make_log_path(ratio_name, n_interleave, seed)
        seed_label = f"seed={seed}"
        p_needed = (P_LIST if FORCE_RERUN
                    else missing_p_list(ratio_cfg, n_interleave, seed))
        if not p_needed:
            print(f"[SKIP models exist] {seed_label} nint={n_interleave}  all P done")
            return None
        if p_needed != P_LIST:
            print(f"[PARTIAL] {seed_label} nint={n_interleave}  missing P={p_needed}")

        cmd = build_command(ratio_cfg, n_interleave, seed, p_needed)

        print("=" * 100)
        print(f"data_mode    : {DATA_MODE}")
        print(f"ratio        : {ratio_name}  "
              f"(a={ratio_cfg['r_anchor']}, l={ratio_cfg['r_local']}, "
              f"b={ratio_cfg['r_bridge']}, lo={ratio_cfg['r_long']})")
        print(f"n_interleave : {n_interleave}")
        print(f"P_list       : {p_needed}  (requested={P_LIST})")
        print(f"seed         : {seed_label}")
        print(f"log          : {log_path}")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 100)

        if DRY_RUN:
            return None

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # limit each process to one thread. torch starts as many intra-op threads
    # as cores by default, but parallel BLAS gains nothing for the tiny M=20
    # model and processes fighting over cores slows everything down.
    env["OMP_NUM_THREADS"]      = "1"
    env["MKL_NUM_THREADS"]      = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"]  = "1"

    start_time = datetime.datetime.now()
    log_file   = open(log_path, "wb")

    _write_text_as_bytes(log_file, f"# Started at: {start_time}\n")
    _write_text_as_bytes(log_file, f"# data_mode: {DATA_MODE}\n")
    _write_text_as_bytes(log_file, f"# ratio: {ratio_name}\n")
    _write_text_as_bytes(log_file, f"# r_anchor={ratio_cfg['r_anchor']}  r_local={ratio_cfg['r_local']}  "
                                    f"r_bridge={ratio_cfg['r_bridge']}  r_long={ratio_cfg['r_long']}\n")
    _write_text_as_bytes(log_file, f"# n_interleave: {n_interleave}\n")
    _write_text_as_bytes(log_file, f"# P_list: {P_LIST}\n")
    _write_text_as_bytes(log_file, f"# seed: {seed}\n")
    if RETRAIN_REDUCED:
        _write_text_as_bytes(log_file, f"# retrain_seed: {retrain_seed}\n")
        _write_text_as_bytes(log_file, f"# candidate_id: {candidate_id}\n")
    _write_text_as_bytes(log_file, "# Command:\n")
    _write_text_as_bytes(log_file, " ".join(cmd) + "\n\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    seed_tag = (f"s{seed}r{retrain_seed}c{candidate_id}"
                if RETRAIN_REDUCED else f"s{seed}")
    prefix = f"[{ratio_name}/nint{n_interleave}/{seed_tag}]"
    t = threading.Thread(target=_reader_thread, args=(proc, log_file, prefix), daemon=True)
    t.start()
    return (proc, log_file, log_path, start_time, ratio_cfg, n_interleave,
            seed_label, t)


def wait_one(handle) -> tuple[str, int, str, int]:
    proc, log_file, log_path, start_time, ratio_cfg, n_interleave, seed_label, t = handle
    ratio_name = ratio_cfg["name"]

    t.join()
    return_code = proc.wait()

    end_time = datetime.datetime.now()
    _write_text_as_bytes(log_file, f"\n# Finished at: {end_time}\n")
    _write_text_as_bytes(log_file, f"# Elapsed: {end_time - start_time}\n")
    _write_text_as_bytes(log_file, f"# Return code: {return_code}\n")
    log_file.close()

    elapsed = end_time - start_time
    if return_code == 0:
        print(f"[OK]     {ratio_name}, nint={n_interleave}, {seed_label}  ({elapsed})")
    else:
        print(f"[FAILED] {ratio_name}, nint={n_interleave}, {seed_label}  "
              f"(return={return_code})  log={log_path}")

    return ratio_name, n_interleave, seed_label, return_code


def run_retrain_condition(ratio_cfg: dict, n_interleave: int,
                          seed_pairs: list, failed: list) -> None:
    """Run one full retraining block (selection -> plan -> teacher-cache
    preparation -> parallel execution -> join) with the current module config.

    Run/skip is decided per candidate.
    Parallel structure: all seeds in parallel (1 seed = 1 worker thread);
    within each seed, candidates run in waves of MAX_PARALLEL_CANDIDATES;
    the whole thing is capped by the MAX_TOTAL_PARALLEL semaphore.
    """
    results = []
    results_lock = threading.Lock()
    print_lock = threading.Lock()   # avoid interleaved prints across threads
    # global cap on concurrent processes (WinError 1455 mitigation).
    # each run acquires a slot before spawning and releases it on exit.
    run_slots = threading.Semaphore(MAX_TOTAL_PARALLEL)

    def _one_run(seed, retrain_seed, it, cid):
        """Complete one run as acquire -> spawn -> wait -> release.

        Important: no thread ever acquires another slot while holding one,
        so the semaphore cannot deadlock in a circular wait.
        (The old implementation had seed threads wait for the next slot
        while holding acquired slots within a wave; once all slots were held
        by mid-wave threads nobody could reach release, and the launcher
        hung forever even after the children finished.)
        """
        run_slots.acquire()   # wait for a free global slot (holding nothing here)
        try:
            with print_lock:
                h = spawn_one(ratio_cfg, n_interleave, seed, retrain_seed,
                              retrain_item=it, candidate_id=cid)
            if h is None:         # DRY_RUN etc.
                return
            r = wait_one(h)
            with results_lock:
                results.append(r)
        except Exception as e:
            # avoid silent thread deaths (unattended operation); the run just
            # counts as incomplete and is retried on the next launch.
            with print_lock:
                print(f"[RUN-ERROR] srcseed={seed} rseed={retrain_seed} "
                      f"cand={cid}: {type(e).__name__}: {e}")
        finally:
            run_slots.release()

    def _run_seed_waves(seed, retrain_seed, runs):
        for i in range(0, len(runs), MAX_PARALLEL_CANDIDATES):
            wave = runs[i:i + MAX_PARALLEL_CANDIDATES]
            wave_ids = [cid for _, cid in wave]
            with print_lock:
                print(f"[WAVE] srcseed={seed} rseed={retrain_seed}: "
                      f"launching candidates {wave_ids} "
                      f"({i + len(wave)}/{len(runs)}, "
                      f"may wait for free slots of the global cap {MAX_TOTAL_PARALLEL})")
            # each run in the wave does acquire->spawn->wait->release in its
            # own thread; the next wave starts after all finish (order kept)
            ths = [threading.Thread(target=_one_run,
                                    args=(seed, retrain_seed, it, cid),
                                    daemon=True)
                   for it, cid in wave]
            for th in ths:
                th.start()
            for th in ths:
                th.join()

    # ── candidate selection stage ──
    # GLOBAL_MIN_ONLY: pick target candidates via the global minimum
    # num_clusters across all eligible sources + the P_eff filter.
    # This selection is independent of DISTILL_MODE (every distillation
    # method compares the same candidate set fairly).
    allowed_by_p = None
    if GLOBAL_MIN_ONLY:
        allowed_by_p = build_global_min_selection(ratio_cfg, n_interleave)

    # ── 1) collect the execution plan ──
    seed_runs = []
    total_runs = 0
    for seed, retrain_seed in seed_pairs:
        if (allowed_by_p is not None
                and not any(seed in allowed_by_p.get(p, {})
                            for p in P_LIST)):
            print(f"[NOT SELECTED] srcseed={seed}: global-min "
                  f"no candidates selected (ineligible or "
                  f"larger-symbol only) — skip")
            continue
        plan = plan_retrain_candidates(ratio_cfg, n_interleave,
                                       seed, retrain_seed,
                                       allowed_by_p=allowed_by_p)
        for it in plan:
            print(f"[RETRAIN plan] srcseed={seed} rseed={retrain_seed} "
                  f"P={it['p']}: selected_candidates={len(it['all_ids'])}  "
                  f"scheduled_runs={len(it['run_ids'])} {it['run_ids']}  "
                  f"skip={it['skip_ids']}")
        runs = [(it, cid) for it in plan for cid in it['run_ids']]
        if not runs:
            print(f"[SKIP models exist] srcseed={seed} "
                  f"rseed={retrain_seed} nint={n_interleave}  "
                  f"all candidates done")
            continue
        total_runs += len(runs)
        seed_runs.append((seed, retrain_seed, runs))

    # ── 2) pre-build teacher caches (distill modes only) ──
    # build serially, once per source checkpoint, to structurally rule out
    # generation races between parallel workers.
    # every subsequent run is guaranteed a cache HIT.
    if DISTILL_MODE != 'none' and seed_runs and not DRY_RUN:
        import importlib.util as _ilu
        import numpy as _np
        _s = _ilu.spec_from_file_location('_ta_cache', TRAIN_SCRIPT)
        _ta = _ilu.module_from_spec(_s)
        _s.loader.exec_module(_ta)
        _raw = _np.load(RAW_DATA_PATH).astype(_np.float32)
        _srcs = {}
        for seed, retrain_seed, runs in seed_runs:
            for it, cid in runs:
                _srcs[(it['p'], seed)] = str(it['src'])
        print(f"\n[Teacher cache] preparing {len(_srcs)} unique source(s) "
              f"(dir={TEACHER_CACHE_DIR})")
        for (p, seed), src in sorted(_srcs.items()):
            _t0 = datetime.datetime.now()
            _, _, _, _, _hit = _ta.build_or_load_teacher_cache(
                src, RAW_DATA_PATH, _raw,
                20, p, _raw.shape[-1],
                cache_dir=str(TEACHER_CACHE_DIR),
                force_rebuild=FORCE_REBUILD_TEACHER_CACHE,
                verbose=False)
            _dt = (datetime.datetime.now() - _t0).total_seconds()
            print(f"  src={seed} P={p}: "
                  + ("HIT" if _hit else f"BUILD ({_dt:.1f}s)"))

    # ── 3) parallel launch ──
    seed_threads = []
    for seed, retrain_seed, runs in seed_runs:
        th = threading.Thread(target=_run_seed_waves,
                              args=(seed, retrain_seed, runs),
                              daemon=True)
        seed_threads.append(th)

    print(f"\n  total retraining runs scheduled: {total_runs}  "
          f"(peak parallel ≈ {len(seed_threads)} seeds × "
          f"{MAX_PARALLEL_CANDIDATES} candidates)")
    for th in seed_threads:
        th.start()
    for th in seed_threads:
        th.join()

    for ratio_name, nint, seed_label, ret in results:
        if ret != 0:
            failed.append((ratio_name, nint, seed_label, ret))
    if failed and STOP_ON_FAILURE:
        print("\nSTOP_ON_FAILURE=True (retrain mode: checked after all runs):")
        for item in failed:
            print(item)
        raise SystemExit(1)


def ablation_preflight() -> None:
    """Collision check + dry-run config table + fixed lambdas of representative candidates."""
    import glob as _glob
    print("=" * 72)
    print("WEIGHTING x MATCHED-FORCING ABLATION (P_parent=10) — "
          "5 conditions, run in order")
    print("=" * 72)
    # 1) tag-collision check. Existing outputs under this ablation's own
    #    _v1 tags are its OWN completed runs: report resume, not abort.
    for c in ABLATION_CONDITIONS:
        hits = _glob.glob(f"models/*_{c['name']}_*")
        if hits:
            print(f"[RESUME] '{c['name']}': {len(hits)} model(s) exist — "
                  f"completed runs are skipped; only unfinished runs execute")
    _csv = Path("results/aug_only") / ABLATION_CSV
    print(f"ablation CSV      = {_csv}  (exists={_csv.exists()})")
    # 2) config table
    hdr = (f"{'condition':>34} {'guidance_mode':>22} {'weighting':>18} "
           f"{'auto':>5} {'alpha':>6} {'forcing':>17}  lambda rule")
    print(hdr); print("-" * len(hdr))
    for c in ABLATION_CONDITIONS:
        print(f"{c['name']:>34} {c['mode']:>22} {c['weighting']:>18} "
              f"{str(c['auto']):>5} {str(c['alpha']):>6} "
              f"{c['sync_scope']:>17}  {c['lam_rule']}")
    print(f"sync interval      = {SYNC_INTERVAL}  "
          "(source = parent_autonomous_cache, shared by all conditions)")
    # 3) fixed lambda of representative candidates (from the selection JSON)
    sel_files = sorted(_glob.glob(
        "results/relu_hierarchy/retraining_specs/"
        "global_min_retrain_selection_*_m20_p10.json"))
    if sel_files:
        with open(sel_files[0]) as f:
            sel = json.load(f)
        cands = sel.get("selected_candidates", [])
        N_ = 3
        M_ = 20
        print(f"\nselected candidates: {len(cands)} "
              f"(must equal the previous P=10 comparison)")
        print("representative fixed lambdas:")
        seen_pe = set()
        for c in cands:
            pe = int(c.get("p_effective", c.get("P_effective")))
            if pe in seen_pe:
                continue
            seen_pe.add(pe)
            print(f"  seed{c.get('source_seed', c.get('seed'))} "
                  f"cand{c['candidate_id']} P_eff={pe}: "
                  f"retained_coordeq lambda={pe / N_:.6g}, "
                  f"full_coordeq lambda={(M_ - N_) / N_:.6g}, "
                  f"symbol fixed lambda=1.0")
    print("=" * 72)


def main() -> None:
    global P_LIST, DISTILL_MODE, CURRENT_COND
    ablation_preflight()
    _orig_p_list = list(P_LIST)
    _orig_distill_mode = DISTILL_MODE

    if not Path(TRAIN_SCRIPT).exists():
        raise FileNotFoundError(f"{TRAIN_SCRIPT} not found.")

    conditions = get_conditions()

    # one run = (seed, retrain_seed); normal mode has retrain_seed=None.
    if RETRAIN_REDUCED:
        seed_pairs = [(s, r) for s in SOURCE_SEEDS for r in RETRAIN_SEEDS]
    else:
        seed_pairs = [(s, None) for s in SEEDS]

    n_total = len(conditions) * len(N_INTERLEAVE_LIST) * len(seed_pairs)

    print("\nExperiment grid")
    print("-" * 80)
    print(f"TRAIN_SCRIPT      = {TRAIN_SCRIPT}")
    print(f"DATA_MODE         = {DATA_MODE}")
    print(f"CONDITIONS        = {[c['name'] for c in conditions]}")
    print(f"N_INTERLEAVE_LIST = {N_INTERLEAVE_LIST}")
    print(f"P_LIST            = {P_LIST}")
    if RETRAIN_REDUCED:
        print(f"SOURCE_SEEDS      = {SOURCE_SEEDS}")
        print(f"RETRAIN_SEEDS     = {RETRAIN_SEEDS}")
    else:
        print(f"SEEDS             = {SEEDS}")
    print(f"NUM_EPOCHS        = {NUM_EPOCHS}")
    print(f"STEPS_PER_EPOCH   = {STEPS_PER_EPOCH}")
    print(f"BATCH_EPISODES    = {BATCH_EPISODES}")
    print(f"TOTAL RUNS        = {n_total}  (each covers P_list={P_LIST})")
    if RETRAIN_REDUCED:
        print(f"PARALLELISM       = parallel over seeds x waves of "
              f"{MAX_PARALLEL_CANDIDATES} candidates per seed, "
              f"global cap {MAX_TOTAL_PARALLEL} processes (WinError 1455 guard)")
    else:
        print(f"PARALLELISM       = {len(seed_pairs)} runs per condition×nint (simultaneous)")
    print(f"RETRAIN_REDUCED   = {RETRAIN_REDUCED}")
    if RETRAIN_REDUCED:
        print(f"REDUCTION_SPEC_DIR= {REDUCTION_SPEC_DIR}")
        print(f"GLOBAL_MIN_ONLY   = {GLOBAL_MIN_ONLY}")
        if GLOBAL_MIN_ONLY:
            print(f"TARGET_P_EFFS     = {TARGET_P_EFFS}")
        print(f"DISTILL_MODES     = {DISTILL_MODES}")
        print(f"SWEEP STRUCTURE   = per P, run DISTILL_MODES sequentially "
              f"({len(_orig_p_list)} P × {len(DISTILL_MODES)} modes = "
              f"{len(_orig_p_list) * len(DISTILL_MODES)} blocks, "
              f"each block at most {MAX_TOTAL_PARALLEL} parallel)")
        print(f"DISTILL_LAMBDAS   = full:{DISTILL_LAMBDA_FULL} "
              f"retained:{DISTILL_LAMBDA_RETAINED} "
              f"symbol:{DISTILL_LAMBDA_SYMBOL} (margin {SYMBOL_MARGIN})")
        print(f"DISTILL_AUTO_LAMBDA = {DISTILL_AUTO_LAMBDA}"
              + (f"  (alpha={DISTILL_GRAD_RATIO_ALPHA}, "
                 f"batches={DISTILL_CALIBRATION_BATCHES}) — "
                 f"manual lambda ignored; auto-calibrated from gradient ratios"
                 if DISTILL_AUTO_LAMBDA else "  (using manual lambda)"))
    print(f"LOG_DIR           = {LOG_DIR}")
    print(f"FORCE_RERUN       = {FORCE_RERUN}")
    print(f"DRY_RUN           = {DRY_RUN}")
    print("-" * 80)

    failed  = []
    run_idx = 0

    for n_interleave in N_INTERLEAVE_LIST:
        for ratio_cfg in conditions:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"[condition {run_idx}/{len(conditions) * len(N_INTERLEAVE_LIST)}]  "
                  f"{ratio_cfg['name']}  nint={n_interleave}")
            if RETRAIN_REDUCED:
                print(f"  sources={SOURCE_SEEDS} × rseeds={RETRAIN_SEEDS}, "
                      f"candidates per seed in waves of {MAX_PARALLEL_CANDIDATES}")
            else:
                print(f"  launching {len(seed_pairs)} seeds in parallel: {SEEDS}")
            print(f"{'='*60}")

            if RETRAIN_REDUCED:
                # serial P x DISTILL_MODE sweep:
                #   for P in P_LIST: for mode in DISTILL_MODES: run the block
                # each (P, mode) block finishes before the next starts, so
                # concurrency stays <= MAX_TOTAL_PARALLEL (no parallel modes).
                # the teacher cache is BUILT by the first distill mode per P;
                # later modes HIT it.
                # weighting ablation: run the 5 conditions serially in ABLATION_CONDITIONS order
                n_blocks = len(ABLATION_CONDITIONS)
                for blk, _cond in enumerate(ABLATION_CONDITIONS, 1):
                    P_LIST = [10]
                    DISTILL_MODE = _cond["mode"]
                    CURRENT_COND = _cond
                    print(f"\n{'#' * 60}")
                    print(f"# block {blk}/{n_blocks}:  P_original=10  "
                          f"condition={_cond['name']}")
                    print(f"#   mode={_cond['mode']}  "
                          f"weighting={_cond['weighting']}  "
                          f"auto={_cond['auto']}  alpha={_cond['alpha']}")
                    print(f"{'#' * 60}")
                    run_retrain_condition(ratio_cfg, n_interleave,
                                          seed_pairs, failed)
                P_LIST = _orig_p_list
                DISTILL_MODE = _orig_distill_mode
            else:
                handles = []
                for seed, retrain_seed in seed_pairs:
                    h = spawn_one(ratio_cfg, n_interleave, seed)
                    if h is not None:
                        handles.append(h)

                for h in handles:
                    ratio_name, nint, seed_label, ret = wait_one(h)
                    if ret != 0:
                        failed.append((ratio_name, nint, seed_label, ret))
                        if STOP_ON_FAILURE:
                            print("\nSTOP_ON_FAILURE=True, stopping.")
                            for item in failed:
                                print(item)
                            raise SystemExit(ret)

    print("\nAll scheduled experiments finished.")
    print(f"Logs saved in: {LOG_DIR}")

    if failed:
        print("\nFailed runs:")
        for ratio_name, nint, seed_label, ret in failed:
            print(f"  ratio={ratio_name}, nint={nint}, {seed_label}, return={ret}")
    else:
        print("No failed runs.")


if __name__ == "__main__":
    main()
