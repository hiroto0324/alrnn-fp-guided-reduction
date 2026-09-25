#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent free-run record cache: one .npz per (checkpoint, steps).

Any analysis that free-runs a saved checkpoint should go through
``freerun_record`` so the rollout is computed ONCE and every downstream
quantity is stored with it:

  readout   (T, N)  float32  readout trajectory z[:, :N]
  bits      (T, P)  uint8    full-P symbol sequence (z[:, M-P:] > 0)
  retained_local    int[]    ReLU slot indices kept by the ckpt's relu_mask
  Dstsp, DH         float    same convention as training-time eval
  Q_vis             int      visited real FPs (official collect_fp_snapshot)
  n_posttransient_symbols    unique retained-bit patterns after `cut`
  fp_region_bits / fp_type / fp_visited / fp_position_z   all real FPs

Conventions are IDENTICAL to evaluate_model_quick in 2_train_test_aug_only:
  z0 = raw[0] @ B (predict_free_sequence init), rollout `steps` steps,
  Dstsp = state_space_divergence_binning(readout, full raw data),
  DH    = power_spectrum_error(readout, raw[:steps]),
  FP snapshot = collect_fp_snapshot over ALL 2^P regions with the
  TRAINING script's mask-aware analyze_fixed_points_continuous
  (the grp version is NOT mask-aware — never use it on reduced models).
Symbol/graph statistics use the post-transient cut (default 1000);
Dstsp/DH use the full rollout (training-eval convention, no cut).
"""

from pathlib import Path
import hashlib

import numpy as np
import torch

RAW_DATA_PATH = "data/chua_3-scroll_train.npy"
CACHE_DIR = Path("results/freerun_cache")
STEPS_DEFAULT = 10000
CUT_DEFAULT = 1000

_ta = None          # lazily-loaded training module (mask-aware FP analyze)
_raw = None


def _training_module():
    global _ta
    if _ta is None:
        import importlib.util, io, contextlib
        spec = importlib.util.spec_from_file_location(
            "_freerun_ta", "2_train_test_aug_only.py")
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        _ta = mod
    return _ta


def _raw_data(data_path=None):
    global _raw
    if not isinstance(_raw, dict):
        _raw = {}
    key = data_path or RAW_DATA_PATH
    if key not in _raw:
        _raw[key] = np.load(key).astype(np.float32)
    return _raw[key]


def load_model(ckpt_path):
    """AL_RNN reconstructed from a checkpoint alone (M, N, relu_mask from
    the state dict). Models WITHOUT a relu_mask buffer (parents, direct
    runs) carry P only in their filename tag (`_p{P}_seed...`), so it is
    parsed from there; M//2 is the last-resort fallback."""
    import re
    from tutorial import AL_RNN
    sd = torch.load(ckpt_path, map_location="cpu")
    M = int(sd["A"].shape[0])
    N = int(sd["B"].shape[0])
    if "relu_mask" in sd:
        P = int(sd["relu_mask"].numel())
        model = AL_RNN(M=M, P=P, N=N, relu_mask=sd["relu_mask"].bool().tolist())
    else:
        m = re.search(r"_p(\d+)_", Path(ckpt_path).stem)
        P = int(m.group(1)) if m else M // 2
        model = AL_RNN(M=M, P=P, N=N)
    model.load_state_dict(sd)
    model.eval()
    return model


def _cache_path(ckpt_path, steps):
    h = hashlib.md5(str(Path(ckpt_path).resolve()).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{Path(ckpt_path).stem}_{h}_freerun{steps}.npz"


def freerun_record(ckpt_path, steps=STEPS_DEFAULT, cut=CUT_DEFAULT,
                   force=False, data_path=None, fp_delta_t=0.01):
    """Load the cached free-run record for `ckpt_path`, computing (and
    persisting) it on first use. Returns a dict of numpy values.

    `data_path` / `fp_delta_t` select the reference trajectory and the
    FP-analysis dt for non-Chua systems (defaults keep every existing
    Chua cache entry's stamp byte-identical)."""
    ckpt_path = Path(ckpt_path)
    data_path = data_path or RAW_DATA_PATH
    stamp = (f"{ckpt_path.resolve()}|{ckpt_path.stat().st_mtime_ns}|"
             f"{steps}|{cut}|{data_path}")
    if abs(fp_delta_t - 0.01) > 1e-12:
        stamp += f"|dt{fp_delta_t:g}"
    cpath = _cache_path(ckpt_path, steps)
    if cpath.exists() and not force:
        with np.load(cpath, allow_pickle=False) as z:
            if str(z["stamp"]) == stamp:
                return {k: z[k] for k in z.files}

    from tutorial import predict_free_sequence
    from fp_tracker import collect_fp_snapshot
    from metrics import state_space_divergence_binning, power_spectrum_error
    torch.set_num_threads(1)

    model = load_model(ckpt_path)
    M, P, N = model.M, model.P, model.N
    mask = (model.relu_mask.bool().numpy() if model.relu_mask is not None
            else np.ones(P, bool))
    raw = _raw_data(data_path)
    with torch.no_grad():
        z = predict_free_sequence(model, torch.tensor(raw[:1, :]), steps)[0]
    z_np = z.numpy()
    readout = z_np[:, :N].astype(np.float32)
    bits = (z_np[:, M - P:] > 0).astype(np.uint8)
    retained_local = np.flatnonzero(mask).astype(np.int64)

    dstsp = float(state_space_divergence_binning(
        z[:, :N], torch.tensor(raw)))
    dh = float(power_spectrum_error(readout, raw[:steps, :]))

    snap = collect_fp_snapshot(model, z_np, -1,
                               _training_module()
                               .analyze_fixed_points_continuous,
                               delta_t=fp_delta_t)
    fps = snap["fps"]
    rec = dict(
        stamp=np.str_(stamp),
        steps=np.int64(steps), cut=np.int64(cut),
        M=np.int64(M), P=np.int64(P), N=np.int64(N),
        readout=readout, bits=bits, retained_local=retained_local,
        Dstsp=np.float64(dstsp), DH=np.float64(dh),
        Q_vis=np.int64(snap["counts"].get("vis_total", 0)),
        n_posttransient_symbols=np.int64(
            len(np.unique(bits[cut:][:, mask], axis=0))),
        fp_region_bits=np.array([fp["fp_region_bits"] for fp in fps],
                                dtype=f"<U{P}"),
        fp_type=np.array([fp["fp_type"] for fp in fps], dtype="<U16"),
        fp_visited=np.array([bool(fp["is_visited"]) for fp in fps],
                            dtype=bool),
        fp_position_z=(np.stack([fp["fp_position_z"] for fp in fps])
                       .astype(np.float32) if fps
                       else np.zeros((0, M), np.float32)),
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **rec)
    tmp.replace(cpath)
    return rec


def posttransient_fidelity(rec, data_path=None):
    """(E_stsp, E_H) over the POST-TRANSIENT readout (rollout[cut:]).

    Official fidelity convention since 2026-09-25 (user decision): the
    stored Dstsp/DH keep the historical full-rollout values; reported
    E_stsp/E_H drop the first `cut` steps of the generated rollout.
    Reference distributions are unchanged (E_stsp: full training
    trajectory; E_H: the first len(rollout)-cut data steps, matching
    the generated length). Derived from the cached readout, so no
    model re-run is needed. Returns (nan, nan) for diverged rollouts.
    """
    from metrics import (state_space_divergence_binning,
                         power_spectrum_error)
    raw = _raw_data(str(data_path or RAW_DATA_PATH))
    cut = int(rec["cut"])
    ro = np.asarray(rec["readout"], np.float64)[cut:]
    try:
        e = float(state_space_divergence_binning(
            torch.tensor(ro), torch.tensor(np.asarray(raw, np.float64))))
    except Exception:
        e = float("nan")
    try:
        with np.errstate(all="ignore"):
            eh = float(power_spectrum_error(ro, raw[:len(ro)]))
    except Exception:
        eh = float("nan")
    return e, eh


def visited_fp_patterns(rec, retained_only=True):
    """Retained-bit (or full-P) patterns of the record's VISITED real FPs."""
    rl = rec["retained_local"]
    out = set()
    for s, v in zip(rec["fp_region_bits"], rec["fp_visited"]):
        if not v:
            continue
        b = tuple(int(c) for c in str(s))
        out.add(tuple(b[j] for j in rl) if retained_only else b)
    return out
