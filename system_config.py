#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-system configuration for the AL-RNN pipeline (Chua / Lorenz-63 /
Rössler).

Conventions
-----------
- `integration_dt` : RK4 step used by the data generator.
- `sample_dt`      : sampling interval of the saved trajectory
                     (= integration_dt * subsample). The AL-RNN sees this
                     discrete-time system.
- `fp_delta_t`     : dt used by the continuous-time fixed-point analysis;
                     ALWAYS equals sample_dt.
- `system_fixed_points` : ALL analytic fixed points of the ODE (raw,
  un-normalized coordinates). This is a property of the equations.
- `target_fp_indices`   : which of those are the attractor-relevant
  structural targets for trajectory-trained models. Fixed a priori from
  the data-support / published minimal representations — NOT tuned on
  model results. `target_fp_count = len(target_fp_indices)` is what the
  training score and parent screening use; it is NOT the total number of
  fixed points of the system.
- Normalization: lorenz63 / rossler trajectories are z-scored with the
  TRAIN trajectory's mean/std; the same affine map is applied to the test
  split and to the analytic fixed points (see the *_meta.json written by
  1_generate_benchmark_data.py). Chua stays un-normalized (legacy).
- N=3, M=20, P_parent=10 for every system. P_direct follows the published
  minimal nonlinear capacities (Chua 3, Lorenz-63 2, Rössler 1).
"""

import json
from pathlib import Path

import numpy as np


def _rossler_fixed_points(a=0.2, b=0.2, c=5.7):
    disc = np.sqrt(c * c - 4 * a * b)
    fps = []
    for s in (+1, -1):
        y = (-c + s * disc) / (2 * a)
        fps.append([-a * y, y, -y])
    # index 0: near the chaotic attractor (|y| ~ 0.035); index 1: far FP
    return np.array(fps)


def _lorenz_fixed_points(sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    r = np.sqrt(beta * (rho - 1))
    # index 0: origin (saddle); 1, 2: wing centers C+/C-
    return np.array([[0.0, 0.0, 0.0],
                     [r, r, rho - 1],
                     [-r, -r, rho - 1]])


SYSTEMS = {
    "chua": dict(
        tag_prefix="chua",
        train_path="data/chua_3-scroll_train.npy",
        test_path="data/chua_3-scroll_test.npy",
        meta_path=None,                  # legacy data, no meta JSON
        integration_dt=0.01, sample_dt=0.01, fp_delta_t=0.01,
        N=3, M=20, P_parent=10, P_direct=3,
        ode_params=None,
        system_fixed_points=None,        # legacy: not tracked analytically
        target_fp_indices=None,
        target_fp_count=5,               # established Chua 3-scroll target
        known_minimal_symbols=5,         # validation only, never selection
        normalized=False,
    ),
    "lorenz63": dict(
        tag_prefix="lor63",
        train_path="data/lorenz63_norm_train.npy",
        test_path="data/lorenz63_norm_test.npy",
        meta_path="data/lorenz63_norm_meta.json",
        integration_dt=0.01, sample_dt=0.01, fp_delta_t=0.01,
        N=3, M=20, P_parent=10, P_direct=2,
        ode_params=dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
        system_fixed_points=_lorenz_fixed_points().tolist(),
        # structural targets = ALL three fixed points (origin saddle +
        # wing centers C+/C-), following the AL-RNN NeurIPS reference
        # where 3 is the correct count for Lorenz-63. (2026-09-22: fixed
        # from an earlier wings-only target=2 misconfiguration that also
        # skewed best-epoch selection scores.)
        target_fp_indices=[0, 1, 2],
        target_fp_count=3,
        known_minimal_symbols=3,         # published minimal realization (validation only)
        normalized=True,
    ),
    # control condition: lorenz63 + observation noise sigma=0.1 (replicates
    # noisy-training setups in the literature; main line is noise-free)
    "lorenz63_noise": dict(
        tag_prefix="lor63n",
        train_path="data/lorenz63_noise_train.npy",
        test_path="data/lorenz63_noise_test.npy",
        meta_path="data/lorenz63_noise_meta.json",
        integration_dt=0.01, sample_dt=0.01, fp_delta_t=0.01,
        N=3, M=20, P_parent=10, P_direct=2,
        ode_params=dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
        system_fixed_points=_lorenz_fixed_points().tolist(),
        target_fp_indices=[0, 1, 2],
        target_fp_count=3,
        known_minimal_symbols=3,
        normalized=True,
    ),
    "rossler": dict(
        tag_prefix="ross",
        train_path="data/rossler_norm_train.npy",
        test_path="data/rossler_norm_test.npy",
        meta_path="data/rossler_norm_meta.json",
        integration_dt=0.01, sample_dt=0.05, fp_delta_t=0.05,
        N=3, M=20, P_parent=10, P_direct=1,
        ode_params=dict(a=0.2, b=0.2, c=5.7),
        system_fixed_points=_rossler_fixed_points().tolist(),
        # only the inner FP sits at the chaotic attractor; the outer FP is
        # far outside the data support. target_fp_count=1 does NOT mean
        # "Rössler has 1 FP" (it has 2).
        target_fp_indices=[0],
        target_fp_count=1,
        known_minimal_symbols=2,         # published: FP-free symbol required (validation only)
        normalized=True,
    ),
}


def get_system(name):
    cfg = dict(SYSTEMS[name])
    cfg["name"] = name
    return cfg


def normalized_target_fps(name):
    """Target fixed points in the coordinates the model is trained on
    (i.e. after the z-score normalization recorded in the meta JSON)."""
    cfg = get_system(name)
    fps = np.array(cfg["system_fixed_points"], dtype=float)
    fps = fps[cfg["target_fp_indices"]]
    if cfg["normalized"]:
        meta = json.load(open(cfg["meta_path"]))
        mean = np.array(meta["normalization"]["mean"])
        std = np.array(meta["normalization"]["std"])
        fps = (fps - mean) / std
    return fps


if __name__ == "__main__":
    for name in SYSTEMS:
        c = get_system(name)
        print(f"{name}: tag={c['tag_prefix']}  P_direct={c['P_direct']}  "
              f"sample_dt={c['sample_dt']}  target_fp_count="
              f"{c['target_fp_count']}  train={c['train_path']} "
              f"(exists={Path(c['train_path']).exists()})")
