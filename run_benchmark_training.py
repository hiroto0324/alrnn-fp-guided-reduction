#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic direct / parent training launcher for the benchmark systems
(rossler, lorenz63) — Chua protocol transferred verbatim.

Runs 2_train_test_aug_only.py in data_mode=original with the exact Chua
Direct settings (nint=128 raw-data readout TF, bs16, lr 1e-3→1e-5,
2000x50, segment 200) and the system's tag_prefix / fp_delta_t /
target_fp_count from system_config. Skip-completed via the model tag
(same convention as every other launcher: models/{tag}.pth exists).

Examples
--------
  # Rossler direct P=1, smoke 3 seeds
  python run_benchmark_training.py --system rossler --p 1 --seeds 0-2
  # Rossler parent P=10, 30 seeds
  python run_benchmark_training.py --system rossler --p 10 --seeds 0-29
  # Lorenz direct P=2
  python run_benchmark_training.py --system lorenz63 --p 2 --seeds 0-29
"""

import argparse
import datetime
import os
import subprocess
import sys
import threading
from pathlib import Path

import system_config as sc

TRAIN_SCRIPT = "2_train_test_aug_only.py"
LOG_DIR = Path("logs/benchmark")

# Chua direct protocol (identical values to run_aug_only_ratio_sweep_original)
NUM_EPOCHS, STEPS_PER_EPOCH, BATCH_EPISODES = 2000, 50, 16
SEGMENT_LEN, MAX_BRIDGE_LEN = 200, 200
N_INTERLEAVE = 128
ALPHA, INPUT_SIGMA, STATE_SIGMA = 1.0, 0.0, 0.0
LR_START, LR_END, SSI = 1e-3, 1e-5, 10
FP_COUNT_WEIGHT = 5.0


def make_tag(cfg, p, seed, n_interleave=N_INTERLEAVE):
    """Must match the tag generation in 2_train_test_aug_only.py."""
    return (f"{cfg['tag_prefix']}_orig_nint{n_interleave}"
            f"_bs{BATCH_EPISODES}_sig{INPUT_SIGMA}"
            f"_lr{LR_START:.0e}-{LR_END:.0e}"
            f"_tfp0_ramp{SEGMENT_LEN}_hLR1_m{cfg['M']}_p{p}_seed{seed}")


def build_command(cfg, p, seed, num_epochs, steps_per_epoch):
    return [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--data_mode", "original",
        "--raw_data_path", cfg["train_path"],
        "--tag_prefix", cfg["tag_prefix"],
        "--fp_delta_t", str(cfg["fp_delta_t"]),
        "--P_list", str(p),
        "--n_interleave", str(N_INTERLEAVE),
        "--num_epochs", str(num_epochs),
        "--steps_per_epoch", str(steps_per_epoch),
        "--batch_episodes", str(BATCH_EPISODES),
        "--segment_len", str(SEGMENT_LEN),
        "--max_bridge_len", str(MAX_BRIDGE_LEN),
        "--alpha", str(ALPHA),
        "--r_anchor", "0.0", "--r_local", "0.0",
        "--r_bridge", "0.0", "--r_long", "1.0",
        "--input_sigma", str(INPUT_SIGMA),
        "--state_sigma", str(STATE_SIGMA),
        "--lr_start", str(LR_START), "--lr_end", str(LR_END),
        "--ssi", str(SSI),
        "--target_fp_count", str(cfg["target_fp_count"]),
        "--fp_count_weight", str(FP_COUNT_WEIGHT),
        "--seed", str(seed),
    ]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_one(cfg, p, seed, args, sem, results):
    tag = make_tag(cfg, p, seed)
    ckpt = Path("models") / f"{tag}.pth"
    if ckpt.exists() and not args.force_rerun:
        print(f"[SKIP] seed={seed}  {ckpt.name} exists")
        results.append((seed, "skip"))
        return
    cmd = build_command(cfg, p, seed, args.num_epochs, args.steps_per_epoch)
    log_path = LOG_DIR / f"{tag}.log"
    if args.dry_run:
        print(f"[DRY] seed={seed} tag={tag}\n  " + " ".join(cmd))
        results.append((seed, "dry"))
        return
    with sem:
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        t0 = datetime.datetime.now()
        print(f"[LAUNCH] {cfg['name']} p={p} seed={seed}  log={log_path}")
        with open(log_path, "wb") as f:
            f.write(f"# {t0}\n# {' '.join(cmd)}\n\n".encode())
            f.flush()
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 env=env).returncode
        dt = datetime.datetime.now() - t0
        print(f"[{'OK' if ret == 0 else 'FAILED'}] seed={seed} "
              f"({dt}, return={ret})")
        results.append((seed, ret))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True,
                    choices=["rossler", "lorenz63", "lorenz63_noise"])
    ap.add_argument("--p", type=int, default=None,
                    help="P (default: system's P_direct; 10 for parent)")
    ap.add_argument("--seeds", default="0-2",
                    help="e.g. 0-2, 0-29, 3,7,11")
    # default 30 so all 30 seeds fit in one wave (~7% slowdown vs
    # 28-parallel on 18 physical cores; far cheaper than a ragged wave)
    ap.add_argument("--parallel", type=int, default=30)
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--steps_per_epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--force_rerun", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = sc.get_system(args.system)
    p = args.p if args.p is not None else cfg["P_direct"]
    seeds = parse_seeds(args.seeds)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"system={args.system}  P={p}  seeds={seeds}  "
          f"nint={N_INTERLEAVE} (raw TF)  target_fp_count="
          f"{cfg['target_fp_count']}  fp_delta_t={cfg['fp_delta_t']}  "
          f"epochs={args.num_epochs}x{args.steps_per_epoch}")

    sem = threading.Semaphore(args.parallel)
    results = []
    ths = [threading.Thread(target=run_one,
                            args=(cfg, p, s, args, sem, results),
                            daemon=True) for s in seeds]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    failed = [s for s, r in results if r not in (0, "skip", "dry")]
    print(f"\ndone: {len(results)} runs, failed={failed or 'none'}")


if __name__ == "__main__":
    main()
