# Fixed-Point-Guided Hierarchical Reduction of AL-RNNs (3-scroll Chua)

Code and data to reproduce the 3-scroll Chua results of the paper
(Figures 1, 2 and Appendix E). Trained checkpoints are included
(AL-RNNs are small), so every figure can be regenerated without
retraining; the full training / reduction / retraining pipeline is
also included for end-to-end reproduction.

## Layout

- `tutorial.py` — AL-RNN model and free-run rollout
- `2_train_test_aug_only.py` — training engine (direct training and
  reduction-guided retraining; teacher cache, parent-state
  synchronization, fixed/auto auxiliary weighting)
- `graph_reduction_with_relu_pruning.py` — fixed-point-preserving
  ReLU-pruning hierarchy (reduction trees, quotient graphs)
- `linear_region_functions.py`, `dataset.py`, `metrics.py`
  (`E_stsp`, `E_H`), `fp_tracker.py` (fixed-point analysis),
  `freerun_cache.py` (official 10k-step evaluation protocol,
  post-transient fidelity)
- launchers:
  - `run_aug_only_ratio_sweep_original.py` — direct training sweep
    (P = 1..10, 30 seeds) and baseline retraining
  - `run_sync_scope_ablation_p10.py` — guidance x forcing ablation
  - `run_weighting_ablation_p10.py` — fixed-lambda cells, including
    the main protocol (full pre-activation guidance,
    lambda = (M-N)/N, full-state forcing)
  - `build_consensus_selection.py` — consensus estimate of the
    symbolic granularity (Appendix B.5; `--system chua --dry_run`)
- figure scripts:
  - Fig. 1: `fig1a_state_space.py`, `fig1b_transition_graph.py`,
    `fig1c_relu_linearization.py`, `fig1d_reduction_tree.py`
  - Fig. 2(a): `plot_direct_training_vs_P.py`
  - Fig. 2(b): `plot_hierarchical_reduction_structure.py`
  - Fig. 2(c): `plot_chua_fig2d_prime.py`
    (+ `plot_chua_direct_retrained_dist.py`)
  - Fig. A1 / A3: `plot_chua_appendix_ablations.py`
  - Fig. A2: `analyze_estsp_threshold_gallery.py`
  - Fig. A4: `analyze_retrain_transition_graphs.py`
- `data/` — 3-scroll Chua training / test trajectories
  (dt = 0.01, 400k / 100k samples)
- `models/` — all trained Chua checkpoints (parents P=10, direct
  P=1..10, retrained candidates of every reported condition)
- `results/relu_hierarchy/` — per-parent reduction trees and the
  candidate/retraining spec JSONs (incl. the 141-candidate selection
  `global_min_retrain_selection_original_nint128_m20_p10.json`)
- `results/aug_only/` — per-run summary CSVs of the reported
  ablations (the figure scripts read these)

## Reproducing the figures

```bash
pip install -r requirements.txt
python plot_direct_training_vs_P.py          # Fig. 2a
python plot_hierarchical_reduction_structure.py   # Fig. 2b
python plot_chua_fig2d_prime.py              # Fig. 2c
python plot_chua_appendix_ablations.py       # Fig. A1 + A3
python analyze_estsp_threshold_gallery.py    # Fig. A2
python analyze_retrain_transition_graphs.py  # Fig. A4
python fig1a_state_space.py                  # Fig. 1 (a-d likewise)
python build_consensus_selection.py --system chua --dry_run  # B.5
```

Figures are written to `figures/`. On first run the scripts build
10k-step free-run records for the referenced checkpoints in
`results/freerun_cache/` (a few minutes; cached afterwards).

Conventions: E_stsp / E_H are computed on the post-transient rollout
(first 1000 of 10000 steps discarded); success rates are
parent-balanced seed-macro over the 30 parent seeds (failed runs
count as failures); Q_vis uses the full-rollout visited set.

## Retraining from scratch (optional)

Direct training and the ablations can be re-run with the launchers
above (several thousand CPU-hours in total); all launchers are
resume-safe and skip existing checkpoints.
