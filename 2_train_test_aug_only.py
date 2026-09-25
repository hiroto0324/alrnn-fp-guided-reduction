#!/usr/bin/env python3
"""
Augmented-data-only training: sample from anchor / local / bridge / long episodes.
The original (global) data is never used directly.

Episode selection:
  FP round-robin -> sample types at the anchor:local:bridge:long ratio
  anchor/local/bridge are balanced per FP
  long cuts a random segment per FP

Evaluation (every SSI epochs):
  - train loss
  - free-run Dstsp, DH
  - real FP counts (fp_tracker)
  - GT FP matching distance (per-FP subplot)

Outputs:
  figures/aug_only/{tag}_loss.png
  figures/aug_only/{tag}_metrics.png
  figures/aug_only/{tag}_fp_counts.png
  figures/aug_only/{tag}_fp_overlay.png
  figures/aug_only/{tag}_fp_matching.png
  results/aug_only/{tag}_train.npz
  results/aug_only/{tag}_fp_matching.npz
  results/aug_only/summary.csv

Usage:
  python 2_train_test_aug_only.py --P 3
  python 2_train_test_aug_only.py --P_list 3 4 5 --r_anchor 0.1 --r_local 0.3 --r_bridge 0.3 --r_long 0.3
  python 2_train_test_aug_only.py --P 3 --lr_fixed --no_periodic_eval
"""

import argparse
import collections
import copy
import csv
import json
import os
import random
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from tutorial import AL_RNN, predict_free_sequence
from metrics import state_space_divergence_binning, power_spectrum_error
import linear_region_functions as lrf
from fp_tracker import (collect_fp_snapshot, save_fp_snapshots,
                        plot_fp_trajectory_overlay, plot_fp_counts,
                        compute_fp_matching, plot_fp_matching_history,
                        save_fp_matching_history)

# ============================================================
DATA_PATH      = "results/chua_3scroll_bridge_augmented_data_m20_p17.npz"
META_PATH      = "results/chua_3scroll_bridge_augmented_data_m20_p17_meta.json"
LONG_DATA_PATH = "results/chua_3scroll_long_trajs_m20_p17.npz"

M       = 20
SSI     = 10
DELTA_T = 0.01

# teacher distillation mode -> suffix appended to checkpoint/log names
# (must match the dict of the same name in run_aug_only_ratio_sweep_original.py)
# 'none' is the empty string = keep the existing baseline naming (backward compat)
DISTILL_TAGS = {
    'none':                    '',
    'full_preactivation':      '_distfullpre',
    'retained_preactivation':  '_distretpre',
    'symbol':                  '_distsym',
}
# ============================================================


# ── Fixed-point analysis ──────────────────────────────────────────────────────

def analyze_fixed_points_continuous(model, unique_symbols_list, delta_t=None):
    if delta_t is None:          # use DELTA_T (--fp_delta_t) at call time
        delta_t = DELTA_T
    M, P = model.M, model.P
    A = np.diag(model.A.detach().cpu().numpy())
    W = model.W.detach().cpu().numpy()
    h = model.h.detach().cpu().numpy()
    # relu_mask support (reduced models): slots with mask[j]=False are identity.
    # Index convention: local slot j → global hidden index = M - P + j。
    # identity slots always pass values regardless of sign -> treat as bit=1 in
    # the linear map, and the bit does not define the region (the 'real' test compares ReLU slots only).
    _mask = getattr(model, 'relu_mask', None)
    mask_np = None if _mask is None else _mask.detach().cpu().numpy().astype(bool)
    results = {}
    for symbol_vec in unique_symbols_list:
        symbol_vec = np.asarray(symbol_vec, dtype=int)
        eff_vec = symbol_vec
        if mask_np is not None:
            eff_vec = symbol_vec.copy()
            eff_vec[~mask_np] = 1
        padded_vec = np.pad(eff_vec, (M - P, 0), 'constant', constant_values=1)
        W_k = A + W @ np.diag(padded_vec)
        I   = np.identity(M)
        eigenvalues = np.linalg.eigvals((W_k - I) / delta_t)
        try:
            fp = np.linalg.inv(I - W_k) @ h
        except np.linalg.LinAlgError:
            continue
        signs = (fp[-P:] > 0).astype(int)
        if mask_np is None:
            sym_str = ''.join(map(str, symbol_vec))
            is_real = np.array_equal(signs, symbol_vec)
        else:
            # symbols differing only in identity-slot bits collapse to the same fixed
            # point, so use a canonical key with identity bits set from the FP's own sign
            is_real = np.array_equal(signs[mask_np], symbol_vec[mask_np])
            canon = symbol_vec.copy()
            canon[~mask_np] = signs[~mask_np]
            sym_str = ''.join(map(str, canon))
        has_complex = np.any(np.abs(eigenvalues.imag) > 1e-9)
        real_parts  = eigenvalues.real
        if np.all(real_parts < 0):
            stab = 'stable spiral'   if has_complex else 'stable node'
        elif np.all(real_parts > 0):
            stab = 'unstable spiral' if has_complex else 'unstable node'
        else:
            stab = 'saddle'
        results[sym_str] = {
            'type': 'real' if is_real else 'virtual',
            'stability': stab, 'location': fp,
            'eigenvalues_continuous': eigenvalues,
        }
    return results


def evaluate_model_quick(model, orig_data):
    """Same computation as the periodic eval (10000-step free run -> Dstsp, visited real FPs).

    Used to record the source / reduced model's performance just before retraining."""
    device = next(model.parameters()).device
    X_torch = torch.tensor(orig_data).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        z_free = predict_free_sequence(model, X_torch[:, 0, :], 10000)[0]
    N = model.N
    Dstsp = state_space_divergence_binning(z_free[:, :N], X_torch[0])
    snap = collect_fp_snapshot(model, z_free.detach().cpu().numpy(), -1,
                               analyze_fixed_points_continuous,
                               delta_t=DELTA_T)
    if was_training:
        model.train()
    return float(Dstsp), int(snap['counts'].get('vis_total', 0))


# ── Teacher forcing ───────────────────────────────────────────────────────────

def teacher_force(z, x, alpha):
    N = x.shape[-1]; z = z.clone()
    z[:, :N] = alpha * x + (1 - alpha) * z[:, :N]
    return z




def _git_commit_or_none():
    """Current git commit for reproducibility metadata (None on failure)."""
    try:
        import subprocess as _sp
        return _sp.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                       text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def apply_parent_state_sync(z, z_parent, sync_scope, n_readout,
                            retained_global_idx):
    """PARENT-STATE SYNCHRONIZATION (distinct from raw-data teacher forcing).

    z        : (B, M) student state. At loop iteration t this is the
               pre-update state = segment-time-t state (z0 = cache[s], so parent time s+t).
    z_parent : (B, M) frozen parent's autonomous-cache state at the SAME time s+t.
    Returns a clone with only the scoped coordinates overwritten (no in-place).
      readout          : first N coordinates only
      readout_retained : first N + global coordinates R of the retained ReLUs
      full             : all M coordinates
    The sync source is always the parent's autonomous cache; raw data is never used.
    """
    z = z.clone()
    if sync_scope == 'readout':
        z[:, :n_readout] = z_parent[:, :n_readout]
    elif sync_scope == 'readout_retained':
        z[:, :n_readout] = z_parent[:, :n_readout]
        z[:, retained_global_idx] = z_parent[:, retained_global_idx]
    elif sync_scope == 'full':
        z = z_parent.clone()
    else:
        raise ValueError(f"unknown sync_scope: {sync_scope}")
    return z


def run_episode_batch(model, x_batch, y_batch, loss_fn, alpha, n_interleave,
                      z_init_batch=None, input_noise_scale=None, state_sigma=0.0,
                      return_states=False,
                      parent_sync_traj=None, sync_scope='none',
                      sync_interval=128, sync_n_readout=None,
                      sync_retained_idx=None, sync_debug=False):
    """
    x_batch: (B, T, N)
    y_batch: (B, T, N)
    z_init_batch: (B, M) or None
    input_noise_scale: (B, N) or None  — anchor/local=0, bridge/long=σ×std
    return_states: if True, return (loss, states); states is the (B, T, M)
        state trajectory. In this AL-RNN formulation the forward applies the
        activation to the INPUT state before the transition, so state z_t
        coincides with the preactivation (z_pre) right before the next
        activation (same convention that FP analysis / symbol bits use).
        Cloning is required because the unmasked AL_RNN.forward ReLUs the
        input tensor's last-P slots in place (keep the raw pre-update values).
    """
    B, T, N = x_batch.shape

    if z_init_batch is not None:
        z = z_init_batch.clone()
    else:
        z = x_batch[:, 0, :] @ model.B
        z = teacher_force(z, x_batch[:, 0, :], alpha=1.0)

    preds = []
    states = [] if return_states else None
    for t in range(T):
        if t > 0 and t % n_interleave == 0:
            x_t = x_batch[:, t, :]
            if input_noise_scale is not None:
                x_t = x_t + input_noise_scale * torch.randn_like(x_t)
            z = teacher_force(z, x_t, alpha)
        # ── parent-state sync (sparse hard sync for distillation) ──
        # TIMING: at loop iteration t, z is the pre-update state; under the
        # z0=cache[s] convention it corresponds to parent time s+t.
        # parent_sync_traj[:, t] = cache[s+t] (same time) is injected here
        # BEFORE the model update, matching the raw-data teacher forcing
        # timing above (t % n_interleave) with zero step offset. t=0 (full
        # state init at segment start) is shared and not counted as a sync event.
        if (parent_sync_traj is not None and sync_scope != 'none'
                and t > 0 and t % sync_interval == 0):
            zp = parent_sync_traj[:, t]
            if sync_debug:
                if sync_scope == 'full':
                    _sidx = list(range(z.shape[1]))
                elif sync_scope == 'readout':
                    _sidx = list(range(sync_n_readout))
                else:
                    _sidx = (list(range(sync_n_readout))
                             + list(sync_retained_idx))
                _before = (z[:, _sidx] - zp[:, _sidx]).abs().max().item()
            z = apply_parent_state_sync(z, zp, sync_scope,
                                        sync_n_readout, sync_retained_idx)
            if sync_debug:
                _after = (z[:, _sidx] - zp[:, _sidx]).abs().max().item()
                print(f"    [parent-sync debug] t={t} scope={sync_scope} "
                      f"n_coords={len(_sidx)} "
                      f"max|before-parent|={_before:.4g} "
                      f"max|after-parent|={_after:.4g}")
        z = model(z)
        if state_sigma > 0.0:
            z = z + state_sigma * torch.randn_like(z)
        preds.append(z[:, :N])
        if return_states:
            states.append(z.clone())

    pred = torch.stack(preds, dim=1)   # (B, T, N)
    loss = loss_fn(pred, y_batch)
    if return_states:
        return loss, torch.stack(states, dim=1)   # (B, T, M)
    return loss


def compute_distill_loss(distill_mode, student_states, teacher_states,
                         retained_global_idx, symbol_margin,
                         teacher_bits=None, n_readout=None):
    """Auxiliary teacher-distillation loss (mean over time x target dims).

    student_states / teacher_states: (B, T, M) state trajectories
        (= preactivation right before each step's activation; see run_episode_batch)
    retained_global_idx: list of global hidden indices of the retained
        (still nonlinear) ReLU slots. Convention: global = M - P_original + local_relu_index.

    The 3 modes (hierarchy: all internal -> retained nonlinear -> symbolic sign):
      A full_preactivation = all NON-READOUT internal preactivation dimensions:
        targets dims N..M-1 only (originally linear hidden units, retained
        ReLUs, identity-fied units). The readout (first N dims) is already
        supervised by L_out (teacher free-run output), so it is excluded
        from aux to avoid double counting (n_readout=N required; normalization 1/(T*(M-N)) mean).
      B retained_preactivation:
        reproduce only the internal coordinates of the retained ReLU slots that define the reduced symbolic partition.
        deleted/identity slots and originally-linear units are excluded.
      C symbol:
        reproduce only the post-reduction linear-region assignment (sign
        pattern of the retained ReLUs), not the teacher's internal values.
        This demands 'same side of the switching hyperplane' rather than
        'same preactivation value': a margin-based soft loss. Hard
        thresholding is only used to build the teacher-side target; the
        student side keeps continuous preactivations (sign / >0 / bool on the student side would block gradients).
    """
    if distill_mode == 'full_preactivation':
        # readout dims (0..N-1) belong to L_out. aux covers internal N..M-1 only
        # (never include the first N dims). mean reduction -> denominator T*B*(M-N)
        assert n_readout is not None and n_readout > 0, \
            "full_preactivation requires n_readout (=N)"
        return F.mse_loss(student_states[..., n_readout:],
                          teacher_states[..., n_readout:])
    elif distill_mode == 'retained_preactivation':
        return F.mse_loss(student_states[..., retained_global_idx],
                          teacher_states[..., retained_global_idx])
    elif distill_mode == 'symbol':
        s_pre = student_states[..., retained_global_idx]          # continuous (with grad)
        if teacher_bits is not None:
            # bits read from the teacher cache (bool/uint8).
            # identity with (teacher_pre[..., retained] > 0) is guaranteed
            # by the cache-generation definition and its test
            t_bits = teacher_bits
        else:
            t_bits = (teacher_states[..., retained_global_idx] > 0)  # hard on the teacher side only
        y = t_bits.to(s_pre.dtype) * 2.0 - 1.0                    # {0,1} → {-1,+1}
        return F.softplus(symbol_margin - y * s_pre).mean()
    raise ValueError(f"unknown distill_mode: {distill_mode}")


# ── Teacher trajectory cache ─────────────────────────────────────────────────
# The distillation teacher target is fixed to ONE fully autonomous free-run
# trajectory of the source over-parameterized model. No teacher forcing, no
# periodic raw-data injection, no per-segment resets (raw data is used only
# to build the initial condition z0). The teacher is fully frozen, so the
# cache is computed once per source checkpoint and shared by every
# candidate / distillation mode / retrain seed. No teacher forward in training.
#
# index convention (watch the off-by-one):
#   cache[t] = teacher state at time t. cache[0] = initial state z0 (init below),
#   cache[t] = state after t autonomous updates from z0.
#   readout / preactivation / symbol all derive from the same cache[t] (same phase).
# alignment with the student:
#   the student initializes its hidden state to cache[s] at segment start s,
#   rolls out L autonomous steps -> student states[t] = state at time s+t+1
#   -> teacher target is cache[s+1 : s+L+1].
#   (convention under which student == teacher gives exactly zero loss)

TEACHER_CACHE_VERSION = 2
TEACHER_CACHE_PROTOCOL = "autonomous_free_run"


def teacher_cache_path(cache_dir, source_checkpoint):
    """The cache path depends only on the source checkpoint stem
    (independent of candidate / distill mode / retrain seed = shared by all)."""
    stem = os.path.splitext(os.path.basename(source_checkpoint))[0]
    return os.path.join(cache_dir, f"teacher_cache_{stem}.npz")


def _teacher_cache_meta(source_checkpoint, raw_data_path, raw_data,
                        M_, P_, N_):
    st = os.stat(source_checkpoint)
    return {
        'cache_version': TEACHER_CACHE_VERSION,
        'rollout_protocol': TEACHER_CACHE_PROTOCOL,
        'teacher_forcing': False,
        'source_checkpoint': str(source_checkpoint),
        'ckpt_mtime': st.st_mtime,
        'ckpt_size': st.st_size,
        'M': int(M_), 'P_original': int(P_), 'N': int(N_),
        'trajectory_length': int(raw_data.shape[0]),
        # raw data is used ONLY to build the initial condition z0 (never
        # injected into the rollout). The checksum is provenance for identity checks.
        'initial_condition_info': {
            'type': 'predict_free_sequence_init',   # z0 = raw[0]@B; z0[:N] = raw[0]
            'data_path': str(raw_data_path),
            'data_len': int(raw_data.shape[0]),
            'data_checksum': float(np.float64(raw_data).sum()),
        },
        'dtype': 'float32',
    }


def build_or_load_teacher_cache(source_checkpoint, raw_data_path, raw_data,
                                M_, P_, N_,
                                cache_dir='results/teacher_cache',
                                force_rebuild=False, verbose=True):
    """Return the teacher's autonomous free-run cache (build if missing).

    Returns (pre, bits, out, meta, cache_hit)
      pre  : torch.FloatTensor (T, M) CPU — cache[t] = teacher state at time t
             (state = preactivation in this AL-RNN). cache[0] = initial state.
      bits : torch.uint8 (T, P_original) — pre[:, -P:] > 0 (teacher-side hard sign)
      out  : torch.FloatTensor (T, N) — teacher output (observable) = pre[:, :N]。
             directly comparable to the student's pred[..., :N].
      cache_hit : True if an existing cache was reused

    rollout protocol: fully autonomous free run. The initial condition equals
    the standard free-run evaluation (predict_free_sequence / Dstsp eval):
    z0 = raw[0] @ B, z0[:N] = raw[0]. No raw-data injection afterwards.

    validity: cache metadata (version/protocol, checkpoint path/mtime/size,
    M/P/N, trajectory length, init provenance) exactly matches the current
    setup. Old teacher-forced caches (version 1) mismatch on version/protocol
    and go stale -> regenerated. Writes are atomic (tmp + os.replace).
    """
    meta = _teacher_cache_meta(source_checkpoint, raw_data_path, raw_data,
                               M_, P_, N_)
    path = teacher_cache_path(cache_dir, source_checkpoint)

    if os.path.exists(path) and not force_rebuild:
        try:
            # close the file via context manager (on Windows an open handle
            # makes the rebuild-time os.replace fail with PermissionError)
            with np.load(path, allow_pickle=False) as d:
                stored = json.loads(str(d['meta']))
                if stored == meta and 'teacher_out' in d.files:
                    pre = torch.from_numpy(d['teacher_pre'].astype(np.float32))
                    bits = torch.from_numpy(d['teacher_bits'].astype(np.uint8))
                    out = torch.from_numpy(d['teacher_out'].astype(np.float32))
                    pre.requires_grad_(False)
                    out.requires_grad_(False)
                    return pre, bits, out, meta, True
                mism = [k for k in meta if stored.get(k) != meta[k]]
            if verbose:
                print(f"  [teacher cache] stale ({mism}) → rebuild: {path}")
        except Exception as e:
            if verbose:
                print(f"  [teacher cache] unreadable ({e}) → rebuild: {path}")

    # ── build: teacher = original over-parameterized AL-RNN (no mask, frozen) ──
    teacher = AL_RNN(M=M_, P=P_, N=N_)
    state = torch.load(source_checkpoint, map_location='cpu')
    teacher.load_state_dict({k: v for k, v in state.items() if k != 'relu_mask'})
    teacher.eval()
    for tp in teacher.parameters():
        tp.requires_grad_(False)

    x0 = torch.tensor(np.asarray(raw_data[:1], dtype=np.float32))   # (1, N)
    T_full = int(raw_data.shape[0])
    with torch.no_grad():
        # initial condition: same as predict_free_sequence (z0 = raw[0]@B, first N overwritten with raw[0])
        z = x0 @ teacher.B
        z[:, :N_] = x0
        # clone required: the unmasked forward ReLUs the input's last-P slots
        # in place (keep the stored values as raw preactivations)
        pre_list = [z.squeeze(0).clone()]        # cache[0] = initial state
        # fully autonomous free run: no teacher forcing / raw-data injection
        for _t in range(T_full - 1):
            z = teacher(z)
            pre_list.append(z.squeeze(0).clone())
    pre = torch.stack(pre_list, dim=0)                     # (T, M)
    out = pre[:, :N_].clone()                              # (T, N) observable
    bits = (pre[:, -P_:] > 0).to(torch.uint8)              # (T, P)
    del teacher

    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"   # np.savez keeps the name if it ends in .npz
    np.savez(tmp, teacher_pre=pre.numpy().astype(np.float32),
             teacher_bits=bits.numpy(),
             teacher_out=out.numpy().astype(np.float32),
             meta=json.dumps(meta))
    # atomic replace (guards against parallel-generation races). On Windows
    # another process reading the same cache can cause PermissionError, so
    # retry briefly; on final failure just return the data (rebuilt/HIT next launch).
    for _attempt in range(3):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(2.0)
    else:
        if verbose:
            print(f"  [teacher cache] WARN: could not replace {path} "
                  f"(in use by another process?) — data used for this run only")
        try:
            os.remove(tmp)
        except OSError:
            pass
    pre.requires_grad_(False)
    out.requires_grad_(False)
    if verbose:
        print(f"  [teacher cache] built → {path}  "
              f"(pre {tuple(pre.shape)} float32, out {tuple(out.shape)}, "
              f"bits {tuple(bits.shape)} uint8, protocol={TEACHER_CACHE_PROTOCOL})")
    return pre, bits, out, meta, False


def compute_auto_lambda(g_out, g_aux, alpha, eps=1e-12):
    """Lambda from the initial gradient-norm ratio: lambda = alpha * g_out / (g_aux + eps).
    Training then starts with ||lambda grad L_aux|| ~= alpha ||grad L_out||."""
    return alpha * g_out / (g_aux + eps)


def _param_grad_norm(loss, params, retain_graph):
    """L2 gradient norm of the loss over all trainable parameters.
    Leaves optimizer / param.grad untouched (uses torch.autograd.grad).
    None gradients count as 0."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum().item())
    return total ** 0.5


def distill_autolambda_tag(auto_lambda, distill_mode, alpha):
    """Tag suffix of auto-lambda calibration runs (e.g. '_ga0p1').

    Manual-lambda runs get '' (existing naming unchanged). Alpha is part of
    the name so runs with different alpha never share a checkpoint.
    Must match the function of the same name in run_aug_only_ratio_sweep_original.py."""
    if not auto_lambda or distill_mode == 'none':
        return ''
    return '_ga' + f"{alpha:g}".replace('.', 'p').replace('-', 'm')


def target_fp_residual_loss(model, gt_x, fp_hidden):
    """
    Build the student state z_fp = [gt_x, fp_hidden] (K, M) from the teacher
    fixed points' observed coordinates gt_x (K, N) and a learnable hidden
    part fp_hidden (K, M-N); compute the FP residual ||model(z_fp) - z_fp||^2 over all M dims.

    No detach: gradients flow into both the model parameters and fp_hidden.
    gt_x is fixed with requires_grad=False.
    """
    z_fp   = torch.cat([gt_x, fp_hidden], dim=1)   # (K, M)
    z_next = model(z_fp)                            # (K, M)
    return ((z_next - z_fp) ** 2).mean()


def build_z_init_batch(model, x_batch, ep_types, fp_idxs, gt_fps_obs=None):
    """Build each episode's initial hidden state; returns a (B, M) tensor."""
    z_list = []
    for i in range(len(ep_types)):
        x0 = x_batch[i:i+1, 0, :]
        z0 = x0 @ model.B
        z0 = teacher_force(z0, x0, alpha=1.0)
        z_list.append(z0)
    return torch.cat(z_list, dim=0)   # (B, M)


def build_input_noise_scale(ep_types, input_noise_vec_gen):
    """Per-episode-type noise scale (B, N); zero for anchor/local."""
    if input_noise_vec_gen is None:
        return None
    zero = torch.zeros_like(input_noise_vec_gen)
    scales = [input_noise_vec_gen if ep in ('bridge', 'long') else zero
              for ep in ep_types]
    return torch.cat(scales, dim=0)   # (B, N)


# ── Aug-only queue manager ────────────────────────────────────────────────────

class AugOnlyQueueManager:
    """
    Sample anchor / local / bridge / long via FP round-robin + type ratios.

    long episodes:
      cut random segments per FP from long_trajs (K, T_long, M).
      Segment length is segment_len.
    """

    def __init__(self, anchor_trajs, anchor_meta,
                 local_trajs, local_meta,
                 bridge_trajs, bridge_lengths, bridge_meta,
                 long_trajs, K, N,
                 max_bridge_len=200, segment_len=200):
        self.K = K; self.N = N; self.rr = 0
        self.max_bridge_len = max_bridge_len
        self.segment_len    = segment_len

        self.fp_anchor = {k: [] for k in range(K)}
        self.fp_local  = {k: [] for k in range(K)}
        self.fp_bridge = {k: [] for k in range(K)}

        for i, m in enumerate(anchor_meta):
            self.fp_anchor[m['fp_index']].append(i)
        for i, m in enumerate(local_meta):
            self.fp_local[m['fp_index']].append(i)
        for i, m in enumerate(bridge_meta):
            self.fp_bridge[m['fp_index']].append(i)

        self.anchor_trajs   = anchor_trajs
        self.local_trajs    = local_trajs
        self.bridge_trajs   = bridge_trajs
        self.bridge_lengths = bridge_lengths
        self.long_trajs     = long_trajs    # (K, T_long+1, M)

        self.q_anchor = {k: self._mq(self.fp_anchor[k]) for k in range(K)}
        self.q_local  = {k: self._mq(self.fp_local[k])  for k in range(K)}
        self.q_bridge = {k: self._mq(self.fp_bridge[k]) for k in range(K)}

        for k in range(K):
            T_long = long_trajs.shape[1] if long_trajs is not None else 0
            print(f"  FP{k}: anchor={len(self.fp_anchor[k])}  "
                  f"local={len(self.fp_local[k])}  "
                  f"bridge={len(self.fp_bridge[k])}  "
                  f"long_len={T_long}")

    @staticmethod
    def _mq(idx_list):
        lst = list(idx_list); random.shuffle(lst)
        return collections.deque(lst)

    def _next_fp(self):
        for _ in range(self.K):
            fp = self.rr % self.K; self.rr += 1
            if (self.fp_anchor[fp] or self.fp_local[fp] or
                    self.fp_bridge[fp] or self.long_trajs is not None):
                return fp
        return None

    def _pop(self, queue, idx_list):
        if not queue:
            lst = list(idx_list); random.shuffle(lst)
            queue.extend(lst)
        return queue.popleft() if queue else None

    def _get(self, ep_type, fp):
        # x_seq/y_seq must both have segment_len entries -> need T_need = segment_len + 1 points
        T_need = self.segment_len + 1

        if ep_type == 'anchor':
            idx = self._pop(self.q_anchor[fp], self.fp_anchor[fp])
            if idx is None: return None, None
            ep_full = self.anchor_trajs[idx, :, :self.N]
            if len(ep_full) < T_need: return None, None
            ep_np = ep_full[:T_need]

        elif ep_type == 'local':
            idx = self._pop(self.q_local[fp], self.fp_local[fp])
            if idx is None: return None, None
            ep_full = self.local_trajs[idx, :, :self.N]
            if len(ep_full) < T_need: return None, None
            ep_np = ep_full[:T_need]

        elif ep_type == 'bridge':
            idx = self._pop(self.q_bridge[fp], self.fp_bridge[fp])
            if idx is None: return None, None
            T_avail = int(self.bridge_lengths[idx])
            if T_avail < T_need: return None, None
            if self.max_bridge_len < self.segment_len: return None, None
            ep_np = self.bridge_trajs[idx, :T_need, :self.N]

        else:  # long
            if self.long_trajs is None: return None, None
            traj_full = self.long_trajs[fp, :, :self.N]
            T_long = len(traj_full)
            if T_long < T_need: return None, None
            start = random.randint(0, T_long - T_need)
            ep_np = traj_full[start: start + T_need]

        return (torch.tensor(ep_np[:-1], dtype=torch.float32),
                torch.tensor(ep_np[1:],  dtype=torch.float32))

    def next_episode(self, r_anchor, r_local, r_bridge, r_long):
        ratios = {
            'anchor': r_anchor,
            'local':  r_local,
            'bridge': r_bridge,
            'long':   r_long,
        }

        # only types with r > 0 are candidates (zero-ratio types are not even fallbacks)
        allowed = [k for k, v in ratios.items() if v > 0]
        if not allowed:
            return None, None, None, None

        probs = np.array([ratios[k] for k in allowed], dtype=np.float64)
        probs = probs / probs.sum()
        ep_type = np.random.choice(allowed, p=probs)

        fp = self._next_fp()
        if fp is None: return None, None, ep_type, None

        # try the requested type first
        x, y = self._get(ep_type, fp)
        if x is not None:
            return x, y, ep_type, fp

        # fallbacks also stay within the allowed set
        for t in allowed:
            if t == ep_type:
                continue
            x, y = self._get(t, fp)
            if x is not None:
                return x, y, t, fp

        return None, None, ep_type, fp

    def next_batch(self, r_anchor, r_local, r_bridge, r_long, batch_episodes):
        """Collect batch_episodes episodes and return them stacked."""
        xs, ys, ep_types, fp_idxs = [], [], [], []

        for _ in range(batch_episodes):
            x, y, ep_type, fp_idx = self.next_episode(
                r_anchor, r_local, r_bridge, r_long)
            if x is None:
                continue
            xs.append(x)
            ys.append(y)
            ep_types.append(ep_type)
            fp_idxs.append(fp_idx)

        if not xs:
            return None, None, [], []

        T0 = xs[0].shape[0]
        for x in xs:
            if x.shape[0] != T0:
                raise ValueError(
                    f"Episode length mismatch: expected {T0}, got {x.shape[0]}")

        x_batch = torch.stack(xs, dim=0)   # (B, T, N)
        y_batch = torch.stack(ys, dim=0)   # (B, T, N)
        return x_batch, y_batch, ep_types, fp_idxs


# ── Raw data manager (original Chua trajectory) ───────────────────────────────

class RawDataManager:
    """Manager sampling random segments from the raw Chua time series.
    Same next_batch interface as AugOnlyQueueManager."""

    def __init__(self, raw_data: np.ndarray, segment_len: int):
        # raw_data: (T, N)  float32
        self.raw_data    = torch.tensor(raw_data)
        self.segment_len = segment_len
        self.T           = raw_data.shape[0]
        # per-episode start times s on the source trajectory of the latest
        # batch. Used for time alignment with the teacher cache (cache[s : s+L]).
        self.last_starts = None

    def next_batch(self, r_anchor, r_local, r_bridge, r_long, batch_episodes):
        """Ratio args ignored; every episode is a random raw-data segment."""
        T   = self.segment_len
        max_start = self.T - T - 1
        if max_start <= 0:
            return None, None, [], []

        xs, ys, ep_types, fp_idxs = [], [], [], []
        starts = []
        for _ in range(batch_episodes):
            s = np.random.randint(0, max_start)
            starts.append(int(s))
            xs.append(self.raw_data[s    : s + T    ])   # (T, N)
            ys.append(self.raw_data[s + 1: s + T + 1])   # (T, N)
            ep_types.append('long')
            fp_idxs.append(-1)
        self.last_starts = starts

        x_batch = torch.stack(xs, dim=0)   # (B, T, N)
        y_batch = torch.stack(ys, dim=0)   # (B, T, N)
        return x_batch, y_batch, ep_types, fp_idxs


# ── Training loop ─────────────────────────────────────────────────────────────

def train_aug_only(
    model, mgr, orig_data_for_eval,
    n_interleave, alpha,
    num_epochs, steps_per_epoch,
    lr_start, lr_end,
    r_anchor, r_local, r_bridge, r_long,
    ssi, tag, P=None,
    input_sigma=0.0, state_sigma=0.0,
    gen_data_std=None,
    target_fp_count=5, fp_count_weight=5.0,
    gt_fps_obs=None, gt_fp_types=None,
    no_periodic_eval=False,
    lr_fixed=False,
    batch_episodes=16,
    lambda_target_fp_residual=0.0,
    target_fp_residual_ramp_epochs=200,
    fp_hidden_lr_mult=1.0,
    distill_mode='none',
    distill_lambda=0.0,
    symbol_margin=1.0,
    retained_global_idx=None,
    retained_local_idx=None,
    teacher_pre_cache=None,
    teacher_bits_cache=None,
    teacher_out_cache=None,
    distill_auto_lambda=False,
    distill_grad_ratio_alpha=0.1,
    distill_calibration_batches=5,
    distill_grad_eps=1e-12,
    distill_sync_scope='none',
    distill_sync_interval=128,
):
    model.train()
    loss_fn   = nn.MSELoss()
    N = model.N
    device = next(model.parameters()).device
    # print the parent-state sync debug log only for the first forward
    _sync_debug_left = [1 if distill_sync_scope != 'none' else 0]

    # ── teacher distillation init (autonomous free-run cache scheme) ──
    # distill modes never use raw Chua data as the teacher target:
    #   readout target  = teacher free-run output (teacher_out_cache)
    #   internal/symbol = teacher free-run preactivations / bits (same trajectory, same phase)
    # the student also rolls out fully autonomously (no teacher forcing).
    # only the initial condition syncs to the teacher: student state = cache[s] at segment start s.
    # the training loop never runs the teacher model's forward; it only
    # reads cache slices. The cache is shared per source checkpoint across
    # all candidates / distill modes / retrain seeds.
    use_distill = (distill_mode != 'none')
    if use_distill:
        assert teacher_pre_cache is not None, \
            "distill requires teacher_pre_cache (build_or_load_teacher_cache)"
        assert teacher_out_cache is not None, \
            "distill requires teacher_out_cache (free-run output target)"
        assert (not teacher_pre_cache.requires_grad
                and not teacher_out_cache.requires_grad), \
            "teacher caches must not carry gradients"
        assert hasattr(mgr, 'last_starts'), \
            "distill needs a manager that records segment start times " \
            "(RawDataManager of data_mode=original) — augmented mode cannot time-align"
        if distill_mode in ('retained_preactivation', 'symbol'):
            assert retained_global_idx, "retained ReLUs required"
        if distill_mode == 'symbol':
            assert teacher_bits_cache is not None, "symbol mode requires the bits cache"
        _ridx = (torch.as_tensor(retained_global_idx, dtype=torch.long)
                 if retained_global_idx else None)
        _rloc = (torch.as_tensor(retained_local_idx, dtype=torch.long)
                 if retained_local_idx else None)
        _arangeL = None   # arange of segment length (fixed at the first batch)
        print(f"  [distill] mode={distill_mode}  lambda={distill_lambda:g}"
              + (f"  margin={symbol_margin:g}" if distill_mode == 'symbol' else "")
              + (f"  retained_global_idx={retained_global_idx}"
                 if _ridx is not None else "  (all M units)")
              + f"  teacher_cache=len {teacher_pre_cache.shape[0]} "
              + f"(autonomous free-run / no raw-data target / no TF / "
              + f"no online teacher)")
    else:
        _ridx = None
        _rloc = None

    def _distill_forward(x_batch):
        """One distillation forward; returns (loss_out, loss_aux).

        Both the training loop and the lambda calibration use this function
        (single implementation -> calibration uses exactly the training protocol).
        Raw data is never a target: mgr's x_batch is only used to obtain the
        segment start times s; the values themselves are never read (TF disabled).
        """
        nonlocal _arangeL
        starts = mgr.last_starts
        assert starts is not None and len(starts) == x_batch.shape[0]
        L = x_batch.shape[1]
        if _arangeL is None or _arangeL.numel() != L:
            _arangeL = torch.arange(L, dtype=torch.long)
        starts_t = torch.as_tensor(starts, dtype=torch.long)

        # student initial state = teacher free-run state cache[s] at time s
        # (same state space as the teacher, so direct sync; treated as constant)
        z0 = teacher_pre_cache[starts_t].to(device)

        # target times: student states[t] = time s+t+1
        #            → teacher target = cache[s+1 : s+L+1]
        idx = starts_t.unsqueeze(1) + 1 + _arangeL   # (B, L)

        # readout target = teacher free-run output (not raw data)
        teacher_y = teacher_out_cache[idx].to(device)   # (B, L, N)

        # the student rolls out fully autonomously:
        #   n_interleave=L+1 structurally disables teacher forcing.
        # only when distill_sync_scope != 'none', add sparse hard sync to the
        # parent cache's same-time states cache[s+t] (t=0..L-1)
        # (raw data is not used for sync either).
        parent_sync_traj = None
        if distill_sync_scope != 'none':
            sync_idx = starts_t.unsqueeze(1) + _arangeL      # (B, L): times s+t
            parent_sync_traj = teacher_pre_cache[sync_idx].to(device)
        _dbg = _sync_debug_left[0] > 0
        if _dbg:
            _sync_debug_left[0] -= 1
        loss_out, student_states = run_episode_batch(
            model, x_batch, teacher_y, loss_fn, alpha,
            n_interleave=L + 1,
            z_init_batch=z0,
            input_noise_scale=None,
            state_sigma=state_sigma,
            return_states=True,
            parent_sync_traj=parent_sync_traj,
            sync_scope=distill_sync_scope,
            sync_interval=distill_sync_interval,
            sync_n_readout=N,
            sync_retained_idx=retained_global_idx,
            sync_debug=_dbg)

        # internal/symbol targets use the same trajectory and times idx (same phase)
        if distill_mode == 'symbol':
            t_bits = (teacher_bits_cache[idx][..., _rloc]
                      .to(device))                 # (B, L, |R|) uint8
            loss_aux = compute_distill_loss(
                'symbol', student_states, None,
                _ridx, symbol_margin, teacher_bits=t_bits)
        else:
            t_pre = teacher_pre_cache[idx].to(device)   # (B, L, M)
            loss_aux = compute_distill_loss(
                distill_mode, student_states, t_pre,
                _ridx, symbol_margin, n_readout=N)
        return loss_out, loss_aux

    # ── auto-lambda calibration (initial gradient-norm ratio) ──
    # decide lambda = alpha*g_out/(g_aux+eps) once before training and keep it
    # fixed for the whole run (NOT adaptive weighting like GradNorm).
    # runs before optimizer creation, no parameter updates (torch.autograd.grad only).
    # RNG state is saved -> restored, so calibration does not change the
    # actual training batch sequence / random state at all.
    distill_calibration = None
    distill_lambda_manual = distill_lambda
    if use_distill and distill_auto_lambda:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        _rng_state = (random.getstate(), np.random.get_state(),
                      torch.get_rng_state())   # no CUDA (CPU execution)
        g_out_list, g_aux_list = [], []
        for _b in range(distill_calibration_batches):
            x_b, _yb, _et, _fi = mgr.next_batch(
                r_anchor, r_local, r_bridge, r_long, batch_episodes)
            if x_b is None:
                continue
            l_out, l_aux = _distill_forward(x_b.to(device))
            # take both norms from the same forward graph (retain, then free)
            g_out_list.append(_param_grad_norm(l_out, trainable_params,
                                               retain_graph=True))
            g_aux_list.append(_param_grad_norm(l_aux, trainable_params,
                                               retain_graph=False))
        random.setstate(_rng_state[0])
        np.random.set_state(_rng_state[1])
        torch.set_rng_state(_rng_state[2])

        if not g_out_list:
            raise RuntimeError("[Distill calibration] could not fetch a batch")
        if not all(np.isfinite(g_out_list)) or not all(np.isfinite(g_aux_list)):
            raise RuntimeError(
                f"[Distill calibration] NaN/Inf gradient norm: "
                f"g_out={g_out_list} g_aux={g_aux_list}")
        g_out_med = float(np.median(g_out_list))   # median (robust to outliers)
        g_aux_med = float(np.median(g_aux_list))
        if g_aux_med < distill_grad_eps:
            raise RuntimeError(
                f"[Distill calibration] auxiliary gradient norm is near zero "
                f"(g_aux={g_aux_med:.3e}) — cannot compute the auto lambda; stopping. "
                f"Use a manual lambda (DISTILL_AUTO_LAMBDA=False) or revisit "
                f"the configuration.")
        distill_lambda = compute_auto_lambda(
            g_out_med, g_aux_med, distill_grad_ratio_alpha, distill_grad_eps)
        distill_calibration = {
            'alpha': float(distill_grad_ratio_alpha),
            'calibration_batches': int(distill_calibration_batches),
            'grad_norm_out_initial': g_out_med,
            'grad_norm_aux_initial': g_aux_med,
            'grad_norm_out_per_calibration_batch': g_out_list,
            'grad_norm_aux_per_calibration_batch': g_aux_list,
            'distill_lambda_manual': float(distill_lambda_manual),
            'distill_lambda_effective': float(distill_lambda),
        }
        print(f"  [Distill calibration]")
        print(f"    mode={distill_mode}  alpha={distill_grad_ratio_alpha:g}  "
              f"batches={distill_calibration_batches}")
        print(f"    g_out={g_out_med:.6g}  g_aux={g_aux_med:.6g}")
        print(f"    lambda_manual={distill_lambda_manual:g}  "
              f"lambda_effective={distill_lambda:.6g}  "
              f"(lambda*g_aux/g_out={distill_lambda * g_aux_med / g_out_med:.4f})")

    # ── init of the target-FP-residual auxiliary variables ──
    use_target_fp_residual = (lambda_target_fp_residual > 0.0
                              and gt_fps_obs is not None)
    fp_hidden = None
    gt_x      = None
    if use_target_fp_residual:
        gt_x = torch.tensor(gt_fps_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            z0 = gt_x @ model.B
            z0[:, :model.N] = gt_x
        fp_hidden = nn.Parameter(z0[:, model.N:].clone())
        print(f"  target FP residual: ON  lambda={lambda_target_fp_residual:g}  "
              f"ramp={target_fp_residual_ramp_epochs}  "
              f"hLR_mult={fp_hidden_lr_mult:g}  "
              f"fp_hidden shape={tuple(fp_hidden.shape)}")
    else:
        print("  target FP residual: OFF")

    # ── optimizer (includes fp_hidden when the target FP residual is on) ──
    param_groups = [{"params": list(model.parameters()), "lr": lr_start}]
    if use_target_fp_residual:
        param_groups.append({"params": [fp_hidden],
                             "lr": lr_start * fp_hidden_lr_mult})
    optimizer = torch.optim.RAdam(param_groups)
    # the teacher does not exist as an object during training (cache-based),
    # so it structurally cannot leak into the optimizer (cache tensors have requires_grad=False)

    if lr_fixed:
        scheduler = None
        print(f"  LR: fixed={lr_start:.0e}")
    else:
        gamma     = np.exp(np.log(lr_end / lr_start) / num_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(f"  LR: {lr_start:.0e} → {lr_end:.0e} (exp decay)")

    # bridge/long noise: follows the std of the bridge/long data
    # gen_data_std is a (1, N) tensor passed from run_one
    if input_sigma > 0.0 and gen_data_std is not None:
        input_noise_vec_gen = (input_sigma * gen_data_std).to(device)
        print(f"  input noise (bridge/long) per-dim std: "
              f"{input_noise_vec_gen.cpu().numpy().flatten()}")
    else:
        input_noise_vec_gen = None
    best_model     = copy.deepcopy(model)
    best_fp_hidden = None
    best_score     = float('inf')
    # keep the Dstsp-best per vis_fps value in parallel (a state_dict is ~2KB
    # at M=20). Insurance that lets us swap the best epoch without retraining
    # if target_fp_count changes later; saved via log['best_by_vis'].
    best_by_vis    = {}   # vis -> (Dstsp, epoch, state_dict[cpu])

    loss_hist  = []
    traj_loss_hist = []
    distill_loss_hist = []   # auxiliary loss (epoch mean; nan when distill off)
    orig_loss_hist    = []   # existing objective without the auxiliary loss (epoch mean)
    tfp_residual_hist = []
    tfp_per_fp_hist = []   # list of (K,) arrays, one per epoch
    tfp_weight_hist = []
    Dstsp_hist = []; DH_hist = []; eval_epochs = []
    fp_snapshots = []; fp_count_history = []
    matching_history = []
    type_counts = {'anchor': 0, 'local': 0, 'bridge': 0, 'long': 0}

    X_torch = torch.tensor(orig_data_for_eval).unsqueeze(0).to(device)

    if no_periodic_eval:
        print("  [fast mode] only Dstsp+vis_fps for best model selection")

    _tty = sys.stdout.isatty()
    with trange(num_epochs, desc="[aug_only]", dynamic_ncols=True,
                disable=not _tty) as pbar:
        for epoch in pbar:
            model.train()
            epoch_losses = []
            epoch_traj_losses = []
            epoch_aux_losses = []
            epoch_orig_losses = []
            epoch_tfp_losses = []
            epoch_tfp_weight = float('nan')

            # ── ramp weight of the target FP residual (constant per epoch) ──
            if use_target_fp_residual:
                if target_fp_residual_ramp_epochs > 0:
                    ramp = min(1.0, epoch / target_fp_residual_ramp_epochs)
                else:
                    ramp = 1.0
                epoch_tfp_weight = ramp * lambda_target_fp_residual
            else:
                ramp = 0.0

            for _ in range(steps_per_epoch):
                optimizer.zero_grad()

                x_batch, y_batch, ep_types, fp_idxs = mgr.next_batch(
                    r_anchor, r_local, r_bridge, r_long, batch_episodes)

                if x_batch is None:
                    continue

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                for ep_type in ep_types:
                    type_counts[ep_type] = type_counts.get(ep_type, 0) + 1

                if use_distill:
                    # distillation forward (same implementation as calibration: _distill_forward)
                    loss_traj, loss_aux = _distill_forward(x_batch)
                else:
                    # ── none: legacy raw-data target + existing teacher forcing ──
                    z_init_batch = build_z_init_batch(
                        model, x_batch, ep_types, fp_idxs,
                        gt_fps_obs=gt_fps_obs)

                    input_noise_scale = build_input_noise_scale(
                        ep_types, input_noise_vec_gen)

                    loss_aux = None
                    loss_traj = run_episode_batch(
                        model, x_batch, y_batch, loss_fn, alpha, n_interleave,
                        z_init_batch=z_init_batch,
                        input_noise_scale=input_noise_scale,
                        state_sigma=state_sigma)

                if use_target_fp_residual:
                    loss_fp = target_fp_residual_loss(model, gt_x, fp_hidden)
                    loss = loss_traj + ramp * lambda_target_fp_residual * loss_fp
                else:
                    loss_fp = None
                    loss = loss_traj

                # the existing objective (= loss_original) ends here; distill adds
                # loss_total = loss_original + lambda_aux * loss_aux
                loss_original = loss
                if loss_aux is not None:
                    loss = loss + distill_lambda * loss_aux

                if not torch.isfinite(loss):
                    warnings.warn("Non-finite batch loss, skip")
                    optimizer.zero_grad(); continue

                loss.backward()
                clip_params = [p for g in param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=10.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                epoch_traj_losses.append(loss_traj.item())
                epoch_orig_losses.append(loss_original.item())
                if loss_aux is not None:
                    epoch_aux_losses.append(loss_aux.item())
                if loss_fp is not None:
                    epoch_tfp_losses.append(loss_fp.item())

            if scheduler is not None:
                scheduler.step()

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            loss_hist.append(avg)
            traj_loss_hist.append(
                float(np.mean(epoch_traj_losses)) if epoch_traj_losses else float('nan'))
            distill_loss_hist.append(
                float(np.mean(epoch_aux_losses)) if epoch_aux_losses else float('nan'))
            orig_loss_hist.append(
                float(np.mean(epoch_orig_losses)) if epoch_orig_losses else float('nan'))
            tfp_residual_hist.append(
                float(np.mean(epoch_tfp_losses)) if epoch_tfp_losses else float('nan'))
            tfp_weight_hist.append(epoch_tfp_weight)

            # per-FP residual (K,) — no gradient, eval only
            if use_target_fp_residual:
                with torch.no_grad():
                    z_fp_eval = torch.cat([gt_x, fp_hidden], dim=1)
                    z_next_eval = model(z_fp_eval)
                    per_fp = ((z_next_eval - z_fp_eval) ** 2).mean(dim=1).cpu().numpy()
                tfp_per_fp_hist.append(per_fp)

            # extra postfix info only when the target FP residual is enabled
            tfp_postfix = {}
            if use_target_fp_residual:
                tfp_postfix = {
                    'loss_traj': f'{traj_loss_hist[-1]:.3e}',
                    'loss_fp':   f'{tfp_residual_hist[-1]:.3e}',
                    'fp_w':      f'{epoch_tfp_weight:.2e}',
                }
            if use_distill:
                tfp_postfix['loss_aux'] = f'{distill_loss_hist[-1]:.3e}'

            if no_periodic_eval:
                if epoch % ssi == 0 or epoch == num_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        z_free = predict_free_sequence(
                            model, X_torch[:, 0, :], 10000)[0]
                    Dstsp   = state_space_divergence_binning(z_free[:, :N], X_torch[0])
                    bits    = lrf.convert_to_bits(z_free.detach().cpu().numpy()[:, -model.P:])
                    _, uniq = lrf.unique_regions_crossed(bits, model.M)
                    fp_data = analyze_fixed_points_continuous(model, uniq)
                    vis_fps = sum(1 for v in fp_data.values() if v['type'] == 'real')
                    score   = float(Dstsp) + fp_count_weight * abs(vis_fps - target_fp_count)
                    if score < best_score:
                        best_score = score
                        best_model = copy.deepcopy(model)
                        if use_target_fp_residual:
                            best_fp_hidden = fp_hidden.detach().clone()
                    _bv = best_by_vis.get(vis_fps)
                    if _bv is None or float(Dstsp) < _bv[0]:
                        best_by_vis[vis_fps] = (
                            float(Dstsp), int(epoch),
                            {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()})

                    Dstsp_hist.append(float(Dstsp))
                    DH_hist.append(float('nan'))
                    eval_epochs.append(epoch)

                    snap = collect_fp_snapshot(
                        model, z_free.detach().cpu().numpy(), epoch,
                        analyze_fixed_points_continuous, delta_t=DELTA_T)
                    fp_snapshots.append(snap)
                    fp_count_history.append(snap['counts'])
                    if gt_fps_obs is not None:
                        matching_history.append(compute_fp_matching(snap, gt_fps_obs))

                    pbar.set_postfix(loss=f'{avg:.3e}', Dstsp=f'{Dstsp:.3f}',
                                     vis_fps=vis_fps, score=f'{score:.3f}',
                                     **tfp_postfix)
                    if not _tty:
                        _kv = {'P': str(P) if P is not None else '?', 'ep': f'{epoch}/{num_epochs}', 'loss': f'{avg:.3e}',
                               'Dstsp': f'{Dstsp:.3f}', 'vis_fps': str(vis_fps),
                               'score': f'{score:.3f}', **tfp_postfix}
                        print('[PROGRESS] ' + ' '.join(f'{k}={v}' for k, v in _kv.items()),
                              flush=True)
                else:
                    pbar.set_postfix(loss=f'{avg:.3e}', **tfp_postfix)
            else:
                if epoch % ssi == 0 or epoch == num_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        z_free = predict_free_sequence(
                            model, X_torch[:, 0, :], 10000)[0]
                    Dstsp = state_space_divergence_binning(z_free[:, :N], X_torch[0])
                    DH    = power_spectrum_error(z_free[:, :N], X_torch[0, :10000, :])
                    Dstsp_hist.append(float(Dstsp))
                    DH_hist.append(float(DH))
                    eval_epochs.append(epoch)

                    snap = collect_fp_snapshot(
                        model, z_free.detach().cpu().numpy(), epoch,
                        analyze_fixed_points_continuous, delta_t=DELTA_T)
                    fp_snapshots.append(snap)
                    fp_count_history.append(snap['counts'])

                    if gt_fps_obs is not None:
                        matching_history.append(compute_fp_matching(snap, gt_fps_obs))

                    vis_fps = snap['counts'].get('vis_total', 0)
                    score   = float(Dstsp) + fp_count_weight * abs(vis_fps - target_fp_count)
                    if score < best_score:
                        best_score = score
                        best_model = copy.deepcopy(model)
                        if use_target_fp_residual:
                            best_fp_hidden = fp_hidden.detach().clone()
                    _bv = best_by_vis.get(vis_fps)
                    if _bv is None or float(Dstsp) < _bv[0]:
                        best_by_vis[vis_fps] = (
                            float(Dstsp), int(epoch),
                            {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()})

                    pbar.set_postfix(loss=f'{avg:.3e}', Dstsp=f'{Dstsp:.3f}',
                                     vis_fps=vis_fps, score=f'{score:.3f}',
                                     **tfp_postfix)
                    if not _tty:
                        _kv = {'P': str(P) if P is not None else '?', 'ep': f'{epoch}/{num_epochs}', 'loss': f'{avg:.3e}',
                               'Dstsp': f'{Dstsp:.3f}', 'vis_fps': str(vis_fps),
                               'score': f'{score:.3f}', **tfp_postfix}
                        print('[PROGRESS] ' + ' '.join(f'{k}={v}' for k, v in _kv.items()),
                              flush=True)
                else:
                    pbar.set_postfix(loss=f'{avg:.3e}', **tfp_postfix)

    model.load_state_dict(best_model.state_dict())
    if use_target_fp_residual and best_fp_hidden is not None:
        with torch.no_grad():
            fp_hidden.copy_(best_fp_hidden)
    print(f"\n  Episode type counts: {type_counts}")

    # target-FP-residual auxiliaries (None / empty arrays when disabled)
    if use_target_fp_residual:
        target_fp_gt_x        = gt_x.detach().cpu().numpy().astype(np.float32)
        target_fp_hidden_final = fp_hidden.detach().cpu().numpy().astype(np.float32)
    else:
        target_fp_gt_x        = np.zeros((0, 0), dtype=np.float32)
        target_fp_hidden_final = np.zeros((0, 0), dtype=np.float32)

    return model, {
        'loss_history':      np.array(loss_hist,   dtype=np.float32),
        'distill_loss_history':  np.array(distill_loss_hist, dtype=np.float32),
        'original_loss_history': np.array(orig_loss_hist,    dtype=np.float32),
        'distill_calibration':      distill_calibration,   # None when auto-lambda is off
        'distill_lambda_effective': float(distill_lambda),
        'traj_loss_history':            np.array(traj_loss_hist,    dtype=np.float32),
        'target_fp_residual_history':   np.array(tfp_residual_hist, dtype=np.float32),
        'target_fp_residual_per_fp_history': (
            np.stack(tfp_per_fp_hist, axis=0).astype(np.float32)
            if tfp_per_fp_hist else np.zeros((0, 0), dtype=np.float32)
        ),
        'target_fp_residual_weight_history': np.array(tfp_weight_hist, dtype=np.float32),
        'Dstsp_history':     np.array(Dstsp_hist,  dtype=np.float32),
        'DH_history':        np.array(DH_hist,     dtype=np.float32),
        'eval_epochs':       np.array(eval_epochs, dtype=np.int32),
        'fp_snapshots':      fp_snapshots,
        'fp_count_history':  fp_count_history,
        'matching_history':  matching_history,
        'type_counts':       type_counts,
        'use_target_fp_residual':  use_target_fp_residual,
        'target_fp_gt_x':          target_fp_gt_x,
        'target_fp_hidden_final':  target_fp_hidden_final,
        'best_by_vis':             best_by_vis,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(log, tag, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    epochs = np.arange(len(log['loss_history']))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(epochs, log['loss_history'], lw=2)
    ax.set_xlabel('epoch'); ax.set_ylabel('total loss')
    ax.set_title(f'Training loss (aug_only)  [{tag}]')
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{tag}_loss.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if len(log['eval_epochs']) > 0:
        ev = log['eval_epochs']
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(ev, log['Dstsp_history'], 'o-', ms=3, lw=1.5, label='$D_{stsp}$')
        axes[0].set_ylabel('$D_{stsp}$'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ev, log['DH_history'], 'o-', ms=3, lw=1.5,
                     color='tomato', label='$D_H$')
        axes[1].set_ylabel('$D_H$'); axes[1].set_xlabel('epoch')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle(f'DS metrics (aug_only)  [{tag}]', fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{tag}_metrics.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(args, P, orig_data,
            anchor_trajs, anchor_meta,
            local_trajs,  local_meta,
            bridge_trajs, bridge_lengths, bridge_meta,
            long_trajs,
            K, gt_fps_obs, gt_fp_types,
            fig_dir, res_dir, csv_path,
            raw_data=None,
            init_model_path='',
            source_checkpoint='',
            reduction_spec='',
            candidate_id=None):
    """
    When raw_data is given: data_mode='original' (train on raw data).
    When None: data_mode='augmented' (legacy augmented-episode training).

    When args.retrain_reduced=True, reduced-model retraining mode:
      build a fixed relu_mask from candidates[candidate_id] of the
      reduction_spec (minimal_symbol_reductions.json) and retrain with the
      source_checkpoint weights as init. One run_one call = one candidate.
    """

    data_mode = 'original' if raw_data is not None else 'augmented'

    # ── reduced-model retraining: load the spec & build the fixed relu_mask ──
    # Index convention (shared with graph reduction):
    #   deleted_relu_local_index j
    #       = j-th element among the last P_original activation slots
    #       → global hidden index = M - P_original + j
    retrain_reduced = bool(getattr(args, 'retrain_reduced', False))
    relu_mask = None
    red_spec = None
    red_cand = None
    retrain_suffix = ''
    if retrain_reduced:
        if not source_checkpoint or not reduction_spec:
            raise ValueError(
                "--retrain_reduced requires source_checkpoint and reduction_spec")
        if candidate_id is None:
            raise ValueError("--retrain_reduced requires candidate_id")
        if not os.path.exists(source_checkpoint):
            raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")
        if not os.path.exists(reduction_spec):
            raise FileNotFoundError(f"reduction spec not found: {reduction_spec}")
        with open(reduction_spec) as f:
            red_spec = json.load(f)
        if 'candidates' not in red_spec or 'minimal_num_clusters' not in red_spec:
            raise ValueError(
                f"{reduction_spec} is the old single-candidate format or invalid. "
                f"Re-run graph reduction to generate "
                f"minimal_symbol_reductions.json.")
        if int(red_spec['M']) != args.M:
            raise ValueError(
                f"reduction spec M={red_spec['M']} != model M={args.M} ({reduction_spec})")
        if int(red_spec['P_original']) != P:
            raise ValueError(
                f"reduction spec P_original={red_spec['P_original']} != P={P} "
                f"({reduction_spec})")
        red_cand = next((c for c in red_spec['candidates']
                         if int(c['candidate_id']) == int(candidate_id)), None)
        if red_cand is None:
            raise ValueError(
                f"candidate_id={candidate_id} not present in the spec "
                f"(available: {[c['candidate_id'] for c in red_spec['candidates']]}) "
                f"({reduction_spec})")
        minimal_num_clusters = int(red_spec['minimal_num_clusters'])
        deleted_local = sorted(int(j) for j in red_cand['deleted_relu_local_indices'])
        if any(j < 0 or j >= P for j in deleted_local):
            raise ValueError(f"deleted_relu_local_indices out of range [0,{P}): {deleted_local}")
        deleted_set = set(deleted_local)
        relu_mask = [j not in deleted_set for j in range(P)]   # True=ReLU, False=identity
        P_eff = sum(relu_mask)
        if 'P_effective' in red_cand and int(red_cand['P_effective']) != P_eff:
            raise ValueError(
                f"candidate {candidate_id} P_effective={red_cand['P_effective']} does "
                f"not match the mask-derived P_eff={P_eff} ({reduction_spec})")
        # separate the source seed (args.seed, tag base) from the retraining
        # seed (training RNG). minsym / cand / rseed are part of the filename so
        # multiple candidates x retrain seeds of one source never overwrite each other.
        retrain_seed = (args.retrain_seed
                        if getattr(args, 'retrain_seed', None) is not None
                        else args.seed)
        # include the distillation mode in the suffix so checkpoints / logs /
        # metadata never collide across modes ('none' is empty = legacy naming)
        distill_mode = getattr(args, 'distill_mode', 'none')
        _ga_tag = distill_autolambda_tag(
            getattr(args, 'distill_auto_lambda', False), distill_mode,
            getattr(args, 'distill_grad_ratio_alpha', 0.1))
        # weighting-ablation runs get unique tags; no collision with existing outputs
        _name_tag = getattr(args, 'distill_name_tag', '')
        _mode_tag = (f"_{_name_tag}" if _name_tag
                     else f"{DISTILL_TAGS[distill_mode]}{_ga_tag}")
        retrain_suffix = (f"_minsym{minimal_num_clusters}_cand{int(candidate_id)}"
                          f"_reducedp{P_eff}{_mode_tag}"
                          f"_rseed{retrain_seed}")
    else:
        retrain_seed = None
        distill_mode = getattr(args, 'distill_mode', 'none')
        if distill_mode != 'none':
            raise ValueError(
                "--distill_mode is only available with --retrain_reduced")

    total = args.r_anchor + args.r_local + args.r_bridge + args.r_long
    r_a = args.r_anchor / total
    r_l = args.r_local  / total
    r_b = args.r_bridge / total
    r_lo= args.r_long   / total

    lr_tag = (f"_lrfix{args.lr_start:.0e}"
              if args.lr_fixed
              else f"_lr{args.lr_start:.0e}-{args.lr_end:.0e}")
    tfp_tag = (f"_tfp{args.lambda_target_fp_residual:g}"
               f"_ramp{args.target_fp_residual_ramp_epochs}"
               f"_hLR{args.fp_hidden_lr_mult:g}")

    init_suffix = getattr(args, 'init_tag_suffix', '_fromTeacher') if init_model_path else ''

    if data_mode == 'original':
        tag = (f"{args.tag_prefix}_orig"
               f"_nint{args.n_interleave}"
               f"_bs{args.batch_episodes}"
               f"_sig{args.input_sigma}"
               f"{lr_tag}"
               f"{tfp_tag}"
               f"_m{args.M}_p{P}_seed{args.seed}{init_suffix}{retrain_suffix}")
    else:
        tag = (f"{args.tag_prefix}_aug_only"
               f"_a{r_a:.2f}_l{r_l:.2f}_b{r_b:.2f}_lo{r_lo:.2f}"
               f"_nint{args.n_interleave}"
               f"_bs{args.batch_episodes}"
               f"_sig{args.input_sigma}"
               f"{lr_tag}"
               f"{tfp_tag}"
               f"_m{args.M}_p{P}_seed{args.seed}{init_suffix}{retrain_suffix}")

    print(f"\n{'='*65}\n  {tag}\n{'='*65}")
    if data_mode == 'original':
        print(f"  data_mode: original  (raw Chua trajectory)")
    else:
        print(f"  ratio  anchor:{r_a:.2f}  local:{r_l:.2f}  "
              f"bridge:{r_b:.2f}  long:{r_lo:.2f}")

    # in retrain mode the training RNG comes from retrain_seed (args.seed is the source seed)
    train_seed = retrain_seed if retrain_reduced else args.seed
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)

    N     = orig_data.shape[-1]
    _run_t0 = time.perf_counter()
    model = AL_RNN(M=args.M, P=P, N=N, relu_mask=relu_mask)
    init_eval = None
    teacher_pre_cache  = None
    teacher_bits_cache = None
    teacher_out_cache  = None
    teacher_cache_info = None
    cache_gen_time = 0.0
    distill_lambda = 0.0
    retained_global_idx = []
    retained_local_idx  = []
    distill_weighting = getattr(args, 'distill_weighting', 'default')
    lambda_formula = None
    _sync_scope = getattr(args, 'distill_sync_scope', 'none')
    _sync_n_coords = 0
    if retrain_reduced:
        # load the source checkpoint BEFORE optimizer creation (before train_aug_only).
        # every parameter except the activation type is inherited from the source.
        src_state = torch.load(source_checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(src_state, strict=False)
        # the only tolerated mismatch: the source (old-format ckpt) lacks the relu_mask buffer
        missing = [k for k in missing if k != 'relu_mask']
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint key mismatch loading {source_checkpoint}: "
                f"missing={missing} unexpected={list(unexpected)}")
        expected_mask = torch.tensor(relu_mask, dtype=torch.bool)
        if not torch.equal(model.relu_mask.detach().cpu(), expected_mask):
            raise RuntimeError(
                f"relu_mask mismatch: checkpoint/spec inconsistency "
                f"(spec={reduction_spec}, ckpt={source_checkpoint})")
        # sanity: every non-activation parameter must equal the source exactly
        for name, p_t in model.named_parameters():
            if not torch.allclose(p_t.detach().cpu(), src_state[name].cpu()):
                raise RuntimeError(f"parameter {name} differs from source checkpoint")
        deleted_local = [j for j in range(P) if not relu_mask[j]]
        print(f"  [retrain_reduced] source ckpt : {source_checkpoint}")
        print(f"  [retrain_reduced] spec        : {reduction_spec}")
        print(f"  [retrain_reduced] source_seed={args.seed}  retrain_seed={retrain_seed}  "
              f"candidate_id={candidate_id}")
        print(f"  [retrain_reduced] minimal_num_clusters={red_spec['minimal_num_clusters']}  "
              f"candidate num_clusters={red_cand['num_clusters']}  "
              f"edge_types={red_cand['edge_types']}  is_leaf={red_cand.get('is_leaf')}")
        print(f"  [retrain_reduced] deleted local ReLU={deleted_local}  "
              f"→ hidden idx={[args.M - P + j for j in deleted_local]}")
        print(f"  [retrain_reduced] relu_mask={[int(b) for b in relu_mask]}  "
              f"P_original={P}  P_effective={model.P_effective}")

        # performance right before retraining: source model and the reduced model right after ReLU->identity
        src_model = AL_RNN(M=args.M, P=P, N=N)
        src_model.load_state_dict({k: v for k, v in src_state.items()
                                   if k != 'relu_mask'})
        d_src, v_src = evaluate_model_quick(src_model, orig_data)
        d_red, v_red = evaluate_model_quick(model, orig_data)
        init_eval = {
            'source_Dstsp':        d_src,
            'source_vis_fps':      v_src,
            'reduced_init_Dstsp':  d_red,
            'reduced_init_vis_fps': v_red,
        }
        print(f"  [retrain_reduced] init eval: "
              f"source Dstsp={d_src:.4f} vis_fps={v_src}  |  "
              f"reduced (pre-retrain) Dstsp={d_red:.4f} vis_fps={v_red}")

        # ── teacher distillation setup (cache scheme) ──
        # teacher target = precomputed cache of the source over-parameterized
        # model's global trajectory; built once per source checkpoint and
        # shared by all candidates / distill modes / retrain seeds.
        # the teacher model is never held or forwarded during training.
        if distill_mode != 'none':
            if raw_data is None:
                raise ValueError(
                    "distillation is only available with data_mode=original "
                    "(time alignment between teacher cache and training segments)")
            _lam_by_mode = {
                'full_preactivation':     args.distill_lambda_full,
                'retained_preactivation': args.distill_lambda_retained,
                'symbol':                 args.distill_lambda_symbol,
            }
            distill_lambda = float(_lam_by_mode[distill_mode])
            # ── weighting ablation: swap ONLY the lambda-decision rule ──
            # (loss definition, normalization, teacher, training procedure unchanged)
            if distill_weighting != 'default':
                assert P == 10, \
                    "the weighting ablation is only defined for P_parent=10 runs"
                if distill_weighting == 'coord_equal_fixed':
                    assert not getattr(args, 'distill_auto_lambda', False), \
                        "coord_equal_fixed requires auto-lambda OFF"
                    if distill_mode == 'retained_preactivation':
                        distill_lambda = P_eff / N
                        lambda_formula = 'P_eff / N'
                    elif distill_mode == 'full_preactivation':
                        distill_lambda = (args.M - N) / N
                        lambda_formula = '(M - N) / N'
                    else:
                        raise ValueError(
                            f"coord_equal_fixed is for preactivation modes only "
                            f"(got {distill_mode})")
                elif distill_weighting == 'fixed_lambda1':
                    assert not getattr(args, 'distill_auto_lambda', False), \
                        "fixed_lambda1 requires auto-lambda OFF"
                    distill_lambda = 1.0
                    lambda_formula = '1.0'
                elif distill_weighting == 'grad_auto_alpha1':
                    assert getattr(args, 'distill_auto_lambda', False), \
                        "grad_auto_alpha1 requires auto-lambda ON"
                    assert abs(getattr(args, 'distill_grad_ratio_alpha', 0.1)
                               - 1.0) < 1e-12, "alpha must be 1.0"
                    lambda_formula = 'alpha * g_out / (g_aux + eps)'
                print(f"  [ablation] weighting={distill_weighting}  "
                      f"lambda_manual={distill_lambda:.6g}  "
                      f"formula={lambda_formula}  "
                      f"(N={N}, M={args.M}, P_eff={P_eff})")
            # global hidden indices of the retained ReLUs (= M - P_original + local index)
            retained_global_idx = [args.M - P + j for j in range(P) if relu_mask[j]]
            retained_local_idx  = [j for j in range(P) if relu_mask[j]]

            # ── pre-validation of the parent-state sync scope ablation ──
            _sync_scope = getattr(args, 'distill_sync_scope', 'none')
            if _sync_scope != 'none':
                assert _sync_scope in ('readout', 'readout_retained', 'full')
                # readout and retained coordinates are disjoint (guaranteed by the architecture)
                assert not (set(range(N)) & set(retained_global_idx)), \
                    "readout indices overlap with retained indices"
                _sync_n_coords = {'readout': N,
                                  'readout_retained': N + P_eff,
                                  'full': args.M}[_sync_scope]
                print(f"  [parent-sync] scope={_sync_scope}  "
                      f"interval={getattr(args, 'distill_sync_interval', 128)}  "
                      f"source=parent_autonomous_cache  "
                      f"synced_coords={_sync_n_coords}  "
                      f"(N={N}, P_eff={P_eff}, M={args.M}, "
                      f"R={retained_global_idx})")
            else:
                _sync_n_coords = 0

            _t0 = time.perf_counter()
            teacher_pre_cache, teacher_bits_cache, teacher_out_cache, \
                _cmeta, _hit = build_or_load_teacher_cache(
                    source_checkpoint, args.raw_data_path, raw_data,
                    args.M, P, N,
                    cache_dir=args.teacher_cache_dir,
                    force_rebuild=args.force_rebuild_teacher_cache)
            cache_gen_time = 0.0 if _hit else (time.perf_counter() - _t0)
            teacher_cache_info = {
                'teacher_cache_path': teacher_cache_path(
                    args.teacher_cache_dir, source_checkpoint),
                'teacher_cache_version': TEACHER_CACHE_VERSION,
                'teacher_cache_hit': bool(_hit),
                'teacher_cache_source_checkpoint': source_checkpoint,
                'teacher_cache_trajectory_length': int(teacher_pre_cache.shape[0]),
            }
            print(f"  [teacher cache] {'HIT' if _hit else f'BUILD ({cache_gen_time:.1f}s)'}"
                  f"  {teacher_cache_info['teacher_cache_path']}")
    elif init_model_path:
        state = torch.load(init_model_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"  Init weights from: {init_model_path}")

    if data_mode == 'original':
        mgr = RawDataManager(raw_data, segment_len=args.segment_len)
        gen_data_std = torch.tensor(raw_data.std(axis=0)).unsqueeze(0)  # (1, N)
        print(f"  raw_data std: {gen_data_std.numpy().flatten()}")
        # original mode has no GT FP information
        gt_fps_obs  = None
        gt_fp_types = None
    else:
        mgr = AugOnlyQueueManager(
            anchor_trajs, anchor_meta,
            local_trajs,  local_meta,
            bridge_trajs, bridge_lengths, bridge_meta,
            long_trajs,
            K, N,
            max_bridge_len = args.max_bridge_len,
            segment_len    = args.segment_len)

        # std of the bridge/long data, shape (1, N)
        gen_parts = [bridge_trajs[:, :, :N].reshape(-1, N)]
        if long_trajs is not None:
            gen_parts.append(long_trajs[:, :, :N].reshape(-1, N))
        gen_obs     = np.concatenate(gen_parts, axis=0).astype(np.float32)
        gen_data_std = torch.tensor(gen_obs.std(axis=0)).unsqueeze(0)  # (1, N)
        print(f"  gen_data std (bridge+long): {gen_data_std.numpy().flatten()}")

    _train_t0 = time.perf_counter()
    model, log = train_aug_only(
        model                = model,
        mgr                  = mgr,
        orig_data_for_eval   = orig_data,
        n_interleave         = args.n_interleave,
        alpha                = args.alpha,
        num_epochs           = args.num_epochs,
        steps_per_epoch      = args.steps_per_epoch,
        lr_start             = args.lr_start,
        lr_end               = args.lr_end,
        r_anchor             = r_a,
        r_local              = r_l,
        r_bridge             = r_b,
        r_long               = r_lo,
        ssi                  = args.ssi,
        tag                  = tag,
        P                    = P,
        input_sigma          = args.input_sigma,
        state_sigma          = args.state_sigma,
        gen_data_std         = gen_data_std,
        target_fp_count      = args.target_fp_count,
        fp_count_weight      = args.fp_count_weight,
        gt_fps_obs           = gt_fps_obs,
        gt_fp_types          = gt_fp_types,
        no_periodic_eval     = args.no_periodic_eval,
        lr_fixed             = args.lr_fixed,
        batch_episodes       = args.batch_episodes,
        lambda_target_fp_residual      = args.lambda_target_fp_residual,
        target_fp_residual_ramp_epochs = args.target_fp_residual_ramp_epochs,
        fp_hidden_lr_mult              = args.fp_hidden_lr_mult,
        distill_mode         = distill_mode,
        distill_lambda       = distill_lambda,
        symbol_margin        = getattr(args, 'symbol_margin', 1.0),
        retained_global_idx  = retained_global_idx,
        retained_local_idx   = retained_local_idx,
        teacher_pre_cache    = teacher_pre_cache,
        teacher_bits_cache   = teacher_bits_cache,
        teacher_out_cache    = teacher_out_cache,
        distill_auto_lambda        = getattr(args, 'distill_auto_lambda', False),
        distill_grad_ratio_alpha   = getattr(args, 'distill_grad_ratio_alpha', 0.1),
        distill_calibration_batches= getattr(args, 'distill_calibration_batches', 5),
        distill_grad_eps           = getattr(args, 'distill_grad_eps', 1e-12),
        distill_sync_scope         = getattr(args, 'distill_sync_scope', 'none'),
        distill_sync_interval      = getattr(args, 'distill_sync_interval', 128),
    )
    training_time = time.perf_counter() - _train_t0

    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/{tag}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # Dstsp-best per vis value (insurance for retargeting; a few KB/run)
    if log.get('best_by_vis'):
        bbv_path = f"models/{tag}_bestbyvis.pth"
        torch.save({v: dict(Dstsp=d, epoch=e, state_dict=sd)
                    for v, (d, e, sd) in log['best_by_vis'].items()},
                   bbv_path)
        print(f"  best-by-vis ({sorted(log['best_by_vis'])}) → {bbv_path}")

    if retrain_reduced:
        # sanity: the relu_mask stayed fixed throughout training
        expected_mask = torch.tensor(relu_mask, dtype=torch.bool)
        if not torch.equal(model.relu_mask.detach().cpu(), expected_mask):
            raise RuntimeError("relu_mask changed during training (must be fixed)")
        # ── post-hoc weighting-ablation assertions (do not silence misconfigured runs) ──
        if distill_weighting != 'default':
            _lam_eff = float(log.get('distill_lambda_effective', distill_lambda))
            _tol = 1e-9
            if distill_weighting == 'coord_equal_fixed':
                _expect = (P_eff / N if distill_mode == 'retained_preactivation'
                           else (args.M - N) / N)
                assert abs(_lam_eff - _expect) < _tol, \
                    f"lambda_eff={_lam_eff} != {lambda_formula}={_expect}"
                assert log.get('distill_calibration') is None
            elif distill_weighting == 'fixed_lambda1':
                assert _lam_eff == 1.0
                assert log.get('distill_calibration') is None
            elif distill_weighting == 'grad_auto_alpha1':
                assert log.get('distill_calibration') is not None
                assert abs(getattr(args, 'distill_grad_ratio_alpha', 0.1)
                           - 1.0) < 1e-12
        # save the metadata needed to restore the reduced checkpoint as a sidecar
        # JSON (relu_mask itself is also inside the state_dict as a buffer)
        deleted_local = [j for j in range(P) if not relu_mask[j]]
        meta = {
            'M': args.M,
            'P_original': P,
            'P_effective': model.P_effective,
            'candidate_id': int(candidate_id),
            'minimal_num_clusters': int(red_spec['minimal_num_clusters']),
            'num_clusters': int(red_cand['num_clusters']),
            # consensus-selected candidates carry no tree edge_types
            'edge_types': (int(red_cand['edge_types'])
                           if red_cand.get('edge_types') is not None
                           else -1),
            'num_deleted': len(deleted_local),
            'deleted_relu_local_indices': deleted_local,
            'remaining_relu_local_indices': [j for j in range(P) if relu_mask[j]],
            'deleted_hidden_indices': [args.M - P + j for j in deleted_local],
            'source_seed': args.seed,
            'retrain_seed': retrain_seed,
            'source_model_path': source_checkpoint,
            'reduction_spec_path': reduction_spec,
            'distill_mode': distill_mode,
            'distill_lambda': distill_lambda,                       # manual (reference value)
            'distill_lambda_manual': distill_lambda,
            'distill_lambda_effective': log.get('distill_lambda_effective',
                                                distill_lambda),
            'distill_auto_lambda': getattr(args, 'distill_auto_lambda', False),
            'distill_grad_ratio_alpha': getattr(args, 'distill_grad_ratio_alpha', 0.1),
            'distill_calibration_batches': getattr(args,
                                                   'distill_calibration_batches', 5),
            'distill_calibration': log.get('distill_calibration'),  # g_out/g_aux etc.
            # ── weighting-ablation metadata ──
            'weighting_scheme': distill_weighting,
            'lambda_formula': lambda_formula,
            'auto_lambda_enabled': bool(getattr(args, 'distill_auto_lambda',
                                                False)),
            'alpha': float(getattr(args, 'distill_grad_ratio_alpha', 0.1)),
            'N': N,
            'M_total': args.M,
            'P_parent': P,
            'P_eff': P_eff,
            'calibration_batch_count': int(getattr(
                args, 'distill_calibration_batches', 5)),
            # ── parent-state sync ablation metadata ──
            'sync_scope': _sync_scope,
            'sync_interval': int(getattr(args, 'distill_sync_interval', 128)),
            'sync_source': ('parent_autonomous_cache'
                            if _sync_scope != 'none' else None),
            'segment_start_sync': ('full_parent_state'
                                   if distill_mode != 'none' else None),
            'retained_indices': retained_global_idx,
            'synchronized_coordinate_count': _sync_n_coords,
            'raw_data_used_for_periodic_sync': False,
            'git_commit': _git_commit_or_none(),
            'symbol_margin': (getattr(args, 'symbol_margin', 1.0)
                              if distill_mode == 'symbol' else None),
            # provenance of the trajectory (readout) teacher signal:
            #   none -> raw Chua data / distill -> teacher's autonomous free run
            'trajectory_target': ('raw_data' if distill_mode == 'none'
                                  else 'teacher_free_run'),
            # full_preactivation = non-readout internal dims (N..M-1) only.
            # loss version marker distinguishes it from the old all-M-dims 'full'.
            'distill_loss_version': ('full_internal_v2'
                                     if distill_mode == 'full_preactivation'
                                     else None),
            'full_aux_start_dim': (N if distill_mode == 'full_preactivation'
                                   else None),
            'full_aux_num_dims': (args.M - N
                                  if distill_mode == 'full_preactivation'
                                  else None),
            'full_aux_includes_readout': (False
                                          if distill_mode == 'full_preactivation'
                                          else None),
            'teacher_rollout_protocol': (None if distill_mode == 'none'
                                         else TEACHER_CACHE_PROTOCOL),
            'teacher_forcing_used': (distill_mode == 'none'),
            'raw_data_used_as_output_target': (distill_mode == 'none'),
            'teacher_cache': teacher_cache_info,   # None in mode 'none'
            'timing': {
                'cache_generation_time_sec': round(cache_gen_time, 3),
                'training_time_sec': round(training_time, 3),
                'total_run_time_sec': round(time.perf_counter() - _run_t0, 3),
            },
            'init_eval': init_eval,
        }
        meta_path = f"models/{tag}_reduction_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Reduction meta → {meta_path}")

    os.makedirs(res_dir, exist_ok=True)
    np.savez(os.path.join(res_dir, f"{tag}_train.npz"),
             loss_history  = log['loss_history'],
             distill_loss_history  = log['distill_loss_history'],
             original_loss_history = log['original_loss_history'],
             Dstsp_history = log['Dstsp_history'],
             DH_history    = log['DH_history'],
             eval_epochs   = log['eval_epochs'],
             traj_loss_history                      = log['traj_loss_history'],
             target_fp_residual_history             = log['target_fp_residual_history'],
             target_fp_residual_per_fp_history      = log['target_fp_residual_per_fp_history'],
             target_fp_residual_weight_history      = log['target_fp_residual_weight_history'],
             target_fp_gt_x                         = log['target_fp_gt_x'],
             target_fp_hidden_final                 = log['target_fp_hidden_final'])

    # save the target-FP-residual auxiliaries to a separate file (only when enabled)
    if log.get('use_target_fp_residual', False):
        np.savez(os.path.join(res_dir, f"{tag}_target_fp_aux.npz"),
                 target_fp_gt_x         = log['target_fp_gt_x'],
                 target_fp_hidden_final = log['target_fp_hidden_final'])

    plot_curves(log, tag, fig_dir)

    if log['fp_count_history']:
        all_train_dict = {tag: {'fp_count_history': log['fp_count_history'],
                                'fp_snapshots':     log['fp_snapshots']}}
        plot_fp_counts(all_train_dict, tag, fig_dir,
                       num_epochs=args.num_epochs, ssi=args.ssi)
        save_fp_snapshots(log['fp_snapshots'],
                          os.path.join(res_dir, f"{tag}_fp_snapshots.npz"))
        plot_fp_trajectory_overlay(
            log['fp_snapshots'], orig_data,
            os.path.join(fig_dir, f"{tag}_fp_overlay.png"),
            label=tag, ssi=args.ssi)

    if gt_fps_obs is not None and log['matching_history']:
        match_fig = os.path.join(fig_dir, f"{tag}_fp_matching.png")
        plot_fp_matching_history(
            log['matching_history'], log['eval_epochs'].tolist(),
            gt_fp_types, save_path=match_fig, tag=tag)
        save_fp_matching_history(
            log['matching_history'], log['eval_epochs'].tolist(),
            gt_fp_types,
            path=os.path.join(res_dir, f"{tag}_fp_matching.npz"))

    final_loss = float(log['loss_history'][-1])
    best_Dstsp = float(np.nanmin(log['Dstsp_history'])) if len(log['Dstsp_history']) else float('nan')
    best_DH    = float(np.nanmin(log['DH_history']))    if len(log['DH_history'])    else float('nan')
    final_vis  = (log['fp_count_history'][-1].get('vis_total', 0)
                  if log['fp_count_history'] else 0)
    final_traj_loss = (float(log['traj_loss_history'][-1])
                       if len(log['traj_loss_history']) else float('nan'))
    final_tfp_residual = (float(log['target_fp_residual_history'][-1])
                          if len(log['target_fp_residual_history']) else float('nan'))

    print(f"\n  final_loss={final_loss:.4e}  best_Dstsp={best_Dstsp:.4f}"
          f"  best_DH={best_DH:.4f}  final_vis_fps={final_vis}"
          f"  final_traj_loss={final_traj_loss:.4e}"
          f"  final_target_fp_residual={final_tfp_residual:.4e}")

    csv_fields = ['tag','M','P','seed',
                  'r_anchor','r_local','r_bridge','r_long',
                  'n_interleave','num_epochs',
                  'batch_episodes','segment_len','max_bridge_len',
                  'input_sigma','state_sigma',
                  'lr_start','lr_end','lr_fixed',
                  'target_fp_count','fp_count_weight',
                  'lambda_target_fp_residual','target_fp_residual_ramp_epochs',
                  'fp_hidden_lr_mult',
                  'final_loss','best_Dstsp','best_DH','final_vis_fps',
                  'final_traj_loss','final_target_fp_residual','ckpt']
    if retrain_reduced:
        # extra columns only for the retrain CSV (summary_retrain.csv).
        # one row = one source model x one reduction candidate x one retrain seed.
        # 'seed' column = source_seed; 'P' column = P_original (explicit columns too).
        _extra = ['source_seed', 'retrain_seed', 'candidate_id',
                  'P_original', 'P_effective', 'num_deleted',
                  'num_clusters', 'minimal_num_clusters', 'deleted_set_str',
                  'distill_mode', 'distill_lambda', 'symbol_margin',
                  'distill_auto_lambda', 'distill_grad_ratio_alpha',
                  'grad_norm_out_initial', 'grad_norm_aux_initial',
                  'distill_lambda_effective']
        if getattr(args, 'ablation_csv', ''):
            # extra columns only in the ablation-specific CSV (the shared
            # summary_retrain.csv schema is unchanged -> backward compatible)
            _extra = _extra + ['weighting_scheme', 'lambda_formula',
                               'alpha', 'auto_lambda_enabled']
            if getattr(args, 'ablation_sync_cols', False):
                # only the sync-scope-ablation CSV is extended further (the
                # in-flight weighting-ablation CSV keeps its 4-column schema).
                # sync=none condition rows keep the schema via this flag
                _extra = _extra + ['sync_scope', 'sync_interval']
        for _i, _name in enumerate(_extra):
            csv_fields.insert(csv_fields.index('seed') + 1 + _i, _name)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if not write_header:
        with open(csv_path, newline='') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != csv_fields:
            raise RuntimeError(
                f"{csv_path} has a header ({len(existing_header)} cols) that "
                f"doesn't match the current csv_fields ({len(csv_fields)} cols). "
                f"Appending would silently misalign columns. Migrate the file to "
                f"the current schema (see _migrate_summary_csv.py) before rerunning."
            )
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        if write_header: w.writeheader()
        csv_row = dict(
            tag=tag, M=args.M, P=P, seed=args.seed,
            r_anchor=f'{r_a:.2f}', r_local=f'{r_l:.2f}',
            r_bridge=f'{r_b:.2f}', r_long=f'{r_lo:.2f}',
            n_interleave=args.n_interleave, num_epochs=args.num_epochs,
            batch_episodes=args.batch_episodes,
            segment_len=args.segment_len,
            max_bridge_len=args.max_bridge_len,
            input_sigma=args.input_sigma,
            state_sigma=args.state_sigma,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            lr_fixed=args.lr_fixed,
            target_fp_count=args.target_fp_count,
            fp_count_weight=args.fp_count_weight,
            lambda_target_fp_residual=args.lambda_target_fp_residual,
            target_fp_residual_ramp_epochs=args.target_fp_residual_ramp_epochs,
            fp_hidden_lr_mult=args.fp_hidden_lr_mult,
            final_loss=f'{final_loss:.6f}',
            best_Dstsp=f'{best_Dstsp:.4f}', best_DH=f'{best_DH:.4f}',
            final_vis_fps=final_vis,
            final_traj_loss=f'{final_traj_loss:.6f}',
            final_target_fp_residual=f'{final_tfp_residual:.6e}',
            ckpt=ckpt_path)
        if retrain_reduced:
            _deleted = [j for j in range(P) if not relu_mask[j]]
            csv_row['source_seed']          = args.seed
            csv_row['retrain_seed']         = retrain_seed
            csv_row['candidate_id']         = int(candidate_id)
            csv_row['P_original']           = P
            csv_row['P_effective']          = model.P_effective
            csv_row['num_deleted']          = len(_deleted)
            csv_row['num_clusters']         = int(red_cand['num_clusters'])
            csv_row['minimal_num_clusters'] = int(red_spec['minimal_num_clusters'])
            csv_row['deleted_set_str']      = ";".join(map(str, _deleted))
            csv_row['distill_mode']         = distill_mode
            csv_row['distill_lambda']       = distill_lambda
            csv_row['symbol_margin']        = (getattr(args, 'symbol_margin', 1.0)
                                               if distill_mode == 'symbol' else '')
            _cal = log.get('distill_calibration') or {}
            csv_row['distill_auto_lambda']  = getattr(args, 'distill_auto_lambda',
                                                      False)
            csv_row['distill_grad_ratio_alpha'] = (
                getattr(args, 'distill_grad_ratio_alpha', 0.1)
                if distill_mode != 'none' else '')
            csv_row['grad_norm_out_initial'] = _cal.get('grad_norm_out_initial', '')
            csv_row['grad_norm_aux_initial'] = _cal.get('grad_norm_aux_initial', '')
            csv_row['distill_lambda_effective'] = log.get('distill_lambda_effective',
                                                          distill_lambda)
            if getattr(args, 'ablation_csv', ''):
                csv_row['weighting_scheme'] = distill_weighting
                csv_row['lambda_formula'] = lambda_formula or ''
                csv_row['alpha'] = getattr(args, 'distill_grad_ratio_alpha',
                                           0.1)
                csv_row['auto_lambda_enabled'] = bool(
                    getattr(args, 'distill_auto_lambda', False))
                if getattr(args, 'ablation_sync_cols', False):
                    csv_row['sync_scope'] = _sync_scope
                    csv_row['sync_interval'] = getattr(
                        args, 'distill_sync_interval', 128)
        w.writerow(csv_row)
    print(f"  CSV → {csv_path}")

    return dict(tag=tag, P=P, final_loss=final_loss,
                best_Dstsp=best_Dstsp, best_DH=best_DH,
                final_vis_fps=final_vis, ckpt=ckpt_path)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_path, meta_path, long_data_path):
    print(f"Loading: {data_path}")
    d = np.load(data_path, allow_pickle=True)
    orig_data      = d['original_data'].astype(np.float32)
    anchor_trajs   = d['anchor_trajs'].astype(np.float32)
    local_trajs    = d['local_trajs_flat'].astype(np.float32)
    bridge_trajs   = d['bridge_trajs'].astype(np.float32)
    bridge_lengths = d['bridge_lengths'].astype(np.int32)
    gt_fps_full    = d['fixed_points'].astype(np.float32)
    gt_fp_types_a  = d['fixed_point_types'].astype(str).tolist()
    N = orig_data.shape[-1]
    gt_fps_obs = gt_fps_full[:, :N]

    with open(meta_path) as f:
        meta = json.load(f)
    anchor_meta = meta.get('anchor', [])
    local_meta  = meta.get('local',  [])
    bridge_meta = meta.get('bridge', [])
    if not anchor_meta:
        K_a = anchor_trajs.shape[0]
        anchor_meta = [{'fp_index': k} for k in range(K_a)]
    K = gt_fps_obs.shape[0]

    # long trajectories
    long_trajs = None
    if os.path.exists(long_data_path):
        dl = np.load(long_data_path)
        long_trajs = dl['long_trajs'].astype(np.float32)  # (K, T_long+1, M)
        print(f"  long_trajs shape: {long_trajs.shape}")
    else:
        print(f"  WARNING: {long_data_path} not found")

    print(f"  K={K}  anchor={len(anchor_meta)}  "
          f"local={len(local_meta)}  bridge={len(bridge_meta)}")
    print(f"  GT FPs: {gt_fps_obs.shape}  types={gt_fp_types_a}")

    return (orig_data,
            anchor_trajs, anchor_meta,
            local_trajs,  local_meta,
            bridge_trajs, bridge_lengths, bridge_meta,
            long_trajs, K, gt_fps_obs, gt_fp_types_a)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aug-only training: anchor/local/bridge/long, no original data')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_mode', default='augmented',
        choices=['augmented', 'original'],
        help='"augmented": train on anchor/local/bridge/long episodes (default). '
             '"original": train on the raw Chua time series (--raw_data_path required).')
    parser.add_argument('--raw_data_path', default='data/chua_3-scroll_train.npy',
        help='raw-data .npy file read when data_mode=original.')
    parser.add_argument('--tag_prefix', default='chua',
        help='model tag prefix (system name). default "chua" is byte-identical '
             'to the existing tags; for lorenz63/rossler pass the '
             'tag_prefix from system_config.')
    parser.add_argument('--fp_delta_t', type=float, default=0.01,
        help='dt for the continuous-time fixed-point analysis (= data sample_dt).')
    parser.add_argument('--data_path',      default=DATA_PATH)
    parser.add_argument('--meta_path',      default=META_PATH)
    parser.add_argument('--long_data_path', default=LONG_DATA_PATH)

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--M',      type=int,          default=20)
    parser.add_argument('--P',      type=int,          default=3)
    parser.add_argument('--P_list', type=int, nargs='+', default=None,
        help='multiple P values; overrides --P.')

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--num_epochs',      type=int,   default=2000)
    parser.add_argument('--steps_per_epoch', type=int,   default=50)
    parser.add_argument('--batch_episodes',  type=int,   default=16,
        help='episodes per gradient step.')
    parser.add_argument('--n_interleave',    type=int,   default=16,
        help='teacher-forcing interval; >= segment_len means effectively free-run.')
    parser.add_argument('--alpha',           type=float, default=1.0,
        help='teacher-forcing mixing rate; 1.0 = full overwrite.')

    # ── episode ratios ────────────────────────────────────────────────────────
    parser.add_argument('--r_anchor',       type=float, default=0.10)
    parser.add_argument('--r_local',        type=float, default=0.40)
    parser.add_argument('--r_bridge',       type=float, default=0.20)
    parser.add_argument('--r_long',         type=float, default=0.30)
    parser.add_argument('--segment_len',    type=int,   default=200,
        help='segment length of long episodes.')
    parser.add_argument('--max_bridge_len', type=int,   default=200,
        help='max length of bridge trajectories used.')

    # ── learning rate ─────────────────────────────────────────────────────────
    parser.add_argument('--lr_start', type=float, default=1e-3)
    parser.add_argument('--lr_end',   type=float, default=1e-5,
        help='final LR of ExponentialLR; ignored with --lr_fixed.')
    parser.add_argument('--lr_fixed', action='store_true',
        help='fix the LR at lr_start.')

    # ── noise ─────────────────────────────────────────────────────────────────
    parser.add_argument('--input_sigma', type=float, default=0.1,
        help='input-noise factor for bridge/long episodes; anchor/local always 0.')
    parser.add_argument('--state_sigma', type=float, default=0.0,
        help='state-noise std per step.')

    # ── best model selection ──────────────────────────────────────────────────
    parser.add_argument('--target_fp_count', type=int,   default=-1,
        help='target number of visited FPs; -1/unset = use the data GT FP count K.')
    parser.add_argument('--fp_count_weight', type=float, default=5.0)

    # ── target FP residual loss ───────────────────────────────────────────────
    parser.add_argument('--lambda_target_fp_residual', type=float, default=0.0,
        help='coefficient of the target FP residual loss; 0 disables it.')
    parser.add_argument('--target_fp_residual_ramp_epochs', type=int, default=200,
        help='ramp-up epochs; <= 0 means no ramp (full coefficient from the start).')
    parser.add_argument('--fp_hidden_lr_mult', type=float, default=1.0,
        help='LR multiplier of fp_hidden; actual LR = lr_start * fp_hidden_lr_mult.')

    # ── evaluation & logging ──────────────────────────────────────────────────
    parser.add_argument('--ssi',             type=int,  default=SSI,
        help='evaluation interval (epochs)。')
    parser.add_argument('--no_periodic_eval', action='store_true',
        help='fast mode: compute only Dstsp + vis_fps, skip DH.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_model_paths', nargs='*', default=[],
        help='model paths in P_list order; used as initial weights when given.')
    parser.add_argument('--init_tag_suffix', type=str, default='_fromTeacher',
        help='tag suffix appended when init_model_paths is given.')

    # ── reduced-model retraining (graph-reduction baseline) ───────────────
    parser.add_argument('--retrain_reduced', action='store_true',
        help='follow the best reduction spec from graph reduction: fix the '
             'deleted ReLUs to identity and retrain from the source checkpoint. '
             '--seed then denotes the source model seed (tag base); the '
             'training RNG comes from --retrain_seed.')
    parser.add_argument('--retrain_seed', type=int, default=None,
        help='training seed for retrain_reduced; defaults to --seed '
             'when unset.')
    parser.add_argument('--source_checkpoints', nargs='*', default=[],
        help='source (over-parameterized) checkpoint paths in P_list order; '
             'required for retrain_reduced.')
    parser.add_argument('--reduction_specs', nargs='*', default=[],
        help='minimal symbol reduction spec JSON paths in P_list order; '
             'required for retrain_reduced.')
    parser.add_argument('--reduction_candidate_ids', nargs='*', default=[],
        help='comma-separated candidate_id groups in P_list order '
             '(e.g. "0,2,3"); unset runs every candidate of each spec.')

    # ── teacher distillation (only with retrain_reduced) ────────────────────
    parser.add_argument('--distill_mode', default='none',
        choices=['none', 'full_preactivation', 'retained_preactivation', 'symbol'],
        help='auxiliary loss with the frozen source over-param model as teacher. '
             'full_preactivation: MSE on all M preactivations / '
             'retained_preactivation: MSE on the retained ReLU slots only / '
             'symbol: match the retained ReLU sign pattern via margin-softplus. '
             'default=none (fully compatible with the existing baseline).')
    parser.add_argument('--distill_lambda_full',     type=float, default=1.0)
    parser.add_argument('--distill_lambda_retained', type=float, default=1.0)
    parser.add_argument('--distill_lambda_symbol',   type=float, default=1.0)
    parser.add_argument('--symbol_margin',           type=float, default=1.0,
        help='margin m of symbol mode: softplus(m - y * z_pre).')
    parser.add_argument('--teacher_cache_dir', default='results/teacher_cache',
        help='where to store teacher trajectory caches; shared per source checkpoint.')
    parser.add_argument('--distill_auto_lambda', action='store_true',
        help='auto-calibrate lambda from the initial gradient-norm ratio: '
             'lambda = alpha*g_out/(g_aux+eps); fixed for the run (not adaptive).')
    parser.add_argument('--distill_grad_ratio_alpha', type=float, default=0.1,
        help='target gradient ratio alpha of the auto lambda (||lambda grad L_aux|| ~= alpha ||grad L_out||).')
    parser.add_argument('--distill_calibration_batches', type=int, default=5,
        help='number of calibration batches (median is the representative).')
    parser.add_argument('--distill_grad_eps', type=float, default=1e-12)
    # ── weighting-ablation (2026-09-14) ────────────────────────────
    # ablation that changes ONLY the guidance-loss weighting rule. \'default\'
    # is exactly the legacy behavior. Other values decide lambda as:
    #   coord_equal_fixed : auto OFF, λ = P_eff/N (retained) / (M-N)/N (full)
    #   fixed_lambda1     : auto OFF, λ = 1.0
    #   grad_auto_alpha1  : auto ON,  alpha = 1.0 (uses the existing calibration)
    parser.add_argument('--distill_weighting', default='default',
        choices=['default', 'coord_equal_fixed', 'grad_auto_alpha1',
                 'fixed_lambda1'],
        help='weighting-ablation rule (default = legacy behavior)')
    parser.add_argument('--distill_name_tag', default='',
        help='weighting-ablation experiment tag; when set, replaces the '
             'distill/ga tag parts of checkpoint/log names (collision guard)')
    parser.add_argument('--ablation_csv', default='',
        help='when set, write retrain CSV rows to results/aug_only/<name> '
             'instead of summary_retrain.csv (keeps the shared CSV clean)')
    # ── parent-state synchronization scope ablation (2026-09-14) ────
    # sparse hard sync to the frozen parent's autonomous cache during
    # distillation. 'none' (default) is exactly the legacy behavior. The sync
    # is independent of raw-data TF and orthogonal to the weighting axis (--distill_weighting / alpha).
    parser.add_argument('--distill_sync_scope', default='none',
        choices=['none', 'readout', 'readout_retained', 'full'],
        help='coordinates targeted by the parent-state sync (none = no sync, legacy)')
    parser.add_argument('--distill_sync_interval', type=int, default=128,
        help='parent-state sync period (128, same as standard sparse TF)')
    parser.add_argument('--ablation_sync_cols', action='store_true',
        help='include sync_scope/sync_interval columns in the ablation CSV '
             '(unifies the schema across all sync-ablation CSV rows; '
             'the weighting-ablation CSV keeps its 4 extra columns)')
    parser.add_argument('--force_rebuild_teacher_cache', action='store_true',
        help='(debug) ignore the existing teacher cache and rebuild it.')

    args = parser.parse_args()

    # propagate --fp_delta_t to the module global (all FP analysis reads DELTA_T)
    globals()['DELTA_T'] = args.fp_delta_t
    if abs(args.fp_delta_t - 0.01) > 1e-12:
        print(f"[system] fp_delta_t = {args.fp_delta_t}")

    if args.retrain_reduced and args.init_model_paths:
        raise ValueError("--retrain_reduced and --init_model_paths are mutually exclusive")

    if args.data_mode == 'original':
        print(f"data_mode=original  raw_data_path={args.raw_data_path}")
        raw_data = np.load(args.raw_data_path).astype(np.float32)
        print(f"  raw_data shape: {raw_data.shape}")
        # the eval orig_data uses the same file
        orig_data = raw_data
        # augmented-mode variables are dummies (unused inside run_one)
        anchor_trajs = anchor_meta = None
        local_trajs  = local_meta  = None
        bridge_trajs = bridge_lengths = bridge_meta = None
        long_trajs   = None
        K            = 0
        gt_fps_obs   = gt_fp_types = None
        if args.target_fp_count is None or args.target_fp_count < 0:
            args.target_fp_count = 5
            print(f"  target_fp_count set to 5 (default for original mode)")
    else:
        (orig_data,
         anchor_trajs, anchor_meta,
         local_trajs,  local_meta,
         bridge_trajs, bridge_lengths, bridge_meta,
         long_trajs, K, gt_fps_obs, gt_fp_types) = load_data(
            args.data_path, args.meta_path, args.long_data_path)
        raw_data = None

        if args.target_fp_count is None or args.target_fp_count < 0:
            args.target_fp_count = K
            print(f"  target_fp_count set to K={K} (auto)")

        if long_trajs is None and args.r_long > 0:
            raise ValueError(
                f"--r_long={args.r_long} > 0 but long_data_path was not found: "
                f"{args.long_data_path}"
            )

    fig_dir  = os.path.join("figures", "aug_only")
    res_dir  = os.path.join("results", "aug_only")
    # retraining results are stored separately from the normal summary.csv
    csv_name = "summary_retrain.csv" if args.retrain_reduced else "summary.csv"
    if getattr(args, 'ablation_csv', ''):
        csv_name = args.ablation_csv    # weighting ablation: separate CSV
    csv_path = os.path.join(res_dir, csv_name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    P_values = args.P_list if args.P_list is not None else [args.P]
    init_paths = getattr(args, 'init_model_paths', []) or []

    cand_groups = None
    if args.retrain_reduced:
        if (len(args.source_checkpoints) != len(P_values)
                or len(args.reduction_specs) != len(P_values)):
            raise ValueError(
                f"--retrain_reduced requires --source_checkpoints / --reduction_specs "
                f"matching P_list ({len(P_values)} entries) "
                f"(got {len(args.source_checkpoints)} / {len(args.reduction_specs)})")
        if args.reduction_candidate_ids:
            if len(args.reduction_candidate_ids) != len(P_values):
                raise ValueError(
                    f"--reduction_candidate_ids must have as many comma-separated "
                    f"groups as P_list ({len(P_values)} entries) "
                    f"(got {len(args.reduction_candidate_ids)})")
            cand_groups = [[int(x) for x in grp.split(',')]
                           for grp in args.reduction_candidate_ids]

    all_results = []
    for i, P_val in enumerate(P_values):
        args.P = P_val
        init_path = init_paths[i] if i < len(init_paths) else ''
        if args.retrain_reduced:
            src_ckpt      = args.source_checkpoints[i]
            red_spec_path = args.reduction_specs[i]
            if cand_groups is not None:
                cand_ids = cand_groups[i]
            else:
                # no candidate ids given -> run every candidate of the spec
                with open(red_spec_path) as f:
                    _sp = json.load(f)
                if 'candidates' not in _sp:
                    raise ValueError(
                        f"{red_spec_path} is the old single-candidate format. "
                        f"Re-run graph reduction.")
                cand_ids = [int(c['candidate_id']) for c in _sp['candidates']]
            print(f"\n[retrain_reduced] P={P_val}: {len(cand_ids)} candidate(s) "
                  f"to run: {cand_ids}")
            for cid in cand_ids:
                r = run_one(args, P_val, orig_data,
                            anchor_trajs, anchor_meta,
                            local_trajs,  local_meta,
                            bridge_trajs, bridge_lengths, bridge_meta,
                            long_trajs, K, gt_fps_obs, gt_fp_types,
                            fig_dir, res_dir, csv_path,
                            raw_data=raw_data,
                            init_model_path=init_path,
                            source_checkpoint=src_ckpt,
                            reduction_spec=red_spec_path,
                            candidate_id=cid)
                all_results.append(r)
        else:
            r = run_one(args, P_val, orig_data,
                        anchor_trajs, anchor_meta,
                        local_trajs,  local_meta,
                        bridge_trajs, bridge_lengths, bridge_meta,
                        long_trajs, K, gt_fps_obs, gt_fp_types,
                        fig_dir, res_dir, csv_path,
                        raw_data=raw_data,
                        init_model_path=init_path)
            all_results.append(r)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    hdr = f"{'tag':<60} {'P':>3} {'loss':>8} {'Dstsp':>7} {'DH':>7} {'vis_fps':>7}"
    print(hdr); print('-'*len(hdr))
    for r in all_results:
        def _f(v):
            try: return f'{v:7.3f}'
            except: return str(v)
        print(f"{r['tag']:<60} {r['P']:>3} "
              f"{_f(r['final_loss'])} {_f(r['best_Dstsp'])} "
              f"{_f(r['best_DH'])} {r['final_vis_fps']:>7}")

    print(f"\nFigures → {fig_dir}/")
    print(f"Results → {res_dir}/")
    print(f"CSV     → {csv_path}")


if __name__ == '__main__':
    main()
