"""
fp_tracker.py
Per-checkpoint fixed-point snapshot collection, saving and plotting.

Analyzes the real fixed points of all 2^P regions and marks visited vs unvisited via the is_visited flag.
"""

from collections import Counter
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import linear_region_functions as lrf

# ── stability classification ───────────────────────────────────────────────

_STABLE   = frozenset(('stable node',   'stable spiral'))
_UNSTABLE = frozenset(('unstable node', 'unstable spiral'))

def _classify(stability):
    if stability in _STABLE:   return 'stable'
    if stability in _UNSTABLE: return 'unstable'
    if stability == 'saddle':  return 'saddle'
    return 'other'

_TYPE_MARKER = {'stable': 'o', 'unstable': 'X', 'saddle': 'D', 'other': 's'}
_TYPE_COLOR  = {'stable': 'royalblue', 'unstable': 'tomato',
                'saddle': 'limegreen',  'other': 'gray'}
_TYPES_MAIN  = ('stable', 'unstable', 'saddle')


# ── snapshot collection ────────────────────────────────────────────────────

def collect_fp_snapshot(model, orbit_np, epoch, analyze_fixed_points_fn, delta_t=0.01):
    """
    Analyze the real fixed points of all 2^P regions and return a snapshot.

    Parameters
    ----------
    model                  : AL_RNN (attributes: N, P, M)
    orbit_np               : (T, M) ndarray — free-run orbit
    epoch                  : int
    analyze_fixed_points_fn: callable compatible with analyze_fixed_points_continuous
    delta_t                : float

    Returns
    -------
    dict with keys:
      epoch   : int
      fps     : list[dict]   — full field set of each real fixed point
      counts  : dict         — all_* / visited_* counts
    """
    N, P = model.N, model.P

    bits = lrf.convert_to_bits(orbit_np[:, -P:])

    # occupancy per region (0 for unvisited regions)
    bit_strs    = [''.join(map(str, map(int, b))) for b in bits]
    occupancy   = Counter(bit_strs)
    visited_set = set(occupancy.keys())

    # enumerate all 2^P regions and analyze their fixed points
    all_regions = [np.array([int(c) for c in format(i, f'0{P}b')], dtype=int)
                   for i in range(2 ** P)]
    fp_data = analyze_fixed_points_fn(model, all_regions, delta_t)

    orbit_obs = orbit_np[:, :N]  # (T, N)

    fps    = []
    counts = {
        'all_total': 0,    'all_stable': 0,    'all_unstable': 0,    'all_saddle': 0,
        'vis_total': 0,    'vis_stable': 0,     'vis_unstable': 0,    'vis_saddle': 0,
    }

    for region_str, v in fp_data.items():
        if v['type'] != 'real':
            continue

        z_star    = v['location']               # (M,)
        y_star    = z_star[:N]                  # (N,)
        eigvals   = v['eigenvalues_continuous'] # (M,) complex
        max_re    = float(np.max(eigvals.real))
        fp_type   = _classify(v['stability'])
        is_vis    = region_str in visited_set

        diffs  = orbit_obs - y_star[np.newaxis, :]
        dists  = np.linalg.norm(diffs, axis=1)
        t_near = int(np.argmin(dists))

        fps.append({
            'fp_position_z':       z_star.copy(),
            'fp_position_y':       y_star.copy(),
            'fp_region_bits':      region_str,
            'fp_type':             fp_type,
            'is_visited':          is_vis,
            'fp_eigvals':          eigvals.copy(),
            'fp_max_real_eig':     max_re,
            'fp_dist_to_orbit':    float(dists[t_near]),
            'fp_nearest_time':     t_near,
            'fp_region_occupancy': occupancy.get(region_str, 0),
        })

        counts['all_total'] += 1
        if fp_type in _TYPES_MAIN:
            counts[f'all_{fp_type}'] += 1
        if is_vis:
            counts['vis_total'] += 1
            if fp_type in _TYPES_MAIN:
                counts[f'vis_{fp_type}'] += 1

    return {'epoch': epoch, 'fps': fps, 'counts': counts}


# ── FP matching ────────────────────────────────────────────────────────────

def compute_fp_matching(snap, gt_fps_obs):
    """
    For each ground-truth FP, find the nearest student FP in observation space.

    Parameters
    ----------
    snap        : return value of collect_fp_snapshot
    gt_fps_obs  : (K, N) ndarray — observed coordinates of the ground-truth FPs

    Returns
    -------
    dict with keys:
      match_dists      : (K,) float  — distance to the nearest student FP (inf if none)
      match_stabilities: (K,) str    — stability of the nearest student FP ('none' if absent)
      match_fp_ids     : (K,) int    — index of the nearest student FP (-1 if absent)
      match_is_visited : (K,) bool   — whether the nearest student FP lies in a visited region
    """
    K = len(gt_fps_obs)
    student_fps = snap['fps']   # list of dicts

    match_dists       = np.full(K, np.inf)
    match_stabilities = ['none'] * K
    match_fp_ids      = np.full(K, -1, dtype=int)
    match_is_visited  = np.zeros(K, dtype=bool)

    if not student_fps:
        return dict(match_dists=match_dists,
                    match_stabilities=match_stabilities,
                    match_fp_ids=match_fp_ids,
                    match_is_visited=match_is_visited)

    # student FP coordinate matrix (S, N)
    stu_pos = np.array([fp['fp_position_y'] for fp in student_fps])  # (S, N)

    for k, gt_pos in enumerate(gt_fps_obs):
        diffs = stu_pos - gt_pos[np.newaxis, :]          # (S, N)
        dists = np.linalg.norm(diffs, axis=1)            # (S,)
        best  = int(np.argmin(dists))
        match_dists[k]       = float(dists[best])
        match_stabilities[k] = student_fps[best]['fp_type']
        match_fp_ids[k]      = best
        match_is_visited[k]  = bool(student_fps[best]['is_visited'])

    return dict(match_dists=match_dists,
                match_stabilities=match_stabilities,
                match_fp_ids=match_fp_ids,
                match_is_visited=match_is_visited)


def plot_fp_matching_history(matching_history, eval_epochs, gt_fp_types,
                             save_path, tag='', vline_epochs=None):
    """
    Plot and save the training history of each ground-truth FP's matching distance.
    One subplot per FP (1 x K).

    Parameters
    ----------
    matching_history : list[dict]  — list of compute_fp_matching return values
    eval_epochs      : (E,) int    — epoch of each snapshot
    gt_fp_types      : (K,) str    — stability of each GT FP
    save_path        : str
    tag              : str
    vline_epochs     : list[int] or None  — stage boundaries
    """
    if not matching_history:
        return

    K          = len(gt_fp_types)
    eval_epochs = list(eval_epochs)

    dist_mat = np.array([m['match_dists']     for m in matching_history])   # (E, K)
    vis_mat  = np.array([m['match_is_visited'] for m in matching_history])  # (E, K) bool
    stab_mat = np.array([[m['match_stabilities'][k] for k in range(K)]
                          for m in matching_history])                        # (E, K) str

    palette    = plt.cm.tab10(np.linspace(0, 0.9, K))
    marker_for = {'saddle': 'D', 'stable': 'o', 'unstable': 'X', 'other': 's',
                  'none': 'x'}
    vline_colors = ['orange', 'red', 'purple']
    vline_labels = ['A→B', 'B→C', 'C→D']

    ncols = K
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=False)
    if K == 1:
        axes = [axes]

    fig.suptitle(f'GT FP matching distance  [{tag}]\n'
                 f'filled=visited region  hollow=unvisited  '
                 f'marker shape=matched student FP type',
                 fontsize=10)

    for k, ax in enumerate(axes):
        color = palette[k]
        dists = dist_mat[:, k]   # (E,)
        vis   = vis_mat[:, k]    # (E,) bool
        stabs = stab_mat[:, k]   # (E,) str

        # solid line
        ax.plot(eval_epochs, dists, '-', color=color, lw=1.8, alpha=0.85)

        # markers: shape = stability of the matched student FP, fill = visited flag
        for e_idx, (ep, d, v, st) in enumerate(zip(eval_epochs, dists, vis, stabs)):
            mk = marker_for.get(st, 's')
            fc = color if v else 'none'
            ax.scatter(ep, d, marker=mk, s=50, color=color,
                       facecolors=fc, edgecolors=color, linewidths=1.2, zorder=4)

        # stage boundaries
        if vline_epochs:
            for i, ve in enumerate(vline_epochs):
                ax.axvline(ve, color=vline_colors[i % len(vline_colors)],
                           ls='--', lw=1.2, alpha=0.7,
                           label=vline_labels[i] if e_idx == 0 else None)
            ax.legend(fontsize=6, loc='upper right')

        ax.set_title(f'GT-FP{k}\n({gt_fp_types[k]})', fontsize=9)
        ax.set_xlabel('epoch', fontsize=8)
        if k == 0:
            ax.set_ylabel('distance (obs space)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # legend (marker shapes) only on the first panel
        if k == 0:
            handles = [
                mlines.Line2D([], [], color='gray', marker=mk, ls='None',
                              markersize=7, markerfacecolor='gray', label=f'{tp} (vis)')
                for tp, mk in marker_for.items() if tp != 'none'
            ] + [
                mlines.Line2D([], [], color='gray', marker='o', ls='None',
                              markersize=7, markerfacecolor='none', label='unvisited')
            ]
            ax.legend(handles=handles, fontsize=6, loc='upper right',
                      title='matched FP type', title_fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_fp_matching_history(matching_history, eval_epochs, gt_fp_types, path):
    """Save matching_history to an npz file."""
    if not matching_history:
        return
    K = len(gt_fp_types)
    dist_mat = np.array([m['match_dists']     for m in matching_history])   # (E, K)
    vis_mat  = np.array([m['match_is_visited'] for m in matching_history])  # (E, K)
    stab_mat = np.array([[m['match_stabilities'][k] for k in range(K)]
                         for m in matching_history])                         # (E, K) str
    np.savez(path,
             eval_epochs  = np.array(eval_epochs),
             match_dists  = dist_mat,
             match_visited= vis_mat,
             match_stabs  = stab_mat,
             gt_fp_types  = np.array(gt_fp_types))


# ── save ──────────────────────────────────────────────────────────────────

def save_fp_snapshots(snapshots, path):
    """
    Save the snapshot list to an npz file. Key format: ep{epoch}_{field}

      fp_position_z       : (n_fps, M)
      fp_position_y       : (n_fps, N)
      fp_eigvals_real/imag: (n_fps, M)
      fp_max_real_eig     : (n_fps,)
      fp_dist_to_orbit    : (n_fps,)
      fp_nearest_time     : (n_fps,)
      fp_region_occupancy : (n_fps,)
      fp_type             : (n_fps,)  str array
      fp_region_bits      : (n_fps,)  str array
      is_visited          : (n_fps,)  bool array
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    payload = {}
    for snap in snapshots:
        ep, fps = snap['epoch'], snap['fps']
        if not fps:
            continue
        pre = f'ep{ep}'
        payload[f'{pre}_fp_position_z']       = np.array([fp['fp_position_z']       for fp in fps])
        payload[f'{pre}_fp_position_y']       = np.array([fp['fp_position_y']       for fp in fps])
        payload[f'{pre}_fp_eigvals_real']     = np.array([fp['fp_eigvals'].real      for fp in fps])
        payload[f'{pre}_fp_eigvals_imag']     = np.array([fp['fp_eigvals'].imag      for fp in fps])
        payload[f'{pre}_fp_max_real_eig']     = np.array([fp['fp_max_real_eig']      for fp in fps])
        payload[f'{pre}_fp_dist_to_orbit']    = np.array([fp['fp_dist_to_orbit']     for fp in fps])
        payload[f'{pre}_fp_nearest_time']     = np.array([fp['fp_nearest_time']      for fp in fps])
        payload[f'{pre}_fp_region_occupancy'] = np.array([fp['fp_region_occupancy']  for fp in fps])
        payload[f'{pre}_fp_type']             = np.array([fp['fp_type']              for fp in fps])
        payload[f'{pre}_fp_region_bits']      = np.array([fp['fp_region_bits']       for fp in fps])
        payload[f'{pre}_is_visited']          = np.array([fp['is_visited']           for fp in fps])
    np.savez(path, **payload)


# ── fp counts plot ─────────────────────────────────────────────────────────

def plot_fp_counts(all_train, tag, save_dir, num_epochs, ssi, vline_epoch=None):
    """
    Plot the training history of the number of real fixed points.

    5 stacked panels (sharex):
      Row 0: total       (all regions)
      Row 1: stable      (all regions)
      Row 2: unstable    (all regions)
      Row 3: saddle      (all regions)
      Row 4: visited FP ratio (vis_total / all_total)

    Multiple labels per panel share one color palette; solid lines.
    """
    os.makedirs(save_dir, exist_ok=True)

    # key: (source, field_or_None), title, ylabel
    ROWS = [
        ('all',   'total',    'Total real FPs',   'count'),
        ('vis',   None,       'Visited FPs',       'count'),
        ('all',   'stable',   'Stable FPs',        'count'),
        ('all',   'unstable', 'Unstable FPs',      'count'),
        ('all',   'saddle',   'Saddle FPs',        'count'),
        ('ratio', None,       'Visited FP ratio (vis / all)', 'vis / all'),
    ]
    N_ROWS = len(ROWS)

    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(all_train), 1)))
    cmap   = {lb: colors[i] for i, lb in enumerate(all_train.keys())}

    fig, axes = plt.subplots(N_ROWS, 1, figsize=(11, 3 * N_ROWS),
                             sharex=True)
    fig.suptitle(f'Real FP counts  {tag}', fontsize=11, y=1.01)

    for row_idx, (src, field, row_title, ylabel) in enumerate(ROWS):
        ax = axes[row_idx]
        for lb, res in all_train.items():
            hist = res['fp_count_history']
            if not hist:
                continue
            x_ep = np.arange(len(hist)) * ssi
            if src == 'all':
                y = [c[f'all_{field}'] for c in hist]
                ax.plot(x_ep, y, '-', lw=1.8, alpha=0.9, color=cmap[lb], label=lb)
            elif src == 'vis':
                y = [c['vis_total'] for c in hist]
                ax.plot(x_ep, y, '-', lw=1.8, alpha=0.9, color=cmap[lb], label=lb)
            else:  # ratio
                y = [c['vis_total'] / max(c['all_total'], 1) for c in hist]
                ax.plot(x_ep, y, '-o', ms=2, lw=1.5, alpha=0.85,
                        color=cmap[lb], label=lb)
        ax.set_ylabel(ylabel)
        ax.set_title(row_title, fontsize=9)
        ax.grid(True, alpha=0.3)
        if src == 'ratio':
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('epoch')
        if vline_epoch is not None:
            ax.axvline(vline_epoch, color='gray', ls='--', lw=1, alpha=0.5)
        if len(all_train) > 1:
            ax.legend(fontsize=7, ncol=2)
        elif src == 'ratio':
            ax.legend(fontsize=7)

    plt.tight_layout()
    fname = f'fp_counts_{tag}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ── trajectory overlay ─────────────────────────────────────────────────────

def plot_fp_trajectory_overlay(snapshots, X_true, save_path, label='', ssi=1):
    """
    Plot fixed-point locations in observation space over training, overlaid on the ground-truth orbit.

    Left panel (3D/2D):
      - ground-truth orbit (gray)
      - FPs in visited regions: filled markers, color = epoch
      - FPs in unvisited regions: open markers, color = epoch, smaller / low alpha

    Right panel: distance-to-orbit history by type
      - solid = all-region FPs (mean), dashed = visited FPs (mean)
    """
    N      = X_true.shape[1]
    use_3d = (N >= 3)

    active_snaps = [s for s in snapshots if s['fps']]
    if not active_snaps:
        return

    epochs  = [s['epoch'] for s in active_snaps]
    ep_min, ep_max = min(epochs), max(epochs)
    cmap_ep = plt.cm.plasma
    norm_ep = plt.Normalize(vmin=ep_min, vmax=ep_max)

    fig = plt.figure(figsize=(15, 6))

    # ── left: orbit + FP locations ──────────────────────────────
    ax = fig.add_subplot(1, 2, 1, projection='3d') if use_3d \
         else fig.add_subplot(1, 2, 1)

    stride = max(1, len(X_true) // 8000)
    xt     = X_true[::stride]
    if use_3d:
        ax.scatter(xt[:, 0], xt[:, 1], xt[:, 2],
                   c='silver', s=1, alpha=0.12, depthshade=False)
    else:
        ax.scatter(xt[:, 0], xt[:, 1], c='silver', s=1, alpha=0.12)

    for snap in active_snaps:
        ep      = snap['epoch']
        col     = cmap_ep(norm_ep(ep))
        is_last = (ep == ep_max)
        al_base = 0.3 + 0.6 * norm_ep(ep)

        for fp in snap['fps']:
            y       = fp['fp_position_y']
            mk      = _TYPE_MARKER.get(fp['fp_type'], 's')
            visited = fp['is_visited']

            if visited:
                sz = 80 if is_last else 28
                al = al_base
                fc = col            # filled
                ec = 'k'
                lw = 0.8 if is_last else 0.3
            else:
                sz = 40 if is_last else 14
                al = al_base * 0.45
                fc = 'none'         # open
                ec = col
                lw = 0.8 if is_last else 0.4

            if use_3d:
                ax.scatter(y[0], y[1], y[2],
                           c=[fc] if fc != 'none' else [[0,0,0,0]],
                           marker=mk, s=sz, alpha=al,
                           edgecolors=ec, linewidths=lw, depthshade=False)
            else:
                ax.scatter(y[0], y[1],
                           c=[fc] if fc != 'none' else [[0,0,0,0]],
                           marker=mk, s=sz, alpha=al,
                           edgecolors=ec, linewidths=lw)

    if use_3d:
        ax.set_xlabel('y₀'); ax.set_ylabel('y₁'); ax.set_zlabel('y₂')
        ax.tick_params(labelsize=7)
    else:
        ax.set_xlabel('y₀'); ax.set_ylabel('y₁')

    ax.set_title('FP positions in obs space\n'
                 'filled=visited region  hollow=unvisited\n'
                 'color=epoch  o=stable  X=unstable  D=saddle',
                 fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap_ep, norm=norm_ep)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='epoch', shrink=0.6, pad=0.08)

    type_handles = [mlines.Line2D([], [], color='dimgray', marker=mk,
                                  ls='None', markersize=7, label=t)
                    for t, mk in _TYPE_MARKER.items() if t != 'other']
    vis_handles  = [
        mlines.Line2D([], [], color='dimgray', marker='o', ls='None',
                      markersize=7, markerfacecolor='dimgray', label='visited'),
        mlines.Line2D([], [], color='dimgray', marker='o', ls='None',
                      markersize=7, markerfacecolor='none',    label='unvisited'),
    ]
    ax.legend(handles=type_handles + vis_handles, fontsize=7,
              loc='upper left', framealpha=0.6, ncol=2, title='stability / visit')

    # ── right: dist_to_orbit history (all vs visited) ───────────
    ax2 = fig.add_subplot(1, 2, 2)
    for fp_type, col in _TYPE_COLOR.items():
        if fp_type == 'other':
            continue
        for vis_only, ls, sfx in [(False, '-', 'all'), (True, '--', 'vis')]:
            ep_list, dmean = [], []
            for snap in active_snaps:
                dlist = [fp['fp_dist_to_orbit']
                         for fp in snap['fps']
                         if fp['fp_type'] == fp_type
                         and (fp['is_visited'] if vis_only else True)]
                if dlist:
                    ep_list.append(snap['epoch'])
                    dmean.append(float(np.mean(dlist)))
            if ep_list:
                ax2.plot(ep_list, dmean, ls=ls, marker='o', ms=2, lw=1.4,
                         color=col, alpha=0.85 if not vis_only else 0.6,
                         label=f'{fp_type} ({sfx})')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('mean dist to free-run orbit (obs space)')
    ax2.set_title('FP proximity to orbit\n(solid=all regions, dashed=visited only)')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(label, fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
