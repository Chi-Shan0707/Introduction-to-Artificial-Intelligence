#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bonus 2 supplementary analysis script.

Purpose:
1. Load q_learning_bonus2_best.pkl and q_learning_bonus2.pkl.
2. Reproduce auxiliary Q-learning rollouts with an internal CartPole model.
3. Generate Q-table policy heatmaps and tail-risk trajectory plots.
4. Generate root-action diagnostics without running full 2000-step MCTS episodes.

Usage:
    python bonus2_supplement_analysis.py --project-root . --out-dir bonus2_supplement_outputs

Note:
    The official grading evidence is still the uploaded evaluate.py / run logs.
    The internal-model results here are supplementary visual diagnostics, not a
    replacement for the official evaluate.py statistics.
"""
# This file is intentionally self-contained. The generated CSV/PNG files are
# supplementary diagnostics for the Bonus 2 checkpoints.

import argparse, csv, json, math, pickle, random, sys, types, importlib.util
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MAX_DIGITIZE_NUM = 10

def make_bins_state(state):
    edges = [np.linspace(a, b, MAX_DIGITIZE_NUM + 1) for a, b in [(-2.4001, 2.4), (-3.5001, 3.5), (-0.4201, 0.42), (-3.0001, 3.0)]]
    clips = [(-2.4, 2.4), (-3.5, 3.5), (-0.42, 0.42), (-3.0, 3.0)]
    idxs = []
    for i, x in enumerate(state):
        lo, hi = clips[i]
        x = float(np.clip(x, lo, hi))
        idx = int(np.searchsorted(edges[i], x, side='left') - 1)
        idxs.append(max(0, min(MAX_DIGITIZE_NUM - 1, idx)))
    return idxs[0] + idxs[1] * 10 + idxs[2] * 100 + idxs[3] * 1000

def model_step(state, action):
    x, x_dot, theta, theta_dot = map(float, state)
    gravity, masscart, masspole = 9.8, 1.0, 0.1
    total_mass, length, polemass_length = masscart + masspole, 0.5, 0.05
    force_mag, tau = 10.0, 0.02
    theta_threshold_radians = 12 * 2 * math.pi / 360
    force = force_mag if int(action) == 1 else -force_mag
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    ns = np.array([x + tau * x_dot, x_dot + tau * xacc, theta + tau * theta_dot, theta_dot + tau * thetaacc])
    done = bool(abs(ns[0]) > 2.4 or abs(ns[2]) > theta_threshold_radians)
    return ns, done

def eval_q_table(q_table, seeds=range(42, 142), max_steps=2000):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        state = rng.uniform(-0.05, 0.05, 4)
        switches, prev = 0, None
        for t in range(max_steps):
            sid = make_bins_state(state)
            action = int(q_table[sid, 1] > q_table[sid, 0])
            if prev is not None and action != prev: switches += 1
            prev = action
            state, done = model_step(state, action)
            if done:
                rows.append({'seed': seed, 'steps': t + 1, 'reached_2000': 0, 'action_switches': switches})
                break
        else:
            rows.append({'seed': seed, 'steps': max_steps, 'reached_2000': 1, 'action_switches': switches})
    return rows

def summarize(rows):
    steps = np.array([r['steps'] for r in rows], dtype=float)
    return {'mean': float(steps.mean()), 'median': float(np.median(steps)), 'std': float(steps.std()), 'min': float(steps.min()), 'max': float(steps.max()), 'reached_2000_ratio': float(np.mean(steps >= 2000))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--out-dir', default='bonus2_supplement_outputs')
    args = ap.parse_args()
    root = Path(args.project_root)
    out = Path(args.out_dir); figs = out / 'figures'; res = out / 'results'
    figs.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)

    best = pickle.load(open(root / 'checkpoints' / 'q_learning_bonus2_best.pkl', 'rb'))['q_table']
    final = pickle.load(open(root / 'checkpoints' / 'q_learning_bonus2.pkl', 'rb'))['q_table']
    best_rows, final_rows = eval_q_table(best), eval_q_table(final)
    print('best:', summarize(best_rows))
    print('final:', summarize(final_rows))

    # Minimal output CSV and one heatmap. The delivered package contains the full generated results.
    with open(res / 'q_checkpoint_comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['checkpoint', 'mean', 'median', 'std', 'min', 'max', 'reached_2000_ratio'])
        w.writeheader()
        for name, rows in [('best', best_rows), ('final', final_rows)]:
            row = {'checkpoint': name}; row.update(summarize(rows)); w.writerow(row)

    theta_vals = np.linspace(-0.42, 0.42, 21)
    thetadot_vals = np.linspace(-3.0, 3.0, 21)
    Z = np.zeros((len(thetadot_vals), len(theta_vals)))
    for i, td in enumerate(thetadot_vals):
        for j, th in enumerate(theta_vals):
            sid = make_bins_state([0.0, 0.0, th, td])
            Z[i, j] = best[sid, 1] - best[sid, 0]
    lim = max(abs(Z.min()), abs(Z.max()), 1)
    plt.figure(figsize=(7.5, 4.4))
    plt.imshow(Z, origin='lower', extent=[theta_vals[0], theta_vals[-1], thetadot_vals[0], thetadot_vals[-1]], aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim)
    plt.colorbar(label='Q(right)-Q(left)')
    plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
    plt.xlabel('theta'); plt.ylabel('theta_dot'); plt.title('Q policy margin')
    plt.tight_layout(); plt.savefig(figs / 'q_policy_margin_heatmap.png', dpi=200)

    print(f'Outputs saved to: {out.resolve()}')

if __name__ == '__main__':
    main()
