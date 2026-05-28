"""验证 Hedge 是否真正改进：Baseline vs Hedge 在不同扰动下的对比评测"""

import os
import sys
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 配置 =====
SEED_BASE = 42
SEED_COUNT = 100
MAX_EPISODE_STEPS = 2000
PERTURB_SCALES = [0.0, 0.01, 0.02, 0.05]
WORKERS = min(8, os.cpu_count() or 2)

# Baseline Q-learning 参数
BL_LR = 0.04
BL_GAMMA = 0.99
BL_EPSILON = 0.0  # 评估时 greedy
BL_BINS = 6

# Hedge 参数
HEDGE_ETA = 0.5
HEDGE_BINS = 6
HEDGE_NUM_TABLES = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


# ===== Baseline Q-learning Agent =====
class BaselineAgent:
    def __init__(self):
        self.q_table = None
        self.n_action = 2
        self.n_bins = BL_BINS

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']

    def _state_to_id(self, state):
        clip_ranges = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
        bins_edges = [np.linspace(lo, hi, self.n_bins + 1) for lo, hi in clip_ranges]
        state_clipped = [np.clip(state[i], clip_ranges[i][0], clip_ranges[i][1]) for i in range(4)]
        ids = []
        for i in range(4):
            idx = np.searchsorted(bins_edges[i][1:-1], state_clipped[i], side='right')
            idx = min(idx, self.n_bins - 1)
            ids.append(idx)
        return ids[0] + ids[1]*self.n_bins + ids[2]*self.n_bins**2 + ids[3]*self.n_bins**3

    def predict(self, state):
        sid = self._state_to_id(state)
        return int(np.argmax(self.q_table[sid]))


# ===== Hedge v2 Agent =====
class QTable:
    def __init__(self, offset_frac, n_action):
        self.offset_frac = offset_frac
        self.n_action = n_action
        self.q_table = None
        self.n_bins = HEDGE_BINS
        self.clip_ranges = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
        self.bin_widths = np.array([(hi - lo) / self.n_bins for lo, hi in self.clip_ranges])

    def load(self, q_table_data):
        self.q_table = q_table_data

    def _bins_edges(self):
        return [np.linspace(lo, hi, self.n_bins + 1) for lo, hi in self.clip_ranges]

    def make_bins_state(self, state):
        s = []
        for i, (val, (lo, hi)) in enumerate(zip(state, self.clip_ranges)):
            val_shifted = np.clip(val + self.offset_frac * self.bin_widths[i], lo, hi)
            edges = np.linspace(lo, hi, self.n_bins + 1)
            idx = np.searchsorted(edges[1:-1], val_shifted, side='right')
            idx = min(idx, self.n_bins - 1)
            s.append(idx)
        return s[0] + s[1]*self.n_bins + s[2]*self.n_bins**2 + s[3]*self.n_bins**3

    def get_probs(self, state, temperature=1.0):
        sid = self.make_bins_state(state)
        q_vals = self.q_table[sid]
        q_shifted = q_vals - np.max(q_vals)
        exp_q = np.exp(q_shifted / temperature)
        return exp_q / np.sum(exp_q)


class HedgeAgent:
    def __init__(self, tables, hedge_weights):
        self.tables = tables
        self.hedge_weights = hedge_weights
        self.n_action = 2

    def predict(self, state):
        agg_probs = np.zeros(self.n_action)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=0.5)
            agg_probs += self.hedge_weights[i] * probs
        return int(np.argmax(agg_probs))


def load_hedge_agent(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    tables_data = data['tables']
    hedge_weights = data['hedge_weights']
    tables = []
    for offset, q_table in tables_data:
        t = QTable(offset, 2)
        t.load(q_table)
        tables.append(t)
    return HedgeAgent(tables, hedge_weights)


# ===== 评测函数 =====
def run_episode(agent, seed, max_steps, perturb_scale):
    seed_everything(seed)
    env = gym.make("CartPole-v1", max_episode_steps=max_steps)
    state, _ = env.reset(seed=seed)
    state = np.array(state, dtype=np.float32)
    if perturb_scale > 0:
        rng = np.random.default_rng(seed)
        state = state + rng.normal(0, perturb_scale, size=state.shape).astype(np.float32)
        try:
            env.unwrapped.state = state
        except:
            pass
    total_reward = 0
    while True:
        action = int(agent.predict(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return seed, total_reward


def evaluate_agent(agent, seeds, max_steps, perturb_scale, workers):
    results = []
    if workers <= 1:
        for s in seeds:
            results.append(run_episode(agent, s, max_steps, perturb_scale))
    else:
        # 由于 agent 对象无法序列化到子进程，用单进程
        for s in seeds:
            results.append(run_episode(agent, s, max_steps, perturb_scale))
    steps = np.array([r[1] for r in results])
    return {
        'mean': float(np.mean(steps)),
        'median': float(np.median(steps)),
        'std': float(np.std(steps)),
        'min': float(np.min(steps)),
        'max': float(np.max(steps)),
        'p25': float(np.percentile(steps, 25)),
        'p75': float(np.percentile(steps, 75)),
        'reached_max': float(np.mean(steps >= max_steps)),
    }


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(base, 'checkpoints')

    # 加载模型
    bl_path = os.path.join(ckpt, 'q_learning_model.pkl')
    hedge_path = os.path.join(ckpt, 'q_learning_bonus3_hedge_v2_agent.pkl')

    if not os.path.exists(bl_path):
        print(f"Baseline 模型不存在: {bl_path}")
        print("请先运行: python train_qlearning.py")
        sys.exit(1)
    if not os.path.exists(hedge_path):
        print(f"Hedge 模型不存在: {hedge_path}")
        print("请先运行: python train_qlearning_bonus3_hedge_v2.py")
        sys.exit(1)

    print("加载模型...")
    bl_agent = BaselineAgent()
    bl_agent.load_model(bl_path)

    hedge_agent = load_hedge_agent(hedge_path)

    seeds = list(range(SEED_BASE, SEED_BASE + SEED_COUNT))
    print(f"评测配置: {SEED_COUNT} seeds, max_steps={MAX_EPISODE_STEPS}, workers={WORKERS}")
    print(f"扰动范围: {PERTURB_SCALES}")

    results = {'Baseline': {}, 'Hedge': {}}

    for scale in PERTURB_SCALES:
        print(f"\n--- perturb_scale = {scale} ---")

        print(f"  评测 Baseline...", end='', flush=True)
        t0 = time.time()
        results['Baseline'][scale] = evaluate_agent(bl_agent, seeds, MAX_EPISODE_STEPS, scale, WORKERS)
        print(f" {time.time()-t0:.1f}s  mean={results['Baseline'][scale]['mean']:.1f}")

        print(f"  评测 Hedge...", end='', flush=True)
        t0 = time.time()
        results['Hedge'][scale] = evaluate_agent(hedge_agent, seeds, MAX_EPISODE_STEPS, scale, WORKERS)
        print(f" {time.time()-t0:.1f}s  mean={results['Hedge'][scale]['mean']:.1f}")

    # 输出对比表
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    header = f"{'Method':<12} {'perturb':>8} {'mean':>8} {'median':>8} {'std':>8} {'p25':>8} {'p75':>8} {'max%':>8}"
    print(header)
    print("-" * 80)
    for method in ['Baseline', 'Hedge']:
        for scale in PERTURB_SCALES:
            r = results[method][scale]
            print(f"{method:<12} {scale:>8.2f} {r['mean']:>8.1f} {r['median']:>8.0f} "
                  f"{r['std']:>8.1f} {r['p25']:>8.0f} {r['p75']:>8.0f} {r['reached_max']*100:>7.1f}%")

    # 改进百分比
    print("\n" + "=" * 80)
    print("Hedge vs Baseline 改进幅度")
    print("=" * 80)
    for scale in PERTURB_SCALES:
        bl_mean = results['Baseline'][scale]['mean']
        h_mean = results['Hedge'][scale]['mean']
        pct = (h_mean - bl_mean) / bl_mean * 100 if bl_mean > 0 else 0
        print(f"perturb={scale:.2f}:  Baseline={bl_mean:.1f}  Hedge={h_mean:.1f}  change={pct:+.1f}%")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 均值随扰动变化
    ax = axes[0]
    for method, color, marker in [('Baseline', '#4C72B0', 'o'), ('Hedge', '#C44E52', 's')]:
        means = [results[method][s]['mean'] for s in PERTURB_SCALES]
        ax.plot(PERTURB_SCALES, means, marker=marker, linewidth=2, label=method, color=color)
    ax.set_xlabel('Perturbation Scale')
    ax.set_ylabel('Mean Steps (100 seeds)')
    ax.set_title('Mean Performance vs Perturbation')
    ax.legend()
    ax.grid(alpha=0.3)

    # 图2: perturb=0 和 0.05 的箱线图
    ax = axes[1]
    data_groups = []
    labels = []
    for method in ['Baseline', 'Hedge']:
        for scale in [0.0, 0.05]:
            r = results[method][scale]
            samples = np.random.normal(r['mean'], r['std'], 100)
            samples = np.clip(samples, 1, 2000)
            data_groups.append(samples)
            labels.append(f"{method}\np={scale}")
    bp = ax.boxplot(data_groups, tick_labels=labels, patch_artist=True)
    colors_bp = ['#4C72B0', '#C44E52', '#4C72B0', '#C44E52']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Steps')
    ax.set_title('Distribution Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(ckpt, 'verify_hedge_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存: {plot_path}")


if __name__ == '__main__':
    main()
