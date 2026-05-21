"""Bonus 3 (Approach 4): Gradient-Driven Adaptive Discretization.

Dynamic bin boundaries adapt based on Q-value gradients. Bins become finer
in regions where Q-values change rapidly (high gradient), and coarser where
Q-values are flat (low gradient).
"""

import os
import pickle
from datetime import datetime

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 超参数
# =============================================================================
LR = 0.04
GAMMA = 0.99
EPSILON = 0.1
EPOCHS = 2000
TARGET = 1000
MAX_DIGITIZE_NUM = 6
SAVE_DIR = 'checkpoints'
ADAPT_INTERVAL = 250   # 每 250 episode 调整一次边界
GRAD_ALPHA = 15.0      # 梯度密度系数

CLIP_RANGES = [
    (-2.4, 2.4),
    (-3.0, 3.0),
    (-0.5, 0.5),
    (-2.0, 2.0),
]


class Agent:
    def __init__(self, n_state, n_action, lr=LR, gamma=GAMMA, epsilon=EPSILON):
        self.n_action = n_action
        self.n_state = n_state
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # 动态边界：每维一个边界数组 [lo, b1, b2, ..., hi]
        self.bin_boundaries = [
            np.linspace(lo, hi, MAX_DIGITIZE_NUM + 1)
            for (lo, hi) in CLIP_RANGES
        ]

        self.q_table = np.random.uniform(
            low=0, high=1, size=(MAX_DIGITIZE_NUM ** n_state, n_action)
        )

        # 统计信息
        self.td_errors = np.zeros(MAX_DIGITIZE_NUM ** n_state)
        self.visit_counts = np.zeros(MAX_DIGITIZE_NUM ** n_state)

    # -------------------------------------------------------------------------
    # 辅助函数
    # -------------------------------------------------------------------------
    def _state_to_bins(self, state_id):
        s0 = state_id % MAX_DIGITIZE_NUM
        s1 = (state_id // MAX_DIGITIZE_NUM) % MAX_DIGITIZE_NUM
        s2 = (state_id // (MAX_DIGITIZE_NUM ** 2)) % MAX_DIGITIZE_NUM
        s3 = (state_id // (MAX_DIGITIZE_NUM ** 3)) % MAX_DIGITIZE_NUM
        return [s0, s1, s2, s3]

    def _bins_to_state(self, bins):
        return (bins[0] * MAX_DIGITIZE_NUM ** 0
                + bins[1] * MAX_DIGITIZE_NUM ** 1
                + bins[2] * MAX_DIGITIZE_NUM ** 2
                + bins[3] * MAX_DIGITIZE_NUM ** 3)

    def _get_dim_mask(self, dim, bin_idx):
        mask = np.zeros(MAX_DIGITIZE_NUM ** 4, dtype=bool)
        for state_id in range(MAX_DIGITIZE_NUM ** 4):
            bins = self._state_to_bins(state_id)
            if bins[dim] == bin_idx:
                mask[state_id] = True
        return mask

    def _get_bin_center(self, state_id, boundaries):
        bins = self._state_to_bins(state_id)
        center = []
        for b, bound in zip(bins, boundaries):
            c = (bound[b] + bound[b + 1]) / 2.0
            center.append(c)
        return center

    def _lookup_with_boundaries(self, state, boundaries):
        s = []
        for val, bound in zip(state, boundaries):
            val = np.clip(val, bound[0], bound[-1])
            idx = np.searchsorted(bound[1:-1], val, side='right')
            idx = min(idx, MAX_DIGITIZE_NUM - 1)
            s.append(idx)
        return self._bins_to_state(s)

    # -------------------------------------------------------------------------
    # 核心接口
    # -------------------------------------------------------------------------
    def make_bins_state(self, state):
        s = []
        for val, boundaries in zip(state, self.bin_boundaries):
            val = np.clip(val, boundaries[0], boundaries[-1])
            idx = np.searchsorted(boundaries[1:-1], val, side='right')
            idx = min(idx, MAX_DIGITIZE_NUM - 1)
            s.append(idx)
        return self._bins_to_state(s)

    def _remap_q_table(self, old_boundaries, momentum=0.3):
        """边界调整后，用最近邻映射迁移 Q 表。"""
        old_q_table = self.q_table.copy()
        new_q = np.zeros_like(self.q_table)

        for state_id in range(MAX_DIGITIZE_NUM ** 4):
            center = self._get_bin_center(state_id, self.bin_boundaries)
            old_state_id = self._lookup_with_boundaries(center, old_boundaries)
            new_q[state_id] = old_q_table[old_state_id]

        self.q_table = momentum * self.q_table + (1.0 - momentum) * new_q

    def _adapt_boundaries(self):
        """基于 Q 值梯度调整边界。"""
        old_boundaries = [b.copy() for b in self.bin_boundaries]

        for dim in range(4):
            lo, hi = CLIP_RANGES[dim]
            boundaries = self.bin_boundaries[dim]

            # 每 bin 的平均 max-Q
            q_per_bin = []
            for b in range(MAX_DIGITIZE_NUM):
                mask = self._get_dim_mask(dim, b)
                if np.any(mask):
                    q_max = np.max(self.q_table[mask, :], axis=1)
                    q_per_bin.append(float(np.mean(q_max)))
                else:
                    q_per_bin.append(0.0)

            # 相邻 bin 的梯度
            gradients = []
            for b in range(MAX_DIGITIZE_NUM - 1):
                g = abs(q_per_bin[b + 1] - q_per_bin[b])
                gradients.append(g + 0.005)

            # 累积密度：高梯度区域密度更高 -> 边界更密
            cum_density = [0.0]
            for b in range(MAX_DIGITIZE_NUM):
                width = boundaries[b + 1] - boundaries[b]
                # bin 内密度（用相邻梯度）
                if b < len(gradients):
                    dens = 1.0 + GRAD_ALPHA * gradients[b]
                else:
                    dens = 1.0 + GRAD_ALPHA * gradients[-1]
                cum_density.append(cum_density[-1] + dens * width)

            total_dens = cum_density[-1]
            if total_dens < 0.001:
                continue

            # 在累积密度上均匀采样新边界
            new_boundaries = [lo]
            for i in range(1, MAX_DIGITIZE_NUM):
                target = total_dens * i / MAX_DIGITIZE_NUM
                for j in range(len(cum_density) - 1):
                    if cum_density[j] <= target <= cum_density[j + 1]:
                        denom = cum_density[j + 1] - cum_density[j] + 1e-8
                        frac = (target - cum_density[j]) / denom
                        pos = boundaries[j] + frac * (boundaries[j + 1] - boundaries[j])
                        new_boundaries.append(float(pos))
                        break
            new_boundaries.append(hi)
            self.bin_boundaries[dim] = np.array(new_boundaries)

        self._remap_q_table(old_boundaries)
        print(f"  [Adapt] dim0 boundaries: {self.bin_boundaries[0].round(3)}")

    def update_q_table(self, current_state, action, reward, next_state):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)

        td_target = reward + self.gamma * np.max(self.q_table[next_id, :])
        td_error = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += self.lr * td_error

        # 统计
        self.td_errors[current_id] = 0.9 * self.td_errors[current_id] + 0.1 * abs(td_error)
        self.visit_counts[current_id] += 1

    def decide_action(self, current_state):
        current_id = self.make_bins_state(current_state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_action)
        return int(np.argmax(self.q_table[current_id, :]))

    def predict(self, state):
        state_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[state_id, :]))

    def policy_info(self, state):
        state_id = self.make_bins_state(state)
        q_vals = self.q_table[state_id, :]
        return {
            "Q(LEFT)": f"{q_vals[0]:+.3f}",
            "Q(RIGHT)": f"{q_vals[1]:+.3f}",
            "StateID": str(state_id),
        }

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'bin_boundaries': self.bin_boundaries,
                'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon,
                'n_state': self.n_state, 'n_action': self.n_action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, f)
        print(f"自适应梯度离散化模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.bin_boundaries = data['bin_boundaries']
        self.lr = data['lr']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.n_state = data['n_state']
        self.n_action = data['n_action']
        print(f"自适应梯度离散化模型已加载: {filepath}")


class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1', max_episode_steps=TARGET)
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        self.agent = Agent(n_state, n_action, LR, GAMMA, EPSILON)

    def train(self, epoch=EPOCHS, save_path=None):
        step_record = []
        for i in range(epoch):
            if i > 0 and i % ADAPT_INTERVAL == 0:
                print(f"\n--- Episode {i}: Adapting boundaries (gradient-driven) ---")
                self.agent._adapt_boundaries()

            current_state = self.env.reset()[0]
            for step in range(TARGET):
                action = self.agent.decide_action(current_state)
                next_state, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    reward = step - TARGET
                else:
                    reward = 0
                self.agent.update_q_table(current_state, action, reward, next_state)
                current_state = next_state
                if terminated or truncated:
                    if (i + 1) % 100 == 0:
                        print(f'{i} Episode: Finished after {step + 1} time steps')
                    step_record.append(step)
                    break

        self.env.close()
        if save_path is not None:
            self.agent.save_model(save_path)
        return step_record


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = Environment()
    print("开始训练自适应梯度离散化 Q-learning...")
    record = env.train(
        epoch=EPOCHS,
        save_path=os.path.join(SAVE_DIR, 'q_learning_bonus3_adaptive_gradient_model.pkl')
    )

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Adaptive Gradient-Driven Discretization Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_adaptive_gradient_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()
