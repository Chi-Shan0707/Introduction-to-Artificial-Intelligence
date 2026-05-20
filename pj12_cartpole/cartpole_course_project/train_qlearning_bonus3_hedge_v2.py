"""Bonus 3 (Approach 2 v2): Multi-Discretization with TRUE Hedge Ensemble.

Phase 1: Train 4 independent Q-tables (different bin offsets), each frozen after training.
Phase 2: Use Hedge algorithm to combine frozen tables online.

Key difference from v1: Q-tables are NEVER updated during Hedge. Only Hedge weights
are updated exponentially based on per-table prediction quality.

Hedge algorithm (Freund & Schapire):
    w_i^{t+1} = w_i^t * exp(-eta * loss_i^t)
    where loss_i^t = 1 - p_i(action_taken)  (how much table i disagreed)
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
EPOCHS_PER_TABLE = 2000   # 每张表独立训练 2000 轮
TARGET = 1000
MAX_DIGITIZE_NUM = 6
NUM_TABLES = 4
HEDGE_ETA = 0.5           # Hedge 学习率
SAVE_DIR = 'checkpoints'

CLIP_RANGES = [
    (-2.4, 2.4),
    (-3.0, 3.0),
    (-0.5, 0.5),
    (-2.0, 2.0),
]
BIN_WIDTHS = np.array([
    (hi - lo) / MAX_DIGITIZE_NUM for (lo, hi) in CLIP_RANGES
])


class QTable:
    """单张 Q 表，训练完成后冻结。"""

    def __init__(self, offset_frac, n_action):
        self.offset_frac = offset_frac
        self.n_action = n_action
        self.q_table = np.random.uniform(
            low=0, high=1, size=(MAX_DIGITIZE_NUM ** 4, n_action)
        )

    def bins(self, clip_min, clip_max):
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def make_bins_state(self, current_state):
        """带偏移的离散化。"""
        cart_pos, cart_v, pole_angle, pole_v = current_state
        cart_pos = np.clip(cart_pos + self.offset_frac * BIN_WIDTHS[0], -2.4, 2.4)
        cart_v = np.clip(cart_v + self.offset_frac * BIN_WIDTHS[1], -3., 3.)
        pole_angle = np.clip(pole_angle + self.offset_frac * BIN_WIDTHS[2], -0.5, 0.5)
        pole_v = np.clip(pole_v + self.offset_frac * BIN_WIDTHS[3], -2., 2.)

        s1 = np.argwhere((cart_pos - self.bins(-2.4001, 2.4)) > 0)[-1, 0]
        s2 = np.argwhere((cart_v - self.bins(-3.001, 3.)) > 0)[-1, 0]
        s3 = np.argwhere((pole_angle - self.bins(-0.5001, 0.5)) > 0)[-1, 0]
        s4 = np.argwhere((pole_v - self.bins(-2.001, 2.)) > 0)[-1, 0]
        return (s1 * MAX_DIGITIZE_NUM ** 0
                + s2 * MAX_DIGITIZE_NUM ** 1
                + s3 * MAX_DIGITIZE_NUM ** 2
                + s4 * MAX_DIGITIZE_NUM ** 3)

    def get_probs(self, state, temperature=1.0):
        """softmax 动作概率。"""
        state_id = self.make_bins_state(state)
        q_vals = self.q_table[state_id, :]
        q_shifted = q_vals - np.max(q_vals)
        exp_q = np.exp(q_shifted / temperature)
        return exp_q / np.sum(exp_q)

    def predict(self, state):
        """贪心动作。"""
        state_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[state_id, :]))

    def update_q(self, state, action, reward, next_state, lr, gamma):
        """标准 Q-learning 更新。"""
        current_id = self.make_bins_state(state)
        next_id = self.make_bins_state(next_state)
        td_target = reward + gamma * np.max(self.q_table[next_id, :])
        td_error = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += lr * td_error


class HedgeAgent:
    """
    Phase 2: Hedge 集成器。
    Q-tables 是 frozen 的（只读），只更新 Hedge 权重。
    """

    def __init__(self, tables, n_action, eta=HEDGE_ETA):
        self.tables = tables          # frozen Q-tables
        self.n_action = n_action
        self.num_tables = len(tables)
        self.eta = eta
        self.hedge_weights = np.ones(self.num_tables) / self.num_tables

    def _compute_loss(self, state, action_taken):
        """
        计算每张表的 loss：
        loss_i = 1 - softmax(Q_i(s))[action_taken]
        表 i 越不支持 taken action，loss 越高。
        """
        losses = np.zeros(self.num_tables)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=1.0)
            losses[i] = 1.0 - probs[action_taken]
        return losses

    def update_weights(self, state, action_taken):
        """Hedge 指数更新。"""
        losses = self._compute_loss(state, action_taken)
        self.hedge_weights *= np.exp(-self.eta * losses)
        # 归一化
        total = np.sum(self.hedge_weights)
        if total > 0:
            self.hedge_weights /= total
        else:
            self.hedge_weights = np.ones(self.num_tables) / self.num_tables

    def decide_action(self, state, epsilon=0.0):
        """softmax 聚合所有表的概率分布，然后采样。"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.n_action)

        # 加权聚合概率
        agg_probs = np.zeros(self.n_action)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=1.0)
            agg_probs += self.hedge_weights[i] * probs

        agg_probs /= np.sum(agg_probs)
        return int(np.random.choice(self.n_action, p=agg_probs))

    def predict(self, state):
        """贪心动作（evaluate/vis 用）。"""
        agg_probs = np.zeros(self.n_action)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=0.5)
            agg_probs += self.hedge_weights[i] * probs
        return int(np.argmax(agg_probs))

    def policy_info(self, state):
        """vis.py 用。"""
        info = {"HEDGE": f"[{', '.join(f'{w:.2f}' for w in self.hedge_weights)}]"}
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=0.5)
            info[f"T{i}"] = f"L={probs[0]:.2f} R={probs[1]:.2f}"
        return info


# =============================================================================
# Phase 1: 独立训练 4 张 Q 表
# =============================================================================

def train_single_table(table, env, epochs, lr, gamma, epsilon):
    """独立训练单张 Q 表，返回训练记录。"""
    step_record = []
    for i in range(epochs):
        current_state = env.reset()[0]
        for step in range(TARGET):
            # ε-greedy
            state_id = table.make_bins_state(current_state)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(table.n_action)
            else:
                action = np.argmax(table.q_table[state_id, :])

            next_state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                reward = step - TARGET
            else:
                reward = 0

            table.update_q(current_state, action, reward, next_state, lr, gamma)
            current_state = next_state

            if terminated or truncated:
                step_record.append(step)
                break

        if (i + 1) % 100 == 0:
            recent = np.mean(step_record[-100:]) if len(step_record) >= 100 else np.mean(step_record)
            print(f"  Episode {i+1}: recent avg = {recent:.1f} steps")

    return step_record


def train_all_tables():
    """Phase 1: 训练 4 张独立的 Q 表。"""
    env = gym.make('CartPole-v1', max_episode_steps=TARGET)
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    tables = []
    for t in range(NUM_TABLES):
        offset = t / NUM_TABLES  # 0, 0.25, 0.5, 0.75
        print(f"\n=== Training Table {t} (offset={offset:.2f}) ===")
        table = QTable(offset, n_action)
        record = train_single_table(table, env, EPOCHS_PER_TABLE, LR, GAMMA, EPSILON)
        print(f"Table {t} done. Last 20 avg = {np.mean(record[-20:]):.1f}")
        tables.append(table)

    env.close()
    return tables, n_state, n_action


# =============================================================================
# Phase 2: Hedge 集成 + 在线测试
# =============================================================================

def test_hedge(tables, n_action, episodes=1000):
    """Phase 2: 冻结所有表，用 Hedge 在线集成。"""
    env = gym.make('CartPole-v1', max_episode_steps=TARGET)
    agent = HedgeAgent(tables, n_action, eta=HEDGE_ETA)

    step_record = []
    for i in range(episodes):
        current_state = env.reset()[0]
        for step in range(TARGET):
            action = agent.decide_action(current_state, epsilon=0.0)
            next_state, _, terminated, truncated, _ = env.step(action)

            # 更新 Hedge 权重（Q 表不动！）
            agent.update_weights(current_state, action)

            current_state = next_state
            if terminated or truncated:
                step_record.append(step)
                break

        if (i + 1) % 100 == 0:
            recent = np.mean(step_record[-100:])
            print(f"  Hedge Episode {i+1}: recent avg = {recent:.1f}, "
                  f"weights=[{', '.join(f'{w:.2f}' for w in agent.hedge_weights)}]")

    env.close()
    return step_record, agent.hedge_weights


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Phase 1: 独立训练 4 张表
    print("=" * 60)
    print("PHASE 1: Training 4 independent Q-tables")
    print("=" * 60)
    tables, n_state, n_action = train_all_tables()

    # 保存所有表
    for i, table in enumerate(tables):
        path = os.path.join(SAVE_DIR, f'q_learning_bonus3_hedge_v2_table{i}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'q_table': table.q_table, 'offset': table.offset_frac}, f)
        print(f"Table {i} saved: {path}")

    # Phase 2: Hedge 在线集成
    print("\n" + "=" * 60)
    print("PHASE 2: Hedge ensemble (frozen tables)")
    print("=" * 60)
    record, final_weights = test_hedge(tables, n_action, episodes=1000)

    print(f"\nFinal Hedge weights: {final_weights}")
    print(f"Last 20 episodes avg: {np.mean(record[-20:]):.1f}")

    # 保存 Hedge agent
    agent_data = {
        'tables': [(t.offset_frac, t.q_table) for t in tables],
        'hedge_weights': final_weights,
        'n_state': n_state, 'n_action': n_action,
    }
    agent_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_hedge_v2_agent.pkl')
    with open(agent_path, 'wb') as f:
        pickle.dump(agent_data, f)
    print(f"Hedge agent saved: {agent_path}")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Hedge v2: Ensemble Training Curve (frozen tables)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_hedge_v2_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Curve saved: {plot_path}")


if __name__ == "__main__":
    main()
