"""Bonus 3 (Approach 2): Multi-Discretization Q-Learning with Hedge Ensemble.

Train multiple Q-tables with different bin offsets. At decision time, each table
votes; Hedge algorithm maintains online weights over tables. Tables that make
better predictions get higher weights.
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
NUM_TABLES = 4          # 4 张 Q 表，偏移量分别为 0, 1/4, 1/2, 3/4 bin 宽度
HEDGE_ETA = 0.1         # Hedge 学习率
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
    """单张 Q 表，带有固定的 bin 偏移。"""

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

        # 应用偏移：每个维度加上 offset_frac * bin_width
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

    def update_q(self, state, action, reward, next_state, lr, gamma):
        current_id = self.make_bins_state(state)
        next_id = self.make_bins_state(next_state)
        td_target = reward + gamma * np.max(self.q_table[next_id, :])
        td_error = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += lr * td_error

    def predict(self, state, temperature=1.0):
        """softmax 采样动作。temperature 控制探索程度。"""
        state_id = self.make_bins_state(state)
        q_vals = self.q_table[state_id, :]
        q_shifted = q_vals - np.max(q_vals)
        exp_q = np.exp(q_shifted / temperature)
        probs = exp_q / np.sum(exp_q)
        return int(np.random.choice(self.n_action, p=probs))

    def get_q(self, state):
        state_id = self.make_bins_state(state)
        return self.q_table[state_id, :]

    def get_probs(self, state, temperature=1.0):
        """返回 softmax 动作概率。"""
        state_id = self.make_bins_state(state)
        q_vals = self.q_table[state_id, :]
        q_shifted = q_vals - np.max(q_vals)
        exp_q = np.exp(q_shifted / temperature)
        return exp_q / np.sum(exp_q)


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, epsilon):
        self.n_action = n_action
        self.n_state = n_state
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # 创建多张 Q 表，偏移量均匀分布
        self.tables = []
        for i in range(NUM_TABLES):
            offset = i / NUM_TABLES  # 0, 0.25, 0.5, 0.75
            self.tables.append(QTable(offset, n_action))

        # Hedge 权重，初始均匀
        self.hedge_weights = np.ones(NUM_TABLES) / NUM_TABLES
        self.hedge_eta = HEDGE_ETA

    def _compute_hedge_loss(self, state, action_taken, next_state, reward, done):
        """
        计算每张表的 loss：
        loss_i = 1 - softmax(Q_i(s))[action_taken]
        即：表 i 认为 taken action 的概率越低，loss 越高。
        """
        losses = np.zeros(NUM_TABLES)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=1.0)
            losses[i] = 1.0 - probs[action_taken]
        return losses

    def update_hedge(self, state, action_taken, next_state, reward, done):
        """用 Hedge 算法更新权重。"""
        losses = self._compute_hedge_loss(state, action_taken, next_state, reward, done)
        self.hedge_weights *= np.exp(-self.hedge_eta * losses)
        # 归一化
        total = np.sum(self.hedge_weights)
        if total > 0:
            self.hedge_weights /= total
        else:
            self.hedge_weights = np.ones(NUM_TABLES) / NUM_TABLES

    def update_q_tables(self, state, action, reward, next_state):
        """同时更新所有 Q 表。"""
        for table in self.tables:
            table.update_q(state, action, reward, next_state, self.lr, self.gamma)

    def decide_action(self, current_state):
        """ε-greedy，ensemble 级别，softmax 聚合。"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_action)

        # 每张表输出 softmax 概率，按 Hedge 权重加权聚合
        agg_probs = np.zeros(self.n_action)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(current_state, temperature=1.0)
            agg_probs += self.hedge_weights[i] * probs

        # 归一化后采样
        agg_probs /= np.sum(agg_probs)
        return int(np.random.choice(self.n_action, p=agg_probs))

    def predict(self, state):
        """贪心动作，evaluate/vis 用（softmax 聚合后 argmax）。"""
        agg_probs = np.zeros(self.n_action)
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=0.5)  # 评估时温度更低，更确定
            agg_probs += self.hedge_weights[i] * probs
        return int(np.argmax(agg_probs))

    def policy_info(self, state):
        """vis.py 用，显示各表 softmax 概率和 Hedge 权重。"""
        info = {"HEDGE": f"[{', '.join(f'{w:.2f}' for w in self.hedge_weights)}]"}
        for i, table in enumerate(self.tables):
            probs = table.get_probs(state, temperature=0.5)
            info[f"T{i}"] = f"L={probs[0]:.2f} R={probs[1]:.2f}(w={self.hedge_weights[i]:.2f})"
        return info

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            'tables': [(t.offset_frac, t.q_table) for t in self.tables],
            'hedge_weights': self.hedge_weights,
            'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon,
            'n_state': self.n_state, 'n_action': self.n_action,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Q-learning Bonus3-Hedge 模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        for i, (offset, q_table) in enumerate(data['tables']):
            self.tables[i].offset_frac = offset
            self.tables[i].q_table = q_table
        self.hedge_weights = data['hedge_weights']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning Bonus3-Hedge 模型已加载: {filepath}")


class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1', max_episode_steps=TARGET)
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        self.agent = Agent(n_state, n_action, LR, GAMMA, EPSILON)

    def train(self, epoch=EPOCHS, save_path=None):
        step_record = []
        for i in range(epoch):
            current_state = self.env.reset()[0]

            for step in range(TARGET):
                action = self.agent.decide_action(current_state)
                next_state, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    reward = step - TARGET
                else:
                    reward = 0

                # 同时更新所有 Q 表
                self.agent.update_q_tables(current_state, action, reward, next_state)
                # 更新 Hedge 权重
                self.agent.update_hedge(current_state, action, next_state, reward, terminated or truncated)

                current_state = next_state

                if terminated or truncated:
                    if (i + 1) % 100 == 0:
                        print(f'{i} Episode: Finished after {step + 1} time steps, '
                              f'Hedge weights=[{", ".join(f"{w:.2f}" for w in self.agent.hedge_weights)}]')
                    step_record.append(step)
                    break

        self.env.close()
        if save_path is not None:
            self.agent.save_model(save_path)
        return step_record


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = Environment()
    print("开始训练 Q-learning Bonus3-Hedge 模型...")
    record = env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_bonus3_hedge_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Bonus3-Hedge Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_hedge_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")
    print(f"最终 Hedge 权重: {env.agent.hedge_weights}")


if __name__ == "__main__":
    main()
