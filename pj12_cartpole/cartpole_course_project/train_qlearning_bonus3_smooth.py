"""Bonus 3 (Approach 1): Neighborhood Smoothing Q-Learning.

Instead of sharp bin boundaries, we linearly interpolate Q-values between adjacent
bins based on where the continuous state falls within its bin. This makes the policy
smooth across boundaries, reducing fragility to perturbations.
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

# 各维度 clip 范围（用于计算 bin 内的相对位置）
CLIP_RANGES = [
    (-2.4, 2.4),   # cart_pos
    (-3.0, 3.0),   # cart_v
    (-0.5, 0.5),   # pole_angle
    (-2.0, 2.0),   # pole_v
]


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, epsilon):
        self.n_action = n_action
        self.n_state = n_state
        self.q_table = np.random.uniform(low=0, high=1,
                                         size=(MAX_DIGITIZE_NUM ** n_state, n_action))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def bins(self, clip_min, clip_max):
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def make_bins_state(self, current_state):
        """把 4 维连续观测映射到离散 state_id（和原始一样）。"""
        cart_pos, cart_v, pole_angle, pole_v = current_state
        cart_pos = np.clip(cart_pos, -2.4, 2.4)
        cart_v = np.clip(cart_v, -3., 3.)
        pole_angle = np.clip(pole_angle, -0.5, 0.5)
        pole_v = np.clip(pole_v, -2., 2.)
        s1 = np.argwhere((cart_pos - self.bins(-2.4001, 2.4)) > 0)[-1, 0]
        s2 = np.argwhere((cart_v - self.bins(-3.001, 3.)) > 0)[-1, 0]
        s3 = np.argwhere((pole_angle - self.bins(-0.5001, 0.5)) > 0)[-1, 0]
        s4 = np.argwhere((pole_v - self.bins(-2.001, 2.)) > 0)[-1, 0]
        return (s1 * MAX_DIGITIZE_NUM ** 0
                + s2 * MAX_DIGITIZE_NUM ** 1
                + s3 * MAX_DIGITIZE_NUM ** 2
                + s4 * MAX_DIGITIZE_NUM ** 3)

    def _get_bin_fractions(self, current_state):
        """计算状态在每个维度 bin 内的相对位置（0=左边界, 1=右边界）。"""
        fractions = []
        for i, (val, (lo, hi)) in enumerate(zip(current_state, CLIP_RANGES)):
            val = np.clip(val, lo, hi)
            bin_width = (hi - lo) / MAX_DIGITIZE_NUM
            # 找到当前在哪个 bin
            bin_idx = int((val - lo) / bin_width)
            bin_idx = min(bin_idx, MAX_DIGITIZE_NUM - 1)
            # 在 bin 内的位置
            bin_lo = lo + bin_idx * bin_width
            frac = (val - bin_lo) / bin_width
            frac = np.clip(frac, 0.0, 1.0)
            fractions.append((bin_idx, frac))
        return fractions

    def _get_neighbor_ids_and_weights(self, current_state):
        """返回相邻 bin 的 state_id 和插值权重（用于邻域平滑）。"""
        fractions = self._get_bin_fractions(current_state)

        # 构建 2^4 = 16 个相邻 bin 的索引组合
        neighbor_ids = []
        weights = []

        for offset_bits in range(16):
            # 每个 bit 表示该维度取上界(1)还是下界(0)
            bin_indices = []
            w = 1.0
            for dim in range(4):
                bin_idx, frac = fractions[dim]
                if (offset_bits >> dim) & 1:
                    # 取右邻居
                    neighbor_idx = min(bin_idx + 1, MAX_DIGITIZE_NUM - 1)
                    w *= frac
                else:
                    # 取左邻居
                    neighbor_idx = bin_idx
                    w *= (1.0 - frac)
                bin_indices.append(neighbor_idx)

            state_id = (bin_indices[0] * MAX_DIGITIZE_NUM ** 0
                      + bin_indices[1] * MAX_DIGITIZE_NUM ** 1
                      + bin_indices[2] * MAX_DIGITIZE_NUM ** 2
                      + bin_indices[3] * MAX_DIGITIZE_NUM ** 3)

            neighbor_ids.append(state_id)
            weights.append(w)

        return neighbor_ids, np.array(weights)

    def _smooth_q_values(self, current_state):
        """对相邻 bin 的 Q 值做线性插值。"""
        neighbor_ids, weights = self._get_neighbor_ids_and_weights(current_state)
        q_vals = np.zeros(self.n_action)
        for sid, w in zip(neighbor_ids, weights):
            q_vals += w * self.q_table[sid, :]
        return q_vals

    def update_q_table(self, current_state, action, reward, next_state):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)

        # {这里原本空的，你后来填上去了}
        # TODO 1: Q-table Bellman TD 更新（更新中心 bin）
        td_target = reward + self.gamma * np.max(self.q_table[next_id, :])
        td_error = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += self.lr * td_error

    def decide_action(self, current_state):
        """ε-greedy，但用平滑后的 Q 值做决策。"""
        current_id = self.make_bins_state(current_state)

        # {这里原本空的，你后来填上去了}
        # TODO 2: epsilon-greedy（用平滑 Q 值）
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            # 用邻域平滑后的 Q 值
            smooth_q = self._smooth_q_values(current_state)
            action = np.argmax(smooth_q)

        return int(action)

    def predict(self, state):
        """贪心动作，evaluate/vis 用（平滑版）。"""
        smooth_q = self._smooth_q_values(state)
        return int(np.argmax(smooth_q))

    def policy_info(self, state):
        current_id = self.make_bins_state(state)
        q_vals = self.q_table[current_id, :]
        smooth_q = self._smooth_q_values(state)
        return {
            "Q(LEFT)": f"{q_vals[0]:+.3f}",
            "Q(RIGHT)": f"{q_vals[1]:+.3f}",
            "SmoothQ(LEFT)": f"{smooth_q[0]:+.3f}",
            "SmoothQ(RIGHT)": f"{smooth_q[1]:+.3f}",
        }

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon,
                'n_state': self.n_state, 'n_action': self.n_action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, f)
        print(f"Q-learning Bonus3-Smooth 模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning Bonus3-Smooth 模型已加载: {filepath}")


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
    print("开始训练 Q-learning Bonus3-Smooth 模型...")
    record = env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_bonus3_smooth_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Bonus3-Smooth Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_smooth_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()
