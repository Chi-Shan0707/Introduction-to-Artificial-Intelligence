"""Bonus 3 (Approach 1 v2): Smoothstep (C1) Neighborhood Smoothing Q-Learning.

Instead of piecewise-constant (original) or piecewise-linear (v1), we use
smoothstep interpolation: weight(t) = 3t^2 - 2t^3, which gives C1 continuity
(smooth first derivative) across bin boundaries.

This is the standard Hermite smoothstep used in computer graphics and gives
a much smoother policy surface than linear interpolation.
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

CLIP_RANGES = [
    (-2.4, 2.4),
    (-3.0, 3.0),
    (-0.5, 0.5),
    (-2.0, 2.0),
]


def smoothstep(t):
    """C1 Hermite smoothstep: 3t^2 - 2t^3."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        """计算状态在每个维度 bin 内的相对位置 t∈[0,1]。"""
        fractions = []
        for val, (lo, hi) in zip(current_state, CLIP_RANGES):
            val = np.clip(val, lo, hi)
            bin_width = (hi - lo) / MAX_DIGITIZE_NUM
            bin_idx = int((val - lo) / bin_width)
            bin_idx = min(bin_idx, MAX_DIGITIZE_NUM - 1)
            bin_lo = lo + bin_idx * bin_width
            frac = (val - bin_lo) / bin_width
            frac = np.clip(frac, 0.0, 1.0)
            fractions.append((bin_idx, frac))
        return fractions

    def _smooth_q_values(self, current_state):
        """4D smoothstep (C1) 插值。"""
        fractions = self._get_bin_fractions(current_state)

        # 16 个相邻 bin 的 smoothstep 加权
        q_vals = np.zeros(self.n_action)
        for offset_bits in range(16):
            bin_indices = []
            w = 1.0
            for dim in range(4):
                bin_idx, frac = fractions[dim]
                if (offset_bits >> dim) & 1:
                    neighbor_idx = min(bin_idx + 1, MAX_DIGITIZE_NUM - 1)
                    w *= smoothstep(frac)  # C1 smoothstep
                else:
                    neighbor_idx = bin_idx
                    w *= (1.0 - smoothstep(frac))
                bin_indices.append(neighbor_idx)

            state_id = (bin_indices[0] * MAX_DIGITIZE_NUM ** 0
                      + bin_indices[1] * MAX_DIGITIZE_NUM ** 1
                      + bin_indices[2] * MAX_DIGITIZE_NUM ** 2
                      + bin_indices[3] * MAX_DIGITIZE_NUM ** 3)
            q_vals += w * self.q_table[state_id, :]

        return q_vals

    def update_q_table(self, current_state, action, reward, next_state):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)
        td_target = reward + self.gamma * np.max(self.q_table[next_id, :])
        td_error = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += self.lr * td_error

    def decide_action(self, current_state):
        current_id = self.make_bins_state(current_state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            smooth_q = self._smooth_q_values(current_state)
            action = np.argmax(smooth_q)
        return int(action)

    def predict(self, state):
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
        print(f"Q-learning Bonus3-Smooth v2 模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning Bonus3-Smooth v2 模型已加载: {filepath}")


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
    print("开始训练 Q-learning Bonus3-Smooth v2 (smoothstep C1) 模型...")
    record = env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_bonus3_smooth_v2_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Bonus3-Smooth v2 (smoothstep C1) Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus3_smooth_v2_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()
