"""Bonus 2: Optimized Q-Learning for CartPole — targeting 100-seed mean >= 500.

Key optimizations vs baseline:
1. Finer discretization: 10 bins per dimension (vs 6) → 10^4 = 10000 states
2. Epsilon decay: starts at 1.0, decays to 0.01 slowly
3. Higher training epochs: 10000
4. Larger TARGET: 2000 (match evaluation TimeLimit)
5. Best-so-far checkpoint saving
6. Dense intermediate reward: +1 per survival step
"""

import os
import pickle
from datetime import datetime
from collections import deque

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Optimized hyperparameters
LR = 0.2
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99975
EPOCHS = 15000
TARGET = 2000
MAX_DIGITIZE_NUM = 12  # 12 bins → 20736 states
SAVE_DIR = 'checkpoints'


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, epsilon):
        self.n_action = n_action
        self.n_state = n_state
        self.q_table = np.zeros((MAX_DIGITIZE_NUM ** n_state, n_action))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def bins(self, clip_min, clip_max):
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def make_bins_state(self, current_state):
        cart_pos, cart_v, pole_angle, pole_v = current_state
        cart_pos = np.clip(cart_pos, -2.4, 2.4)
        cart_v = np.clip(cart_v, -2., 2.)
        pole_angle = np.clip(pole_angle, -0.3, 0.3)
        pole_v = np.clip(pole_v, -1.5, 1.5)
        s1 = np.argwhere((cart_pos - self.bins(-2.4001, 2.4)) > 0)[-1, 0]
        s2 = np.argwhere((cart_v - self.bins(-2.001, 2.)) > 0)[-1, 0]
        s3 = np.argwhere((pole_angle - self.bins(-0.3001, 0.3)) > 0)[-1, 0]
        s4 = np.argwhere((pole_v - self.bins(-1.5001, 1.5)) > 0)[-1, 0]
        return (s1 * MAX_DIGITIZE_NUM ** 0
                + s2 * MAX_DIGITIZE_NUM ** 1
                + s3 * MAX_DIGITIZE_NUM ** 2
                + s4 * MAX_DIGITIZE_NUM ** 3)

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
            action = np.argmax(self.q_table[current_id, :])
        return int(action)

    def predict(self, state):
        current_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[current_id, :]))

    def policy_info(self, state):
        current_id = self.make_bins_state(state)
        q_vals = self.q_table[current_id, :]
        return {"Q(LEFT)": f"{q_vals[0]:+.3f}", "Q(RIGHT)": f"{q_vals[1]:+.3f}"}

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon,
                'n_state': self.n_state, 'n_action': self.n_action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, f)
        print(f"Q-learning Bonus2 模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning Bonus2 模型已加载: {filepath}")


class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1', max_episode_steps=TARGET)
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        self.agent = Agent(n_state, n_action, LR, GAMMA, EPSILON_START)

    def train(self, epoch=EPOCHS, save_path=None):
        step_record = []
        scores_deque = deque(maxlen=100)
        best_avg = float('-inf')
        epsilon = EPSILON_START

        for i in range(epoch):
            self.agent.epsilon = epsilon
            current_state = self.env.reset()[0]

            for step in range(TARGET):
                action = self.agent.decide_action(current_state)
                next_state, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    reward = -100  # big penalty for failing
                else:
                    reward = 1  # +1 for each survival step

                self.agent.update_q_table(current_state, action, reward, next_state)
                current_state = next_state

                if terminated or truncated:
                    step_record.append(step)
                    scores_deque.append(step)
                    if (i + 1) % 100 == 0:
                        print(f'{i} Episode: Finished after {step + 1} time steps, epsilon={epsilon:.4f}')
                    break

            # Epsilon decay
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            # Best-so-far save
            if save_path is not None and len(scores_deque) == scores_deque.maxlen:
                current_avg = float(np.mean(scores_deque))
                if current_avg > best_avg:
                    best_avg = current_avg
                    self.agent.epsilon = 0.0  # save with greedy epsilon
                    self.agent.save_model(save_path)
                    self.agent.epsilon = epsilon

        self.env.close()
        if save_path is not None and best_avg == float('-inf'):
            self.agent.save_model(save_path)
        return step_record


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = Environment()
    print("开始训练 Q-learning Bonus2 模型...")
    record = env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_bonus2_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Bonus2 Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus2_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()
