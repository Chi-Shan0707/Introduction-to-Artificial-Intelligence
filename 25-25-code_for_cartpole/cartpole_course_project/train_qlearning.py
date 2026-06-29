"""
Q-Learning for CartPole-v1.

Baseline tabular Q-learning with state discretization.
- 4 discrete dimensions (cart_pos, cart_vel, pole_angle, pole_angvel) each with 6 bins
- Total states: 6^4 = 1296
- Epsilon-greedy exploration during training
"""

import os
import pickle
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
LR = 0.04
GAMMA = 0.99
EPSILON = 0.1
EPOCHS = 2000
TARGET = 1000
MAX_DIGITIZE_NUM = 6
SAVE_DIR = 'checkpoints'


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
        """Map 4D continuous observation to discrete state_id in 0..6^4-1."""
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

    def update_q_table(self, current_state, action, reward, next_state):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)
        current_q = self.q_table[current_id, action]
        next_q_max = np.max(self.q_table[next_id, :])
        td_target = reward + self.gamma * next_q_max
        self.q_table[current_id, action] = current_q + self.lr * (td_target - current_q)

    def decide_action(self, current_state):
        current_id = self.make_bins_state(current_state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            action = np.argmax(self.q_table[current_id, :])
        return int(action)

    def predict(self, state):
        """Greedy action for evaluation."""
        current_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[current_id, :]))

    def policy_info(self, state):
        """Display Q values for visualization."""
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
        print(f"Q-learning model saved to: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning model loaded: {filepath} (trained at {data['timestamp']})")


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
                    reward = (step + 1) - TARGET
                else:
                    reward = 0
                self.agent.update_q_table(current_state, action, reward, next_state)
                current_state = next_state
                if terminated or truncated:
                    if (i + 1) % 100 == 0:
                        print(f'{i} Episode: Finished after {step + 1} steps')
                    step_record.append(step)
                    break
        self.env.close()
        if save_path is not None:
            self.agent.save_model(save_path)
        return step_record


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    test_env = Environment()
    print("Training Q-learning model...")
    record = test_env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curve saved to: {plot_path}")
    print(f"Last 20 episodes avg steps: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()