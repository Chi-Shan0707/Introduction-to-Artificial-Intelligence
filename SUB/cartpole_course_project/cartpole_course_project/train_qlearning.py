"""
Q-Learning 学生填空版。

目标:
    通过三个 TODO 实现离散化 Q-learning,能在 CartPole-v1 上持续平衡。
    完成后用 evaluate.py / vis.py 指向本文件的 Agent 类即可评估:

    python evaluate.py \\
        --agent-class train_qlearning:Agent \\
        --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' \\
        --checkpoint checkpoints/base/q_learning_model.pkl \\
        --seed-base 42 --seed-count 100

固定基座(不需要修改):
    make_bins_state / bins        —— 状态离散化
    save_model / load_model       —— checkpoint 序列化
    predict / policy_info         —— evaluate / vis 接口
    Environment.__init__          —— env 和 agent 构造
"""

import os
import pickle
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 超参数(学生可以调,但建议先用默认值跑通 TODO,再回过头来调)
# =============================================================================
LR = 0.04
GAMMA = 0.99
EPSILON = 0.1
EPOCHS = 2000
TARGET = 1000            # 训练内层 TimeLimit,越大信号越明显
MAX_DIGITIZE_NUM = 6     # 每维离散 bin 数(固定)
SAVE_DIR = 'checkpoints/base'


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, epsilon):
        self.n_action = n_action
        self.n_state = n_state
        self.q_table = np.random.uniform(low=0, high=1,
                                         size=(MAX_DIGITIZE_NUM ** n_state, n_action))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    # ----- 固定基座:状态离散化 -----------------------------------------------
    def bins(self, clip_min, clip_max):
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def make_bins_state(self, current_state):
        """把 4 维连续观测映射到 0..6^4-1 的离散 state_id。"""
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

    # =========================================================================
    # TODO 1: Q-table 更新(Bellman TD)
    # 提示:
    #   标准 Q-learning 更新公式
    #       Q(s,a) <- Q(s,a) + lr * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
    #   需要的变量:
    #     current_id  = self.make_bins_state(current_state)
    #     next_id     = self.make_bins_state(next_state)
    #     self.q_table[current_id, action] 是当前 Q(s,a)
    #     self.q_table[next_id, :]         是下一状态各动作 Q 值数组
    # =========================================================================
    def update_q_table(self, current_state, action, reward, next_state):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)

        # ------ 你的代码开始 ------
        # 占位符:不更新 —— 训练曲线会一直趴在低分上
        current_q = self.q_table[current_id, action]
        next_q_max = np.max(self.q_table[next_id, :])
        td_target = reward + self.gamma * next_q_max
        self.q_table[current_id, action] = current_q + self.lr * (td_target - current_q)
        # ------ 你的代码结束 ------

    # =========================================================================
    # TODO 2: epsilon-greedy 动作选择
    # 提示:
    #   以 epsilon 的概率随机选动作,否则按 argmax Q(s,:) 贪心选。
    #   可用:
    #     np.random.uniform(0, 1)    —— 抽一个 [0,1) 均匀随机数
    #     np.random.choice(self.n_action)  —— 均匀随机动作
    #     np.argmax(self.q_table[id, :])   —— 贪心动作
    # =========================================================================
    def decide_action(self, current_state):
        current_id = self.make_bins_state(current_state)

        # ------ 你的代码开始 ------
        # 占位符:永远选动作 0,等价于完全不平衡的随机策略
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            action = np.argmax(self.q_table[current_id, :])
        # ------ 你的代码结束 ------

        return int(action)

    # ----- 固定基座:evaluate / vis 的统一接口 -------------------------------
    def predict(self, state):
        """贪心动作,用于 evaluate.py / vis.py(不做随机探索)。"""
        current_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[current_id, :]))

    def policy_info(self, state):
        """vis.py 右上角叠加显示 Q 值。"""
        current_id = self.make_bins_state(state)
        q_vals = self.q_table[current_id, :]
        return {"Q(LEFT)": f"{q_vals[0]:+.3f}", "Q(RIGHT)": f"{q_vals[1]:+.3f}"}

    # ----- 固定基座:checkpoint --------------------------------------------
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon,
                'n_state': self.n_state, 'n_action': self.n_action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, f)
        print(f"Q-learning 模型已保存到: {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.lr, self.gamma, self.epsilon = data['lr'], data['gamma'], data['epsilon']
        self.n_state, self.n_action = data['n_state'], data['n_action']
        print(f"Q-learning 模型已加载: {filepath} (训练时间 {data['timestamp']})")


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

                # =============================================================
                # TODO 3: reward shaping
                # 提示:
                #   CartPole 的原始 reward 就是 +1/step(只要杆子没倒),这个
                #   信号稀疏且方差大。常见做法:
                #     * 只在 episode 结束时给一次总结性 reward
                #     * 结束时 reward = step - target:
                #         - 撑满 target 步 -> reward = 0
                #         - 只撑 100 步   -> reward = -900
                #       即"离目标越远惩罚越大",让 agent 把"延长存活"当目标
                #   中间步 (terminated/truncated 都为 False) 可直接给 0。
                #   下面在 "occur_end" 分支里给出 terminal reward 即可。
                # =============================================================
                if terminated or truncated:
                    # ------ 你的代码开始 ------
                    reward = (step + 1) - TARGET
                    # ------ 你的代码结束 ------
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
    print("开始训练 Q-learning 模型...")
    record = env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'q_learning_model.pkl'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('Q-Learning Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plot_path = os.path.join(SAVE_DIR, 'q_learning_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均步数: {np.mean(record[-20:]):.2f}")


if __name__ == "__main__":
    main()
