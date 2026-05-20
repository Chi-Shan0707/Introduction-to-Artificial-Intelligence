"""Evaluate wrapper for Bonus 3 Hedge v2 (frozen tables).

Usage:
    python evaluate.py \
        --agent-class train_qlearning_bonus3_hedge_v2_eval:Agent \
        --agent-init-kwargs '{"n_state":4,"n_action":2}' \
        --checkpoint checkpoints/q_learning_bonus3_hedge_v2_agent.pkl \
        --seed-base 42 --seed-count 100 --perturb-scale 0.05
"""

import pickle
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_qlearning_bonus3_hedge_v2 import QTable, HedgeAgent


class Agent:
    """evaluate.py / vis.py 兼容接口。"""

    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.hedge_agent = None

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 重建 frozen Q-tables
        tables = []
        for offset, q_table in data['tables']:
            t = QTable(offset, self.n_action)
            t.q_table = q_table
            tables.append(t)

        self.hedge_agent = HedgeAgent(tables, self.n_action, eta=0.0)
        # 加载保存的权重（evaluate 时固定，不更新）
        if 'hedge_weights' in data:
            self.hedge_agent.hedge_weights = np.array(data['hedge_weights'])
        print(f"Hedge v2 agent loaded: {filepath}")
        print(f"  Weights: {self.hedge_agent.hedge_weights}")

    def predict(self, state):
        return self.hedge_agent.predict(state)

    def policy_info(self, state):
        return self.hedge_agent.policy_info(state)
