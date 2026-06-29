"""
MCTS 学生填空版。

目标:
    通过四个 TODO 实现 UCT 版 MCTS,在 CartPole-v1 上做在线规划并保持平衡。
    MCTS 不需要 checkpoint,可以直接跑本文件看分数,或用 vis_mcts.py 录视频:

    python train_mcts.py                         # 跑几轮看分数
    python vis_mcts.py --episodes 1              # 录一段视频(默认指向 backup,
                                                 # 学生可改成 train_mcts 验证)

固定基座(不需要修改):
    MCTSNode               —— 树节点结构
    Agent.__init__ / act   —— 入口和状态管理
    _tree_policy           —— 外层搜索控制
    _expand                —— 节点扩展
    Environment            —— 训练/演示壳子
"""

import itertools
import math
import random

import gymnasium as gym
import numpy as np


# =============================================================================
# 超参数(学生可以调)
# =============================================================================
ITERATION_BUDGET = 100
C_P_INIT = 200
EPOCHS = 5
LOOKAHEAD_TARGET = 200


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth):
        self.params = params
        self.children = {}
        self.parent = None
        self.Q = 0
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = done
        self.depth = depth
        self.action = None


class Agent:
    def __init__(self, iteration_budget, env_id):
        self.env_id = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions = None

    # ----- 固定基座:外层入口 ----------------------------------------------
    def act(self, state, n_actions, node=None, C_p=C_P_INIT, lookahead_target=LOOKAHEAD_TARGET):
        self.n_actions = n_actions
        return self._uct_search(state, n_actions, node=node, C_p=C_p,
                                lookahead_target=lookahead_target)

    def _uct_search(self, state, n_actions, node=None, C_p=C_P_INIT, lookahead_target=LOOKAHEAD_TARGET):
        root_node = node if node is not None else MCTSNode(state, False, 0)
        # 复用 best_child 做新根时切断父链,避免老树挂着 + backward 越界。
        root_node.parent = None
        max_depth = 0

        for _ in range(self.iteration_budget):
            c_node = self._tree_policy(root_node, n_actions, C_p)
            max_depth = max(c_node.depth - root_node.depth, max_depth)
            reward = self._default_policy(c_node)
            self._backward(c_node, reward, root_node)

        # =====================================================================
        # TODO 4: C_p 自适应调整
        # 提示:
        #   C_p 是 UCT 的探索常数。期望它让搜索平均能展开到 lookahead_target
        #   深度左右:
        #     * 如果本轮最大深度 max_depth 还没到 lookahead_target —— 说明
        #       搜索"发散得太广、不够深",应该让 C_p 减小(减少探索)。
        #     * 反之,如果搜索深度已经够了,应该稍微加大 C_p(鼓励探索)。
        #   一个简单的规则就是每步 ±1。
        # =====================================================================
        # ------ 你的代码开始 ------
        pass  # 占位符:不调整 C_p,搜索深度难以自校准
        # ------ 你的代码结束 ------

        best_child_node = max(root_node.children.values(), key=lambda x: x.N)
        return best_child_node.action, best_child_node, C_p

    # ----- 固定基座:selection -> expansion 的外层循环 --------------------
    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
        exp_env = gym.make(self.env_id)
        exp_env.reset()
        exp_env.unwrapped.state = np.array(node.params)

        unchosen_actions = [a for a in range(n_actions) if a not in node.children]
        a = random.choice(unchosen_actions)
        params, _, terminated, truncated, _ = exp_env.step(a)
        done = terminated or truncated
        child_node = MCTSNode(params, done, node.depth + 1)
        child_node.parent = node
        child_node.action = a
        node.children[a] = child_node
        exp_env.close()
        return child_node

    # =========================================================================
    # TODO 1: UCT 打分公式(bestchild 的核心)
    # 提示:
    #   UCT 分数 = Q/N + C_p * sqrt( 2 * ln(N_parent) / N_child )
    #     * 第一项 exploitation:子节点平均回报
    #     * 第二项 exploration:节点被访问得少 -> 鼓励多探索
    #   需要:child.Q, child.N, node.N, C_p
    #   返回分数最高的 child 节点(可用 max(..., key=lambda c: uct_score(c)))。
    # 注意:node.children 是 {action: child_node} 字典。
    # =========================================================================
    def _bestchild(self, node, C_p):
        # ------ 你的代码开始 ------
        # 占位符:随便挑第一个子节点,丢失 UCT 的平衡性
        return next(iter(node.children.values()))
        # ------ 你的代码结束 ------

    # =========================================================================
    # TODO 2: 默认策略 rollout
    # 提示:
    #   从 node 对应的状态出发,**随机**走到 episode 终止,累积 reward。
    #   常用骨架:
    #       new_env = gym.make(self.env_id); new_env.reset()
    #       new_env.unwrapped.state = np.array(node.params)
    #       done = node.done
    #       reward = node.depth    # 原作者设计:用深度作初值鼓励深搜索
    #       while not done:
    #           a = random.randrange(self.n_actions)
    #           _, step_reward, terminated, truncated, _ = new_env.step(a)
    #           done = terminated or truncated
    #           reward += step_reward
    #       return reward
    # =========================================================================
    def _default_policy(self, node):
        # ------ 你的代码开始 ------
        # 占位符:直接返回节点深度,根本没做 rollout —— 树根完全得不到分数差异
        return node.depth
        # ------ 你的代码结束 ------

    # =========================================================================
    # TODO 3: 回溯更新(backward)
    # 提示:
    #   从 node(叶子)向上爬到新根,路径上每个节点:
    #       node.N += 1
    #       node.Q += reward
    #       node = node.parent
    #   停在 root_node.parent(= None,因为 _uct_search 切过 parent 了)。
    #   注意 while 条件用 `node is not stop`,到达 stop 就退出,不把 stop
    #   自身再更新。
    # =========================================================================
    def _backward(self, node, reward, root_node):
        stop = root_node.parent
        # ------ 你的代码开始 ------
        pass  # 占位符:不更新任何 N/Q,搜索树永远是噪声
        # ------ 你的代码结束 ------


class Environment:
    def __init__(self, env_id="CartPole-v1", iteration_budget=ITERATION_BUDGET,
                 C_p=C_P_INIT, epochs=EPOCHS, lookahead_target=LOOKAHEAD_TARGET):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.agent = Agent(iteration_budget, env_id)
        self.epochs = epochs
        self.C_p = C_p
        self.lookahead_target = lookahead_target
        self.n_action = self.env.action_space.n

    def train(self):
        record = []
        for i in range(self.epochs):
            self.env.reset()
            sum_reward = 0
            node = None
            C_p = self.C_p
            while True:
                action, node, C_p = self.agent.act(
                    self.env.unwrapped.state,
                    n_actions=self.n_action,
                    node=node,
                    C_p=C_p,
                    lookahead_target=self.lookahead_target,
                )
                _, reward, terminated, truncated, _ = self.env.step(action)
                sum_reward += reward
                if terminated or truncated:
                    break
            record.append(sum_reward)
            print(f"Epoch {i}: Score = {sum_reward}")
        self.env.close()
        return np.mean(record)


if __name__ == "__main__":
    exp_env = Environment()
    final = exp_env.train()
    print(f"Average score: {final}")
