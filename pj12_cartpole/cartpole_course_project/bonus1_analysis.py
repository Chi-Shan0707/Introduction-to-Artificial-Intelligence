"""Bonus 1: 三种算法对比分析 —— 生成 boxplot + JSON 报告"""

import json
import random

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用 MCTS Agent
from train_mcts import Agent as MCTSAgent

def eval_mcts(n_seeds=100, seed_base=42, iteration_budget=100, max_steps=2000):
    results = []
    for i in range(n_seeds):
        seed = seed_base + i
        random.seed(seed)
        np.random.seed(seed)
        env = gym.make("CartPole-v1", max_episode_steps=max_steps)
        state, _ = env.reset(seed=seed)
        agent = MCTSAgent(iteration_budget, "CartPole-v1")
        total_reward = 0.0
        node = None
        C_p = 200
        while True:
            action, node, C_p = agent.act(
                env.unwrapped.state,
                n_actions=env.action_space.n,
                node=node,
                C_p=C_p,
                lookahead_target=200,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()
        results.append({"seed": seed, "steps": int(total_reward), "reward": float(total_reward)})
        if (i + 1) % 10 == 0:
            print(f"MCTS eval: {i+1}/{n_seeds} done")
    return results


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    # Load Q-learning and REINFORCE reports
    with open(os.path.join(base, "checkpoints", "q_learning_report.json")) as f:
        q_report = json.load(f)
    with open(os.path.join(base, "checkpoints", "reinforce_report.json")) as f:
        r_report = json.load(f)

    # Run MCTS evaluation
    print("Running MCTS evaluation (this takes a while)...")
    mcts_results = eval_mcts(n_seeds=100)  # 100 seeds, unified with Q/REINFORCE
    mcts_steps = [r["steps"] for r in mcts_results]

    mcts_report = {
        "agent_class": "train_mcts:Agent",
        "per_episode": mcts_results,
        "summary": {
            "n_episodes": len(mcts_results),
            "max_episode_steps": 2000,
            "steps": {
                "mean": float(np.mean(mcts_steps)),
                "median": float(np.median(mcts_steps)),
                "std": float(np.std(mcts_steps)),
                "min": float(np.min(mcts_steps)),
                "max": float(np.max(mcts_steps)),
            },
        },
    }
    with open(os.path.join(base, "checkpoints", "mcts_report.json"), "w") as f:
        json.dump(mcts_report, f, indent=2, ensure_ascii=False)

    # Extract steps
    q_steps = [e["steps"] for e in q_report["per_episode"]]
    r_steps = [e["steps"] for e in r_report["per_episode"]]
    m_steps = mcts_steps

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [q_steps, r_steps, m_steps],
        tick_labels=["Q-Learning", "REINFORCE", "MCTS"],
        patch_artist=True,
    )
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Steps")
    ax.set_title("CartPole Algorithm Comparison (100 seeds each)")
    ax.grid(axis="y", alpha=0.3)

    # 添加定量信息
    info_lines = [
        f"Q-Learning:  mean={np.mean(q_steps):.1f}, std={np.std(q_steps):.1f}",
        f"REINFORCE:   mean={np.mean(r_steps):.1f}, std={np.std(r_steps):.1f}",
        f"MCTS:        mean={np.mean(m_steps):.1f}, std={np.std(m_steps):.1f}",
    ]
    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plot_path = os.path.join(base, "checkpoints", "bonus1_boxplot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to: {plot_path}")

    # Print analysis summary
    print("\n=== Bonus 1 对比分析 ===")
    print(f"Q-Learning:  mean={np.mean(q_steps):.1f}, median={np.median(q_steps):.0f}, std={np.std(q_steps):.1f}")
    print(f"REINFORCE:   mean={np.mean(r_steps):.1f}, median={np.median(r_steps):.0f}, std={np.std(r_steps):.1f}")
    print(f"MCTS:        mean={np.mean(m_steps):.1f}, median={np.median(m_steps):.0f}, std={np.std(m_steps):.1f}")


if __name__ == "__main__":
    main()
