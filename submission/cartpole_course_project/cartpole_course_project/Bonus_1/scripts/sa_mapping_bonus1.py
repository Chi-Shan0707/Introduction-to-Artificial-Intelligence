from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ANGLE_LIMIT = 12 * 2 * np.pi / 360


def build_q_agent(project_dir: Path):
    class QAgent:
        def __init__(self, q_table):
            self.q_table = q_table

        def bins(self, clip_min, clip_max):
            return np.linspace(clip_min, clip_max, 7)

        def make_bins_state(self, current_state):
            cart_pos, cart_v, pole_angle, pole_v = current_state
            cart_pos = np.clip(cart_pos, -2.4, 2.4)
            cart_v = np.clip(cart_v, -3.0, 3.0)
            pole_angle = np.clip(pole_angle, -0.5, 0.5)
            pole_v = np.clip(pole_v, -2.0, 2.0)
            s1 = np.argwhere((cart_pos - self.bins(-2.4001, 2.4)) > 0)[-1, 0]
            s2 = np.argwhere((cart_v - self.bins(-3.001, 3.0)) > 0)[-1, 0]
            s3 = np.argwhere((pole_angle - self.bins(-0.5001, 0.5)) > 0)[-1, 0]
            s4 = np.argwhere((pole_v - self.bins(-2.001, 2.0)) > 0)[-1, 0]
            return s1 + s2 * 6 + s3 * 36 + s4 * 216

        def predict(self, state):
            current_id = self.make_bins_state(state)
            return int(np.argmax(self.q_table[current_id, :]))

    with open(project_dir / "checkpoints" / "q_learning_model.pkl", "rb") as f:
        data = pickle.load(f)
    return QAgent(data["q_table"])


def build_r_agent(project_dir: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

        def predict(self, state):
            state = torch.from_numpy(np.asarray(state, dtype=np.float32)).float().unsqueeze(0)
            with torch.no_grad():
                probs = self.forward(state)
            return int(probs.argmax(dim=1).item())

    agent = RAgent()
    agent.load_state_dict(torch.load(project_dir / "checkpoints" / "reinforce_model.pt", map_location="cpu"))
    agent.eval()
    return agent


def build_mcts_agent():
    from train_mcts import Agent as MCTSAgent

    return MCTSAgent(iteration_budget=80, env_id="CartPole-v1")


def grid_actions_2d(agent, x_idx, y_idx, x_vals, y_vals):
    out = np.zeros((len(y_vals), len(x_vals)), dtype=np.int32)
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            s = np.zeros(4, dtype=np.float32)
            s[x_idx] = xv
            s[y_idx] = yv
            out[i, j] = int(agent.predict(s))
    return out


def collect_mcts_trajectory(seed: int = 42, max_steps: int = 2000):
    import gymnasium as gym

    agent = build_mcts_agent()
    env = gym.make("CartPole-v1", max_episode_steps=max_steps)
    state, _ = env.reset(seed=seed)
    node = None
    c_p = 200
    lookahead_target = 200

    states = []
    actions = []
    while True:
        action, node, c_p = agent.act(
            env.unwrapped.state,
            n_actions=env.action_space.n,
            node=node,
            C_p=c_p,
            lookahead_target=lookahead_target,
        )
        states.append(np.array(state, dtype=np.float32))
        actions.append(int(action))
        state, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    env.close()
    return np.array(states), np.array(actions, dtype=np.int32)


def padded_quantile_range(
    arr: np.ndarray,
    q_low: float = 0.02,
    q_high: float = 0.98,
    pad_ratio: float = 0.18,
    hard_min: float | None = None,
    hard_max: float | None = None,
):
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    if hi <= lo:
        hi = lo + 1e-6
    span = hi - lo
    lo -= pad_ratio * span
    hi += pad_ratio * span
    if hard_min is not None:
        lo = max(lo, hard_min)
    if hard_max is not None:
        hi = min(hi, hard_max)
    return lo, hi


def plot_mcts_scatter(
    ax,
    states,
    actions,
    x_idx,
    y_idx,
    title,
    xlabel,
    ylabel,
    xlim,
    ylim,
):
    left = actions == 0
    right = actions == 1
    # Softer colors
    ax.scatter(states[left, x_idx], states[left, y_idx], s=16, alpha=0.72, c="#87c5da", label="LEFT(0)")
    ax.scatter(states[right, x_idx], states[right, y_idx], s=16, alpha=0.72, c="#e8a0a0", label="RIGHT(1)")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)


def plot_pair(ax, data, extent, title, xlabel, ylabel):
    # Softer binary palette
    cmap = ListedColormap(["#a8d8ea", "#f6b8b8"]).copy()
    cmap.set_bad(color="#ececec")
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def build_projection_specs():
    return [
        {
            "x_idx": 0,
            "y_idx": 2,
            "x_vals": np.linspace(-2.4, 2.4, 121),
            "y_vals": np.linspace(-ANGLE_LIMIT, ANGLE_LIMIT, 121),
            "xlabel": "cart_pos",
            "ylabel": "pole_angle",
            "name": "Position vs Angle",
            "x_hard": (-2.4, 2.4),
            "y_hard": (-ANGLE_LIMIT, ANGLE_LIMIT),
        },
        {
            "x_idx": 1,
            "y_idx": 3,
            "x_vals": np.linspace(-3.0, 3.0, 121),
            "y_vals": np.linspace(-2.5, 2.5, 121),
            "xlabel": "cart_vel",
            "ylabel": "pole_angular_velocity",
            "name": "Velocity vs Angular Velocity",
            "x_hard": (-3.0, 3.0),
            "y_hard": (-2.5, 2.5),
        },
        {
            "x_idx": 0,
            "y_idx": 1,
            "x_vals": np.linspace(-2.4, 2.4, 121),
            "y_vals": np.linspace(-3.0, 3.0, 121),
            "xlabel": "cart_pos",
            "ylabel": "cart_vel",
            "name": "Position vs Velocity",
            "x_hard": (-2.4, 2.4),
            "y_hard": (-3.0, 3.0),
        },
        {
            "x_idx": 2,
            "y_idx": 3,
            "x_vals": np.linspace(-ANGLE_LIMIT, ANGLE_LIMIT, 121),
            "y_vals": np.linspace(-2.5, 2.5, 121),
            "xlabel": "pole_angle",
            "ylabel": "pole_angular_velocity",
            "name": "Angle vs Angular Velocity",
            "x_hard": (-ANGLE_LIMIT, ANGLE_LIMIT),
            "y_hard": (-2.5, 2.5),
        },
    ]


def plot_policy_4in1(agent, specs, algo_name, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, spec in zip(axes.ravel(), specs):
        data = grid_actions_2d(
            agent,
            x_idx=spec["x_idx"],
            y_idx=spec["y_idx"],
            x_vals=spec["x_vals"],
            y_vals=spec["y_vals"],
        )
        plot_pair(
            ax,
            data,
            extent=[
                spec["x_vals"].min(),
                spec["x_vals"].max(),
                spec["y_vals"].min(),
                spec["y_vals"].max(),
            ],
            title=f"{algo_name}: {spec['name']}",
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
        )
    fig.suptitle(f"{algo_name} S-A Maps (4 Projections)", y=0.985)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def plot_mcts_4in1(states, actions, specs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    for ax, spec in zip(axes.ravel(), specs):
        x_arr = states[:, spec["x_idx"]]
        y_arr = states[:, spec["y_idx"]]
        xlim = padded_quantile_range(
            x_arr,
            hard_min=spec["x_hard"][0],
            hard_max=spec["x_hard"][1],
        )
        ylim = padded_quantile_range(
            y_arr,
            hard_min=spec["y_hard"][0],
            hard_max=spec["y_hard"][1],
        )
        plot_mcts_scatter(
            ax,
            states,
            actions,
            spec["x_idx"],
            spec["y_idx"],
            title=f"MCTS: {spec['name']}",
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            xlim=xlim,
            ylim=ylim,
        )
    handles, labels = axes.ravel()[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("MCTS S-A Maps (Trajectory, 4 Projections)", y=0.985)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_path, dpi=190, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Bonus 1 4-in-1 S-A maps.")
    parser.add_argument("--agent", choices=["qlearning", "reinforce", "mcts", "all"], default="all")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    results_dir = Path(__file__).resolve().parents[1] / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # import path for train_* modules
    import sys

    sys.path.insert(0, str(project_dir))

    specs = build_projection_specs()

    generated = []
    if args.agent in ("qlearning", "all"):
        q_agent = build_q_agent(project_dir)
        plot_policy_4in1(
            q_agent,
            specs=specs,
            algo_name="Q-learning",
            out_path=figures_dir / "bonus1_sa_map_qlearning_4in1.png",
        )
        generated.append("bonus1_sa_map_qlearning_4in1.png")
    if args.agent in ("reinforce", "all"):
        r_agent = build_r_agent(project_dir)
        plot_policy_4in1(
            r_agent,
            specs=specs,
            algo_name="REINFORCE",
            out_path=figures_dir / "bonus1_sa_map_reinforce_4in1.png",
        )
        generated.append("bonus1_sa_map_reinforce_4in1.png")
    if args.agent in ("mcts", "all"):
        # MCTS: trajectory scatter with projection-wise zoom
        states, actions = collect_mcts_trajectory(seed=42, max_steps=2000)
        plot_mcts_4in1(
            states,
            actions,
            specs=specs,
            out_path=figures_dir / "bonus1_sa_map_mcts_4in1.png",
        )
        generated.append("bonus1_sa_map_mcts_4in1.png")

    print("Done. Generated:")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
