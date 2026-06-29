"""
Robust Q-Learning with Inference-Time Smoothing (Bonus 3).

Key improvements over baseline:
1. Confidence-gated neighborhood smoothing during inference
2. Training-time observation noise (optional)
3. Adaptive bin discretization (optional)

Smoothing parameters (from PPT sensitivity analysis):
- radius=1: Single-axis neighbor extension
- alpha=0.9: Center state weight
- smooth_dims="angle_only": Only smooth pole angle dimension (dim 3)
- confidence_tau: Gate threshold for deciding when to smooth
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Default hyperparameters
LR = 0.04
GAMMA = 0.99
EPSILON = 0.1
EPOCHS = 2000
TARGET = 1000
MAX_DIGITIZE_NUM = 6
SAVE_DIR = "results"

LOW = np.array([-2.4, -3.0, -0.5, -2.0], dtype=np.float32)
HIGH = np.array([2.4, 3.0, 0.5, 2.0], dtype=np.float32)

# Default smoothing configuration (optimal from PPT experiments)
DEFAULT_SMOOTH_MODE = "confidence"
DEFAULT_SMOOTH_ALPHA = 0.9
DEFAULT_SMOOTH_RADIUS = 1
DEFAULT_SMOOTH_DIMS = "angle_only"
DEFAULT_CONFIDENCE_TAU = 0.8


def clip_state(state: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(state, dtype=np.float32), LOW, HIGH)


def parse_smooth_dims(smooth_dims: str) -> Tuple[int, ...]:
    """Map smoothing dimension name to state dimension indices."""
    name = str(smooth_dims).strip().lower()
    mapping = {
        "angle_angular": (2, 3),
        "angle_vel": (2, 3),
        "pole": (2, 3),
        "critical": (2, 3),
        "all_single": (0, 1, 2, 3),
        "all": (0, 1, 2, 3),
        "pos_only": (0,),
        "cart_pos_only": (0,),
        "velocity_only": (1,),
        "cart_velocity_only": (1,),
        "cart_vel_only": (1,),
        "angle_only": (2,),
        "angular_only": (3,),
        "pos_angle": (0, 2),
        "vel_angular": (1, 3),
    }
    if name not in mapping:
        raise ValueError(f"Unknown smooth_dims={smooth_dims!r}. Valid keys: {sorted(mapping)}")
    return mapping[name]


class Agent:
    def __init__(
        self,
        n_state: int,
        n_action: int,
        lr: float = LR,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
        train_obs_noise_std: float = 0.0,
        smooth_mode: str = DEFAULT_SMOOTH_MODE,
        smooth_alpha: float = DEFAULT_SMOOTH_ALPHA,
        smooth_radius: int = DEFAULT_SMOOTH_RADIUS,
        smooth_dims: str = DEFAULT_SMOOTH_DIMS,
        confidence_tau: float = DEFAULT_CONFIDENCE_TAU,
    ):
        self.n_state = int(n_state)
        self.n_action = int(n_action)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.train_obs_noise_std = float(train_obs_noise_std)
        self.smooth_mode = str(smooth_mode).lower()
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_radius = int(smooth_radius)
        self.smooth_dims = str(smooth_dims)
        self.confidence_tau = float(confidence_tau)

        self.q_table = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(MAX_DIGITIZE_NUM ** self.n_state, self.n_action),
        )

    def bins(self, clip_min: float, clip_max: float) -> np.ndarray:
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def state_to_digits(self, current_state: np.ndarray) -> Tuple[int, int, int, int]:
        current_state = clip_state(current_state)
        cart_pos, cart_v, pole_angle, pole_v = current_state

        s1 = np.argwhere((cart_pos - self.bins(-2.4001, 2.4)) > 0)[-1, 0]
        s2 = np.argwhere((cart_v - self.bins(-3.001, 3.0)) > 0)[-1, 0]
        s3 = np.argwhere((pole_angle - self.bins(-0.5001, 0.5)) > 0)[-1, 0]
        s4 = np.argwhere((pole_v - self.bins(-2.001, 2.0)) > 0)[-1, 0]
        return int(s1), int(s2), int(s3), int(s4)

    def digits_to_id(self, digits: Sequence[int]) -> int:
        base = MAX_DIGITIZE_NUM
        d0, d1, d2, d3 = [int(x) for x in digits]
        return d0 * (base ** 0) + d1 * (base ** 1) + d2 * (base ** 2) + d3 * (base ** 3)

    def id_to_digits(self, state_id: int) -> Tuple[int, int, int, int]:
        base = MAX_DIGITIZE_NUM
        state_id = int(state_id)
        return (
            state_id % base,
            (state_id // base) % base,
            (state_id // (base ** 2)) % base,
            (state_id // (base ** 3)) % base,
        )

    def make_bins_state(self, current_state: np.ndarray) -> int:
        return self.digits_to_id(self.state_to_digits(current_state))

    def observe_for_train(self, state: np.ndarray) -> np.ndarray:
        """Add observation noise during training (simulates imperfect sensing)."""
        state = clip_state(state)
        if self.train_obs_noise_std <= 0:
            return state
        noise = np.random.normal(
            loc=0.0,
            scale=self.train_obs_noise_std,
            size=state.shape,
        ).astype(np.float32)
        return clip_state(state + noise)

    def _one_axis_neighbor_ids(self, center_id: int) -> List[int]:
        """Conservative neighborhood: center point + single-axis extension.

        Example: angle_only + radius=1 gives at most 5 states.
        Avoids over-smoothing (3^4=81 neighbors) in higher dimensions.
        """
        radius = max(0, int(self.smooth_radius))
        center = list(self.id_to_digits(center_id))
        ids = {int(center_id)}
        if radius == 0:
            return [int(center_id)]

        dims = parse_smooth_dims(self.smooth_dims)
        for dim in dims:
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                neighbor = center.copy()
                neighbor[dim] = int(np.clip(neighbor[dim] + delta, 0, MAX_DIGITIZE_NUM - 1))
                ids.add(self.digits_to_id(neighbor))
        return sorted(ids)

    def _neighbor_mean_q(self, center_id: int) -> np.ndarray:
        ids = self._one_axis_neighbor_ids(center_id)
        neighbor_ids = [x for x in ids if x != int(center_id)]
        if len(neighbor_ids) == 0:
            return self.q_table[int(center_id), :].copy()
        return np.mean(self.q_table[neighbor_ids, :], axis=0)

    def _weighted_smoothed_q(self, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        center_id = self.make_bins_state(state)
        center_q = self.q_table[center_id, :].copy()
        neighbor_q = self._neighbor_mean_q(center_id)
        alpha = float(np.clip(self.smooth_alpha, 0.0, 1.0))
        q_smooth = alpha * center_q + (1.0 - alpha) * neighbor_q
        info = {
            "center_id": center_id,
            "center_q": center_q,
            "neighbor_q": neighbor_q,
            "margin": float(abs(center_q[0] - center_q[1])),
            "used_smoothing": True,
        }
        return q_smooth, info

    def _robust_q(self, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        center_id = self.make_bins_state(state)
        center_q = self.q_table[center_id, :].copy()
        margin = float(abs(center_q[0] - center_q[1]))

        mode = self.smooth_mode.lower()
        if mode in {"none", "off", "baseline"}:
            return center_q, {
                "center_id": center_id,
                "center_q": center_q,
                "margin": margin,
                "used_smoothing": False,
            }

        if mode in {"weighted", "smooth", "always"}:
            return self._weighted_smoothed_q(state)

        if mode in {"confidence", "gated", "confidence_gated"}:
            if margin > self.confidence_tau:
                return center_q, {
                    "center_id": center_id,
                    "center_q": center_q,
                    "margin": margin,
                    "used_smoothing": False,
                }
            return self._weighted_smoothed_q(state)

        raise ValueError(f"smooth_mode must be: none, weighted, confidence. Got {self.smooth_mode!r}")

    def update_q_table(self, current_state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        current_id = self.make_bins_state(current_state)
        next_id = self.make_bins_state(next_state)

        current_q = self.q_table[current_id, int(action)]
        next_q_max = np.max(self.q_table[next_id, :])
        td_target = float(reward) + self.gamma * next_q_max
        self.q_table[current_id, int(action)] = current_q + self.lr * (td_target - current_q)

    def decide_action(self, current_state: np.ndarray) -> int:
        current_id = self.make_bins_state(current_state)
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            return int(np.random.choice(self.n_action))
        return int(np.argmax(self.q_table[current_id, :]))

    def predict(self, state: np.ndarray) -> int:
        """Greedy action selection during evaluation."""
        q_vals, _ = self._robust_q(state)
        return int(np.argmax(q_vals))

    def policy_info(self, state: np.ndarray) -> dict:
        q_vals, info = self._robust_q(state)
        out = {
            "Quse(LEFT)": f"{q_vals[0]:+.3f}",
            "Quse(RIGHT)": f"{q_vals[1]:+.3f}",
            "margin": f"{info['margin']:.3f}",
            "smooth": str(bool(info["used_smoothing"])),
        }
        return out

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "n_state": self.n_state,
                    "n_action": self.n_action,
                    "train_obs_noise_std": self.train_obs_noise_std,
                    "smooth_mode": self.smooth_mode,
                    "smooth_alpha": self.smooth_alpha,
                    "smooth_radius": self.smooth_radius,
                    "smooth_dims": self.smooth_dims,
                    "confidence_tau": self.confidence_tau,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )
        print(f"Model saved: {filepath}")

    def load_model(self, filepath: str):
        smooth_mode = self.smooth_mode
        smooth_alpha = self.smooth_alpha
        smooth_radius = self.smooth_radius
        smooth_dims = self.smooth_dims
        confidence_tau = self.confidence_tau

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = data["q_table"]
        self.lr = float(data.get("lr", self.lr))
        self.gamma = float(data.get("gamma", self.gamma))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.n_state = int(data.get("n_state", self.n_state))
        self.n_action = int(data.get("n_action", self.n_action))
        self.train_obs_noise_std = float(data.get("train_obs_noise_std", self.train_obs_noise_std))

        self.smooth_mode = smooth_mode
        self.smooth_alpha = smooth_alpha
        self.smooth_radius = smooth_radius
        self.smooth_dims = smooth_dims
        self.confidence_tau = confidence_tau

        print(f"Model loaded: {filepath}")


def train(train_obs_noise_std: float, out_ckpt: Path, out_curve: Path, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)

    env = gym.make("CartPole-v1", max_episode_steps=TARGET)
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    agent = Agent(
        n_state=n_state,
        n_action=n_action,
        lr=LR,
        gamma=GAMMA,
        epsilon=EPSILON,
        train_obs_noise_std=train_obs_noise_std,
        smooth_mode="none",
    )

    record: List[int] = []
    for i in range(EPOCHS):
        reset_kwargs = {"seed": seed + i} if seed is not None else {}
        real_state, _ = env.reset(**reset_kwargs)

        obs_state = agent.observe_for_train(real_state)

        for step in range(TARGET):
            action = agent.decide_action(obs_state)
            next_real_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = (step + 1) - TARGET if done else 0

            next_obs_state = agent.observe_for_train(next_real_state)
            agent.update_q_table(obs_state, action, reward, next_obs_state)

            obs_state = next_obs_state
            if done:
                if (i + 1) % 100 == 0:
                    print(f"{i} Episode: Finished after {step + 1} steps")
                record.append(step + 1)
                break

    env.close()
    agent.save_model(str(out_ckpt))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title(f"Q-learning Training Curve (train_obs_noise_std={train_obs_noise_std})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.savefig(out_curve)
    plt.close()

    print(f"Curve saved: {out_curve}")
    if len(record) > 0:
        print(f"Last 20 avg steps: {np.mean(record[-20:]):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train robust Q-learning agent for Bonus 3")
    parser.add_argument("--train-obs-noise-std", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=str, default="q_robust_base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        out_dir = Path(__file__).resolve().parent / SAVE_DIR
    else:
        out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = out_dir / f"{args.output_prefix}_model.pkl"
    curve = out_dir / f"{args.output_prefix}_training_curve.png"
    train(args.train_obs_noise_std, ckpt, curve, seed=args.seed)


if __name__ == "__main__":
    main()