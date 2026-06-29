"""
Bonus 3 diagnostic: compare terminal states for smoothing variants.

The regular evaluate.py reports only steps/reward. This script records the
terminal state of each episode, so we can see whether failures are mostly
caused by cart-position boundary, pole-angle boundary, or time limit.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


LIMITS = {
    "cart_pos": 2.4,
    # Gymnasium CartPole-v1 terminates at +/- 12 degrees.
    "pole_angle": 12 * 2 * np.pi / 360,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_agent(agent_class: str, kwargs: dict, checkpoint: Path):
    module_name, class_name = agent_class.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    agent = cls(**kwargs)
    agent.load_model(str(checkpoint))
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def classify_terminal(state: np.ndarray, truncated: bool) -> str:
    if truncated:
        return "time_limit"
    x = float(state[0])
    theta = float(state[2])
    over_x = abs(x) > LIMITS["cart_pos"]
    over_theta = abs(theta) > LIMITS["pole_angle"]
    if over_x and over_theta:
        return "cart_pos_and_pole_angle"
    if over_x:
        return "cart_pos"
    if over_theta:
        return "pole_angle"
    return "other_terminated"


def make_agent_kwargs(dims: str, alpha: float, tau: float, radius: int) -> dict:
    return {
        "n_state": 4,
        "n_action": 2,
        "lr": 0.04,
        "gamma": 0.99,
        "epsilon": 0.0,
        "train_obs_noise_std": 0.0,
        "smooth_mode": "confidence",
        "smooth_alpha": alpha,
        "smooth_radius": radius,
        "smooth_dims": dims,
        "confidence_tau": tau,
    }


def run_variant(
    *,
    name: str,
    agent_class: str,
    kwargs: dict,
    checkpoint: Path,
    seed_base: int,
    seed_count: int,
    max_episode_steps: int,
    perturb_scale: float,
) -> list[dict]:
    rows = []
    for seed in range(seed_base, seed_base + seed_count):
        seed_everything(seed)
        agent = load_agent(agent_class, kwargs, checkpoint)
        env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
        state, _ = env.reset(seed=seed)
        state = np.asarray(state, dtype=np.float32)

        if perturb_scale > 0:
            rng = np.random.default_rng(seed)
            state = state + rng.normal(0.0, perturb_scale, size=state.shape).astype(np.float32)
            try:
                env.unwrapped.state = state
            except Exception:
                pass

        steps = 0
        terminated = False
        truncated = False
        while True:
            action = int(agent.predict(state))
            state, _, terminated, truncated, _ = env.step(action)
            state = np.asarray(state, dtype=np.float32)
            steps += 1
            if terminated or truncated:
                break
        env.close()

        reason = classify_terminal(state, truncated)
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "steps": steps,
                "terminal_reason": reason,
                "cart_pos": float(state[0]),
                "cart_vel": float(state[1]),
                "pole_angle": float(state[2]),
                "pole_angular_velocity": float(state[3]),
                "abs_cart_pos": abs(float(state[0])),
                "abs_pole_angle": abs(float(state[2])),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for variant in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        reasons = sorted({r["terminal_reason"] for r in sub})
        base = {
            "variant": variant,
            "episodes": len(sub),
            "mean_steps": float(np.mean([r["steps"] for r in sub])),
            "median_steps": float(np.median([r["steps"] for r in sub])),
            "time_limit_count": sum(r["terminal_reason"] == "time_limit" for r in sub),
            "cart_pos_count": sum("cart_pos" in r["terminal_reason"] for r in sub),
            "pole_angle_count": sum("pole_angle" in r["terminal_reason"] for r in sub),
            "mean_abs_terminal_x": float(np.mean([r["abs_cart_pos"] for r in sub])),
            "mean_abs_terminal_theta": float(np.mean([r["abs_pole_angle"] for r in sub])),
        }
        for reason in reasons:
            base[f"count_{reason}"] = sum(r["terminal_reason"] == reason for r in sub)
        out.append(base)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose terminal failure states for Bonus 3 smoothing variants.")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--dims", nargs="*", default=["baseline", "pos_only", "angular_only", "angle_only", "velocity_only"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_DIR / "Bonus_3" / "results" / "failure_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = PROJECT_DIR / "checkpoints" / "q_learning_model.pkl"
    all_rows = []
    for dims in args.dims:
        if dims == "baseline":
            name = "baseline"
            agent_class = "train_qlearning:Agent"
            kwargs = {"n_state": 4, "n_action": 2, "lr": 0.04, "gamma": 0.99, "epsilon": 0.0}
        else:
            name = dims
            agent_class = "Bonus_3.train_qlearning_bonus3:Agent"
            kwargs = make_agent_kwargs(dims, args.alpha, args.tau, args.radius)
        all_rows.extend(
            run_variant(
                name=name,
                agent_class=agent_class,
                kwargs=kwargs,
                checkpoint=baseline_ckpt,
                seed_base=args.seed_base,
                seed_count=args.seed_count,
                max_episode_steps=args.max_episode_steps,
                perturb_scale=args.perturb_scale,
            )
        )

    summary = summarize(all_rows)
    write_csv(all_rows, out_dir / "terminal_states.csv")
    write_csv(summary, out_dir / "terminal_summary.csv")
    with (out_dir / "terminal_states.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    with (out_dir / "terminal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done. Failure diagnostics written to:", out_dir)
    for row in summary:
        print(
            row["variant"],
            f"mean={row['mean_steps']:.1f}",
            f"time_limit={row['time_limit_count']}",
            f"cart_pos={row['cart_pos_count']}",
            f"pole_angle={row['pole_angle_count']}",
        )


if __name__ == "__main__":
    main()
