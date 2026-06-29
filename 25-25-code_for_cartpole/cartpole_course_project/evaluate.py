"""
General CartPole Model Evaluation Script.

Supports:
- Dynamic loading of student Agent classes
- Parallel evaluation across multiple seeds
- Initial state perturbation (perturb_scale)
- Per-step observation noise perturbation (obs_noise_scale)

Perturbation Implementation in CartPole-v1:
- Initial perturbation: Added to state at reset() via `env.unwrapped.state = perturbed_state`
- Observation noise: Added to each observation during inference before action prediction

Usage:
    python evaluate.py \
        --agent-class train_qlearning:Agent \
        --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' \
        --checkpoint checkpoints/q_learning_model.pkl \
        --seed-base 42 --seed-count 100
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _load_agent_class(agent_class_spec: str):
    if ":" not in agent_class_spec:
        raise ValueError(f"--agent-class must be 'module:Class' format, got: {agent_class_spec!r}")
    module_name, class_name = agent_class_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_agent(agent_class_spec: str, init_kwargs: dict, checkpoint: str):
    AgentCls = _load_agent_class(agent_class_spec)
    agent = AgentCls(**init_kwargs)
    agent.load_model(checkpoint)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def _clip_state(state: np.ndarray) -> np.ndarray:
    """Clip state to CartPole observation bounds."""
    bounds_low = np.array([-2.4, -3.0, -0.5, -2.0], dtype=np.float32)
    bounds_high = np.array([2.4, 3.0, 0.5, 2.0], dtype=np.float32)
    return np.clip(state, bounds_low, bounds_high)


def _run_one_episode(
    agent_class_spec: str,
    init_kwargs: dict,
    checkpoint: str,
    seed: int,
    max_episode_steps: int,
    perturb_scale: float,
    obs_noise_scale: float,
) -> Tuple[int, int, float]:
    """Run a single episode. Returns (seed, steps, total_reward)."""
    _seed_everything(seed)
    agent = _build_agent(agent_class_spec, init_kwargs, checkpoint)

    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    state, _ = env.reset(seed=seed)
    state = np.asarray(state, dtype=np.float32)

    # Initial perturbation: modify starting state with Gaussian noise
    if perturb_scale > 0:
        rng = np.random.default_rng(seed)
        state = state + rng.normal(0.0, perturb_scale, size=state.shape).astype(np.float32)
        state = _clip_state(state)
        try:
            env.unwrapped.state = state
        except Exception:
            pass

    steps = 0
    total_reward = 0.0
    while True:
        # Observation noise: add noise to each observation during inference
        obs = state
        if obs_noise_scale > 0:
            rng_obs = np.random.default_rng(seed + 12345)
            obs = state + rng_obs.normal(0.0, obs_noise_scale, size=state.shape).astype(np.float32)
            obs = _clip_state(obs)

        action = int(agent.predict(obs))
        state, reward, terminated, truncated, _ = env.step(action)
        state = np.asarray(state, dtype=np.float32)

        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return seed, steps, total_reward


def evaluate(
    agent_class_spec: str,
    init_kwargs: dict,
    checkpoint: str,
    seeds: List[int],
    max_episode_steps: int,
    workers: int,
    perturb_scale: float,
    obs_noise_scale: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """Return list of (seed, steps, reward) results, sorted by seed."""
    results: List[Tuple[int, int, float]] = []
    if workers <= 1:
        for s in seeds:
            results.append(
                _run_one_episode(
                    agent_class_spec, init_kwargs, checkpoint, s, max_episode_steps,
                    perturb_scale, obs_noise_scale,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _run_one_episode,
                    agent_class_spec, init_kwargs, checkpoint, s, max_episode_steps,
                    perturb_scale, obs_noise_scale,
                )
                for s in seeds
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    results.sort(key=lambda r: r[0])
    return results


def _print_report(results: List[Tuple[int, int, float]], max_steps: int) -> dict:
    steps = np.array([r[1] for r in results], dtype=np.float64)
    rewards = np.array([r[2] for r in results], dtype=np.float64)
    report = {
        "n_episodes": len(results),
        "max_episode_steps": max_steps,
        "steps": {
            "mean": float(np.mean(steps)),
            "median": float(np.median(steps)),
            "std": float(np.std(steps)),
            "min": float(np.min(steps)),
            "max": float(np.max(steps)),
            "p25": float(np.percentile(steps, 25)),
            "p75": float(np.percentile(steps, 75)),
            "reached_max_ratio": float(np.mean(steps >= max_steps)),
        },
        "rewards": {
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "std": float(np.std(rewards)),
        },
    }
    print(f"\n=== Evaluation Results ({report['n_episodes']} episodes, TimeLimit={max_steps}) ===")
    s = report["steps"]
    print(f"Steps: mean={s['mean']:.1f} median={s['median']:.0f} std={s['std']:.1f} "
          f"min={s['min']:.0f} max={s['max']:.0f} (p25={s['p25']:.0f}/p75={s['p75']:.0f})")
    print(f"      Reached TimeLimit ratio: {s['reached_max_ratio']*100:.1f}%")
    r = report["rewards"]
    print(f"Reward: mean={r['mean']:.2f} median={r['median']:.2f} std={r['std']:.2f}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="CartPole general evaluation - supports dynamic Agent loading + multi-seed parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent-class", required=True, help="Agent class locator, format 'module:ClassName'")
    parser.add_argument("--agent-init-kwargs", default="{}", help="Agent constructor kwargs (JSON string)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint file path")
    parser.add_argument("--seed-base", type=int, default=42, help="Seed start")
    parser.add_argument("--seed-count", type=int, default=100, help="Seed count")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Parallel workers (default: CPU half)")
    parser.add_argument("--max-episode-steps", type=int, default=2000,
                        help="Single episode TimeLimit (overrides CartPole-v1 default 500)")
    parser.add_argument("--perturb-scale", type=float, default=0.0,
                        help="Initial state Gaussian perturbation scale (0 disables)")
    parser.add_argument("--obs-noise-scale", type=float, default=0.0,
                        help="Per-step observation noise perturbation scale (0 disables)")
    parser.add_argument("--report-json", type=str, default=None, help="Optional: save stats to JSON")
    parser.add_argument("--print-per-seed", action="store_true", help="Print each seed's steps/reward")

    args = parser.parse_args()

    try:
        init_kwargs = json.loads(args.agent_init_kwargs)
    except json.JSONDecodeError as e:
        print(f"--agent-init-kwargs is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(init_kwargs, dict):
        print("--agent-init-kwargs must parse to a dict", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)

    seeds = list(range(args.seed_base, args.seed_base + args.seed_count))
    print(f"Agent:    {args.agent_class}")
    print(f"Kwargs:   {init_kwargs}")
    print(f"Ckpt:     {args.checkpoint}")
    print(f"Seeds:    [{seeds[0]}..{seeds[-1]}] ({len(seeds)} seeds)")
    print(f"Workers:  {args.workers}")
    print(f"InitPert: {args.perturb_scale}")
    print(f"ObsNoise: {args.obs_noise_scale}")

    results = evaluate(
        agent_class_spec=args.agent_class,
        init_kwargs=init_kwargs,
        checkpoint=args.checkpoint,
        seeds=seeds,
        max_episode_steps=args.max_episode_steps,
        workers=args.workers,
        perturb_scale=args.perturb_scale,
        obs_noise_scale=args.obs_noise_scale,
    )

    if args.print_per_seed:
        print("\nper-seed:")
        for s, steps, rew in results:
            print(f"  seed={s:>5d}  steps={steps:>5d}  reward={rew:>8.2f}")

    report = _print_report(results, args.max_episode_steps)

    if args.report_json:
        payload = {
            "agent_class": args.agent_class,
            "init_kwargs": init_kwargs,
            "checkpoint": args.checkpoint,
            "seeds": seeds,
            "perturb_scale": args.perturb_scale,
            "obs_noise_scale": args.obs_noise_scale,
            "per_episode": [
                {"seed": s, "steps": int(steps), "reward": float(rew)}
                for s, steps, rew in results
            ],
            "summary": report,
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {args.report_json}")


if __name__ == "__main__":
    main()