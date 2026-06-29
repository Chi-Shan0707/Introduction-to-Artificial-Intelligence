"""
观测扰动评测脚本：
在每一步决策前给“观测状态”加噪声，再让 agent 基于噪声观测选动作。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import List, Tuple

import gymnasium as gym
import numpy as np

# 允许直接运行本脚本时导入项目根目录模块
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


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
        raise ValueError(f"--agent-class 需要是 'module:Class' 格式，收到: {agent_class_spec!r}")
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
    low = np.array([-2.4, -3.0, -0.5, -2.0], dtype=np.float32)
    high = np.array([2.4, 3.0, 0.5, 2.0], dtype=np.float32)
    return np.clip(state, low, high)


def _run_one_episode(
    agent_class_spec: str,
    init_kwargs: dict,
    checkpoint: str,
    seed: int,
    max_episode_steps: int,
    init_perturb_scale: float,
    obs_noise_scale: float,
) -> Tuple[int, int, float]:
    _seed_everything(seed)
    agent = _build_agent(agent_class_spec, init_kwargs, checkpoint)

    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    state, _ = env.reset(seed=seed)
    state = np.asarray(state, dtype=np.float32)

    if init_perturb_scale > 0:
        # 初始状态扰动：只在 reset 后加一次
        rng_init = np.random.default_rng(seed)
        state = state + rng_init.normal(0.0, init_perturb_scale, size=state.shape).astype(np.float32)
        state = _clip_state(state)
        try:
            env.unwrapped.state = state
        except Exception:
            pass

    # 观测噪声序列：同一 seed 下可复现
    rng_obs = np.random.default_rng(seed + 12345)
    steps = 0
    total_reward = 0.0
    while True:
        obs_state = state
        if obs_noise_scale > 0:
            # 观测扰动：每一步都加噪，再裁剪回有效范围
            obs_state = state + rng_obs.normal(0.0, obs_noise_scale, size=state.shape).astype(np.float32)
            obs_state = _clip_state(obs_state)

        action = int(agent.predict(obs_state))
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
    init_perturb_scale: float,
    obs_noise_scale: float,
) -> List[Tuple[int, int, float]]:
    results = []
    for seed in seeds:
        results.append(
            _run_one_episode(
                agent_class_spec=agent_class_spec,
                init_kwargs=init_kwargs,
                checkpoint=checkpoint,
                seed=seed,
                max_episode_steps=max_episode_steps,
                init_perturb_scale=init_perturb_scale,
                obs_noise_scale=obs_noise_scale,
            )
        )
    results.sort(key=lambda x: x[0])
    return results


def _build_summary(results: List[Tuple[int, int, float]], max_steps: int):
    steps = np.array([r[1] for r in results], dtype=np.float64)
    rewards = np.array([r[2] for r in results], dtype=np.float64)
    return {
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


def _print_summary(summary: dict):
    s = summary["steps"]
    r = summary["rewards"]
    print(f"\n=== 评测结果 ({summary['n_episodes']} episodes, TimeLimit={summary['max_episode_steps']}) ===")
    print(
        f"步数: mean={s['mean']:.1f} median={s['median']:.0f} std={s['std']:.1f} "
        f"min={s['min']:.0f} max={s['max']:.0f} (p25={s['p25']:.0f}/p75={s['p75']:.0f})"
    )
    print(f"      达到 TimeLimit 比例: {s['reached_max_ratio'] * 100:.1f}%")
    print(f"回报: mean={r['mean']:.2f} median={r['median']:.2f} std={r['std']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="按每步观测扰动进行评测")
    parser.add_argument("--agent-class", required=True)
    parser.add_argument("--agent-init-kwargs", default="{}")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--init-perturb-scale", type=float, default=0.0)
    parser.add_argument("--obs-noise-scale", type=float, default=0.0)
    parser.add_argument("--report-json", type=str, default=None)
    parser.add_argument("--print-per-seed", action="store_true")
    args = parser.parse_args()

    try:
        init_kwargs = json.loads(args.agent_init_kwargs)
    except json.JSONDecodeError as e:
        print(f"--agent-init-kwargs 不是合法 JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(init_kwargs, dict):
        print("--agent-init-kwargs 必须是 JSON 对象", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint 不存在: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)

    seeds = list(range(args.seed_base, args.seed_base + args.seed_count))
    print(f"Agent:      {args.agent_class}")
    print(f"Kwargs:     {init_kwargs}")
    print(f"Ckpt:       {args.checkpoint}")
    print(f"Seeds:      [{seeds[0]}..{seeds[-1]}] ({len(seeds)})")
    print(f"InitPert:   {args.init_perturb_scale}")
    print(f"ObsNoise:   {args.obs_noise_scale}")

    results = evaluate(
        agent_class_spec=args.agent_class,
        init_kwargs=init_kwargs,
        checkpoint=args.checkpoint,
        seeds=seeds,
        max_episode_steps=args.max_episode_steps,
        init_perturb_scale=args.init_perturb_scale,
        obs_noise_scale=args.obs_noise_scale,
    )
    if args.print_per_seed:
        print("\nper-seed:")
        for s, steps, rew in results:
            print(f"  seed={s:>5d}  steps={steps:>5d}  reward={rew:>8.2f}")

    summary = _build_summary(results, args.max_episode_steps)
    _print_summary(summary)

    if args.report_json:
        payload = {
            "agent_class": args.agent_class,
            "init_kwargs": init_kwargs,
            "checkpoint": args.checkpoint,
            "seeds": seeds,
            "init_perturb_scale": args.init_perturb_scale,
            "obs_noise_scale": args.obs_noise_scale,
            "per_episode": [
                {"seed": s, "steps": int(steps), "reward": float(rew)} for s, steps, rew in results
            ],
            "summary": summary,
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n详细报告已写入: {args.report_json}")


if __name__ == "__main__":
    main()
