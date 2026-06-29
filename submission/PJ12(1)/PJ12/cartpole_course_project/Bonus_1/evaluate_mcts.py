"""
MCTS 专用评测脚本（补充脚本，不替代 evaluate.py）。

用途:
1) 对 train_mcts:Agent 做多 seed 评测
2) 输出和 evaluate.py 类似的统计信息
3) 可选导出 JSON，便于 Bonus 报告画图/对比

示例:
    python evaluate_mcts.py --seed-base 42 --seed-count 100 \
        --iteration-budget 80 --lookahead-target 200 --start-cp 200 \
        --report-json ../Bonus_1/mcts_eval_report.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

# 允许在 Bonus_1 子目录运行时导入上层项目模块（如 train_mcts.py）
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
        raise ValueError(
            f"--agent-class 应为 'module:Class' 格式,收到: {agent_class_spec!r}"
        )
    module_name, class_name = agent_class_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _run_one_episode(
    agent_class_spec: str,
    agent_init_kwargs: Dict,
    env_id: str,
    seed: int,
    max_episode_steps: int,
    start_cp: int,
    lookahead_target: int,
    perturb_scale: float,
) -> Tuple[int, int, float]:
    _seed_everything(seed)
    AgentCls = _load_agent_class(agent_class_spec)
    agent = AgentCls(**agent_init_kwargs)

    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    state, _ = env.reset(seed=seed)
    state = np.array(state, dtype=np.float32)

    if perturb_scale > 0:
        rng = np.random.default_rng(seed)
        state = state + rng.normal(0.0, perturb_scale, size=state.shape).astype(np.float32)
        try:
            env.unwrapped.state = state
        except Exception:
            pass

    node = None
    c_p = int(start_cp)
    steps = 0
    total_reward = 0.0

    while True:
        action, node, c_p = agent.act(
            env.unwrapped.state,
            n_actions=env.action_space.n,
            node=node,
            C_p=c_p,
            lookahead_target=lookahead_target,
        )
        _, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return seed, steps, total_reward


def _summary(values: List[float], max_episode_steps: int) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "hit_limit_ratio": float(np.mean(arr >= max_episode_steps)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CartPole MCTS 专用评测入口（含 report-json）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent-class", type=str, default="train_mcts:Agent")
    parser.add_argument("--agent-init-kwargs", type=str, default=None,
                        help="可选 JSON。若不传则使用 iteration_budget + env_id 组装")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--iteration-budget", type=int, default=80)
    parser.add_argument("--lookahead-target", type=int, default=200)
    parser.add_argument("--start-cp", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--perturb-scale", type=float, default=0.0)
    parser.add_argument("--report-json", type=str, default=None)
    parser.add_argument("--print-per-seed", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.agent_init_kwargs is None:
        init_kwargs = {
            "iteration_budget": args.iteration_budget,
            "env_id": args.env_id,
        }
    else:
        try:
            init_kwargs = json.loads(args.agent_init_kwargs)
        except json.JSONDecodeError as e:
            print(f"--agent-init-kwargs 不是合法 JSON: {e}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(init_kwargs, dict):
            print("--agent-init-kwargs 必须解析为 JSON 对象(dict)", file=sys.stderr)
            sys.exit(2)

    seeds = list(range(args.seed_base, args.seed_base + args.seed_count))

    print(f"Agent:    {args.agent_class}")
    print(f"Kwargs:   {init_kwargs}")
    print(f"Seeds:    [{seeds[0]}..{seeds[-1]}] ({len(seeds)} 个)")
    print(f"Env:      {args.env_id}")
    print(f"MaxStep:  {args.max_episode_steps}")
    print(f"Perturb:  {args.perturb_scale}")

    results: List[Tuple[int, int, float]] = []
    for idx, s in enumerate(seeds, start=1):
        one = _run_one_episode(
            agent_class_spec=args.agent_class,
            agent_init_kwargs=init_kwargs,
            env_id=args.env_id,
            seed=s,
            max_episode_steps=args.max_episode_steps,
            start_cp=args.start_cp,
            lookahead_target=args.lookahead_target,
            perturb_scale=args.perturb_scale,
        )
        results.append(one)
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(seeds)):
            print(f"[progress] {idx}/{len(seeds)} seeds done")

    results.sort(key=lambda x: x[0])
    steps = [r[1] for r in results]
    rewards = [r[2] for r in results]

    steps_s = _summary(steps, args.max_episode_steps)
    rewards_s = _summary(rewards, args.max_episode_steps)

    print(f"\n=== 评估结果({len(results)} episodes, TimeLimit={args.max_episode_steps}) ===")
    print(
        "步数:"
        f"mean={steps_s['mean']:.1f}  median={steps_s['median']:.0f}  std={steps_s['std']:.1f}"
        f"  min={steps_s['min']:.0f}  max={steps_s['max']:.0f}"
        f"  (p25={steps_s['p25']:.0f}/p75={steps_s['p75']:.0f})"
    )
    print(f"      达到 TimeLimit 上限的比例: {steps_s['hit_limit_ratio'] * 100:.1f}%")
    print(
        "回报:"
        f"mean={rewards_s['mean']:.2f}  median={rewards_s['median']:.2f}  std={rewards_s['std']:.2f}"
    )

    if args.print_per_seed:
        print("\nper-seed:")
        for s, st, rw in results:
            print(f"  seed={s:>5d}  steps={st:>5d}  reward={rw:>8.2f}")

    if args.report_json:
        payload = {
            "agent_class": args.agent_class,
            "agent_init_kwargs": init_kwargs,
            "env_id": args.env_id,
            "seed_base": args.seed_base,
            "seed_count": args.seed_count,
            "seeds": seeds,
            "max_episode_steps": args.max_episode_steps,
            "perturb_scale": args.perturb_scale,
            "summary": {
                "steps": steps_s,
                "rewards": rewards_s,
            },
            "episodes": [
                {"seed": s, "steps": int(st), "reward": float(rw)}
                for s, st, rw in results
            ],
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存: {args.report_json}")


if __name__ == "__main__":
    main()
