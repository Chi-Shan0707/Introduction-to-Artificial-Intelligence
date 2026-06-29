"""
Bonus3 统一评测+可视化脚本。

会对三种策略在两类扰动下做四挡对比：
1) 初始状态扰动（evaluate.py --perturb-scale）
2) 观测噪声扰动（evaluate_obs_perturb.py --obs-noise-scale）
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Bonus_3.train_qlearning_bonus3 import (
    DEFAULT_CONFIDENCE_TAU,
    DEFAULT_SMOOTH_ALPHA,
    DEFAULT_SMOOTH_DIMS,
    DEFAULT_SMOOTH_RADIUS,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run_cmd(cmd: list[str], cwd: Path):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_steps_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["summary"]["steps"]


def read_episode_steps(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("per_episode") or payload.get("episodes")
    if episodes is None:
        raise KeyError(f"Missing per-episode results in {path}")
    return [float(x["steps"]) for x in episodes]


def eval_job(py, project_dir, report, cmd_args, force=False):
    if report.exists() and not force:
        print(f"\n[SKIP] report exists: {report}")
    else:
        run_cmd([py] + cmd_args + ["--report-json", str(report)], project_dir)
    return read_steps_summary(report)


def save_summary(rows, out_json: Path, out_csv: Path):
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_type",
                "agent",
                "scale",
                "mean_steps",
                "std_steps",
                "reached_max_ratio",
                "report_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(rows, out_png: Path):
    tests = ["init_perturb", "obs_perturb"]
    agents = ["baseline", "inference_smooth", "obsnoise_train"]
    colors = {
        "baseline": "#5b8ff9",
        "inference_smooth": "#61dDAa",
        "obsnoise_train": "#f6bd16",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, test in zip(axes, tests):
        for agent in agents:
            pts = [r for r in rows if r["test_type"] == test and r["agent"] == agent]
            pts = sorted(pts, key=lambda x: x["scale"])
            x = [p["scale"] for p in pts]
            y = [p["mean_steps"] for p in pts]
            ax.plot(x, y, marker="o", linewidth=2, label=agent, color=colors[agent])
        ax.set_title("初始状态扰动" if test == "init_perturb" else "观测噪声扰动")
        ax.set_xlabel("scale")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("平均步数")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.suptitle("Bonus3 鲁棒性对比（四挡）", y=1.03)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def quantile(values: list[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def box_stats(values: list[float]) -> dict:
    vals = sorted(values)
    return {
        "min": vals[0],
        "q1": quantile(vals, 0.25),
        "median": quantile(vals, 0.5),
        "q3": quantile(vals, 0.75),
        "max": vals[-1],
    }


def ymap(value: float, top: float, panel_h: float, max_steps: float) -> float:
    return top + panel_h - (value / max_steps) * panel_h


def plot_boxplot_svg(rows: list[dict], out_svg: Path, max_steps: int):
    tests = [("init_perturb", "Initial perturb"), ("obs_perturb", "Observation noise")]
    agents = ["baseline", "inference_smooth", "obsnoise_train"]
    labels = {"baseline": "Base", "inference_smooth": "Smooth", "obsnoise_train": "NoiseTrain"}
    colors = {"baseline": "#5b8ff9", "inference_smooth": "#61dDAa", "obsnoise_train": "#f6bd16"}
    scales = sorted({float(r["scale"]) for r in rows})

    width, height = 1240, 760
    left, right, top, bottom = 70, 30, 92, 70
    panel_gap = 56
    panel_w = (width - left - right - panel_gap) / 2
    panel_h = height - top - bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="44" y="42" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700" fill="#222">Bonus 3 Robustness Boxplots</text>',
        '<text x="44" y="68" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#555">100 seeds per setting. Boxes show p25 / median / p75; whiskers show min / max.</text>',
    ]

    legend_x = 760
    for i, agent in enumerate(agents):
        x = legend_x + i * 130
        parts.append(f'<rect x="{x}" y="35" width="16" height="16" fill="{colors[agent]}" opacity="0.72"/>')
        parts.append(f'<text x="{x+22}" y="48" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333">{labels[agent]}</text>')

    report_map = {(r["test_type"], r["agent"], float(r["scale"])): Path(r["report_json"]) for r in rows}

    for panel_i, (test, title) in enumerate(tests):
        x0 = left + panel_i * (panel_w + panel_gap)
        parts.append(f'<text x="{x0}" y="{top-18}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#222">{title}</text>')
        parts.append(f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fbfbfb" stroke="#ddd"/>')
        for tick in [0, 500, 1000, 1500, 2000]:
            y = ymap(tick, top, panel_h, max_steps)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            if panel_i == 0:
                parts.append(f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#555">{tick}</text>')

        group_w = panel_w / len(scales)
        box_w = 18
        for scale_i, scale in enumerate(scales):
            center = x0 + group_w * (scale_i + 0.5)
            parts.append(f'<text x="{center}" y="{top+panel_h+30}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#444">{scale:.2f}</text>')
            for agent_i, agent in enumerate(agents):
                report = report_map.get((test, agent, scale))
                if report is None or not report.exists():
                    continue
                stats = box_stats(read_episode_steps(report))
                bx = center + (agent_i - 1) * 24
                y_min = ymap(stats["min"], top, panel_h, max_steps)
                y_q1 = ymap(stats["q1"], top, panel_h, max_steps)
                y_med = ymap(stats["median"], top, panel_h, max_steps)
                y_q3 = ymap(stats["q3"], top, panel_h, max_steps)
                y_max = ymap(stats["max"], top, panel_h, max_steps)
                color = colors[agent]
                parts.append(f'<line x1="{bx:.1f}" y1="{y_max:.1f}" x2="{bx:.1f}" y2="{y_min:.1f}" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<line x1="{bx-7:.1f}" y1="{y_max:.1f}" x2="{bx+7:.1f}" y2="{y_max:.1f}" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<line x1="{bx-7:.1f}" y1="{y_min:.1f}" x2="{bx+7:.1f}" y2="{y_min:.1f}" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<rect x="{bx-box_w/2:.1f}" y="{y_q3:.1f}" width="{box_w}" height="{max(1, y_q1-y_q3):.1f}" fill="{color}" opacity="0.55" stroke="{color}"/>')
                parts.append(f'<line x1="{bx-box_w/2:.1f}" y1="{y_med:.1f}" x2="{bx+box_w/2:.1f}" y2="{y_med:.1f}" stroke="#222" stroke-width="2"/>')

    parts.append(f'<text x="22" y="{top+panel_h/2}" transform="rotate(-90 22 {top+panel_h/2})" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333">Steps</text>')
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行 Bonus3 四挡评测并作图")
    parser.add_argument("--force", action="store_true", help="即使已有结果也重新评测")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02, 0.05],
        help="初始扰动与观测扰动共用的四挡强度",
    )
    parser.add_argument(
        "--obsnoise-prefix",
        type=str,
        default="q_bonus3_obsnoise",
        help="训练加噪模型前缀（对应 results/<prefix>_model.pkl）",
    )
    parser.add_argument(
        "--obsnoise-std",
        type=float,
        default=0.005,
        help="obsnoise_train 评测时传入的 train_obs_noise_std（与训练值保持一致）",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=DEFAULT_SMOOTH_ALPHA,
        help="推理平滑中心权重",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=DEFAULT_SMOOTH_RADIUS,
        help="推理平滑邻域半径",
    )
    parser.add_argument(
        "--confidence-tau",
        type=float,
        default=DEFAULT_CONFIDENCE_TAU,
        help="置信门控阈值",
    )
    parser.add_argument(
        "--smooth-dims",
        type=str,
        default=DEFAULT_SMOOTH_DIMS,
        help="推理平滑维度；主方案使用 angular_only",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    bonus_dir = Path(__file__).resolve().parent
    results_dir = bonus_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    baseline_ckpt = project_dir / "checkpoints" / "q_learning_model.pkl"
    obsnoise_ckpt = results_dir / f"{args.obsnoise_prefix}_model.pkl"

    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_ckpt}")
    agents = [
        {
            "name": "baseline",
            "agent_class": "train_qlearning:Agent",
            "kwargs": '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}',
            "checkpoint": baseline_ckpt,
        },
        {
            "name": "inference_smooth",
            "agent_class": "Bonus_3.train_qlearning_bonus3:Agent",
            "kwargs": json.dumps(
                {
                    "n_state": 4,
                    "n_action": 2,
                    "lr": 0.04,
                    "gamma": 0.99,
                    "epsilon": 0.0,
                    "train_obs_noise_std": 0.0,
                    "smooth_mode": "confidence",
                    "smooth_alpha": args.smooth_alpha,
                    "smooth_radius": args.smooth_radius,
                    "smooth_dims": args.smooth_dims,
                    "confidence_tau": args.confidence_tau,
                },
                ensure_ascii=False,
            ),
            "checkpoint": baseline_ckpt,
        },
        {
            "name": "obsnoise_train",
            "agent_class": "Bonus_3.train_qlearning_bonus3:Agent",
            "kwargs": json.dumps(
                {
                    "n_state": 4,
                    "n_action": 2,
                    "lr": 0.04,
                    "gamma": 0.99,
                    "epsilon": 0.0,
                    "train_obs_noise_std": args.obsnoise_std,
                    "smooth_mode": "none",
                },
                ensure_ascii=False,
            ),
            "checkpoint": obsnoise_ckpt,
        },
    ]
    if not obsnoise_ckpt.exists():
        print(
            f"[WARN] 未找到加噪训练模型: {obsnoise_ckpt}\n"
            "       将仅评测 baseline 与 inference_smooth。\n"
            f"       如需加入 obsnoise_train，请先运行：\n"
            f"       python .\\Bonus_3\\train_qlearning_bonus3.py --train-obs-noise-std {args.obsnoise_std} --output-prefix {args.obsnoise_prefix}"
        )
        agents = [a for a in agents if a["name"] != "obsnoise_train"]

    rows = []
    for agent in agents:
        for scale in args.scales:
            tag = str(scale).replace(".", "")
            init_report = results_dir / f"init_{agent['name']}_ps{tag}.json"
            init_summary = eval_job(
                py,
                project_dir,
                init_report,
                [
                    "evaluate.py",
                    "--agent-class",
                    agent["agent_class"],
                    "--agent-init-kwargs",
                    agent["kwargs"],
                    "--checkpoint",
                    str(agent["checkpoint"]),
                    "--seed-base",
                    str(args.seed_base),
                    "--seed-count",
                    str(args.seed_count),
                    "--max-episode-steps",
                    str(args.max_episode_steps),
                    "--perturb-scale",
                    str(scale),
                ],
                force=args.force,
            )
            rows.append(
                {
                    "test_type": "init_perturb",
                    "agent": agent["name"],
                    "scale": scale,
                    "mean_steps": init_summary["mean"],
                    "std_steps": init_summary["std"],
                    "reached_max_ratio": init_summary["reached_max_ratio"],
                    "report_json": str(init_report),
                }
            )

            obs_report = results_dir / f"obs_{agent['name']}_ns{tag}.json"
            obs_summary = eval_job(
                py,
                project_dir,
                obs_report,
                [
                    "Bonus_3/evaluate_obs_perturb.py",
                    "--agent-class",
                    agent["agent_class"],
                    "--agent-init-kwargs",
                    agent["kwargs"],
                    "--checkpoint",
                    str(agent["checkpoint"]),
                    "--seed-base",
                    str(args.seed_base),
                    "--seed-count",
                    str(args.seed_count),
                    "--max-episode-steps",
                    str(args.max_episode_steps),
                    "--init-perturb-scale",
                    "0.0",
                    "--obs-noise-scale",
                    str(scale),
                ],
                force=args.force,
            )
            rows.append(
                {
                    "test_type": "obs_perturb",
                    "agent": agent["name"],
                    "scale": scale,
                    "mean_steps": obs_summary["mean"],
                    "std_steps": obs_summary["std"],
                    "reached_max_ratio": obs_summary["reached_max_ratio"],
                    "report_json": str(obs_report),
                }
            )

    out_json = results_dir / "bonus3_summary_4scales.json"
    out_csv = results_dir / "bonus3_summary_4scales.csv"
    out_png = results_dir / "bonus3_robustness_4scales.png"
    out_box_svg = results_dir / "bonus3_boxplot_4scales.svg"
    save_summary(rows, out_json, out_csv)
    plot_lines(rows, out_png)
    plot_boxplot_svg(rows, out_box_svg, args.max_episode_steps)

    print("\n完成，已生成：")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", out_png)
    print(" -", out_box_svg)


if __name__ == "__main__":
    main()
