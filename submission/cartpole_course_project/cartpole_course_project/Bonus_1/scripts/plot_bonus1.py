from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_rows(report: dict):
    if "episodes" in report:
        return report["episodes"]
    if "per_episode" in report:
        return report["per_episode"]
    raise KeyError("report missing both 'episodes' and 'per_episode'")


def get_seed_step_pairs(report: dict):
    rows = get_episode_rows(report)
    pairs = [(int(x["seed"]), float(x["steps"])) for x in rows]
    pairs.sort(key=lambda t: t[0])
    seeds = [p[0] for p in pairs]
    steps = [p[1] for p in pairs]
    return seeds, steps


def get_hit_limit_ratio(step_summary: dict) -> float:
    if "hit_limit_ratio" in step_summary:
        return float(step_summary["hit_limit_ratio"])
    if "reached_max_ratio" in step_summary:
        return float(step_summary["reached_max_ratio"])
    return float("nan")


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    eval_dir = results_dir / "evaluations"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    speed_dir = results_dir / "speed"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scales = [0.0, 0.01, 0.02, 0.05]
    tags = ["ps00", "ps001", "ps002", "ps005"]
    reports = {}
    for sc, tag in zip(scales, tags):
        reports[sc] = {
            "Q-learning": load_report(eval_dir / f"qlearning_eval_{tag}.json"),
            "REINFORCE": load_report(eval_dir / f"reinforce_eval_{tag}.json"),
            "MCTS": load_report(eval_dir / f"mcts_eval_{tag}.json"),
        }

    # 1) 箱线图四合一（4 个扰动档）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for ax, sc in zip(axes.ravel(), scales):
        q_steps = [x["steps"] for x in get_episode_rows(reports[sc]["Q-learning"])]
        r_steps = [x["steps"] for x in get_episode_rows(reports[sc]["REINFORCE"])]
        m_steps = [x["steps"] for x in get_episode_rows(reports[sc]["MCTS"])]
        ax.boxplot([q_steps, r_steps, m_steps], tick_labels=["Q-learning", "REINFORCE", "MCTS"])
        ax.set_title(f"perturb={sc}")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Bonus1 Seed-Step Boxplot (4 Perturb Levels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "bonus1_seed_step_boxplot_4in1.png", dpi=150)
    plt.close(fig)

    # 2) 直方图四合一（4 个扰动档）
    bins = np.linspace(0, 2000, 41)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, sc in zip(axes.ravel(), scales):
        q_steps = [x["steps"] for x in get_episode_rows(reports[sc]["Q-learning"])]
        r_steps = [x["steps"] for x in get_episode_rows(reports[sc]["REINFORCE"])]
        m_steps = [x["steps"] for x in get_episode_rows(reports[sc]["MCTS"])]
        ax.hist(q_steps, bins=bins, alpha=0.45, label="Q-learning")
        ax.hist(r_steps, bins=bins, alpha=0.45, label="REINFORCE")
        ax.hist(m_steps, bins=bins, alpha=0.45, label="MCTS")
        ax.set_title(f"perturb={sc}")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Bonus1 Seed-Step Histogram (4 Perturb Levels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "bonus1_seed_step_hist_4in1.png", dpi=150)
    plt.close(fig)

    # 3) seed-step 折线图四合一（x=seed, y=steps）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, sc in zip(axes.ravel(), scales):
        q_seed, q_step_by_seed = get_seed_step_pairs(reports[sc]["Q-learning"])
        r_seed, r_step_by_seed = get_seed_step_pairs(reports[sc]["REINFORCE"])
        m_seed, m_step_by_seed = get_seed_step_pairs(reports[sc]["MCTS"])
        ax.plot(q_seed, q_step_by_seed, marker="o", markersize=2.2, linewidth=1.0, alpha=0.85, label="Q-learning")
        ax.plot(r_seed, r_step_by_seed, marker="o", markersize=2.2, linewidth=1.0, alpha=0.85, label="REINFORCE")
        ax.plot(m_seed, m_step_by_seed, marker="o", markersize=2.2, linewidth=1.0, alpha=0.85, label="MCTS")
        ax.set_title(f"perturb={sc}")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Bonus1 Seed vs Steps (4 Perturb Levels)")
    fig.tight_layout()
    fig.savefig(figures_dir / "bonus1_seed_step_by_seed_4in1.png", dpi=150)
    plt.close(fig)

    # 2) 定量对比表（CSV）
    table_csv = tables_dir / "bonus1_quant_table.csv"
    rows = []
    for sc in scales:
        for name, rep in reports[sc].items():
            s = rep["summary"]["steps"]
            rows.append(
                {
                    "perturb_scale": sc,
                    "algorithm": name,
                    "mean": s["mean"],
                    "median": s["median"],
                    "std": s["std"],
                    "min": s["min"],
                    "max": s["max"],
                    "p25": s["p25"],
                    "p75": s["p75"],
                    "hit_limit_ratio": get_hit_limit_ratio(s),
                }
            )
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "perturb_scale",
                "algorithm",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "p25",
                "p75",
                "hit_limit_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 4) perturb-scale 对比图（mean steps）
    algo_map = {
        "Q-learning": [],
        "REINFORCE": [],
        "MCTS": [],
    }
    for sc in scales:
        algo_map["Q-learning"].append(reports[sc]["Q-learning"]["summary"]["steps"]["mean"])
        algo_map["REINFORCE"].append(reports[sc]["REINFORCE"]["summary"]["steps"]["mean"])
        algo_map["MCTS"].append(reports[sc]["MCTS"]["summary"]["steps"]["mean"])

    plt.figure(figsize=(8, 5))
    for name, means in algo_map.items():
        plt.plot(scales, means, marker="o", label=name)
    plt.title("Bonus1 Perturb-Scale Robustness (Mean Steps)")
    plt.xlabel("perturb_scale")
    plt.ylabel("Mean Steps")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "bonus1_perturb_mean_steps.png", dpi=150)
    plt.close()

    # 5) 速率对比图（读 timing_summary.csv）
    timing_csv = speed_dir / "timing_summary.csv"
    if timing_csv.exists():
        grouped = {}
        with timing_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "skipped_existing_report":
                    continue
                if not row.get("episodes_per_sec"):
                    continue
                algo = row["algo"]
                ps = float(row["perturb_scale"])
                eps = float(row["episodes_per_sec"])
                grouped.setdefault(algo, []).append((ps, eps))
        if grouped:
            plt.figure(figsize=(8, 5))
            for algo, items in grouped.items():
                items.sort(key=lambda x: x[0])
                xs = [x for x, _ in items]
                ys = [y for _, y in items]
                plt.plot(xs, ys, marker="o", label=algo)
            plt.title("Bonus1 Compute Rate (episodes/s)")
            plt.xlabel("perturb_scale")
            plt.ylabel("episodes_per_sec")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "bonus1_compute_rate.png", dpi=150)
            plt.close()

    print("Done. Generated in:", results_dir)
    print(" - bonus1_seed_step_boxplot_4in1.png")
    print(" - bonus1_seed_step_hist_4in1.png")
    print(" - bonus1_seed_step_by_seed_4in1.png")
    print(" - bonus1_quant_table.csv")
    print(" - bonus1_perturb_mean_steps.png")
    print(" - bonus1_compute_rate.png (if measured timing rows exist)")


if __name__ == "__main__":
    main()
