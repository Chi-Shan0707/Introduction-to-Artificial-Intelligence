from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> float:
    print("\n[RUN]", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Run Bonus 1 evaluations for Q-learning, REINFORCE, and MCTS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even when the target JSON report already exists.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    bonus_dir = Path(__file__).resolve().parents[1]
    results_dir = bonus_dir / "results"
    eval_dir = results_dir / "evaluations"
    speed_dir = results_dir / "speed"
    eval_dir.mkdir(parents=True, exist_ok=True)
    speed_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    seed_base = 42
    seed_count = 100
    max_steps = 2000
    perturb_scales = [0.0, 0.01, 0.02, 0.05]

    jobs: list[dict] = []

    for ps in perturb_scales:
        tag = f"ps{str(ps).replace('.', '')}"
        jobs.append(
            {
                "algo": "qlearning",
                "cmd": [
                    py,
                    "evaluate.py",
                    "--agent-class",
                    "train_qlearning:Agent",
                    "--agent-init-kwargs",
                    '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}',
                    "--checkpoint",
                    "checkpoints/q_learning_model.pkl",
                    "--seed-base",
                    str(seed_base),
                    "--seed-count",
                    str(seed_count),
                    "--max-episode-steps",
                    str(max_steps),
                    "--perturb-scale",
                    str(ps),
                    "--report-json",
                    str(eval_dir / f"qlearning_eval_{tag}.json"),
                ],
            }
        )
        jobs.append(
            {
                "algo": "reinforce",
                "cmd": [
                    py,
                    "evaluate.py",
                    "--agent-class",
                    "train_reinforce:Agent",
                    "--agent-init-kwargs",
                    '{"n_state":4,"hidden_c":16,"n_action":2}',
                    "--checkpoint",
                    "checkpoints/reinforce_model.pt",
                    "--seed-base",
                    str(seed_base),
                    "--seed-count",
                    str(seed_count),
                    "--max-episode-steps",
                    str(max_steps),
                    "--perturb-scale",
                    str(ps),
                    "--report-json",
                    str(eval_dir / f"reinforce_eval_{tag}.json"),
                ],
            }
        )
        jobs.append(
            {
                "algo": "mcts",
                "cmd": [
                    py,
                    "Bonus_1/evaluate_mcts.py",
                    "--agent-class",
                    "train_mcts:Agent",
                    "--seed-base",
                    str(seed_base),
                    "--seed-count",
                    str(seed_count),
                    "--max-episode-steps",
                    str(max_steps),
                    "--iteration-budget",
                    "80",
                    "--lookahead-target",
                    "200",
                    "--start-cp",
                    "200",
                    "--perturb-scale",
                    str(ps),
                    "--progress-every",
                    "10",
                    "--report-json",
                    str(eval_dir / f"mcts_eval_{tag}.json"),
                ],
            }
        )

    timing_rows = []
    for job in jobs:
        report_path = Path(job["cmd"][job["cmd"].index("--report-json") + 1])
        was_skipped = report_path.exists() and not args.force
        if was_skipped:
            print(f"\n[SKIP] report exists: {report_path}")
            elapsed = None
        else:
            elapsed = run_cmd(job["cmd"], project_dir)
        episodes_per_sec = seed_count / elapsed if elapsed and elapsed > 0 else None
        timing_rows.append(
            {
                "algo": job["algo"],
                "perturb_scale": float(
                    job["cmd"][job["cmd"].index("--perturb-scale") + 1]
                ),
                "seed_count": seed_count,
                "elapsed_sec": elapsed,
                "episodes_per_sec": episodes_per_sec,
                "status": "skipped_existing_report" if was_skipped else "measured",
            }
        )

    timing_json = speed_dir / "timing_summary.json"
    timing_csv = speed_dir / "timing_summary.csv"
    with timing_json.open("w", encoding="utf-8") as f:
        json.dump(timing_rows, f, ensure_ascii=False, indent=2)

    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "perturb_scale",
                "seed_count",
                "elapsed_sec",
                "episodes_per_sec",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    print("\nDone. Files written to:")
    print(" ", results_dir)
    print(" - timing_summary.json")
    print(" - timing_summary.csv")


if __name__ == "__main__":
    main()
