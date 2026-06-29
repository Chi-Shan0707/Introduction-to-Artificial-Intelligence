from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run_timed(cmd: list[str], cwd: Path) -> float:
    print("\n[RUN]", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    return time.perf_counter() - t0


def write_svg(rows: list[dict], out_path: Path) -> None:
    width = 980
    row_h = 46
    top = 112
    height = top + row_h * (len(rows) + 1) + 52

    def esc(x) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="36" y="46" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="700" fill="#222222">Basic Evaluation Compute Speed</text>',
        '<text x="36" y="76" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#555555">100 seeds, max_episode_steps=2000, perturb_scale=0.0. Reports are written separately under Bonus_1/results/speed.</text>',
        f'<rect x="36" y="{top}" width="908" height="{row_h}" fill="#263238" rx="4"/>',
        f'<text x="58" y="{top + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Algorithm</text>',
        f'<text x="245" y="{top + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Elapsed sec</text>',
        f'<text x="425" y="{top + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Episodes/sec</text>',
        f'<text x="625" y="{top + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#ffffff">Report</text>',
    ]

    for i, row in enumerate(rows):
        y = top + row_h * (i + 1)
        fill = "#f7f9fb" if i % 2 == 0 else "#ffffff"
        parts.extend(
            [
                f'<rect x="36" y="{y}" width="908" height="{row_h}" fill="{fill}"/>',
                f'<text x="58" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#222222">{esc(row["algorithm"])}</text>',
                f'<text x="245" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#222222">{row["elapsed_sec"]:.2f}</text>',
                f'<text x="425" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#222222">{row["episodes_per_sec"]:.3f}</text>',
                f'<text x="625" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#555555">{esc(Path(row["report_json"]).name)}</text>',
            ]
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    out_dir = project_dir / "Bonus_1" / "results" / "speed"
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    seed_base = 42
    seed_count = 100
    max_steps = 2000

    jobs = [
        {
            "algorithm": "Q-learning",
            "report": out_dir / "qlearning_basic_speed_eval.json",
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
                "0.0",
            ],
        },
        {
            "algorithm": "REINFORCE",
            "report": out_dir / "reinforce_basic_speed_eval.json",
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
                "0.0",
            ],
        },
        {
            "algorithm": "MCTS",
            "report": out_dir / "mcts_basic_speed_eval.json",
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
                "0.0",
                "--progress-every",
                "10",
            ],
        },
    ]

    rows = []
    for job in jobs:
        cmd = job["cmd"] + ["--report-json", str(job["report"])]
        elapsed = run_timed(cmd, project_dir)
        rows.append(
            {
                "algorithm": job["algorithm"],
                "seed_count": seed_count,
                "max_episode_steps": max_steps,
                "perturb_scale": 0.0,
                "elapsed_sec": elapsed,
                "episodes_per_sec": seed_count / elapsed if elapsed > 0 else 0.0,
                "report_json": str(job["report"]),
            }
        )

    csv_path = out_dir / "basic_compute_rate.csv"
    json_path = out_dir / "basic_compute_rate.json"
    svg_path = out_dir / "basic_compute_rate_table.svg"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "seed_count",
                "max_episode_steps",
                "perturb_scale",
                "elapsed_sec",
                "episodes_per_sec",
                "report_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    write_svg(rows, svg_path)

    print("\nDone. Basic speed outputs:")
    print(" -", csv_path)
    print(" -", json_path)
    print(" -", svg_path)


if __name__ == "__main__":
    main()
