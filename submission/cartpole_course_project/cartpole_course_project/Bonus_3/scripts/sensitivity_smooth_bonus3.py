from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def tag(x) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def read_steps(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["summary"]["steps"]


def color_for(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if t < 0.5:
        k = t / 0.5
        r, g, b = int(70 + k * 176), int(130 + k * 59), int(180 - k * 110)
    else:
        k = (t - 0.5) / 0.5
        r, g, b = int(246 - k * 32), int(189 - k * 120), int(70 - k * 5)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_rows(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def make_kwargs(alpha: float, tau: float, radius: int, dims: str) -> str:
    return json.dumps(
        {
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
        },
        ensure_ascii=False,
    )


def eval_one(
    *,
    project_dir: Path,
    out_dir: Path,
    py: str,
    ckpt: Path,
    seed_base: int,
    seed_count: int,
    max_steps: int,
    perturb_scale: float,
    alpha: float,
    tau: float,
    radius: int,
    dims: str,
    force: bool,
    prefix: str,
) -> dict:
    report = out_dir / (
        f"{prefix}_alpha{tag(alpha)}_tau{tag(tau)}_r{radius}_dims{dims}_ps{tag(perturb_scale)}.json"
    )
    if force or not report.exists():
        run_cmd(
            [
                py,
                "evaluate.py",
                "--agent-class",
                "Bonus_3.train_qlearning_bonus3:Agent",
                "--agent-init-kwargs",
                make_kwargs(alpha, tau, radius, dims),
                "--checkpoint",
                str(ckpt),
                "--seed-base",
                str(seed_base),
                "--seed-count",
                str(seed_count),
                "--max-episode-steps",
                str(max_steps),
                "--perturb-scale",
                str(perturb_scale),
                "--report-json",
                str(report),
            ],
            project_dir,
        )
    else:
        print(f"[SKIP] report exists: {report}")

    s = read_steps(report)
    return {
        "smooth_alpha": alpha,
        "confidence_tau": tau,
        "smooth_radius": radius,
        "smooth_dims": dims,
        "perturb_scale": perturb_scale,
        "seed_count": seed_count,
        "mean_steps": s["mean"],
        "median_steps": s["median"],
        "std_steps": s["std"],
        "reached_max_ratio": s["reached_max_ratio"],
        "report_json": str(report),
    }


def write_strength_svg(rows: list[dict], alphas: list[float], taus: list[float], radii: list[int], out: Path) -> None:
    values = {
        (float(r["smooth_alpha"]), float(r["confidence_tau"]), int(r["smooth_radius"])): float(r["mean_steps"])
        for r in rows
    }
    all_vals = list(values.values())
    vmin, vmax = min(all_vals), max(all_vals)

    cell_w, cell_h = 112, 62
    left, top = 104, 116
    gap = 54
    panel_w = cell_w * len(alphas)
    panel_h = cell_h * len(taus)
    width = left + len(radii) * panel_w + (len(radii) - 1) * gap + 70
    height = top + panel_h + 96

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="36" y="42" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#222">Smoothing Strength Sensitivity</text>',
        '<text x="36" y="70" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#555">Mean steps at perturb_scale=0.05. Each panel is one smooth_radius.</text>',
    ]
    for pi, radius in enumerate(radii):
        x0 = left + pi * (panel_w + gap)
        parts.append(f'<text x="{x0}" y="{top-22}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#222">radius={radius}</text>')
        for i, alpha in enumerate(alphas):
            x = x0 + i * cell_w + cell_w / 2
            parts.append(f'<text x="{x:.1f}" y="{top-6}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333">{alpha:g}</text>')
        for j, tau in enumerate(taus):
            y = top + j * cell_h + cell_h / 2
            if pi == 0:
                parts.append(f'<text x="{left-16}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333">{tau:g}</text>')
            for i, alpha in enumerate(alphas):
                x = x0 + i * cell_w
                yy = top + j * cell_h
                mean = values[(alpha, tau, radius)]
                parts.append(f'<rect x="{x}" y="{yy}" width="{cell_w}" height="{cell_h}" fill="{color_for(mean, vmin, vmax)}" stroke="#fff" stroke-width="2"/>')
                parts.append(f'<text x="{x+cell_w/2:.1f}" y="{yy+37:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#1f1f1f">{mean:.0f}</text>')
    parts.append(f'<text x="{left + (panel_w * len(radii) + gap * (len(radii)-1))/2:.1f}" y="{height-24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333">smooth_alpha; rows are confidence_tau</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def write_dims_svg(rows: list[dict], out: Path) -> None:
    vals = [(r["smooth_dims"], float(r["mean_steps"])) for r in rows]
    vmax = max(v for _, v in vals)
    width, height = 820, 430
    left, top, bar_h, gap = 190, 92, 36, 24
    scale_w = 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="36" y="42" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#222">Smoothing Dimension Sensitivity</text>',
        '<text x="36" y="70" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#555">Mean steps at fixed alpha/tau/radius. Higher is better.</text>',
    ]
    for i, (name, value) in enumerate(vals):
        y = top + i * (bar_h + gap)
        w = scale_w * value / vmax
        parts.append(f'<text x="{left-14}" y="{y+24}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333">{name}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="#61dDAa"/>')
        parts.append(f'<text x="{left+w+10:.1f}" y="{y+24}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#222">{value:.1f}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus 3 smoothing sensitivity.")
    parser.add_argument("--mode", choices=["strength", "dims", "both"], default="both")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.8, 0.9, 0.95])
    parser.add_argument("--taus", type=float, nargs="*", default=[0.4, 0.8, 1.2])
    parser.add_argument("--radii", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--dims", nargs="*", default=["pos_only", "velocity_only", "angle_only", "angular_only"])
    parser.add_argument("--strength-dims", type=str, default="angular_only")
    parser.add_argument("--fixed-alpha", type=float, default=0.9)
    parser.add_argument("--fixed-tau", type=float, default=0.8)
    parser.add_argument("--fixed-radius", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run evaluate.py.")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    out_dir = project_dir / "Bonus_3" / "results" / "sensitivity_smooth"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = args.python
    ckpt = project_dir / "checkpoints" / "q_learning_model.pkl"

    common = {
        "project_dir": project_dir,
        "out_dir": out_dir,
        "py": py,
        "ckpt": ckpt,
        "seed_base": args.seed_base,
        "seed_count": args.seed_count,
        "max_steps": args.max_episode_steps,
        "perturb_scale": args.perturb_scale,
        "force": args.force,
    }

    if args.mode in {"strength", "both"}:
        rows = []
        for radius in args.radii:
            for alpha in args.alphas:
                for tau in args.taus:
                    rows.append(
                        eval_one(
                            **common,
                            alpha=alpha,
                            tau=tau,
                            radius=radius,
                            dims=args.strength_dims,
                            prefix="strength",
                        )
                    )
        write_rows(rows, out_dir / "strength_sensitivity_summary.csv", out_dir / "strength_sensitivity_summary.json")
        write_strength_svg(rows, args.alphas, args.taus, args.radii, out_dir / "strength_sensitivity_heatmap.svg")
        best = max(rows, key=lambda r: r["mean_steps"])
        print("Best strength:", best["smooth_alpha"], best["confidence_tau"], best["smooth_radius"], best["mean_steps"])

    if args.mode in {"dims", "both"}:
        rows = []
        for dims in args.dims:
            rows.append(
                eval_one(
                    **common,
                    alpha=args.fixed_alpha,
                    tau=args.fixed_tau,
                    radius=args.fixed_radius,
                    dims=dims,
                    prefix="dims",
                )
            )
        write_rows(rows, out_dir / "dims_sensitivity_summary.csv", out_dir / "dims_sensitivity_summary.json")
        write_dims_svg(rows, out_dir / "dims_sensitivity_bar.svg")
        best = max(rows, key=lambda r: r["mean_steps"])
        print("Best dims:", best["smooth_dims"], best["mean_steps"])

    print("\nOutputs written to:", out_dir)


if __name__ == "__main__":
    main()
