from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from models.benchmark import best_record, score_prediction_set


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate direct regression in Blender against Random Best@K and GT rerender."
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--random-k", type=int, default=8)
    p.add_argument("--render-samples", type=int, default=16)
    p.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.random_k < 1:
        raise ValueError("--random-k must be >= 1")

    data = args.data if args.data.is_absolute() else (REPO_ROOT / args.data)
    run_dir = args.run if args.run.is_absolute() else (REPO_ROOT / args.run)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = run_dir / "best.pt"
    elif not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint

    blender = args.blender
    if not blender.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = run_dir / "regression_predictions.jsonl"
    pred_renders = run_dir / "regression_renders"
    baseline_targets = run_dir / "regression_baseline_targets.jsonl"
    random_predictions = run_dir / "random_predictions.jsonl"
    gt_predictions = run_dir / "gt_predictions.jsonl"
    random_renders = run_dir / "random_renders"
    gt_renders = run_dir / "gt_renders"

    print("=== Direct regression evaluation ===")
    print(f"Targets:  {args.limit}")
    print(f"Random K: {args.random_k}")

    # 1) Deterministic image -> state inference: one prediction per target.
    run(
        [
            sys.executable,
            "-m",
            "models.infer_regression",
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(predictions),
            "--split",
            args.split,
            "--limit",
            str(args.limit),
        ]
    )

    # 2) Render the single deterministic regression prediction per target.
    run(
        [
            str(blender),
            "--background",
            "--python",
            "render_from_params/render_predictions.py",
            "--",
            "--predictions",
            str(predictions),
            "--out",
            str(pred_renders),
            "--samples",
            str(args.render_samples),
        ]
    )

    # 3) Ask the existing baseline generator for Random Best@K. We repeat only
    # target metadata here; the regression prediction itself remains deterministic.
    regression_rows = read_jsonl(predictions)
    repeated_rows: list[dict] = []
    for row in regression_rows:
        for sample_index in range(args.random_k):
            copy = dict(row)
            copy["sample_index"] = sample_index
            repeated_rows.append(copy)
    write_jsonl(baseline_targets, repeated_rows)

    run(
        [
            sys.executable,
            "-m",
            "models.make_baselines",
            "--data",
            str(data),
            "--targets-from",
            str(baseline_targets),
            "--out",
            str(run_dir),
        ]
    )

    for baseline_predictions, renders in (
        (random_predictions, random_renders),
        (gt_predictions, gt_renders),
    ):
        run(
            [
                str(blender),
                "--background",
                "--python",
                "render_from_params/render_predictions.py",
                "--",
                "--predictions",
                str(baseline_predictions),
                "--out",
                str(renders),
                "--samples",
                str(args.render_samples),
            ]
        )

    regression = score_prediction_set(data, predictions, pred_renders)
    random_baseline = score_prediction_set(data, random_predictions, random_renders)
    gt = score_prediction_set(data, gt_predictions, gt_renders)

    target_ids = sorted(set(regression) & set(random_baseline) & set(gt))
    rows = []
    for target_id in target_ids:
        r = best_record(regression[target_id])
        rand = best_record(random_baseline[target_id])
        g = best_record(gt[target_id])
        rows.append(
            {
                "target_id": target_id,
                "regression_score": float(r["main_score"]),
                "random_best_score": float(rand["main_score"]),
                "gt_score": float(g["main_score"]),
                "regression_minus_random": float(r["main_score"] - rand["main_score"]),
                "regression_beats_random": int(r["main_score"] > rand["main_score"]),
            }
        )

    regression_mean = float(np.mean([r["regression_score"] for r in rows]))
    random_mean = float(np.mean([r["random_best_score"] for r in rows]))
    gt_mean = float(np.mean([r["gt_score"] for r in rows]))
    win_fraction = float(np.mean([r["regression_beats_random"] for r in rows]))

    summary = {
        "targets": len(rows),
        "regression": {"main_score_mean": regression_mean, "k": 1},
        "random_best_of_k": {"main_score_mean": random_mean, "k": args.random_k},
        "gt_rerender": {"main_score_mean": gt_mean},
        "regression_vs_random": {
            "mean_score_advantage": regression_mean - random_mean,
            "fraction_targets_regression_wins": win_fraction,
        },
        "note": (
            "Regression is deterministic (K=1). Random uses Best@K so this intentionally "
            "keeps the same strong random baseline used by the diffusion benchmark."
        ),
    }

    csv_path = run_dir / "regression_benchmark_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = run_dir / "regression_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Direct Regression Benchmark ===")
    print(f"Targets:             {len(rows)}")
    print(f"GT rerender:         {gt_mean:.4f}")
    print(f"Regression @1:       {regression_mean:.4f}")
    print(f"Random best@{args.random_k}:      {random_mean:.4f}")
    print(f"Regression - random: {regression_mean - random_mean:+.4f}")
    print(f"Regression wins:     {win_fraction:.1%} of targets")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
