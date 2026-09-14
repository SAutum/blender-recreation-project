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
        description=(
            "Small image-conditioning ablation: compare normal conditioning with a "
            "deterministically shuffled conditioning image/pair on the same targets and seed."
        )
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--samples-per-image", type=int, default=4)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--render-samples", type=int, default=8)
    p.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def main() -> None:
    args = parse_args()

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
    if not (data / "metadata.jsonl").exists():
        raise FileNotFoundError(f"Dataset metadata not found: {data / 'metadata.jsonl'}")

    out_dir = run_dir / "conditioning_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_predictions = out_dir / "normal_predictions.jsonl"
    shuffled_predictions = out_dir / "shuffled_predictions.jsonl"
    normal_renders = out_dir / "normal_renders"
    shuffled_renders = out_dir / "shuffled_renders"

    common_infer = [
        sys.executable,
        "-m",
        "models.infer",
        "--data",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--split",
        args.split,
        "--samples-per-image",
        str(args.samples_per_image),
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
    ]

    print("=== Conditioning ablation ===")
    print(f"Targets:   {args.limit}")
    print(f"Best-of-K: {args.samples_per_image}")
    print("Normal and shuffled runs use the same diffusion seed.")

    run(common_infer + ["--out", str(normal_predictions)])
    run(
        common_infer
        + [
            "--out",
            str(shuffled_predictions),
            "--shuffle-conditioning",
        ]
    )

    for predictions, renders in (
        (normal_predictions, normal_renders),
        (shuffled_predictions, shuffled_renders),
    ):
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
                str(renders),
                "--samples",
                str(args.render_samples),
            ]
        )

    normal = score_prediction_set(data, normal_predictions, normal_renders)
    shuffled = score_prediction_set(data, shuffled_predictions, shuffled_renders)
    target_ids = sorted(set(normal) & set(shuffled))
    if not target_ids:
        raise RuntimeError("No common targets between normal and shuffled runs")

    rows = []
    for target_id in target_ids:
        n = best_record(normal[target_id])
        s = best_record(shuffled[target_id])
        rows.append(
            {
                "target_id": target_id,
                "normal_best_score": float(n["main_score"]),
                "shuffled_best_score": float(s["main_score"]),
                "normal_minus_shuffled": float(n["main_score"] - s["main_score"]),
                "normal_wins": int(n["main_score"] > s["main_score"]),
            }
        )

    normal_mean = mean([r["normal_best_score"] for r in rows])
    shuffled_mean = mean([r["shuffled_best_score"] for r in rows])
    advantage = normal_mean - shuffled_mean
    win_fraction = mean([r["normal_wins"] for r in rows])

    summary = {
        "targets": len(rows),
        "samples_per_image": args.samples_per_image,
        "normal_best_of_k_mean": normal_mean,
        "shuffled_best_of_k_mean": shuffled_mean,
        "normal_minus_shuffled": advantage,
        "fraction_targets_normal_wins": win_fraction,
        "interpretation": (
            "If normal conditioning clearly beats shuffled conditioning, the denoiser is using "
            "image information. If the two are similar, generation is dominated by the scene prior."
        ),
    }

    csv_path = out_dir / "conditioning_ablation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "conditioning_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Normal vs Shuffled Conditioning ===")
    print(f"Targets:          {len(rows)}")
    print(f"Normal best@K:    {normal_mean:.4f}")
    print(f"Shuffled best@K:  {shuffled_mean:.4f}")
    print(f"Normal - shuffled:{advantage:+.4f}")
    print(f"Normal wins:      {win_fraction:.1%} of targets")
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
