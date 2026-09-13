from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from models.scorer import score_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--diffusion-predictions", type=Path, required=True)
    p.add_argument("--diffusion-renders", type=Path, required=True)
    p.add_argument("--random-predictions", type=Path, required=True)
    p.add_argument("--random-renders", type=Path, required=True)
    p.add_argument("--gt-predictions", type=Path, required=True)
    p.add_argument("--gt-renders", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_prediction_set(data: Path, predictions: Path, renders: Path) -> dict[int, list[dict]]:
    rows = read_jsonl(predictions)
    grouped: dict[int, list[dict]] = defaultdict(list)

    for row in rows:
        target_id = int(row["target_id"])
        sample_index = int(row["sample_index"])
        target_path = data / row["target_image"]
        render_path = renders / f"{target_id:07d}_s{sample_index:02d}.png"
        if not render_path.exists():
            raise FileNotFoundError(render_path)

        metrics = score_pair(target_path, render_path)
        grouped[target_id].append(
            {
                "target_id": target_id,
                "sample_index": sample_index,
                **metrics,
            }
        )

    if not grouped:
        raise RuntimeError(f"No rows scored from {predictions}")
    return grouped


def best_record(records: list[dict]) -> dict:
    return max(records, key=lambda r: r["main_score"])


def mean(records: list[dict], key: str) -> float:
    return float(np.mean([float(r[key]) for r in records]))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    diffusion = score_prediction_set(
        args.data, args.diffusion_predictions, args.diffusion_renders
    )
    random_baseline = score_prediction_set(
        args.data, args.random_predictions, args.random_renders
    )
    gt = score_prediction_set(args.data, args.gt_predictions, args.gt_renders)

    target_ids = sorted(set(diffusion) & set(random_baseline) & set(gt))
    if not target_ids:
        raise RuntimeError("No common targets across diffusion/random/GT sets")

    per_target = []
    for target_id in target_ids:
        d = best_record(diffusion[target_id])
        r = best_record(random_baseline[target_id])
        g = best_record(gt[target_id])
        per_target.append(
            {
                "target_id": target_id,
                "k_diffusion": len(diffusion[target_id]),
                "k_random": len(random_baseline[target_id]),
                "diffusion_best_score": d["main_score"],
                "random_best_score": r["main_score"],
                "gt_score": g["main_score"],
                "diffusion_best_iou": d["mask_iou"],
                "random_best_iou": r["mask_iou"],
                "gt_iou": g["mask_iou"],
                "diffusion_best_ssim": d["ssim"],
                "random_best_ssim": r["ssim"],
                "gt_ssim": g["ssim"],
                "diffusion_minus_random": d["main_score"] - r["main_score"],
                "diffusion_beats_random": int(d["main_score"] > r["main_score"]),
            }
        )

    csv_path = args.out / "benchmark_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_target[0].keys()))
        writer.writeheader()
        writer.writerows(per_target)

    summary = {
        "targets": len(per_target),
        "k_diffusion_mean": float(np.mean([r["k_diffusion"] for r in per_target])),
        "k_random_mean": float(np.mean([r["k_random"] for r in per_target])),
        "gt_rerender": {
            "main_score_mean": mean(per_target, "gt_score"),
            "mask_iou_mean": mean(per_target, "gt_iou"),
            "ssim_mean": mean(per_target, "gt_ssim"),
        },
        "diffusion_best_of_k": {
            "main_score_mean": mean(per_target, "diffusion_best_score"),
            "mask_iou_mean": mean(per_target, "diffusion_best_iou"),
            "ssim_mean": mean(per_target, "diffusion_best_ssim"),
        },
        "random_best_of_k": {
            "main_score_mean": mean(per_target, "random_best_score"),
            "mask_iou_mean": mean(per_target, "random_best_iou"),
            "ssim_mean": mean(per_target, "random_best_ssim"),
        },
        "diffusion_vs_random": {
            "mean_score_advantage": mean(per_target, "diffusion_minus_random"),
            "fraction_targets_diffusion_wins": mean(per_target, "diffusion_beats_random"),
        },
        "interpretation": (
            "The key test is whether diffusion Best-of-K clearly exceeds a random valid-state "
            "Best-of-K drawn from the same training distribution. GT rerender should be near 1.0 "
            "and acts as a scorer/renderer sanity check."
        ),
    }

    summary_path = args.out / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Blender Recreation Benchmark ===")
    print(f"Targets: {summary['targets']}")
    print(f"GT rerender:       {summary['gt_rerender']['main_score_mean']:.4f}")
    print(
        f"Diffusion best@K:  {summary['diffusion_best_of_k']['main_score_mean']:.4f}"
    )
    print(f"Random best@K:     {summary['random_best_of_k']['main_score_mean']:.4f}")
    print(
        "Diffusion - random: "
        f"{summary['diffusion_vs_random']['mean_score_advantage']:+.4f}"
    )
    print(
        "Diffusion wins:     "
        f"{summary['diffusion_vs_random']['fraction_targets_diffusion_wins']:.1%} of targets"
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
