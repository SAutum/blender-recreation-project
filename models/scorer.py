from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from br_scene_state import parameter_errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def composite_gray(rgba: np.ndarray, gray: float = 32.0 / 255.0) -> np.ndarray:
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    return rgb * a + gray * (1.0 - a)


def mask_iou(a_rgba: np.ndarray, b_rgba: np.ndarray) -> float:
    a = a_rgba[..., 3] > 0.5
    b = b_rgba[..., 3] > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def score_pair(target_path: Path, pred_path: Path) -> dict:
    target = load_rgba(target_path)
    pred = load_rgba(pred_path)
    if target.shape != pred.shape:
        pred_img = Image.open(pred_path).convert("RGBA").resize((target.shape[1], target.shape[0]))
        pred = np.asarray(pred_img, dtype=np.float32) / 255.0

    iou = mask_iou(target, pred)
    target_rgb = composite_gray(target)
    pred_rgb = composite_gray(pred)
    ssim = float(structural_similarity(target_rgb, pred_rgb, channel_axis=2, data_range=1.0))
    ssim01 = float(np.clip((ssim + 1.0) * 0.5, 0.0, 1.0))
    mse = float(np.mean((target_rgb - pred_rgb) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))
    main_score = 0.70 * iou + 0.30 * ssim01
    return {
        "main_score": main_score,
        "mask_iou": iou,
        "ssim": ssim,
        "mse": mse,
        "psnr": psnr,
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]

    metrics = []
    grouped = defaultdict(list)
    for row in rows:
        target_id = int(row["target_id"])
        sample_index = int(row["sample_index"])
        target_path = args.data / row["target_image"]
        pred_path = args.renders / f"{target_id:07d}_s{sample_index:02d}.png"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)

        image_metrics = score_pair(target_path, pred_path)
        param_metrics = parameter_errors(row["target_scene"], row["pred_scene"])
        record = {
            "target_id": target_id,
            "sample_index": sample_index,
            **image_metrics,
            **param_metrics,
            "target_image": str(target_path),
            "pred_image": str(pred_path),
        }
        metrics.append(record)
        grouped[target_id].append(record)

    if not metrics:
        raise RuntimeError("No prediction rows were scored")

    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    best = [max(group, key=lambda x: x["main_score"]) for group in grouped.values()]
    summary = {
        "targets": len(grouped),
        "samples_total": len(metrics),
        "samples_per_target_mean": float(np.mean([len(v) for v in grouped.values()])),
        "best_of_k": {
            "main_score_mean": float(np.mean([x["main_score"] for x in best])),
            "main_score_median": float(np.median([x["main_score"] for x in best])),
            "mask_iou_mean": float(np.mean([x["mask_iou"] for x in best])),
            "ssim_mean": float(np.mean([x["ssim"] for x in best])),
            "shape_accuracy": float(np.mean([x["shape_correct"] for x in best])),
            "camera_l2_mean": float(np.mean([x["camera_l2"] for x in best])),
            "geometry_mae_mean": float(np.mean([x["geometry_mae"] for x in best])),
        },
        "all_samples": {
            "main_score_mean": float(np.mean([x["main_score"] for x in metrics])),
            "mask_iou_mean": float(np.mean([x["mask_iou"] for x in metrics])),
            "ssim_mean": float(np.mean([x["ssim"] for x in metrics])),
        },
        "primary_metric_note": "Best-of-K image-space reconstruction score is primary; parameter errors are secondary diagnostics.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
