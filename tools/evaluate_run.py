from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full post-training evaluation pipeline: infer -> render diffusion -> "
            "make baselines -> render random/GT -> benchmark."
        )
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--samples-per-image", type=int, default=8)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--render-samples", type=int, default=16)
    p.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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

    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = run_dir / "predictions.jsonl"
    pred_renders = run_dir / "pred_renders"
    random_predictions = run_dir / "random_predictions.jsonl"
    gt_predictions = run_dir / "gt_predictions.jsonl"
    random_renders = run_dir / "random_renders"
    gt_renders = run_dir / "gt_renders"

    print("=== Post-training evaluation ===")
    print(f"Data:       {data}")
    print(f"Run:        {run_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Targets:    {args.limit}")
    print(f"Best-of-K:  {args.samples_per_image}")

    # 1) Diffusion inference.
    run(
        [
            sys.executable,
            "-m",
            "models.infer",
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(predictions),
            "--split",
            args.split,
            "--samples-per-image",
            str(args.samples_per_image),
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
        ]
    )

    # 2) Render diffusion predictions in Blender.
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

    # 3) Create random-valid-state and GT baselines with matching K/targets.
    run(
        [
            sys.executable,
            "-m",
            "models.make_baselines",
            "--data",
            str(data),
            "--targets-from",
            str(predictions),
            "--out",
            str(run_dir),
        ]
    )

    # 4) Render random baseline.
    run(
        [
            str(blender),
            "--background",
            "--python",
            "render_from_params/render_predictions.py",
            "--",
            "--predictions",
            str(random_predictions),
            "--out",
            str(random_renders),
            "--samples",
            str(args.render_samples),
        ]
    )

    # 5) Render GT rerender sanity check.
    run(
        [
            str(blender),
            "--background",
            "--python",
            "render_from_params/render_predictions.py",
            "--",
            "--predictions",
            str(gt_predictions),
            "--out",
            str(gt_renders),
            "--samples",
            str(args.render_samples),
        ]
    )

    # 6) Benchmark Diffusion Best@K vs Random Best@K vs GT rerender.
    run(
        [
            sys.executable,
            "-m",
            "models.benchmark",
            "--data",
            str(data),
            "--diffusion-predictions",
            str(predictions),
            "--diffusion-renders",
            str(pred_renders),
            "--random-predictions",
            str(random_predictions),
            "--random-renders",
            str(random_renders),
            "--gt-predictions",
            str(gt_predictions),
            "--gt-renders",
            str(gt_renders),
            "--out",
            str(run_dir),
        ]
    )

    print("\n=== Evaluation complete ===")
    print(run_dir / "benchmark_summary.json")
    print(run_dir / "benchmark_metrics.csv")


if __name__ == "__main__":
    main()
