from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_renderer.generate_dataset import build_scene


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    return p.parse_args(argv)


def render_prediction(scene_spec: dict, path: Path, width: int, height: int) -> None:
    # Important: unlike synthetic data generation, do NOT auto-fit a predicted
    # camera. The predicted state must be rendered exactly as predicted so the
    # image-space scorer can penalize bad camera estimates.
    from batch_renderer.generate_dataset import clear_scene, setup_render, setup_fixed_lighting, create_shape, create_camera

    clear_scene()
    setup_render(width, height)
    setup_fixed_lighting()
    create_shape(scene_spec)
    create_camera(scene_spec)

    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        target_id = int(row["target_id"])
        sample_index = int(row["sample_index"])
        out_path = args.out / f"{target_id:07d}_s{sample_index:02d}.png"
        render_prediction(row["pred_scene"], out_path, args.width, args.height)
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Rendered {i + 1}/{len(rows)} predictions")

    print(f"Prediction renders written to {args.out}")


if __name__ == "__main__":
    main()
