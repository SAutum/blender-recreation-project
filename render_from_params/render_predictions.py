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
    p.add_argument("--samples", type=int, default=16)
    return p.parse_args(argv)


def render_prediction(
    scene_spec: dict,
    path: Path,
    width: int,
    height: int,
    samples: int,
) -> None:
    # Deliberately do not fit or correct the predicted camera.
    build_scene(
        scene_spec,
        width,
        height,
        samples=samples,
        fit_camera=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for i, row in enumerate(rows):
        target_id = int(row["target_id"])
        sample_index = int(row["sample_index"])
        out_path = args.out / f"{target_id:07d}_s{sample_index:02d}.png"
        render_prediction(
            row["pred_scene"],
            out_path,
            args.width,
            args.height,
            args.samples,
        )
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Rendered {i + 1}/{len(rows)} predictions")

    print(f"Prediction renders written to {args.out}")


if __name__ == "__main__":
    main()
