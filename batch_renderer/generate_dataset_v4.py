from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_renderer.generate_dataset import assert_cycles_active
from batch_renderer.generate_dataset_v3 import (
    _fully_in_frame,
    _scene_projected_extent,
    build_scene,
    look_at_rotation,
    sample_scene,
    split_for_index,
)
from br_scene_state_v3 import (
    STATE_DIM,
    camera_location_from_parameters,
    encode_scene,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=5000, help="Number of scenes/pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-scene-attempts", type=int, default=100)
    p.add_argument(
        "--stereo-baseline",
        type=float,
        default=0.35,
        help=(
            "Left-to-right camera separation in Blender units. Default 0.35 BU is "
            "roughly equivalent to a 4 degree separation at a 5 BU viewing distance. "
            "Use 0.063 BU for an approximately 63 mm human IPD if 1 BU = 1 m."
        ),
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _make_right_view(left_spec: dict, stereo_baseline: float) -> dict:
    """Translate the second camera laterally by a fixed binocular baseline.

    The right camera still converges on the exact same target as the anchor camera.
    We convert the translated camera location back into v3's legal spherical camera
    parameters so rendering remains fully compatible with the v3 state model.
    """
    right = copy.deepcopy(left_spec)
    cam = right["camera"]
    target = Vector(cam["target"])

    left_location = Vector(
        camera_location_from_parameters(
            cam["azimuth"],
            cam["elevation"],
            cam["distance"],
            cam["target"],
        )
    )

    # Camera-local +X is the natural left/right eye axis. Including roll here keeps
    # the stereo translation aligned with the actual camera frame rather than a
    # fixed world-space horizontal direction.
    rotation = look_at_rotation(left_location, target, cam["roll"])
    camera_right_axis = rotation.to_matrix() @ Vector((1.0, 0.0, 0.0))
    if camera_right_axis.length < 1e-8:
        raise RuntimeError("Could not determine stereo camera right axis")
    camera_right_axis.normalize()

    right_location = left_location + camera_right_axis * float(stereo_baseline)
    outward = right_location - target
    distance = outward.length
    if distance < 1e-8:
        raise RuntimeError("Stereo companion camera collapsed onto target")

    azimuth = math.atan2(outward.y, outward.x)
    elevation = math.asin(max(-1.0, min(1.0, outward.z / distance)))

    cam["azimuth"] = float(azimuth)
    cam["elevation"] = float(elevation)
    cam["distance"] = float(distance)
    cam["location"] = [float(v) for v in right_location]
    # target, roll and focal length intentionally remain unchanged.
    return right


def _validate_view(spec: dict, width: int, height: int, samples: int) -> None:
    objects, camera_obj = build_scene(
        spec,
        width,
        height,
        samples=samples,
        fit_camera=False,
    )
    bpy.context.view_layer.update()
    if not all(_fully_in_frame(obj, camera_obj) for obj in objects):
        raise RuntimeError("Stereo companion view puts an object outside the camera frame")
    projected_w, projected_h = _scene_projected_extent(objects, camera_obj)
    if max(projected_w, projected_h) < 0.22:
        raise RuntimeError("Stereo companion view is too small in the camera frame")


def _read_completed_rows(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        return []
    rows: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cannot resume: invalid JSON in {metadata_path} at line {line_no}"
                ) from exc
    return rows


def _resume_start_index(metadata_path: Path, image_dir: Path, count: int) -> int:
    rows = _read_completed_rows(metadata_path)
    if not rows:
        return 0

    ids = [int(row["id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise RuntimeError("Cannot resume safely: metadata ids are not contiguous from 0")
    if len(rows) > count:
        raise RuntimeError(
            f"Existing dataset already has {len(rows)} rows, more than requested --count {count}"
        )

    last_id = ids[-1]
    for suffix in ("left", "right"):
        image_path = image_dir / f"{last_id:07d}_{suffix}.png"
        if not image_path.exists():
            raise RuntimeError(
                f"Cannot resume safely: metadata says sample {last_id} is complete but "
                f"{image_path} is missing"
            )
    return len(rows)


def _rng_for_sample(seed: int, idx: int) -> random.Random:
    mixed = (int(seed) * 1_000_003 + int(idx) * 97_409 + 0x6A09E667) & 0xFFFFFFFFFFFFFFFF
    return random.Random(mixed)


def _render(spec: dict, output_path: Path, width: int, height: int, samples: int) -> None:
    build_scene(spec, width, height, samples=samples, fit_camera=False)
    bpy.context.scene.render.filepath = str(output_path)
    assert_cycles_active()
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.stereo_baseline <= 1.0:
        raise ValueError("--stereo-baseline must be > 0 and <= 1.0 Blender units")

    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out
    args.out = args.out.resolve()
    print(f"[blender-recreation] v4 paired-view dataset output: {args.out}")
    print(
        f"[blender-recreation] Stereo baseline: {args.stereo_baseline:.4f} BU "
        "(use 0.063 for approximate human IPD at 1 BU = 1 m)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.jsonl"

    if args.resume:
        start_idx = _resume_start_index(metadata_path, image_dir, args.count)
        mode = "a"
        print(
            f"[blender-recreation] Resume enabled: {start_idx}/{args.count} "
            "scene pairs already complete"
        )
    else:
        start_idx = 0
        mode = "w"
        if metadata_path.exists():
            print("[blender-recreation] Resume disabled: existing metadata will be overwritten")

    if start_idx < args.count:
        with metadata_path.open(mode, encoding="utf-8") as f:
            for idx in range(start_idx, args.count):
                rng = _rng_for_sample(args.seed, idx)
                left_spec: dict | None = None
                right_spec: dict | None = None
                last_error: Exception | None = None

                for _attempt in range(args.max_scene_attempts):
                    try:
                        candidate = sample_scene(rng)

                        # The left image is the anchor/benchmark view and therefore defines
                        # the target Blender state. Fit it exactly as in v3 first.
                        build_scene(
                            candidate,
                            args.width,
                            args.height,
                            samples=args.samples,
                            fit_camera=True,
                        )

                        companion = _make_right_view(candidate, args.stereo_baseline)
                        _validate_view(
                            companion,
                            args.width,
                            args.height,
                            args.samples,
                        )
                        left_spec = candidate
                        right_spec = companion
                        last_error = None
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        left_spec = None
                        right_spec = None

                if left_spec is None or right_spec is None:
                    raise RuntimeError(
                        f"Failed to generate valid v4 stereo pair for sample {idx} after "
                        f"{args.max_scene_attempts} attempts"
                    ) from last_error

                left_rel = Path("images") / f"{idx:07d}_left.png"
                right_rel = Path("images") / f"{idx:07d}_right.png"
                _render(
                    left_spec,
                    args.out / left_rel,
                    args.width,
                    args.height,
                    args.samples,
                )
                _render(
                    right_spec,
                    args.out / right_rel,
                    args.width,
                    args.height,
                    args.samples,
                )

                row = {
                    "id": idx,
                    # Keep `image` as the anchor view so all existing Blender scoring /
                    # random-baseline code still compares against the exact target state.
                    "image": left_rel.as_posix(),
                    "images": [left_rel.as_posix(), right_rel.as_posix()],
                    "split": split_for_index(idx, args.count),
                    "scene": left_spec,
                    "state": encode_scene(left_spec).tolist(),
                    "stereo": {
                        "baseline": float(args.stereo_baseline),
                        "right_camera": right_spec["camera"],
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                if (idx + 1) % 100 == 0 or idx == start_idx:
                    print(
                        f"Rendered {idx + 1}/{args.count} scene pairs "
                        f"({2 * (idx + 1)} images)"
                    )

    config = {
        "dataset_version": 4,
        "scene_state_version": 3,
        "count_scenes": args.count,
        "count_images": args.count * 2,
        "views_per_scene": 2,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "engine": "CYCLES",
        "samples": args.samples,
        "lighting": "fixed Blender-startup-style point light + fixed world",
        "film_transparent": True,
        "state_dim": STATE_DIM,
        "objects_per_scene": [1, 2],
        "primitive_types": ["cube", "sphere", "cylinder"],
        "conditioning": "two RGB views concatenated channel-wise (6 channels)",
        "anchor_view": "left image exactly matches the target v3 Blender state",
        "stereo_baseline": float(args.stereo_baseline),
        "stereo_relation": (
            "right camera is translated along anchor-camera local X by a fixed baseline; "
            "both views converge on the same target with matching roll and focal length"
        ),
        "human_ipd_reference_bu": 0.063,
        "default_baseline_note": (
            "0.35 BU is intentionally exaggerated for stronger disparity; it is roughly "
            "equivalent to a 4 degree separation at a 5 BU viewing distance"
        ),
        "resumable_generation": True,
    }
    (args.out / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"v4 paired-view dataset written to {args.out}")


if __name__ == "__main__":
    main()
