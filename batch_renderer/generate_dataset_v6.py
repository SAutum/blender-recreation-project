from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_renderer.generate_dataset import assert_cycles_active
from batch_renderer.generate_dataset_v3 import (
    build_scene,
    look_at_rotation,
    sample_objects,
    sort_objects_by_screen_position,
    split_for_index,
)
from br_scene_state_v3 import (
    CAMERA_DISTANCE_MAX,
    CAMERA_ELEVATION_MAX_RAD,
    CAMERA_ROLL_MAX_RAD,
    FOCAL_MAX_MM,
    FOCAL_MIN_MM,
    STATE_DIM,
    TARGET_OFFSET_MAX,
    camera_location_from_parameters,
    encode_scene,
    object_centroid,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=5000, help="Number of scene pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-scene-attempts", type=int, default=100)
    p.add_argument(
        "--stereo-baseline",
        type=float,
        default=0.35,
        help="Left-to-right camera separation in Blender units.",
    )
    p.add_argument(
        "--camera-distance-min",
        type=float,
        default=3.0,
        help="Independent camera-distance sampling lower bound.",
    )
    p.add_argument(
        "--camera-distance-max",
        type=float,
        default=10.5,
        help="Independent camera-distance sampling upper bound.",
    )
    p.add_argument(
        "--min-visible-area",
        type=float,
        default=0.005,
        help="Minimum clipped projected bbox area in normalized image coordinates.",
    )
    p.add_argument(
        "--min-visible-span",
        type=float,
        default=0.10,
        help="Minimum clipped projected bbox width or height in normalized image coordinates.",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _sample_unframed_camera(rng: random.Random, objects: list[dict], dmin: float, dmax: float) -> dict:
    """Sample camera independently of scene radius / object size.

    Unlike v3/v4, distance is not derived from object extent or focal length. This
    intentionally preserves image-size variation so scale, distance and focal length
    remain visible cues instead of being normalized away by automatic framing.
    """
    center = Vector(object_centroid(objects))
    azimuth = rng.uniform(-math.pi, math.pi)
    elevation = rng.uniform(-CAMERA_ELEVATION_MAX_RAD * 0.90, CAMERA_ELEVATION_MAX_RAD * 0.90)
    distance = rng.uniform(dmin, dmax)
    roll = rng.uniform(-CAMERA_ROLL_MAX_RAD * 0.75, CAMERA_ROLL_MAX_RAD * 0.75)
    lens_mm = rng.uniform(FOCAL_MIN_MM, FOCAL_MAX_MM)

    target = center + Vector(
        (
            rng.uniform(-TARGET_OFFSET_MAX * 0.55, TARGET_OFFSET_MAX * 0.55),
            rng.uniform(-TARGET_OFFSET_MAX * 0.55, TARGET_OFFSET_MAX * 0.55),
            rng.uniform(-TARGET_OFFSET_MAX * 0.55, TARGET_OFFSET_MAX * 0.55),
        )
    )

    return {
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "distance": float(distance),
        "target": [float(v) for v in target],
        "roll": float(roll),
        "lens_mm": float(lens_mm),
        "location": camera_location_from_parameters(azimuth, elevation, distance, target),
    }


def _sample_unframed_scene(rng: random.Random, dmin: float, dmax: float) -> dict:
    num_objects = rng.choice([1, 2])
    objects = sample_objects(rng, num_objects)
    return {
        "version": 3,
        "camera": _sample_unframed_camera(rng, objects, dmin, dmax),
        "objects": objects,
    }


def _make_right_view(left_spec: dict, stereo_baseline: float) -> dict:
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
    if distance > CAMERA_DISTANCE_MAX:
        raise RuntimeError("Stereo companion camera exceeds encoded distance range")

    cam["azimuth"] = float(math.atan2(outward.y, outward.x))
    cam["elevation"] = float(math.asin(max(-1.0, min(1.0, outward.z / distance))))
    cam["distance"] = float(distance)
    cam["location"] = [float(v) for v in right_location]
    return right


def _projection_stats(objects, camera_obj) -> dict:
    scene = bpy.context.scene
    xs: list[float] = []
    ys: list[float] = []
    for obj in objects:
        for local_corner in obj.bound_box:
            p = world_to_camera_view(scene, camera_obj, obj.matrix_world @ Vector(local_corner))
            if p.z > 0:
                xs.append(float(p.x))
                ys.append(float(p.y))

    if not xs:
        return {
            "full_width": 0.0,
            "full_height": 0.0,
            "visible_width": 0.0,
            "visible_height": 0.0,
            "visible_area": 0.0,
        }

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    visible_width = max(0.0, min(max_x, 1.0) - max(min_x, 0.0))
    visible_height = max(0.0, min(max_y, 1.0) - max(min_y, 0.0))
    return {
        "full_width": float(max_x - min_x),
        "full_height": float(max_y - min_y),
        "visible_width": float(visible_width),
        "visible_height": float(visible_height),
        "visible_area": float(visible_width * visible_height),
    }


def _validate_relaxed_view(
    spec: dict,
    width: int,
    height: int,
    samples: int,
    min_visible_area: float,
    min_visible_span: float,
    sort_slots: bool = False,
) -> dict:
    objects, camera_obj = build_scene(
        spec,
        width,
        height,
        samples=samples,
        fit_camera=False,
    )
    bpy.context.view_layer.update()
    stats = _projection_stats(objects, camera_obj)

    # This is intentionally only a visibility guard, not a framing operation.
    # Large scenes may be cropped and small scenes remain small. We merely reject
    # scenes that are effectively absent from the image.
    visible_span = max(stats["visible_width"], stats["visible_height"])
    if stats["visible_area"] < min_visible_area or visible_span < min_visible_span:
        raise RuntimeError(
            "Scene is not sufficiently visible without auto-framing: "
            f"area={stats['visible_area']:.4f}, span={visible_span:.4f}"
        )

    if sort_slots:
        sort_objects_by_screen_position(spec, objects, camera_obj)
    return stats


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
    mixed = (int(seed) * 1_000_003 + int(idx) * 97_409 + 0x243F6A88) & 0xFFFFFFFFFFFFFFFF
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
    if not 2.5 <= args.camera_distance_min < args.camera_distance_max <= CAMERA_DISTANCE_MAX:
        raise ValueError(
            "Camera distance range must satisfy 2.5 <= min < max <= "
            f"{CAMERA_DISTANCE_MAX}"
        )
    if args.min_visible_area <= 0 or args.min_visible_span <= 0:
        raise ValueError("Visibility thresholds must be positive")

    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out
    args.out = args.out.resolve()

    print(f"[blender-recreation] v6 no-framing paired dataset output: {args.out}")
    print(
        "[blender-recreation] Camera distance sampled independently: "
        f"{args.camera_distance_min:.2f}..{args.camera_distance_max:.2f} BU"
    )
    print(
        "[blender-recreation] No automatic camera fitting; only relaxed visibility rejection is used"
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
                left_stats: dict | None = None
                right_stats: dict | None = None
                last_error: Exception | None = None

                for _attempt in range(args.max_scene_attempts):
                    try:
                        candidate = _sample_unframed_scene(
                            rng,
                            args.camera_distance_min,
                            args.camera_distance_max,
                        )
                        left_projection = _validate_relaxed_view(
                            candidate,
                            args.width,
                            args.height,
                            args.samples,
                            args.min_visible_area,
                            args.min_visible_span,
                            sort_slots=True,
                        )

                        companion = _make_right_view(candidate, args.stereo_baseline)
                        right_projection = _validate_relaxed_view(
                            companion,
                            args.width,
                            args.height,
                            args.samples,
                            args.min_visible_area,
                            args.min_visible_span,
                            sort_slots=False,
                        )

                        left_spec = candidate
                        right_spec = companion
                        left_stats = left_projection
                        right_stats = right_projection
                        last_error = None
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        left_spec = None
                        right_spec = None

                if left_spec is None or right_spec is None:
                    raise RuntimeError(
                        f"Failed to generate visible v6 pair for sample {idx} after "
                        f"{args.max_scene_attempts} attempts"
                    ) from last_error

                left_rel = Path("images") / f"{idx:07d}_left.png"
                right_rel = Path("images") / f"{idx:07d}_right.png"
                _render(left_spec, args.out / left_rel, args.width, args.height, args.samples)
                _render(right_spec, args.out / right_rel, args.width, args.height, args.samples)

                row = {
                    "id": idx,
                    "image": left_rel.as_posix(),
                    "images": [left_rel.as_posix(), right_rel.as_posix()],
                    "split": split_for_index(idx, args.count),
                    "scene": left_spec,
                    "state": encode_scene(left_spec).tolist(),
                    "stereo": {
                        "baseline": float(args.stereo_baseline),
                        "right_camera": right_spec["camera"],
                    },
                    "framing": {
                        "mode": "none",
                        "camera_distance_sampled_independently": True,
                        "left_projection": left_stats,
                        "right_projection": right_stats,
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
        "dataset_version": 6,
        "scene_state_version": 3,
        "count_scenes": args.count,
        "count_images": args.count * 2,
        "views_per_scene": 2,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "engine": "CYCLES",
        "samples": args.samples,
        "state_dim": STATE_DIM,
        "objects_per_scene": [1, 2],
        "primitive_types": ["cube", "sphere", "cylinder"],
        "conditioning": "paired RGB views; train with view_mode mono=(left,left) or stereo=(left,right)",
        "anchor_view": "left image exactly matches the target v3 Blender state",
        "stereo_baseline": float(args.stereo_baseline),
        "framing": "none; camera distance and focal length are sampled independently of object extent",
        "camera_distance_range": [args.camera_distance_min, args.camera_distance_max],
        "focal_range_mm": [FOCAL_MIN_MM, FOCAL_MAX_MM],
        "visibility_guard": {
            "min_visible_area": args.min_visible_area,
            "min_visible_span": args.min_visible_span,
            "note": "reject only effectively absent scenes; no camera fitting or distance correction",
        },
        "resumable_generation": True,
    }
    (args.out / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"v6 no-framing paired dataset written to {args.out}")


if __name__ == "__main__":
    main()
