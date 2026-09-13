from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_renderer.generate_dataset import (
    assert_cycles_active,
    clear_scene,
    make_material,
    setup_fixed_lighting,
    setup_render,
)
from br_scene_state_v3 import (
    CAMERA_DISTANCE_MAX,
    CAMERA_ELEVATION_MAX_RAD,
    CAMERA_ROLL_MAX_RAD,
    CAMERA_SENSOR_WIDTH_MM,
    FOCAL_MAX_MM,
    FOCAL_MIN_MM,
    STATE_DIM,
    TARGET_OFFSET_MAX,
    bounding_radius,
    camera_location_from_parameters,
    encode_scene,
    object_centroid,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max-scene-attempts", type=int, default=100)
    return p.parse_args(argv)


def sample_geometry(rng: random.Random, shape: str) -> dict:
    if shape == "cube":
        return {
            "size_x": rng.uniform(0.40, 1.80),
            "size_y": rng.uniform(0.40, 1.80),
            "size_z": rng.uniform(0.40, 1.80),
        }
    if shape == "sphere":
        return {"radius": rng.uniform(0.30, 1.20)}
    if shape == "cylinder":
        return {
            "radius": rng.uniform(0.30, 1.00),
            "depth": rng.uniform(0.50, 2.00),
        }
    raise ValueError(shape)


def sample_object(rng: random.Random) -> dict:
    shape = rng.choice(["cube", "sphere", "cylinder"])
    return {
        "shape": shape,
        "geometry": sample_geometry(rng, shape),
        "location": [
            rng.uniform(-0.95, 0.95),
            rng.uniform(-0.95, 0.95),
            rng.uniform(-0.55, 0.55),
        ],
        "rotation_euler": [rng.uniform(-math.pi, math.pi) for _ in range(3)],
    }


def sample_objects(rng: random.Random, num_objects: int) -> list[dict]:
    first = sample_object(rng)
    objects = [first]
    if num_objects == 1:
        return objects

    first_pos = Vector(first["location"])
    first_r = bounding_radius(first["shape"], first["geometry"])
    for _ in range(100):
        second = sample_object(rng)
        second_pos = Vector(second["location"])
        second_r = bounding_radius(second["shape"], second["geometry"])
        if (second_pos - first_pos).length >= 0.70 * (first_r + second_r):
            objects.append(second)
            return objects
    raise RuntimeError("Could not sample two sufficiently separated primitives")


def scene_center_radius(objects: list[dict]) -> tuple[Vector, float]:
    center = Vector(object_centroid(objects))
    radius = 0.0
    for obj in objects:
        r = bounding_radius(obj["shape"], obj["geometry"])
        radius = max(radius, (Vector(obj["location"]) - center).length + r)
    return center, max(radius, 0.25)


def sample_camera(rng: random.Random, objects: list[dict]) -> dict:
    center, scene_radius = scene_center_radius(objects)
    lens_mm = rng.uniform(FOCAL_MIN_MM, FOCAL_MAX_MM)
    half_fov = math.atan(CAMERA_SENSOR_WIDTH_MM / (2.0 * lens_mm))
    safe_distance = scene_radius / max(math.sin(half_fov), 1e-6) * 1.22
    distance = min(CAMERA_DISTANCE_MAX * 0.90, max(2.5, safe_distance) * rng.uniform(1.00, 1.15))

    azimuth = rng.uniform(-math.pi, math.pi)
    elevation = rng.uniform(-CAMERA_ELEVATION_MAX_RAD * 0.90, CAMERA_ELEVATION_MAX_RAD * 0.90)
    roll = rng.uniform(-CAMERA_ROLL_MAX_RAD * 0.75, CAMERA_ROLL_MAX_RAD * 0.75)

    # The target is deliberately tied to the actual object centroid. This is the key
    # v3 legality constraint: camera orientation can vary, but it cannot independently
    # drift away from the predicted/generated scene.
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


def sample_scene(rng: random.Random) -> dict:
    num_objects = rng.choice([1, 2])
    objects = sample_objects(rng, num_objects)
    return {
        "version": 3,
        "camera": sample_camera(rng, objects),
        "objects": objects,
    }


def create_object(spec: dict, index: int):
    shape = spec["shape"]
    g = spec["geometry"]
    location = spec["location"]
    rotation = spec["rotation_euler"]

    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
        obj = bpy.context.object
        obj.dimensions = (g["size_x"], g["size_y"], g["size_z"])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=g["radius"],
            location=location,
            rotation=rotation,
        )
        obj = bpy.context.object
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=g["radius"],
            depth=g["depth"],
            location=location,
            rotation=rotation,
        )
        obj = bpy.context.object
    else:
        raise ValueError(f"Unknown shape: {shape}")

    obj.name = f"TargetPrimitive_{index}"
    obj.data.materials.append(make_material())
    return obj


def look_at_rotation(location, target, roll_rad: float):
    direction = Vector(target) - Vector(location)
    if direction.length < 1e-8:
        direction = Vector((0.0, 0.0, -1.0))
    look_q = direction.to_track_quat("-Z", "Y")
    roll_q = Quaternion((0.0, 0.0, 1.0), float(roll_rad))
    return (look_q @ roll_q).to_euler("XYZ")


def create_camera(spec: dict):
    cam = spec["camera"]
    location = camera_location_from_parameters(
        cam["azimuth"], cam["elevation"], cam["distance"], cam["target"]
    )
    cam["location"] = location

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = float(cam["lens_mm"])
    camera_data.sensor_width = CAMERA_SENSOR_WIDTH_MM
    camera_data.clip_start = 0.01
    camera_data.clip_end = 1000.0
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.location = location
    camera_obj.rotation_mode = "XYZ"
    camera_obj.rotation_euler = look_at_rotation(location, cam["target"], cam["roll"])
    bpy.context.scene.camera = camera_obj
    return camera_obj


def _fully_in_frame(obj, camera_obj, border: float = 0.025) -> bool:
    scene = bpy.context.scene
    for local_corner in obj.bound_box:
        p = world_to_camera_view(scene, camera_obj, obj.matrix_world @ Vector(local_corner))
        if (
            p.z <= 0
            or p.x < border
            or p.x > 1.0 - border
            or p.y < border
            or p.y > 1.0 - border
        ):
            return False
    return True


def _scene_projected_extent(objects, camera_obj) -> tuple[float, float]:
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
        return 0.0, 0.0
    return max(xs) - min(xs), max(ys) - min(ys)


def refresh_camera_from_parameters(spec: dict, camera_obj) -> None:
    cam = spec["camera"]
    location = camera_location_from_parameters(
        cam["azimuth"], cam["elevation"], cam["distance"], cam["target"]
    )
    cam["location"] = location
    camera_obj.location = location
    camera_obj.rotation_euler = look_at_rotation(location, cam["target"], cam["roll"])


def ensure_scene_in_frame(spec: dict, objects, camera_obj) -> None:
    for _ in range(45):
        bpy.context.view_layer.update()
        if all(_fully_in_frame(obj, camera_obj) for obj in objects):
            width, height = _scene_projected_extent(objects, camera_obj)
            if max(width, height) < 0.22:
                raise RuntimeError("Generated scene is too small in the camera frame")
            return

        spec["camera"]["distance"] *= 1.07
        if spec["camera"]["distance"] > CAMERA_DISTANCE_MAX:
            raise RuntimeError("Required camera distance exceeds v3 encoding range")
        refresh_camera_from_parameters(spec, camera_obj)

    raise RuntimeError("Could not fit all v3 primitives inside the camera frame")


def sort_objects_by_screen_position(spec: dict, objects, camera_obj) -> None:
    scene = bpy.context.scene
    pairs = []
    for obj_spec, obj in zip(spec["objects"], objects):
        p = world_to_camera_view(scene, camera_obj, obj.matrix_world.translation)
        pairs.append((float(p.x), float(p.y), float(p.z), obj_spec))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    spec["objects"] = [item[3] for item in pairs]


def build_scene(
    spec: dict,
    width: int,
    height: int,
    samples: int = 16,
    fit_camera: bool = True,
):
    clear_scene()
    setup_render(width, height, samples)
    assert_cycles_active()
    setup_fixed_lighting()

    objects = [create_object(obj_spec, i) for i, obj_spec in enumerate(spec["objects"])]
    camera_obj = create_camera(spec)
    bpy.context.view_layer.update()

    if fit_camera:
        ensure_scene_in_frame(spec, objects, camera_obj)
        sort_objects_by_screen_position(spec, objects, camera_obj)

    return objects, camera_obj


def split_for_index(idx: int, count: int) -> str:
    if count < 20:
        return "train" if idx < max(1, count - 2) else ("val" if idx == count - 2 else "test")
    x = idx % 20
    if x == 18:
        return "val"
    if x == 19:
        return "test"
    return "train"


def main() -> None:
    args = parse_args()
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out
    args.out = args.out.resolve()
    print(f"[blender-recreation] v3 dataset output: {args.out}")

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for idx in range(args.count):
            last_error: Exception | None = None
            for _attempt in range(args.max_scene_attempts):
                spec = sample_scene(rng)
                try:
                    build_scene(
                        spec,
                        args.width,
                        args.height,
                        samples=args.samples,
                        fit_camera=True,
                    )
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError(
                    f"Failed to generate valid v3 scene for sample {idx} after "
                    f"{args.max_scene_attempts} attempts"
                ) from last_error

            image_rel = Path("images") / f"{idx:07d}.png"
            image_abs = args.out / image_rel
            bpy.context.scene.render.filepath = str(image_abs)
            assert_cycles_active()
            bpy.ops.render.render(write_still=True)

            row = {
                "id": idx,
                "image": image_rel.as_posix(),
                "split": split_for_index(idx, args.count),
                "scene": spec,
                "state": encode_scene(spec).tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"Rendered {idx + 1}/{args.count}")

    config = {
        "scene_version": 3,
        "count": args.count,
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
        "camera_parameterization": (
            "azimuth + elevation + distance + target_offset_from_object_centroid + roll + focal; "
            "Euler rotation is derived, never predicted"
        ),
        "camera_legality": "camera always looks at decoded scene centroid plus bounded target offset",
    }
    (args.out / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"v3 dataset written to {args.out}")


if __name__ == "__main__":
    main()
