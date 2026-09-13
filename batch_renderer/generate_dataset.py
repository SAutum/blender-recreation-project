from __future__ import annotations

import argparse
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

from br_scene_state import CAMERA_LENS_MM, encode_scene, safe_camera_distance


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--samples", type=int, default=16, help="Cycles samples per render")
    return p.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def setup_render(width: int, height: int, samples: int = 16) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.seed = 0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def assert_cycles_active() -> None:
    engine = bpy.context.scene.render.engine
    print(f"[blender-recreation] Active render engine: {engine}")
    if engine != "CYCLES":
        raise RuntimeError(f"Expected CYCLES, but Blender reports {engine!r}")


def setup_fixed_lighting() -> None:
    """Fixed Blender-startup-style point light; never randomized in v1."""
    light_data = bpy.data.lights.new(name="FixedLight", type="POINT")
    light_data.energy = 1000.0
    light_obj = bpy.data.objects.new(name="FixedLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (4.076, -1.005, 5.904)


def make_material():
    mat = bpy.data.materials.new("PrimitiveMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.72, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.48
    return mat


def create_shape(spec: dict):
    shape = spec["shape"]
    g = spec["geometry"]

    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        obj = bpy.context.object
        obj.dimensions = (g["size_x"], g["size_y"], g["size_z"])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=g["radius"],
            location=(0, 0, 0),
        )
        obj = bpy.context.object
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=g["radius"],
            depth=g["depth"],
            location=(0, 0, 0),
        )
        obj = bpy.context.object
    else:
        raise ValueError(f"Unknown shape: {shape}")

    obj.name = "TargetPrimitive"
    obj.data.materials.append(make_material())
    return obj


def point_camera_at(camera_obj, target=(0.0, 0.0, 0.0)) -> None:
    direction = Vector(target) - camera_obj.location
    if direction.length < 1e-8:
        direction = Vector((0.0, 0.0, -1.0))
    camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera(spec: dict):
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = float(spec["camera"].get("lens_mm", CAMERA_LENS_MM))
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.01
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.location = spec["camera"]["location"]
    point_camera_at(camera_obj)
    bpy.context.scene.camera = camera_obj
    return camera_obj


def _fully_in_frame(obj, camera_obj, border: float = 0.025) -> bool:
    scene = bpy.context.scene
    for local_corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(local_corner)
        p = world_to_camera_view(scene, camera_obj, world_corner)
        if (
            p.z <= 0
            or p.x < border
            or p.x > 1.0 - border
            or p.y < border
            or p.y > 1.0 - border
        ):
            return False
    return True


def ensure_in_frame(spec: dict, obj, camera_obj) -> None:
    for _ in range(30):
        bpy.context.view_layer.update()
        if _fully_in_frame(obj, camera_obj):
            spec["camera"]["location"] = [float(v) for v in camera_obj.location]
            return
        camera_obj.location *= 1.06
        point_camera_at(camera_obj)
    raise RuntimeError("Could not place generated primitive fully inside the camera frame")


def sample_scene(rng: random.Random) -> dict:
    shape = rng.choice(["cube", "sphere", "cylinder"])

    if shape == "cube":
        geometry = {
            "size_x": rng.uniform(0.50, 1.80),
            "size_y": rng.uniform(0.50, 1.80),
            "size_z": rng.uniform(0.50, 1.80),
        }
    elif shape == "sphere":
        geometry = {"radius": rng.uniform(0.35, 1.20)}
    else:
        geometry = {
            "radius": rng.uniform(0.35, 1.10),
            "depth": rng.uniform(0.50, 2.00),
        }

    azimuth = rng.uniform(-math.pi, math.pi)
    elevation = math.radians(rng.uniform(-65.0, 65.0))
    distance = safe_camera_distance(shape, geometry, margin=1.18) * rng.uniform(1.00, 1.18)

    ce = math.cos(elevation)
    location = [
        distance * ce * math.cos(azimuth),
        distance * ce * math.sin(azimuth),
        distance * math.sin(elevation),
    ]

    return {
        "shape": shape,
        "geometry": geometry,
        "camera": {"location": location, "lens_mm": CAMERA_LENS_MM},
    }


def choose_split(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.90:
        return "train"
    if x < 0.95:
        return "val"
    return "test"


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
    obj = create_shape(spec)
    cam = create_camera(spec)
    if fit_camera:
        ensure_in_frame(spec, obj, cam)
    return obj, cam


def render_spec(
    spec: dict,
    output_path: Path,
    width: int = 128,
    height: int = 128,
    samples: int = 16,
) -> None:
    build_scene(spec, width, height, samples=samples, fit_camera=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    assert_cycles_active()
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out
    args.out = args.out.resolve()
    print(f"[blender-recreation] Dataset output: {args.out}")

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for idx in range(args.count):
            spec = sample_scene(rng)
            build_scene(
                spec,
                args.width,
                args.height,
                samples=args.samples,
                fit_camera=True,
            )

            image_rel = Path("images") / f"{idx:07d}.png"
            image_abs = args.out / image_rel
            bpy.context.scene.render.filepath = str(image_abs)
            assert_cycles_active()
            bpy.ops.render.render(write_still=True)

            row = {
                "id": idx,
                "image": image_rel.as_posix(),
                "split": choose_split(rng),
                "scene": spec,
                "state": encode_scene(spec).tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"Rendered {idx + 1}/{args.count}")

    config = {
        "count": args.count,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "engine": "CYCLES",
        "samples": args.samples,
        "lighting": "fixed Blender-startup-style point light + fixed world",
        "film_transparent": True,
        "state_dim": 9,
    }
    (args.out / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Dataset written to {args.out}")


if __name__ == "__main__":
    main()
