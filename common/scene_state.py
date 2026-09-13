from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

SHAPES = ("cube", "sphere", "cylinder")
STATE_DIM = 9
CAMERA_COORD_MAX = 8.0
GEOM_MAX = 2.5
CAMERA_LENS_MM = 50.0
CAMERA_SENSOR_WIDTH_MM = 36.0


def _geom_triplet(scene: Dict) -> List[float]:
    shape = scene["shape"]
    g = scene["geometry"]
    if shape == "cube":
        return [float(g["size_x"]), float(g["size_y"]), float(g["size_z"])]
    if shape == "sphere":
        return [float(g["radius"]), 0.0, 0.0]
    if shape == "cylinder":
        return [float(g["radius"]), float(g["depth"]), 0.0]
    raise ValueError(f"Unknown shape: {shape}")


def encode_scene(scene: Dict) -> np.ndarray:
    """Encode a Blender-readable scene dict into a compact [-1, 1]-ish state."""
    state = np.full(STATE_DIM, -1.0, dtype=np.float32)
    state[SHAPES.index(scene["shape"])] = 1.0

    camera = np.asarray(scene["camera"]["location"], dtype=np.float32)
    state[3:6] = np.clip(camera / CAMERA_COORD_MAX, -1.0, 1.0)

    geom = np.asarray(_geom_triplet(scene), dtype=np.float32)
    state[6:9] = np.clip((2.0 * geom / GEOM_MAX) - 1.0, -1.0, 1.0)
    return state


def decode_state(state: Sequence[float]) -> Dict:
    """Decode a network state vector into Blender-readable parameters.

    Values are clamped to the supported v1 ranges. Inactive geometry fields are
    ignored according to the selected primitive class.
    """
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape ({STATE_DIM},), got {state.shape}")

    shape = SHAPES[int(np.argmax(state[:3]))]
    camera = np.clip(state[3:6], -1.0, 1.0) * CAMERA_COORD_MAX
    geom = (np.clip(state[6:9], -1.0, 1.0) + 1.0) * 0.5 * GEOM_MAX

    if shape == "cube":
        geometry = {
            "size_x": float(np.clip(geom[0], 0.35, 2.2)),
            "size_y": float(np.clip(geom[1], 0.35, 2.2)),
            "size_z": float(np.clip(geom[2], 0.35, 2.2)),
        }
    elif shape == "sphere":
        geometry = {"radius": float(np.clip(geom[0], 0.30, 1.40))}
    else:
        geometry = {
            "radius": float(np.clip(geom[0], 0.30, 1.30)),
            "depth": float(np.clip(geom[1], 0.40, 2.30)),
        }

    return {
        "shape": shape,
        "camera": {
            "location": [float(v) for v in camera],
            "lens_mm": CAMERA_LENS_MM,
        },
        "geometry": geometry,
    }


def bounding_radius(shape: str, geometry: Dict) -> float:
    if shape == "cube":
        hx = float(geometry["size_x"]) * 0.5
        hy = float(geometry["size_y"]) * 0.5
        hz = float(geometry["size_z"]) * 0.5
        return math.sqrt(hx * hx + hy * hy + hz * hz)
    if shape == "sphere":
        return float(geometry["radius"])
    if shape == "cylinder":
        r = float(geometry["radius"])
        hz = float(geometry["depth"]) * 0.5
        return math.sqrt(r * r + hz * hz)
    raise ValueError(f"Unknown shape: {shape}")


def safe_camera_distance(
    shape: str,
    geometry: Dict,
    margin: float = 1.25,
    lens_mm: float = CAMERA_LENS_MM,
    sensor_width_mm: float = CAMERA_SENSOR_WIDTH_MM,
) -> float:
    """Conservative distance based on a bounding sphere and horizontal FOV."""
    radius = bounding_radius(shape, geometry)
    half_fov = math.atan(sensor_width_mm / (2.0 * lens_mm))
    return radius / max(math.sin(half_fov), 1e-6) * margin


def parameter_errors(target: Dict, pred: Dict) -> Dict[str, float]:
    """Secondary diagnostics only; image-space reconstruction is the main metric."""
    t_cam = np.asarray(target["camera"]["location"], dtype=np.float32)
    p_cam = np.asarray(pred["camera"]["location"], dtype=np.float32)
    camera_l2 = float(np.linalg.norm(t_cam - p_cam))

    shape_correct = float(target["shape"] == pred["shape"])

    t_geom = _geom_triplet(target)
    p_geom = _geom_triplet(pred)
    geometry_mae = float(np.mean(np.abs(np.asarray(t_geom) - np.asarray(p_geom))))

    return {
        "shape_correct": shape_correct,
        "camera_l2": camera_l2,
        "geometry_mae": geometry_mae,
    }
