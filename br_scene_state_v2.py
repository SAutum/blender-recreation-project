from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

SHAPES = ("cube", "sphere", "cylinder")
MAX_OBJECTS = 2
SLOT_DIM = 13
STATE_DIM = 34

CAMERA_POS_MAX = 20.0
OBJECT_POS_MAX = 2.0
GEOM_MAX = 2.5
FOCAL_MIN_MM = 35.0
FOCAL_MAX_MM = 70.0
CAMERA_SENSOR_WIDTH_MM = 36.0

# State layout:
# 0                  : num_objects signal (-1 = one, +1 = two)
# 1:4                : camera location xyz
# 4:7                : camera Euler rotation xyz
# 7                  : focal length
# 8:21, 21:34        : object slots
# each object slot:
#   present, shape[3], position[3], rotation[3], geometry[3]


def geometry_triplet(obj: Dict) -> List[float]:
    shape = obj["shape"]
    g = obj["geometry"]
    if shape == "cube":
        return [float(g["size_x"]), float(g["size_y"]), float(g["size_z"])]
    if shape == "sphere":
        return [float(g["radius"]), 0.0, 0.0]
    if shape == "cylinder":
        return [float(g["radius"]), float(g["depth"]), 0.0]
    raise ValueError(f"Unknown shape: {shape}")


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


def _normalize_focal(lens_mm: float) -> float:
    x = (float(lens_mm) - FOCAL_MIN_MM) / (FOCAL_MAX_MM - FOCAL_MIN_MM)
    return float(np.clip(x * 2.0 - 1.0, -1.0, 1.0))


def _denormalize_focal(x: float) -> float:
    x = float(np.clip(x, -1.0, 1.0))
    return FOCAL_MIN_MM + (x + 1.0) * 0.5 * (FOCAL_MAX_MM - FOCAL_MIN_MM)


def encode_scene(scene: Dict) -> np.ndarray:
    objects = list(scene["objects"])
    if not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"v2 expects 1..{MAX_OBJECTS} objects, got {len(objects)}")

    state = np.full(STATE_DIM, -1.0, dtype=np.float32)
    state[0] = -1.0 if len(objects) == 1 else 1.0

    camera = scene["camera"]
    state[1:4] = np.clip(
        np.asarray(camera["location"], dtype=np.float32) / CAMERA_POS_MAX,
        -1.0,
        1.0,
    )
    state[4:7] = np.clip(
        np.asarray(camera["rotation_euler"], dtype=np.float32) / math.pi,
        -1.0,
        1.0,
    )
    state[7] = _normalize_focal(camera["lens_mm"])

    for i, obj in enumerate(objects):
        off = 8 + i * SLOT_DIM
        state[off] = 1.0
        state[off + 1 : off + 4] = -1.0
        state[off + 1 + SHAPES.index(obj["shape"])] = 1.0
        state[off + 4 : off + 7] = np.clip(
            np.asarray(obj["location"], dtype=np.float32) / OBJECT_POS_MAX,
            -1.0,
            1.0,
        )
        state[off + 7 : off + 10] = np.clip(
            np.asarray(obj["rotation_euler"], dtype=np.float32) / math.pi,
            -1.0,
            1.0,
        )
        geom = np.asarray(geometry_triplet(obj), dtype=np.float32)
        state[off + 10 : off + 13] = np.clip(
            (2.0 * geom / GEOM_MAX) - 1.0,
            -1.0,
            1.0,
        )

    return state


def _decode_geometry(shape: str, geom: np.ndarray) -> Dict:
    if shape == "cube":
        return {
            "size_x": float(np.clip(geom[0], 0.40, 1.80)),
            "size_y": float(np.clip(geom[1], 0.40, 1.80)),
            "size_z": float(np.clip(geom[2], 0.40, 1.80)),
        }
    if shape == "sphere":
        return {"radius": float(np.clip(geom[0], 0.30, 1.20))}
    return {
        "radius": float(np.clip(geom[0], 0.30, 1.00)),
        "depth": float(np.clip(geom[1], 0.50, 2.00)),
    }


def decode_state(state: Sequence[float]) -> Dict:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape ({STATE_DIM},), got {state.shape}")

    camera_location = np.clip(state[1:4], -1.0, 1.0) * CAMERA_POS_MAX
    camera_rotation = np.clip(state[4:7], -1.0, 1.0) * math.pi
    lens_mm = _denormalize_focal(float(state[7]))

    slot2_present = float(state[8 + SLOT_DIM])
    num_objects = 2 if float(state[0]) + slot2_present > 0.0 else 1

    objects = []
    for i in range(num_objects):
        off = 8 + i * SLOT_DIM
        shape = SHAPES[int(np.argmax(state[off + 1 : off + 4]))]
        location = np.clip(state[off + 4 : off + 7], -1.0, 1.0) * OBJECT_POS_MAX
        rotation = np.clip(state[off + 7 : off + 10], -1.0, 1.0) * math.pi
        geom = (
            (np.clip(state[off + 10 : off + 13], -1.0, 1.0) + 1.0)
            * 0.5
            * GEOM_MAX
        )
        objects.append(
            {
                "shape": shape,
                "location": [float(v) for v in location],
                "rotation_euler": [float(v) for v in rotation],
                "geometry": _decode_geometry(shape, geom),
            }
        )

    return {
        "version": 2,
        "camera": {
            "location": [float(v) for v in camera_location],
            "rotation_euler": [float(v) for v in camera_rotation],
            "lens_mm": float(lens_mm),
        },
        "objects": objects,
    }


def _wrapped_angle_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return np.abs((d + math.pi) % (2.0 * math.pi) - math.pi)


def parameter_errors(target: Dict, pred: Dict) -> Dict[str, float]:
    t_cam = target["camera"]
    p_cam = pred["camera"]
    camera_l2 = float(
        np.linalg.norm(
            np.asarray(t_cam["location"], dtype=np.float32)
            - np.asarray(p_cam["location"], dtype=np.float32)
        )
    )
    camera_rotation_mae_deg = float(
        np.degrees(
            np.mean(
                _wrapped_angle_abs(
                    np.asarray(t_cam["rotation_euler"]),
                    np.asarray(p_cam["rotation_euler"]),
                )
            )
        )
    )
    focal_abs_mm = abs(float(t_cam["lens_mm"]) - float(p_cam["lens_mm"]))

    t_objs = target["objects"]
    p_objs = pred["objects"]
    n = min(len(t_objs), len(p_objs))
    shape_scores = []
    position_errors = []
    rotation_errors = []
    geometry_errors = []
    for i in range(n):
        t_obj = t_objs[i]
        p_obj = p_objs[i]
        shape_scores.append(float(t_obj["shape"] == p_obj["shape"]))
        position_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(t_obj["location"], dtype=np.float32)
                    - np.asarray(p_obj["location"], dtype=np.float32)
                )
            )
        )
        rotation_errors.append(
            float(
                np.degrees(
                    np.mean(
                        _wrapped_angle_abs(
                            np.asarray(t_obj["rotation_euler"]),
                            np.asarray(p_obj["rotation_euler"]),
                        )
                    )
                )
            )
        )
        geometry_errors.append(
            float(
                np.mean(
                    np.abs(
                        np.asarray(geometry_triplet(t_obj))
                        - np.asarray(geometry_triplet(p_obj))
                    )
                )
            )
        )

    return {
        "object_count_correct": float(len(t_objs) == len(p_objs)),
        "shape_accuracy": float(np.mean(shape_scores)) if shape_scores else 0.0,
        "camera_l2": camera_l2,
        "camera_rotation_mae_deg": camera_rotation_mae_deg,
        "focal_abs_mm": float(focal_abs_mm),
        "object_position_l2_mean": float(np.mean(position_errors)) if position_errors else 0.0,
        "object_rotation_mae_deg": float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        "geometry_mae": float(np.mean(geometry_errors)) if geometry_errors else 0.0,
    }
