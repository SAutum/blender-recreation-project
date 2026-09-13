from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

SHAPES = ("cube", "sphere", "cylinder")
MAX_OBJECTS = 2
SLOT_DIM = 13

# v3 scene-level layout:
# 0      : num_objects signal (-1 = one, +1 = two)
# 1      : camera azimuth
# 2      : camera elevation
# 3      : camera distance
# 4:7    : camera target offset from decoded object centroid xyz
# 7      : camera roll
# 8      : focal length
# 9:22   : object slot 1
# 22:35  : object slot 2
#
# Each object slot:
# present, shape[3], position[3], rotation[3], geometry[3]
STATE_DIM = 35

OBJECT_POS_MAX = 2.0
GEOM_MAX = 2.5
TARGET_OFFSET_MAX = 0.35
CAMERA_DISTANCE_MIN = 2.5
CAMERA_DISTANCE_MAX = 12.0
CAMERA_ELEVATION_MAX_RAD = math.radians(60.0)
CAMERA_ROLL_MAX_RAD = math.radians(20.0)
FOCAL_MIN_MM = 35.0
FOCAL_MAX_MM = 70.0
CAMERA_SENSOR_WIDTH_MM = 36.0


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


def object_centroid(objects: Sequence[Dict]) -> np.ndarray:
    if not objects:
        return np.zeros(3, dtype=np.float32)
    return np.mean(
        [np.asarray(obj["location"], dtype=np.float32) for obj in objects],
        axis=0,
    ).astype(np.float32)


def camera_location_from_parameters(
    azimuth: float,
    elevation: float,
    distance: float,
    target: Sequence[float],
) -> List[float]:
    ce = math.cos(float(elevation))
    outward = np.asarray(
        [
            ce * math.cos(float(azimuth)),
            ce * math.sin(float(azimuth)),
            math.sin(float(elevation)),
        ],
        dtype=np.float32,
    )
    location = np.asarray(target, dtype=np.float32) + outward * float(distance)
    return [float(v) for v in location]


def _norm_linear(value: float, lo: float, hi: float) -> float:
    x = (float(value) - lo) / max(hi - lo, 1e-8)
    return float(np.clip(x * 2.0 - 1.0, -1.0, 1.0))


def _denorm_linear(value: float, lo: float, hi: float) -> float:
    x = float(np.clip(value, -1.0, 1.0))
    return lo + (x + 1.0) * 0.5 * (hi - lo)


def encode_scene(scene: Dict) -> np.ndarray:
    objects = list(scene["objects"])
    if not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"v3 expects 1..{MAX_OBJECTS} objects, got {len(objects)}")

    state = np.full(STATE_DIM, -1.0, dtype=np.float32)
    state[0] = -1.0 if len(objects) == 1 else 1.0

    camera = scene["camera"]
    state[1] = float(np.clip(float(camera["azimuth"]) / math.pi, -1.0, 1.0))
    state[2] = float(
        np.clip(
            float(camera["elevation"]) / CAMERA_ELEVATION_MAX_RAD,
            -1.0,
            1.0,
        )
    )
    state[3] = _norm_linear(
        camera["distance"], CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX
    )

    centroid = object_centroid(objects)
    target = np.asarray(camera["target"], dtype=np.float32)
    target_offset = np.clip(
        (target - centroid) / TARGET_OFFSET_MAX,
        -1.0,
        1.0,
    )
    state[4:7] = target_offset
    state[7] = float(
        np.clip(float(camera["roll"]) / CAMERA_ROLL_MAX_RAD, -1.0, 1.0)
    )
    state[8] = _norm_linear(camera["lens_mm"], FOCAL_MIN_MM, FOCAL_MAX_MM)

    for i, obj in enumerate(objects):
        off = 9 + i * SLOT_DIM
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

    slot2_present = float(state[9 + SLOT_DIM])
    num_objects = 2 if float(state[0]) + slot2_present > 0.0 else 1

    objects: List[Dict] = []
    for i in range(num_objects):
        off = 9 + i * SLOT_DIM
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

    centroid = object_centroid(objects)
    target_offset = np.clip(state[4:7], -1.0, 1.0) * TARGET_OFFSET_MAX
    target = centroid + target_offset

    azimuth = float(np.clip(state[1], -1.0, 1.0) * math.pi)
    elevation = float(
        np.clip(state[2], -1.0, 1.0) * CAMERA_ELEVATION_MAX_RAD
    )
    distance = _denorm_linear(
        float(state[3]), CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX
    )
    roll = float(np.clip(state[7], -1.0, 1.0) * CAMERA_ROLL_MAX_RAD)
    lens_mm = _denorm_linear(float(state[8]), FOCAL_MIN_MM, FOCAL_MAX_MM)
    location = camera_location_from_parameters(azimuth, elevation, distance, target)

    return {
        "version": 3,
        "camera": {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": float(distance),
            "target": [float(v) for v in target],
            "roll": roll,
            "lens_mm": float(lens_mm),
            # Convenience/debug value. The renderer does not trust a predicted Euler angle;
            # it deterministically reconstructs orientation from target + roll.
            "location": location,
        },
        "objects": objects,
    }
