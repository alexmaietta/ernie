# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ernie_bench.scene import Scene, State
from ernie_bench.config import Workload

TAXONOMY = "class"
SUN_ANGLE = 0.53

OUTPUT_FORMATS = {
    "rgb": (3, np.float32),
    "depth": (1, np.float32),
    "segmentation": (1, np.int32),
}


def linear_from_srgb(encoded: np.ndarray) -> np.ndarray:
    value = encoded.astype(np.float32) / 255
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb_from_linear(linear) -> np.ndarray:
    value = np.clip(np.asarray(linear, np.float64), 0, 1)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)


def hits(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def single(renderer: str, values: np.ndarray, what: str) -> float:
    found = np.unique(values)
    if len(found) != 1:
        raise ValueError(f"{renderer} needs one {what} for every view, but the scene has {found.tolist()}")
    return float(found[0])


def apertures(fov_y, focal_length: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    vertical = 2 * focal_length * np.tan(np.radians(np.asarray(fov_y, np.float64)) / 2)
    return vertical, vertical * width / height


def sun_basis(direction: tuple[float, float, float]) -> np.ndarray:
    backward = -np.asarray(direction, np.float64) / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(backward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, backward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(backward, right), backward], axis=1)


def class_labels(id_map: np.ndarray, ids: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    blob = id_map.tobytes()
    count = struct.unpack_from("<I", blob, len(blob) - 4)[0]
    records = [struct.unpack_from("<6I", blob, record * 24) for record in range(count)]
    lookup = np.zeros(max([2, *(record[0] + 1 for record in records)]), np.int32)
    for semantic_id, _, _, _, length, offset in records:
        label = blob[offset : offset + length].decode().strip().removeprefix(f"{TAXONOMY}:").strip().removesuffix(";").strip()
        if label in classes:
            lookup[semantic_id] = classes.index(label) + 1
    ids = ids.astype(np.int64)
    return np.where(ids < len(lookup), lookup[np.minimum(ids, len(lookup) - 1)], 0).astype(np.int32)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str
    device: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    tiers: frozenset[str]
    outputs: frozenset[str]
    max_spp: int | None = None
    square_only: bool = False
    sm_limit: bool = False

    def unsupported(self, workload: Workload) -> list[str]:
        reasons = []
        if workload.tier not in self.tiers:
            reasons.append(f"tier {workload.tier}")
        missing = set(workload.outputs) - self.outputs
        if missing:
            reasons.append(f"outputs {sorted(missing)}")
        if self.max_spp is not None and workload.spp > self.max_spp:
            reasons.append(f"{workload.spp} spp exceeds {self.max_spp}")
        if self.square_only and workload.width != workload.height:
            reasons.append(f"{workload.width}x{workload.height} is not square")
        if workload.sm_percent < 100 and not self.sm_limit:
            reasons.append("MPS cannot limit its SMs")
        return reasons


class Backend(Protocol):
    def info(self) -> BackendInfo: ...
    def capabilities(self) -> Capabilities: ...
    def load(self, scene: Scene, workload: Workload) -> None: ...
    def update(self, state: State) -> None: ...
    def render(self) -> None: ...
    def sync(self) -> None: ...
    def readback(self) -> dict[str, np.ndarray]: ...
    def stream(self) -> int | None: ...
    def close(self) -> None: ...
