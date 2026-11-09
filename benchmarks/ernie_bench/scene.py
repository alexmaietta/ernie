# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray


@dataclass
class Material:
    base_color: tuple[float, float, float]


@dataclass
class Instances:
    world: np.ndarray
    mesh: np.ndarray
    material: np.ndarray
    semantic: np.ndarray
    transform: np.ndarray

    def __len__(self) -> int:
        return len(self.world)


@dataclass
class Cameras:
    world: np.ndarray
    parent: np.ndarray
    local: np.ndarray
    fov_y: np.ndarray
    near: np.ndarray
    far: np.ndarray

    def __len__(self) -> int:
        return len(self.world)


@dataclass
class DistantLight:
    direction: tuple[float, float, float]
    irradiance: float
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class LissajousMotion:
    instance: np.ndarray
    center: np.ndarray
    amplitude: np.ndarray
    frequency: np.ndarray
    phase: np.ndarray
    height: np.ndarray
    scale: np.ndarray

    def poses(self, time: float) -> np.ndarray:
        angle = self.frequency * time + self.phase
        velocity = self.amplitude * self.frequency * np.cos(angle)
        position = np.column_stack([self.center + self.amplitude * np.sin(angle), self.height])
        yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
        return compose(position, yaw, np.ones_like(self.scale))


@dataclass
class State:
    time: float
    instance: np.ndarray
    transform: np.ndarray
    camera: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    camera_translation: np.ndarray
    camera_rotation: np.ndarray


@dataclass
class Scene:
    name: str
    world_count: int
    meshes: list[Mesh]
    materials: list[Material]
    classes: tuple[str, ...]
    instances: Instances
    cameras: Cameras
    lights: list[DistantLight]
    motion: LissajousMotion
    ambient: float = 0.0
    colors: np.ndarray = field(init=False, repr=False)
    members: list[np.ndarray] = field(init=False, repr=False)
    views: list[np.ndarray] = field(init=False, repr=False)
    _parent_row: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.colors = np.array([material.base_color for material in self.materials], np.float64)[self.instances.material]
        self.members = [np.flatnonzero(self.instances.world == world) for world in range(self.world_count)]
        self.views = [np.flatnonzero(self.cameras.world == world) for world in range(self.world_count)]
        row = np.full(len(self.instances), -1, np.int64)
        row[self.motion.instance] = np.arange(len(self.motion.instance))
        parent = self.cameras.parent
        self._parent_row = np.where(parent >= 0, row[np.maximum(parent, 0)], -1)
        if np.any((parent >= 0) & (self._parent_row < 0)):
            raise ValueError("a camera's parent must be a moving instance")

    @property
    def triangle_count(self) -> int:
        per_mesh = np.array([len(mesh.indices) for mesh in self.meshes])
        return int(per_mesh[self.instances.mesh].sum())

    def transforms(self, state: State) -> np.ndarray:
        transforms = self.instances.transform.copy()
        transforms[state.instance] = state.transform
        return transforms

    def state(self, time: float) -> State:
        poses = self.motion.poses(time)
        transform = poses.copy()
        transform[:, :3, :3] *= self.motion.scale[:, None, :].astype(np.float32)
        camera = self.cameras.local.copy()
        mounted = self._parent_row >= 0
        camera[mounted] = np.einsum("nij,njk->nik", poses[self._parent_row[mounted]], self.cameras.local[mounted])
        translation, rotation, scale = decompose(transform)
        camera_translation, camera_rotation, _ = decompose(camera)
        return State(time, self.motion.instance, transform, camera, translation, rotation, scale, camera_translation, camera_rotation)


def compose(translation: np.ndarray, yaw: np.ndarray, scale: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    matrix = np.zeros((len(translation), 4, 4), np.float32)
    matrix[:, 0, 0] = cos * scale[:, 0]
    matrix[:, 0, 1] = -sin * scale[:, 1]
    matrix[:, 1, 0] = sin * scale[:, 0]
    matrix[:, 1, 1] = cos * scale[:, 1]
    matrix[:, 2, 2] = scale[:, 2]
    matrix[:, :3, 3] = translation
    matrix[:, 3, 3] = 1
    return matrix


def quaternions(rotation: np.ndarray) -> np.ndarray:
    m = rotation.astype(np.float64)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    squares = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=1)
    largest = np.argmax(squares, axis=1)
    s = 2 * np.sqrt(np.maximum(squares[np.arange(len(m)), largest], 1e-300))
    candidates = np.stack(
        [
            np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4], axis=1),
            np.stack([s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], axis=1),
            np.stack([(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s], axis=1),
            np.stack([(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s], axis=1),
        ]
    )
    quaternion = candidates[largest, np.arange(len(m))]
    return quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)


def decompose(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = matrices[:, :3, :3].astype(np.float64)
    scale = np.linalg.norm(linear, axis=1)
    rotation = linear / np.where(scale > 0, scale, 1.0)[:, None, :]
    return matrices[:, :3, 3].astype(np.float64), quaternions(rotation), scale.astype(np.float32)
