# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from ernie_bench.backends.base import BackendInfo, Capabilities, hits, linear_from_srgb, single
from ernie_bench.reference import camera_rays
from ernie_bench.scene import Scene, State, decompose
from ernie_bench.config import Workload

VERSION = "madrona_benchmark b908c43 + cuda 12.8 headers"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "madrona-cache" / "native"
MADRONA = ROOT / "external" / "madrona_benchmark" / "external" / "madrona"
RUNTIME_SOURCES = (MADRONA / "src" / "mw" / "device", MADRONA / "include" / "madrona", ROOT / "native" / "madrona" / "src")
CAMERA_AXES = np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def scalar_first(quaternions: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.roll(quaternions, 1, axis=-1), np.float32)


def kernel_caches(module: Path, mode: str) -> dict[str, str]:
    digest = hashlib.sha256(module.read_bytes())
    for root in RUNTIME_SOURCES:
        for source in sorted(path for path in root.rglob("*") if path.is_file()):
            digest.update(source.read_bytes())
    digest = digest.hexdigest()[:16]
    folder = CACHE / f"{digest}-{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    return {"MADRONA_MWGPU_KERNEL_CACHE": str(folder / "kernels"), "MADRONA_BVH_KERNEL_CACHE": str(folder / "bvh")}


class MadronaBackend:
    def __init__(self, raytrace: bool = True):
        import _madrona_bench

        self._module = _madrona_bench
        self._raytrace = raytrace
        for name, path in kernel_caches(Path(_madrona_bench.__file__), "rt" if raytrace else "rast").items():
            os.environ.setdefault(name, path)

    def info(self) -> BackendInfo:
        deviations = (
            "white albedo, fixed sun towards (0.5, 0.5, 2) with ambient 0.4, lighting clamped at 1"
            if self._raytrace
            else "samples pixel corners, row 0 is garbage"
        )
        return BackendInfo("madrona", VERSION, "cuda:0", {"renderer": "raytracer" if self._raytrace else "rasterizer", "deviations": deviations})

    def capabilities(self) -> Capabilities:
        tiers = {"depth", "direct"} if self._raytrace else {"depth", "unlit"}
        return Capabilities(frozenset(tiers), frozenset({"rgb", "depth"}), max_spp=1, square_only=self._raytrace, sm_limit=self._raytrace)

    def load(self, scene: Scene, workload: Workload) -> None:
        layout = [(tuple(scene.instances.mesh[found]), tuple(scene.instances.material[found])) for found in scene.members]
        if any(entry != layout[0] for entry in layout) or len({len(found) for found in scene.views}) != 1:
            raise ValueError("madrona builds every world from one layout, but the scene's worlds differ")

        pairs = sorted(set(zip(*layout[0])))
        instance_objects = np.array([pairs.index(pair) for pair in zip(*layout[0])], np.int32)
        vertex_offsets = np.cumsum([0] + [len(mesh.positions) for mesh in scene.meshes[:-1]]).astype(np.uint32)
        triangle_offsets = np.cumsum([0] + [len(mesh.indices) for mesh in scene.meshes[:-1]]).astype(np.uint32)
        direction = -np.asarray(scene.lights[0].direction if scene.lights else (0.0, 0.0, -1.0), np.float32)
        fov = single("madrona", scene.cameras.fov_y, "field of view")
        self._far = single("madrona", scene.cameras.far, "far plane")

        self._renderer = self._module.Renderer(
            0,
            scene.world_count,
            workload.width,
            workload.height,
            self._raytrace,
            fov,
            single("madrona", scene.cameras.near, "near plane"),
            self._far,
            direction / np.linalg.norm(direction),
            np.concatenate([mesh.positions for mesh in scene.meshes]).astype(np.float32),
            np.concatenate([mesh.normals for mesh in scene.meshes]).astype(np.float32),
            np.concatenate([mesh.indices for mesh in scene.meshes]).astype(np.uint32),
            vertex_offsets,
            triangle_offsets,
            np.array([mesh for mesh, _ in pairs], np.int32),
            np.array([scene.materials[material].base_color for _, material in pairs], np.float32),
            instance_objects,
            len(scene.views[0]),
        )

        self._rows = np.empty(len(scene.instances), np.int64)
        self._rows[np.concatenate(scene.members)] = np.arange(len(scene.instances))
        self._camera_rows = np.empty(len(scene.cameras), np.int64)
        self._camera_rows[np.concatenate(scene.views)] = np.arange(len(scene.cameras))
        translation, rotation, scale = decompose(scene.instances.transform)
        self._positions = np.zeros((len(scene.instances), 3), np.float32)
        self._rotations = np.zeros((len(scene.instances), 4), np.float32)
        self._scales = np.zeros((len(scene.instances), 3), np.float32)
        self._positions[self._rows] = translation
        self._rotations[self._rows] = scalar_first(rotation)
        self._scales[self._rows] = scale
        self._camera_positions = np.zeros((len(scene.cameras), 3), np.float32)
        self._camera_rotations = np.zeros((len(scene.cameras), 4), np.float32)
        self._moving = self._rows[scene.motion.instance]
        self._size = (workload.height, workload.width)
        self._ray_lengths = (1 / camera_rays(np.eye(4), fov, workload.width, workload.height)[2]).reshape(workload.height, workload.width).astype(np.float32)
        self._outputs = workload.outputs

    def update(self, state: State) -> None:
        self._positions[self._moving] = state.translation
        self._rotations[self._moving] = scalar_first(state.rotation)
        self._scales[self._moving] = state.scale
        self._camera_positions[self._camera_rows] = state.camera_translation
        self._camera_rotations[self._camera_rows] = scalar_first(multiply(state.camera_rotation, CAMERA_AXES))
        self._renderer.set_instances(self._positions, self._rotations, self._scales)
        self._renderer.set_cameras(self._camera_positions, self._camera_rotations)

    def render(self) -> None:
        self._renderer.render()

    def sync(self) -> None:
        self._renderer.synchronize()

    def readback(self) -> dict[str, np.ndarray]:
        views = len(self._camera_rows)
        height, width = self._size
        frames = {}
        if "rgb" in self._outputs:
            color = np.empty((views, height, width, 4), np.uint8)
            self._renderer.read_color(color)
            if self._raytrace:
                color = color.transpose(0, 2, 1, 3)
                frames["rgb"] = np.ascontiguousarray(color[self._camera_rows, ..., :3], np.float32) / 255
            else:
                frames["rgb"] = linear_from_srgb(color[self._camera_rows, ..., :3])
        if "depth" in self._outputs:
            depth = np.empty((views, height, width), np.float32)
            self._renderer.read_depth(depth)
            if self._raytrace:
                depth = depth.transpose(0, 2, 1)
            else:
                depth = depth * self._ray_lengths
            depth = depth[self._camera_rows]
            frames["depth"] = np.where(hits(depth) & (depth < self._far), depth, np.inf)[..., None]
        return frames

    def stream(self) -> int | None:
        return self._renderer.stream or None

    def close(self) -> None:
        self._renderer = None
