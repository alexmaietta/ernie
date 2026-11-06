# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench.backends.base import BackendInfo, Capabilities
from ernie_bench.scene import Scene, State
from ernie_bench.config import OUTPUTS, TIERS, Workload

PATH_TRACED_DEPTH = 5
REFERENCE_SPP = 256
INTEGRATORS = {"unlit": {"type": "aov", "aovs": "albedo:albedo"}, "pathtraced": {"type": "path", "max_depth": PATH_TRACED_DEPTH}}


def baked(positions: np.ndarray, normals: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = matrix.astype(np.float64)
    world = positions @ matrix[:3, :3].T + matrix[:3, 3]
    turned = normals @ np.linalg.inv(matrix[:3, :3])
    return world.astype(np.float32), (turned / np.linalg.norm(turned, axis=1, keepdims=True)).astype(np.float32)


class MitsubaBackend:
    def __init__(self, variant: str = "cuda_ad_rgb"):
        import drjit as dr
        import mitsuba as mi

        mi.set_variant(variant)
        self._mi = mi
        self._dr = dr
        self._variant = variant

    def info(self) -> BackendInfo:
        return BackendInfo("mitsuba", self._mi.__version__, self._variant, {"role": "reference", "integrators": "aov albedo, direct, path", "minimum_spp": REFERENCE_SPP})

    def capabilities(self) -> Capabilities:
        return Capabilities(frozenset(TIERS), frozenset(OUTPUTS), sm_limit=self._variant.startswith("cuda"))

    def load(self, scene: Scene, workload: Workload) -> None:
        mi = self._mi
        self._scene = scene
        self._workload = workload
        self._worlds = []
        light = scene.lights[0] if scene.lights else None
        for instances, cameras in zip(scene.members, scene.views):
            description = {"type": "scene", "integrator": INTEGRATORS.get(workload.tier, {"type": "direct"})}
            for slot, index in enumerate(instances):
                mesh = scene.meshes[scene.instances.mesh[index]]
                properties = mi.Properties()
                properties["bsdf"] = mi.load_dict({"type": "diffuse", "reflectance": {"type": "rgb", "value": scene.colors[index].tolist()}})
                shape = mi.Mesh(f"I{slot:05d}", len(mesh.positions), len(mesh.indices), properties, has_vertex_normals=True)
                parameters = mi.traverse(shape)
                positions, normals = baked(mesh.positions, mesh.normals, scene.instances.transform[index])
                parameters["vertex_positions"] = mi.Float(positions.ravel())
                parameters["vertex_normals"] = mi.Float(normals.ravel())
                parameters["faces"] = mi.UInt32(mesh.indices.astype(np.uint32).ravel())
                parameters.update()
                description[f"I{slot:05d}"] = shape
            sensors = []
            for index in cameras:
                sensors.append(mi.load_dict({
                    "type": "perspective",
                    "fov": float(scene.cameras.fov_y[index]),
                    "fov_axis": "y",
                    "near_clip": float(scene.cameras.near[index]),
                    "far_clip": float(scene.cameras.far[index]),
                    "film": {"type": "hdrfilm", "width": workload.width, "height": workload.height, "rfilter": {"type": "box"}, "pixel_format": "rgb"},
                    "sampler": {"type": "independent", "sample_count": max(workload.spp, REFERENCE_SPP)},
                }))
            if light is not None:
                description["sun"] = {
                    "type": "directional",
                    "direction": list(light.direction),
                    "irradiance": {"type": "rgb", "value": [light.irradiance * c for c in light.color]},
                }
            if scene.ambient > 0:
                description["ambient"] = {"type": "constant", "radiance": {"type": "rgb", "value": scene.ambient}}
            loaded = mi.load_dict(description)
            self._worlds.append((loaded, mi.traverse(loaded), instances, cameras, sensors))
        self._frames = {}
        self._origins = np.zeros((len(scene.cameras), 3))

    def update(self, state: State) -> None:
        mi = self._mi
        transforms = self._scene.transforms(state)
        moving = set(state.instance.tolist())
        for loaded, parameters, instances, cameras, sensors in self._worlds:
            for slot, index in enumerate(instances):
                if index in moving:
                    mesh = self._scene.meshes[self._scene.instances.mesh[index]]
                    positions, normals = baked(mesh.positions, mesh.normals, transforms[index])
                    parameters[f"I{slot:05d}.vertex_positions"] = mi.Float(positions.ravel())
                    parameters[f"I{slot:05d}.vertex_normals"] = mi.Float(normals.ravel())
            parameters.update()
            for sensor, index in zip(sensors, cameras):
                camera = state.camera[index].astype(np.float64)
                origin = camera[:3, 3]
                self._origins[index] = origin
                lens = mi.traverse(sensor)
                lens["to_world"] = mi.ScalarTransform4f().look_at(origin=origin.tolist(), target=(origin - camera[:3, 2]).tolist(), up=camera[:3, 1].tolist())
                lens.update()

    def render(self) -> None:
        mi, dr = self._mi, self._dr
        workload = self._workload
        height, width = workload.height, workload.width
        views = len(self._scene.cameras)
        rgb = np.zeros((views, height, width, 3), np.float32)
        depth = np.full((views, height, width, 1), np.inf, np.float32)
        classes = np.zeros((views, height, width, 1), np.int32)
        columns, rows = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
        film = mi.Point2f(columns.ravel().astype(np.float32), rows.ravel().astype(np.float32))
        for loaded, _, instances, cameras, sensors in self._worlds:
            labels = self._scene.instances.semantic[instances] + 1
            shapes = loaded.shapes()
            lookup = np.zeros(len(shapes) + 1, np.int32)
            for position, shape in enumerate(shapes):
                lookup[position + 1] = labels[int(shape.id()[1:])]
            for slot, index in enumerate(cameras):
                if "rgb" in workload.outputs:
                    rgb[index] = np.asarray(mi.render(loaded, sensor=sensors[slot], spp=max(workload.spp, REFERENCE_SPP)))[..., :3]
                if "depth" in workload.outputs or "segmentation" in workload.outputs:
                    rays, _ = sensors[slot].sample_ray(0.0, 0.5, film, mi.Point2f(0.5, 0.5))
                    hit = loaded.ray_intersect(rays)
                    points = np.stack([np.asarray(hit.p.x), np.asarray(hit.p.y), np.asarray(hit.p.z)], axis=1)
                    distance = np.where(np.asarray(hit.is_valid()), np.linalg.norm(points - self._origins[index], axis=1), np.inf)
                    depth[index, ..., 0] = distance.reshape(height, width)
                    found = mi.UInt32(0)
                    for order, shape in enumerate(shapes):
                        found = dr.select(hit.shape == mi.ShapePtr(shape), mi.UInt32(order + 1), found)
                    classes[index, ..., 0] = lookup[np.asarray(found).reshape(height, width)]
        self._frames = {"rgb": rgb, "depth": depth, "segmentation": classes}

    def sync(self) -> None:
        self._dr.sync_thread()

    def readback(self) -> dict[str, np.ndarray]:
        return {output: self._frames[output] for output in self._workload.outputs}

    def stream(self) -> int | None:
        return None

    def close(self) -> None:
        self._worlds = []
        self._frames = {}
