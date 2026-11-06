# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench.backends.base import BackendInfo, Capabilities, apertures, hits, linear_from_srgb, single, srgb_from_linear
from ernie_bench.scene import Scene, State, decompose
from ernie_bench.config import OUTPUTS, Workload

FOCAL_LENGTH = 10.0
MISSED = 0xFFFFFFFF


class NewtonBackend:
    def __init__(self):
        import newton
        import warp as wp
        from newton.sensors import SensorCamera

        self._newton = newton
        self._wp = wp
        self._camera = SensorCamera
        self._device = wp.get_device("cuda:0")

    def info(self) -> BackendInfo:
        return BackendInfo(
            "newton",
            self._newton.__version__,
            f"{self._device.alias} {self._device.name}",
            {
                "warp": self._wp.__version__,
                "architecture": f"sm_{self._device.arch}",
                "renderer": "SensorCamera",
                "deviations": "shadowed points keep 30% of the sun, ambient is a sky and ground gradient",
            },
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(frozenset({"depth", "segmentation", "unlit", "direct"}), frozenset(OUTPUTS), max_spp=1, sm_limit=True)

    def load(self, scene: Scene, workload: Workload) -> None:
        newton, wp, camera = self._newton, self._wp, self._camera
        fov = single("newton", scene.cameras.fov_y, "field of view")

        meshes = [newton.Mesh(mesh.positions, mesh.indices.reshape(-1), normals=mesh.normals, compute_inertia=False) for mesh in scene.meshes]
        translation, rotation, scale = decompose(scene.instances.transform)
        moving = np.zeros(len(scene.instances), bool)
        moving[scene.motion.instance] = True
        body = np.full(len(scene.instances), -1, np.int64)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        lookup = {}
        for found in scene.members:
            builder.begin_world()
            for index in found:
                pose = wp.transform(wp.vec3(*translation[index]), wp.quat(*rotation[index]))
                if moving[index]:
                    body[index] = builder.add_body(xform=pose, is_kinematic=True)
                shape = builder.add_shape_mesh(
                    int(body[index]),
                    xform=wp.transform_identity() if moving[index] else pose,
                    mesh=meshes[scene.instances.mesh[index]],
                    scale=wp.vec3(*scale[index]),
                    color=wp.vec3(*srgb_from_linear(scene.colors[index])),
                )
                lookup[shape] = scene.instances.semantic[index] + 1
            builder.end_world()

        self._model = builder.finalize(device=self._device)
        self._state = self._model.state()
        self._classes = np.zeros(self._model.shape_count, np.int32)
        self._classes[list(lookup)] = list(lookup.values())

        config = camera.RenderConfig()
        config.enable_shadows = workload.tier == "direct"
        config.enable_ambient_lighting = scene.ambient > 0
        self._sensor = camera(self._model, default_render_config=config)
        if scene.lights:
            self._sensor.create_default_light(enable_shadows=workload.tier == "direct", direction=wp.vec3f(*scene.lights[0].direction))

        vertical, horizontal = apertures(fov, FOCAL_LENGTH, workload.width, workload.height)
        self._rays = camera.compute_camera_rays_pinhole(
            workload.width,
            workload.height,
            focal_length=FOCAL_LENGTH,
            vertical_aperture=float(vertical),
            horizontal_aperture=float(horizontal),
            device=self._device,
        )
        views = len(scene.cameras)
        allocate = {
            "rgb": ("albedo_image", self._sensor.allocate_image_albedo) if workload.tier == "unlit" else ("color_image", self._sensor.allocate_image_color),
            "depth": ("depth_image", self._sensor.allocate_image_depth),
            "segmentation": ("shape_index_image", self._sensor.allocate_image_shape_index),
        }
        self._images = {output: (argument, allocate[output][1](views, workload.width, workload.height)) for output, (argument, _) in allocate.items() if output in workload.outputs}
        self._views = wp.zeros(views, dtype=wp.transformf, device=self._device)
        self._worlds = wp.array(scene.cameras.world.astype(np.int32), dtype=wp.int32, device=self._device)
        self._moving_bodies = body[scene.motion.instance]
        self._body_poses = self._state.body_q.numpy()

    def update(self, state: State) -> None:
        self._body_poses[self._moving_bodies, :3] = state.translation
        self._body_poses[self._moving_bodies, 3:] = state.rotation
        self._state.body_q.assign(self._body_poses)
        self._views.assign(np.concatenate([state.camera_translation, state.camera_rotation], axis=1).astype(np.float32))

    def render(self) -> None:
        self._model.bvh_refit_shapes(self._state)
        self._sensor.update(
            self._state,
            self._views,
            self._rays,
            world_indices=self._worlds,
            **{argument: image for argument, image in self._images.values()},
        )

    def sync(self) -> None:
        self._wp.synchronize_device(self._device)

    def readback(self) -> dict[str, np.ndarray]:
        frames = {}
        for output, (_, image) in self._images.items():
            values = image.numpy()
            if output == "rgb":
                frames[output] = linear_from_srgb(values.view(np.uint8).reshape(*values.shape, 4)[..., :3])
            elif output == "depth":
                frames[output] = np.where(hits(values), values, np.inf).astype(np.float32)[..., None]
            else:
                hit = values != MISSED
                frames[output] = np.where(hit, self._classes[np.where(hit, values, 0)], 0).astype(np.int32)[..., None]
        return frames

    def stream(self) -> int | None:
        return self._device.stream.cuda_stream

    def close(self) -> None:
        self._sensor = None
        self._state = None
        self._model = None
