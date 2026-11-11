# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench.backends.base import SUN_ANGLE, TAXONOMY, BackendInfo, Capabilities, apertures, class_labels, hits, linear_from_srgb, sun_basis
from ernie_bench.scene import Scene, State, decompose, quaternions
from ernie_bench.config import OUTPUTS, TIERS, Workload
from ernie_bench.system import cuda

PRODUCT = "/Render/Batch"
PROTOTYPES = "/Prototypes"
SOURCES = {"rgb": "LdrColor", "depth": "DistanceToCamera", "segmentation": "SemanticSegmentation"}
COLOR_SOURCES = {"unlit": "DiffuseAlbedo"}
SUN_SOLID_ANGLE = 2 * np.pi * (1 - np.cos(np.radians(SUN_ANGLE) / 2))
FOCAL_LENGTH = 10.0
PATH_TRACED_BOUNCES = 4
SHADING = {
    "depth": {"render_mode": "Minimal", "mode": "data"},
    "segmentation": {"render_mode": "Minimal", "mode": "data"},
    "unlit": {"render_mode": "Minimal", "mode": "displayColor", "cast_shadows": False},
    "direct": {"render_mode": "Minimal", "mode": "displayColor", "cast_shadows": True},
    "pathtraced": {"render_mode": "RealTime", "max_bounces": PATH_TRACED_BOUNCES},
}


def world_path(world: int) -> str:
    return f"/W{world:05d}"


def prototype_path(slot: int) -> str:
    return f"{PROTOTYPES}/P{slot:05d}"


class ErnieBackend:
    def __init__(self, api: str | None = None, traversal: str = "AUTO", sampling: str = "AUTO"):
        import ernie

        self._ernie = ernie
        self._device = ernie.Device(
            backend=getattr(ernie.Backend, api) if api else None,
            traversal=getattr(ernie.Traversal, traversal),
            sampling=getattr(ernie.Sampling, sampling),
        )
        for asked, given, what in ((traversal, self._device.traversal.name, "traversal"), (sampling, self._device.sampling.name, "sampling")):
            if asked != "AUTO" and given != asked:
                raise RuntimeError(f"asked for {asked.lower()} {what} but {self._device.adapter.alias} gave {given.lower()}")
        self._output = None
        self._stream = None

    def info(self) -> BackendInfo:
        adapter = self._device.adapter
        return BackendInfo(
            "ernie",
            self._ernie.__version__,
            f"{adapter.alias} {adapter.name}",
            {
                "backend": self._ernie.backend_name(adapter.backend),
                "traversal": self._device.traversal.name,
                "sampling": self._device.sampling.name,
                "driver": adapter.driver_version,
                "api_version": adapter.api_version,
                "architecture": adapter.architecture,
            },
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(frozenset(TIERS), frozenset(OUTPUTS), max_spp=1, sm_limit=self._device.adapter.backend == self._ernie.Backend.CUDA)

    def load(self, scene: Scene, workload: Workload) -> None:
        ernie = self._ernie
        stage = ernie.Scene(self._device)
        layout = [
            (tuple(scene.instances.mesh[found]), tuple(scene.instances.material[found]), tuple(scene.instances.semantic[found]), len(seen))
            for found, seen in zip(scene.members, scene.views)
        ]

        keys = np.stack([scene.instances.mesh, scene.instances.material, scene.instances.semantic], axis=1)
        prototypes, prototype = np.unique(keys, axis=0, return_inverse=True)
        prototype = prototype.reshape(-1)
        ernie.Scope.define(stage, PROTOTYPES)
        for slot, (mesh_index, material, semantic) in enumerate(prototypes):
            mesh = scene.meshes[mesh_index]
            ernie.Mesh.define(
                stage,
                prototype_path(slot),
                points=mesh.positions,
                face_vertex_indices=mesh.indices.reshape(-1).astype(np.int32),
                face_vertex_counts=np.full(len(mesh.indices), 3, np.int32),
                normals=mesh.normals,
                normals_interpolation="vertex",
                display_color=tuple(scene.materials[material].base_color),
                semantics_labels={TAXONOMY: [scene.classes[semantic]]},
            )

        def define_world(world: int) -> None:
            root = world_path(world)
            ernie.World.define(stage, root)
            for slot, index in enumerate(scene.members[world]):
                ernie.Xform.define(stage, f"{root}/I{slot:05d}", prototype={"prototype": prototype_path(prototype[index])})
            for slot in range(len(scene.views[world])):
                ernie.Camera.define(stage, f"{root}/C{slot:03d}")
            for slot, light in enumerate(scene.lights):
                ernie.DistantLight.define(
                    stage,
                    f"{root}/L{slot:02d}",
                    orient=quaternions(sun_basis(light.direction)[None])[0],
                    angle=SUN_ANGLE,
                    intensity=float(light.irradiance / SUN_SOLID_ANGLE),
                    color=light.color,
                )

        groups = {}
        for world, entry in enumerate(layout):
            groups.setdefault(entry, []).append(world)
        for first, *rest in groups.values():
            define_world(first)
            if rest:
                stage.clone(stage.find(world_path(first)), [world_path(world) for world in rest])

        instance_paths = np.empty(len(scene.instances), object)
        for world, found in enumerate(scene.members):
            instance_paths[found] = [f"{world_path(world)}/I{slot:05d}" for slot in range(len(found))]
        camera_paths = [None] * len(scene.cameras)
        for world, found in enumerate(scene.views):
            for slot, index in enumerate(found):
                camera_paths[index] = f"{world_path(world)}/C{slot:03d}"

        everything = stage.select(list(instance_paths))
        translation, rotation, scale = decompose(scene.instances.transform)
        everything.write("translate", translation)
        everything.write("orient", rotation)
        everything.write("scale", scale)
        lenses = stage.select(camera_paths)
        vertical, horizontal = apertures(scene.cameras.fov_y, FOCAL_LENGTH, workload.width, workload.height)
        lenses.write("focalLength", np.full(len(camera_paths), FOCAL_LENGTH, np.float32))
        lenses.write("verticalAperture", vertical.astype(np.float32))
        lenses.write("horizontalAperture", horizontal.astype(np.float32))
        lenses.write("clippingRange", np.stack([scene.cameras.near, scene.cameras.far], axis=1).astype(np.float32))

        self._moving = stage.select(list(instance_paths[scene.motion.instance])) if len(scene.motion.instance) else None
        if self._moving is not None:
            self._moving.write("scale", scene.motion.scale.astype(np.float32))
        self._lenses = lenses

        ernie.Scope.define(stage, "/Render")
        variables = []
        self._color_source = COLOR_SOURCES.get(workload.tier, SOURCES["rgb"])
        for output in workload.outputs:
            variables.append(f"/Render/{output}")
            ernie.RenderVar.define(stage, variables[-1], source_name=self._color_source if output == "rgb" else SOURCES[output])
        if "segmentation" in workload.outputs:
            variables.append("/Render/ids")
            ernie.RenderVar.define(stage, variables[-1], source_name="SemanticIdMap")
        ernie.RenderProduct.define(
            stage,
            PRODUCT,
            camera=camera_paths,
            resolution=(workload.width, workload.height),
            ordered_vars=variables,
            semantic_taxonomy=TAXONOMY,
            ambient_light_color=(1.0, 1.0, 1.0),
            ambient_light_intensity=float(scene.ambient),
            **SHADING[workload.tier],
        )
        self._stage = stage
        self._classes = scene.classes
        self._dt = workload.dt
        self._renderer = ernie.Renderer(stage)
        try:
            stream = cuda.Stream()
        except cuda.CudaUnavailable:
            return
        try:
            self._renderer.step([PRODUCT], self._dt, stream=stream.handle.value).wait(stream=stream.handle.value)
            stream.synchronize()
        except RuntimeError:
            stream.close()
            return
        finally:
            self._stage.time = 0.0
        self._stream = stream

    def update(self, state: State) -> None:
        if self._moving is not None:
            self._moving.write_poses(np.hstack([state.translation, state.rotation]))
        self._lenses.write_poses(np.hstack([state.camera_translation, state.camera_rotation]))
        self._stage.publish()

    def render(self) -> None:
        stream = self.stream()
        self._output = self._renderer.step([PRODUCT], self._dt, stream=stream)
        if stream is not None:
            self._output.wait(stream=stream)

    def sync(self) -> None:
        self._output.wait()

    def readback(self) -> dict[str, np.ndarray]:
        frame = self._output[PRODUCT].frames[0]
        frames = {}
        for output, source in SOURCES.items():
            if f"/Render/{output}" not in frame.render_vars:
                continue
            values = frame.source(self._color_source if output == "rgb" else source).numpy()
            if output == "rgb" and values.dtype == np.uint8:
                frames[output] = linear_from_srgb(values[..., :3])
            elif output == "rgb":
                frames[output] = np.ascontiguousarray(values[..., :3], np.float32)
            elif output == "depth":
                depth = values.reshape(*values.shape[:3], 1).astype(np.float32)
                frames[output] = np.where(hits(depth), depth, np.float32(np.inf))
            else:
                frames[output] = class_labels(frame.source("SemanticIdMap").numpy(), values.reshape(*values.shape[:3], 1), self._classes)
        return frames

    def stream(self) -> int | None:
        return self._stream.handle.value if self._stream else None

    def close(self) -> None:
        self._output = None
        self._renderer = None
        self._stage = None
        if self._stream:
            self._stream.close()
