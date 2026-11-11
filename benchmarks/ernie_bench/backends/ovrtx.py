# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ernie_bench.backends.base import SUN_ANGLE, TAXONOMY, BackendInfo, Capabilities, apertures, class_labels, hits, linear_from_srgb, sun_basis
from ernie_bench.scene import Scene, State, compose
from ernie_bench.config import OUTPUTS, TIERS, Workload

FOCAL_LENGTH = 1.0
SOURCES = {"rgb": "LdrColor", "depth": "DistanceToCameraSD", "segmentation": "SemanticSegmentation"}
SHADING = {
    "depth": {"omni:rtx:rendermode": "MinimalRendering", "omni:rtx:minimal:mode": 0},
    "segmentation": {"omni:rtx:rendermode": "MinimalRendering", "omni:rtx:minimal:mode": 0},
    "direct": {"omni:rtx:rendermode": "MinimalRendering", "omni:rtx:minimal:mode": 2, "omni:rtx:minimal:castShadows": True},
    "pathtraced": {"omni:rtx:rendermode": "PathTracing", "omni:rtx:pt:limits:maxBounces": 4, "omni:rtx:pt:denoising:enabled": False},
}
TYPES = {
    "omni:rtx:rendermode": "token",
    "omni:rtx:minimal:mode": "int",
    "omni:rtx:minimal:castShadows": "bool",
    "omni:rtx:pt:limits:maxBounces": "uint",
    "omni:rtx:pt:denoising:enabled": "bool",
    "omni:rtx:pt:samplesPerPixel": "uint",
}
UNIT_SUN_INTENSITY = 1000.0
COLOR_SOURCES = {"pathtraced": "HdrColor"}


def tile_grid(count: int) -> tuple[int, int]:
    columns = int(np.ceil(np.sqrt(count)))
    return columns, int(np.ceil(count / columns))


def author_layer(scene: Scene, workload: Workload, tiled: bool = False):
    from pxr import Gf, Sdf, Vt

    names = Sdf.ValueTypeNames
    layer = Sdf.Layer.CreateAnonymous(".usda")

    def define(path: str, type_name: str, specifier=Sdf.SpecifierDef):
        return Sdf.PrimSpec(layer.GetPrimAtPath(path.rpartition("/")[0] or "/"), path.rpartition("/")[2], specifier, type_name)

    def attribute(prim, name: str, type_name, value, custom: bool = False, variability=Sdf.VariabilityVarying):
        spec = Sdf.AttributeSpec(prim, name, type_name, variability, custom)
        spec.default = value
        return spec

    def transform(prim, matrix: np.ndarray) -> None:
        attribute(prim, "xformOp:transform", names.Matrix4d, Gf.Matrix4d(*np.asarray(matrix, np.float64).T.ravel().tolist()))
        attribute(prim, "xformOpOrder", names.TokenArray, Vt.TokenArray(["xformOp:transform"]), variability=Sdf.VariabilityUniform)

    with Sdf.ChangeBlock():
        layer.defaultPrim = "World"
        layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
        layer.pseudoRoot.SetInfo("upAxis", "Z")
        define("/Prototypes", "Scope", Sdf.SpecifierClass)
        for index, mesh in enumerate(scene.meshes):
            prototype = define(f"/Prototypes/M{index:03d}", "Mesh")
            attribute(prototype, "points", names.Point3fArray, Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.positions, np.float32)))
            normals = attribute(prototype, "normals", names.Normal3fArray, Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.normals, np.float32)))
            normals.SetInfo("interpolation", "vertex")
            attribute(prototype, "faceVertexCounts", names.IntArray, Vt.IntArray([3] * len(mesh.indices)))
            attribute(prototype, "faceVertexIndices", names.IntArray, Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.indices.reshape(-1), np.int32)))
            attribute(prototype, "subdivisionScheme", names.Token, "none", variability=Sdf.VariabilityUniform)
        define("/World", "Xform")
        for slot, light in enumerate(scene.lights):
            matrix = np.eye(4)
            matrix[:3, :3] = sun_basis(light.direction)
            sun = define(f"/World/L{slot:02d}", "DistantLight")
            attribute(sun, "inputs:angle", names.Float, SUN_ANGLE)
            attribute(sun, "inputs:intensity", names.Float, UNIT_SUN_INTENSITY * light.irradiance / np.pi)
            attribute(sun, "inputs:color", names.Color3f, Gf.Vec3f(*map(float, light.color)))
            transform(sun, matrix)
        instance_paths = np.empty(len(scene.instances), object)
        camera_paths = [""] * len(scene.cameras)
        vertical, horizontal = apertures(scene.cameras.fov_y, FOCAL_LENGTH, workload.width, workload.height)
        semantics = Sdf.TokenListOp.Create(prependedItems=[f"SemanticsAPI:{TAXONOMY}"])
        for world in range(scene.world_count):
            partition = f"W{world:05d}"
            root = f"/World/{partition}"
            attribute(define(root, "Xform"), "primvars:omni:scenePartition", names.Token, partition).SetInfo("interpolation", "constant")
            for slot, index in enumerate(scene.members[world]):
                instance_paths[index] = f"{root}/I{slot:05d}"
                instance = define(instance_paths[index], "Mesh")
                instance.referenceList.Prepend(Sdf.Reference(primPath=f"/Prototypes/M{scene.instances.mesh[index]:03d}"))
                instance.SetInfo("apiSchemas", semantics)
                attribute(instance, "primvars:displayColor", names.Color3fArray, Vt.Vec3fArray([Gf.Vec3f(*map(float, scene.colors[index]))]))
                transform(instance, scene.instances.transform[index])
                attribute(instance, f"semantic:{TAXONOMY}:params:semanticData", names.String, scene.classes[scene.instances.semantic[index]])
                attribute(instance, f"semantic:{TAXONOMY}:params:semanticType", names.String, TAXONOMY)
            for slot, index in enumerate(scene.views[world]):
                camera_paths[index] = f"{root}/C{slot:03d}"
                camera = define(camera_paths[index], "Camera")
                attribute(camera, "omni:scenePartition", names.Token, partition, custom=True)
                attribute(camera, "focalLength", names.Float, FOCAL_LENGTH)
                attribute(camera, "verticalAperture", names.Float, float(vertical[index]))
                attribute(camera, "horizontalAperture", names.Float, float(horizontal[index]))
                attribute(camera, "clippingRange", names.Float2, Gf.Vec2f(float(scene.cameras.near[index]), float(scene.cameras.far[index])))
                transform(camera, np.eye(4))
        define("/Render", "Scope")
        settings = dict(SHADING[workload.tier])
        if workload.tier == "pathtraced":
            settings["omni:rtx:pt:samplesPerPixel"] = workload.spp
        variables = [output for output in OUTPUTS if output in workload.outputs]
        outputs = variables + (["ids"] if "segmentation" in variables else [])
        columns, rows = tile_grid(len(camera_paths))
        groups = [("Tiled", camera_paths, (columns * workload.width, rows * workload.height))] if tiled else [
            (f"P{index:05d}", [camera], (workload.width, workload.height)) for index, camera in enumerate(camera_paths)
        ]
        product_paths = []
        for product, targets, (width, height) in groups:
            path = f"/Render/{product}"
            product_paths.append(path)
            render = define(path, "RenderProduct")
            Sdf.RelationshipSpec(render, "camera").targetPathList.explicitItems = targets
            attribute(render, "resolution", names.Int2, Gf.Vec2i(width, height), variability=Sdf.VariabilityUniform)
            Sdf.RelationshipSpec(render, "orderedVars").targetPathList.explicitItems = [f"{path}/{name}" for name in outputs]
            for key, value in settings.items():
                attribute(render, key, names.Find(TYPES[key]), value)
            for name in outputs:
                source = COLOR_SOURCES.get(workload.tier, SOURCES["rgb"]) if name == "rgb" else SOURCES.get(name, "SemanticIdMap")
                attribute(define(f"{path}/{name}", "RenderVar"), "sourceName", names.String, source, variability=Sdf.VariabilityUniform)
    return layer, list(instance_paths[scene.motion.instance]), camera_paths, product_paths


class OvrtxBackend:
    def __init__(self, tiled: bool = False):
        import ovrtx
        import ovstage

        self._tiled = tiled
        self._ovrtx = ovrtx
        self._ovstage = ovstage
        cache = os.environ.get("ERNIE_BENCH_CACHE")
        options = {"keep_system_alive": True, "enable_crash_reporter": False, "read_gpu_transforms": True} | ({"cache_root_path": cache} if cache else {})
        self._renderer = ovrtx.Renderer(ovrtx.RendererConfig(**options))
        self._stage = None
        self._pending = None
        self._output = None
        self._resources = ExitStack()

    def info(self) -> BackendInfo:
        return BackendInfo(
            "ovrtx",
            self._ovrtx.__version__,
            "cuda:0",
            {"products": "tiled" if self._tiled else "per camera", "deviations": "aces tone curve on LdrColor, no ambient, no unlit"},
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(frozenset(TIERS) - {"unlit"}, frozenset(OUTPUTS))

    def load(self, scene: Scene, workload: Workload) -> None:
        ovstage = self._ovstage
        layer, moving, cameras, products = author_layer(scene, workload, self._tiled)
        path = Path(self._resources.enter_context(TemporaryDirectory(prefix="ernie-bench-ovrtx-"))) / "stage.usda"
        layer.Export(str(path))
        self._stage = ovstage.Stage("ernie-bench")
        self._renderer.attach_ovstage(self._stage)
        self._ordinal = 1
        ovstage.population.open_usd(self._stage, str(path), ordinal=self._ordinal)
        self._stage.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()
        paths = self._resources.enter_context(ovstage.PathDictionary(self._stage))
        self._attribute = paths.intern_token("omni:xform")
        self._queries = []
        for selected in (moving, cameras):
            if not selected:
                self._queries.append((None, 0))
                continue
            path_list = paths.create_path_list_from_strings(selected)
            self._resources.callback(paths.destroy_path_list, path_list)
            self._queries.append((self._resources.enter_context(self._stage.query_from_path_list(path_list)), len(selected)))
        self._matrix = ovstage.numpy_to_dldatatype(np.dtype(np.float64), lanes=16)
        self._products = set(products)
        self._product_order = products
        self._grid = tile_grid(len(cameras))
        self._cameras = len(cameras)
        self._size = (workload.height, workload.width)
        self._classes = scene.classes
        self._outputs = [output for output in OUTPUTS if output in workload.outputs]
        self._tier = workload.tier
        self._dt = workload.dt

    def update(self, state: State) -> None:
        ovstage = self._ovstage
        self._ordinal += 1
        for (query, count), matrices in zip(self._queries, (state.transform, state.camera)):
            if query is None:
                continue
            rows = np.ascontiguousarray(np.transpose(matrices, (0, 2, 1)), np.float64)
            tensor = ovstage.make_dltensor(rows, dtype=self._matrix, shape=[count], ndim=1)
            self._stage.write_attribute(query, self._attribute, ordinal=self._ordinal, tensors=tensor, is_array=False, semantic=ovstage.AttributeSemantic.MATRIX).wait()
        self._stage.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()

    def render(self) -> None:
        self._output = None
        self._pending = self._renderer.step_async(self._products, self._dt, ordinal=self._ordinal)

    def sync(self) -> None:
        self._output = self._pending.wait().fetch()
        self._pending = None

    def readback(self) -> dict[str, np.ndarray]:
        def mapped(frame, product: str, name: str) -> np.ndarray:
            var = frame.render_vars[f"{product}/{name}"].map(device=self._ovrtx.Device.CPU)
            array = np.from_dlpack(var).copy()
            var.unmap()
            return array

        def tiles(values: np.ndarray) -> list[np.ndarray]:
            if not self._tiled:
                return [values]
            height, width = self._size
            columns, _ = self._grid
            return [values[(i // columns) * height : (i // columns + 1) * height, (i % columns) * width : (i % columns + 1) * width] for i in range(self._cameras)]

        frames = {output: [] for output in self._outputs}
        for product in self._product_order:
            frame = self._output[product].frames[0]
            for output in self._outputs:
                for values in tiles(mapped(frame, product, output)):
                    if output == "rgb" and self._tier == "pathtraced":
                        frames[output].append(values[..., :3].astype(np.float32))
                    elif output == "rgb":
                        frames[output].append(linear_from_srgb(values[..., :3]))
                    elif output == "depth":
                        frames[output].append(np.where(hits(values), values, np.inf).astype(np.float32))
                    else:
                        frames[output].append(class_labels(mapped(frame, product, "ids"), values, self._classes))
        return {output: np.stack(images) for output, images in frames.items()}

    def stream(self) -> int | None:
        return None

    def close(self) -> None:
        self._output = None
        self._pending = None
        self._resources.close()
        if self._stage is not None:
            self._renderer.detach_ovstage()
            self._stage.destroy()
            self._stage = None
        self._renderer.destroy()
