# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from contextlib import ExitStack

import numpy as np

from ernie_bench.backends.base import SUN_ANGLE, TAXONOMY, BackendInfo, Capabilities, apertures, class_labels, hits, linear_from_srgb, sun_basis
from ernie_bench.scene import Scene, State, compose
from ernie_bench.config import OUTPUTS, TIERS, Workload

FOCAL_LENGTH = 1.0
SOURCES = {"rgb": "LdrColor", "depth": "DistanceToCameraSD", "segmentation": "SemanticSegmentation"}
SHADING = {
    "depth": {"omni:rtx:rendermode": '"MinimalRendering"', "omni:rtx:minimal:mode": "0"},
    "segmentation": {"omni:rtx:rendermode": '"MinimalRendering"', "omni:rtx:minimal:mode": "0"},
    "direct": {"omni:rtx:rendermode": '"MinimalRendering"', "omni:rtx:minimal:mode": "2", "omni:rtx:minimal:castShadows": "true"},
    "pathtraced": {"omni:rtx:rendermode": '"PathTracing"', "omni:rtx:pt:limits:maxBounces": "4", "omni:rtx:pt:denoising:enabled": "false"},
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


def world_path(world: int) -> str:
    return f"/World/W{world:05d}"


def tuples(values: np.ndarray) -> str:
    return ", ".join("(" + ", ".join(f"{v:.9g}" for v in row) + ")" for row in np.asarray(values).reshape(len(values), -1))


def usd_matrix(matrix: np.ndarray) -> str:
    return "(" + tuples(np.asarray(matrix, np.float64).T) + ")"


def tile_grid(count: int) -> tuple[int, int]:
    columns = int(np.ceil(np.sqrt(count)))
    return columns, int(np.ceil(count / columns))


def stage_usda(scene: Scene, workload: Workload, tiled: bool = False) -> tuple[str, list[str], list[str], list[str]]:
    lines = ['#usda 1.0', '(', '    defaultPrim = "World"', "    metersPerUnit = 1", '    upAxis = "Z"', ")", 'class Scope "Prototypes"', "{"]
    for index, mesh in enumerate(scene.meshes):
        lines += [
            f'    def Mesh "M{index:03d}"',
            "    {",
            f"        point3f[] points = [{tuples(mesh.positions)}]",
            f"        normal3f[] normals = [{tuples(mesh.normals)}] (",
            '            interpolation = "vertex"',
            "        )",
            f"        int[] faceVertexCounts = [{', '.join(['3'] * len(mesh.indices))}]",
            f"        int[] faceVertexIndices = [{', '.join(str(i) for i in mesh.indices.reshape(-1))}]",
            '        uniform token subdivisionScheme = "none"',
            "    }",
        ]
    lines += ["}", 'def Xform "World"', "{"]
    for slot, light in enumerate(scene.lights):
        matrix = np.eye(4)
        matrix[:3, :3] = sun_basis(light.direction)
        lines += [
            f'    def DistantLight "L{slot:02d}"',
            "    {",
            f"        float inputs:angle = {SUN_ANGLE}",
            f"        float inputs:intensity = {UNIT_SUN_INTENSITY * light.irradiance / np.pi:.9g}",
            f"        color3f inputs:color = ({', '.join(f'{c:.9g}' for c in light.color)})",
            f"        matrix4d xformOp:transform = {usd_matrix(matrix)}",
            '        uniform token[] xformOpOrder = ["xformOp:transform"]',
            "    }",
        ]
    instance_paths = np.empty(len(scene.instances), object)
    camera_paths = [""] * len(scene.cameras)
    vertical, horizontal = apertures(scene.cameras.fov_y, FOCAL_LENGTH, workload.width, workload.height)
    for world in range(scene.world_count):
        root = world_path(world)
        partition = f"W{world:05d}"
        lines += [f'    def Xform "{partition}"', "    {", f'        token primvars:omni:scenePartition = "{partition}" (', '            interpolation = "constant"', "        )"]
        for slot, index in enumerate(scene.members[world]):
            instance_paths[index] = f"{root}/I{slot:05d}"
            lines += [
                f'        def Mesh "I{slot:05d}" (',
                f"            prepend references = </Prototypes/M{scene.instances.mesh[index]:03d}>",
                f'            prepend apiSchemas = ["SemanticsAPI:{TAXONOMY}"]',
                "        )",
                "        {",
                f"            color3f[] primvars:displayColor = [({', '.join(f'{c:.9g}' for c in scene.colors[index])})]",
                f"            matrix4d xformOp:transform = {usd_matrix(scene.instances.transform[index])}",
                '            uniform token[] xformOpOrder = ["xformOp:transform"]',
                f'            string semantic:{TAXONOMY}:params:semanticData = "{scene.classes[scene.instances.semantic[index]]}"',
                f'            string semantic:{TAXONOMY}:params:semanticType = "{TAXONOMY}"',
                "        }",
            ]
        for slot, index in enumerate(scene.views[world]):
            camera_paths[index] = f"{root}/C{slot:03d}"
            lines += [
                f'        def Camera "C{slot:03d}"',
                "        {",
                f'            custom token omni:scenePartition = "{partition}"',
                f"            float focalLength = {FOCAL_LENGTH}",
                f"            float verticalAperture = {vertical[index]:.9g}",
                f"            float horizontalAperture = {horizontal[index]:.9g}",
                f"            float2 clippingRange = ({scene.cameras.near[index]:.9g}, {scene.cameras.far[index]:.9g})",
                f"            matrix4d xformOp:transform = {usd_matrix(np.eye(4))}",
                '            uniform token[] xformOpOrder = ["xformOp:transform"]',
                "        }",
            ]
        lines += ["    }"]
    lines += ["}", 'def Scope "Render"', "{"]
    settings = dict(SHADING[workload.tier])
    if workload.tier == "pathtraced":
        settings["omni:rtx:pt:samplesPerPixel"] = str(workload.spp)
    variables = [output for output in OUTPUTS if output in workload.outputs]
    names = variables + (["ids"] if "segmentation" in variables else [])
    columns, rows = tile_grid(len(camera_paths))
    groups = [("Tiled", camera_paths, (columns * workload.width, rows * workload.height))] if tiled else [
        (f"P{index:05d}", [camera], (workload.width, workload.height)) for index, camera in enumerate(camera_paths)
    ]
    product_paths = []
    for product, targets, (width, height) in groups:
        product_paths.append(f"/Render/{product}")
        lines += [
            f'    def RenderProduct "{product}"',
            "    {",
            f"        rel camera = [{', '.join(f'<{camera}>' for camera in targets)}]",
            f"        int2 resolution = ({width}, {height})",
            f"        rel orderedVars = [{', '.join(f'<{name}>' for name in names)}]",
        ]
        lines += [f"        {TYPES[key]} {key} = {value}" for key, value in settings.items()]
        for name in names:
            source = COLOR_SOURCES.get(workload.tier, SOURCES["rgb"]) if name == "rgb" else SOURCES.get(name, "SemanticIdMap")
            lines += [f'        def RenderVar "{name}"', "        {", f'            string sourceName = "{source}"', "        }"]
        lines += ["    }"]
    lines += ["}", ""]
    return "\n".join(lines), list(instance_paths[scene.motion.instance]), camera_paths, product_paths


class OvrtxBackend:
    def __init__(self, tiled: bool = False):
        import ovrtx
        import ovstage

        self._tiled = tiled
        self._ovrtx = ovrtx
        self._ovstage = ovstage
        cache = os.environ.get("ERNIE_BENCH_CACHE")
        options = {"keep_system_alive": True, "enable_crash_reporter": False} | ({"cache_root_path": cache} if cache else {})
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
        usda, moving, cameras, products = stage_usda(scene, workload, self._tiled)
        self._stage = ovstage.Stage("ernie-bench")
        self._renderer.attach_ovstage(self._stage)
        self._ordinal = 1
        ovstage.population.open_usd_from_string(self._stage, usda, ordinal=self._ordinal)
        self._stage.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()
        paths = self._resources.enter_context(ovstage.PathDictionary(self._stage))
        self._attribute = paths.intern_token("omni:xform")
        self._queries = []
        for selected in (moving, cameras):
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
