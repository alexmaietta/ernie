"""Batched cameras for reinforcement learning: one camera renders a view per environment, each
from its own pose of its own world, and every step hands back the observations as GPU arrays.

    import ernie
    from ernie.batch import BatchCamera

    renderer = ernie.Renderer()
    scene = ...  # environments cloned under prims with tokens.WORLD 0, 1, ... (see Scene.clone)
    camera = BatchCamera(renderer, scene, "/World/Camera", views=4096, resolution=(64, 64),
                         render_vars=("LdrColor", "DistanceToCamera"))
    camera.set_worlds(numpy.arange(4096, dtype=numpy.int32))
    for step in range(steps):
        poses = simulator.camera_poses()            # (4096, 4, 4) float32, e.g. a CUDA tensor
        observations = camera.step(poses, stream=torch.cuda.current_stream().cuda_stream)
        rgb = torch.from_dlpack(observations["LdrColor"])[..., :3]  # (4096, 64, 64, 3) uint8

Nothing waits on the host: poses written as a contiguous float32 CUDA array stay on the GPU
(`tokens.POSES_DEVICE`), the render is submitted and the next step is prepared while it runs,
and the observations are ready for work queued on `stream` after the call. Each step's arrays
belong to that step's frame: keep them as long as needed, but drop them before many more steps,
or each step renders into a frame of its own.

Observation shapes: LdrColor (views, rows, cols, 4) uint8 sRGB; HdrColor, Electrons and Normal
(views, rows, cols, 4) float32; DistanceToCamera and DistanceToImagePlane (views, rows, cols)
float32; PrimId and InstanceId (views, rows, cols) uint32; RawDN (views, rows, cols, 4) uint16;
SpectralRadiance (views, bands, rows, cols) float32. Only the vars asked for are rendered and
held.

References:
- B. Shacklett et al., "An Extensible, Data-Oriented Architecture for High-Performance,
  Many-World Simulation", SIGGRAPH 2023 (Madrona's batch renderer).
  https://madrona-engine.github.io/
- M. Mittal et al., "Orbit: A Unified Simulation Framework for Interactive Robot Learning
  Environments", IEEE RA-L 2023 (Isaac Lab's tiled cameras). https://isaac-sim.github.io/IsaacLab/
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ._ernie import Device, Path, Renderer, Scene, tokens

__all__ = ["BatchCamera"]


class BatchCamera:
    """A batch of `views` views through the Camera prim `camera` of `scene`, rendered by
    `renderer` as the render product `product` at `resolution` (columns, rows) with
    `render_vars`.

    The camera keeps its lens and sensor; each view replaces its transform with a pose under the
    camera's parent (`tokens.POSES`) and sees one world (`tokens.WORLDS`) besides what every
    world shares, or every world for -1. A missing camera is defined with default settings. Until
    the first poses, every view takes the camera's own transform, and views see every world.
    """

    def __init__(
        self,
        renderer: Renderer,
        scene: Scene,
        camera: str | Path,
        views: int,
        *,
        resolution: tuple[int, int] = (64, 64),
        render_vars: Iterable[str] = ("LdrColor",),
        product: str | Path = "/Render/BatchCamera",
    ) -> None:
        if views < 1:
            raise ValueError("A BatchCamera needs at least one view")
        self.renderer = renderer
        self.scene = scene
        self.camera = Path(str(camera))
        self.product = Path(str(product))
        self.views = int(views)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.render_vars = tuple(render_vars)
        if not self.render_vars:
            raise ValueError("A BatchCamera needs at least one render var")

        if not scene.has(self.camera):
            scene.define(self.camera, tokens.CAMERA)
        # Poses replace the camera's own transform, under its parent.
        transform = scene.get(self.camera, tokens.TRANSFORM)
        pose = np.eye(4) if transform is None else np.asarray(transform, np.float64)
        scene.write(self.camera, tokens.POSES, np.repeat(pose[None].astype(np.float32), views, 0))
        scene.write(self.camera, tokens.WORLDS, np.full(views, -1, np.int32))

        self._paths: dict[str, str] = {}
        var_paths = []
        for name in self.render_vars:
            path = Path(f"{self.product}/Vars/{name}")
            scene.define(path, tokens.RENDER_VAR).set(tokens.SOURCE_NAME, name)
            var_paths.append(path)
            self._paths[name] = str(path)
        scene.define(self.product, tokens.RENDER_PRODUCT).set(tokens.CAMERA_REL, self.camera).set(
            tokens.RESOLUTION, self.resolution
        ).set(tokens.ORDERED_VARS, var_paths)

    def set_poses(self, poses: Any, stream: Any = None) -> None:
        """Sets every view's camera-to-world pose under the camera's parent: a (views, 4, 4)
        array in the math convention (translation in the last column) from any DLPack framework.
        Contiguous float32 CUDA arrays stay on the GPU, copied after the work queued on `stream`.
        """
        self._check_views(poses, "poses")
        self.scene.write(self.camera, tokens.POSES, poses, stream=stream)

    def set_worlds(self, worlds: Any, stream: Any = None) -> None:
        """Sets the world each view sees (`tokens.WORLD` of the environments' prims): a (views,)
        int32 array, -1 for every world. A CUDA array is copied to the host once per write, after
        the work queued on `stream`, for planning the views' shadows."""
        self._check_views(worlds, "worlds")
        self.scene.write(self.camera, tokens.WORLDS, worlds, stream=stream)

    def step(
        self, poses: Any = None, worlds: Any = None, *, stream: Any = None, device: str = "cuda"
    ) -> Mapping[str, Any]:
        """Renders a step, after setting `poses` and `worlds` when given, and returns each render
        var's observations by name.

        With device="cuda", the arrays are in GPU memory (DLPack, e.g. torch.from_dlpack) and the
        call returns once the step is submitted: work queued on CUDA stream `stream` (a CUstream
        handle or DLPack stream number, default the legacy stream) after it sees the finished
        render. With device="cpu", it waits for the render and returns NumPy arrays.
        """
        if poses is not None:
            self.set_poses(poses, stream)
        if worlds is not None:
            self.set_worlds(worlds, stream)
        if device == "cuda":
            target = Device.CUDA
        elif device == "cpu":
            target = Device.CPU
        else:
            raise ValueError(f'device must be "cuda" or "cpu", not {device!r}')
        operation = self.renderer.step_async(self.scene, render_products=[self.product])
        frame = operation.outputs()[str(self.product)].frames[0]
        return {
            name: frame.render_vars[path].map(device=target, sync_stream=stream)
            for name, path in self._paths.items()
        }

    def _check_views(self, array: Any, what: str) -> None:
        shape = getattr(array, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) != self.views:
            raise ValueError(f"{what} must have {self.views} rows, one per view, not {shape[0]}")
