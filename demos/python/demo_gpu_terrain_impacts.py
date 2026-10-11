# SPDX-License-Identifier: BSD-3-Clause
"""A lunar plain cratered by impacts as you watch: Warp kernels make the terrain and carve every
crater on the GPU, and Ernie rebuilds only the patch each one changed.

    PYTHONPATH=build/python python demos/python/demo_gpu_terrain_impacts.py [--rate 3]
        [--viewer null --output impacts.png]

Needs Warp (`uv pip install warp-lang`) and CUDA interop. A kernel fills a Warp array with
rolling fractal relief, written once as the terrain's heights. From then on each frame maps the
renderer's own copy of the heights (`Viewer.map` or `Renderer.map`, a CUDA array over Ernie's
memory) and each impact's kernel carves a bowl with a raised rim and an ejecta blanket straight
into it; unmapping on the kernels' stream with `ernie:heights:dirty`, the rectangle of cells the
impacts touched, has the renderer recompute the gradients, their mips and the level-of-detail
bounds of that rectangle alone, on the GPU. Nothing is copied, and the host never reads the heights.

References:
- H. J. Melosh, "Impact Cratering: A Geologic Process", Oxford University Press, 1989 (crater
  depth about a fifth of the diameter for simple craters, a raised rim, ejecta thinning with the
  cube of the distance).
- T. R. McGetchin, M. Settle and J. W. Head, "Radial Thickness Variation in Impact Crater
  Ejecta", Earth and Planetary Science Letters 20(2):226-236, 1973.
"""

import math

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

CELLS = 1024  # cells along each side
CELL = 0.25  # metres
EJECTA = 2.5  # radii the ejecta blanket reaches


@wp.kernel
def relief(heights: wp.array2d(dtype=float), seed: int):
    """Rolling fractal relief: five octaves of Perlin noise."""
    row, col = wp.tid()
    state = wp.rand_init(seed)
    p = wp.vec2(float(col), float(row)) * CELL
    height = float(0.0)
    scale = float(1.0 / 60.0)
    amplitude = float(2.5)
    for _ in range(5):
        height += amplitude * wp.noise(state, p * scale)
        scale *= 2.1
        amplitude *= 0.45
    heights[row, col] = height


@wp.kernel
def impact(heights: wp.array2d(dtype=float), first: wp.vec2i, centre: wp.vec2, radius: float):
    """A simple crater of `radius` metres at `centre` (in cells): a bowl a fifth of its diameter
    deep, a rim 4% of it high, and ejecta thinning as the cube of the distance beyond."""
    dy, dx = wp.tid()
    row = first[1] + dy
    col = first[0] + dx
    if row < 0 or row >= CELLS or col < 0 or col >= CELLS:
        return
    d = wp.length(wp.vec2(float(col), float(row)) - centre) * CELL / radius
    if d > EJECTA:
        return
    depth = 0.4 * radius
    rim = 0.04 * radius
    change = float(0.0)
    if d < 1.0:
        change = depth * (d * d - 1.0) + rim
    else:
        # Thinning as the cube of the distance, eased to nothing at the blanket's edge.
        edge = 1.0 / (EJECTA * EJECTA * EJECTA)
        change = rim * (1.0 / (d * d * d) - edge) / (1.0 - edge)
    heights[row, col] = heights[row, col] + change


def build_scene(scene, heights):
    regolith = scene.define("/Looks/Regolith", ernie.Material, brdf="hapke",
                            albedo=(0.25, 0.25, 0.25))
    extent = CELLS * CELL
    terrain = scene.define("/World/Terrain", ernie.Terrain, origin=(-0.5 * extent, 0.5 * extent),
                           cell_size=CELL, material_binding=regolith)
    # The relief kernel's Warp array as the heights.
    terrain.write(tokens.HEIGHTS, heights)
    # A low sun, so the craters cast long shadows.
    altitude, azimuth = math.radians(12.0), math.radians(250.0)
    towards = (math.cos(altitude) * math.sin(azimuth), math.cos(altitude) * math.cos(azimuth),
               math.sin(altitude))
    scene.define("/World/Sun", ernie.DistantLight, angle=0.53).set(
        tokens.TRANSFORM, ernie.distant_light_transform(towards))
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=24.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.0, -150.0, 70.0), (0.0, 0.0, 0.0)))
    scene.define("/Render/Settings", ernie.RenderSettings, camera=camera, ambient=0.02,
                 color_mode=tokens.HILLSHADE)
    return terrain, camera


class Impacts:
    """Random impacts with a power-law size distribution: many small craters, few large ones."""

    def __init__(self, seed=3):
        self.random = np.random.default_rng(seed)

    def next(self):
        # Cumulative number above a radius falling as its square, from 1 m to 25 m.
        radius = 1.0 / math.sqrt(self.random.uniform(1.0 / 25.0**2, 1.0))
        centre = self.random.uniform(0.0, CELLS, size=2)
        return centre, radius


def carve(heights, impacts, count, stream_device):
    """Carves `count` impacts and returns the rectangle of cells they touched, as
    (first column, first row, columns, rows)."""
    low = np.array([CELLS, CELLS])
    high = np.array([0, 0])
    for _ in range(count):
        centre, radius = impacts.next()
        reach = int(math.ceil(EJECTA * radius / CELL)) + 1
        first = np.floor(centre).astype(int) - reach
        size = 2 * reach + 1
        wp.launch(impact, dim=(size, size),
                  inputs=[heights, wp.vec2i(*first), wp.vec2(*centre), radius],
                  device=stream_device)
        low = np.minimum(low, np.clip(first, 0, CELLS))
        high = np.maximum(high, np.clip(first + size, 0, CELLS))
    return (int(low[0]), int(low[1]), int(high[0] - low[0]), int(high[1] - low[1]))


class Demo(ernie.demo.Demo):
    num_frames = 120

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--rate", type=int, default=3, help="impacts per frame")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        self.device = "cuda:0"
        self.stream = wp.get_stream(self.device)
        heights = wp.zeros((CELLS, CELLS), dtype=float, device=self.device)
        wp.launch(relief, dim=(CELLS, CELLS), inputs=[heights, 7], device=self.device)
        self.relief = heights.numpy()
        self.terrain, camera = build_scene(self.scene, heights)
        self.impacts = Impacts()
        self.carved = 0
        if viewer is not None:
            viewer.active_camera = camera.path
            if args.output:
                display = viewer.viewport_display
                display.lights = display.grid = display.cameras = False
                viewer.viewport_display = display

    def step(self):
        # Ernie's own copy of the heights, carved straight into: nothing is copied at all.
        path = self.terrain.path
        mapped = wp.from_dlpack(self.map(path, tokens.HEIGHTS, stream=self.stream))
        dirty = carve(mapped, self.impacts, self.args.rate, self.device)
        self.scene.set(path, tokens.HEIGHTS_DIRTY, [float(x) for x in dirty])
        self.scene.unmap(path, tokens.HEIGHTS, stream=self.stream)
        self.carved += self.args.rate

    def test_final(self):
        # The renderer's heights, read back once for the test.
        path = self.terrain.path
        heights = wp.from_dlpack(self.map(path, tokens.HEIGHTS, stream=self.stream)).numpy()
        self.scene.unmap(path, tokens.HEIGHTS, stream=self.stream)
        changed = np.abs(heights - self.relief) > 0.01
        assert self.carved > 0 and changed.any(), "the impacts should carve the terrain"
        assert changed.mean() < 0.5, "the craters should change patches, not the whole plain"
        image = self.render_image()[..., :3]
        assert image.std() > 1.0, "the hillshaded terrain should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
