"""A lunar plain cratered by impacts as you watch: Warp kernels make the terrain and carve every
crater on the GPU, and Ernie rebuilds only the patch each one changed.

    PYTHONPATH=build/python python demos/python/demo_gpu_terrain_impacts.py [--capture impacts.png]

Needs Warp (`uv pip install warp-lang`) and CUDA interop. The heights live in one Warp array the
whole time: a kernel fills it with rolling fractal relief, and each impact's kernel carves a
bowl with a raised rim and an ejecta blanket into it. Each frame `scene.write` hands Ernie the
array on the kernels' stream along with `ernie:heights:dirty`, the rectangle of cells the impacts
touched, so the renderer copies the heights device to device and recomputes the gradients,
their mips and the level-of-detail bounds of that rectangle alone, on the GPU. The host never
reads the heights.

References:
- H. J. Melosh, "Impact Cratering: A Geologic Process", Oxford University Press, 1989 (crater
  depth about a fifth of the diameter for simple craters, a raised rim, ejecta thinning with the
  cube of the distance).
- T. R. McGetchin, M. Settle and J. W. Head, "Radial Thickness Variation in Impact Crater
  Ejecta", Earth and Planetary Science Letters 20(2):226-236, 1973.
"""

import argparse
import math

import numpy as np
import warp as wp

import ernie
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


def build_scene(heights):
    scene = ernie.Scene()
    regolith = ernie.Path("/Looks/Regolith")
    scene.define(regolith, tokens.MATERIAL).set(tokens.BRDF, "hapke").set(
        tokens.ALBEDO, (0.25, 0.25, 0.25))
    extent = CELLS * CELL
    terrain = ernie.Path("/World/Terrain")
    scene.define(terrain, tokens.TERRAIN).set(tokens.ORIGIN, (-0.5 * extent, 0.5 * extent)).set(
        tokens.CELL_SIZE, CELL).set(tokens.MATERIAL_BINDING, regolith)
    scene.write(terrain, tokens.HEIGHTS, heights)
    # A low sun, so the craters cast long shadows.
    altitude, azimuth = math.radians(12.0), math.radians(250.0)
    towards = (math.cos(altitude) * math.sin(azimuth), math.cos(altitude) * math.cos(azimuth),
               math.sin(altitude))
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform(towards)).set(tokens.ANGLE, 0.53)
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.0, -150.0, 70.0), (0.0, 0.0, 0.0))).set(
        tokens.FOCAL_LENGTH, 24.0)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(tokens.AMBIENT, 0.02).set(
        tokens.COLOR_MODE, tokens.HILLSHADE)
    return scene


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", help="render without a window and save the image here")
    parser.add_argument("--rate", type=int, default=3, help="impacts per frame")
    args = parser.parse_args()

    wp.init()
    device = "cuda:0"
    stream = wp.get_stream(device)
    heights = wp.zeros((CELLS, CELLS), dtype=float, device=device)
    wp.launch(relief, dim=(CELLS, CELLS), inputs=[heights, 7], device=device)
    scene = build_scene(heights)
    impacts = Impacts()
    terrain = ernie.Path("/World/Terrain")

    from ernie.app import Viewer
    viewer = Viewer(scene, title="Ernie: impacts carved by Warp kernels",
                    hidden=args.capture is not None)

    def update(v, info):
        dirty = carve(heights, impacts, args.rate, device)
        if dirty[2] > 0 and dirty[3] > 0:
            v.scene.set(terrain, tokens.HEIGHTS_DIRTY, [float(x) for x in dirty])
            v.scene.write(terrain, tokens.HEIGHTS, heights, stream=stream)

    viewer.active_camera = ernie.Path("/World/Camera")
    if args.capture is not None:
        display = viewer.viewport_display
        display.lights = display.grid = display.cameras = False
        viewer.viewport_display = display
    viewer.on_frame(update)
    if args.capture is None:
        viewer.run()
        return
    for _ in range(120):
        viewer.step()
    viewer.capture(args.capture)
    viewer.step()


if __name__ == "__main__":
    main()
