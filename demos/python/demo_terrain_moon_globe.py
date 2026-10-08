"""The whole Moon as one terrain: a Moon-sized body of cratered relief drawn with planet-scale
level of detail, from orbit down to a few kilometres above the ground.

    PYTHONPATH=build/python python demos/python/demo_terrain_moon_globe.py
    PYTHONPATH=build/python python demos/python/demo_terrain_moon_globe.py --output moon.png
    PYTHONPATH=build/python python demos/python/demo_terrain_moon_globe.py \
        --textures ~/Downloads/moon_01_4k.blend.zip

A Terrain with `ernie:radius` is a planet: its heights are six cube faces (Ernie's
equiangular cube sphere, faces 0 to 3 round the equator from longitude 0 eastwards, 4 the north
pole and 5 the south), each a grid of (n + 1) x (n + 1) heights above the radius. Each frame the
renderer picks patches from all six faces at continuous levels of detail, culls those behind the
horizon or out of view, and draws them relative to the camera, so the body stays steady at
centimetre camera moves although it is 3,475 km across.

The relief is illustrative: a few low basins and several thousand bowl craters with raised
rims, their sizes drawn from a cumulative power law and the larger ones' depths capped. Shading uses Hapke's lunar BRDF; with
Poly Haven's moon_01 scan (CC0, https://polyhaven.com/a/moon_01) as detail textures, the regolith
gains texture close up. Without --output the viewer opens, starting in orbit.

References:
- NASA NSSDCA, "Moon Fact Sheet", updated 11 January 2024 (volumetric mean radius 1737.4 km).
  https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
- R. J. Pike, "Size-dependence in the shape of fresh impact craters on the Moon", in Impact and
  Explosion Cratering, Pergamon, 1977, 489-509 (depth about 0.2 and rim height about 0.04 of the
  diameter for simple craters).
- C. Ronchi, R. Iacono and P. S. Paolucci, "The 'Cubed Sphere'", Journal of Computational Physics
  124(1):93-114, 1996 (the equiangular cube sphere).
"""

import argparse
import math
import os
import tempfile

import numpy as np

import ernie
from ernie import tokens

from demo_support import side_by_side, write_png

RADIUS = 1737400.0  # metres, the Moon's mean radius
FACE_CELLS = 1024   # grid quads along each cube face, about 2.7 km each
SIZE = (960, 540)
SUN = np.array([0.35, 0.92, 0.18]) / np.linalg.norm([0.35, 0.92, 0.18])  # towards the Sun

# Each face's normal, column axis and row axis: Ernie's CubeSphereLod.
FRAMES = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
], dtype=np.float64)


def plane_offset(s):
    """tan(pi s / 4), exact at the face's edges, so vertices shared by two faces agree."""
    return np.where(np.abs(s) == 1.0, s, np.tan(0.25 * np.pi * s))


def face_directions(face, cols, rows):
    """Unit directions of grid points (cols, rows) of `face`: the equiangular mapping."""
    normal, u, v = FRAMES[face]
    s = 2.0 * cols / FACE_CELLS - 1.0
    t = 2.0 * rows / FACE_CELLS - 1.0
    p = normal + plane_offset(s)[..., None] * u + plane_offset(t)[..., None] * v
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def random_directions(rng, count):
    d = rng.normal(size=(count, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def crater_profile(r, diameter):
    """Height of a crater at `r` crater radii from its centre: a parabolic bowl 0.2 of the
    diameter deep and a rim 0.04 of it high falling off outside, as simple craters (Pike 1977).
    Larger, complex craters are much shallower; here their depth and rim are simply capped at
    3 km and 1 km, an illustration rather than a fit."""
    depth = min(0.2 * diameter, 3000.0)
    rim = min(0.04 * diameter, 1000.0)
    inside = rim - (depth + rim) * (1.0 - r * r)
    outside = rim * np.exp(-3.0 * (r - 1.0))
    return np.where(r < 1.0, inside, outside)


def moon_heights(seed=7, craters=6000, smallest=8000.0, largest=250000.0):
    """Six faces of heights: smooth basins, then `craters` craters between `smallest` and
    `largest` metres across with a cumulative D^-2 size distribution, each added only over the
    window of each face it can reach."""
    rng = np.random.default_rng(seed)
    side = FACE_CELLS + 1
    index = np.arange(side, dtype=np.float64)
    cols, rows = np.meshgrid(index, index)
    directions = [face_directions(face, cols, rows) for face in range(6)]
    heights = np.zeros((6, side, side))

    for centre in random_directions(rng, 8):
        angle = np.arccos(np.clip(np.stack([d @ centre for d in directions]), -1.0, 1.0))
        heights -= 1500.0 * np.exp(-(angle / 0.35) ** 2)

    u = rng.random(craters)
    diameters = smallest / np.sqrt(1.0 - u * (1.0 - (smallest / largest) ** 2))
    for centre, diameter in zip(random_directions(rng, craters), diameters):
        reach = 3.0 * 0.5 * diameter / RADIUS  # radians: the rim's fall-off ends by 3 radii
        half = int(math.ceil(2.0 * reach * FACE_CELLS / (0.5 * np.pi))) + 2
        for face in range(6):
            normal, axis_u, axis_v = FRAMES[face]
            along = centre @ normal
            if along < math.cos(math.radians(55.0) + reach) or along <= 0.0:
                continue
            s = math.atan(centre @ axis_u / along) * 4.0 / math.pi
            t = math.atan(centre @ axis_v / along) * 4.0 / math.pi
            col = int(round((s + 1.0) * 0.5 * FACE_CELLS))
            row = int(round((t + 1.0) * 0.5 * FACE_CELLS))
            c0, c1 = max(col - half, 0), min(col + half + 1, side)
            r0, r1 = max(row - half, 0), min(row + half + 1, side)
            if c0 >= c1 or r0 >= r1:
                continue
            window = directions[face][r0:r1, c0:c1]
            angle = np.arccos(np.clip(window @ centre, -1.0, 1.0))
            heights[face, r0:r1, c0:c1] += crater_profile(angle * RADIUS / (0.5 * diameter),
                                                          diameter)
    return heights.astype(np.float32)


def build_scene(heights, maps=None):
    scene = ernie.Scene()
    regolith = ernie.Path("/Looks/Regolith")
    material = scene.define(regolith, tokens.MATERIAL).set(tokens.BRDF, "hapke").set(
        tokens.ALBEDO, (0.25, 0.25, 0.25))
    if maps is not None:
        from demo_materials_lunar_regolith import TILE
        material.write(tokens.DETAIL_ALBEDO, maps["Diffuse"]).write(
            tokens.DETAIL_NORMAL, maps["nor_gl"]).set(tokens.DETAIL_TILE_SIZE, TILE).set(
            tokens.DETAIL_HEX_TILING, True)
    scene.define("/World/Moon", tokens.TERRAIN).write(tokens.HEIGHTS, heights).set(
        tokens.PLANET_RADIUS, RADIUS).set(tokens.MATERIAL_BINDING, regolith)

    # The Sun, as bright as above an atmosphere.
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform(tuple(SUN))).set(
        tokens.PHOTOMETRIC_ILLUMINANCE, 128000.0).set(tokens.ANGLE, 0.53).set(
        tokens.PHYSICAL_ILLUMINANT, tokens.BLACKBODY).set(tokens.COLOR_TEMPERATURE, 5772.0)
    stars = ernie.synthetic_stars(9000, seed=11)
    scene.define("/World/Stars", tokens.STAR_FIELD).write(
        tokens.STARS_RIGHT_ASCENSION, np.array(stars["right_ascension"], np.float32)).write(
        tokens.STARS_DECLINATION, np.array(stars["declination"], np.float32)).write(
        tokens.STARS_MAGNITUDE, np.array(stars["magnitude"], np.float32)).write(
        tokens.STARS_COLOR_INDEX, np.array(stars["color_index"], np.float32))

    scene.define("/World/Camera", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 24.0).set(
        tokens.SENSOR_NOISE, False).set(tokens.EXPOSURE_ISO, 100.0).set(
        tokens.EXPOSURE_TIME, 1.0 / 250.0).set(tokens.EXPOSURE_FSTOP, 5.6)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(tokens.RESOLUTION, SIZE).set(
        tokens.SHADOW_FIT, tokens.VISIBLE).set(tokens.AMBIENT, 0.0).set(
        tokens.COLOR_MODE, tokens.HILLSHADE)
    return scene


def in_orbit(scene):
    """The near side from 4 radii out, the terminator crossing it."""
    eye = (4.0 * RADIUS, 0.8 * RADIUS, 1.0 * RADIUS)
    scene.set("/World/Camera", tokens.TRANSFORM, ernie.look_at_transform(eye, (0.0, 0.0, 0.0)))
    scene.set("/World/Camera", tokens.CLIPPING_RANGE, (1.0e4, 1.0e8))


def low_over_the_terminator(scene):
    """15 km up where the Sun stands 8 degrees above the horizon, looking across its light so
    the craters cast long shadows."""
    north = np.array([0.0, 0.0, 1.0]) - SUN * SUN[2]
    north /= np.linalg.norm(north)
    elevation = math.radians(8.0)
    up = math.sin(elevation) * SUN + math.cos(elevation) * north
    ahead = np.cross(up, SUN)
    ahead /= np.linalg.norm(ahead)
    eye = up * (RADIUS + 15000.0)
    target = eye + 100000.0 * ahead - 15000.0 * up
    scene.set("/World/Camera", tokens.TRANSFORM,
              ernie.look_at_transform(tuple(eye), tuple(target), tuple(up)))
    scene.set("/World/Camera", tokens.CLIPPING_RANGE, (10.0, 4.0e5))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write orbit and low-altitude images here instead of "
                                         "opening the viewer")
    parser.add_argument("--textures", help="Poly Haven's moon_01 download (a .zip of any "
                                           "resolution) or a folder of its maps, for detail")
    args = parser.parse_args()
    maps = None
    if args.textures:
        from demo_materials_lunar_regolith import load_maps
        with tempfile.TemporaryDirectory() as scratch:
            maps = load_maps(os.path.expanduser(args.textures), scratch)
    scene = build_scene(moon_heights(), maps)
    in_orbit(scene)
    if not args.output:
        from ernie.app import Viewer
        Viewer(scene).run()
        return
    renderer = ernie.Renderer()
    orbit = np.array(renderer.render(scene))[..., :3]
    low_over_the_terminator(scene)
    low = np.array(renderer.render(scene))[..., :3]
    write_png(args.output, side_by_side([orbit, low]))


if __name__ == "__main__":
    main()
