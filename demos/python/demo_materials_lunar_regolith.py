# SPDX-License-Identifier: BSD-3-Clause
"""Lunar regolith under a low Sun and a black sky: Poly Haven's "Moon 01" scan as a tiling detail
material, shaded with Hapke's lunar BRDF, on a terrain raised from its displacement map, with
boulders, the Sun at the solar constant (about 128,000 lux, with no atmosphere to dim or redden
it) and a star field, seen at two exposures.

At a daylight exposure (ISO 100, 1/250 s, f/5.6) the regolith, of albedo about 0.12, is mid-grey
and the stars are far too faint to register, as in the Apollo surface photographs. A 15 s night
exposure (ISO 1600, f/2.8) shows the stars and blows out the lit ground.

    PYTHONPATH=build/python python demos/python/demo_materials_lunar_regolith.py \
        --textures ~/Downloads/moon_01_4k.blend.zip
    PYTHONPATH=build/python python demos/python/demo_materials_lunar_regolith.py \
        --textures ~/Downloads/moon_01_4k.blend.zip --output moon.png

Download the textures (CC0) from https://polyhaven.com/a/moon_01 at any resolution, as the zip
or its maps; the demo reads the diffuse, OpenGL normal (nor_gl) and displacement maps. The scan
covers 3.1 m, so each texture tile does; its displacement is taken as 12 cm from black to white.

References:
- Poly Haven, "Moon 01" by G. Zaal, R. Cilliers, J. van Heerden and D. Barresi, CC0.
  https://polyhaven.com/a/moon_01
- G. H. Heiken, D. T. Vaniman and B. M. French (eds.), "Lunar Sourcebook", Cambridge University
  Press, 1991 (regolith albedo, about 0.07 in the maria to 0.2 in the highlands).
- B. Hapke, "Theory of Reflectance and Emittance Spectroscopy", 2nd edition, Cambridge
  University Press, 2012; H. Sato et al., "Resolved Hapke parameter maps of the Moon", JGR
  Planets 119(8):1775-1805, 2014 (the BRDF and its lunar parameters).
- G. Kopp and J. L. Lean, "A new, lower value of total solar irradiance", Geophysical Research
  Letters 38(1), 2011 (1361 W/m^2, about 128,000 lux in the visible).
"""

import argparse
import math
import os
import tempfile
import zipfile

import numpy as np

import ernie
from ernie import tokens

from demo_support import side_by_side, write_png

SIZE = (960, 540)
TILE = 3.1          # metres one texture tile covers
RELIEF = 0.12       # metres from the displacement map's black to white
ASSET = "moon_01"


MAPS = {"Diffuse": "_diff_", "nor_gl": "_nor_gl_", "Displacement": "_disp_"}


def find_maps(names):
    """The diffuse, OpenGL normal and displacement maps among file `names`, by Poly Haven's
    naming (moon_01_diff_4k.jpg and so on)."""
    found = {}
    for kind, part in MAPS.items():
        matches = sorted(n for n in names if part in os.path.basename(n) and ASSET in n)
        if not matches:
            raise SystemExit(f"No {ASSET} {kind} map ({part}) among the textures")
        found[kind] = matches[0]
    return found


def load_maps(source, scratch):
    """The maps as float arrays, from a folder of them or a Poly Haven .zip download (unpacked
    into `scratch`). Maps above 2k are halved, a box filter at a time, to keep uploads small."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = find_maps(archive.namelist())
            paths = {kind: archive.extract(name, scratch) for kind, name in names.items()}
    else:
        names = find_maps(os.listdir(source))
        paths = {kind: os.path.join(source, name) for kind, name in names.items()}
    maps = {}
    for kind, path in paths.items():
        # The normal map is a DWAA-compressed half-float EXR, as Poly Haven ships them.
        image = np.asarray(ernie.load_image(path, srgb=kind == "Diffuse"))
        while image.shape[0] > 2048:
            image = 0.25 * (image[0::2, 0::2] + image[1::2, 0::2] + image[0::2, 1::2] +
                            image[1::2, 1::2])
        maps[kind] = np.ascontiguousarray(image, dtype=np.float32)
    return maps


def placed(x, y, z, sx=1.0, sy=1.0, sz=1.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def terrain_heights(displacement, tiles=32, per_tile=128):
    """Heights over `tiles` x `tiles` texture tiles (about 100 m, its far edge the horizon): the
    displacement map tiled (it is seamless) at `per_tile` samples a tile, on a few shallow
    craters. Rows run north to south. The normal map carries the finer relief."""
    step = max(displacement.shape[0] // per_tile, 1)
    tile = displacement[::step, ::step, 0][:per_tile, :per_tile]
    heights = np.tile(tile, (tiles, tiles)) * RELIEF
    n = heights.shape[0]
    cell = TILE / per_tile
    coords = (np.arange(n) + 0.5) * cell - 0.5 * n * cell
    x, y = np.meshgrid(coords, -coords)
    for cx, cy, radius, depth in [(-3.5, 2.0, 2.4, 0.35), (4.0, 5.5, 1.5, 0.2),
                                  (1.5, -1.0, 0.9, 0.12), (-6.0, -5.0, 1.8, 0.25)]:
        r = np.hypot(x - cx, y - cy) / radius
        bowl = np.where(r < 1.0, -depth * (1.0 - r * r), 0.0)
        rim = 0.25 * depth * np.exp(-((r - 1.0) / 0.18) ** 2)
        heights += bowl + rim
    return heights.astype(np.float32), cell, n * cell


def build_scene(maps):
    scene = ernie.Scene()
    regolith = ernie.Path("/Looks/Regolith")
    # Hapke's model with Sato et al.'s lunar parameters (the BRDF's defaults): its albedo input
    # is the single-scattering albedo w, about 0.25 for lunar soil, which gives the regolith's
    # normal albedo of about 0.12; the scan's colour varies it about that mean.
    scene.define(regolith, tokens.MATERIAL).set(tokens.BRDF, "hapke").set(
        tokens.ALBEDO, (0.25, 0.25, 0.25)).write(tokens.DETAIL_ALBEDO, maps["Diffuse"]).write(
        tokens.DETAIL_NORMAL, maps["nor_gl"]).set(tokens.DETAIL_TILE_SIZE, TILE).set(
        tokens.DETAIL_NORMAL_STRENGTH, 1.0)

    heights, cell, extent = terrain_heights(maps["Displacement"])
    scene.define("/World/Terrain", tokens.TERRAIN).write(tokens.HEIGHTS, heights).set(
        tokens.ORIGIN, (-0.5 * extent, 0.5 * extent)).set(tokens.CELL_SIZE, cell).set(
        tokens.MATERIAL_BINDING, regolith)
    for k, (x, y, r) in enumerate([(2.2, 1.5, 0.45), (-1.0, 4.0, 0.7), (0.4, 0.2, 0.18),
                                   (3.5, 7.5, 1.1), (-2.4, 0.8, 0.25)]):
        scene.define(f"/World/Boulders/B{k}", tokens.SPHERE).set(tokens.GEOM_RADIUS, 1.0).set(
            tokens.TRANSFORM, placed(x, y, 0.35 * r, r, 0.8 * r, 0.6 * r)).set(
            tokens.MATERIAL_BINDING, regolith)

    # The Sun 15 degrees up in the west-south-west, as bright as above the atmosphere.
    altitude, azimuth = math.radians(15.0), math.radians(250.0)
    towards = (math.cos(altitude) * math.sin(azimuth), math.cos(altitude) * math.cos(azimuth),
               math.sin(altitude))
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform(towards)).set(
        tokens.PHOTOMETRIC_ILLUMINANCE, 128000.0).set(tokens.ANGLE, 0.53).set(
        tokens.PHYSICAL_ILLUMINANT, tokens.BLACKBODY).set(tokens.COLOR_TEMPERATURE, 5772.0)
    stars = ernie.synthetic_stars(9000, seed=11)
    scene.define("/World/Stars", tokens.STAR_FIELD).write(
        tokens.STARS_RIGHT_ASCENSION, np.array(stars["right_ascension"], np.float32)).write(
        tokens.STARS_DECLINATION, np.array(stars["declination"], np.float32)).write(
        tokens.STARS_MAGNITUDE, np.array(stars["magnitude"], np.float32)).write(
        tokens.STARS_COLOR_INDEX, np.array(stars["color_index"], np.float32))

    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.5, -7.0, 1.7), (0.0, 6.0, 0.6))).set(
        tokens.FOCAL_LENGTH, 20.0).set(tokens.SENSOR_NOISE, False)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(tokens.RESOLUTION, SIZE).set(
        tokens.SHADOW_FIT, tokens.VISIBLE).set(tokens.AMBIENT, 0.0).set(
        tokens.COLOR_MODE, tokens.HILLSHADE)  # the material's own colour, not elevation's
    return scene


def expose(scene, iso, seconds, fstop):
    scene.set("/World/Camera", tokens.EXPOSURE_ISO, iso)
    scene.set("/World/Camera", tokens.EXPOSURE_TIME, seconds)
    scene.set("/World/Camera", tokens.EXPOSURE_FSTOP, fstop)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write the image here instead of opening a window")
    parser.add_argument("--textures", required=True,
                        help="Poly Haven's moon_01 download (a .zip of any resolution) or a "
                             "folder of its diff, nor_gl and disp maps")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        maps = load_maps(os.path.expanduser(args.textures), scratch)
    scene = build_scene(maps)
    renderer = ernie.Renderer()
    expose(scene, 100.0, 1.0 / 250.0, 5.6)
    day = np.array(renderer.render(scene))[..., :3]
    if not args.output:
        from ernie.app import Viewer
        Viewer(scene).run()
        return
    expose(scene, 1600.0, 15.0, 2.8)
    night = np.array(renderer.render(scene))[..., :3]
    write_png(args.output, side_by_side([day, night]))


if __name__ == "__main__":
    main()
