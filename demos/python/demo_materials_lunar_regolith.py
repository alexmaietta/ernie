# SPDX-License-Identifier: BSD-3-Clause
"""Lunar regolith under a low Sun and a black sky: Poly Haven's "Moon 01" scan as a tiling detail
material, shaded with Hapke's lunar BRDF, on a terrain raised from its displacement map, with
boulders, the Sun at the solar constant (about 128,000 lux, with no atmosphere to dim or redden
it) and a star field, seen at two exposures.

At a daylight exposure (ISO 100, 1/250 s, f/5.6) the regolith, of albedo about 0.12, is mid-grey
and the stars are far too faint to register, as in the Apollo surface photographs. A 15 s night
exposure (ISO 1600, f/2.8) shows the stars and blows out the lit ground.

    PYTHONPATH=build/python python demos/python/demo_materials_lunar_regolith.py \
        [--textures ~/Downloads/moon_01_4k.blend.zip]
    PYTHONPATH=build/python python demos/python/demo_materials_lunar_regolith.py \
        --textures ~/Downloads/moon_01_4k.blend.zip --viewer null --output moon.png

The output sets the day exposure beside the night one. Download the textures (CC0) from
https://polyhaven.com/a/moon_01 at any resolution, as the zip or its maps; the demo reads the
diffuse, OpenGL normal (nor_gl) and displacement maps. The scan covers 3.1 m, so each texture tile
does; its displacement is taken as 12 cm from black to white. Without `--textures` the regolith
is plain Hapke, of the same albedo, on a smooth random relief tiled the same way.

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

import math
import os
import tempfile
import zipfile

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

from demo_support import side_by_side, write_png

SIZE = (960, 540)
TILE = 3.1          # metres one texture tile covers
RELIEF = 0.12       # metres from the displacement map's black to white
ASSET = "moon_01"
DAY = (100.0, 1.0 / 250.0, 5.6)     # ISO, seconds, f-number
NIGHT = (1600.0, 15.0, 2.8)


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


def procedural_displacement(size=128, seed=3):
    """A seamless stand-in for the scan's displacement map, in [0, 1]: white noise low-passed
    with a Gaussian over its periodic spectrum, so it tiles as the scan does."""
    rng = np.random.default_rng(seed)
    frequency = np.fft.fftfreq(size) * size
    k = np.hypot(*np.meshgrid(frequency, frequency))
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) *
                         np.exp(-(k / 6.0) ** 2)).real
    field = (field - field.min()) / (field.max() - field.min())
    return field[..., None].astype(np.float32)


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


class Demo(ernie.demo.Demo):
    num_frames = 1

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--textures", default=None,
                            help="Poly Haven's moon_01 download (a .zip of any resolution) or a "
                                 "folder of its diff, nor_gl and disp maps; plain Hapke regolith "
                                 "without")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        maps = None
        if args.textures:
            with tempfile.TemporaryDirectory() as scratch:
                maps = load_maps(os.path.expanduser(args.textures), scratch)
        scene = self.scene
        # Hapke's model with Sato et al.'s lunar parameters (the BRDF's defaults): its albedo
        # input is the single-scattering albedo w, about 0.25 for lunar soil, which gives the
        # regolith's normal albedo of about 0.12; the scan's colour varies it about that mean.
        detail = {} if maps is None else dict(
            detail_albedo=maps["Diffuse"], detail_normal=maps["nor_gl"], detail_tile_size=TILE,
            detail_normal_strength=1.0)
        regolith = scene.define("/Looks/Regolith", ernie.Material, brdf="hapke",
                                albedo=(0.25, 0.25, 0.25), **detail)

        world = scene.define("/World", ernie.Xform)
        displacement = procedural_displacement() if maps is None else maps["Displacement"]
        heights, cell, extent = terrain_heights(displacement)
        world.define("Terrain", ernie.Terrain, heights=heights,
                     origin=(-0.5 * extent, 0.5 * extent), cell_size=cell,
                     material_binding=regolith)
        boulders = world.define("Boulders", ernie.Xform)
        for k, (x, y, r) in enumerate([(2.2, 1.5, 0.45), (-1.0, 4.0, 0.7), (0.4, 0.2, 0.18),
                                       (3.5, 7.5, 1.1), (-2.4, 0.8, 0.25)]):
            boulders.define(f"B{k}", ernie.Sphere, radius=1.0, translate=(x, y, 0.35 * r),
                            scale=(r, 0.8 * r, 0.6 * r), material_binding=regolith)

        # The Sun 15 degrees up in the west-south-west, as bright as above the atmosphere.
        altitude, azimuth = math.radians(15.0), math.radians(250.0)
        towards = (math.cos(altitude) * math.sin(azimuth),
                   math.cos(altitude) * math.cos(azimuth), math.sin(altitude))
        world.define("Sun", ernie.DistantLight, photometric_illuminance=128000.0, angle=0.53,
                     physical_illuminant=tokens.BLACKBODY, color_temperature=5772.0).set(
            tokens.TRANSFORM, ernie.distant_light_transform(towards))
        stars = ernie.synthetic_stars(9000, seed=11)
        world.define("Stars", ernie.StarField,
                     right_ascension=np.array(stars["right_ascension"], np.float32),
                     declination=np.array(stars["declination"], np.float32),
                     magnitude=np.array(stars["magnitude"], np.float32),
                     color_index=np.array(stars["color_index"], np.float32))

        self.camera = world.define("Camera", ernie.Camera, focal_length=20.0,
                                   sensor_noise=False).set(
            tokens.TRANSFORM, ernie.look_at_transform((0.5, -7.0, 1.7), (0.0, 6.0, 0.6)))
        scene.define("/Render/Settings", ernie.RenderSettings, camera=self.camera,
                     resolution=SIZE, shadow_fit=tokens.VISIBLE, ambient=0.0,
                     color_mode=tokens.HILLSHADE)  # the material's own colour, not elevation's
        self.expose(*DAY)
        if viewer is not None:
            # Look through the scene's camera, with its own (daylight) exposure.
            viewer.active_camera = self.camera.path

    def expose(self, iso, seconds, fstop):
        self.camera.set(exposure_iso=iso, exposure_time=seconds, exposure_f_stop=fstop)

    def render_at(self, exposure):
        """The scene's camera at `exposure` (ISO, seconds, f-number), RGB; leaves it at day."""
        self.expose(*exposure)
        image = np.array(self.renderer.render(self.scene))[..., :3]
        self.expose(*DAY)
        return image

    def save_output(self, path):
        write_png(path, side_by_side([self.render_at(DAY), self.render_at(NIGHT)]))

    def test_final(self):
        day, night = self.render_at(DAY), self.render_at(NIGHT)
        # The near ground, below the horizon: mid-grey at the day exposure, not black or white.
        ground = np.median(day[SIZE[1] * 2 // 3:].astype(np.float32))
        assert 50.0 < ground < 200.0, f"the regolith should be mid-grey by day, not {ground}"
        # The black sky above the horizon: no stars register by day, some do by night.
        sky_day, sky_night = day[:SIZE[1] // 8], night[:SIZE[1] // 8]
        assert np.percentile(sky_day, 99.9) < 20, "stars should not register by day"
        assert sky_night.max() > 100, "the night exposure should show stars"
        assert np.median(night[SIZE[1] * 2 // 3:]) > ground, "night should overexpose the ground"


if __name__ == "__main__":
    ernie.demo.main(Demo)
