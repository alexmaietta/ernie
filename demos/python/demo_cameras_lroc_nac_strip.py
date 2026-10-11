# SPDX-License-Identifier: BSD-3-Clause
"""An LROC Narrow Angle Camera image, strip by strip: the NAC-L's real geometry from its ISD over
cratered ground near the Moon's south pole, through the NAC's published sensor and readout to
the 8-bit numbers it sends.

    PYTHONPATH=build/python python demos/python/demo_cameras_lroc_nac_strip.py \
        [--viewer null --output nac.png] [--isd ISD]

The ISD (ALE's image support data, from usgscsm's tests) gives the camera's orbit, pointing,
line time, 2 x 2 summed detector and lens distortion; `IsdLineScanner.author_strip` turns each run
of 64 lines into a pushbroom camera posed from the strip's first line to its last, and stepping
the scene through the strips renders the image in order. `author_lroc_nac` sets the NAC-L's
sensor (gain, detector noise, full well, 12-bit ADC, f-number, 400-760 nm response) and readout:
noise, ADC, its even and odd channels and the 12- to 8-bit companding table, so RawDN holds what
the NAC downlinks. The sun stands where the ISD puts it, about a degree over the horizon near the
pole. This image's lines are long (11.9 ms, 2 x 2 summed): about 20 m along the track against 0.72
m across it, so craters come out squashed along the track, as in the raw image.

The scene is built in a local frame on the strip's ground point, z up from the Moon's centre, so
terrain heights stay small: a 2 m heightfield following the sphere's curve with illustrative bowl
craters (not a real NAC footprint), and the ISD's poses and sun moved into the same frame.

References:
- M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument Overview", Space
  Science Reviews 150:81-124, 2010 (the NAC's design, Table 2 and Appendix A).
- USGS Astrogeology, usgscsm, tests/data/toughLroNacLineScan.json (the ISD).
- R. J. Pike, "Size-dependence in the shape of fresh impact craters on the Moon", in Impact and
  Explosion Cratering, Pergamon, 1977, 489-509 (depth about 0.2 and rim height about 0.04 of the
  diameter for simple craters).
"""

import math
import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

ISD = os.path.join(os.path.dirname(__file__), "..", "..", "ernie", "test", "fixtures", "isd",
                   "toughLroNacLineScan.json")
FIRST_LINE = 3968  # the image's middle 256 lines
STRIP_LINES = 64
STRIPS = 4
CELL = 2.0  # metres per height
EXTENT = 7000.0  # metres of terrain each way around the strip's ground point


def local_frame(centre):
    """The rigid transform from body-fixed metres to a frame on `centre`, z away from the body."""
    up = centre / np.linalg.norm(centre)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    rotation = np.array([east, north, up])
    frame = np.eye(4)
    frame[:3, :3] = rotation
    frame[:3, 3] = -rotation @ centre
    return frame


def terrain_heights(radius, seed=4):
    """Heights over the frame's plane: the sphere's curve and illustrative bowl craters."""
    n = int(EXTENT / CELL)
    x = (np.arange(n) + 0.5) * CELL - 0.5 * EXTENT
    y = 0.5 * EXTENT - (np.arange(n) + 0.5) * CELL  # rows run south from the north edge
    xx, yy = np.meshgrid(x, y)
    heights = np.sqrt(radius * radius - xx * xx - yy * yy) - radius
    rng = np.random.default_rng(seed)
    # Diameters from 8 to 600 m, fewer of the larger (a cumulative power law of slope -2).
    count = 900
    diameters = 8.0 / np.sqrt(1.0 - rng.random(count) * (1.0 - (8.0 / 600.0) ** 2))
    centres = rng.uniform(-0.5 * EXTENT, 0.5 * EXTENT, (count, 2))
    for (cx, cy), d in zip(centres, diameters):
        r = 0.5 * d
        cols = slice(max(int((cx - 2 * r + 0.5 * EXTENT) / CELL), 0),
                     min(int((cx + 2 * r + 0.5 * EXTENT) / CELL) + 1, n))
        rows = slice(max(int((0.5 * EXTENT - cy - 2 * r) / CELL), 0),
                     min(int((0.5 * EXTENT - cy + 2 * r) / CELL) + 1, n))
        dx = xx[rows, cols] - cx
        dy = yy[rows, cols] - cy
        s = np.sqrt(dx * dx + dy * dy) / r
        # Pike: a bowl 0.2 D deep inside, a rim 0.04 D high falling away outside.
        bowl = np.where(s < 1.0, 0.24 * d * (s * s - 1.0) + 0.04 * d, 0.0)
        rim = np.where(s >= 1.0, 0.04 * d * np.exp(-6.0 * (s - 1.0)), 0.0)
        heights[rows, cols] += bowl + rim
    return heights.astype(np.float32)


def build_scene(scene, scanner):
    """Builds the terrain, sun, NAC and its render settings into `scene`; returns the local
    frame."""
    middle = FIRST_LINE + 0.5 * STRIP_LINES * STRIPS
    centre = np.array(scanner.image_to_ground(middle, 0.5 * scanner.samples))
    frame = local_frame(centre)
    radius = float(np.linalg.norm(centre))

    # Hapke's model with its lunar defaults; single-scattering albedo about 0.25 for lunar soil.
    regolith = scene.define("/Looks/Regolith", ernie.Material, brdf="hapke",
                            albedo=(0.25, 0.25, 0.25))
    scene.define("/World/Terrain", ernie.Terrain, heights=terrain_heights(radius),
                 origin=(-0.5 * EXTENT, 0.5 * EXTENT), cell_size=CELL, material_binding=regolith)

    # The sun where the ISD has it at the image's middle, as bright as at the Moon.
    sun = np.array(scanner.sun_position(scanner.image_time(middle))) - centre
    towards = frame[:3, :3] @ (sun / np.linalg.norm(sun))
    print(f"sun {math.degrees(math.asin(towards[2])):.1f} degrees above the horizon")
    scene.define("/World/Sun", ernie.DistantLight, photometric_illuminance=128000.0, angle=0.53,
                 physical_illuminant=tokens.BLACKBODY, color_temperature=5772.0).set(
        tokens.TRANSFORM, ernie.distant_light_transform(tuple(towards)))

    camera = scene.define("/World/Camera", ernie.Camera)
    raw_dn = scene.define("/Render/Vars/RawDN", ernie.RenderVar, source_name="RawDN")
    nac = scene.define("/Render/Nac", ernie.RenderProduct, camera=camera).set(
        tokens.ORDERED_VARS, [raw_dn.path])
    ernie.author_lroc_nac(scene, camera, nac, side="left")
    # No sky to light the shadows: only the sun (the default ambient term would saturate the
    # NAC everywhere).
    scene.define("/Render/Settings", ernie.RenderSettings, ambient=0.0, camera=camera).set(
        tokens.PRODUCTS, [nac.path])
    return frame


def author_strip(scanner, scene, frame, first, at_end):
    """Authors the strip from line `first` on the camera, posed in the local frame."""
    resolution = scanner.author_strip(scene, "/World/Camera", first, STRIP_LINES, at_end=at_end)
    pose = np.array(scene.get("/World/Camera", tokens.TRANSFORM))
    scene.set("/World/Camera", tokens.TRANSFORM, frame @ pose)
    scene.set("/Render/Nac", tokens.RESOLUTION, resolution)


class Demo(ernie.demo.Demo):
    num_frames = STRIPS  # one strip a step; a window starts the image over after the last

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--isd", default=ISD, help="the NAC image's ISD (ALE JSON)")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.scanner = ernie.IsdLineScanner.from_file(args.isd)
        print(f"{self.scanner.sensor}: {self.scanner.samples} samples x "
              f"{self.scanner.lines} lines")
        self.local = build_scene(self.scene, self.scanner)
        self.strips = [None] * STRIPS

    def step(self):
        k = self.frame % STRIPS
        if k == 0:
            # The first strip's start, rendered and dropped: each strip then runs to its end, the
            # camera's last pose being the next strip's first.
            self.strips = [None] * STRIPS
            author_strip(self.scanner, self.scene, self.local, FIRST_LINE, at_end=False)
            self.renderer.step(self.scene, render_products=["/Render/Nac"],
                               settings="/Render/Settings")
        author_strip(self.scanner, self.scene, self.local, FIRST_LINE + k * STRIP_LINES,
                     at_end=True)
        out = self.renderer.step(self.scene, render_products=["/Render/Nac"],
                                 settings="/Render/Settings")["/Render/Nac"].frames[0]
        self.strips[k] = np.array(out.render_vars["/Render/Vars/RawDN"].map())[..., 0]
        if k == STRIPS - 1:
            image = self.mosaic()
            print(f"8-bit DN from {image.min()} to {image.max()}, mean {image.mean():.1f}")

    def render(self):
        # `step` renders the NAC's product; without a viewer there is nothing more to draw.
        if self.viewer is not None:
            super().render()

    def mosaic(self):
        """The strips rendered so far, top to bottom, as 8-bit DN; strips still to come black."""
        width = next(s.shape[1] for s in self.strips if s is not None)
        return np.concatenate([s if s is not None else np.zeros((STRIP_LINES, width))
                               for s in self.strips], axis=0).astype(np.uint8)

    def save_output(self, path):
        ernie.demo.write_png(path, self.mosaic())

    def test_final(self):
        assert all(s is not None for s in self.strips), "every strip should have rendered"
        image = self.mosaic()
        assert image.shape[0] == STRIPS * STRIP_LINES
        assert image.std() > 2.0, "the craters and their shadows should not render flat"
        assert image.mean() > 5.0 and np.mean(image == 255) < 0.05, \
            "the companded DN should be neither black nor saturated"


if __name__ == "__main__":
    ernie.demo.main(Demo)
