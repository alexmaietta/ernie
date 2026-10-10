# SPDX-License-Identifier: BSD-3-Clause
"""An LROC Wide Angle Camera image, framelet by framelet: the WAC's real push-frame geometry from
its ISD over cratered ground, through one band's published sensor and readout to the 11-bit
numbers it digitises.

    PYTHONPATH=build/python python demos/python/demo_cameras_lroc_wac_framelets.py
    PYTHONPATH=build/python python demos/python/demo_cameras_lroc_wac_framelets.py --band 415

The ISD (ALE's image support data, from usgscsm's tests) is a visible band's image: 364 framelets
of 14 lines by 1024 samples, each exposed for 39 ms, one every 0.516 s, the last exposed at the
image's top (`framelet_order_reversed`), through the WAC's wide-angle lens and its distortion.
`IsdPushFrame.author_framelet` turns each framelet into a frame camera posed when it is exposed;
rendering a run of them and stacking their rows gives the image as the WAC returns it, framelets
overlapping a little on the ground. `author_lroc_wac` sets the band's sensor and readout from
Robinson et al. 2010: the passband (Table 4's effective wavelength and width) at the efficiency
that reproduces Table 4's responsivity, the KAI-1001's gain, noise and full well and the 11-bit
ADC (Table 3). The sun stands where the ISD puts it, about 25 degrees up. The PNG shows RawDN
stretched linearly from 0 to its largest (not the WAC's 11- to 8-bit companding, which the paper
gives only as a figure). The WAC's framelets overlap on the ground by a few lines, so stacked
they stretch the ground along the track a little, as raw WAC images do.

The scene is built in a local frame on the run's ground point, z up from the Moon's centre: a
120 km heightfield at 50 m following the sphere's curve with illustrative bowl craters (not a real
WAC footprint), and the ISD's poses and sun moved into the same frame.

References:
- M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument Overview", Space
  Science Reviews 150:81-124, 2010 (the WAC's design, Tables 3 and 4).
- USGS Astrogeology, usgscsm, tests/data/lroWacPushFrame.json (the ISD), and its push-frame model
  `UsgsAstroPushFrameSensorModel`.
- R. J. Pike, "Size-dependence in the shape of fresh impact craters on the Moon", in Impact and
  Explosion Cratering, Pergamon, 1977, 489-509 (depth about 0.2 and rim height about 0.04 of the
  diameter for simple craters).
"""

import argparse
import math
import os

import numpy as np

import ernie
from ernie import tokens

from demo_support import write_png

ISD = os.path.join(os.path.dirname(__file__), "..", "..", "ernie", "test", "fixtures", "isd",
                   "lroWacPushFrame.json")
FIRST_FRAMELET = 170  # 24 framelets about the image's middle
FRAMELETS = 24
CELL = 50.0  # metres per height
EXTENT = 120000.0  # metres of terrain each way around the run's ground point


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


def terrain_heights(radius, seed=7):
    """Heights over the frame's plane: the sphere's curve and illustrative bowl craters."""
    n = int(EXTENT / CELL)
    x = (np.arange(n) + 0.5) * CELL - 0.5 * EXTENT
    y = 0.5 * EXTENT - (np.arange(n) + 0.5) * CELL  # rows run south from the north edge
    xx, yy = np.meshgrid(x, y)
    heights = np.sqrt(radius * radius - xx * xx - yy * yy) - radius
    rng = np.random.default_rng(seed)
    # Diameters from 300 m to 15 km, fewer of the larger (a cumulative power law of slope -2).
    count = 2500
    diameters = 300.0 / np.sqrt(1.0 - rng.random(count) * (1.0 - (300.0 / 15000.0) ** 2))
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


def build_scene(camera, band):
    lines = camera.framelet_lines
    middle = (FIRST_FRAMELET + 0.5 * FRAMELETS) * lines
    centre = np.array(camera.image_to_ground(middle, 0.5 * camera.samples))
    frame = local_frame(centre)
    radius = float(np.linalg.norm(centre))

    scene = ernie.Scene()
    regolith = ernie.Path("/Looks/Regolith")
    # Hapke's model with its lunar defaults; single-scattering albedo about 0.25 for lunar soil.
    scene.define(regolith, tokens.MATERIAL).set(tokens.BRDF, "hapke").set(
        tokens.ALBEDO, (0.25, 0.25, 0.25))
    scene.define("/World/Terrain", tokens.TERRAIN).write(
        tokens.HEIGHTS, terrain_heights(radius)).set(
        tokens.ORIGIN, (-0.5 * EXTENT, 0.5 * EXTENT)).set(tokens.CELL_SIZE, CELL).set(
        tokens.MATERIAL_BINDING, regolith)

    # The sun where the ISD has it at the run's middle, as bright as at the Moon.
    sun = np.array(camera.sun_position(camera.image_time(middle))) - centre
    towards = frame[:3, :3] @ (sun / np.linalg.norm(sun))
    print(f"sun {math.degrees(math.asin(towards[2])):.1f} degrees above the horizon")
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform(tuple(towards))).set(
        tokens.PHOTOMETRIC_ILLUMINANCE, 128000.0).set(tokens.ANGLE, 0.53).set(
        tokens.PHYSICAL_ILLUMINANT, tokens.BLACKBODY).set(tokens.COLOR_TEMPERATURE, 5772.0)

    scene.define("/World/Camera", tokens.CAMERA)
    scene.define("/Render/Vars/RawDN", tokens.RENDER_VAR).set(tokens.SOURCE_NAME, "RawDN")
    scene.define("/Render/Wac", tokens.RENDER_PRODUCT).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(
        tokens.ORDERED_VARS, [ernie.Path("/Render/Vars/RawDN")])
    ernie.author_lroc_wac(scene, "/World/Camera", "/Render/Wac", band=band)
    # No sky to light the shadows: only the sun.
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(tokens.AMBIENT, 0.0).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(
        tokens.PRODUCTS, [ernie.Path("/Render/Wac")])
    return scene, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--isd", default=ISD, help="the WAC image's ISD (ALE JSON)")
    parser.add_argument("--band", type=int, default=605,
                        help="the WAC band to model: 320, 360, 415, 565, 605, 645 or 690 nm")
    parser.add_argument("--output", default="lroc_wac_framelets.png",
                        help="where to write the image")
    args = parser.parse_args()
    camera = ernie.IsdPushFrame.from_file(args.isd)
    print(f"{camera.sensor}: {camera.framelets} framelets of {camera.samples} x "
          f"{camera.framelet_lines}")
    scene, frame = build_scene(camera, args.band)
    renderer = ernie.Renderer()
    framelets = []
    for index in range(FIRST_FRAMELET, FIRST_FRAMELET + FRAMELETS):
        samples, lines, _, rows_reversed, _ = camera.author_framelet(scene, "/World/Camera",
                                                                    index)
        pose = np.array(scene.get("/World/Camera", tokens.TRANSFORM))
        scene.set("/World/Camera", tokens.TRANSFORM, frame @ pose)
        scene.set("/Render/Wac", tokens.RESOLUTION, (samples, lines))
        out = renderer.step(scene, render_products=["/Render/Wac"],
                            settings="/Render/Settings")["/Render/Wac"].frames[0]
        dn = np.array(out.render_vars["/Render/Vars/RawDN"].map())[..., 0]
        framelets.append(dn[::-1] if rows_reversed else dn)
    image = np.concatenate(framelets, axis=0)
    print(f"11-bit DN from {image.min():.0f} to {image.max():.0f}, mean {image.mean():.1f}")
    shown = np.clip(image * (255.0 / max(image.max(), 1.0)), 0, 255).astype(np.uint8)
    write_png(args.output, np.repeat(shown[..., None], 3, axis=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
