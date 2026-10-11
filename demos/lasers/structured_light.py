# SPDX-License-Identifier: BSD-3-Clause
"""Structured light: a 450 nm projector shows a six-bit Gray code, one
bit plane per frame of its sequence (`ernie:laser:sequence:count`), to
a mono camera beside it. Each frame of the sensor clock shows the next
plane; the six frames decode to the projector column each pixel sees,
from which a scanner would triangulate depth.

References:
- J. Salvi, J. Pages and J. Batlle, "Pattern codification strategies
  in structured light systems", Pattern Recognition 37(4), 2004.
- J. L. Posdamer and M. D. Altschuler, "Surface measurement by
  space-encoded projected beam systems", CGIP 18(1), 1982.
"""

from pathlib import Path

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import cache_dir, side_by_side, write_png

BITS = 6
COLUMNS = 1 << BITS
PERIOD = 1.0 / 15.0  # s each plane shows
EXPOSURE = 1.0 / 60.0


def gray_code():
    """The Gray code's bit planes, most significant first, stacked top
    to bottom in one image of COLUMNS x 4 texels each; returns its path
    in the demos' cache."""
    columns = np.arange(COLUMNS)
    gray = columns ^ (columns >> 1)
    planes = [
        np.tile(((gray >> (BITS - 1 - plane)) & 1) * 255, (4, 1))
        for plane in range(BITS)
    ]
    image = np.repeat(
        np.concatenate(planes).astype(np.uint8)[:, :, None], 3, axis=2
    )
    path = Path(cache_dir()) / "laser_gray_code.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, image)
    return path


def decoded(planes):
    """Each pixel's projector column from the planes: a bit set where a
    plane is brighter than half the pixel's brightest, the Gray code to
    binary, as opaque grey (R = G = B = column / 63 x 255) with G 0
    where nothing lights the pixel."""
    values = np.stack(
        [plane[:, :, 0].astype(int) for plane in planes]
    )
    brightest = values.max(axis=0)
    gray = np.zeros(brightest.shape, int)
    for value in values:
        gray = (gray << 1) | (2 * value > brightest)
    column = gray.copy()
    shift = gray >> 1
    while shift.any():
        column ^= shift
        shift >>= 1
    grey = (column * 255 // (COLUMNS - 1)).astype(np.uint8)
    lit = brightest >= 32
    out = np.zeros(brightest.shape + (4,), np.uint8)
    out[..., 3] = 255
    out[..., 0] = np.where(lit, grey, 0)
    out[..., 1] = np.where(lit, np.maximum(grey, 1), 0)
    out[..., 2] = np.where(lit, grey, 0)
    return out


class Demo(ernie.demo.Demo):
    num_frames = BITS
    fixed_dt = PERIOD

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.6,) * 3,
            checker_color1=(0.55,) * 3,
        )
        for name, centre, size in (
            ("Wall", (0.0, 1.2, 1.0), (3.0, 0.1, 2.0)),
            ("Block", (-0.35, 0.4, 0.2), (0.4, 0.4, 0.4)),
        ):
            world.define(
                name,
                ernie.Cube,
                size=1.0,
                display_color=(0.7,) * 3,
                translate=centre,
                scale=size,
            )
        world.define(
            "Ball",
            ernie.Sphere,
            radius=0.25,
            display_color=(0.7,) * 3,
            translate=(0.4, 0.3, 0.25),
        )
        # The projector: 40 mW at 450 nm, a DLP scanner's blue, its six
        # planes in turn, each shown for 1/15 s.
        world.define(
            "Projector",
            ernie.LaserLight,
            wavelength=450.0,
            power=0.04,
            waist_radius=(0.001, 0.001),
            pattern="image",
            pattern_image=str(gray_code()),
            pattern_field_of_view=(50.0, 40.0),
            sequence_count=BITS,
            sequence_period=PERIOD,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.3, -1.8, 1.2), (0.0, 0.6, 0.4)
            ),
        )
        # A mono camera 30 cm to the projector's left, exposing 1/60 s
        # within each plane.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=20.0,
            sensor_noise=False,
            sensor_type="mono",
            exposure_time=EXPOSURE,
            exposure_iso=3200.0,
            exposure_f_stop=2.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.8, 1.2), (0.0, 0.6, 0.4)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(320, 240),
            ambient=0.0,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def planes(self):
        """The camera's frame in the middle of each plane's showing,
        uint8 (rows, cols, 4)."""
        now = self.renderer.time
        images = []
        for plane in range(BITS):
            self.renderer.reset_time((plane + 0.75) * PERIOD)
            images.append(np.array(self.renderer.render(self.scene)))
        self.renderer.reset_time(now)
        return images

    def save_output(self, path):
        """The six planes as the camera saw them, then the decoded
        columns as grey, side by side."""
        images = self.planes()
        write_png(path, side_by_side(images + [decoded(images)]))

    def test_final(self):
        out = decoded(self.planes())
        # Along the middle row, the columns the projector lit, left to
        # right: increasing, as the projector's x is the camera's.
        middle = out[out.shape[0] // 2]
        seen = [
            int(pixel[0]) * (COLUMNS - 1) // 255
            for pixel in middle
            if pixel[1] != 0
        ]
        steps = np.diff(seen)
        forward = int((steps > 0).sum())
        backward = int((steps < 0).sum())
        distinct = len(set(seen))
        print(
            f"middle row: {len(seen)} pixels lit, {distinct} columns, "
            f"{forward} steps up and {backward} down"
        )
        assert distinct >= COLUMNS // 2, (
            "the middle row should see most of the projector's columns"
        )
        assert backward * 5 < forward, (
            "the decoded columns should rise from left to right"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
