# SPDX-License-Identifier: BSD-3-Clause
"""A rover's structured-light dots smearing over a long night exposure:
still on flat ground, which moves with the rig, but drawn out along
their row on a rock face the rover drives towards.

References:
- A. Nefian, U. Wong, M. Dille et al., "Structured Light-based Hazard
  Detection For Planetary Surface Navigation", NASA NTRS 20170009200,
  2017 (motion blur's night false negatives, Sec. VI).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

DOTS = 11
GAMMA = 1.52  # degrees between beams
BASELINE = 0.26  # m, camera to projector
PITCH = 28.0  # degrees down
HEIGHT = 0.6  # m, the rig above the ground
FOCAL = 3.5  # mm
PIXEL = 4.65e-3  # mm, the ICX267's
RESOLUTION = (1392, 1040)
SPEED = 0.1  # m/s, their most common
STEP = 0.5  # s between frames
# Exposures (s): a short one, and a long one at night; each frame's
# shutter is open for its exposure before the frame's time.
EXPOSURES = {"1 ms": 0.001, "500 ms": 0.5}
START = -0.3  # m, the rover's first position
END = 0.0  # m, where it turns back


def rig(x):
    """A transform `x` m to the right on the rover, `PITCH` down."""
    pitch = math.radians(PITCH)
    return ernie.look_at_transform(
        (x, 0.0, HEIGHT),
        (x, math.cos(pitch), HEIGHT - math.sin(pitch)),
        (0.0, 0.0, 1.0),
    )


def dots(row):
    """(first, last) columns of each dot along `row` above half its own
    peak: the dots on the rock face outshine those on the ground."""
    columns = np.flatnonzero(row > 0.01 * row.max())
    if columns.size == 0:
        return []
    runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    spans = []
    for run in runs:
        half = run[row[run] > 0.5 * row[run].max()]
        spans.append((int(half[0]), int(half[-1])))
    return spans


class Demo(ernie.demo.Demo):
    num_frames = 4
    fixed_dt = STEP

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.Cube,
            size=1.0,
            display_color=(0.1,) * 3,
            translate=(0.0, 20.0, -0.05),
            scale=(60.0, 60.0, 0.1),
        )
        # A rock face 1 m ahead, under the left dots (the projector is
        # 26 cm right of the camera).
        world.define(
            "Rock",
            ernie.Cube,
            size=1.0,
            display_color=(0.3,) * 3,
            translate=(0.12, 1.2, 0.15),
            scale=(0.24, 0.4, 0.3),
        )
        self.rover = world.define("Rover", ernie.Xform)
        self.rover.define(
            "Projector",
            ernie.LaserLight,
            wavelength=660.0,
            power=0.05,
            waist_radius=(0.001, 0.001),
            pattern="grid",
            grid_columns=DOTS,
            grid_rows=1,
            grid_pitch=(GAMMA, GAMMA),
            efficiency=0.8,
            zero_order=0.0,
        ).set(tokens.TRANSFORM, rig(BASELINE))
        self.camera = self.rover.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=PIXEL * RESOLUTION[0],
            vertical_aperture=PIXEL * RESOLUTION[1],
            sensor_type="mono",
            sensor_qe_mono=[600, 0.3, 720, 0.3],
            spectral_range=(600.0, 720.0),
            spectral_bands=3,
            sensor_noise=False,
            exposure_f_stop=5.6,
            clipping_range=(0.05, 1e4),
            shutter_samples=8,
        ).set(tokens.TRANSFORM, rig(0.0))
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        # Radiance, which the shutter averages and nothing saturates.
        radiance = scene.define(
            "/Render/Camera/SpectralRadiance", ernie.RenderVar
        )
        radiance.source_name = "SpectralRadiance"
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=[radiance.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.frames = {}  # per exposure, as `radiance` renders it
        self.position = START
        self.place(START)
        self.expose("500 ms")
        self.widgets.combo(
            "Exposure",
            list(EXPOSURES),
            1,
            on_change=lambda i: self.expose(list(EXPOSURES)[i]),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE

    def place(self, y):
        """Puts the rover `y` m along its track."""
        self.rover.set(translate=(0.0, y, 0.0))

    def expose(self, name):
        """Opens the shutter for exposure `name` before each frame's
        time."""
        exposure = EXPOSURES[name]
        self.camera.set(
            exposure_time=exposure,
            shutter_open=-exposure / STEP,
            shutter_close=0.0,
        )

    def step(self):
        self.position += SPEED * STEP
        if self.position > END:
            self.position = START
        self.place(self.position)

    def radiance(self, name):
        """The camera's radiance over its bands for exposure `name`,
        the rover arriving at `END` from where it was a frame before
        (drawn with the shutter shut, for its poses alone). Each
        exposure renders once."""
        if name in self.frames:
            return self.frames[name]
        for at, exposure in ((END - SPEED * STEP, None), (END, name)):
            self.place(at)
            if exposure is None:
                self.camera.set(shutter_open=0.0, shutter_close=0.0)
            else:
                self.expose(exposure)
            outputs = self.renderer.step(
                self.scene, render_products=["/Render/Camera"]
            )
        bands = np.array(
            outputs["/Render/Camera"]
            .frames[0]
            .render_vars["/Render/Camera/SpectralRadiance"]
            .map()
        ).astype(np.float64)
        self.frames[name] = bands.sum(axis=0)
        return self.frames[name]

    def render_all(self):
        """Each exposure's image about the dots, as uint8 grey: a square
        root, so the dim dots on the ground show, three times larger."""
        images = []
        for name in EXPOSURES:
            image = self.radiance(name)
            row = int(np.argmax(image.max(axis=1)))
            crop = image[row - 20 : row + 21, 730:980]
            grey = 255.0 * np.sqrt(np.clip(crop / crop.max(), 0, 1))
            big = np.kron(grey, np.ones((3, 3)))
            images.append(
                np.repeat(big[..., None], 3, axis=2).astype(np.uint8)
            )
        return images

    def save_output(self, path):
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        widths = {}
        for name in EXPOSURES:
            image = self.radiance(name)
            row = int(np.argmax(image.max(axis=1)))
            band = image[row - 3 : row + 4].max(axis=0)
            spans = dots(band)
            widths[name] = [last - first + 1 for first, last in spans]
            print(f"{name}: dot widths {widths[name]}")
        short, long = widths["1 ms"], widths["500 ms"]
        # The rock's edge hides a dot.
        assert len(short) == len(long) == DOTS - 1, "10 dots each"
        # The left five on the rock face, the right five on the ground.
        rock = [b - a for a, b in zip(short[:5], long[:5])]
        ground = [b - a for a, b in zip(short[-5:], long[-5:])]
        assert min(rock) >= 5, (
            f"dots on the rock face should smear: {rock} px longer"
        )
        assert max(abs(g) for g in ground) <= 1, (
            f"dots on flat ground should not: {ground} px longer"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
