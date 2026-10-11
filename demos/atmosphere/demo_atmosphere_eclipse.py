# SPDX-License-Identifier: BSD-3-Clause
"""Solar eclipses by a CelestialBody Moon: partial and annular through
a solar filter, totality from the ground, and its shadow from orbit."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    mean_linear_rgb,
    side_by_side,
    srgb_to_linear,
    write_png,
)

SUN_ELEVATION = 40.0  # degrees

# A camera, the Moon's distance and offset from the sun, and an exposure
# (-13.3 stops is an ND 4.0 solar filter). At 360,000 km the Moon covers
# the sun; at 405,500 km it leaves a ring.
VIEWS = [
    (
        "Partial, through a solar filter",
        (0.0, 0.0, 2.0),
        SUN_ELEVATION,
        (0.0, 0.0, 1.0),
        1200.0,
        384400e3,
        0.3,
        16.0,
        1.0 / 1000.0,
        -13.3,
    ),
    (
        "Annular, through a solar filter",
        (0.0, 0.0, 2.0),
        SUN_ELEVATION,
        (0.0, 0.0, 1.0),
        1200.0,
        405500e3,
        0.0,
        16.0,
        1.0 / 1000.0,
        -13.3,
    ),
    (
        "Totality, from the ground",
        (0.0, 0.0, 2.0),
        22.0,
        (0.0, 0.0, 1.0),
        14.0,
        360000e3,
        0.0,
        2.8,
        1.0 / 2.0,
        0.0,
    ),
    (
        "The Moon's shadow from geostationary orbit",
        (0.0, 0.0, 35786e3),
        -90.0,
        (1.0, 0.0, 0.0),
        100.0,
        360000e3,
        0.0,
        16.0,
        1.0 / 125.0,
        0.0,
    ),
]


def region_luminance(image, rows, cols):
    """The mean linear luminance of a region of a uint8 image."""
    rgb = srgb_to_linear(np.asarray(image)[rows, cols, :3] / 255.0)
    return float(
        (rgb.reshape(-1, 3) @ [0.2126, 0.7152, 0.0722]).mean()
    )


def bright_pixels(image):
    """The pixels of a uint8 image whose green is above half white."""
    return int((np.asarray(image)[..., 1] > 128).sum())


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Air", ernie.Atmosphere, preset=tokens.EARTH)
        e = math.radians(SUN_ELEVATION)
        sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        )
        sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (math.cos(e), 0.0, math.sin(e))
            ),
        )
        # The Moon: its mean radius and normal albedo.
        self.moon = world.define(
            "Moon",
            ernie.CelestialBody,
            radius=1737.4e3,
            albedo=(0.12, 0.12, 0.12),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            sensor_noise=False,
            exposure_iso=100.0,
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(480, 320),
            ambient=0.0,
        )
        self.show(0)

        self.widgets.combo(
            "View",
            [name for name, *_ in VIEWS],
            0,
            on_change=self.show,
        )
        self.widgets.slider(
            "Moon offset",
            -1.0,
            1.0,
            VIEWS[0][6],
            on_change=lambda degrees: self.moon_at(
                VIEWS[self.view][5], degrees
            ),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def show(self, index):
        self.view = index
        (
            _,
            eye,
            elevation,
            up,
            focal_length,
            distance,
            offset,
            f_stop,
            exposure_time,
            exposure,
        ) = VIEWS[index]
        a = math.radians(elevation)
        self.camera.focal_length = focal_length
        self.camera.exposure_f_stop = f_stop
        self.camera.exposure_time = exposure_time
        self.camera.exposure = exposure
        target = np.add(eye, (math.cos(a), 0.0, math.sin(a)))
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, tuple(target), up),
        )
        self.moon_at(distance, offset)

    def moon_at(self, distance, degrees):
        """The Moon `distance` from the world's origin, `degrees` from
        the sun's direction towards the zenith."""
        e = math.radians(SUN_ELEVATION + degrees)
        self.moon.set_xform(
            translate=(
                distance * math.cos(e),
                0.0,
                distance * math.sin(e),
            )
        )

    def render_without_moon(self, index):
        """View `index` with the Moon far from the sun; leaves the
        view as it was."""
        current = self.view
        self.show(index)
        self.moon_at(VIEWS[index][5], -30.0)
        image = np.array(self.renderer.render(self.scene))
        self.show(current)
        return image

    def render_all(self):
        """The scene from each view in turn, RGBA uint8; leaves the
        view as it was."""
        current = self.view
        images = []
        for index in range(len(VIEWS)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(current)
        return images

    def save_output(self, path):
        """The four views side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        images = self.render_all()
        for (name, *_), image in zip(VIEWS, images):
            m = mean_linear_rgb(image)
            print(
                f"{name}: mean linear RGB {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}"
            )
        # A crescent and a ring: less of the sun than its whole disc.
        whole = bright_pixels(self.render_without_moon(0))
        crescent = bright_pixels(images[0])
        ring = bright_pixels(images[1])
        print(
            f"bright pixels: whole {whole}, crescent {crescent}, ring {ring}"
        )
        assert 0 < crescent < 0.8 * whole, (
            "the partial eclipse should leave a crescent"
        )
        assert 0 < ring < 0.3 * whole, (
            "the annular eclipse should leave a thin ring"
        )
        rows, cols = images[1].shape[:2]
        middle = (
            slice(rows // 2 - 2, rows // 2 + 2),
            slice(cols // 2 - 2, cols // 2 + 2),
        )
        assert region_luminance(images[1], *middle) < 0.01, (
            "the Moon should be dark against the sun"
        )
        # Totality: a dark sky, brightest along the horizon.
        everything = (slice(None), slice(None))
        total = region_luminance(images[2], *everything)
        daylight = region_luminance(
            self.render_without_moon(2), *everything
        )
        print(f"totality {total / daylight:.5f} of daylight")
        assert total < 0.05 * daylight, (
            "totality should darken the sky"
        )
        horizon = region_luminance(
            images[2], slice(rows - rows // 5, rows), slice(None)
        )
        overhead = region_luminance(
            images[2], slice(0, rows // 5), slice(None)
        )
        print(f"totality: horizon {horizon:.5f}, top {overhead:.5f}")
        assert horizon > 2.0 * overhead, (
            "the horizon should glow in totality"
        )
        # From orbit: a dark umbra under the camera.
        umbra = region_luminance(images[3], *middle)
        around = region_luminance(
            images[3], slice(0, rows // 8), slice(None)
        )
        print(f"from orbit: umbra {umbra:.5f}, around {around:.5f}")
        assert umbra < 0.1 * around, (
            "the umbra should be dark from orbit"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
