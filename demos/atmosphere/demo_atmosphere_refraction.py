# SPDX-License-Identifier: BSD-3-Clause
"""Refraction lifts and flattens the low sun and Moon: a telephoto
sunset, the sun seen after it sets, moonrise, and sunset from the ISS."""

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

MOON_DISTANCE = 384400e3

# A camera's altitude, aim and focal length, the sun's true elevation
# towards +x, and an exposure.
VIEWS = [
    (
        "Sunset, the sun flattened",
        2.0,
        0.0,
        0.5,
        1200.0,
        0.0,
        16.0,
        1.0 / 1000.0,
        -3.0,
    ),
    (
        "Set, and still seen",
        2.0,
        0.0,
        0.2,
        1200.0,
        -0.47,
        16.0,
        1.0 / 1000.0,
        -1.0,
    ),
    (
        "Full Moon rising",
        2.0,
        183.0,
        1.9,
        600.0,
        -0.8,
        11.0,
        1.0 / 30.0,
        0.0,
    ),
    # Bent going into the air and again coming out.
    (
        "Sunset from the ISS",
        400e3,
        0.0,
        -19.6,
        600.0,
        -19.95,
        16.0,
        1.0 / 1000.0,
        -11.0,
    ),
]


def towards(azimuth, elevation):
    a = math.radians(azimuth)
    e = math.radians(elevation)
    return (
        math.cos(e) * math.cos(a),
        math.cos(e) * math.sin(a),
        math.sin(e),
    )


def bright_extent(image):
    """The (top, bottom, left, right) extent of the pixels of a uint8
    image whose green is above half its brightest, or None."""
    green = np.asarray(image)[..., 1].astype(np.int32)
    rows, cols = np.nonzero(2 * green > green.max())
    if rows.size == 0:
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()


def region_linear_rgb(image, rows, cols):
    """The mean linear colour of a region of a uint8 image."""
    rgb = srgb_to_linear(np.asarray(image)[rows, cols, :3] / 255.0)
    return rgb.reshape(-1, 3).mean(axis=0)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        ernie.demo.define_planet_surface(
            self.scene,
            "/World/Earth",
            6371.0088e3,
            (0.3, 0.3, 0.3),
            "/World/Looks/Ground",
        )
        world.define("Air", ernie.Atmosphere, preset=tokens.EARTH)
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        )
        # The full Moon opposite the sun, aside from the Earth's shadow.
        world.define(
            "Moon",
            ernie.CelestialBody,
            radius=1737.4e3,
            albedo=(0.12, 0.12, 0.12),
            translate=tuple(
                MOON_DISTANCE * c for c in towards(183.0, 1.5)
            ),
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
            "Sun elevation",
            -25.0,
            3.0,
            VIEWS[0][5],
            on_change=self.sun_at,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def show(self, index):
        self.view = index
        (
            _,
            altitude,
            azimuth,
            elevation,
            focal_length,
            sun_elevation,
            f_stop,
            exposure_time,
            exposure,
        ) = VIEWS[index]
        self.camera.focal_length = focal_length
        self.camera.exposure_f_stop = f_stop
        self.camera.exposure_time = exposure_time
        self.camera.exposure = exposure
        eye = (0.0, 0.0, altitude)
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                eye,
                tuple(np.add(eye, towards(azimuth, elevation))),
                (0.0, 0.0, 1.0),
            ),
        )
        self.sun_at(sun_elevation)

    def sun_at(self, degrees):
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(towards(0.0, degrees)),
        )

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
        # The low sun: flattened and lifted.
        pixels_per_degree = 480.0 / math.degrees(
            2.0 * math.atan(20.955 / 2.0 / 1200.0)
        )
        for index in (0, 1):
            name, _, _, elevation, _, sun_elevation, *_ = VIEWS[index]
            extent = bright_extent(images[index])
            assert extent is not None, "the low sun should show"
            top, bottom, left, right = extent
            wide = right - left + 1
            high = bottom - top + 1
            centre = (
                elevation
                + (
                    images[index].shape[0] / 2.0
                    - 0.5 * (top + bottom)
                )
                / pixels_per_degree
            )
            print(
                f"{name}: {wide} x {high} pixels, centre {centre:.2f} "
                f"degrees up (true {sun_elevation:.2f})"
            )
            assert high < 0.92 * wide, (
                "the low sun should show flattened"
            )
            assert centre - sun_elevation > 0.4, (
                "the low sun should show lifted"
            )
        # The rising Moon: redder than the sky beside it.
        middle = images[2].shape[0] // 2
        rows = slice(middle - 4, middle + 4)
        moon = region_linear_rgb(images[2], rows, slice(236, 244))
        sky = region_linear_rgb(images[2], rows, slice(0, 40))
        print(f"rising Moon {moon}, sky {sky}")
        assert moon[0] > sky[0] and (
            moon[0] / moon[2] > 1.3 * sky[0] / sky[2]
        ), "the rising Moon should show reddened"
        # From orbit: squashed.
        extent = bright_extent(images[3])
        assert extent is not None, "the sun should show from orbit"
        top, bottom, left, right = extent
        wide = right - left + 1
        high = bottom - top + 1
        print(f"{VIEWS[3][0]}: {wide} x {high} pixels")
        assert high < 0.8 * wide, (
            "the sun should show squashed from orbit"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
