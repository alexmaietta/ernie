# SPDX-License-Identifier: BSD-3-Clause
"""A ridge 6 km high on a Moon-sized planet Terrain under a sun 10
degrees up, seen from 300 km: its shadow through the cascades alone,
through the terrain's horizon maps alone, and through both. The cascades
show acne and ragged edges on the curved ground far from the ridge."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, srgb_to_linear, write_png

MOON_RADIUS = 1737.4e3
CELLS = 1024
RIDGE_HEIGHT = 6000.0
ALTITUDE = 300e3
EYE_Y = -30e3
WIDTH, HEIGHT = 640, 360

# What shadows the ground: horizon maps on or off, and how far the
# cascades reach (0 for as far as the view sees).
MODES = [
    ("Cascades alone", False, 0.0),
    ("Horizon maps alone", True, 10.0),
    ("Both", True, 0.0),
]


def ridge_heights():
    """Six cube faces of heights, a ridge along face 0's meridian."""
    points = CELLS + 1
    heights = np.zeros((6, points, points), np.float32)
    heights[0, :, CELLS // 2 - 2 : CELLS // 2 + 3] = RIDGE_HEIGHT
    return heights


def ground_green(image, y):
    """The linear green along the middle rows at ground y (m)."""
    u = (y - EYE_Y) / (ALTITUDE * 18.0 / 24.0)
    col = int((u + 1.0) * 0.5 * WIDTH)
    rows = slice(HEIGHT // 2 - 8, HEIGHT // 2 + 8)
    green = np.asarray(image)[rows, col, 1] / 255.0
    return float(srgb_to_linear(green).mean())


def acne(image):
    """How much the lit ground far west of the ridge varies from column
    to column, mean over rows: rings of acne raise it."""
    green = srgb_to_linear(np.asarray(image)[:, :64, 1] / 255.0)
    return float(np.abs(np.diff(green, axis=1)).mean())


class Demo(ernie.demo.Demo):
    num_frames = 3

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        self.scene.define(
            "/World/Looks/Regolith", ernie.Material, albedo=(0.3,) * 3
        )
        moon = world.define(
            "Moon",
            ernie.Terrain,
            planet_radius=MOON_RADIUS,
            material_binding="/World/Looks/Regolith",
        )
        self.scene.write(moon.path, tokens.HEIGHTS, ridge_heights())
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            horizontal_aperture=36.0,
            vertical_aperture=36.0 * HEIGHT / WIDTH,
            sensor_noise=False,
            exposure=-14.0,
            clipping_range=(1e3, 1e7),
        )
        camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (MOON_RADIUS + ALTITUDE, EYE_Y, 0.0),
                (MOON_RADIUS, EYE_Y, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.settings = self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(WIDTH, HEIGHT),
            ambient=0.0,
        )
        self.sun_at(10.0)
        self.show(2)

        self.widgets.combo(
            "Shadows",
            [name for name, *_ in MODES],
            2,
            on_change=self.show,
        )
        self.widgets.slider(
            "Sun elevation", 2.0, 30.0, 10.0, on_change=self.sun_at
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def sun_at(self, degrees):
        """The sun `degrees` up over the ridge, from the east (+y)."""
        e = math.radians(degrees)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (math.sin(e), math.cos(e), 0.0)
            ),
        )

    def show(self, index):
        self.mode = index
        _, horizons, distance = MODES[index]
        self.settings.set(tokens.SHADOW_HORIZONS, horizons)
        self.settings.set(tokens.SHADOW_DISTANCE, distance)

    def render_all(self):
        """Each mode's image, RGBA uint8; leaves the mode as it was."""
        current = self.mode
        images = []
        for index in range(len(MODES)):
            self.show(index)
            for _ in range(self.num_frames):
                image = np.array(self.renderer.render(self.scene))
            images.append(image)
        self.show(current)
        return images

    def save_output(self, path):
        """The three modes side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        images = self.render_all()
        for (name, *_), image in zip(MODES, images):
            print(f"{name}: acne {acne(image):.4f}")
        # The ridge's foot at y = -5.4 km; the ground falling away as
        # d^2 / 2R meets the ray from its top 36 km on, at -41.6 km.
        for (name, *_), image in zip(MODES, images):
            lit = ground_green(image, -60e3)
            shadow = [ground_green(image, y) for y in (-15e3, -35e3)]
            print(f"{name}: lit {lit:.4f}, shadow {shadow}")
            assert 0.03 < lit < 0.9, "the open ground should show lit"
            assert ground_green(image, 40e3) > lit, (
                "the sunward side, under a higher sun, should be brighter"
            )
            assert all(s < 0.1 * lit for s in shadow), (
                f"{name}: the ridge should shadow 36 km of ground"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
