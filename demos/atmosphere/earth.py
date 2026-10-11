# SPDX-License-Identifier: BSD-3-Clause
"""Earth's atmosphere from the ground and from orbit: a blue noon sky,
a red sunset, and the planet's limb from the International Space
Station's 400 km, under the Sun's 1361 W/m^2 as a 5772 K black
body."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import mean_linear_rgb, side_by_side, write_png

# A camera and the sun's elevation (degrees above the horizon at the
# origin), with an exposure for its light: the "sunny 16" rule (f/16 at
# 1/ISO s) in daylight, opened up at sunset.
VIEWS = [
    (
        "Ground, noon",
        (0.0, 0.0, 2.0),
        (-1000.0, 0.0, 250.0),
        60.0,
        16.0,
    ),
    (
        "Ground, sunset",
        (0.0, 0.0, 2.0),
        (1000.0, 0.0, 60.0),
        2.0,
        5.6,
    ),
    ("ISS, 400 km", (0.0, 0.0, 400e3), (0.0, 1e6, 0.0), 30.0, 16.0),
    # Over the night side, looking towards the sun at the limb, 19.8
    # degrees below the horizontal from 400 km, where the sun sets.
    (
        "ISS, limb at sunset",
        (0.0, 0.0, 400e3),
        (1e6, 0.0, 40e3),
        -19.0,
        4.0,
    ),
]


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        # Earth's air, ozone and haze around a planet whose surface is
        # the world's z = 0, the ground beyond any geometry its own.
        world.define("Air", ernie.Atmosphere, preset=tokens.EARTH)
        # The Sun above the atmosphere.
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.BLACKBODY,
            color_temperature=5772.0,
            angle=0.53,
        )
        # Grey spheres on the ground, lit by the sun through the air.
        for i in range(3):
            world.define(
                f"Rock{i}",
                ernie.Sphere,
                radius=1.0,
                display_color=(0.5, 0.5, 0.5),
                translate=(
                    -25.0 if i == 0 else 30.0,
                    -6.0 + 6.0 * i,
                    1.0,
                ),
            )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=14.0,
            sensor_noise=False,
            exposure_time=1.0 / 125.0,
            exposure_iso=125.0,
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
            -20.0,
            90.0,
            VIEWS[0][3],
            on_change=self.sun_at,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def show(self, index):
        self.view = index
        _, eye, target, elevation, f_stop = VIEWS[index]
        self.camera.exposure_f_stop = f_stop
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0)),
        )
        self.sun_at(elevation)

    def sun_at(self, degrees):
        e = math.radians(degrees)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (math.cos(e), 0.0, math.sin(e))
            ),
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
        means = [
            mean_linear_rgb(image) for image in self.render_all()
        ]
        for (name, *_), m in zip(VIEWS, means):
            print(
                f"{name}: mean linear RGB {m[0]:.3f} {m[1]:.3f} {m[2]:.3f}"
            )
        # Rayleigh scattering: a blue sky at noon, and a red sun low
        # through the long path at sunset.
        assert means[0][2] > 1.2 * means[0][0], (
            "the noon sky should be blue"
        )
        assert (
            means[1][0] / means[1][2]
            > 2.0 * means[0][0] / means[0][2]
        ), "the sunset should be redder than noon"
        assert means[2].sum() > 0.05, (
            "the planet should be lit from orbit"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
