# SPDX-License-Identifier: BSD-3-Clause
"""Mars's atmosphere from a rover and from orbit: the dusty noon sky,
the blue sunset and the thin limb."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import mean_linear_rgb, side_by_side, write_png

# The Sun's irradiance at Mars's mean distance, 1.5237 au (its orbit's
# semi-major axis): the total solar irradiance at 1 au over the
# distance squared.
MARS_IRRADIANCE = 1361.0 / 1.5237**2

# A camera and the sun's elevation (degrees above the horizon at the
# origin), with an exposure for its light: a stop more than on Earth in
# daylight (the sun is 43% as bright).
VIEWS = [
    (
        "Rover, noon",
        (0.0, 0.0, 2.0),
        (-1000.0, 0.0, 350.0),
        60.0,
        11.0,
    ),
    (
        "Rover, sunset",
        (0.0, 0.0, 2.0),
        (1000.0, 0.0, 60.0),
        3.0,
        16.0,
    ),
    (
        "Orbiter, 300 km",
        (0.0, 0.0, 300e3),
        (0.0, 1e6, 0.0),
        30.0,
        11.0,
    ),
    # Looking towards the sun at the limb, 23.3 degrees below the
    # horizontal from 300 km, where the sun sets.
    (
        "Limb at sunset",
        (0.0, 0.0, 300e3),
        (1e6, 0.0, -131e3),
        -22.0,
        2.8,
    ),
]


def mean_linear_rgb_in(image, top, bottom, left, right):
    """The mean linear colour of the rows [top, bottom) and columns
    [left, right) of an image, as fractions of its size."""
    image = np.asarray(image)
    rows, cols = image.shape[:2]
    return mean_linear_rgb(
        image[
            int(top * rows) : int(bottom * rows),
            int(left * cols) : int(right * cols),
        ]
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        # Mars's surface at the world's z = 0 (reddish soil, albedo about
        # its mean 0.25), and its carbon dioxide and dust (feldspar with
        # hematite).
        ernie.demo.define_planet_surface(
            self.scene,
            "/World/Mars",
            3389.5e3,
            (0.33, 0.22, 0.14),
            "/World/Looks/Soil",
        )
        self.air = world.define(
            "Air",
            ernie.Atmosphere,
            preset=tokens.MARS,
            dust_optical_depth=0.5,
            ground_albedo=0.25,
        )
        # The Sun's measured spectrum at Mars's distance.
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=MARS_IRRADIANCE,
            physical_illuminant=tokens.SUN,
            angle=0.35,
        )
        # Reddish rocks on the ground (albedo about Mars's mean 0.25).
        for i in range(3):
            world.define(
                f"Rock{i}",
                ernie.Sphere,
                radius=0.6 + 0.3 * i,
                display_color=(0.35, 0.2, 0.12),
                translate=(
                    -12.0 if i == 0 else 15.0,
                    -5.0 + 5.0 * i,
                    0.5,
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
            -25.0,
            90.0,
            VIEWS[0][3],
            on_change=self.sun_at,
        )
        # About 0.3 to 0.9 in clear seasons; dust storms above 1.
        self.widgets.slider(
            "Dust optical depth",
            0.0,
            5.0,
            0.5,
            on_change=self.dust,
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

    def dust(self, depth):
        self.air.dust_optical_depth = depth

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
        # The noon sky away from the sun, the top of the frame.
        noon = mean_linear_rgb_in(images[0], 0.0, 0.3, 0.0, 1.0)
        # At sunset, the sky around the sun and far to its side.
        aureole = mean_linear_rgb_in(images[1], 0.25, 0.45, 0.4, 0.6)
        aside = mean_linear_rgb_in(images[1], 0.0, 0.2, 0.0, 0.15)
        orbit = mean_linear_rgb(images[2])
        for name, c in [
            ("noon sky", noon),
            ("around the setting sun", aureole),
            ("aside at sunset", aside),
        ]:
            print(f"{name}: {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        # Lemmon et al. 2004: dust makes the daytime sky butterscotch,
        # and scatters blue light forward around the setting sun.
        assert noon[0] > 1.1 * noon[2], (
            "the noon sky should be redder than it is blue"
        )
        assert aureole[2] / aureole[0] > 1.2 * aside[2] / aside[0], (
            "the sky around the setting sun should be bluer"
        )
        assert orbit.sum() > 0.03, (
            "the planet should be lit from orbit"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
