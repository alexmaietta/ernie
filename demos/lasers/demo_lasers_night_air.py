# SPDX-License-Identifier: BSD-3-Clause
"""Laser beams in the night air: a green and a blue beam and a red fan
rise from the ground, shown only by the light the air scatters, the
haze lighting them more and a balloon in the green beam stopping it.

References:
- C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
  by Small Particles", Wiley, 1983 (a beam seen side-on by the light
  its path scatters: the gas's Rayleigh scattering, the haze's or
  dust's forward peak).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 200)
RIG = (0.0, 0.0, 1.5)  # m, the lasers' aperture
HAZE = 0.1  # the haze's (Earth's) or dust's (Mars's) optical depth
AIRS = ["Earth", "Mars"]


def aim(elevation, azimuth):
    """The lasers' rig turned to `elevation` above the horizon and
    `azimuth` from +y towards +x, in degrees."""
    e = math.radians(elevation)
    a = math.radians(azimuth)
    target = (
        RIG[0] + math.cos(e) * math.sin(a),
        RIG[1] + math.cos(e) * math.cos(a),
        RIG[2] + math.sin(e),
    )
    return ernie.look_at_transform(RIG, target, (0.0, 0.0, 1.0))


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        looks.define("Ground", ernie.Material, albedo=(0.1,) * 3)
        looks.define("Balloon", ernie.Material, albedo=(0.7,) * 3)
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Ground",
        )
        self.air = world.define(
            "Air", ernie.Atmosphere, haze_optical_depth=HAZE
        )
        world.define(
            "Green",
            ernie.LaserLight,
            wavelength=532.0,
            power=5.0,
            waist_radius=(0.003, 0.003),
        ).set(tokens.TRANSFORM, aim(50.0, 0.0))
        world.define(
            "Blue",
            ernie.LaserLight,
            wavelength=450.0,
            power=3.0,
            waist_radius=(0.003, 0.003),
        ).set(tokens.TRANSFORM, aim(70.0, -30.0))
        world.define(
            "Red",
            ernie.LaserLight,
            wavelength=638.0,
            power=3.0,
            waist_radius=(0.003, 0.003),
            pattern="line",
            fan_angle=40.0,
        ).set(tokens.TRANSFORM, aim(30.0, 25.0))
        # A balloon in the green beam, 15 m along it.
        e = math.radians(50.0)
        world.define(
            "Balloon",
            ernie.Sphere,
            radius=1.0,
            translate=(
                RIG[0],
                RIG[1] + 15.0 * math.cos(e),
                RIG[2] + 15.0 * math.sin(e),
            ),
            material_binding="/World/Looks/Balloon",
        )
        # A long, wide-open exposure, as for the night sky.
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=14.0,
            sensor_noise=False,
            exposure_time=1.0,
            exposure_f_stop=1.4,
            exposure_iso=6400.0,
            clipping_range=(0.05, 1e5),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-20.0, -12.0, 1.7),
                (0.0, 12.0, 14.0),
                (0.0, 0.0, 1.0),
            ),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        var = scene.define(
            "/Render/Camera/SpectralRadiance", ernie.RenderVar
        )
        var.source_name = "SpectralRadiance"
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=[var.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.widgets.combo("Air", AIRS, 0, on_change=self.breathe)
        self.widgets.slider(
            "Haze or dust optical depth",
            0.0,
            2.0,
            HAZE,
            on_change=self.thicken,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def breathe(self, index):
        self.air.preset = tokens.MARS if index == 1 else tokens.EARTH

    def thicken(self, depth):
        self.air.haze_optical_depth = depth
        self.air.dust_optical_depth = depth

    def capture(self, lasers=True):
        """The radiance summed over the bands, with the lasers on or
        off."""
        for name in ("Green", "Blue", "Red"):
            self.scene.set(
                f"/World/{name}",
                tokens.VISIBILITY,
                tokens.INHERITED if lasers else tokens.INVISIBLE,
            )
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        frame = outputs["/Render/Camera"].frames[0].render_vars
        radiance = np.array(
            frame["/Render/Camera/SpectralRadiance"].map()
        ).astype(np.float64)
        return radiance.sum(axis=0)

    def save_output(self, path):
        write_png(path, self.render_image())

    def test_final(self):
        clear = self.capture()
        self.thicken(1.0)
        hazy = self.capture()
        self.thicken(HAZE)
        dark = self.capture(lasers=False)
        # The beams by the brightest of their pixels, past the few the
        # green spot lights on the balloon.
        threshold = 0.05 * np.percentile(clear, 99.9)
        visible = (clear > threshold).mean()
        ratio = np.percentile(hazy, 99.0) / np.percentile(clear, 99.0)
        print(
            f"the beams light {100 * visible:.1f}% of the pixels in "
            f"clear air; haze of optical depth 1 scatters {ratio:.1f} "
            f"times their light; with the lasers off, the brightest "
            f"pixel is {dark.max():.3g}"
        )
        assert visible > 0.003, "the beams show in the air"
        assert ratio > 1.5, "haze shows them more"
        assert dark.max() == 0.0, "nothing else lights the night"


if __name__ == "__main__":
    ernie.demo.main(Demo)
