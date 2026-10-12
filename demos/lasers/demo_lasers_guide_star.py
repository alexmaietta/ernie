# SPDX-License-Identifier: BSD-3-Clause
"""A sodium laser guide star: a 20 W beam tuned to sodium's 589 nm D2
line rises to the mesosphere, where the sodium layer 92 km up glows
back as an artificial star; seen from behind the laser, and through
a telescope 30 m away that sees it drawn out along the beam.

References:
- R. Foy and A. Labeyrie, "Feasibility of adaptive telescope with
  laser probe", Astronomy and Astrophysics 152:L29-L31, 1985 (the
  sodium laser guide star).
- D. A. Steck, "Sodium D Line Data" (the D2 line).
  https://steck.us/alkalidata/
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 240)
D2 = 589.158  # nm, sodium's D2 line in vacuum
LAUNCH = (0.0, 0.0, 2.0)  # m, the laser's aperture
TELESCOPE = (30.0, 0.0, 2.0)  # m, the telescope beside it
ZENITH = 10.0  # degrees, the beam's angle from the zenith
LAYER = 92e3  # m, the sodium layer's height

# Where the beam meets the sodium layer.
SPOT = (
    LAUNCH[0],
    LAUNCH[1] + LAYER * math.tan(math.radians(ZENITH)),
    LAUNCH[2] + LAYER,
)
# A camera's place, target, focal length (mm) and exposure (s, f-number
# and ISO).
VIEWS = {
    "Behind the laser": (
        (8.0, -30.0, 1.7),
        (0.0, 400.0, 1200.0),
        12.0,
        (4.0, 1.4, 6400.0),
    ),
    "Telescope": (TELESCOPE, SPOT, 50e3, (1.0, 12.0, 1600.0)),
}


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        scene.define("/World/Looks", ernie.Xform).define(
            "Ground", ernie.Material, albedo=(0.1,) * 3
        )
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Ground",
        )
        self.air = world.define("Air", ernie.Atmosphere)
        # A launch telescope's beam, 0.3 m across, its line a few
        # megahertz wide.
        self.laser = world.define(
            "Laser",
            ernie.LaserLight,
            wavelength=D2,
            linewidth=1e-5,
            power=20.0,
            waist_radius=(0.15, 0.15),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(LAUNCH, SPOT, (0.0, 1.0, 0.0)),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            sensor_noise=False,
            clipping_range=(0.05, 1e6),
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
        self.look(0)
        self.widgets.combo(
            "View", list(VIEWS), 0, on_change=self.look
        )
        self.widgets.checkbox(
            "Tuned to the D2 line", True, on_change=self.tune
        )
        # About 2e13 to 8e13 atoms per square metre over the seasons.
        self.widgets.slider(
            "Sodium column (1e13 / m^2)",
            0.0,
            10.0,
            4.0,
            on_change=self.sodium,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def look(self, index):
        eye, target, focal, (time, f_stop, iso) = list(
            VIEWS.values()
        )[index]
        self.camera.set(
            focal_length=focal,
            exposure_time=time,
            exposure_f_stop=f_stop,
            exposure_iso=iso,
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0)),
        )

    def tune(self, tuned):
        # Two hundredths of a nanometre off, far past the line's width.
        self.laser.wavelength = D2 if tuned else D2 - 0.02

    def sodium(self, column):
        self.air.sodium_column = column * 1e13

    def capture(self):
        """The radiance summed over the bands."""
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
        self.look(1)
        tuned = self.capture()
        self.tune(False)
        detuned = self.capture()
        self.tune(True)
        self.look(0)
        # The star's pixels, and how far they reach along and across
        # the image.
        rows, cols = np.nonzero(tuned > 0.1 * tuned.max())
        extent = (np.ptp(rows) + 1, np.ptp(cols) + 1)
        print(
            f"the guide star peaks at {tuned.max():.3g} W m^-2 sr^-1 "
            f"nm^-1, {tuned.max() / max(detuned.max(), 1e-30):.3g} "
            f"times the detuned beam's; {extent[0]} x {extent[1]} "
            "pixels, drawn out along the beam"
        )
        assert tuned.max() > 100.0 * detuned.max(), "sodium glows"
        assert max(extent) > 2 * min(extent), "elongated"


if __name__ == "__main__":
    ernie.demo.main(Demo)
