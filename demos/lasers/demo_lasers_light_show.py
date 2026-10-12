# SPDX-License-Identifier: BSD-3-Clause
"""A laser show: three projectors scan Lissajous figures through a
hazy hall faster than the camera sees, so each draws a hollow cone of
light in the haze and its figure on the far wall, the figures turning
as their phases drift.

References:
- J. A. Lissajous, "Memoire sur l'etude optique des mouvements
  vibratoires", Annales de Chimie et de Physique 51:147-231, 1857
  (the figures).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 200)
DENSITY = 0.08  # 1/m, the haze's extinction at its densest
STAGE = (0.0, 0.0, 2.0)  # m, the projectors' row, centre
WALL = 16.0  # m, the far wall along +y
DRIFT = 0.5  # radians a second, the figures' phase drift
# Each projector: its colour's wavelength (nm), power (W), place along
# the row, the figure's full angles (degrees), frequencies and phase.
PROJECTORS = {
    "Red": (638.0, 1.0, -1.5, (24.0, 24.0), (1.0, 1.0), 90.0),
    "Green": (532.0, 1.0, 0.0, (30.0, 20.0), (3.0, 2.0), 90.0),
    "Blue": (450.0, 1.5, 1.5, (24.0, 18.0), (5.0, 4.0), 0.0),
}


def make_haze():
    """A hall's haze, 12 m x 18 m x 6 m in 0.25 m voxels, thinning
    upwards, its billows soft."""
    voxel = 0.25
    low = np.array([-6.0, -1.0, 0.0])
    shape = (48, 72, 24)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    points = 0.3 * np.stack([x, y, z], -1) + 7.0
    billows = np.asarray(
        ernie.fbm_noise(points.reshape(-1, 3), octaves=3, seed=11)
    )
    value = np.clip(0.75 + 0.4 * billows.reshape(shape), 0.0, 1.0)
    value *= np.exp(-z / 8.0)
    density = value.astype(np.float32)
    return ernie.dense_nanovdb_grid(
        density, voxel, tuple(low + 0.5 * voxel), "density"
    )


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        looks.define("Floor", ernie.Material, albedo=(0.05,) * 3)
        looks.define("Wall", ernie.Material, albedo=(0.7,) * 3)
        world.define(
            "Floor",
            ernie.GroundPlane,
            material_binding="/World/Looks/Floor",
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            translate=(0.0, WALL, 3.0),
            scale=(12.0, 0.2, 6.0),
            material_binding="/World/Looks/Wall",
        )
        self.phases = {}
        self.lasers = {}
        for name, (
            wavelength,
            power,
            x,
            fov,
            frequencies,
            phase,
        ) in PROJECTORS.items():
            aperture = (STAGE[0] + x, STAGE[1], STAGE[2])
            self.lasers[name] = world.define(
                name,
                ernie.LaserLight,
                wavelength=wavelength,
                power=power,
                waist_radius=(0.002, 0.002),
                pattern="scan",
                pattern_field_of_view=fov,
                scan_frequencies=frequencies,
                scan_phase=phase,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    aperture,
                    (aperture[0], WALL, 3.2),
                    (0.0, 0.0, 1.0),
                ),
            )
            self.phases[name] = phase
        haze = world.define(
            "Haze", ernie.Volume, material_binding="/World/Looks/Haze"
        )
        haze.define("Density", ernie.NanoVDBField, buffer=make_haze())
        haze.set_field("density", "/World/Haze/Density")
        self.haze_look = looks.define(
            "Haze",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.95, 0.95, 0.95),
            anisotropy=0.5,
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=16.0,
            sensor_noise=False,
            exposure_time=1.0 / 30.0,
            exposure_f_stop=2.0,
            exposure_iso=400.0,
            clipping_range=(0.05, 1e4),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (5.5, -3.0, 1.6), (0.0, 8.0, 2.8), (0.0, 0.0, 1.0)
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
        self.drift = DRIFT
        self.widgets.slider(
            "Haze density (1/m)",
            0.0,
            0.3,
            DENSITY,
            on_change=self.thicken,
        )
        self.widgets.slider(
            "Phase drift (rad/s)",
            0.0,
            3.0,
            DRIFT,
            on_change=self.speed,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def thicken(self, density):
        self.haze_look.set(density=density)

    def speed(self, drift):
        self.drift = drift

    def step(self):
        # Each figure turns at its own rate.
        for i, name in enumerate(PROJECTORS):
            self.phases[name] += math.degrees(
                self.drift * self.frame_dt
            ) * (1.0 + 0.5 * i)
            self.lasers[name].scan_phase = self.phases[name] % 360.0

    def capture(self, lasers=True, density=DENSITY):
        """The radiance summed over the bands, with the projectors on
        or off and the haze at `density`."""
        for name in PROJECTORS:
            self.scene.set(
                f"/World/{name}",
                tokens.VISIBILITY,
                tokens.INHERITED if lasers else tokens.INVISIBLE,
            )
        self.haze_look.set(density=density)
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        frame = outputs["/Render/Camera"].frames[0].render_vars
        radiance = np.array(
            frame["/Render/Camera/SpectralRadiance"].map()
        ).astype(np.float64)
        self.haze_look.set(density=DENSITY)
        return radiance.sum(axis=0)

    def save_output(self, path):
        write_png(path, self.render_image())

    def test_final(self):
        lit = self.capture()
        clear = self.capture(density=0.0)
        dark = self.capture(lasers=False)
        # What the haze adds: the cones of light the scans sweep.
        cones = lit - clear
        threshold = 0.05 * np.percentile(cones, 99.9)
        swept = (cones > threshold).mean()
        # The figures on the wall, in clear air.
        drawn = (clear > 0.05 * np.percentile(clear, 99.9)).mean()
        print(
            f"the scans' cones light {100 * swept:.1f}% of the pixels "
            f"in the haze; their figures {100 * drawn:.1f}% on the "
            f"wall; with the projectors off, the brightest pixel is "
            f"{dark.max():.3g}"
        )
        assert swept > 0.05, "the cones show in the haze"
        assert 0.001 < drawn < 0.2, (
            "the figures draw lines on the wall"
        )
        assert dark.max() == 0.0, "nothing else lights the hall"


if __name__ == "__main__":
    ernie.demo.main(Demo)
