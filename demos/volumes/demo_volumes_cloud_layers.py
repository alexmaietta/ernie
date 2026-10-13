# SPDX-License-Identifier: BSD-3-Clause
"""Procedural cloud layers over the Earth, after three-geospatial's
clouds: low and mid cumulus and cirrus carved from a weather map and
Perlin-Worley noise at each sample (a CloudLayersField), drifting on the
wind, lit through the atmosphere by a two-lobed phase function, the
powder effect and multiple-scattering octaves. Seen from above, as over
Fuji, or from below, as over Tokyo."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    define_planet_surface,
    mean_linear_rgb,
    write_png,
)

COVERAGE = 0.4
SUN_ELEVATION = 20.0
SUN_AZIMUTH = 30.0
WIND = 15.0
# Each view: eye (m) and pitch (degrees), looking along -x.
VIEWS = {
    "Above the clouds": ((0.0, 0.0, 3800.0), -27.0),
    "Below the clouds": ((0.0, 0.0, 160.0), 9.0),
}


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        define_planet_surface(
            self.scene,
            "/World/Earth",
            6371.0088e3,
            (0.08, 0.12, 0.06),
            "/World/Looks/Ground",
        )
        world.define("Air", ernie.Atmosphere, preset=tokens.EARTH)
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
            shadow_enable=True,
        )
        self.elevation = SUN_ELEVATION
        self.azimuth = SUN_AZIMUTH
        self.place_sun()

        # three-geospatial's default layers (low and mid cumulus, cirrus)
        # over 60 km, their density scales in 1/m.
        clouds = world.define(
            "Clouds",
            ernie.Volume,
            material_binding="/World/Looks/Cloud",
        )
        self.layers = clouds.define(
            "Layers",
            ernie.CloudLayersField,
            coverage=COVERAGE,
            extent=30000.0,
        )
        clouds.set_field("density", "/World/Clouds/Layers")
        self.scene.define(
            "/World/Looks/Cloud",
            ernie.Material,
            density=1.0,
            volume_albedo=(1.0, 1.0, 1.0),
            phase_function=tokens.DUAL_HENYEY_GREENSTEIN,
            anisotropy=0.7,
            second_anisotropy=-0.2,
            lobe_mix=0.5,
            powder=0.8,
            powder_length=150.0,
            scattering_octaves=8,
            octave_attenuation=0.5,
            octave_extinction=0.5,
            octave_phase=0.5,
        )
        self.wind = WIND

        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=12.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=11.0,
            exposure_time=1.0 / 250.0,
        )
        self.look_from(0)
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(640, 360),
            ambient=0.0,
        )

        self.widgets.combo(
            "View", list(VIEWS), 0, on_change=self.look_from
        )
        self.widgets.slider(
            "Coverage", 0.0, 1.0, COVERAGE, on_change=self.cover
        )
        self.widgets.slider(
            "Sun elevation",
            1.0,
            90.0,
            SUN_ELEVATION,
            on_change=self.sun_elevation,
        )
        self.widgets.slider(
            "Sun azimuth",
            0.0,
            360.0,
            SUN_AZIMUTH,
            on_change=self.sun_azimuth,
        )
        self.widgets.slider(
            "Wind (m/s)",
            0.0,
            60.0,
            WIND,
            on_change=lambda speed: setattr(self, "wind", speed),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def step(self):
        # The weather drifts downwind, the clouds with it.
        self.layers.weather_offset = (-self.wind * self.time, 0.0)

    def look_from(self, index):
        eye, pitch = list(VIEWS.values())[index]
        p = math.radians(pitch)
        target = (
            eye[0] - 1000.0 * math.cos(p),
            eye[1],
            eye[2] + 1000.0 * math.sin(p),
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0)),
        )

    def cover(self, coverage):
        self.layers.coverage = coverage

    def sun_elevation(self, degrees):
        self.elevation = degrees
        self.place_sun()

    def sun_azimuth(self, degrees):
        self.azimuth = degrees
        self.place_sun()

    def place_sun(self):
        a = math.radians(self.azimuth)
        e = math.radians(self.elevation)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (
                    math.cos(a) * math.cos(e),
                    math.sin(a) * math.cos(e),
                    math.sin(e),
                )
            ),
        )

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        cloudy = mean_linear_rgb(
            np.array(self.renderer.render(self.scene))
        )
        self.cover(0.0)
        clear = mean_linear_rgb(
            np.array(self.renderer.render(self.scene))
        )
        self.cover(COVERAGE)
        print(
            f"Mean linear RGB with clouds {cloudy[0]:.3f} {cloudy[1]:.3f} "
            f"{cloudy[2]:.3f}, clear {clear[0]:.3f} {clear[1]:.3f} "
            f"{clear[2]:.3f}"
        )
        # Above the clouds, they whiten the view of the ground below.
        assert np.linalg.norm(cloudy - clear) > 0.01, (
            "the clouds show"
        )
        assert cloudy[0] / cloudy[2] > clear[0] / clear[2], (
            "the clouds are whiter than the ground and air"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
