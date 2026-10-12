# SPDX-License-Identifier: BSD-3-Clause
"""Cumulus clouds over the Earth under its atmosphere, after
three-geospatial's clouds: a Volume whose grid is built here from a
coverage map, a height profile and eroding noise, scattering many times
and lit by the sun, the sky and the ground."""

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

COVERAGE = 0.55
SUN_ELEVATION = 25.0


def make_clouds(coverage):
    """A field of cumulus over a 20 km square, from 600 m up to 2100 m,
    in 50 m voxels, its density over 0 to 1: where a low-frequency
    coverage map lets clouds form, a profile rounding their bases and
    tops, and billowing noise eroded by finer noise at their edges.

    References: A. Schneider and N. Vos, "The Real-Time Volumetric
    Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances in
    Real-Time Rendering; takram, three-geospatial's clouds, 2024."""
    voxel = 50.0
    base_height = 600.0
    thickness = 1500.0
    low = np.array([-10000.0, -10000.0, base_height])
    shape = (400, 400, 30)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    # Where clouds form: coverage's share of the map, its edges soft.
    columns = np.stack(
        [
            x[..., 0] / 4000.0,
            y[..., 0] / 4000.0,
            np.full(shape[:2], 0.5),
        ],
        -1,
    )
    cloud_map = 0.5 + 0.5 * ernie.fbm_noise(
        columns.reshape(-1, 3), octaves=3, seed=21
    ).reshape(shape[:2])
    cover = np.clip((cloud_map - (1.0 - coverage)) / 0.25, 0.0, 1.0)[
        ..., None
    ]
    # Flat bases and rounded tops, taller where cover is full.
    h = (z - base_height) / thickness
    profile = np.clip(h / 0.08, 0.0, 1.0) * np.clip(
        (0.35 + 0.65 * cover - h) / 0.25, 0.0, 1.0
    )
    points = np.stack([x, y, z], -1).reshape(-1, 3)
    billows = 0.5 + 0.5 * ernie.fbm_noise(
        points / 900.0, octaves=4, seed=21
    ).reshape(shape)
    base = np.clip((billows * profile * cover - 0.35) / 0.4, 0.0, 1.0)
    # Finer noise wears the edges away, wispier below.
    detail = 0.5 + 0.5 * ernie.fbm_noise(
        points / 180.0 + 31.0, octaves=3, seed=21
    ).reshape(shape)
    value = base - (1.0 - base) * detail * (0.6 - 0.3 * h)
    keep = (
        (cover > 0.0)
        & (profile > 0.0)
        & (base > 0.0)
        & (value > 1e-3)
    )
    density = np.where(keep, value, 0.0).astype(np.float32)
    return ernie.dense_nanovdb_grid(
        density, voxel, tuple(low + 0.5 * voxel), "density"
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        # The Earth's surface at z = 0, grassland's albedo, and its air.
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
        self.sun_at(SUN_ELEVATION)

        # The clouds: water droplets scatter nearly all the light,
        # mostly forwards, many times over.
        clouds = world.define(
            "Clouds",
            ernie.Volume,
            material_binding="/World/Looks/Cloud",
        )
        self.density = clouds.define("Density", ernie.NanoVDBField)
        clouds.set_field("density", "/World/Clouds/Density")
        self.scene.define(
            "/World/Looks/Cloud",
            ernie.Material,
            density=0.04,
            volume_albedo=(0.99, 0.99, 0.99),
            anisotropy=0.6,
            scattering_octaves=6,
            octave_attenuation=0.5,
            octave_extinction=0.5,
            octave_phase=0.5,
        )
        self.cover(COVERAGE)

        # From 300 m up, looking across the field, the sun behind.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=18.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=11.0,
            exposure_time=1.0 / 250.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -9000.0, 300.0),
                (2000.0, 0.0, 1500.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(480, 320),
            ambient=0.0,
        )

        self.widgets.slider(
            "Sun elevation",
            2.0,
            90.0,
            SUN_ELEVATION,
            on_change=self.sun_at,
        )
        self.widgets.slider(
            "Coverage", 0.1, 0.9, COVERAGE, on_change=self.cover
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def sun_at(self, degrees):
        e = math.radians(degrees)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (-0.8 * math.cos(e), -0.6 * math.cos(e), math.sin(e))
            ),
        )

    def cover(self, coverage):
        self.density.buffer = make_clouds(coverage)

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        cloudy = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Clouds", tokens.VISIBILITY, tokens.INVISIBLE
        )
        clear = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Clouds", tokens.VISIBILITY, tokens.INHERITED
        )
        with_clouds = mean_linear_rgb(cloudy)
        sky = mean_linear_rgb(clear)
        print(
            "Mean linear RGB with clouds "
            f"{with_clouds[0]:.3f} {with_clouds[1]:.3f} {with_clouds[2]:.3f}, "
            f"clear {sky[0]:.3f} {sky[1]:.3f} {sky[2]:.3f}"
        )
        assert sky[2] > sky[0], "the clear sky is blue"
        # The clouds change the view, white where the sky was blue.
        assert np.linalg.norm(with_clouds - sky) > 0.01, (
            "the clouds show"
        )
        assert with_clouds[0] / with_clouds[2] > sky[0] / sky[2], (
            "the clouds are whiter than the sky"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
