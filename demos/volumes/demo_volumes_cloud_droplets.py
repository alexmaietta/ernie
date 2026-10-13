# SPDX-License-Identifier: BSD-3-Clause
"""A cumulus lit from behind by a low sun: its water droplets scatter by
Mie theory (the silver lining about its edges, a fogbow and glory
opposite the sun), their light scattered many times found by collectors
on its lit surface (Bouthors et al. 2008) rather than octaves."""

import math
import time

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    define_planet_surface,
    mean_linear_rgb,
    write_png,
)

SUN_AZIMUTH = 20.0
SUN_ELEVATION = 8.0
DROPLET_RADIUS = 7.0
# Blobs of the cloud: centre (m) and radius (m).
BLOBS = (
    ((0.0, 0.0, 1900.0), 1100.0),
    ((-900.0, 300.0, 1500.0), 800.0),
    ((850.0, -200.0, 1450.0), 750.0),
    ((300.0, 700.0, 2500.0), 650.0),
    ((-300.0, -500.0, 1300.0), 700.0),
)


def make_cumulus():
    """A cumulus congestus of a few rounded blobs on a flat base at 900 m,
    in 40 m voxels, its density over 0 to 1: billowing noise on the blobs,
    finer noise wearing their edges away."""
    voxel = 40.0
    low = np.array([-2400.0, -2400.0, 800.0])
    shape = (120, 120, 80)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    points = np.stack([x, y, z], -1).reshape(-1, 3)
    inside = np.zeros(len(points))
    for centre, radius in BLOBS:
        distance = np.linalg.norm(points - np.array(centre), axis=1)
        inside = np.maximum(inside, 1.0 - distance / radius)
    inside *= np.clip((points[:, 2] - 900.0) / 150.0, 0.0, 1.0)
    billows = 0.5 + 0.5 * ernie.fbm_noise(
        points / 450.0, octaves=5, seed=7
    )
    base = np.clip((inside - 0.55 * (1.0 - billows)) * 3.0, 0.0, 1.0)
    detail = 0.5 + 0.5 * ernie.fbm_noise(
        points / 120.0 + 17.0, octaves=3, seed=7
    )
    value = base - (1.0 - base) * detail * 0.8
    density = np.where(value > 1e-3, value, 0.0).astype(np.float32)
    return ernie.dense_nanovdb_grid(
        density.reshape(shape),
        voxel,
        tuple(low + 0.5 * voxel),
        "density",
    )


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
        self.azimuth = SUN_AZIMUTH
        self.elevation = SUN_ELEVATION
        self.place_sun()

        cloud = world.define(
            "Cloud",
            ernie.Volume,
            material_binding="/World/Looks/Cloud",
        )
        cloud.define(
            "Density", ernie.NanoVDBField
        ).buffer = make_cumulus()
        cloud.set_field("density", "/World/Cloud/Density")
        # Water droplets of 7 um, their surface where the density
        # reaches 0.1; octaves for the other setting.
        self.look = self.scene.define(
            "/World/Looks/Cloud",
            ernie.Material,
            density=0.05,
            volume_albedo=(1.0, 1.0, 1.0),
            phase_function=tokens.CLOUD_DROPLETS,
            droplet_radius=DROPLET_RADIUS,
            surface_density=0.1,
            anisotropy=0.6,
            scattering_octaves=6,
            octave_attenuation=0.5,
            octave_extinction=0.5,
            octave_phase=0.5,
        )

        # From 6.5 km off at the cloud's middle, the sun behind it.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=11.0,
            exposure_time=1.0 / 250.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -6500.0, 1300.0),
                (0.0, 0.0, 1700.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.settings = self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(640, 400),
            ambient=0.0,
            volume_multiple_scattering="collectors",
        )

        self.widgets.slider(
            "Sun azimuth (0 behind the cloud)",
            0.0,
            360.0,
            SUN_AZIMUTH,
            on_change=self.sun_azimuth,
        )
        self.widgets.slider(
            "Sun elevation",
            1.0,
            60.0,
            SUN_ELEVATION,
            on_change=self.sun_elevation,
        )
        self.widgets.combo(
            "Multiple scattering",
            ["Collectors", "Octaves"],
            0,
            on_change=self.scatter_by,
        )
        self.widgets.combo(
            "Phase function",
            ["Cloud droplets (Mie)", "Henyey-Greenstein"],
            0,
            on_change=self.phase,
        )
        self.widgets.slider(
            "Droplet radius (um)",
            2.0,
            20.0,
            DROPLET_RADIUS,
            on_change=lambda radius: setattr(
                self.look, "droplet_radius", radius
            ),
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def place_sun(self):
        a = math.radians(self.azimuth)
        e = math.radians(self.elevation)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (
                    math.sin(a) * math.cos(e),
                    math.cos(a) * math.cos(e),
                    math.sin(e),
                )
            ),
        )

    def sun_azimuth(self, degrees):
        self.azimuth = degrees
        self.place_sun()

    def sun_elevation(self, degrees):
        self.elevation = degrees
        self.place_sun()

    def scatter_by(self, index):
        self.settings.volume_multiple_scattering = (
            "collectors" if index == 0 else "octaves"
        )

    def phase(self, index):
        self.look.phase_function = (
            tokens.CLOUD_DROPLETS
            if index == 0
            else tokens.HENYEY_GREENSTEIN
        )

    def save_output(self, path):
        write_png(path, self.settled())

    def settled(self):
        """The view once the collectors' transport table is ready, read
        or simulated off the frame's thread (octaves meanwhile)."""
        octaves = self.with_scattering("octaves").astype(np.float32)
        image = self.with_scattering("collectors")
        start = time.monotonic()
        # Beyond the frames' noise, which differs frame to frame.
        while (
            np.abs(image - octaves).mean() < 1.0
            and time.monotonic() - start < 120
        ):
            time.sleep(0.2)
            image = self.with_scattering("collectors")
        return image

    def with_scattering(self, mode):
        self.settings.volume_multiple_scattering = mode
        return np.array(self.renderer.render(self.scene))

    def test_final(self):
        collected = mean_linear_rgb(self.settled())
        octaves = mean_linear_rgb(self.with_scattering("octaves"))
        self.settings.volume_multiple_scattering = "collectors"
        self.look.phase_function = tokens.HENYEY_GREENSTEIN
        henyey = mean_linear_rgb(
            np.array(self.renderer.render(self.scene))
        )
        self.look.phase_function = tokens.CLOUD_DROPLETS
        self.scene.set(
            ernie.Path("/World/Cloud"),
            tokens.VISIBILITY,
            tokens.INVISIBLE,
        )
        clear = mean_linear_rgb(
            np.array(self.renderer.render(self.scene))
        )
        self.scene.set(
            ernie.Path("/World/Cloud"),
            tokens.VISIBILITY,
            tokens.INHERITED,
        )
        print(
            f"Mean linear RGB by collectors {collected[0]:.3f} "
            f"{collected[1]:.3f} {collected[2]:.3f}, octaves "
            f"{octaves[0]:.3f} {octaves[1]:.3f} {octaves[2]:.3f}, "
            f"Henyey-Greenstein {henyey[0]:.3f} {henyey[1]:.3f} "
            f"{henyey[2]:.3f}, clear {clear[0]:.3f} {clear[1]:.3f} "
            f"{clear[2]:.3f}"
        )
        assert np.linalg.norm(collected - clear) > 0.01, (
            "the cloud shows"
        )
        assert np.linalg.norm(collected - octaves) > 1e-4, (
            "the collectors light the cloud otherwise than octaves"
        )
        assert np.linalg.norm(collected - henyey) > 1e-4, (
            "the droplets scatter otherwise than Henyey-Greenstein"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
