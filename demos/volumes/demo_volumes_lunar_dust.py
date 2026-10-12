"""A lander hovering over the Moon, its exhaust blowing a sheet of dust
outward: a low sun lights the dust, which shades the ground under it."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear, write_png

DENSITY = 0.3


def make_dust():
    """The dust's density over a 48 m square, 6 m deep, in 0.25 m
    voxels: a sheet streaming out from under the engine, streaked along
    its rays and billowing, thinning with height and distance."""
    voxel = 0.25
    low = np.array([-24.0, -24.0, 0.0])
    shape = (192, 192, 24)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    # Out from 1.5 m, fading over 10 m; thicker further out.
    radial = np.clip((r - 1.5) / 2.0, 0.0, 1.0) * np.exp(-r / 10.0)
    vertical = np.exp(-z / (0.2 + 0.05 * r))
    # Streaks along the rays, and billows.
    rays = np.stack(
        [6.0 * np.cos(theta), 6.0 * np.sin(theta), 0.15 * r], -1
    )
    streaks = 0.5 + 0.5 * ernie.fbm_noise(
        rays.reshape(-1, 3), octaves=3, seed=11
    ).reshape(shape)
    points = 0.6 * np.stack([x, y, z], -1) + 17.0
    billows = 0.5 + 0.5 * ernie.fbm_noise(
        points.reshape(-1, 3), octaves=4, seed=11
    ).reshape(shape)
    value = (
        radial
        * vertical
        * np.clip(2.5 * streaks - 1.0, 0.0, 1.0)
        * (0.5 + billows)
    )
    density = np.where(value > 1e-3, value, 0.0).astype(np.float32)
    return ernie.dense_nanovdb_grid(
        density, voxel, tuple(low + 0.5 * voxel), "density"
    )


def mean_rows(image, top, bottom):
    """The image's mean linear RGB over rows [top, bottom)."""
    pixels = np.asarray(image)[top:bottom, :, :3]
    return srgb_to_linear(pixels.reshape(-1, 3).mean(axis=0) / 255.0)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Regolith",
        )
        self.scene.define(
            "/World/Looks/Regolith",
            ernie.Material,
            albedo=(0.12, 0.12, 0.12),
        )
        # The lander 4 m up: its body, legs and engine bell.
        lander = world.define("Lander", ernie.Xform)
        lander.define(
            "Body",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 0.0, 5.5),
            scale=(3.0, 3.0, 2.0),
            material_binding="/World/Looks/Foil",
        )
        for leg in range(4):
            angle = math.pi * (0.25 + 0.5 * leg)
            lander.define(
                f"Leg{leg}",
                ernie.Cylinder,
                radius=0.08,
                height=2.0,
                translate=(
                    2.0 * math.cos(angle),
                    2.0 * math.sin(angle),
                    4.0,
                ),
                material_binding="/World/Looks/Metal",
            )
        lander.define(
            "Bell",
            ernie.Cone,
            radius=0.5,
            height=0.8,
            translate=(0.0, 0.0, 4.1),
            material_binding="/World/Looks/Metal",
        )
        self.scene.define(
            "/World/Looks/Foil",
            ernie.Material,
            albedo=(0.35, 0.28, 0.12),
        )
        self.scene.define(
            "/World/Looks/Metal",
            ernie.Material,
            albedo=(0.3, 0.3, 0.3),
        )

        # The dust, built here: grains that scatter about half the
        # light, mostly back towards the sun, as regolith does (Hapke's
        # double Henyey-Greenstein lobes).
        dust = world.define(
            "Dust", ernie.Volume, material_binding="/World/Looks/Dust"
        )
        dust.define("Density", ernie.NanoVDBField, buffer=make_dust())
        dust.set_field("density", "/World/Dust/Density")
        self.dust_look = self.scene.define(
            "/World/Looks/Dust",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.55, 0.55, 0.55),
            phase_function=tokens.DOUBLE_HENYEY_GREENSTEIN,
        )
        self.scene.set("/World/Looks/Dust", tokens.PHASE_B, 0.3)
        self.scene.set("/World/Looks/Dust", tokens.PHASE_C, 0.6)

        # A low sun from the left.
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-1.0, -0.7, 0.3)),
        )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 250.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (8.0, -22.0, 2.5), (0.0, 0.0, 2.5), (0.0, 0.0, 1.0)
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
            "Dust density (1/m)",
            0.0,
            2.0,
            DENSITY,
            on_change=self.set_density,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def set_density(self, density):
        self.dust_look.density = density

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        dusty = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Dust", tokens.VISIBILITY, tokens.INVISIBLE
        )
        clear = np.array(self.renderer.render(self.scene))
        self.scene.set(
            "/World/Dust", tokens.VISIBILITY, tokens.INHERITED
        )
        rows = dusty.shape[0]
        # The sky above, and the ground beyond the lander, seen through
        # the dust or without it.
        sky = mean_rows(dusty, 0, rows // 8)
        band = (rows // 2 + rows // 16, rows // 2 + rows // 8)
        with_dust = mean_rows(dusty, *band)
        without = mean_rows(clear, *band)
        print(
            f"Sky {sky.sum():.4f}, ground band with dust "
            f"{with_dust[0]:.4f} {with_dust[1]:.4f} {with_dust[2]:.4f}, "
            f"without {without[0]:.4f} {without[1]:.4f} {without[2]:.4f}"
        )
        assert sky.sum() < 0.01, "the Moon's sky is black"
        assert abs(with_dust.sum() - without.sum()) > 0.01, (
            "the dust shows over the ground"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
