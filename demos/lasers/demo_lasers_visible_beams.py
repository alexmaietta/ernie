# SPDX-License-Identifier: BSD-3-Clause
"""Laser beams made visible by drifting dust at night: a green grid of
beamlets, a red line and a blue spot cross the haze to a wall, the dust
scattering a little of each beam towards the camera, and a rock in the
way leaves a dark channel behind it.

References:
- C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light
  by Small Particles", Wiley, 1983 (a beam seen side-on by the light
  its path scatters, forward-peaked for dust).
"""

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 200)
DENSITY = 0.15  # 1/m, the dust's extinction at its densest
RIG = (0.0, 0.0, 1.2)  # m, the lasers' aperture
# The wind's direction, rising a little.
GUST = np.array([1.0, 0.0, 0.15])


def make_dust(drift=(0.0, 0.0, 0.0)):
    """Billowing dust over 12 m x 12 m, 3 m deep, in 0.2 m voxels,
    carried `drift` m by the wind."""
    voxel = 0.2
    low = np.array([-6.0, -2.0, 0.0])
    shape = (60, 60, 15)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    points = 0.4 * (np.stack([x, y, z], -1) - drift) + 3.0
    billows = ernie.fbm_noise(
        points.reshape(-1, 3), octaves=4, seed=5
    )
    value = np.clip(0.55 + 0.6 * billows.reshape(shape), 0.0, 1.0)
    value *= np.exp(-z / 2.0)
    density = np.where(value > 1e-3, value, 0.0).astype(np.float32)
    return ernie.dense_nanovdb_grid(
        density, voxel, tuple(low + 0.5 * voxel), "density"
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        looks.define("Ground", ernie.Material, albedo=(0.15,) * 3)
        looks.define("Wall", ernie.Material, albedo=(0.6,) * 3)
        looks.define("Rock", ernie.Material, albedo=(0.25,) * 3)
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Ground",
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 9.0, 1.5),
            scale=(10.0, 0.2, 3.0),
            material_binding="/World/Looks/Wall",
        )
        # A rock in the green grid's left beams.
        world.define(
            "Rock",
            ernie.Sphere,
            radius=0.35,
            translate=(-0.45, 4.0, 0.95),
            material_binding="/World/Looks/Rock",
        )
        target = (0.0, 9.0, 1.2)
        aim = ernie.look_at_transform(RIG, target, (0.0, 0.0, 1.0))
        world.define(
            "Grid",
            ernie.LaserLight,
            wavelength=532.0,
            power=0.2,
            waist_radius=(0.002, 0.002),
            pattern="grid",
            grid_columns=5,
            grid_rows=3,
            grid_pitch=(4.0, 3.0),
        ).set(tokens.TRANSFORM, aim)
        world.define(
            "Line",
            ernie.LaserLight,
            wavelength=638.0,
            power=0.3,
            waist_radius=(0.002, 0.002),
            pattern="line",
            fan_angle=30.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                RIG, (0.0, 9.0, 0.3), (0.0, 0.0, 1.0)
            ),
        )
        world.define(
            "Spot",
            ernie.LaserLight,
            wavelength=450.0,
            power=0.5,
            waist_radius=(0.004, 0.004),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                RIG, (2.5, 9.0, 2.4), (0.0, 0.0, 1.0)
            ),
        )
        # Dust scattering most of what it stops, forwards.
        dust = world.define(
            "Dust", ernie.Volume, material_binding="/World/Looks/Dust"
        )
        self.dust = dust.define(
            "Density", ernie.NanoVDBField, buffer=make_dust()
        )
        dust.set_field("density", "/World/Dust/Density")
        self.dust_look = looks.define(
            "Dust",
            ernie.Material,
            density=DENSITY,
            volume_albedo=(0.9, 0.9, 0.9),
            anisotropy=0.6,
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=18.0,
            sensor_noise=False,
            exposure_time=1.0 / 30.0,
            exposure_f_stop=2.0,
            exposure_iso=1600.0,
            clipping_range=(0.05, 1e4),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (4.5, 0.0, 4.5), (0.0, 4.2, 1.0), (0.0, 0.0, 1.0)
            ),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        variables = []
        for name in ("SpectralRadiance", "PrimId"):
            var = scene.define(
                f"/Render/Camera/{name}", ernie.RenderVar
            )
            var.source_name = name
            variables.append(var.path)
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=variables,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.widgets.slider(
            "Dust density (1/m)",
            0.0,
            1.0,
            DENSITY,
            on_change=lambda d: self.dust_look.set(density=d),
        )
        # Wind carrying the dust through the beams, which then flicker.
        self.wind = 0.0
        self.drift = np.zeros(3)
        self.widgets.slider(
            "Wind (m/s)",
            0.0,
            3.0,
            self.wind,
            on_change=lambda w: setattr(self, "wind", w),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def step(self):
        if self.wind > 0.0:
            self.drift += self.wind * self.frame_dt * GUST
            self.dust.buffer = make_dust(self.drift)

    def capture(self, lasers=True, density=DENSITY):
        """The radiance over the bands and the prim ids, with the
        lasers on or off and the dust at `density`."""
        for name in ("Grid", "Line", "Spot"):
            self.scene.set(
                f"/World/{name}",
                tokens.VISIBILITY,
                "inherited" if lasers else "invisible",
            )
        self.dust_look.set(density=density)
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        frame = outputs["/Render/Camera"].frames[0].render_vars
        radiance = np.array(
            frame["/Render/Camera/SpectralRadiance"].map()
        ).astype(np.float64)
        ids = np.array(frame["/Render/Camera/PrimId"].map())
        self.dust_look.set(density=DENSITY)
        return radiance.sum(axis=0), ids

    def save_output(self, path):
        write_png(path, self.render_image())

    def test_final(self):
        lit, _ = self.capture()
        clear, _ = self.capture(density=0.0)
        dark, _ = self.capture(lasers=False)
        # What the dust adds to the frame in clear air: the beams.
        beams = lit - clear
        threshold = 0.05 * beams.max()
        visible = (beams > threshold).mean()
        print(
            f"the beams light {100 * visible:.1f}% of the pixels in the "
            f"dust; with the lasers off, the brightest pixel is "
            f"{dark.max():.3g}"
        )
        assert visible > 0.003, "the beams show in the dust"
        assert dark.max() == 0.0, "nothing else lights it"


if __name__ == "__main__":
    ernie.demo.main(Demo)
