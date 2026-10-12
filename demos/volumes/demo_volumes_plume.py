"""A rocket engine's plume in sunlight: hot gas glowing as a black body
at the nozzle, cooling to soot that shadows and dims what is behind."""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear, write_png

# The plume's hottest, at the nozzle.
PEAK_KELVIN = 3000.0


def make_plume(time):
    """A rocket engine's plume `time` seconds in, from a nozzle 1.2 m
    across at the origin down -z for 12 m, in 0.1 m voxels: widening as
    it goes, hottest at the nozzle with shock diamonds, cooling to soot
    downstream; turbulence streams down it and the core flickers. Its
    density and temperature (over 0 to 1) as NanoVDB grids."""
    voxel = 0.1
    speed = 20.0  # m/s the turbulence streams down.
    flicker = 1.0 + 0.08 * np.sin(2.0 * np.pi * 12.0 * time)
    low = np.array([-2.4, -2.4, -12.0])
    shape = (48, 48, 120)
    x, y, z = np.meshgrid(
        *[
            low[a] + voxel * (np.arange(shape[a]) + 0.5)
            for a in range(3)
        ],
        indexing="ij",
    )
    s = np.maximum(-z, 0.0)  # Downstream.
    radius = 0.6 + 0.12 * s
    core = np.exp(-2.0 * (x * x + y * y) / (radius * radius))
    diamonds = 1.0 + 0.35 * np.cos(2.0 * np.pi * s / 1.4) * np.exp(
        -s / 3.0
    )
    fade = np.clip(s / 0.3, 0.0, 1.0) * np.clip(
        (12.0 - s) / 2.0, 0.0, 1.0
    )
    points = np.stack(
        [1.5 * x, 1.5 * y, 0.8 * (z + speed * time)], -1
    )
    turbulence = 0.5 + 0.5 * ernie.fbm_noise(
        points.reshape(-1, 3), octaves=3, seed=5
    ).reshape(x.shape)
    hot = (
        core
        * diamonds
        * np.exp(-s / 5.0)
        * fade
        * flicker
        * (0.85 + 0.3 * turbulence)
    )
    soot = (
        core
        * (0.8 + 0.4 * (1.0 - np.exp(-s / 4.0)))
        * np.exp(-s / 14.0)
        * fade
        * (0.4 + 1.2 * turbulence)
    )
    origin = tuple(low + 0.5 * voxel)

    def grid(values, name):
        # Nothing below 1e-3, so the grids stay sparse.
        values = np.where(values > 1e-3, values, 0.0).astype(
            np.float32
        )
        return ernie.dense_nanovdb_grid(values, voxel, origin, name)

    return grid(soot, "density"), grid(hot, "temperature")


def mean_in(image, left, right, top, bottom):
    """The image's mean linear RGB over columns [left, right) and rows
    [top, bottom)."""
    pixels = np.asarray(image)[top:bottom, left:right, :3]
    return srgb_to_linear(pixels.reshape(-1, 3).mean(axis=0) / 255.0)


class Demo(ernie.demo.Demo):
    num_frames = 12

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        # The rocket above its nozzle at the origin, 1.2 m across.
        rocket = world.define("Rocket", ernie.Xform)
        rocket.define(
            "Body",
            ernie.Cylinder,
            radius=0.6,
            height=6.0,
            translate=(0.0, 0.0, 3.4),
            material_binding="/World/Looks/Paint",
        )
        rocket.define(
            "Nose",
            ernie.Cone,
            radius=0.6,
            height=1.6,
            translate=(0.0, 0.0, 7.2),
            material_binding="/World/Looks/Paint",
        )
        rocket.define(
            "Nozzle",
            ernie.Cone,
            radius=0.4,
            height=0.8,
            translate=(0.0, 0.0, 0.4),
            material_binding="/World/Looks/Metal",
        )
        self.scene.define(
            "/World/Looks/Paint",
            ernie.Material,
            albedo=(0.7, 0.7, 0.7),
        )
        self.scene.define(
            "/World/Looks/Metal",
            ernie.Material,
            albedo=(0.25, 0.25, 0.25),
        )

        # The plume's density and temperature, built here as NanoVDB
        # grids each step.
        plume = world.define(
            "Plume",
            ernie.Volume,
            material_binding="/World/Looks/Exhaust",
        )
        self.density = plume.define("Density", ernie.NanoVDBField)
        self.temperature = plume.define(
            "Temperature", ernie.NanoVDBField
        )
        plume.set_field("density", "/World/Plume/Density")
        plume.set_field("temperature", "/World/Plume/Temperature")
        self.time = 0.0
        self.burn()
        # Soot: mostly absorbing, so it glows as a black body where hot
        # (Kirchhoff) and scatters a little sunlight.
        self.exhaust = self.scene.define(
            "/World/Looks/Exhaust",
            ernie.Material,
            density=1.5,
            volume_albedo=(0.3, 0.3, 0.3),
            anisotropy=0.4,
            temperature_scale=PEAK_KELVIN,
        )

        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.5, -1.0, 0.6)),
        )
        # Daylight exposure, from the side 40 m off.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=35.0,
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -40.0, -1.0), (0.0, 0.0, -1.5), (0.0, 0.0, 1.0)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(320, 480),
            ambient=0.0,
        )

        self.widgets.slider(
            "Peak temperature (K)",
            800.0,
            3500.0,
            PEAK_KELVIN,
            on_change=self.set_peak_temperature,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def step(self):
        self.time += self.frame_dt
        self.burn()

    def burn(self):
        """The plume's grids at the time."""
        density, temperature = make_plume(self.time)
        self.density.buffer = density
        self.temperature.buffer = temperature

    def set_peak_temperature(self, kelvin):
        self.exhaust.temperature_scale = kelvin

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        image = np.array(self.renderer.render(self.scene))
        rows, cols = image.shape[:2]
        # The brightest spot down the plume's axis (a shock diamond),
        # and the empty sky beside it.
        core = max(
            (
                mean_in(
                    image, cols // 2 - 4, cols // 2 + 4, row, row + 8
                )
                for row in range(rows // 4, rows - 7, 4)
            ),
            key=lambda spot: spot[0],
        )
        sky = mean_in(image, 0, 16, rows - 32, rows)
        print(
            f"Plume core {core[0]:.4f} {core[1]:.4f} {core[2]:.4f}, "
            f"sky {sky[0]:.4f} {sky[1]:.4f} {sky[2]:.4f}"
        )
        assert sky.sum() < 0.01, "space is black"
        assert core[0] > 0.2, "the hot core glows"
        assert core[0] > core[2], (
            "the glow is warm, as a black body's"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
