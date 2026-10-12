# SPDX-License-Identifier: BSD-3-Clause
"""A polarization camera's view of the sky and of glare on glass: its
mosaic of 0, 45, 90 and 135 degree polarizers gives the degree and angle
of linear polarization per 2 x 2 cell.

References:
- Sony Semiconductor Solutions, "Polarization Image Sensor
  IMX250MZR/MYR", 2018 (the mosaic, and Stokes from its four angles).
- S. Chandrasekhar, "Radiative Transfer", Oxford, 1950 (the sky's
  Rayleigh pattern).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import heatmap, side_by_side, write_png

RESOLUTION = (320, 256)
FOCAL = 10.0  # mm
VERTICAL_FIELD = 80.0  # degrees
HEIGHT = 2.0  # m, the camera above the ground
SUN = math.atan(
    1.0 / 1.5
)  # elevation: its glint on glass at Brewster's angle
SKY = 0.75  # the sky's greatest degree of polarization


def stokes(image):
    """S0, S1 and S2 per 2 x 2 cell of the mosaic's image (rows, cols):
    90 and 45 degrees over 135 and 0."""
    i90 = image[0::2, 0::2]
    i45 = image[0::2, 1::2]
    i135 = image[1::2, 0::2]
    i0 = image[1::2, 1::2]
    return (i0 + i45 + i90 + i135) / 2.0, i0 - i90, i45 - i135


def cell_of(azimuth, elevation):
    """The mosaic cell (row, col) seeing along `azimuth` right of ahead
    and `elevation` up, in degrees."""
    a = math.radians(azimuth)
    e = math.radians(elevation)
    x = math.sin(a) * math.cos(e)
    y = math.sin(e)
    z = math.cos(a) * math.cos(e)
    focal = (
        0.5
        * RESOLUTION[1]
        / math.tan(math.radians(0.5 * VERTICAL_FIELD))
    )
    col = 0.5 * RESOLUTION[0] + focal * x / z
    row = 0.5 * RESOLUTION[1] - focal * y / z
    return int(row) // 2, int(col) // 2


def rayleigh(azimuth, elevation):
    """The sky's degree of polarization along a direction: Rayleigh's
    sin^2 g / (1 + cos^2 g) at its angle g from the sun, times SKY."""
    a = math.radians(azimuth)
    e = math.radians(elevation)
    cosine = math.cos(a) * math.cos(e) * math.cos(SUN) + math.sin(
        e
    ) * math.sin(SUN)
    return SKY * (1.0 - cosine**2) / (1.0 + cosine**2)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define("Air", ernie.Atmosphere, preset=tokens.EARTH)
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (0.0, math.cos(SUN), math.sin(SUN))
            ),
        )
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.3,) * 3,
            checker_color1=(0.25,) * 3,
        )
        # Black glass (index 1.5) where the camera sees the sun's glint.
        glass = world.define(
            "Glass",
            ernie.Material,
            brdf="preview_surface",
            albedo=(0.02,) * 3,
        )
        glass.prim.set("inputs:roughness", 0.2)
        glass.prim.set("inputs:ior", 1.5)
        world.define(
            "Sheet",
            ernie.Cube,
            size=1.0,
            material_binding=glass.path,
            translate=(0.0, HEIGHT / math.tan(SUN), 0.01),
            scale=(4.0, 4.0, 0.02),
        )
        world.define(
            "Block",
            ernie.Cube,
            size=1.0,
            display_color=(0.5,) * 3,
            translate=(2.5, 6.0, 0.5),
        )
        aperture = (
            2.0 * FOCAL * math.tan(math.radians(0.5 * VERTICAL_FIELD))
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=aperture
            * RESOLUTION[0]
            / RESOLUTION[1],
            vertical_aperture=aperture,
            spectral_range=(400.0, 700.0),
            spectral_bands=6,
            sensor_noise=False,
            polarizer_type="mosaic",
            clipping_range=(0.05, 1e5),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, 0.0, HEIGHT),
                (0.0, 1.0, HEIGHT),
                (0.0, 0.0, 1.0),
            ),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        radiance = scene.define(
            "/Render/Camera/SpectralRadiance", ernie.RenderVar
        )
        radiance.source_name = "SpectralRadiance"
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=[radiance.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
        )
        self.measured = None
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def measure(self):
        """Per mosaic cell: its intensity S0, degree and angle (degrees,
        0 to 180) of linear polarization, over the camera's bands."""
        if self.measured is None:
            outputs = self.renderer.step(
                self.scene, render_products=["/Render/Camera"]
            )
            bands = np.array(
                outputs["/Render/Camera"]
                .frames[0]
                .render_vars["/Render/Camera/SpectralRadiance"]
                .map()
            ).astype(np.float64)
            s0, s1, s2 = stokes(bands.sum(axis=0))
            degree = np.sqrt(s1 * s1 + s2 * s2) / np.maximum(
                s0, 1e-30
            )
            angle = np.degrees(0.5 * np.arctan2(s2, s1)) % 180.0
            self.measured = (s0, degree, angle)
        return self.measured

    def render_all(self):
        """Its intensity (a square root of S0 over the median's 4 times),
        degree (0 to 1) and angle (0 to 180 degrees)."""
        s0, degree, angle = self.measure()
        flat = np.sort(s0.ravel())
        scale = 4.0 * flat[flat.size // 2]
        grey = (
            255.0 * np.sqrt(np.clip(s0 / scale, 0.0, 1.0))
        ).astype(np.uint8)
        return [
            np.repeat(grey[..., None], 3, axis=2),
            heatmap(degree, 0.0, 1.0),
            heatmap(angle, 0.0, 180.0),
        ]

    def save_output(self, path):
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        _, degree, angle = self.measure()

        # The median over 5 x 5 cells about a direction: edges, whose
        # four pixels see different things, read as polarized.
        def at(azimuth, elevation):
            row, col = cell_of(azimuth, elevation)
            return float(
                np.median(
                    degree[row - 2 : row + 3, col - 2 : col + 3]
                )
            )

        glare = at(0.0, -math.degrees(SUN))
        near = at(0.0, math.degrees(SUN) - 10.0)
        away = at(40.0, 10.0)
        ground = at(-30.0, -10.0)
        print(
            f"degree of polarization: glare {glare:.2f}, sky 10 deg from the "
            f"sun {near:.3f}, 44 deg from it {away:.3f} "
            f"(Rayleigh {rayleigh(40.0, 10.0):.3f}), ground {ground:.3f}"
        )
        assert glare > 0.6, "glare at Brewster's angle is polarized"
        assert near < 0.05, "the sky by the sun is nearly unpolarized"
        assert abs(away - rayleigh(40.0, 10.0)) < 0.05, (
            "the sky away from the sun follows Rayleigh's pattern"
        )
        assert ground < 0.05, "the grey ground scatters unpolarized"


if __name__ == "__main__":
    ernie.demo.main(Demo)
