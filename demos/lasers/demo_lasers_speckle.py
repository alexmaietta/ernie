# SPDX-License-Identifier: BSD-3-Clause
"""Laser speckle on a matte wall through a machine-vision camera: the
grain grows as the lens stops down, and a surface that depolarizes the
light shows two patterns at once, lowering the contrast.

References:
- J. W. Goodman, "Speckle Phenomena in Optics", Roberts and Company,
  2007 (grain size lambda z / D, contrast 1 / sqrt(M)).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

WAVELENGTH = 532.0  # nm
PIXEL = 5e-3  # mm
FOCAL = 8.0  # mm
RESOLUTION = (256, 192)
DISTANCE = 1.0  # m, camera to wall
F_NUMBERS = (2.0, 8.0, 16.0)


def expected_contrast(f_number, patterns):
    """Goodman's 1 / sqrt(M): `patterns` independent patterns, each
    pixel summing 1 + p^2 / A_c of them for the grain's correlation
    area A_c = 1.08 (1.086 lambda N)^2 on the sensor."""
    cell = 1.086 * WAVELENGTH * 1e-6 * f_number  # mm
    return 1.0 / math.sqrt(
        patterns * (1.0 + PIXEL * PIXEL / (1.08 * cell * cell))
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        # The wall's left half keeps the laser's polarization (a
        # surface scattering once), its right half scrambles it.
        for name, retention, x in (
            ("Keeps", 1.0, -0.25),
            ("Scrambles", 0.0, 0.25),
        ):
            look = world.define(
                f"{name}Look",
                ernie.Material,
                albedo=(0.8,) * 3,
                polarization_retention=retention,
            )
            world.define(
                name,
                ernie.Cube,
                size=1.0,
                material_binding=look.path,
                translate=(x, DISTANCE, 0.0),
                scale=(0.5, 0.01, 0.5),
            )
        self.laser = world.define(
            "Laser",
            ernie.LaserLight,
            wavelength=WAVELENGTH,
            linewidth=0.0,
            power=0.01,
            waist_radius=(0.04, 0.04),
            polarization_degree=1.0,
            speckle=True,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.05, 0.0, 0.0),
                (0.0, DISTANCE, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=PIXEL * RESOLUTION[0],
            vertical_aperture=PIXEL * RESOLUTION[1],
            sensor_type="mono",
            spectral_range=(520.0, 545.0),
            spectral_bands=1,
            sensor_noise=False,
            clipping_range=(0.05, 1e4),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, 0.0, 0.0), (0.0, DISTANCE, 0.0), (0.0, 0.0, 1.0)
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
            ambient=0.0,
        )
        self.images = {}  # per f-number (None: speckle off)
        self.stop_down(1)
        self.widgets.combo(
            "f-number",
            [f"f/{n:g}" for n in F_NUMBERS],
            1,
            on_change=self.stop_down,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE

    def stop_down(self, index):
        """Sets the lens to F_NUMBERS[index], the exposure time keeping
        the image as bright."""
        n = F_NUMBERS[index]
        self.camera.set(exposure_f_stop=n, exposure_time=1e-3 * n * n)

    def radiance(self, f_number):
        """The wall's radiance at `f_number`, or without speckle for
        None. Each renders once."""
        if f_number in self.images:
            return self.images[f_number]
        self.laser.speckle = f_number is not None
        if f_number is not None:
            self.stop_down(F_NUMBERS.index(f_number))
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        bands = np.array(
            outputs["/Render/Camera"]
            .frames[0]
            .render_vars["/Render/Camera/SpectralRadiance"]
            .map()
        ).astype(np.float64)
        self.images[f_number] = bands.sum(axis=0)
        self.laser.speckle = True
        return self.images[f_number]

    def contrasts(self, f_number):
        """The speckle's contrast (deviation over mean) on each half of
        the wall, over the beam's core, against the light without it."""
        plain = self.radiance(None)
        ratio = self.radiance(f_number) / np.maximum(plain, 1e-30)
        core = plain > 0.5 * plain.max()
        half = RESOLUTION[0] // 2
        result = []
        for columns in (slice(8, half - 8), slice(half + 8, -8)):
            values = ratio[:, columns][core[:, columns]]
            result.append(float(values.std() / values.mean()))
        return result

    def render_all(self):
        """Each f-number's image as uint8 grey, the light without
        speckle shown at half white."""
        peak = self.radiance(None).max()
        images = []
        for n in F_NUMBERS:
            grey = 255.0 * np.clip(
                self.radiance(n) / (2.0 * peak), 0, 1
            )
            images.append(
                np.repeat(grey[..., None], 3, axis=2).astype(np.uint8)
            )
        return images

    def save_output(self, path):
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        for n in F_NUMBERS:
            keeps, scrambles = self.contrasts(n)
            one, two = (
                expected_contrast(n, 1),
                expected_contrast(n, 2),
            )
            print(
                f"f/{n:g}: contrast {keeps:.2f} (Goodman {one:.2f}) where "
                f"the wall keeps the polarization, {scrambles:.2f} "
                f"({two:.2f}) where it scrambles it"
            )
            assert abs(keeps - one) < 0.1, "one pattern"
            assert abs(scrambles - two) < 0.1, "two patterns"


if __name__ == "__main__":
    ernie.demo.main(Demo)
