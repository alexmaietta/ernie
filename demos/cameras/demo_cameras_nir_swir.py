# SPDX-License-Identifier: BSD-3-Clause
"""Measured materials seen by visible, near-infrared and
shortwave-infrared cameras: grass, snow and asphalt each change with
the band.

References:
- S. K. Meerdink et al., "The ECOSTRESS spectral library version 1.0",
  Remote Sensing of Environment 230:111196, 2019.
- G. P. Asner, "Biophysical and biochemical sources of variability in
  canopy reflectance", Remote Sensing of Environment 64(3), 1998 (the
  red edge).
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

MATERIALS = [
    "grass",
    "dry_grass",
    "loam",
    "sand",
    "concrete",
    "asphalt",
    "brick",
    "white_coating",
    "aluminium",
    "fine_snow",
]

# The cameras: name, product, spectral range, bands, and for a mono one
# its quantum efficiency as (nm, value) pairs.
BANDS = [
    ("visible RGB", "/Render/Rgb", (380.0, 720.0), 17, None),
    # A visible-blocking filter on silicon.
    (
        "NIR 760-1000 nm",
        "/Render/Nir",
        (760.0, 1000.0),
        12,
        [760, 0, 779, 0, 781, 0.35, 900, 0.25, 1000, 0.05],
    ),
    # InGaAs behind a 1.57 to 1.65 um filter (Landsat 8's band 6).
    (
        "SWIR 1.57-1.65 um",
        "/Render/Swir",
        (1550.0, 1670.0),
        6,
        [1550, 0, 1569, 0, 1571, 0.8, 1649, 0.8, 1651, 0, 1670, 0],
    ),
]

RESOLUTION = (480, 120)
HEIGHT = 9.0  # the cameras above the patches
FOCAL_LENGTH = 24.0  # mm
APERTURE = 20.955  # mm, USD's horizontal default
PITCH = 0.6  # between patches' centres, m
PATCH = 0.5  # a patch's side, m


def patch_x(i):
    return PITCH * (i - 0.5 * (len(MATERIALS) - 1))


def brightness(image, x):
    """The mean value (0 to 1, of the channels' mean) over the middle
    half of the patch centred at `x`."""
    rows, cols = image.shape[:2]
    pixels = FOCAL_LENGTH / APERTURE * cols / HEIGHT
    col = int(0.5 * cols + x * pixels)
    half = int(0.25 * PATCH * pixels)
    window = image[
        rows // 2 - half : rows // 2 + half + 1,
        col - half : col + half + 1,
        :3,
    ]
    return float(window.mean()) / 255.0


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        # Black ground, so only the patches show.
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.02,) * 3,
            checker_color1=(0.02,) * 3,
        )
        looks = world.define("Looks", ernie.Xform)
        patches = world.define("Patches", ernie.Xform)
        for i, material in enumerate(MATERIALS):
            look = looks.define(
                material, ernie.Material, reflectance_library=material
            )
            patches.define(
                material,
                ernie.Cube,
                size=1.0,
                material_binding=look.path,
                translate=(patch_x(i), 0.0, 0.01),
                scale=(PATCH, PATCH, 0.02),
            )
        # The Sun, its measured spectrum, 50 degrees up.
        direction = np.array([0.3, -0.5, 1.0])
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1000.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                tuple(direction / np.linalg.norm(direction))
            ),
        )
        # The cameras straight down, exposed for sunlight by the sunny
        # 16 rule: f/16, 1/100 s at ISO 100; the SWIR camera's narrow
        # band four times as long.
        self.cameras = []
        for c, (_, product, spectral, bands, qe) in enumerate(BANDS):
            camera = world.define(
                f"Camera{c}",
                ernie.Camera,
                focal_length=FOCAL_LENGTH,
                horizontal_aperture=APERTURE,
                vertical_aperture=APERTURE
                * RESOLUTION[1]
                / RESOLUTION[0],
                sensor_noise=False,
                spectral_range=spectral,
                spectral_bands=bands,
                exposure_f_stop=16.0,
                exposure_time=0.04 if c == 2 else 0.01,
                exposure_iso=100.0,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (0.0, 0.0, HEIGHT),
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
            )
            if qe is not None:
                camera.sensor_type = "mono"
                camera.sensor_qe_mono = [float(v) for v in qe]
            self.cameras.append(camera)
            ldr = scene.define(f"{product}/LdrColor", ernie.RenderVar)
            ldr.source_name = "LdrColor"
            scene.define(
                product,
                ernie.RenderProduct,
                camera=camera,
                resolution=RESOLUTION,
                ordered_vars=[ldr.path],
            )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.cameras[0],
            resolution=RESOLUTION,
            ambient=0.0,
        )

        def look_through(index):
            if self.viewer is not None:
                self.viewer.active_camera = self.cameras[index].path

        self.widgets.combo(
            "Camera",
            [name for name, *_ in BANDS],
            0,
            on_change=look_through,
        )
        if viewer is not None:
            viewer.active_camera = self.cameras[0].path

    def render_all(self):
        """Every camera's LdrColor, uint8 (rows, cols, 4)."""
        products = [product for _, product, *_ in BANDS]
        outputs = self.renderer.step(
            self.scene, render_products=products
        )
        return [
            np.array(
                outputs[product]
                .frames[0]
                .render_vars[f"{product}/LdrColor"]
                .map()
            )
            for product in products
        ]

    def save_output(self, path):
        """The three cameras' images side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        images = self.render_all()
        mean = []
        for (band, *_), image in zip(BANDS, images):
            values = [
                brightness(image, patch_x(i))
                for i in range(len(MATERIALS))
            ]
            mean.append(values)
            print(
                f"{band}: "
                + " ".join(
                    f"{m} {v:.2f}" for m, v in zip(MATERIALS, values)
                )
            )
        grass = MATERIALS.index("grass")
        concrete = MATERIALS.index("concrete")
        snow = MATERIALS.index("fine_snow")
        asphalt = MATERIALS.index("asphalt")
        assert mean[0][grass] < mean[0][concrete], (
            "grass should look darker than concrete in the visible"
        )
        assert mean[1][grass] > mean[1][concrete], (
            "grass should outshine concrete in the near infrared"
        )
        assert mean[0][snow] > mean[0][concrete], (
            "snow should be the brighter in the visible"
        )
        assert mean[2][snow] < mean[2][concrete], (
            "snow should be darker than concrete in the SWIR"
        )
        for (band, *_), values in zip(BANDS, mean):
            assert values[asphalt] < values[concrete], (
                f"asphalt should be darker than concrete in the {band}"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
