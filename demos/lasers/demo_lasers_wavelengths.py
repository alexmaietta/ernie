# SPDX-License-Identifier: BSD-3-Clause
"""Six lasers from violet to near infrared seen by four cameras, each
seeing only the lines its bands reach."""

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

# Typical diode and DPSS lasers: name, nm, mW.
#
# References:
# - A. E. Siegman, Lasers, University Science Books, 1986.
# - Thorlabs, laser diode selection guide (405 to 940 nm diodes).
LASERS = [
    ("405 nm violet", 405.0, 5.0),
    ("450 nm blue", 450.0, 10.0),
    ("532 nm green", 532.0, 1.0),
    ("650 nm red", 650.0, 5.0),
    ("850 nm NIR", 850.0, 10.0),
    ("940 nm NIR", 940.0, 20.0),
]

# The cameras: name, product, spectral range, bands, and for a mono
# one its quantum efficiency as (nm, value) pairs.
BANDS = [
    ("visible RGB", "/Render/Rgb", (380.0, 720.0), 17, None),
    # A 10 nm interference filter at 532 nm.
    (
        "532 nm narrowband",
        "/Render/Green",
        (517.0, 547.0),
        6,
        [517, 0, 526, 0, 527, 0.6, 537, 0.6, 538, 0, 547, 0],
    ),
    # A visible-blocking filter on silicon: 780 to 1000 nm.
    (
        "NIR 780-1000 nm",
        "/Render/Nir",
        (760.0, 1000.0),
        12,
        [760, 0, 779, 0, 781, 0.35, 900, 0.25, 1000, 0.05],
    ),
    # Bare silicon, 400 to 1000 nm.
    (
        "wideband mono",
        "/Render/Mono",
        (380.0, 1000.0),
        31,
        [380, 0.1, 500, 0.5, 700, 0.6, 850, 0.4, 1000, 0.05],
    ),
]

# Per camera, which lasers' spots stand out of the wall around them.
EXPECTED = [
    [True, True, True, True, False, False],
    [False, False, True, False, False, False],
    [False, False, False, False, True, True],
    [True, True, True, True, True, True],
]

RESOLUTION = (400, 240)
DISTANCE = 2.0  # camera to wall, m
HEIGHT = 1.2  # the cameras' and spots' height
FOCAL_LENGTH = 50.0  # mm
APERTURE = 20.955  # mm, USD's horizontal default
SPACING = 0.15  # between spots on the wall


def spot_x(i):
    """Where on the wall laser `i`'s spot lies along x."""
    return SPACING * (i - 0.5 * (len(LASERS) - 1))


def brightness(image, x, z):
    """The brightest value (0 to 1, the brightest channel) within 3
    pixels of the wall's point `x` along it and `z` above the spots."""
    rows, cols = image.shape[:2]
    pixels = FOCAL_LENGTH / APERTURE * cols / DISTANCE
    col = int(0.5 * cols + x * pixels)
    row = int(0.5 * rows - z * pixels)
    window = image[
        max(row - 3, 0) : row + 4, max(col - 3, 0) : col + 4, :3
    ]
    return float(window.max()) / 255.0


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.3,) * 3,
            checker_color1=(0.25,) * 3,
        )
        # A matte white wall at y = 0.
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            display_color=(0.8,) * 3,
            translate=(0.0, 0.05, 2.0),
            scale=(6.0, 0.1, 4.0),
        )
        # A dim incandescent ceiling lamp, whose blackbody reaches the
        # near infrared too.
        world.define(
            "Lamp",
            ernie.SphereLight,
            radius=0.1,
            photometric_power=200.0,
            physical_illuminant=tokens.BLACKBODY,
            color_temperature=2800.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.5, 3.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            ),
        )
        # The lasers on a bar 1.5 m from the wall, beams level and
        # square to it: collimated, 4 mm in radius, M^2 1.2.
        lasers = world.define("Lasers", ernie.Xform)
        for i, (_, nm, milliwatts) in enumerate(LASERS):
            origin = (spot_x(i), -1.5, HEIGHT)
            lasers.define(
                f"Laser{i}",
                ernie.LaserLight,
                wavelength=nm,
                power=milliwatts * 1e-3,
                waist_radius=(0.004, 0.004),
                beam_quality=(1.2, 1.2),
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    origin,
                    (origin[0], origin[1] + 1.0, origin[2]),
                    (0.0, 0.0, 1.0),
                ),
            )
        # The four cameras side by side, square to the wall, exposed
        # automatically and without noise.
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
                exposure_mode=tokens.AUTO,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (0.0, -DISTANCE, HEIGHT), (0.0, 0.0, HEIGHT)
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

        # The demo's option: which camera the viewport looks through.
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
        """The four cameras' images side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        for (band, *_), image, expected in zip(
            BANDS, self.render_all(), EXPECTED
        ):
            around = brightness(image, 0.0, 0.12)
            for (laser, *_), i, wanted in zip(
                LASERS, range(len(LASERS)), expected
            ):
                spot = brightness(image, spot_x(i), 0.0)
                print(
                    f"{band} sees {laser}: {spot:.3f} (wall {around:.3f})"
                )
                seen = spot > around + 0.1
                assert seen == wanted, (
                    f"{band} should {'' if wanted else 'not '}see the "
                    f"{laser} laser"
                )


if __name__ == "__main__":
    ernie.demo.main(Demo)
