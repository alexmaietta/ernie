# SPDX-License-Identifier: BSD-3-Clause
"""NASA Ames' structured-light virtual bumper: 11 dots at 660 nm and a
filtered mono camera on a rover, at night, in Earth daylight and in
lunar daylight.

References:
- A. Nefian, U. Wong, M. Dille et al., "Structured Light-based Hazard
  Detection For Planetary Surface Navigation", NASA NTRS 20170009200,
  2017.
- C. M. Pieters, L. A. Taylor, S. K. Noble et al., "Spectral
  characterization of lunar mare soils", LPSC XXXI, 2000 (the soils'
  spectra, read off the paper's Fig. 1).
- A. S. McEwen, "Photometric functions for photoclinometry and other
  applications", Icarus 92:298-311, 1991 (Lunar-Lambert).
- Osela Inc., "SL Streamline Laser" datasheet, 2020 (660 nm at 50 or
  100 mW).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

CASES = ["night", "Earth day", "lunar day"]
DOTS = 11
GAMMA = 1.52  # degrees between beams: 20 pixels apart, as theirs
BASELINE = 0.26  # m, camera to projector
PITCH = 28.0  # degrees down
HEIGHT = 0.6  # m, the camera and projector above the ground
FOCAL = 3.5  # mm, at f/5.6
PIXEL = 4.65e-3  # mm, the ICX267's
RESOLUTION = (1392, 1040)
DISPARITY = 15  # pixels a dot moves before it is an obstacle
# Their binary thresholds (Fig. 11), of 255.
THRESHOLD = {"night": 0.2, "Earth day": 0.675, "lunar day": 0.675}
# Shutters (s) putting the dots at 90% of white on clear ground, as
# `expose` finds them; the viewer starts each case with its own.
SHUTTER = {"night": 8.2e-4, "Earth day": 6.1e-4, "lunar day": 6.0e-4}

# The ICX267's quantum efficiency (Fig. 4) behind a 100 nm wide
# Gaussian band-pass at 660 nm: their C(lambda) F(lambda).
QE = [
    (400, 0.23), (450, 0.33), (500, 0.39), (520, 0.40), (550, 0.38),
    (600, 0.31), (650, 0.27), (700, 0.23), (750, 0.18), (800, 0.13),
    (850, 0.09), (900, 0.06), (950, 0.035), (1000, 0.02),
]  # fmt: skip


def filtered_qe():
    """(nm, value) pairs of the QE times the band-pass."""
    nm = np.arange(560.0, 761.0, 10.0)
    table = np.array(QE)
    qe = np.interp(nm, table[:, 0], table[:, 1])
    sigma = 100.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    band = np.exp(-0.5 * ((nm - 660.0) / sigma) ** 2)
    return [float(v) for pair in zip(nm, qe * band) for v in pair]


# Apollo soils' reflectance (Fig. 1): 12001 mare and 67641 highland.
MARE = [400, 0.05, 500, 0.06, 600, 0.07, 700, 0.08, 800, 0.09,
        900, 0.092, 1000, 0.095]  # fmt: skip
HIGHLAND = [400, 0.245, 500, 0.285, 600, 0.315, 700, 0.345,
            800, 0.365, 900, 0.36, 1000, 0.38]  # fmt: skip


def rig(x):
    """A transform `x` m to the right on the rover, `PITCH` down."""
    pitch = math.radians(PITCH)
    eye = (x, 0.0, HEIGHT)
    target = (x, math.cos(pitch), HEIGHT - math.sin(pitch))
    return ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0))


def clusters(row_mask):
    """Centroids of the runs of set columns in `row_mask`."""
    columns = np.flatnonzero(row_mask)
    if columns.size == 0:
        return []
    runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    return [float(run.mean()) for run in runs if run.size >= 2]


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = world.define("Looks", ernie.Xform)
        mare = looks.define(
            "Mare",
            ernie.Material,
            brdf="lunar_lambert",
            reflectance=MARE,
        )
        highland = looks.define(
            "Highland",
            ernie.Material,
            brdf="lunar_lambert",
            reflectance=HIGHLAND,
        )
        world.define(
            "Ground",
            ernie.Cube,
            size=1.0,
            material_binding=mare.path,
            translate=(0.0, 20.0, -0.05),
            scale=(60.0, 60.0, 0.1),
        )
        # A 10 cm rock, their smallest hazard, under two of the dots.
        self.rock = world.define(
            "Rock",
            ernie.Cube,
            size=1.0,
            material_binding=highland.path,
            translate=(0.18, 1.13, 0.05),
            scale=(0.08, 0.12, 0.1),
        )
        self.sun = world.define(
            "Sun", ernie.DistantLight, physical_illuminant=tokens.SUN
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.5, -0.5, 0.707)),
        )
        self.air = world.define(
            "Air", ernie.Atmosphere, preset=tokens.EARTH
        )
        # 660 nm in 11 beams 1.52 degrees apart: 50 mW, the Streamline's
        # lowest 660 nm option (the paper gives none), 80% into the dots.
        self.laser = world.define(
            "Projector",
            ernie.LaserLight,
            wavelength=660.0,
            power=0.05,
            waist_radius=(0.001, 0.001),
            pattern="grid",
            grid_columns=DOTS,
            grid_rows=1,
            grid_pitch=(GAMMA, GAMMA),
            efficiency=0.8,
            zero_order=0.0,
        ).set(tokens.TRANSFORM, rig(BASELINE))
        width = PIXEL * RESOLUTION[0]
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=width,
            vertical_aperture=PIXEL * RESOLUTION[1],
            sensor_type="mono",
            sensor_qe_mono=filtered_qe(),
            spectral_range=(560.0, 760.0),
            spectral_bands=20,
            sensor_noise=False,
            exposure_f_stop=5.6,
            exposure_iso=400.0,
            exposure_time=SHUTTER["night"],
            clipping_range=(0.05, 1e4),
        ).set(tokens.TRANSFORM, rig(0.0))
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        self.vars = []
        for name in ("LdrColor", "Electrons"):
            var = scene.define(
                f"/Render/Camera/{name}", ernie.RenderVar
            )
            var.source_name = name
            self.vars.append(var.path)
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=self.vars,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.show("night")
        self.widgets.combo(
            "Case", CASES, 0, on_change=lambda i: self.show(CASES[i])
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE

    def show(self, case):
        """Lights and exposes the scene for `case`."""
        self.camera.exposure_time = SHUTTER[case]
        self.sun.irradiance = 0.0 if case == "night" else 1361.0
        self.air.set(
            tokens.VISIBILITY,
            "inherited" if case == "Earth day" else "invisible",
        )

    def capture(self):
        """The camera's (LdrColor, Electrons) as float arrays."""
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        render_vars = outputs["/Render/Camera"].frames[0].render_vars
        ldr, electrons = (
            np.array(render_vars[str(path)].map()).astype(np.float64)
            for path in self.vars
        )
        return ldr[..., 0], electrons.reshape(*ldr.shape[:2], -1)[
            ..., 0
        ]

    def expose(self):
        """Sets the shutter so the brightest dot is 90% of white, as
        theirs, and returns the frame."""
        self.camera.exposure_time = 0.001
        for _ in range(8):
            ldr, _ = self.capture()
            peak = ldr.max()
            if abs(peak - 230.0) <= 4.0:
                break
            # LdrColor is gamma encoded: about the 2.2th root of light.
            scale = (
                (230.0 / max(peak, 1.0)) ** 2.2 if peak < 254 else 0.3
            )
            self.camera.exposure_time *= scale
        return self.capture()

    def run_case(self, case):
        """The case's clear path and rock under one shutter, set on the
        clear path as theirs was per condition: for each (rock or not)
        the image, the dots' sunlight ratio and the dots found."""
        self.show(case)
        results = {}
        for rock in (False, True):
            self.rock.set(
                tokens.VISIBILITY,
                "inherited" if rock else "invisible",
            )
            ldr, on = self.capture() if rock else self.expose()
            self.laser.power = 0.0
            _, off = self.capture()
            self.laser.power = 0.05
            laser = on - off
            row = int(np.argmax(laser.max(axis=1)))
            band = slice(max(row - 40, 0), row + 41)
            peaks = laser[band].max(axis=0)
            dots = clusters(peaks > 0.2 * peaks.max())
            sun = max(float(np.median(off[band])), 1e-9)
            ratios = [
                float(on[band, int(c)].max()) / sun for c in dots
            ]
            found = clusters(
                (ldr[band] > THRESHOLD[case] * 255.0).any(axis=0)
            )
            results[rock] = (
                ldr,
                min(ratios) if ratios else 0.0,
                found,
            )
        return results

    def obstacle(self, found, expected):
        """Their decision: fewer dots than beams, or one moved more
        than `DISPARITY` from where flat ground puts it."""
        if len(found) != len(expected):
            return True
        return any(
            abs(f - e) > DISPARITY for f, e in zip(found, expected)
        )

    def render_all(self):
        return [self.run_case(case)[True][0] for case in CASES]

    def save_output(self, path):
        images = [
            np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
            for image in self.render_all()
        ]
        write_png(path, side_by_side(images))

    def test_final(self):
        expected = self.run_case("night")[False][2]
        assert len(expected) == DOTS, (
            f"flat ground at night shows {len(expected)} dots"
        )
        ratios = {}
        for case in CASES:
            results = self.run_case(case)
            _, clear_ratio, clear = results[False]
            rocky = results[True][2]
            ratios[case] = clear_ratio
            print(
                f"{case}: (L_laser + L_sun) / L_sun {clear_ratio:.3g}, "
                f"clear path {len(clear)} dots "
                f"({'obstacle' if self.obstacle(clear, expected) else 'clear'}),"
                f" rock {len(rocky)} dots "
                f"({'obstacle' if self.obstacle(rocky, expected) else 'clear'})"
            )
            assert self.obstacle(rocky, expected), (
                f"{case}: the rock should be found"
            )
        assert ratios["night"] > ratios["Earth day"], (
            "daylight should fade the dots"
        )
        assert ratios["Earth day"] > ratios["lunar day"], (
            "the bare Sun should fade them more"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
