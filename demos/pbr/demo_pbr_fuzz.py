# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's fuzz: velvet balls of dark cloth under a sheen of fibres
(fuzz_weight, fuzz_color, fuzz_roughness), lit by a low sun from behind.
The fibres catch the light at grazing, so the balls glow about their
rims.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (the fuzz layer).
- T. Zeltner, B. Burley and M. J.-Y. Chiang, "Practical
  Multiple-Scattering Sheen Using Linearly Transformed Cosines",
  SIGGRAPH 2022 Talks.
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear

RESOLUTION = (720, 320)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -6.0, 1.0)
TARGET = (0.0, 0.0, 0.5)
RADIUS = 0.5
FUZZ_WEIGHT = 1.0
FUZZ_ROUGHNESS = 0.5
SUN_AZIMUTH = 30.0  # degrees from behind the balls
SUN_ELEVATION = 15.0
# Name, cloth colour and fuzz colour.
CLOTHS = (
    ("Crimson", (0.2, 0.005, 0.02), (1.0, 0.45, 0.55)),
    ("Navy", (0.01, 0.02, 0.12), (0.45, 0.6, 1.0)),
    ("Black", (0.01, 0.01, 0.01), (0.8, 0.8, 0.8)),
)


def studio(rows=64, cols=128):
    """A latitude-longitude studio: a sky brightening upwards, a dim
    floor and two soft boxes, linear RGB."""
    elevation = 90.0 - 180.0 * (np.arange(rows) + 0.5) / rows
    azimuth = 360.0 * (np.arange(cols) + 0.5) / cols
    e, a = np.meshgrid(elevation, azimuth, indexing="ij")
    sky = np.where(
        e > 0.0, 0.2 + 0.6 * np.sin(np.radians(np.maximum(e, 0.0))), 0.03
    )
    for centre in (240.0, 330.0):
        box = (np.abs(e - 35.0) < 15.0) & (np.abs(a - centre) < 18.0)
        sky = sky + 6.0 * box
    return np.repeat(sky[..., None], 3, axis=2).astype(np.float32)


def project(point):
    """The pixel (col, row) of a world point, and pixels per metre at
    its depth."""
    eye, target = np.array(EYE), np.array(TARGET)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = np.asarray(point) - eye
    scale = FOCAL / APERTURE * RESOLUTION[0] / float(d @ forward)
    return (
        0.5 * RESOLUTION[0] + scale * float(d @ right),
        0.5 * RESOLUTION[1] - scale * float(d @ up),
        scale,
    )


def disc_mean(image, point, inner, outer):
    """The mean linear RGB of the pixels between `inner` and `outer`
    radii (fractions of the ball's) about the ball at `point`."""
    col, row, scale = project(point)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(cols + 0.5 - col, rows + 0.5 - row) / (scale * RADIUS)
    mask = (r >= inner) & (r < outer)
    return srgb_to_linear(
        image[..., :3][mask].astype(np.float64).mean(axis=0) / 255.0
    )


def ball_centre(i):
    return (1.3 * (i - 1), 0.0, RADIUS)


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
            checker_size=0.5,
            checker_color0=(0.06,) * 3,
            checker_color1=(0.03,) * 3,
        )
        # A dim studio, so the sun's grazing light shows.
        world.define("Sky", ernie.DomeLight, intensity=0.15).write(
            tokens.LIGHT_TEXTURE, studio()
        )
        self.sun = world.define(
            "Sun", ernie.DistantLight, intensity=3.0, angle=2.0
        )
        self.light_from(SUN_AZIMUTH)
        self.looks = []
        for i, (name, cloth, fuzz) in enumerate(CLOTHS):
            look = scene.define(
                f"/World/Looks/{name}",
                ernie.Material,
                brdf="open_pbr",
                albedo=cloth,
            )
            # Matte cloth: rough diffuse, no specular.
            look.prim.set("inputs:base_diffuse_roughness", 1.0)
            look.prim.set("inputs:specular_weight", 0.0)
            look.prim.set("inputs:fuzz_weight", FUZZ_WEIGHT)
            look.prim.set("inputs:fuzz_roughness", FUZZ_ROUGHNESS)
            look.prim.set("inputs:fuzz_color", fuzz)
            self.looks.append(look)
            world.define(
                name,
                ernie.Sphere,
                radius=RADIUS,
                translate=ball_centre(i),
                material_binding=look.path,
            )
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=APERTURE,
            vertical_aperture=APERTURE * RESOLUTION[1] / RESOLUTION[0],
            sensor_noise=False,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(EYE, TARGET, (0.0, 0.0, 1.0)),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )

        self.widgets.slider(
            "Fuzz weight",
            0.0,
            1.0,
            FUZZ_WEIGHT,
            on_change=lambda value: self.fuzz("fuzz_weight", value),
        )
        self.widgets.slider(
            "Fuzz roughness",
            0.0,
            1.0,
            FUZZ_ROUGHNESS,
            on_change=lambda value: self.fuzz("fuzz_roughness", value),
        )
        self.widgets.slider(
            "Sun azimuth (0 behind the balls)",
            0.0,
            180.0,
            SUN_AZIMUTH,
            on_change=self.light_from,
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def fuzz(self, name, value):
        """Sets fuzz input `name` on every ball."""
        for look in self.looks:
            look.prim.set(f"inputs:{name}", value)

    def light_from(self, degrees):
        a = math.radians(degrees)
        e = math.radians(SUN_ELEVATION)
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

    def rims(self):
        """Each ball's rim, in linear luminance."""
        image = np.asarray(self.renderer.render(self.scene))
        weights = np.array([0.2126, 0.7152, 0.0722])
        return [
            float(disc_mean(image, ball_centre(i), 0.8, 0.97) @ weights)
            for i in range(len(CLOTHS))
        ]

    def test_final(self):
        fuzzy = self.rims()
        self.fuzz("fuzz_weight", 0.0)
        bare = self.rims()
        self.fuzz("fuzz_weight", FUZZ_WEIGHT)
        self.fuzz("fuzz_roughness", 1.0)
        rough = self.rims()
        self.fuzz("fuzz_roughness", FUZZ_ROUGHNESS)
        for (name, *_), f, b, r in zip(CLOTHS, fuzzy, bare, rough):
            print(
                f"{name}'s rim: fuzz {f:.4f}, none {b:.4f}, roughest "
                f"fuzz {r:.4f}"
            )
        for f, b in zip(fuzzy, bare):
            assert f > 2.0 * b, "the fuzz lights the rims"
        assert rough[2] > 1.5 * fuzzy[2], (
            "rougher fibres spread the sheen further over the balls"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
