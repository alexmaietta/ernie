# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's subsurface scattering: balls of skin, marble and jade lit
from the side (subsurface_weight, subsurface_color, subsurface_radius
and its scale per channel). Light entering each ball leaves it further
on, softening the terminator and the shadow one ball casts on the
next; in skin red travels furthest, so their edges glow red.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (the subsurface lobe).
- P. H. Christensen and B. Burley, "Approximate Reflectance Profiles
  for Efficient Subsurface Scattering", Pixar, 2015.
- J. Jimenez et al., "Separable Subsurface Scattering", Computer
  Graphics Forum 34(6), 2015.
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
SUN_AZIMUTH = 90.0  # degrees from behind the balls: from the right
SUN_ELEVATION = 20.0
# Name, base colour, subsurface colour, radius (m) and its scale per
# channel.
MATERIALS = (
    (
        "Skin",
        (0.8, 0.55, 0.45),
        (0.9, 0.55, 0.45),
        0.04,
        (1.0, 0.35, 0.2),
    ),
    (
        "Marble",
        (0.85, 0.85, 0.83),
        (0.92, 0.92, 0.9),
        0.03,
        (1.0, 1.0, 1.0),
    ),
    (
        "Jade",
        (0.3, 0.6, 0.4),
        (0.35, 0.8, 0.5),
        0.05,
        (0.6, 1.0, 0.6),
    ),
)


def studio(rows=64, cols=128):
    """A dim latitude-longitude studio: a sky brightening upwards and a
    dark floor, linear RGB."""
    elevation = 90.0 - 180.0 * (np.arange(rows) + 0.5) / rows
    sky = np.where(
        elevation > 0.0,
        0.1 + 0.3 * np.sin(np.radians(np.maximum(elevation, 0.0))),
        0.02,
    )
    sky = np.repeat(sky[:, None], cols, axis=1)
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


def shadow_side(image, point):
    """The mean linear RGB of a strip just past a ball's terminator, on
    its side away from the sun."""
    col, row, scale = project(point)
    size = scale * RADIUS
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    x = (cols + 0.5 - col) / size
    y = (rows + 0.5 - row) / size
    mask = (x > -0.3) & (x < -0.05) & (np.abs(y) < 0.3)
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
        world.define("Sky", ernie.DomeLight, intensity=0.1).write(
            tokens.LIGHT_TEXTURE, studio()
        )
        sun = world.define(
            "Sun", ernie.DistantLight, intensity=3.0, angle=1.0
        )
        a = math.radians(SUN_AZIMUTH)
        e = math.radians(SUN_ELEVATION)
        sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (
                    math.sin(a) * math.cos(e),
                    math.cos(a) * math.cos(e),
                    math.sin(e),
                )
            ),
        )
        self.looks = []
        for i, (name, base, colour, radius, scale) in enumerate(
            MATERIALS
        ):
            look = scene.define(
                f"/World/Looks/{name}",
                ernie.Material,
                brdf="open_pbr",
                albedo=base,
            )
            look.prim.set("inputs:specular_roughness", 0.35)
            look.prim.set("inputs:subsurface_weight", 1.0)
            look.prim.set("inputs:subsurface_color", colour)
            look.prim.set("inputs:subsurface_radius", radius)
            for channel, value in zip("rgb", scale):
                look.prim.set(
                    f"inputs:subsurface_radius_scale_{channel}", value
                )
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
            vertical_aperture=APERTURE
            * RESOLUTION[1]
            / RESOLUTION[0],
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
            "Subsurface weight",
            0.0,
            1.0,
            1.0,
            on_change=self.weight,
        )
        self.widgets.slider(
            "Radius (times each material's)",
            0.1,
            4.0,
            1.0,
            on_change=self.radius,
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def weight(self, value):
        for look in self.looks:
            look.prim.set("inputs:subsurface_weight", value)

    def radius(self, times):
        for look, (*_, radius, _) in zip(self.looks, MATERIALS):
            look.prim.set("inputs:subsurface_radius", radius * times)

    def shadow_sides(self):
        image = np.asarray(self.renderer.render(self.scene))
        return [
            shadow_side(image, ball_centre(i))
            for i in range(len(MATERIALS))
        ]

    def test_final(self):
        weights = np.array([0.2126, 0.7152, 0.0722])
        scattered = self.shadow_sides()
        self.weight(0.0)
        opaque = self.shadow_sides()
        self.weight(1.0)
        for (name, *_), s, o in zip(MATERIALS, scattered, opaque):
            print(
                f"{name} past its terminator: subsurface {s @ weights:.4f}"
                f" (red/blue {s[0] / s[2]:.2f}), none {o @ weights:.4f}"
                f" (red/blue {o[0] / o[2]:.2f})"
            )
        for s, o in zip(scattered, opaque):
            assert s @ weights > 1.2 * (o @ weights), (
                "light scattered beneath the surface lights its shadowed side"
            )
        skin, bare = scattered[0], opaque[0]
        assert skin[0] / skin[2] > 1.1 * bare[0] / bare[2], (
            "red light travels furthest in skin"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
