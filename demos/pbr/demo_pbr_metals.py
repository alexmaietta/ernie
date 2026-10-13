# SPDX-License-Identifier: BSD-3-Clause
"""Metals in OpenPBR: base_metalness 1, base_color their reflectance
head on and specular_color their F82 tint towards grazing (Hoffman 2023),
in a studio of two soft boxes over a dim floor.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (the metal's F82-tint Fresnel).
- N. Hoffman, "Generalization of Adobe's Fresnel Model", 2023.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear

RESOLUTION = (800, 320)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -7.0, 1.2)
TARGET = (0.0, 0.0, 0.45)
RADIUS = 0.4
ROUGHNESS = 0.2
# Linear sRGB reflectance head on and F82 tint, approximated from
# measured indices of refraction.
METALS = (
    ("Gold", (1.0, 0.78, 0.34), (0.98, 0.93, 0.74)),
    ("Copper", (0.96, 0.64, 0.54), (0.99, 0.89, 0.83)),
    ("Aluminium", (0.92, 0.92, 0.93), (0.97, 0.98, 0.99)),
    ("Silver", (0.97, 0.96, 0.92), (1.0, 1.0, 0.99)),
    ("Titanium", (0.54, 0.50, 0.45), (0.85, 0.83, 0.80)),
)


def studio(rows=64, cols=128):
    """A latitude-longitude studio: a sky brightening upwards, a dim
    floor and two soft boxes, linear RGB."""
    elevation = 90.0 - 180.0 * (np.arange(rows) + 0.5) / rows
    azimuth = 360.0 * (np.arange(cols) + 0.5) / cols
    e, a = np.meshgrid(elevation, azimuth, indexing="ij")
    sky = np.where(
        e > 0.0,
        0.2 + 0.6 * np.sin(np.radians(np.maximum(e, 0.0))),
        0.03,
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
    depth = float(d @ forward)
    scale = FOCAL / APERTURE * RESOLUTION[0] / depth
    return (
        0.5 * RESOLUTION[0] + scale * float(d @ right),
        0.5 * RESOLUTION[1] - scale * float(d @ up),
        scale,
    )


def disc_mean(image, point, inner, outer, upper=False):
    """The mean linear RGB of the pixels between `inner` and `outer`
    radii (fractions of the ball's) about the ball at `point`; with
    `upper`, its upper half only."""
    col, row, scale = project(point)
    radius = scale * RADIUS
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(cols + 0.5 - col, rows + 0.5 - row) / radius
    mask = (r >= inner) & (r < outer)
    if upper:
        mask &= rows + 0.5 < row
    return srgb_to_linear(
        image[..., :3][mask].astype(np.float64).mean(axis=0) / 255.0
    )


def ball_centre(i):
    return (1.0 * (i - 2), 0.0, RADIUS)


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
        world.define("Sky", ernie.DomeLight, intensity=1.0).write(
            tokens.LIGHT_TEXTURE, studio()
        )
        world.define(
            "Sun", ernie.DistantLight, intensity=2.0, angle=2.0
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.4, -0.5, 0.77)),
        )
        self.looks = []
        for i, (name, colour, tint) in enumerate(METALS):
            look = scene.define(
                f"/World/Looks/{name}",
                ernie.Material,
                brdf="open_pbr",
                albedo=colour,
            )
            look.prim.set("inputs:base_metalness", 1.0)
            look.prim.set("inputs:specular_roughness", ROUGHNESS)
            look.prim.set("inputs:specular_color", tint)
            self.looks.append(look)
            world.define(
                name,
                ernie.Sphere,
                radius=RADIUS,
                translate=ball_centre(i),
                material_binding=look.path,
            )
        aperture = APERTURE * RESOLUTION[1] / RESOLUTION[0]
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=APERTURE,
            vertical_aperture=aperture,
            sensor_noise=False,
        ).set(tokens.TRANSFORM, ernie.look_at_transform(EYE, TARGET))
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )

        self.widgets.slider(
            "Roughness", 0.0, 1.0, ROUGHNESS, on_change=self.roughen
        )
        self.widgets.combo(
            "Edge tint",
            ["F82 tint", "None (white)"],
            0,
            on_change=self.tint,
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def roughen(self, roughness):
        for look in self.looks:
            look.prim.set("inputs:specular_roughness", roughness)

    def tint(self, index):
        for look, (_, _, tint) in zip(self.looks, METALS):
            look.prim.set(
                "inputs:specular_color",
                tint if index == 0 else (1.0, 1.0, 1.0),
            )

    def render_now(self):
        return np.asarray(self.renderer.render(self.scene))

    def test_final(self):
        image = self.render_now()
        balls = [
            disc_mean(image, ball_centre(i), 0.0, 0.9)
            for i in range(len(METALS))
        ]
        # Gold's upper rim, which reflects the sky near grazing.
        tinted = disc_mean(image, ball_centre(0), 0.93, 0.99, True)
        self.tint(1)
        plain = disc_mean(
            self.render_now(), ball_centre(0), 0.93, 0.99, True
        )
        self.tint(0)
        self.roughen(0.7)
        rough = self.render_now()
        self.roughen(ROUGHNESS)
        for (name, _, _), ball in zip(METALS, balls):
            print(f"{name}: {ball.round(3)}")
        print(
            f"Gold's rim tinted {tinted.round(3)}, untinted "
            f"{plain.round(3)}"
        )
        gold, copper, aluminium = balls[0], balls[1], balls[2]
        assert gold[0] > 1.5 * gold[2], "gold is yellow"
        assert copper[0] > 1.3 * copper[1], "copper is red"
        assert max(aluminium) < 1.1 * min(aluminium), (
            "aluminium is neutral"
        )
        assert tinted[2] / tinted[0] < 0.96 * plain[2] / plain[0], (
            "the F82 tint colours gold's grazing reflection"
        )
        assert (
            np.abs(rough.astype(int) - image.astype(int)).mean() > 1.0
        ), "roughness blurs the reflections"


if __name__ == "__main__":
    ernie.demo.main(Demo)
