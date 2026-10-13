# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's rough diffuse: a Lambert ball beside one of
base_diffuse_roughness (EON, an energy-preserving Oren-Nayar), lit by a
sun near the camera. Rough, the ball flattens towards a full moon's look,
its limb as bright as its middle.

References:
- J. Portsmouth, P. Kutz and S. Hill, "EON: A practical
  energy-preserving rough diffuse BRDF", JCGT 14(1), 2025.
- M. Oren and S. K. Nayar, "Generalization of Lambert's reflectance
  model", SIGGRAPH 1994.
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear

RESOLUTION = (720, 360)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -7.0, 0.0)
TARGET = (0.0, 0.0, 0.0)
RADIUS = 0.8
ROUGHNESS = 1.0
PHASE = 20.0  # degrees between the sun and the camera, seen from the balls
ALBEDO = (0.6, 0.58, 0.55)
CENTRES = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


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
    r = np.hypot(cols + 0.5 - col, rows + 0.5 - row) / (
        scale * RADIUS
    )
    mask = (r >= inner) & (r < outer)
    return srgb_to_linear(
        image[..., :3][mask].astype(np.float64).mean(axis=0) / 255.0
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        # A dim sky, so the sun's shading shows.
        world.define("Sky", ernie.DomeLight, intensity=0.02)
        self.sun = world.define(
            "Sun", ernie.DistantLight, intensity=0.8, angle=0.53
        )
        self.light_at(PHASE)
        self.looks = []
        for name, centre, roughness in (
            ("Lambert", CENTRES[0], 0.0),
            ("Rough", CENTRES[1], ROUGHNESS),
        ):
            look = scene.define(
                f"/World/Looks/{name}",
                ernie.Material,
                brdf="open_pbr",
                albedo=ALBEDO,
            )
            # No specular, the diffuse alone.
            look.prim.set("inputs:specular_weight", 0.0)
            look.prim.set("inputs:base_diffuse_roughness", roughness)
            self.looks.append(look)
            world.define(
                name,
                ernie.Sphere,
                radius=RADIUS,
                translate=centre,
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
            "Diffuse roughness (right)",
            0.0,
            1.0,
            ROUGHNESS,
            on_change=self.roughen,
        )
        self.widgets.slider(
            "Sun phase angle",
            0.0,
            150.0,
            PHASE,
            on_change=self.light_at,
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def roughen(self, roughness):
        self.looks[1].prim.set(
            "inputs:base_diffuse_roughness", roughness
        )

    def light_at(self, degrees):
        """The sun `degrees` round from behind the camera to the right,
        a little above."""
        a = math.radians(degrees)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (math.sin(a), -math.cos(a), 0.1)
            ),
        )

    def limb_darkening(self, image, i):
        """The ball's limb over its middle, in linear luminance."""
        weights = np.array([0.2126, 0.7152, 0.0722])
        middle = disc_mean(image, CENTRES[i], 0.0, 0.3) @ weights
        limb = disc_mean(image, CENTRES[i], 0.8, 0.95) @ weights
        return float(limb / middle)

    def test_final(self):
        self.light_at(0.0)
        image = np.asarray(self.renderer.render(self.scene))
        lambert = self.limb_darkening(image, 0)
        rough = self.limb_darkening(image, 1)
        self.roughen(0.0)
        smooth = np.asarray(self.renderer.render(self.scene))
        self.roughen(ROUGHNESS)
        self.light_at(PHASE)
        print(
            f"Limb over middle with the sun behind the camera: Lambert "
            f"{lambert:.3f}, rough {rough:.3f}"
        )
        assert 0.35 < lambert < 0.75, "Lambert's limb darkens as cos"
        assert rough > lambert + 0.1, (
            "the rough ball's limb is flatter"
        )
        assert abs(self.limb_darkening(smooth, 1) - lambert) < 0.02, (
            "roughness 0 is Lambert"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
