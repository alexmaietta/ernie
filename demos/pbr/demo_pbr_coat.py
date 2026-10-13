# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's coat: a clear (or tinted) layer over four rough bases, red
and white plastic, black and a blue metal. A smooth coat lays a sharp
reflection over the base's blurred one; its colour tints what passes
through it and its darkening deepens the base beneath.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (the coat layer).
- C. Kulla and A. Conty, "Revisiting Physically Based Shading at
  Imageworks", SIGGRAPH 2017 course (layering by albedo scaling).
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import srgb_to_linear

RESOLUTION = (800, 320)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -6.0, 1.0)
TARGET = (0.0, 0.0, 0.45)
RADIUS = 0.4
# Name, base colour, metalness and the base's specular roughness.
BASES = (
    ("Red", (0.6, 0.04, 0.03), 0.0, 0.5),
    ("White", (0.8, 0.8, 0.8), 0.0, 0.5),
    ("Black", (0.02, 0.02, 0.02), 0.0, 0.6),
    ("BlueMetal", (0.15, 0.3, 0.75), 1.0, 0.45),
)
COLOURS = (
    ("Clear", (1.0, 1.0, 1.0)),
    ("Amber varnish", (1.0, 0.7, 0.35)),
    ("Blue tint", (0.6, 0.8, 1.0)),
)
COAT = {
    "coat_weight": 1.0,
    "coat_roughness": 0.0,
    "coat_ior": 1.5,
    "coat_darkening": 1.0,
}


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


def disc_peak(image, point):
    """The brightest pixel's mean of R, G and B (0 to 255) on the ball
    at `point`."""
    col, row, scale = project(point)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(cols + 0.5 - col, rows + 0.5 - row) / (
        scale * RADIUS
    )
    return float(
        image[..., :3][r < 0.95].astype(np.float64).mean(axis=1).max()
    )


def ball_centre(i):
    return (1.0 * i - 1.5, 0.0, RADIUS)


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
        for i, (name, colour, metalness, roughness) in enumerate(
            BASES
        ):
            look = scene.define(
                f"/World/Looks/{name}",
                ernie.Material,
                brdf="open_pbr",
                albedo=colour,
            )
            look.prim.set("inputs:base_metalness", metalness)
            look.prim.set("inputs:specular_roughness", roughness)
            for input_name, value in COAT.items():
                look.prim.set(f"inputs:{input_name}", value)
            look.prim.set("inputs:coat_color", COLOURS[0][1])
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

        for label, input_name, low, high in (
            ("Coat weight", "coat_weight", 0.0, 1.0),
            ("Coat roughness", "coat_roughness", 0.0, 1.0),
            ("Coat IOR", "coat_ior", 1.0, 2.5),
            ("Coat darkening", "coat_darkening", 0.0, 1.0),
        ):
            self.widgets.slider(
                label,
                low,
                high,
                COAT[input_name],
                on_change=lambda value, name=input_name: self.coat(
                    name, value
                ),
            )
        self.widgets.combo(
            "Coat colour",
            [name for name, _ in COLOURS],
            0,
            on_change=lambda index: self.coat(
                "coat_color", COLOURS[index][1]
            ),
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def coat(self, name, value):
        """Sets coat input `name` on every ball."""
        for look in self.looks:
            look.prim.set(f"inputs:{name}", value)

    def balls(self):
        image = np.asarray(self.renderer.render(self.scene))
        return [
            disc_mean(image, ball_centre(i), 0.0, 0.95)
            for i in range(len(BASES))
        ]

    def black_peak(self):
        image = np.asarray(self.renderer.render(self.scene))
        return disc_peak(image, ball_centre(2))

    def test_final(self):
        coated = self.balls()
        sharp = self.black_peak()
        self.coat("coat_weight", 0.0)
        bare = self.balls()
        blurred = self.black_peak()
        self.coat("coat_weight", 1.0)
        self.coat("coat_color", COLOURS[1][1])
        amber = self.balls()
        self.coat("coat_color", COLOURS[0][1])
        self.coat("coat_darkening", 0.0)
        undarkened = self.balls()
        self.coat("coat_darkening", 1.0)
        for (name, *_), c, b, a, u in zip(
            BASES, coated, bare, amber, undarkened
        ):
            print(
                f"{name}: coated {c.round(3)}, bare {b.round(3)}, "
                f"amber {a.round(3)}, undarkened {u.round(3)}"
            )
        print(
            f"Black ball's brightest pixel coated {sharp:.0f}, bare "
            f"{blurred:.0f}"
        )
        white = 1
        assert sharp > 200.0 and blurred < 0.7 * sharp, (
            "the smooth coat reflects the soft boxes sharply off the "
            "black ball"
        )
        assert (
            amber[white][0] / amber[white][2]
            > 1.5 * coated[white][0] / coated[white][2]
        ), "an amber coat warms the white ball"
        assert undarkened[0][0] > coated[0][0] * 1.02, (
            "darkening deepens the red ball under its coat"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
