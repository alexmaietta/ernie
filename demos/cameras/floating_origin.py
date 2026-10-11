# SPDX-License-Identifier: BSD-3-Clause
"""Metre-sized probes from the origin out to Neptune's distance, each
as sharp as the last: the renderer draws relative to the eye."""

import functools
import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

# Metres in an astronomical unit.
AU = 1.495978707e11

# The places, each a probe at its distance from the origin along one
# direction: a red cube, a green sphere and a blue cube a metre apart,
# with a 10 cm cube on the sphere. Doubles hold such detail to about a
# millimetre out to the solar system's edge (1e13 m); beyond, they too
# run short.
PLACES = [
    ("Home", "Home", 0.0),
    ("Moon", "The Moon's distance", 3.844e8),
    ("Mars", "Mars (1.52 AU)", 1.524 * AU),
    ("Saturn", "Saturn (9.58 AU)", 9.583 * AU),
    ("Neptune", "Neptune (30.07 AU)", 30.07 * AU),
]
DIRECTION = np.array([1.0, 0.35, 0.12]) / np.linalg.norm(
    [1.0, 0.35, 0.12]
)
# Where the camera starts, and the test looks: the farthest.
START = len(PLACES) - 1


def position(distance):
    return tuple(float(c) for c in DIRECTION * distance)


def step(distance, mantissa):
    """The spacing of floating-point numbers near `distance` with a
    `mantissa`-bit significand (23 for float32, 52 for float64)."""
    if distance <= 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(distance)) - mantissa)


def length(metres):
    for unit, size in (("km", 1e3), ("m", 1.0), ("mm", 1e-3)):
        if metres >= size:
            return f"{metres / size:.3g} {unit}"
    return f"{metres * 1e6:.3g} um"


def describe(index):
    _, label, distance = PLACES[index]
    if distance == 0.0:
        return f"{label}: at the origin."
    return (
        f"{label}: {length(distance)} out, where a float32 "
        f"coordinate steps by {length(step(distance, 23))} and a "
        f"float64 by "
        f"{length(step(distance, 52))}."
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        self.probes = []
        for key, _, distance in PLACES:
            probe = world.define(
                key, ernie.Xform, translate=position(distance)
            )
            probe.define(
                "Red",
                ernie.Cube,
                size=0.6,
                translate=(-1.0, 0.0, 0.0),
                display_color=(0.8, 0.1, 0.1),
            )
            probe.define(
                "Green",
                ernie.Sphere,
                radius=0.35,
                display_color=(0.1, 0.7, 0.2),
            )
            probe.define(
                "Blue",
                ernie.Cube,
                size=0.6,
                translate=(1.0, 0.0, 0.0),
                display_color=(0.1, 0.2, 0.8),
            )
            probe.define(
                "Tip",
                ernie.Cube,
                size=0.1,
                translate=(0.0, 0.0, 0.4),
                display_color=(0.9, 0.9, 0.9),
            )
            self.probes.append(probe)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.4, -0.8, 0.45)),
        )
        # Looking at the farthest probe from 3.5 m; the viewer's
        # navigation drives it, fitting its clipping to the distance.
        target = np.array(position(PLACES[START][2]))
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(0.05, 1000.0),
            exposure_mode=tokens.AUTO,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(target + (0.0, -3.5, 0.6)), tuple(target)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(480, 270),
        )
        self.viewer = viewer
        if viewer is not None:
            viewer.active_camera = camera.path

        # The places, a button each, and where the view is.
        w = self.widgets
        self.where = w.text(describe(START))
        w.separator()
        for i, (_, label, _) in enumerate(PLACES):
            w.button(label, functools.partial(self.go, i))

    def go(self, index):
        """Frames place `index`'s probe with the viewer's camera."""
        self.widgets.set(self.where, describe(index))
        if self.viewer is not None:
            self.viewer.frame([self.probes[index].path])

    def test_final(self):
        image = self.render_image()[..., :3].astype(np.float64)
        rows, cols, _ = image.shape
        band = image[rows // 3 : 2 * rows // 3]
        # The probe's three shapes, left to right, 30 AU out.
        thirds = [
            band[:, i * cols // 3 : (i + 1) * cols // 3]
            for i in range(3)
        ]
        lit = [
            third[third.sum(axis=2) > 30.0].mean(axis=0)
            for third in thirds
        ]
        for colour, (name, channel) in zip(
            lit, (("red", 0), ("green", 1), ("blue", 2))
        ):
            assert (
                colour[channel]
                > 1.3 * np.delete(colour, channel).max()
            ), f"the {name} shape should stand in its place: {colour}"


if __name__ == "__main__":
    ernie.demo.main(Demo)
