# SPDX-License-Identifier: BSD-3-Clause
"""Adaptive spheres from a marble to a moon: an Earth-sized sphere as the
ground, a spinning globe, a ball, a tilted egg (a scaled sphere) and the
Moon at its distance, each refined towards the eye. The panel picks the
view and tints the LOD levels."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

EARTH = 6371000.0
MOON = 1737400.0
MOON_DISTANCE = 384400000.0
MARBLE = 0.02

MOON_CENTRE = (
    MOON_DISTANCE
    * np.array([1.0, 0.4, 0.12])
    / np.linalg.norm([1.0, 0.4, 0.12])
)

# Views: a name, the eye and what it looks at.
VIEWS = [
    ("The scene", (-2.0, -9.0, 2.5), (0.0, 0.0, 0.8)),
    ("The marble", (0.05, -0.05, 0.04), (0.0, 0.0, MARBLE)),
    ("The horizon and the Moon", (0.0, 0.0, 1.7), tuple(MOON_CENTRE)),
    ("From orbit", (0.0, -1.6e7, 1.2e7), (0.0, 0.0, -EARTH)),
]


def globe_map():
    """A globe's map, latitude by longitude (rows from the north): blue and
    white tiles of 22.5 degrees, the equator and prime meridian red."""
    rows, cols = 64, 128
    r, c = np.mgrid[0:rows, 0:cols]
    white = ((r // 8) + (c // 8)) % 2 == 0
    line = (r == rows // 2) | (r == rows // 2 - 1)
    line |= (c == cols // 2) | (c == cols // 2 - 1)
    texels = np.empty((rows, cols, 4), np.uint8)
    texels[...] = (40, 90, 200, 255)
    texels[white] = (230, 230, 230, 255)
    texels[line] = (220, 30, 30, 255)
    return texels


def at(p):
    """A translation to `p`."""
    m = np.eye(4)
    m[:3, 3] = p
    return m


def turned(angle, axis):
    """A rotation by `angle` radians about unit `axis`."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + s * k + (1.0 - c) * (k @ k)
    return m


def define_look(scene, look, colour, texture=None):
    """A material of `colour`, or of `texture` read by latitude and
    longitude."""
    look = ernie.Path(look)
    scene.define(look, ernie.Material)
    scene.define(
        look.child("Surface"),
        ernie.Shader,
        info_id=tokens.USD_PREVIEW_SURFACE,
        diffuse_color=colour,
        roughness=0.6,
    )
    scene.connect(
        look,
        tokens.SURFACE_OUTPUT,
        look.child("Surface"),
        tokens.SURFACE_OUTPUT,
    )
    if texture is not None:
        scene.define(
            look.child("Map"),
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
        ).write(tokens.SHADER_TEXTURE, texture)
        scene.connect(
            look.child("Surface"),
            tokens.DIFFUSE_COLOR,
            look.child("Map"),
            "outputs:rgb",
        )


def define_sphere(scene, path, radius, transform, look):
    """An adaptive sphere of `radius` at `transform`, wearing `look`."""
    scene.define(
        path,
        ernie.Sphere,
        radius=radius,
        tessellation=tokens.ADAPTIVE_SPHERE,
        material_binding=ernie.Path(look),
    )
    scene.set(ernie.Path(path), tokens.TRANSFORM, transform)


class Demo(ernie.demo.Demo):
    num_frames = 3

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World", ernie.Xform)
        define_look(scene, "/World/Looks/Ground", (0.35, 0.42, 0.3))
        define_look(
            scene, "/World/Looks/Globe", (1.0, 1.0, 1.0), globe_map()
        )
        define_look(scene, "/World/Looks/Ball", (0.8, 0.35, 0.2))
        define_look(scene, "/World/Looks/Egg", (0.9, 0.85, 0.7))
        define_look(scene, "/World/Looks/Moon", (0.5, 0.5, 0.5))
        # The ground is the top of an Earth-sized sphere.
        define_sphere(
            scene,
            "/World/Earth",
            EARTH,
            at((0, 0, -EARTH)),
            "/World/Looks/Ground",
        )
        define_sphere(
            scene,
            "/World/Globe",
            MARBLE,
            at((0, 0, MARBLE)),
            "/World/Looks/Globe",
        )
        define_sphere(
            scene,
            "/World/Ball",
            1.0,
            at((3, 1, 1)),
            "/World/Looks/Ball",
        )
        # An egg: a unit sphere squashed along x and y, tipped over.
        egg = at((-3, 1, 1)) @ turned(math.radians(35.0), (1, 0, 0))
        egg = egg @ np.diag([0.6, 0.6, 1.0, 1.0])
        define_sphere(
            scene, "/World/Egg", 1.0, egg, "/World/Looks/Egg"
        )
        define_sphere(
            scene,
            "/World/Moon",
            MOON,
            at(MOON_CENTRE),
            "/World/Looks/Moon",
        )
        scene.define(
            "/World/Sun", ernie.DistantLight, intensity=1.0
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.5, 0.75)),
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=35.0,
            clipping_range=(1e-3, math.inf),
        )
        self.settings = scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
        )
        self.spin = 0.0

        self.widgets.combo(
            "View", [v[0] for v in VIEWS], 0, on_change=self.look
        )
        self.widgets.checkbox(
            "Tint LOD levels",
            False,
            on_change=lambda on: setattr(
                self.settings, "lod_show_levels", on
            ),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
        self.look(0)

    def step(self):
        """The globe turns a degree a frame."""
        self.spin += 1.0
        self.scene.set(
            ernie.Path("/World/Globe"),
            tokens.TRANSFORM,
            at((0, 0, MARBLE))
            @ turned(math.radians(self.spin), (0, 0, 1)),
        )

    def test_final(self):
        # Each view's centre pixel shows its sphere, and the marble's
        # close-up fills most of the frame with its map.
        centres = {}
        marble_cover = 0.0
        for i in range(len(VIEWS)):
            if i == 2:
                continue  # The Moon is small and may be unlit.
            self.look(i)
            image = np.asarray(self.renderer.render(self.scene))
            rows, cols = image.shape[:2]
            centres[i] = float(
                image[rows // 2, cols // 2, :3].astype(np.int64).sum()
            )
            if i == 1:
                lit = (
                    image[:, :, :3].astype(np.int64).sum(axis=2) > 60
                )
                marble_cover = float(lit.mean())
        print(
            f"centres {centres[0]:.0f} {centres[1]:.0f} {centres[3]:.0f}; "
            f"marble covers {marble_cover:.2f}"
        )
        assert centres[0] > 60.0, "the scene's ground is lit"
        assert centres[1] > 60.0, "the marble is lit"
        assert centres[3] > 60.0, "the Earth from orbit is lit"
        assert marble_cover > 0.3, "the marble fills its close-up"

    def look(self, i):
        _, eye, target = VIEWS[i]
        if self.viewer is not None:
            self.viewer.look_at(eye, target)
        else:
            self.camera.set(
                tokens.TRANSFORM,
                ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0)),
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
