# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's specular anisotropy: brushed aluminium. Its lobe is rougher
along the surface's tangent, the way its texture coordinates' u grows
(specular_roughness_anisotropy), so highlights stretch along u, across
the grooves. Each mesh carries texture coordinates: a plate brushed side
to side (u up it), a disc brushed in circles (u outwards, so its
highlights streak along radii) and a ball whose own u runs round it.
A lamp in front of each lights a highlight on it.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (specular_roughness_anisotropy).
- B. Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012
  course (anisotropic GGX).
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

RESOLUTION = (800, 360)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -6.5, 1.6)
TARGET = (0.0, 0.0, 0.7)
ANISOTROPY = 0.7
ROUGHNESS = 0.35
SKY = 0.4  # the studio's intensity
SUN = 0.6
TILT = 35.0  # degrees the panels lean back from upright
RADIUS = 0.6  # the ball's
CENTRES = ((-1.6, 0.0, 0.75), (0.0, 0.0, 0.75), (1.6, 0.0, 0.65))
ALUMINIUM = (0.92, 0.92, 0.93)


def studio(rows=64, cols=128):
    """A latitude-longitude studio without soft boxes, which the lamps
    light instead: a sky brightening upwards over a dim floor, linear
    RGB."""
    elevation = 90.0 - 180.0 * (np.arange(rows) + 0.5) / rows
    e = np.repeat(elevation[:, None], cols, axis=1)
    sky = np.where(
        e > 0.0, 0.2 + 0.6 * np.sin(np.radians(np.maximum(e, 0.0))), 0.03
    )
    return np.repeat(sky[..., None], 3, axis=2).astype(np.float32)


def panel_axes():
    """A panel's right, up and normal: leaning back by TILT, facing the
    camera."""
    t = math.radians(TILT)
    return (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, math.sin(t), math.cos(t)]),
        np.array([0.0, -math.cos(t), math.sin(t)]),
    )


def panel(centre, local):
    """Points (x, y) in a panel's plane at `centre`, in the world."""
    right, up, _ = panel_axes()
    return np.asarray(centre) + local[:, :1] * right + local[:, 1:] * up


def plate(centre, size=1.0):
    """A square plate: points, face counts, indices and texture
    coordinates, u up it."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    st = corners[:, ::-1].astype(np.float32)
    return panel(centre, (corners - 0.5) * size), np.array([4]), np.arange(4), st


def disc(centre, inner=0.05, outer=0.6, rings=16, segments=256):
    """An annulus of quads: points, face counts, indices and texture
    coordinates, u outwards and v round it (its seam's points
    doubled)."""
    u = np.arange(rings + 1) / rings
    v = np.arange(segments + 1) / segments
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radius = inner + (outer - inner) * uu
    angle = 2.0 * math.pi * vv
    local = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle)], -1
    ).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(rings), np.arange(segments), indexing="ij")
    first = (i * (segments + 1) + j).ravel()
    # Counter-clockwise seen from the front: out, round, then in.
    indices = np.stack(
        [first, first + segments + 1, first + segments + 2, first + 1], -1
    ).ravel()
    st = np.stack([uu, vv], -1).reshape(-1, 2).astype(np.float32)
    return panel(centre, local), np.full(rings * segments, 4), indices, st


def lamp_for(point, normal, distance=2.0):
    """Where a lamp shows its reflection at `point` (of `normal`) to the
    camera."""
    view = np.array(EYE) - point
    view /= np.linalg.norm(view)
    reflected = 2.0 * float(normal @ view) * normal - view
    return tuple(point + distance * reflected)


def ball_normal():
    """The ball's normal where its lamp shows: on its upper front."""
    ball = np.array(EYE) - np.array(CENTRES[2])
    ball = ball / np.linalg.norm(ball) + np.array([0.0, 0.0, 0.5])
    return ball / np.linalg.norm(ball)


def ball_highlight():
    """The point on the ball where its lamp shows."""
    return np.array(CENTRES[2]) + RADIUS * ball_normal()


def lamps():
    """A lamp for each subject: on the plate's middle, up the disc and on
    the ball's upper front."""
    _, up, normal = panel_axes()
    return (
        lamp_for(np.array(CENTRES[0]), normal),
        lamp_for(np.array(CENTRES[1]) + 0.3 * up, normal),
        lamp_for(ball_highlight(), ball_normal()),
    )


def project(point):
    """The pixel (col, row) of a world point."""
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
    )


def stretch(image, point, half=40):
    """A highlight's spread across the image over its spread down it,
    within `half` pixels of `point`: the second moments of the pixels
    brighter than the window's mean, weighted by how much."""
    col, row = (int(x) for x in project(point))
    window = image[row - half : row + half, col - half : col + half, :3]
    brightness = window.astype(np.float64).mean(axis=2)
    weight = np.maximum(brightness - brightness.mean(), 0.0)
    rows, cols = np.mgrid[0 : 2 * half, 0 : 2 * half]
    total = weight.sum()
    mean_row = (weight * rows).sum() / total
    mean_col = (weight * cols).sum() / total
    across = (weight * (cols - mean_col) ** 2).sum() / total
    down = (weight * (rows - mean_row) ** 2).sum() / total
    return math.sqrt(across / down)


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
        self.sky = world.define("Sky", ernie.DomeLight, intensity=SKY)
        self.sky.write(tokens.LIGHT_TEXTURE, studio())
        self.sun = world.define(
            "Sun", ernie.DistantLight, intensity=SUN, angle=1.0
        )
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.5, -0.6, 0.62)),
        )
        for i, at in enumerate(lamps()):
            world.define(
                f"Lamp{i}",
                ernie.SphereLight,
                radius=0.04,
                intensity=150.0,
                translate=at,
            )
        self.look = scene.define(
            "/World/Looks/Brushed",
            ernie.Material,
            brdf="open_pbr",
            albedo=ALUMINIUM,
        )
        self.look.prim.set("inputs:base_metalness", 1.0)
        self.look.prim.set("inputs:specular_roughness", ROUGHNESS)
        self.look.prim.set(
            "inputs:specular_roughness_anisotropy", ANISOTROPY
        )
        for name, (points, counts, indices, st) in (
            ("Plate", plate(CENTRES[0])),
            ("Disc", disc(CENTRES[1])),
        ):
            world.define(
                name,
                ernie.Mesh,
                points=points,
                face_vertex_counts=counts,
                face_vertex_indices=indices,
                material_binding=self.look.path,
            ).write("primvars:st", st)
        # Finely tessellated: its tangents are each triangle's.
        world.define(
            "Ball",
            ernie.Sphere,
            radius=RADIUS,
            tessellation_level=256,
            translate=CENTRES[2],
            material_binding=self.look.path,
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
            "Anisotropy",
            0.0,
            1.0,
            ANISOTROPY,
            on_change=lambda value: self.look.prim.set(
                "inputs:specular_roughness_anisotropy", value
            ),
        )
        self.widgets.slider(
            "Roughness",
            0.0,
            1.0,
            ROUGHNESS,
            on_change=lambda value: self.look.prim.set(
                "inputs:specular_roughness", value
            ),
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def stretches(self):
        """The plate's and the ball's lamps' highlights' spreads across
        over down."""
        image = np.asarray(self.renderer.render(self.scene))
        return (
            stretch(image, CENTRES[0]),
            stretch(image, ball_highlight(), 16),
        )

    def test_final(self):
        # The lamps alone.
        self.sky.intensity = 0.0
        self.sun.intensity = 0.0
        brushed = self.stretches()
        self.look.prim.set("inputs:specular_roughness_anisotropy", 0.0)
        plain = self.stretches()
        self.look.prim.set(
            "inputs:specular_roughness_anisotropy", ANISOTROPY
        )
        self.sky.intensity = SKY
        self.sun.intensity = SUN
        print(
            f"Highlights' spread across over down: plate {brushed[0]:.2f}"
            f" (isotropic {plain[0]:.2f}), ball {brushed[1]:.2f} "
            f"(isotropic {plain[1]:.2f})"
        )
        assert brushed[0] < 0.6 * plain[0], (
            "the plate's highlight stretches up it, along u"
        )
        assert brushed[1] > 1.5 * plain[1], (
            "the ball's highlight stretches round it, along u"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
