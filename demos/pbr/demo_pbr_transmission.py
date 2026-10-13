# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's transmission before a wall of coloured tiles: a solid glass
ball refracting the wall by its specular_ior (geometry_thin_walled 0), a
tinted thin-walled pane letting it through unbent (transmission_color),
and beside them plain alpha blending, a UsdPreviewSurface panel of
opacity below 1. Transparent surfaces draw over the opaque scene, the
nearest one per pixel.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (transmission and thin walls).
- Pixar, "UsdPreviewSurface Specification" (opacity).
"""

import colorsys

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

RESOLUTION = (800, 360)
FOCAL = 40.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -6.0, 1.1)
TARGET = (0.0, 0.0, 0.8)
IOR = 1.5
OPACITY = 0.5
RADIUS = 0.6
BALL = (-1.6, 0.0, 0.8)
PANE = (0.0, 0.0, 0.8)
PANEL = (1.6, 0.0, 0.8)
TINTS = (
    ("Green", (0.2, 0.9, 0.3)),
    ("Amber", (1.0, 0.6, 0.1)),
    ("Clear", (1.0, 1.0, 1.0)),
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


def tile_colour(i, j):
    """The wall's tile in column `i` and row `j`: hues round the wheel
    along it, light and dark rows alternating, linear RGB."""
    value = 0.8 if (i + j) % 2 == 0 else 0.35
    return colorsys.hsv_to_rgb(i / 32.0, 0.8, value)


def quad(centre, width=1.0, height=1.2):
    """A vertical quad facing the camera: points, counts and indices."""
    x, _, z = centre
    w, h = 0.5 * width, 0.5 * height
    points = np.array(
        [
            [x - w, 0.0, z - h],
            [x + w, 0.0, z - h],
            [x + w, 0.0, z + h],
            [x - w, 0.0, z + h],
        ]
    )
    return points, np.array([4]), np.arange(4)


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


def window(image, point, half=20):
    """The RGB (0 to 255) of the pixels within `half` of `point`."""
    col, row = (int(x) for x in project(point))
    return image[row - half : row + half, col - half : col + half, :3].astype(
        np.float64
    )


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
        # High, away from the panes' mirror direction to the camera.
        world.define(
            "Sun", ernie.DistantLight, intensity=2.0, angle=2.0
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.3, -0.3, 0.9)),
        )
        # A wall of 32 x 12 tiles of 0.25 m, 2.5 m behind.
        wall = world.define("Wall", ernie.Xform)
        for i in range(32):
            for j in range(12):
                wall.define(
                    f"T{i}_{j}",
                    ernie.Cube,
                    size=1.0,
                    translate=(
                        -4.0 + 0.25 * (i + 0.5),
                        2.5,
                        0.25 * (j + 0.5),
                    ),
                    scale=(0.25, 0.1, 0.25),
                    display_color=tile_colour(i, j),
                )

        self.glass = scene.define(
            "/World/Looks/Glass", ernie.Material, brdf="open_pbr"
        )
        self.glass.prim.set("inputs:specular_roughness", 0.0)
        self.glass.prim.set("inputs:specular_ior", IOR)
        self.glass.prim.set("inputs:transmission_weight", 1.0)
        world.define(
            "Ball",
            ernie.Sphere,
            radius=RADIUS,
            translate=BALL,
            material_binding=self.glass.path,
        )
        self.tinted = scene.define(
            "/World/Looks/Tinted", ernie.Material, brdf="open_pbr"
        )
        self.tinted.prim.set("inputs:specular_roughness", 0.0)
        self.tinted.prim.set("inputs:transmission_weight", 1.0)
        self.tinted.prim.set("inputs:transmission_color", TINTS[0][1])
        self.tinted.prim.set("inputs:geometry_thin_walled", 1.0)
        points, counts, indices = quad(PANE)
        world.define(
            "Pane",
            ernie.Mesh,
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            material_binding=self.tinted.path,
        )
        # Alpha blending: a white UsdPreviewSurface, partly opaque.
        look = ernie.Path("/World/Looks/Frosted")
        scene.define(look, ernie.Material)
        self.surface = scene.define(
            look.child("Surface"),
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.6, 0.6, 0.6),
            opacity=OPACITY,
        )
        scene.connect(
            look,
            tokens.SURFACE_OUTPUT,
            look.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        points, counts, indices = quad(PANEL)
        world.define(
            "Panel",
            ernie.Mesh,
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            material_binding=look,
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
            "Ball's IOR",
            1.0,
            2.5,
            IOR,
            on_change=lambda ior: self.glass.prim.set(
                "inputs:specular_ior", ior
            ),
        )
        self.widgets.checkbox(
            "Ball thin-walled",
            False,
            on_change=lambda thin: self.glass.prim.set(
                "inputs:geometry_thin_walled", 1.0 if thin else 0.0
            ),
        )
        self.widgets.slider(
            "Transmission",
            0.0,
            1.0,
            1.0,
            on_change=self.transmit,
        )
        self.widgets.combo(
            "Pane's colour",
            [name for name, _ in TINTS],
            0,
            on_change=lambda index: self.tinted.prim.set(
                "inputs:transmission_color", TINTS[index][1]
            ),
        )
        self.widgets.slider(
            "Panel's opacity",
            0.0,
            1.0,
            OPACITY,
            on_change=lambda opacity: setattr(
                self.surface, "opacity", opacity
            ),
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def transmit(self, weight):
        for look in (self.glass, self.tinted):
            look.prim.set("inputs:transmission_weight", weight)

    def render_now(self):
        return np.asarray(self.renderer.render(self.scene))

    def hidden(self):
        """The view without the ball, pane and panel."""
        for name in ("Ball", "Pane", "Panel"):
            self.scene.set(
                ernie.Path(f"/World/{name}"),
                tokens.VISIBILITY,
                tokens.INVISIBLE,
            )
        image = self.render_now()
        for name in ("Ball", "Pane", "Panel"):
            self.scene.set(
                ernie.Path(f"/World/{name}"),
                tokens.VISIBILITY,
                tokens.INHERITED,
            )
        return image

    def test_final(self):
        image = self.render_now()
        wall = self.hidden()
        self.glass.prim.set("inputs:specular_ior", 1.0)
        unbent = self.render_now()
        self.glass.prim.set("inputs:specular_ior", IOR)
        self.surface.opacity = 1.0
        opaque = self.render_now()
        self.surface.opacity = OPACITY

        # Most of the ball, which turns the wall over about its edge.
        behind_ball = window(wall, BALL, 55)
        bent = np.abs(window(image, BALL, 55) - behind_ball).mean()
        straight = np.abs(window(unbent, BALL, 55) - behind_ball).mean()
        pane = window(image, PANE).mean(axis=(0, 1))
        behind = window(wall, PANE).mean(axis=(0, 1))
        half = window(image, PANEL).mean()
        full = window(opaque, PANEL).mean()
        through = window(wall, PANEL).mean()
        print(
            f"Ball against the wall: refracted {bent:.1f}, at IOR 1 "
            f"{straight:.1f}; pane {pane.round(1)} over the wall "
            f"{behind.round(1)}; panel {half:.1f} between opaque "
            f"{full:.1f} and the wall {through:.1f}"
        )
        assert bent > 2.0 * straight + 10.0, "the glass ball refracts"
        assert pane[1] / pane[0] > 1.3 * behind[1] / behind[0], (
            "the pane tints the wall green"
        )
        assert pane[1] / pane[2] > 1.3 * behind[1] / behind[2], (
            "the pane tints the wall green"
        )
        low, high = sorted((full, through))
        assert low + 0.2 * (high - low) < half < high - 0.2 * (high - low), (
            "the half-opaque panel blends with the wall"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
