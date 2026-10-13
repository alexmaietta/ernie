# SPDX-License-Identifier: BSD-3-Clause
"""OpenPBR's thin film: iridescence from a film a fraction of a micron
thick on the base's specular (thin_film_weight, thin_film_thickness,
thin_film_ior), like the colours of heat-tinted steel, anodized titanium
or oil on black glass. The balls' films thicken from left to right
through the interference colours, which shift with the angle of view.

References:
- J. Andersson, A. Portsmouth et al., "OpenPBR Surface Specification",
  v1.1, 2024 (the thin film).
- L. Belcour and P. Barla, "A Practical Extension to Microfacet Theory
  for the Modeling of Varying Iridescence", ACM Transactions on Graphics
  36(4), 2017.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

RESOLUTION = (800, 320)
FOCAL = 50.0  # mm
APERTURE = 36.0  # mm, horizontal
EYE = (0.0, -7.0, 1.2)
TARGET = (0.0, 0.0, 0.45)
RADIUS = 0.4
THICKNESS = 0.25  # um, the middle ball's film
# Each ball's film over the middle one's.
SCALES = (0.4, 0.7, 1.0, 1.3, 1.6)
FILM_IOR = 2.0
# Name, base colour, metalness and specular roughness.
BASES = (
    ("Steel", (0.56, 0.57, 0.58), 1.0, 0.15),
    ("Titanium", (0.54, 0.50, 0.45), 1.0, 0.15),
    ("Black glass", (0.01, 0.01, 0.01), 0.0, 0.05),
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


def disc_pixels(image, point):
    """The RGB (0 to 255) of the pixels on the ball at `point`."""
    col, row, scale = project(point)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(cols + 0.5 - col, rows + 0.5 - row) / (scale * RADIUS)
    return image[..., :3][r < 0.95].astype(np.float64)


def chroma(pixels):
    """The mean of each pixel's largest channel less its smallest: how
    coloured they look."""
    return float((pixels.max(axis=1) - pixels.min(axis=1)).mean())


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
        for i in range(len(SCALES)):
            look = scene.define(
                f"/World/Looks/Film{i}", ernie.Material, brdf="open_pbr"
            )
            look.prim.set("inputs:thin_film_weight", 1.0)
            look.prim.set("inputs:thin_film_ior", FILM_IOR)
            self.looks.append(look)
            world.define(
                f"Ball{i}",
                ernie.Sphere,
                radius=RADIUS,
                translate=ball_centre(i),
                material_binding=look.path,
            )
        self.base(0)
        self.thicken(THICKNESS)
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

        self.widgets.combo(
            "Base", [name for name, *_ in BASES], 0, on_change=self.base
        )
        self.widgets.slider(
            "Middle ball's film (um)",
            0.0,
            1.0,
            THICKNESS,
            on_change=self.thicken,
        )
        self.widgets.slider(
            "Film IOR",
            1.0,
            2.5,
            FILM_IOR,
            on_change=lambda ior: self.film("thin_film_ior", ior),
        )
        self.widgets.slider(
            "Film weight",
            0.0,
            1.0,
            1.0,
            on_change=lambda weight: self.film("thin_film_weight", weight),
        )
        if viewer is not None:
            viewer.active_camera = camera.path
            viewer.look_at(EYE, TARGET)

    def film(self, name, value):
        """Sets input `name` on every ball."""
        for look in self.looks:
            look.prim.set(f"inputs:{name}", value)

    def thicken(self, thickness):
        for look, scale in zip(self.looks, SCALES):
            look.prim.set("inputs:thin_film_thickness", scale * thickness)

    def base(self, index):
        _, colour, metalness, roughness = BASES[index]
        for look in self.looks:
            look.albedo = colour
            look.prim.set("inputs:base_metalness", metalness)
            look.prim.set("inputs:specular_roughness", roughness)

    def balls(self):
        image = np.asarray(self.renderer.render(self.scene))
        return [disc_pixels(image, ball_centre(i)) for i in range(len(SCALES))]

    def test_final(self):
        filmed = self.balls()
        self.film("thin_film_weight", 0.0)
        bare = self.balls()
        self.film("thin_film_weight", 1.0)
        means = [pixels.mean(axis=0) for pixels in filmed]
        for i, (f, b, mean) in enumerate(zip(filmed, bare, means)):
            print(
                f"Film {SCALES[i] * THICKNESS:.2f} um: chroma {chroma(f):.1f}"
                f" (bare {chroma(b):.1f}), mean RGB {mean.round(1)}"
            )
        for f, b in zip(filmed, bare):
            assert chroma(f) > chroma(b) + 10.0, "the film colours the steel"
        hues = np.array([mean / mean.sum() for mean in means])
        assert np.abs(hues[1:] - hues[:-1]).sum(axis=1).min() > 0.03, (
            "each thickness has a colour of its own"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
