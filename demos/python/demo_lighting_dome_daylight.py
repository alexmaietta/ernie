# SPDX-License-Identifier: BSD-3-Clause
"""Daylight from a dome light of 100,000 lux (photometric:illuminance) seen through a camera set by
the "sunny 16" rule (ISO 100, 1/100 s, f/16), after the Omniverse RTX physical lighting page's
"dome-light-100k-lx" figure: a house with a pool and deck among trees.

With physical units throughout, the exposure needs no tuning: under 100,000 lux a white wall of
albedo 0.8 has a luminance of 0.8 x 100000 / pi, about 25,000 cd/m^2, and the sunny 16 camera's
exposure scale (1 x 0.01 x 1 / 16^2) brings it just under white.

    PYTHONPATH=build/python python demos/python/demo_lighting_dome_daylight.py
    PYTHONPATH=build/python python demos/python/demo_lighting_dome_daylight.py --viewer null \
        --output daylight.png

The dome holds a made-up sky map with the sun in it, whose photometric illuminance scales it to
100,000 lux. Ernie's dome light shades diffusely and casts no shadows, so unlike the page's image
nothing here shadows the ground; a DistantLight sun would.

Reference: NVIDIA, "Omniverse RTX Renderer: Lighting, Physical Lighting",
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIZE = (900, 520)
EYE = np.array([-16.0, -24.0, 20.0])
TARGET = np.array([1.0, -1.0, 1.0])
FOCAL_LENGTH = 24.0


def sky_map(rows=64, cols=128, sun=(-0.45, -0.55, 0.7)):
    """A latitude-longitude sky (rows from +z down to -z): blue sky above, brightening towards
    the horizon, a bright sun region around `sun`, and dim ground light below. The dome's
    photometric illuminance scales it, so only its shape matters."""
    theta = (np.arange(rows) + 0.5) / rows * np.pi
    phi = (np.arange(cols) + 0.5) / cols * 2.0 * np.pi
    t, p = np.meshgrid(theta, phi, indexing="ij")
    direction = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
    up = direction[..., 2]
    sky = np.where(up[..., None] > 0.0,
                   np.array([0.45, 0.6, 1.0]) * (0.6 + 0.4 * (1.0 - up[..., None])),
                   np.array([0.12, 0.1, 0.08]))
    toward = np.asarray(sun) / np.linalg.norm(sun)
    glow = np.exp((direction @ toward - 1.0) / 0.02)[..., None]
    return (sky + 60.0 * glow * np.array([1.0, 0.95, 0.85])).astype(np.float32)


def project(point, aperture_mm=15.2908):
    """The pixel (col, row) the camera images the world point at."""
    forward = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = np.asarray(point, dtype=np.float64) - EYE
    focal = FOCAL_LENGTH / aperture_mm * SIZE[1]
    depth = d @ forward
    return (SIZE[0] / 2 + focal * (d @ right) / depth, SIZE[1] / 2 - focal * (d @ up) / depth)


class Demo(ernie.demo.Demo):
    num_frames = 1
    title = "Daylight dome, 100,000 lux"

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        looks = scene.define("/Looks", ernie.Scope)

        def material(name, albedo):
            return looks.define(name, ernie.Material, albedo=albedo)

        grass = material("DryGrass", (0.22, 0.16, 0.09))
        white = material("Render", (0.8, 0.8, 0.78))
        wood = material("Deck", (0.32, 0.18, 0.1))
        water = material("Water", (0.15, 0.55, 0.6))
        paving = material("Paving", (0.55, 0.53, 0.5))
        leaves = material("Leaves", (0.08, 0.2, 0.06))
        autumn = material("Autumn", (0.35, 0.2, 0.05))
        bark = material("Bark", (0.15, 0.1, 0.07))

        world = scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, material_binding=grass)

        def cube(path, centre, size, look):
            world.define(path, ernie.Cube, size=1.0, translate=centre, scale=size,
                         material_binding=look)

        # The house: two stacked white blocks with a terrace, beside a deck, a pool and a patio.
        cube("House/Ground", (0.0, 4.0, 1.6), (10.0, 7.0, 3.2), white)
        cube("House/Upper", (1.5, 5.0, 4.6), (7.0, 5.0, 2.8), white)
        cube("Deck", (0.0, -4.5, 0.1), (12.0, 10.0, 0.2), wood)
        cube("Pool", (0.5, -4.0, 0.21), (5.0, 7.0, 0.02), water)
        cube("Patio", (-4.5, -8.0, 0.21), (4.0, 4.0, 0.02), paving)
        for k, (x, y) in enumerate([(-6.5, 3.0), (6.8, 1.0), (8.0, -6.0), (-8.0, -3.5),
                                    (-3.0, 10.0), (7.0, 9.0), (-9.0, 8.0), (10.0, 3.5),
                                    (-10.5, -9.0), (3.0, -12.0)]):
            tree = world.define(f"Trees/T{k}", ernie.Xform)
            tree.define("Trunk", ernie.Cylinder, radius=0.15, height=2.0, translate=(x, y, 1.0),
                        material_binding=bark)
            tree.define("Crown", ernie.Sphere, radius=1.4 + 0.3 * (k % 3), translate=(x, y, 3.0),
                        material_binding=autumn if k % 4 == 1 else leaves)

        # 100,000 lux on the ground: bright daylight, from a sky with the sun in it.
        world.define("Sky", ernie.DomeLight, photometric_illuminance=100000.0, texture=sky_map())
        # Sunny 16: ISO 100, 1/100 s, f/16.
        camera = world.define("Camera", ernie.Camera, focal_length=FOCAL_LENGTH,
                              exposure_iso=100.0, exposure_time=0.01, exposure_f_stop=16.0,
                              sensor_noise=False).set(
            tokens.TRANSFORM, ernie.look_at_transform(tuple(EYE), tuple(TARGET)))
        scene.define("/Render/Settings", ernie.RenderSettings, camera=camera, resolution=SIZE)
        if viewer is not None:
            viewer.active_camera = camera.path

    def test_final(self):
        image = self.render_image()[..., :3].astype(np.float64)
        # The upper block's flat roof faces the whole dome: 100,000 lux on albedo 0.8.
        col, row = (int(round(v)) for v in project((1.5, 5.0, 6.0)))
        roof = image[row - 3:row + 4, col - 3:col + 4].mean(axis=(0, 1))
        print(f"white roof under 100,000 lux: {roof.round(1)}")
        assert roof.min() > 200.0, "the sunny 16 white roof should be near white"
        # Albedo 0.8 of 100,000 lux is 25,000 cd/m^2, 0.995 of the sunny 16 camera's white: at
        # least one channel stays short of clipping.
        assert roof.min() < 254.5, "the sunny 16 white roof should stay below white"
        ground = image[-20:, :].mean()
        assert ground < roof.mean() * 0.7, "the dry grass should be darker than the white roof"


if __name__ == "__main__":
    ernie.demo.main(Demo)
