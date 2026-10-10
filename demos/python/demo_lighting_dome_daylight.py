# SPDX-License-Identifier: BSD-3-Clause
"""Daylight from a dome light of 100,000 lux (photometric:illuminance) seen through a camera set by
the "sunny 16" rule (ISO 100, 1/100 s, f/16), after the Omniverse RTX physical lighting page's
"dome-light-100k-lx" figure: a house with a pool and deck among trees.

With physical units throughout, the exposure needs no tuning: under 100,000 lux a white wall of
albedo 0.8 has a luminance of 0.8 x 100000 / pi, about 25,000 cd/m^2, and the sunny 16 camera's
exposure scale (1 x 0.01 x 1 / 16^2) brings it just under white.

    PYTHONPATH=build/python python demos/python/demo_lighting_dome_daylight.py
    PYTHONPATH=build/python python demos/python/demo_lighting_dome_daylight.py --output daylight.png

The dome holds a made-up sky map with the sun in it, whose photometric illuminance scales it to
100,000 lux. Ernie's dome light shades diffusely and casts no shadows, so unlike the page's image
nothing here shadows the ground; a DistantLight sun would.

Reference: NVIDIA, "Omniverse RTX Renderer: Lighting, Physical Lighting",
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
"""

import argparse

import numpy as np

import ernie
from ernie import tokens

from demo_support import write_png

SIZE = (900, 520)


def placed(x, y, z, sx=1.0, sy=1.0, sz=1.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


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


def material(scene, name, albedo):
    path = ernie.Path(f"/Looks/{name}")
    scene.define(path, tokens.MATERIAL).set(tokens.ALBEDO, albedo)
    return path


def build_scene():
    scene = ernie.Scene()
    grass = material(scene, "DryGrass", (0.22, 0.16, 0.09))
    white = material(scene, "Render", (0.8, 0.8, 0.78))
    wood = material(scene, "Deck", (0.32, 0.18, 0.1))
    water = material(scene, "Water", (0.15, 0.55, 0.6))
    paving = material(scene, "Paving", (0.55, 0.53, 0.5))
    leaves = material(scene, "Leaves", (0.08, 0.2, 0.06))
    autumn = material(scene, "Autumn", (0.35, 0.2, 0.05))
    bark = material(scene, "Bark", (0.15, 0.1, 0.07))

    scene.define("/World/Ground", tokens.GROUND_PLANE).set(tokens.MATERIAL_BINDING, grass)

    def cube(path, transform, look):
        scene.define(path, tokens.CUBE).set(tokens.SIZE, 1.0).set(
            tokens.TRANSFORM, transform
        ).set(tokens.MATERIAL_BINDING, look)

    # The house: two stacked white blocks with a terrace, beside a deck, a pool and a patio.
    cube("/World/House/Ground", placed(0.0, 4.0, 1.6, 10.0, 7.0, 3.2), white)
    cube("/World/House/Upper", placed(1.5, 5.0, 4.6, 7.0, 5.0, 2.8), white)
    cube("/World/Deck", placed(0.0, -4.5, 0.1, 12.0, 10.0, 0.2), wood)
    cube("/World/Pool", placed(0.5, -4.0, 0.21, 5.0, 7.0, 0.02), water)
    cube("/World/Patio", placed(-4.5, -8.0, 0.21, 4.0, 4.0, 0.02), paving)
    for k, (x, y) in enumerate([(-6.5, 3.0), (6.8, 1.0), (8.0, -6.0), (-8.0, -3.5), (-3.0, 10.0),
                                (7.0, 9.0), (-9.0, 8.0), (10.0, 3.5), (-10.5, -9.0), (3.0, -12.0)]):
        scene.define(f"/World/Trees/T{k}/Trunk", tokens.CYLINDER).set(tokens.GEOM_RADIUS, 0.15).set(
            tokens.GEOM_HEIGHT, 2.0
        ).set(tokens.TRANSFORM, placed(x, y, 1.0)).set(tokens.MATERIAL_BINDING, bark)
        scene.define(f"/World/Trees/T{k}/Crown", tokens.SPHERE).set(
            tokens.GEOM_RADIUS, 1.4 + 0.3 * (k % 3)
        ).set(tokens.TRANSFORM, placed(x, y, 3.0)).set(
            tokens.MATERIAL_BINDING, autumn if k % 4 == 1 else leaves
        )

    # 100,000 lux on the ground: bright daylight, from a sky with the sun in it.
    scene.define("/World/Sky", tokens.DOME_LIGHT).set(
        tokens.PHOTOMETRIC_ILLUMINANCE, 100000.0
    ).write(tokens.LIGHT_TEXTURE, sky_map())
    # Sunny 16: ISO 100, 1/100 s, f/16.
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((-16.0, -24.0, 20.0), (1.0, -1.0, 1.0))
    ).set(tokens.FOCAL_LENGTH, 24.0).set(tokens.EXPOSURE_ISO, 100.0).set(
        tokens.EXPOSURE_TIME, 0.01
    ).set(tokens.EXPOSURE_FSTOP, 16.0).set(tokens.SENSOR_NOISE, False)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.RESOLUTION, SIZE)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write the image here instead of opening a window")
    args = parser.parse_args()
    scene = build_scene()
    if args.output:
        image = np.array(ernie.Renderer().render(scene))
        write_png(args.output, image[..., :3])
        print(f"wrote {args.output}")
        return
    from ernie.app import Viewer

    viewer = Viewer(scene, title="Daylight dome, 100,000 lux")
    viewer.active_camera = ernie.Path("/World/Camera")
    viewer.run()


if __name__ == "__main__":
    main()
