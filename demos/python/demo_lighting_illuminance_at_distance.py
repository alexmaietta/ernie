# SPDX-License-Identifier: BSD-3-Clause
"""A rect light 5 m above a floor, set to give 1000 lux at 5 m (photometric:illuminance and
photometric:illuminance:distance), after the Omniverse RTX physical lighting page's
"illuminance-at-distance" figure.

The floor is a grey Lambertian surface of albedo 0.5, so its luminance L (HdrColor, in cd/m^2)
gives the illuminance falling on it: E = pi L / albedo. The demo measures that illuminance and
shows it as a heatmap; directly below the light it reads close to 1000 lux (a little less, as the
light is a 1 m square, not a point).

    PYTHONPATH=build/python python demos/python/demo_lighting_illuminance_at_distance.py
    PYTHONPATH=build/python python demos/python/demo_lighting_illuminance_at_distance.py \
        --output illuminance.png

Reference: NVIDIA, "Omniverse RTX Renderer: Lighting, Physical Lighting",
https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
"""

import argparse
import math

import numpy as np

import ernie
from ernie import tokens

from demo_support import heatmap, side_by_side, write_png

ALBEDO = 0.5
SIZE = (640, 400)
EYE = np.array([0.0, -16.0, 11.0])


def build_scene():
    scene = ernie.Scene()
    grey = scene.define("/Looks/Grey", ernie.Material, albedo=(ALBEDO, ALBEDO, ALBEDO))
    scene.define("/World/Floor", ernie.GroundPlane, material_binding=grey)
    # 1 m square, 5 m up, facing down: 1000 lux on a surface facing it 5 m away.
    scene.define("/World/Panel", ernie.RectLight, width=1.0, height=1.0,
                 photometric_illuminance=1000.0, photometric_illuminance_distance=5.0,
                 shadow_enable=False).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    # Exposed so that the floor under the light (about 160 cd/m^2) is a light grey.
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=24.0, sensor_noise=False,
                          exposure_time=1.0 / 60.0, exposure_iso=200.0, exposure_f_stop=2.8).set(
        tokens.TRANSFORM, ernie.look_at_transform(tuple(EYE), (0.0, 0.0, 0.0))
    )
    render = scene.define("/Render", ernie.Scope)
    render_vars = [render.define(f"Vars/{name}", ernie.RenderVar, source_name=name).path
                   for name in ("HdrColor", "LdrColor")]
    product = render.define("Product", ernie.RenderProduct, camera=camera, resolution=SIZE).set(
        tokens.ORDERED_VARS, render_vars
    )
    render.define("Settings", ernie.RenderSettings, camera=camera, resolution=SIZE,
                  ambient=0.0).set(tokens.PRODUCTS, [product.path])
    return scene


def measure(renderer, scene):
    """The floor's illuminance in lux per pixel, and the photographic image."""
    frame = renderer.step(scene)["/Render/Product"].frames[0]
    hdr = np.array(frame.render_vars["/Render/Vars/HdrColor"].map())
    ldr = np.array(frame.render_vars["/Render/Vars/LdrColor"].map())
    luminance = hdr[..., 0] * 0.2126 + hdr[..., 1] * 0.7152 + hdr[..., 2] * 0.0722
    return math.pi * luminance / ALBEDO, ldr


def project(point, focal_mm=24.0, aperture_mm=15.2908):
    """The pixel (col, row) the camera images the world point at."""
    forward = -EYE / np.linalg.norm(EYE)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = np.asarray(point, dtype=np.float64) - EYE
    focal = focal_mm / aperture_mm * SIZE[1]
    depth = d @ forward
    return (SIZE[0] / 2 + focal * (d @ right) / depth, SIZE[1] / 2 - focal * (d @ up) / depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write the heatmap and image here instead of a window")
    args = parser.parse_args()

    renderer = ernie.Renderer()
    scene = build_scene()
    lux, ldr = measure(renderer, scene)
    col, row = (int(round(v)) for v in project((0.0, 0.0, 0.0)))
    below = float(lux[row, col])
    print(f"illuminance below the light: {below:.0f} lux (asked for 1000 lux at 5 m)")

    if args.output:
        heat = heatmap(lux, 0.0, lux.max())
        write_png(args.output, side_by_side([heat, ldr[..., :3]]))
        print(f"wrote {args.output}: heatmap from 0 to {lux.max():.0f} lux, and the image")
        return

    # In the viewer, the measured illuminance labels points of the floor.
    from ernie.app import Viewer

    labels = scene.overlay("illuminance")
    for x, y in [(0, 0), (2, 0), (4, 0), (0, 3), (-3, -3), (6, 4)]:
        c, r = (int(round(v)) for v in project((x, y, 0.0)))
        if 0 <= c < SIZE[0] and 0 <= r < SIZE[1]:
            labels.marker((x, y, 0.0), shape="cross", size=10, color=(1.0, 0.3, 0.2))
            labels.label((x, y, 0.0), f"{lux[r, c]:.0f} lx", size=14)
    viewer = Viewer(scene, title="Illuminance at distance")
    viewer.active_camera = ernie.Path("/World/Camera")
    viewer.run()


if __name__ == "__main__":
    main()
