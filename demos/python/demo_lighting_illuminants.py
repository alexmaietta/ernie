# SPDX-License-Identifier: BSD-3-Clause
"""One spot light, four illuminants (physical:illuminant), after the Omniverse RTX physical
lighting page's illuminant figure: "white", "illuminant D" at 4000 K, "blackbody" at 2000 K and a
"custom" fluorescent spectrum, each at the same 1600 lumens (photometric:power), lighting a
checkered wall with a colour chart and two spheres in a dark room.

The lumens stay the same, so each image has about the same brightness; the illuminant changes
only the colour of the light, and how the chart's patches render under it.

    PYTHONPATH=build/python python demos/python/demo_lighting_illuminants.py
    PYTHONPATH=build/python python demos/python/demo_lighting_illuminants.py --output illuminants.png

The page's custom illuminant is CIE F5; here the custom spectrum is fluorescent-like rather than F5
itself: mercury emission lines at 405, 436, 546 and 578 nm over a broad halophosphate band.

References:
- NVIDIA, "Omniverse RTX Renderer: Lighting, Physical Lighting",
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
- D. Pascale, "RGB coordinates of the Macbeth ColorChecker", BabelColor, 2006 (the patches' sRGB
  values).
"""

import argparse

import numpy as np

import ernie
from ernie import tokens

from demo_support import side_by_side, srgb_to_linear, write_png

SIZE = (300, 448)

# The ColorChecker's 24 patches in sRGB, row by row from dark skin to black.
CHART = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177),
    (103, 189, 170), (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108),
    (157, 188, 64), (224, 163, 46), (56, 61, 150), (70, 148, 73), (175, 54, 60),
    (231, 199, 31), (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]

ILLUMINANTS = [
    ("white", tokens.WHITE, 6500.0),
    ("illuminant D 4000 K", tokens.ILLUMINANT_D, 4000.0),
    ("blackbody 2000 K", tokens.BLACKBODY, 2000.0),
    ("custom (fluorescent)", tokens.CUSTOM, 6500.0),
]


def fluorescent():
    """(nm, value) pairs of a fluorescent-like spectrum: mercury lines over broad yellow and blue
    phosphor bands, weighted to CIE F5's white point (x 0.311, y 0.337 against F5's 0.314,
    0.335)."""
    wavelengths = np.arange(380.0, 781.0, 1.0)
    band = np.exp(-0.5 * ((wavelengths - 570.0) / 80.0) ** 2)
    band += 0.8 * np.exp(-0.5 * ((wavelengths - 470.0) / 30.0) ** 2)
    for line, weight in [(404.7, 0.8), (435.8, 2.0), (546.1, 0.5), (578.0, 0.2)]:
        band += weight * np.exp(-0.5 * ((wavelengths - line) / 1.5) ** 2)
    return np.stack([wavelengths, band], axis=1).astype(np.float32).ravel().tolist()


def box(scene, path, centre, size, colour):
    m = np.eye(4)
    m[:3, 3] = centre
    m[0, 0], m[1, 1], m[2, 2] = size
    scene.define(path, tokens.CUBE).set(tokens.SIZE, 1.0).set(tokens.TRANSFORM, m).set(
        tokens.DISPLAY_COLOR, tuple(float(c) for c in colour)
    )


def build_scene():
    scene = ernie.Scene()
    scene.define("/World/Floor", tokens.GROUND_PLANE).set(tokens.CHECKER, True).set(
        tokens.CHECKER_SIZE, 0.25
    ).set(tokens.CHECKER_COLOR0, (0.35, 0.35, 0.35)).set(tokens.CHECKER_COLOR1, (0.2, 0.2, 0.2))
    # A checkered back wall 3 m wide and 3 m high at y = 1.
    for i in range(12):
        for j in range(12):
            grey = 0.32 if (i + j) % 2 == 0 else 0.2
            box(scene, f"/World/Wall/T{i}_{j}", (-1.5 + 0.25 * (i + 0.5), 1.05, 0.25 * (j + 0.5)),
                (0.25, 0.1, 0.25), (grey, grey, grey))
    # The colour chart on the wall, 6 x 4 patches of 10 cm on a black card.
    box(scene, "/World/Chart/Card", (-0.25, 0.99, 1.3), (0.68, 0.01, 0.47), (0.03, 0.03, 0.03))
    for k, rgb in enumerate(CHART):
        col, row = k % 6, k // 6
        box(scene, f"/World/Chart/P{k}", (-0.525 + 0.11 * col, 0.98, 1.465 - 0.11 * row),
            (0.1, 0.02, 0.1), srgb_to_linear(np.array(rgb) / 255.0))
    scene.define("/World/Ball", tokens.SPHERE).set(tokens.GEOM_RADIUS, 0.15).set(
        tokens.TRANSFORM, [[1, 0, 0, 0.25], [0, 1, 0, 0.6], [0, 0, 1, 0.15], [0, 0, 0, 1]]
    ).set(tokens.DISPLAY_COLOR, (0.5, 0.5, 0.5))
    scene.define("/World/Ball2", tokens.SPHERE).set(tokens.GEOM_RADIUS, 0.15).set(
        tokens.TRANSFORM, [[1, 0, 0, -0.1], [0, 1, 0, 0.55], [0, 0, 1, 0.15], [0, 0, 0, 1]]
    ).set(tokens.DISPLAY_COLOR, (0.8, 0.8, 0.8))
    # A spot near the ceiling at the left, grazing the wall: 1600 lumens.
    scene.define("/World/Spot", tokens.SPHERE_LIGHT).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((-1.2, 0.6, 2.9), (-0.25, 1.0, 1.0), (0.0, 1.0, 0.0)),
    ).set(tokens.RADIUS, 0.05).set(tokens.SHAPING_CONE_ANGLE, 40.0).set(
        tokens.SHAPING_CONE_SOFTNESS, 0.4
    ).set(tokens.PHOTOMETRIC_POWER, 1600.0).set(tokens.PHYSICAL_CUSTOM_ILLUMINANT, fluorescent())
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.0, -2.6, 1.3), (0.0, 1.0, 1.2))
    ).set(tokens.FOCAL_LENGTH, 18.0).set(tokens.SENSOR_NOISE, False).set(
        tokens.EXPOSURE_TIME, 1.0 / 15.0
    ).set(tokens.EXPOSURE_ISO, 400.0).set(tokens.EXPOSURE_FSTOP, 2.0)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.RESOLUTION, SIZE).set(tokens.AMBIENT, 0.0)
    return scene


def light_with(scene, illuminant, kelvin):
    scene.set("/World/Spot", tokens.PHYSICAL_ILLUMINANT, illuminant)
    scene.set("/World/Spot", tokens.COLOR_TEMPERATURE, kelvin)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write the four images side by side here")
    args = parser.parse_args()
    scene = build_scene()
    if args.output:
        renderer = ernie.Renderer()
        images = []
        for name, illuminant, kelvin in ILLUMINANTS:
            light_with(scene, illuminant, kelvin)
            images.append(np.array(renderer.render(scene))[..., :3])
            print(f"rendered {name}")
        write_png(args.output, side_by_side(images))
        print(f"wrote {args.output}")
        return

    # In the viewer, a panel switches the illuminant.
    from ernie.app import Dock, Viewer

    viewer = Viewer(scene, title="Illuminants")
    viewer.active_camera = ernie.Path("/World/Camera")
    panel = viewer.add_panel("Illuminant", Dock.RIGHT)
    light_with(scene, *ILLUMINANTS[0][1:])
    panel.combo("Illuminant", [name for name, _, _ in ILLUMINANTS], 0,
                on_change=lambda index: light_with(scene, *ILLUMINANTS[index][1:]))
    viewer.run()


if __name__ == "__main__":
    main()
