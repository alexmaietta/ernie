# SPDX-License-Identifier: BSD-3-Clause
"""Opens the viewer on a scene built in code, with a panel of its own that moves the sun.

    PYTHONPATH=build/python python demos/python/demo_viewer_sun_panel.py
"""

import math

import ernie
from ernie import tokens
from ernie.app import Dock, Viewer


def build_scene():
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    for i in range(5):
        scene.define(f"/World/Box{i}", tokens.CUBE).set(tokens.SIZE, 1.0 + 0.25 * i).set(
            tokens.TRANSFORM,
            [[1, 0, 0, 3.0 * (i - 2)], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        ).set(tokens.DISPLAY_COLOR, (0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i))
    scene.define("/World/Sun", tokens.DISTANT_LIGHT)
    return scene


def main():
    viewer = Viewer(build_scene(), title="Ernie demo")
    sun = {"elevation": 45.0, "azimuth": 30.0}
    panel = viewer.add_panel("Sun", Dock.LEFT)
    panel.slider("Elevation", 1.0, 90.0, sun["elevation"],
                 on_change=lambda value: sun.update(elevation=value))
    panel.slider("Azimuth", 0.0, 360.0, sun["azimuth"],
                 on_change=lambda value: sun.update(azimuth=value))

    def aim_sun(v, info):
        e, a = math.radians(sun["elevation"]), math.radians(sun["azimuth"])
        direction = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
        v.scene.set("/World/Sun", tokens.TRANSFORM, ernie.distant_light_transform(direction))

    viewer.on_pre_render(aim_sun)
    viewer.run()


if __name__ == "__main__":
    main()
