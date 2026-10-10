# SPDX-License-Identifier: BSD-3-Clause
"""Opens the viewer on a scene built in code, with a panel of its own that moves the sun.

    PYTHONPATH=build/python python demos/python/demo_viewer_sun_panel.py
"""

import math

import ernie
import ernie.demo
from ernie import tokens


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane)
        for i in range(5):
            world.define(f"Box{i}", ernie.Cube, size=1.0 + 0.25 * i,
                         translate=(3.0 * (i - 2), 0, 1),
                         display_color=(0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i))
        self.sun = world.define("Sun", ernie.DistantLight)
        self.elevation = 45.0
        self.azimuth = 30.0
        self.aim_sun()
        if viewer is not None:
            from ernie.app import Dock

            panel = viewer.add_panel("Sun", Dock.LEFT)
            panel.slider("Elevation", 1.0, 90.0, self.elevation,
                         on_change=lambda value: self.aim_sun(elevation=value))
            panel.slider("Azimuth", 0.0, 360.0, self.azimuth,
                         on_change=lambda value: self.aim_sun(azimuth=value))

    def aim_sun(self, elevation=None, azimuth=None):
        self.elevation = self.elevation if elevation is None else elevation
        self.azimuth = self.azimuth if azimuth is None else azimuth
        e, a = math.radians(self.elevation), math.radians(self.azimuth)
        direction = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
        self.sun.set(tokens.TRANSFORM, ernie.distant_light_transform(direction))

    def test_final(self):
        image = self.render_image()[..., :3]
        assert image.std() > 1.0, "the boxes and ground should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
