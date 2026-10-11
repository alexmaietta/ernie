# SPDX-License-Identifier: BSD-3-Clause
"""A scene built in code with a custom panel of options: every kind of
widget the Demos panel offers, each changing the scene."""

import math

import ernie
import ernie.demo
from ernie import tokens

BOXES = 5
# How far the boxes rise and fall when bobbing, in metres.
BOB = 0.5


def box_colour(i):
    return (0.2 + 0.15 * i, 0.4, 0.8 - 0.15 * i)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane)
        self.boxes = [
            world.define(
                f"Box{i}",
                ernie.Cube,
                size=1.0 + 0.25 * i,
                translate=(3.0 * (i - 2), 0, 1),
                display_color=box_colour(i),
            )
            for i in range(BOXES)
        ]
        self.sun = world.define("Sun", ernie.DistantLight)
        # The viewer frames the scene with its own camera; this one
        # renders it without a viewer and for --test.
        world.define("Camera", ernie.Camera, focal_length=24.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -14.0, 6.0), (0.0, 0.0, 0.5)
            ),
        )
        self.elevation = 45.0
        self.azimuth = 30.0
        self.aim_sun()
        # Each box's colour and visibility, and its height over the
        # last frames.
        self.colours = [box_colour(i) for i in range(BOXES)]
        self.shown = [True] * BOXES
        self.selected = 0
        self.bobbing = False
        self.heights: list[float] = []
        # A panel of the demo's own, docked beside the viewport; the
        # Demos panel's options without a viewer.
        if viewer is not None:
            self.panel = viewer.add_panel("Custom panel")
            viewer.focus_panel("Custom panel")
        else:
            self.panel = self.widgets
        self.add_widgets()

    def add_widgets(self):
        w = self.panel
        # Text, and a field whose text it shows.
        self.caption = w.text("Five boxes under a sun.")
        w.input_text(
            "Caption",
            "Five boxes under a sun.",
            on_change=lambda text: w.set(self.caption, text),
        )
        w.separator()
        # Sliders, and a button that sets them back.
        self.elevation_slider = w.slider(
            "Elevation",
            1.0,
            90.0,
            self.elevation,
            on_change=lambda value: self.aim_sun(elevation=value),
        )
        self.azimuth_slider = w.slider(
            "Azimuth",
            0.0,
            360.0,
            self.azimuth,
            on_change=lambda value: self.aim_sun(azimuth=value),
        )
        w.button("Reset the sun", self.reset_sun)
        w.separator()
        # A drop-down of the boxes, and the chosen one's colour and
        # visibility.
        w.combo(
            "Box",
            [f"Box {i}" for i in range(BOXES)],
            self.selected,
            on_change=self.select,
        )
        self.colour = w.color(
            "Colour", box_colour(self.selected), on_change=self.paint
        )
        self.visible = w.checkbox(
            "Visible", True, on_change=self.show
        )
        w.separator()
        # A checkbox that animates the boxes, and a plot of the
        # chosen box's height.
        w.checkbox("Bob", self.bobbing, on_change=self.set_bobbing)
        w.plot("Height", lambda: self.heights)

    def aim_sun(self, elevation=None, azimuth=None):
        if elevation is not None:
            self.elevation = elevation
        if azimuth is not None:
            self.azimuth = azimuth
        e = math.radians(self.elevation)
        a = math.radians(self.azimuth)
        direction = (
            math.cos(e) * math.cos(a),
            math.cos(e) * math.sin(a),
            math.sin(e),
        )
        self.sun.set(
            tokens.TRANSFORM, ernie.distant_light_transform(direction)
        )

    def reset_sun(self):
        self.aim_sun(elevation=45.0, azimuth=30.0)
        self.panel.set(self.elevation_slider, self.elevation)
        self.panel.set(self.azimuth_slider, self.azimuth)

    def select(self, index):
        self.selected = index
        self.panel.set(self.colour, self.colours[index])
        self.panel.set(self.visible, self.shown[index])
        self.heights = []

    def paint(self, rgb):
        self.colours[self.selected] = tuple(rgb)
        self.boxes[self.selected].display_color = rgb

    def show(self, visible):
        self.shown[self.selected] = visible
        self.boxes[self.selected].visibility = (
            tokens.INHERITED if visible else tokens.INVISIBLE
        )

    def set_bobbing(self, bobbing):
        self.bobbing = bobbing

    def step(self):
        height = 1.0
        for i, box in enumerate(self.boxes):
            z = 1.0
            if self.bobbing:
                z += BOB * math.sin(2.0 * self.time + i)
                box.set(translate=(3.0 * (i - 2), 0.0, z))
            if i == self.selected:
                height = z
        # The last 240 heights, for the plot.
        self.heights = (self.heights + [height])[-240:]

    def test_final(self):
        image = self.render_image()[..., :3]
        assert image.std() > 1.0, (
            "the boxes and ground should not render flat"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
