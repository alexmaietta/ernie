# SPDX-License-Identifier: BSD-3-Clause
"""Ripples spreading over a pond by the wave equation, the height
field logged each step as a live heatmap (log_array)."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIZE = 64  # heights a side
CELL = 0.25  # metres between heights
COURANT = 0.2  # c^2, in cells per step squared: stable below 0.5
DAMPING = 0.995


def drop(size, row, col, width=4.0, amplitude=1.0):
    """A Gaussian bump of `amplitude` metres centred on (row, col),
    `width` cells across."""
    rows, cols = np.mgrid[0:size, 0:size]
    return amplitude * np.exp(
        -((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * width**2)
    )


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        half = 0.5 * (SIZE - 1) * CELL
        self.terrain = world.define(
            "Pond",
            ernie.Terrain,
            cell_size=CELL,
            origin=(-half, half),
            display_color=(0.25, 0.45, 0.7),
        )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.5, -0.4, 0.6)),
        )
        world.define("Camera", ernie.Camera, focal_length=20.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -16.0, 11.0), (0.0, 0.0, 0.0)
            ),
        )
        self.heights = drop(SIZE, SIZE // 2, SIZE // 2)
        self.previous = self.heights.copy()
        self.damping = DAMPING
        self.write_heights()
        self.widgets.slider(
            "Damping",
            0.95,
            1.0,
            DAMPING,
            on_change=lambda value: setattr(self, "damping", value),
        )
        self.widgets.button(
            "Drop", lambda: self.add_drop(SIZE // 3, 2 * SIZE // 3)
        )

    def add_drop(self, row, col):
        self.heights = self.heights + drop(SIZE, row, col)

    def write_heights(self):
        # A new array each time: the scene keeps the one written.
        self.scene.write(
            self.terrain.path,
            tokens.HEIGHTS,
            self.heights.astype(np.float32),
        )

    def step(self):
        h = self.heights
        laplacian = np.zeros_like(h)
        laplacian[1:-1, 1:-1] = (
            h[2:, 1:-1]
            + h[:-2, 1:-1]
            + h[1:-1, 2:]
            + h[1:-1, :-2]
            - 4.0 * h[1:-1, 1:-1]
        )
        following = (
            2.0 * h - self.previous + COURANT * laplacian
        ) * self.damping
        following[0, :] = following[-1, :] = following[
            :, 0
        ] = following[:, -1] = 0.0
        self.previous, self.heights = h, following
        self.write_heights()
        self.log_array("pond/heights", self.heights)
        self.log_array(
            "pond/row energy", (self.heights**2).sum(axis=1)
        )

    def test_final(self):
        heights = self.logged_array("pond/heights")
        energy = self.logged_array("pond/row energy")
        assert heights is not None and energy is not None
        assert heights.shape == (SIZE, SIZE) and energy.shape == (
            SIZE,
        )
        assert np.isfinite(heights).all()
        # The drop spread: the centre fell, and ripples remain.
        assert abs(heights[SIZE // 2, SIZE // 2]) < 0.5, (
            "the drop should have spread"
        )
        assert np.abs(heights).max() > 1e-3, "ripples should remain"
        assert math.isclose(
            float(energy.sum()),
            float((heights**2).sum()),
            rel_tol=1e-9,
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
