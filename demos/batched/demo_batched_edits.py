# SPDX-License-Identifier: BSD-3-Clause
"""Conway's Game of Life on 10,000 cubes: each generation recolours
only the cells that flipped, one prim at a time, and lifts one row of
them (its parent's transform), so a frame syncs only what changed."""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIDE = 100  # cells along each side
SPACING = 1.1  # m between cells
ALIVE = (0.2, 0.85, 0.3)
DEAD = (0.12, 0.12, 0.14)


def row_path(r):
    return f"/World/Board/Row{r}"


def cell_path(r, c):
    return f"{row_path(r)}/Cell{c}"


def soup():
    """The C++ twin's fixed random soup, a third alive."""
    alive = np.zeros((SIDE, SIDE), bool)
    state = 0x853C49E6748FEA9B
    for r in range(SIDE):
        for c in range(SIDE):
            state = (
                state * 6364136223846793005 + 1442695040888963407
            ) % (1 << 64)
            alive[r, c] = (state >> 33) % 3 == 0
    return alive


class Demo(ernie.demo.Demo):
    num_frames = 60

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.3, -0.4, 0.9)),
        )
        world.define("Camera", ernie.Camera, focal_length=18.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -95.0, 90.0), (0.0, 0.0, 0.0)
            ),
        )
        self.alive = soup()
        half = 0.5 * (SIDE - 1) * SPACING
        for r in range(SIDE):
            scene.define(row_path(r), ernie.Xform).set(
                translate=(0, 0, 0)
            )
            for c in range(SIDE):
                scene.define(
                    cell_path(r, c),
                    ernie.Cube,
                    size=1.0,
                    display_color=ALIVE if self.alive[r, c] else DEAD,
                ).set(
                    translate=(
                        c * SPACING - half,
                        r * SPACING - half,
                        0.0,
                    )
                )
        self.row = 0

    def step(self):
        scene = self.scene
        # The next generation on a torus; only the flipped cells are
        # written.
        around = sum(
            np.roll(np.roll(self.alive, dr, 0), dc, 1)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        )
        alive = (around == 3) | (self.alive & (around == 2))
        for r, c in zip(*np.nonzero(alive != self.alive)):
            scene.set(
                cell_path(r, c),
                tokens.DISPLAY_COLOR,
                ALIVE if alive[r, c] else DEAD,
            )
        self.alive = alive
        # A scanning row: the last one down, the next one up.
        scene.get_prim(row_path(self.row)).set_xform(
            translate=(0, 0, 0)
        )
        self.row = (self.row + 1) % SIDE
        scene.get_prim(row_path(self.row)).set_xform(
            translate=(0, 0, 1.0)
        )

    def test_post_step(self):
        scene = self.scene
        for r in range(SIDE):
            for c in range(SIDE):
                colour = tuple(
                    scene.get(cell_path(r, c), tokens.DISPLAY_COLOR)
                )
                assert colour == (
                    ALIVE if self.alive[r, c] else DEAD
                ), "every cell shows its state"
        lifted = scene.world_transform(cell_path(self.row, 0))
        assert lifted[2][3] == 1.0, "the scanning row is lifted"


if __name__ == "__main__":
    ernie.demo.main(Demo)
