# SPDX-License-Identifier: BSD-3-Clause
"""A field of cubes riding a travelling wave, every cube's pose, colour
and temperature (seen with thermal emission on) written each frame as
three columns (one array each, mapped and written in place): one
change per array however many cubes. With
`--rows cuda` the wave is computed with PyTorch on the GPU and
written into the renderer's own arrays."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIDE = 64  # cubes along each side
SPACING = 1.2  # m between cubes
AMPLITUDE = 1.5  # m
WAVELENGTH = 18.0  # m
SPEED = 6.0  # m/s


def wave(t, xp, device=None):
    """Every cube's place, colour and temperature at `t` seconds, (n,
    3), (n, 3) and (n,) arrays of `xp` (NumPy, or PyTorch on `device`):
    on the grid, lifted by a ring wave travelling out from the centre,
    blue and 250 K in its troughs, orange and 350 K on its crests."""
    half = 0.5 * (SIDE - 1) * SPACING
    if xp is np:
        i = np.arange(SIDE * SIDE, dtype=np.float64)
    else:
        i = xp.arange(SIDE * SIDE, dtype=xp.float64, device=device)
    x = (i % SIDE) * SPACING - half
    y = (i // SIDE) * SPACING - half
    r = xp.sqrt(x * x + y * y)
    z = AMPLITUDE * xp.sin(
        2.0 * math.pi * (r - SPEED * t) / WAVELENGTH
    )
    h = 0.5 + 0.5 * z / AMPLITUDE
    places = xp.stack([x, y, z], 1)
    colours = xp.stack(
        [0.1 + 0.9 * h, 0.3 + 0.2 * h, 0.9 - 0.8 * h], 1
    )
    kelvin = 300.0 + 50.0 * z / AMPLITUDE
    return places, colours, kelvin


class Demo(ernie.demo.Demo):
    num_frames = 120

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument(
            "--rows",
            default="cpu",
            choices=("cpu", "cuda"),
            help="where the wave is computed and written",
        )
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.device = args.rows
        if self.device == "cuda":
            try:
                import torch
            except ImportError:
                torch = None
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError(
                    "--rows cuda needs PyTorch with CUDA"
                )
            self.torch = torch
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.5, 0.8)),
        )
        world.define("Camera", ernie.Camera, focal_length=18.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -70.0, 45.0), (0.0, 0.0, 0.0)
            ),
        )
        places = wave(0.0, np)[0]
        paths = []
        for i in range(SIDE * SIDE):
            path = f"/World/Field/Cube{i}"
            scene.define(path, ernie.Cube, size=1.0).set(
                translate=tuple(float(v) for v in places[i])
            )
            paths.append(path)
        # Each attribute of every cube as one column.
        field = ernie.PrimSet(paths)
        self.poses = scene.column(field, tokens.TRANSLATE)
        self.colours = scene.column(field, tokens.DISPLAY_COLOR)
        self.colours.write(np.zeros((SIDE * SIDE, 3), np.float32))
        self.kelvin = scene.column(field, tokens.THERMAL_TEMPERATURE)
        self.kelvin.write(np.zeros(SIDE * SIDE, np.float32))
        self.time = 0.0
        self.place()

    def place(self):
        """Every cube's row of each column, written in place."""
        columns = (self.poses, self.colours, self.kelvin)
        if self.device == "cuda":
            torch = self.torch
            for column, rows in zip(
                columns, wave(self.time, torch, "cuda")
            ):
                with column.map("cuda") as mapped:
                    torch.from_dlpack(mapped).copy_(rows)
        else:
            for column, rows in zip(columns, wave(self.time, np)):
                with column.map("cpu") as mapped:
                    np.asarray(mapped)[:] = rows

    def step(self):
        self.time += self.frame_dt
        self.place()

    def test_post_step(self):
        # Rows read as each cube's own attribute.
        places = wave(self.time, np)[0]
        for i in (0, SIDE * SIDE // 2 + 7, SIDE * SIDE - 1):
            at = self.scene.world_transform(f"/World/Field/Cube{i}")
            assert all(
                abs(at[k][3] - places[i][k]) < 1e-4 for k in range(3)
            ), "each cube is where its row puts it"

    def test_final(self):
        rows = (
            np.asarray(self.colours.read())
            if self.device == "cpu"
            else None
        )
        if rows is not None:
            assert rows.shape == (SIDE * SIDE, 3), (
                "one colour per cube"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
