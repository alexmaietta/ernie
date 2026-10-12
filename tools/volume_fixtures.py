# SPDX-License-Identifier: BSD-3-Clause
"""Writes the NanoVDB test fixtures in ernie/test/fixtures/volumes with Warp.

cube.nvdb: a float grid of 16^3 ones, 0.1 m voxels, index (0, 0, 0) at
(-0.75, -0.75, -0.75) m, so its values span +-0.75 m about the origin, as
Warp's Volume.save_to_nvdb writes a file (its grid unnamed, without
statistics). Trilinear between the voxels, a ray along an axis through the
middle crosses an optical depth of 16 voxels (1.6 m) times the density.

Run from the repository: .venv/bin/python tools/volume_fixtures.py
"""

import pathlib

import numpy as np
import warp as wp

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "ernie/test/fixtures/volumes"
)


def main():
    wp.config.quiet = True
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    ones = np.ones((16, 16, 16), dtype=np.float32)
    cube = wp.Volume.load_from_numpy(
        ones,
        min_world=(-0.75, -0.75, -0.75),
        voxel_size=0.1,
        bg_value=0.0,
        device="cpu",
    )
    cube.save_to_nvdb(str(OUT / "cube.nvdb"))


if __name__ == "__main__":
    main()
