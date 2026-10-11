# SPDX-License-Identifier: BSD-3-Clause
"""Small helpers the demos share: heatmaps, side by side, sRGB, data files, and `write_png` (from
ernie.demo).

Not a demo itself; the demos import it from this directory.
"""

import os

import numpy as np

from ernie.demo import write_png  # noqa: F401 - shared with the demos that import it from here

HERE = os.path.dirname(os.path.abspath(__file__))


def data_file(name):
    """The data file `name` the demos read, such as an ISD: in `data` beside the demos where they
    are installed (`python -m ernie.demos`), else the repository's ISD test fixture."""
    installed = os.path.join(HERE, "data", name)
    if os.path.exists(installed):
        return installed
    return os.path.join(HERE, "..", "..", "ernie", "test", "fixtures", "isd", name)


def heatmap(values, low, high):
    """`values` from `low` (blue) through cyan, green and yellow to `high` (red), as RGB uint8."""
    t = np.clip((np.asarray(values, dtype=np.float64) - low) / max(high - low, 1e-12), 0.0, 1.0)
    stops = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]],
                     dtype=np.float64)
    position = t * (len(stops) - 1)
    index = np.minimum(position.astype(int), len(stops) - 2)
    fraction = (position - index)[..., None]
    return ((1.0 - fraction) * stops[index] + fraction * stops[index + 1]).astype(np.uint8)


def side_by_side(images, gap=4):
    """RGB(A) uint8 images of one height next to each other with black gaps."""
    height = images[0].shape[0]
    parts = []
    for i, image in enumerate(images):
        if i > 0:
            parts.append(np.zeros((height, gap, image.shape[2]), dtype=np.uint8))
        parts.append(image)
    return np.concatenate(parts, axis=1)


def srgb_to_linear(rgb):
    """sRGB-encoded colour in [0, 1] to linear."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
