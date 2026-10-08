"""Small helpers the demos share: writing PNGs without extra packages, heatmaps, side by side.

Not a demo itself; the demos import it from this directory.
"""

import struct
import zlib

import numpy as np


def write_png(path, image):
    """Writes an (rows, cols, 3 or 4) uint8 image as PNG with only the standard library."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    rows, cols, channels = image.shape
    colour_type = {3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(rows))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", cols, rows, 8, colour_type, 0, 0, 0)
    with open(path, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                   chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


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
