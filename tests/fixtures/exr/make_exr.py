# SPDX-License-Identifier: BSD-3-Clause
"""Writes the EXR fixtures from the OpenEXR file layout alone, so the reader is checked against a
writer other than the OpenEXR library: python3 make_exr.py."""

import struct
import zlib
from pathlib import Path

HALF, FLOAT = 1, 2
NONE, ZIPS, ZIP = 0, 2, 3
LINES_PER_BLOCK = {NONE: 1, ZIPS: 1, ZIP: 16}


def attribute(name, kind, value):
    return f"{name}\0{kind}\0".encode() + struct.pack("<i", len(value)) + value


def box(x0, y0, x1, y1):
    return struct.pack("<4i", x0, y0, x1, y1)


def zip_block(raw):
    """Zlib of the bytes split into even and odd halves and differenced; raw if no smaller."""
    reordered = raw[0::2] + raw[1::2]
    predicted = bytes([reordered[0]]) + bytes(
        (reordered[i] - reordered[i - 1] + 128) & 0xFF for i in range(1, len(reordered))
    )
    packed = zlib.compress(predicted, 9)
    return packed if len(packed) < len(raw) else raw


def write(path, channels, window, compression, display=None):
    """`channels` maps names to (pixel type, value at window-relative (x, y))."""
    names = sorted(channels)
    chlist = b"".join(
        name.encode() + b"\0" + struct.pack("<iB3xii", channels[name][0], 0, 1, 1) for name in names
    )
    header = b"".join(
        [
            attribute("channels", "chlist", chlist + b"\0"),
            attribute("compression", "compression", bytes([compression])),
            attribute("dataWindow", "box2i", box(*window)),
            attribute("displayWindow", "box2i", box(*(display or window))),
            attribute("lineOrder", "lineOrder", b"\0"),
            attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
            attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0)),
            attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
        ]
    )
    x0, y0, x1, y1 = window
    width = x1 - x0 + 1

    def line(y):
        return b"".join(
            struct.pack(
                "<" + ("e" if kind == HALF else "f") * width,
                *(value(x, y - y0) for x in range(width)),
            )
            for kind, value in (channels[name] for name in names)
        )

    blocks = []
    step = LINES_PER_BLOCK[compression]
    for top in range(y0, y1 + 1, step):
        raw = b"".join(line(y) for y in range(top, min(top + step, y1 + 1)))
        data = raw if compression == NONE else zip_block(raw)
        blocks.append(struct.pack("<ii", top, len(data)) + data)
    offsets = []
    at = 8 + len(header) + 1 + 8 * len(blocks)
    for block in blocks:
        offsets.append(at)
        at += len(block)
    path.write_bytes(
        struct.pack("<ii", 20000630, 2)
        + header
        + b"\0"
        + struct.pack(f"<{len(offsets)}Q", *offsets)
        + b"".join(blocks)
    )


here = Path(__file__).parent
write(
    here / "rgba_half_window.exr",
    {
        "R": (HALF, lambda x, y: 0.25 * x),
        "G": (HALF, lambda x, y: 2.0**y),
        "B": (HALF, lambda x, y: -0.5),
        "A": (HALF, lambda x, y: 0.5),
    },
    window=(2, 1, 6, 3),
    display=(0, 0, 7, 4),
    compression=NONE,
)
write(
    here / "rgb_float_zip.exr",
    {
        "R": (FLOAT, lambda x, y: x / 16),
        "G": (FLOAT, lambda x, y: y / 16),
        "B": (FLOAT, lambda x, y: 100.0 + x),
    },
    window=(0, 0, 15, 15),
    compression=ZIP,
)
write(
    here / "y_half_zips.exr",
    {"Y": (HALF, lambda x, y: 0.5 * x + y)},
    window=(0, 0, 3, 1),
    compression=ZIPS,
)
