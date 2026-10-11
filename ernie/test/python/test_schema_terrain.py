# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


def test_terrain_class_reads_defaults_and_writes_tensors():
    scene = ernie.Scene()
    heights = np.arange(16, dtype=np.float32).reshape(4, 4)
    terrain = scene.define(
        "/World/Terrain",
        ernie.Terrain,
        heights=heights,
        origin=(10.0, 20.0),
        cell_size=30,
    )
    assert isinstance(terrain, ernie.Terrain) and isinstance(
        terrain, ernie.Gprim
    )
    assert ernie.Terrain.prim_type.base.name == "Gprim"
    np.testing.assert_array_equal(terrain.heights, heights)
    assert terrain.origin == (10.0, 20.0)
    assert terrain.cell_size == 30.0 and isinstance(
        terrain.cell_size, float
    )
    assert scene.get("/World/Terrain", tokens.CELL_SIZE) == 30.0
    assert (
        terrain.exaggeration == 1.0
        and terrain.no_data == 0.0
        and terrain.planet_radius == 0.0
    )
    assert terrain.display_color == (0.5, 0.5, 0.5)
    terrain.planet_radius = 1737.4e3
    assert (
        scene.get("/World/Terrain", tokens.PLANET_RADIUS) == 1737.4e3
    )
    assert ernie.Terrain.cell_size.group == "Terrain"
    assert ernie.Terrain.heights.kind == ernie.AttributeKind.TENSOR


def test_star_field_class():
    scene = ernie.Scene()
    magnitudes = np.array([1.0, 2.5, 7.0], dtype=np.float32)
    stars = scene.define(
        "/World/Stars",
        ernie.StarField,
        magnitude=magnitudes,
        magnitude_limit=6,
    )
    assert isinstance(stars, ernie.Imageable) and not isinstance(
        stars, ernie.Gprim
    )
    np.testing.assert_array_equal(stars.magnitude, magnitudes)
    assert stars.right_ascension is None and stars.color_index is None
    assert stars.magnitude_limit == 6.0
    assert (
        stars.zero_point == pytest.approx(2.54e-6)
        and stars.exposure == 0.0
    )
    assert (
        scene.get("/World/Stars", tokens.STARS_MAGNITUDE_LIMIT) == 6.0
    )
    with pytest.raises(TypeError, match="not a StarField"):
        ernie.StarField(
            scene.define("/World/Terrain", ernie.Terrain).prim
        )
