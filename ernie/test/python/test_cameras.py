# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest

import ernie
from ernie import tokens


def test_prim_paths_are_strs_paths_prims_or_typed_prims():
    scene = ernie.Scene()
    camera = scene.define(
        "/World/Camera", ernie.Camera, focal_length=50.0
    )
    for given in (camera, camera.prim, ernie.Path("/World/Camera")):
        ernie.author_singlet_chromatic(scene, given)
        assert (
            scene.get("/World/Camera", tokens.LENS_CHROMATIC_FOCUS)
            is not None
        )
    with pytest.raises(
        TypeError, match="incompatible function arguments"
    ):
        ernie.author_singlet_chromatic(scene, 3)
    with pytest.raises(ValueError, match="not absolute"):
        ernie.author_singlet_chromatic(scene, "World/Camera")


def test_singlet_chromatic_follows_its_glass():
    scene = ernie.Scene()
    scene.define("/Camera", ernie.Camera, focal_length=50.0)
    # SCHOTT N-BK7 by default: the C line focuses about f / V = 0.78
    # mm further than the F line.
    ernie.author_singlet_chromatic(scene, "/Camera", lateral=0.5)
    focus = dict(
        zip(
            *[iter(scene.get("/Camera", tokens.LENS_CHROMATIC_FOCUS))]
            * 2
        )
    )
    scale = dict(
        zip(
            *[iter(scene.get("/Camera", tokens.LENS_CHROMATIC_SCALE))]
            * 2
        )
    )
    assert 0.6 < focus[660.0] - focus[490.0] < 0.8
    assert scale[700.0] > 1.0 > scale[400.0]
    with pytest.raises(ValueError):
        ernie.author_singlet_chromatic(scene, "/Camera", nd=0.5)


def write_dem(directory, heights, max_latitude, west_longitude, ppd):
    """A LOLA LDEM-style PDS3 label and 16-bit image of `heights`
    (metres, rows from the northern), half-metre steps above a 1737.4
    km sphere."""
    lines, samples = heights.shape
    (directory / "TEST_DEM.IMG").write_bytes(
        np.round(heights / 0.5).astype("<i2").tobytes()
    )
    (directory / "TEST_DEM.LBL").write_text(
        f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {2 * samples}
FILE_RECORDS = {lines}
^IMAGE = "TEST_DEM.IMG"
/* A comment, which the reader skips. */
OBJECT = IMAGE
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  UNIT = METER
  SCALING_FACTOR = 0.5
  OFFSET = 1737400.
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"
  MAP_RESOLUTION = {ppd} <pix/deg>
  A_AXIS_RADIUS = 1737.4 <km>
  MAXIMUM_LATITUDE = {max_latitude} <deg>
  MINIMUM_LATITUDE = {max_latitude - lines / ppd} <deg>
  WESTERNMOST_LONGITUDE = {west_longitude} <deg>
  EASTERNMOST_LONGITUDE = {west_longitude + samples / ppd} <deg>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
    )
    return str(directory / "TEST_DEM.LBL")


def test_cylindrical_dems_read_as_lolas_ldem_labels_say(tmp_path):
    # Heights rising 10 m a line southwards and 1 m a sample eastwards,
    # 4 samples a degree from 40 S, 340 E.
    lines, samples = 8, 12
    heights = (
        10.0 * np.arange(lines)[:, None]
        + 1.0 * np.arange(samples)[None, :]
    )
    dem = ernie.read_cylindrical_dem(
        write_dem(tmp_path, heights, -40.0, 340.0, 4)
    )
    assert (dem.lines, dem.samples) == (lines, samples)
    assert dem.reference_radius == 1737400.0
    # Cell (line 2, sample 3) is centred at 40.625 S, 340.875 E;
    # longitudes wrap.
    assert math.isclose(dem.height_at(-40.625, 340.875), 23.0)
    assert math.isclose(dem.height_at(-40.625, -19.125), 23.0)
    # Halfway between cells, bilinear.
    assert math.isclose(dem.height_at(-40.75, 340.875), 28.0)


def test_local_terrains_hold_the_planets_curvature(tmp_path):
    # A flat sphere (every height 0) about 45 S 345 E: the heightfield
    # drops as the sphere does, about x^2 / 2R, and the local frame
    # takes the point to its origin, up along +z.
    flat = np.zeros((40, 80))
    dem = ernie.read_cylindrical_dem(
        write_dem(tmp_path, flat, -35.0, 330.0, 2)
    )
    cells, cell = 41, 2000.0
    terrain = ernie.local_terrain_of(dem, -45.0, 345.0, cells, cell)
    heights = np.asarray(terrain["heights"])
    assert heights.shape == (cells, cells)
    assert terrain["origin"] == (-40000.0, 40000.0)
    radius = 1737400.0
    assert math.isclose(heights[20, 20], 0.0, abs_tol=1e-3)
    edge = 20 * cell
    expected = math.sqrt(radius**2 - edge**2) - radius
    assert math.isclose(heights[20, 40], expected, rel_tol=1e-6)
    assert math.isclose(heights[0, 20], expected, rel_tol=1e-6)
    lat, lon = math.radians(-45.0), math.radians(345.0)
    point = radius * np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )
    local = np.asarray(terrain["body_to_local"]) @ np.append(
        point, 1.0
    )
    assert np.allclose(local[:3], 0.0, atol=1e-6)
    above = np.asarray(terrain["body_to_local"]) @ np.append(
        point * (1.0 + 1000.0 / radius), 1.0
    )
    assert np.allclose(above[:3], (0.0, 0.0, 1000.0), atol=1e-6)


def test_polar_dems_read_as_lolas_polar_labels_say(tmp_path):
    # A south polar grid of 10 m pixels, the pole at the centre of 21 x
    # 21, heights rising 10 m a line and 1 m a sample: 20 m from the
    # pole along 90 E lies two samples right of it, along the prime
    # meridian two lines up.
    heights = 10.0 * np.arange(21)[:, None] + np.arange(21)[None, :]
    (tmp_path / "POLAR.IMG").write_bytes(
        np.round(heights / 0.5).astype("<i2").tobytes()
    )
    (tmp_path / "POLAR.LBL").write_text(
        """PDS_VERSION_ID = PDS3
^IMAGE = "POLAR.IMG"
OBJECT = IMAGE
  LINES = 21
  LINE_SAMPLES = 21
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  SCALING_FACTOR = 0.5
  OFFSET = 1737400.
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "POLAR STEREOGRAPHIC"
  A_AXIS_RADIUS = 1737.4 <km>
  CENTER_LATITUDE = -90 <deg>
  CENTER_LONGITUDE = 0 <deg>
  MAP_SCALE = 10 <m/pix>
  LINE_PROJECTION_OFFSET = 10.0 <pix>
  SAMPLE_PROJECTION_OFFSET = 10.0 <pix>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
    )
    dem = ernie.read_polar_stereographic_dem(
        str(tmp_path / "POLAR.LBL")
    )
    assert (dem.lines, dem.map_scale) == (21, 10.0)
    assert np.allclose(dem.to_grid(-90.0, 45.0), (10.0, 10.0))
    radius = 1737400.0
    latitude = math.degrees(
        2.0 * math.atan(20.0 / (2.0 * radius)) - 0.5 * math.pi
    )
    assert math.isclose(dem.height_at(latitude, 90.0), 112.0)
    assert math.isclose(dem.height_at(latitude, 0.0), 90.0)
    terrain = ernie.local_terrain_of(dem, -89.9999, 0.0, 3, 1.0)
    assert np.asarray(terrain["heights"]).shape == (3, 3)
