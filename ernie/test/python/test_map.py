# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from test_cuda import instancer_scene

import ernie
from ernie import tokens


@pytest.fixture
def cuda_renderer(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    return renderer


def green_columns(image):
    pixels = np.asarray(image).astype(np.int32)
    green = (pixels[..., 1] > 200) & (
        2 * (pixels[..., 0] + pixels[..., 2]) < pixels[..., 1]
    )
    return np.nonzero(green.any(axis=0))[0]


def test_mapped_cpu_arrays_are_written_in_place(renderer):
    scene = instancer_scene(np.array([[-10.0, 0.0, 0.0]], np.float32))
    columns = green_columns(renderer.render(scene))
    assert columns.size and columns.max() < 24
    with scene.map("/World/Rocks", tokens.POSITIONS) as positions:
        positions[0, 0] = 10.0
    np.testing.assert_array_equal(
        scene.read("/World/Rocks", tokens.POSITIONS), [[10, 0, 0]]
    )
    columns = green_columns(renderer.render(scene))
    assert columns.size and columns.min() > 24, (
        "the instance moved right"
    )
    assert renderer.take_validation_messages() == []


def test_mapped_cuda_arrays_are_written_in_place_every_step(
    cuda_renderer,
):
    scene = instancer_scene(np.array([[-10.0, 0.0, 0.0]], np.float32))
    scene.write(
        "/World/Rocks",
        tokens.POSITIONS,
        cuda_renderer.cuda_upload(
            np.array([[-10.0, 0.0, 0.0]], np.float32)
        ),
        stream=1,
    )
    for x in (10.0, -10.0, 10.0, -10.0):
        mapping = scene.map(
            "/World/Rocks", tokens.POSITIONS, stream=1
        )
        assert mapping.array.__dlpack_device__() == (2, 0)
        value = cuda_renderer.cuda_upload(
            np.array([[x, 0.0, 0.0]], np.float32)
        )
        cuda_renderer.cuda_copy(value, mapping.array, stream=1)
        mapping.unmap()
        columns = green_columns(cuda_renderer.render(scene))
        assert columns.size
        assert (columns.min() > 24) == (x > 0), (
            f"step drew the array written for x={x}"
        )
    assert cuda_renderer.take_validation_messages() == []


def test_only_written_arrays_map(renderer):
    scene = instancer_scene(np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="not a written array"):
        scene.map("/World/Rocks", tokens.POINTS)


def test_columns_write_and_map_rows_of_many_prims():
    scene = ernie.Scene()
    for path, x in (("/A", 1.0), ("/B", 4.0)):
        scene.define(path, "Xform").set_xform(translate=(x, 0.0, 0.0))
    bodies = ernie.PrimSet(["/A", "/B"])
    poses = scene.column(bodies, tokens.TRANSLATE)
    np.testing.assert_array_equal(poses.read()[:, 0], [1.0, 4.0])
    with poses.map("cpu") as rows:
        rows[:, 0] += 10.0
    np.testing.assert_array_equal(
        np.asarray(scene.read("/B", tokens.TRANSLATE)), [14.0, 0.0, 0.0]
    )
    poses.write(np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(poses.read(), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        poses.map("cuda")


def test_column_writes_come_as_one_column_change():
    scene = ernie.Scene()
    for path in ("/A", "/B"):
        scene.define(path, "Xform").set_xform(translate=(0.0, 0.0, 0.0))
    poses = scene.column(ernie.PrimSet(["/A", "/B"]), tokens.TRANSLATE)
    poses.write(np.zeros((2, 3), np.float32))
    cursor = ernie.ChangeCursor()
    scene.changes(cursor)
    poses.write(np.ones((2, 3), np.float32))
    changes, columns = scene.column_changes(cursor)
    assert changes == []
    assert len(columns) == 1 and len(columns[0]) == 2
    assert columns[0].attribute == tokens.TRANSLATE
    assert [str(p) for p in columns[0].paths] == ["/A", "/B"]
    assert columns[0].bits & ernie.Dirty.TRANSFORM
    expanded = scene.expand_columns(changes, columns)
    assert [str(p) for p, _ in expanded] == ["/A", "/B"]
    assert all(bits & ernie.Dirty.TRANSFORM for _, bits in expanded)


def test_columns_read_without_a_change_and_hold_arrays_of_any_length():
    scene = ernie.Scene()
    for path in ("/A", "/B"):
        scene.define(path, "Xform").set_xform(translate=(0.0, 0.0, 0.0))
    poses = scene.column(ernie.PrimSet(["/A", "/B"]), tokens.TRANSLATE)
    poses.write(np.ones((2, 3), np.float32))
    cursor = ernie.ChangeCursor()
    scene.changes(cursor)
    np.testing.assert_array_equal(poses.read("cpu"), np.ones((2, 3)))
    assert scene.column_changes(cursor) == ([], [])
    with pytest.raises(ValueError):
        poses.read("cuda")

    for path in ("/M", "/N"):
        scene.define(path, ernie.Mesh)
    points = scene.column(ernie.PrimSet(["/M", "/N"]), tokens.POINTS)
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    points.write_arrays(values, [2, 3])
    np.testing.assert_array_equal(
        np.asarray(scene.read("/N", tokens.POINTS)), values[2:]
    )


def test_selections_map_every_attribute_of_a_kind_at_once():
    scene = ernie.Scene()
    for i, path in enumerate(("/A", "/B", "/C")):
        scene.define(path, ernie.Cube, translate=(i, 0.0, 0.0))
    scene.define("/S", ernie.Sphere, translate=(0.0, 0.0, 0.0))
    cubes = scene.select(ernie.Cube, [tokens.TRANSLATE])
    assert len(cubes) == 3
    with cubes.map([tokens.TRANSLATE], "cpu") as (poses,):
        poses[:, 2] = 5.0
    (poses,) = cubes.read([tokens.TRANSLATE])
    np.testing.assert_array_equal(poses[:, 2], [5.0, 5.0, 5.0])
    scene.define("/D", ernie.Cube, translate=(3.0, 0.0, 0.0))
    assert [str(p) for p in cubes.prims.paths] == ["/A", "/B", "/C", "/D"]
