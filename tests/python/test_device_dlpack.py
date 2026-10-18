# SPDX-License-Identifier: BSD-3-Clause
"""Tests of device columns shared with Warp through DLPack: written in place, never copied."""

import pytest

import ernie

np = pytest.importorskip("numpy")
wp = pytest.importorskip("warp")


@pytest.fixture(scope="module")
def context():
    try:
        context = ernie.DeviceContext()
    except RuntimeError as error:
        pytest.skip(str(error))
    if context.cuda_ordinal is None:
        pytest.skip(f"No CUDA device pairs with {context.name}")
    wp.init()
    return context


@pytest.fixture
def device(context):
    return wp.get_device(f"cuda:{context.cuda_ordinal}")


@wp.kernel
def step(poses: wp.array2d(dtype=wp.float32), time: float):
    world, component = wp.tid()
    poses[world, component] = float(world) + 0.5 * float(component) + time


def test_warp_writes_a_column_in_place(context, device):
    poses = context.column("poses", "float32", (16384, 8))
    before = context.transfers
    with wp.ScopedDevice(device):
        view = wp.from_dlpack(poses)
        assert view.shape == (16384, 8)
        for time in range(3):
            wp.launch(step, dim=view.shape, inputs=[view, float(time)])
        poses.ready(wp.get_stream().cuda_stream)
    assert context.transfers == before, "Warp's steps copied nothing"

    values = np.frombuffer(poses.read(), dtype=np.float32).reshape(16384, 8)
    expected = np.arange(16384)[:, None] + 0.5 * np.arange(8)[None, :] + 2.0
    np.testing.assert_allclose(values, expected)


def test_assigning_a_view_is_free_and_other_arrays_copy_once(context, device):
    column = context.column("heights", "float32", (1000,))
    with wp.ScopedDevice(device):
        stream = wp.get_stream().cuda_stream
        before = context.transfers
        column.assign(wp.from_dlpack(column), stream)
        assert context.transfers == before, "its own view is taken in place"

        column.assign(wp.full(1000, 3.0, dtype=wp.float32), stream)
    copied = context.transfers["copied"] - before["copied"]
    assert copied == 4000, "one copy on the device"
    assert context.transfers["uploaded"] == before["uploaded"]
    assert np.all(np.frombuffer(column.read(), dtype=np.float32) == 3.0)


def test_columns_report_their_layout(context):
    column = context.column("colors", "uint8", (10, 4))
    assert column.shape == [10, 4]
    assert column.dtype == "uint8"
    assert column.__dlpack_device__() == (2, context.cuda_ordinal)
    with pytest.raises(BufferError):
        column.__dlpack__(copy=True)
    with pytest.raises(ValueError):
        context.column("bad", "complex64", (3,))
