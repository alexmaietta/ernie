# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np

import ernie


def boxes(scene, count):
    for i in range(count):
        ernie.Cube.define(scene, f"/Box{i}")
    return ernie.Cube.select(scene, "/Box*").selection


def test_poses_round_trip_in_either_quaternion_order(scene):
    selection = boxes(scene, 3)
    half = math.sqrt(0.5)
    poses = np.array(
        [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, half, half], [7, 8, 9, half, 0, 0, half]],
        dtype=np.float32,
    )
    selection.write_poses(poses)
    np.testing.assert_allclose(selection.read_poses().numpy(), poses, atol=1e-6)

    wxyz = selection.read_poses(order="wxyz").numpy()
    np.testing.assert_allclose(wxyz[:, 3], poses[:, 6], atol=1e-6)
    np.testing.assert_allclose(wxyz[:, 4:], poses[:, 3:6], atol=1e-6)
    np.testing.assert_allclose(
        ernie.Cube.select(scene, "/Box1").translate[0], (4, 5, 6), atol=1e-6
    )


def test_poses_write_in_parts_and_masked_rows(scene):
    selection = boxes(scene, 4)
    translation = np.arange(12, dtype=np.float32).reshape(4, 3)
    rotation = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 1))
    selection.write_poses(translation=translation, rotation=rotation, order="wxyz")
    poses = selection.read_poses().numpy()
    np.testing.assert_allclose(poses[:, :3], translation)
    np.testing.assert_allclose(poses[:, 3:], np.tile([0, 0, 0, 1], (4, 1)))

    moved = np.zeros((4, 7), dtype=np.float32)
    moved[:, 6] = 1
    selection.write_poses(moved, mask=np.array([0b1010], dtype=np.uint64))
    np.testing.assert_allclose(selection.read_poses().numpy()[:, 0], [0, 0, 6, 0])


def test_tensors_move_between_numpy_and_the_device(device):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = device.import_tensor(values)
    assert list(tensor.shape) == [2, 3]
    assert tensor.dtype == ernie.DType.FLOAT32
    np.testing.assert_array_equal(tensor.numpy(), values)
    copy = device.create_tensor(ernie.DType.FLOAT32, (2, 3))
    device.copy(tensor, copy).wait()
    np.testing.assert_array_equal(copy.numpy(), values)
