# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest

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


def paired_devices():
    adapters = ernie.list_adapters()
    for cuda in adapters:
        if cuda.backend != ernie.Backend.CUDA or cuda.unsupported:
            continue
        for vulkan in adapters:
            if vulkan.backend != ernie.Backend.VULKAN or vulkan.unsupported:
                continue
            if vulkan.type == ernie.AdapterType.CPU:
                continue
            render = ernie.Device(backend=vulkan.backend, adapter=vulkan.alias)
            if render.shared_device == (2, cuda.ordinal):
                return ernie.Device(backend=cuda.backend, adapter=cuda.alias), render
    pytest.skip("no Vulkan device shares a GPU with CUDA")


class Foreign:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, **options):
        return self.tensor.__dlpack__(**options)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def test_poses_from_a_busy_cuda_stream_land_after_its_work():
    cuda, render = paired_devices()
    scene = ernie.Scene(render)
    for i in range(4):
        ernie.Xform.define(scene, f"/P{i}")
    selection = scene.select([f"/P{i}" for i in range(4)])
    scene.publish(wait=True)
    big = cuda.create_tensor(ernie.DType.FLOAT32, (32 << 20,))
    spare = cuda.create_tensor(ernie.DType.FLOAT32, (32 << 20,))
    poses = cuda.create_tensor(ernie.DType.FLOAT32, (4, 7))
    for step in range(1, 4):
        wanted = np.zeros((4, 7), dtype=np.float32)
        wanted[:, :3] = step
        wanted[:, 6] = 1
        filled = cuda.import_tensor(wanted)
        for _ in range(8):
            cuda.copy(big, spare)
        cuda.copy(filled, poses)
        selection.write_poses(Foreign(poses))
        np.testing.assert_array_equal(selection.read_poses().numpy(), wanted)
