# SPDX-License-Identifier: BSD-3-Clause
"""Tests of sharing device memory with PyTorch: Ernie's columns as tensors, and tensors as
Ernie's columns, in place, with nothing copied."""

import os
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def context(cuda_context):
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device for PyTorch")
    return cuda_context


def test_pytorch_writes_a_column_in_place(context):
    poses = context.column("poses", "float32", (16384, 8))
    before = context.transfers
    stream = torch.cuda.current_stream(context.cuda_ordinal)
    view = torch.from_dlpack(poses)
    assert view.shape == (16384, 8) and view.device.index == context.cuda_ordinal
    view.copy_(torch.arange(8, device=view.device).expand(16384, 8))
    view.mul_(2)
    poses.ready(stream.cuda_stream)
    assert context.transfers == before, "PyTorch's writes copied nothing"
    values = np.frombuffer(poses.read(), dtype=np.float32).reshape(16384, 8)
    np.testing.assert_array_equal(values, np.tile(2.0 * np.arange(8), (16384, 1)))


def test_tensors_of_the_default_allocator_are_refused(context):
    tensor = torch.ones(100, device=f"cuda:{context.cuda_ordinal}")
    with pytest.raises(BufferError, match="cuda_allocator"):
        context.wrap(tensor, "plain")


WRAP = textwrap.dedent(
    """
    import sys

    import numpy as np
    import torch

    import ernie

    if sys.argv[1] == "ernie":
        torch.cuda.memory.change_current_allocator(ernie.cuda_allocator())
    context = ernie.DeviceContext()
    device = f"cuda:{context.cuda_ordinal}"
    state = torch.full((4096, 8), 2.5, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream
    poses = context.wrap(state[1:], "poses", stream)
    state.mul_(2)
    poses.ready(stream)
    values = np.frombuffer(poses.read(), dtype=np.float32)
    assert poses.shape == [4095, 8], poses.shape
    assert np.all(values == 5.0), values[:4]
    transfers = context.transfers
    assert transfers["copied"] == 0 and transfers["uploaded"] == 0, transfers
    """
)


@pytest.mark.parametrize(
    ("mode", "allocator_config"),
    [("ernie", ""), ("expandable", "expandable_segments:True")],
    ids=["ernie_allocator", "expandable_segments"],
)
def test_exportable_tensors_are_wrapped_in_place(context, mode, allocator_config):
    # The allocator is chosen before PyTorch first allocates, so each runs in its own process.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=allocator_config)
    result = subprocess.run(
        [sys.executable, "-c", WRAP, mode], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
