# SPDX-License-Identifier: BSD-3-Clause
"""Fixtures shared by the Python tests."""

import pytest

import ernie


@pytest.fixture(scope="module")
def cuda_context():
    """A device context paired with a CUDA device, skipping the test without one."""
    try:
        context = ernie.DeviceContext()
    except RuntimeError as error:
        pytest.skip(str(error))
    if context.cuda_ordinal is None:
        pytest.skip(f"No CUDA device pairs with {context.name}")
    return context
