# SPDX-License-Identifier: BSD-3-Clause
import pytest

import ernie


@pytest.fixture(scope="session")
def device():
    try:
        return ernie.Device()
    except RuntimeError as error:
        pytest.skip(f"no supported adapter: {error}")


@pytest.fixture
def scene(device):
    return ernie.Scene(device)
