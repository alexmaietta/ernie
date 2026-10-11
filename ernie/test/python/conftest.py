# SPDX-License-Identifier: BSD-3-Clause
import os

import numpy as np
import pytest

import ernie
from ernie import tokens

# CI sets ERNIE_TEST_REQUIRE_VALIDATION=1: without a Vulkan device or
# the validation layer the tests fail rather than skipping or running
# unvalidated (as the C++ tests' VulkanTest does).
REQUIRE_VALIDATION = (
    os.environ.get("ERNIE_TEST_REQUIRE_VALIDATION") == "1"
)


@pytest.fixture(scope="session")
def renderer():
    """A validating renderer, or a skip without a Vulkan device."""
    try:
        renderer = ernie.Renderer(validation=True)
    except RuntimeError as error:
        if REQUIRE_VALIDATION:
            pytest.fail(f"No Vulkan device: {error}")
        pytest.skip(f"No Vulkan device: {error}")
    if REQUIRE_VALIDATION and not renderer.validation_enabled:
        pytest.fail("The Vulkan validation layer is not installed")
    return renderer


@pytest.fixture(autouse=True)
def no_validation_messages(request):
    """Fails a test whose renderer reported Vulkan validation
    messages."""
    yield
    if "renderer" in request.fixturenames:
        messages = request.getfixturevalue(
            "renderer"
        ).take_validation_messages()
        assert messages == [], "Vulkan validation:\n" + "\n".join(
            messages
        )


def rolling_heights(rows=65, cols=65):
    """About +/-40 m of smooth relief on a grid with 2 m cells."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32) * 2.0
    return (
        30.0 * np.sin(x / 45.0) * np.cos(y / 55.0)
        + 10.0 * np.sin((x + y) / 20.0)
    ).astype(np.float32)


@pytest.fixture
def terrain_scene():
    return make_terrain_scene()


def make_terrain_scene():
    """A lit terrain seen from above at 48 x 48 pixels against a blue
    background."""
    heights = rolling_heights()
    rows, cols = heights.shape
    scene = ernie.Scene()
    scene.define(
        "/World/Terrain",
        ernie.Terrain,
        heights=heights,
        origin=(-cols, rows),
        cell_size=2.0,
    )
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM,
        ernie.distant_light_transform((1.0, 1.0, 0.6)),
    )
    scene.define(
        "/World/Camera", ernie.Camera, focal_length=18.0
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform(
            (0.0, -150.0, 200.0), (0.0, 0.0, 0.0)
        ),
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera="/World/Camera",
        resolution=(48, 48),
        clear_color=(0.0, 0.0, 1.0),
    )
    return scene
