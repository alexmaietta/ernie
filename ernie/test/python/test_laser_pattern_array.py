# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


@pytest.fixture
def cuda_renderer(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    return renderer


def halves(left, right):
    """A 4 x 8 pattern image, `left` in its left half and `right` in
    its right."""
    image = np.full((4, 8), right, np.float32)
    image[:, :4] = left
    return image


def laser_scene(image):
    """A green laser 3 m above the floor projecting `image`, seen
    from above."""
    scene = ernie.Scene()
    scene.define(
        "/World/Camera", ernie.Camera, focal_length=18.0
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0, 0, 5), (0, 0, 0), up=(0, 1, 0)),
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        resolution=(64, 64),
        ambient=0.0,
    )
    scene.define("/World/Floor", ernie.GroundPlane)
    scene.define(
        "/World/Laser",
        ernie.LaserLight,
        wavelength=532.0,
        power=0.01,
        pattern="image",
        pattern_image_array=image,
        pattern_field_of_view=(60.0, 30.0),
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0, 0, 3), (0, 0, 0), up=(0, 1, 0)),
    )
    return scene


def green(image):
    return np.asarray(image)[..., 1].astype(np.int32)


def sides(image):
    """The mean green of the image's left and right thirds."""
    values = green(image)
    third = values.shape[1] // 3
    return values[:, :third].mean(), values[:, -third:].mean()


def test_host_arrays_light_as_their_image(renderer):
    image = renderer.render(laser_scene(halves(0.2, 0.6)))
    left, right = sides(image)
    assert 0 < left < right < 255, (left, right)
    assert green(image).max() < 255, "exposed below white"


def test_mapped_pattern_arrays_are_drawn_without_copies(
    cuda_renderer,
):
    expected = green(
        cuda_renderer.render(laser_scene(halves(0.2, 0.6)))
    )
    swapped = green(
        cuda_renderer.render(laser_scene(halves(0.6, 0.2)))
    )

    # On the GPU, the same light as on the host.
    scene = laser_scene(cuda_renderer.cuda_upload(halves(0.2, 0.6)))
    np.testing.assert_allclose(
        green(cuda_renderer.render(scene)), expected, atol=1
    )

    # Mapped: the renderer's own memory, holding the image, written
    # in place and drawn as it is.
    mapped = cuda_renderer.map(
        scene, "/World/Laser", tokens.LASER_PATTERN_IMAGE_ARRAY
    )
    assert mapped.__dlpack_device__() == (2, 0)
    np.testing.assert_array_equal(
        np.array(cuda_renderer.cuda_download(mapped)),
        halves(0.2, 0.6),
    )
    cuda_renderer.cuda_copy(
        cuda_renderer.cuda_upload(halves(0.6, 0.2)), mapped
    )
    scene.unmap("/World/Laser", tokens.LASER_PATTERN_IMAGE_ARRAY)
    np.testing.assert_allclose(
        green(cuda_renderer.render(scene)), swapped, atol=1
    )

    # Mapped again, it holds what was last written.
    again = cuda_renderer.map(
        scene, "/World/Laser", tokens.LASER_PATTERN_IMAGE_ARRAY
    )
    np.testing.assert_array_equal(
        np.array(cuda_renderer.cuda_download(again)),
        halves(0.6, 0.2),
    )
    scene.unmap("/World/Laser", tokens.LASER_PATTERN_IMAGE_ARRAY)


def test_only_gpu_pattern_arrays_map(cuda_renderer):
    scene = laser_scene(halves(0.2, 0.6))
    with pytest.raises(ValueError):
        cuda_renderer.map(
            scene, "/World/Laser", tokens.LASER_PATTERN_IMAGE_ARRAY
        )
