# SPDX-License-Identifier: BSD-3-Clause
"""Tests of rendering from Python: the README's first render as written there, and the authoring
it uses."""

import os
import pathlib
import re

import numpy as np
import pytest

import ernie

Image = pytest.importorskip("PIL.Image")

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"


def readme_python():
    """The README's first-render Python example."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## Your First Render", 1)[1]
    return re.search(r"### Python\s+```python\n(.*?)```", section, re.S).group(1)


def skip_without_ray_tracing():
    """Skips the test without a device that ray traces; no Vulkan at all fails where CI requires
    it."""
    try:
        devices = [device for device in ernie.vulkan_devices() if device.supported]
    except RuntimeError as error:
        devices, reason = [], str(error)
    else:
        reason = "No Vulkan device"
    if not devices:
        if os.environ.get("ERNIE_TEST_REQUIRE_VULKAN") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    if not any(device.ray_query for device in devices):
        pytest.skip("No Vulkan device that ray traces")


def test_the_readmes_first_render_runs_as_written(monkeypatch, tmp_path):
    skip_without_ray_tracing()
    monkeypatch.chdir(tmp_path)

    exec(compile(readme_python(), str(README), "exec"), {})

    image = np.asarray(Image.open(tmp_path / "ernie.png"))
    assert image.shape == (480, 640, 4)
    assert image.dtype == np.uint8
    assert image[240, 320, 0] > 100, "the rock, lit, where the camera looks"
    assert image[470, 320, 0] > image[5, 320, 0], "near ground lit more steeply than far"
    assert (image[..., 3] == 255).all()


def test_define_authors_transform_ops_and_attributes():
    scene = ernie.Scene()
    rock = scene.define("/Rock", ernie.Sphere, translate=(1, 2, 3), radius=2.0)
    assert scene.type_name(rock) == "Sphere"
    assert scene.get(rock, "radius") == 2.0
    assert scene.local_transform(rock)[0][3] == 1.0
    assert rock.set("radius", 3.0) == rock
    assert scene.get("/Rock", "radius") == 3.0
    assert "Sphere" in ernie.PRIM_TYPES and ernie.RenderSettings == "RenderSettings"


def test_prims_keep_their_scene_alive():
    prim = ernie.Scene().define("/Rock", ernie.Sphere).set("radius", 2.0)
    assert prim.set("radius", 3.0) == prim


def test_prims_belong_to_their_own_scene():
    first, second = ernie.Scene(), ernie.Scene()
    rock = first.define("/Rock", ernie.Sphere)
    second.define("/Rock", ernie.Sphere)
    assert rock != second.find("/Rock")
    with pytest.raises(ValueError):
        second.set(rock, "radius", 1.0)


def test_define_authors_no_ops_for_a_rotation_order_alone():
    scene = ernie.Scene()
    scene.define("/Rock", ernie.Sphere, rotation_order="ZYX")
    assert scene.get("/Rock", "xformOpOrder") is None


def test_transform_helpers_point_cameras_and_lights():
    camera = np.array(ernie.look_at_transform((0, -8, 0), (0, 0, 0)))
    # Cameras look down their -z: here along +y.
    assert np.allclose(-camera[:3, 2], (0, 1, 0))
    assert np.allclose(camera[:3, 3], (0, -8, 0))
    light = np.array(ernie.distant_light_transform((0, 0, 2)))
    assert np.allclose(light[:3, 2], (0, 0, 1)), "its +z toward the light"
    assert ernie.tokens.TRANSFORM == "xformOp:transform"
