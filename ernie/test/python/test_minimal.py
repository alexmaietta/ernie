# SPDX-License-Identifier: BSD-3-Clause
"""Minimal shading (ernie:minimal:mode): one forward pass straight into the outputs, RTX
Minimal's modes, its geometry outputs as the full pipeline's."""

import numpy as np
import pytest

import ernie
from ernie import tokens

GEOMETRY = ("DistanceToCamera", "DistanceToImagePlane", "PrimId", "InstanceId", "Normal")
VARS = ("LdrColor",) + GEOMETRY


def boxes_scene(mode=None, color=None, views=None):
    """Red, green and blue boxes on a ground plane under the sun, seen by one camera or a batch
    of `views` views around them."""
    scene = ernie.Scene()
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, -24, 14), (0, 0, 1))
    ).set(tokens.FOCAL_LENGTH, 35.0)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, -0.5, 0.8))
    )
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    for i, rgb in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        scene.define(f"/World/Box{i}", tokens.CUBE).set_xform(
            translate=(-6.0 + 6.0 * i, 0.0, 2.0), scale=(2.0, 2.0, 2.0)
        ).set(tokens.DISPLAY_COLOR, rgb)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.SHADOW_ENABLE, False)
    product = scene.define("/Render/Camera", tokens.RENDER_PRODUCT).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.RESOLUTION, (96, 64))
    for name in VARS:
        scene.define(f"/Render/Vars/{name}", tokens.RENDER_VAR).set(tokens.SOURCE_NAME, name)
    product.set(tokens.ORDERED_VARS, [ernie.Path(f"/Render/Vars/{n}") for n in VARS])
    if mode is not None:
        product.set(tokens.MINIMAL_MODE, mode)
    if color is not None:
        product.set(tokens.MINIMAL_CONSTANT_COLOR, color)
    if views is not None:
        poses = np.stack(
            [
                np.array(
                    ernie.look_at_transform(
                        (24 * np.sin(a), -24 * np.cos(a), 14), (0, 0, 1)
                    ),
                    np.float32,
                )
                for a in np.linspace(-0.6, 0.6, views)
            ]
        )
        scene.write("/World/Camera", tokens.POSES, poses)
    return scene


def outputs(renderer, scene, device=None):
    frame = renderer.step(scene, render_products=["/Render/Camera"])["/Render/Camera"].frames[0]
    vars = frame.render_vars
    if device is None:
        return {name: np.array(vars[f"/Render/Vars/{name}"].map()) for name in VARS}
    return {name: vars[f"/Render/Vars/{name}"].map(device=device) for name in VARS}


def assert_geometry_matches(minimal, full):
    hit = full["PrimId"] != 0
    assert hit.mean() > 0.2, "the boxes and ground fill much of the view"
    np.testing.assert_array_equal(minimal["PrimId"], full["PrimId"])
    np.testing.assert_array_equal(minimal["InstanceId"], full["InstanceId"])
    for name in ("DistanceToCamera", "DistanceToImagePlane"):
        assert np.isinf(minimal[name][~hit]).all(), name
        np.testing.assert_allclose(minimal[name][hit], full[name][hit], rtol=1e-4)
    # Octahedral 16-bit normals against float ones.
    np.testing.assert_allclose(minimal["Normal"][hit], full["Normal"][hit], atol=2e-3)


@pytest.mark.parametrize(
    "mode", [tokens.NONE, tokens.CONSTANT_DIFFUSE, tokens.TEXTURED_DIFFUSE]
)
def test_geometry_outputs_match_the_full_pipeline(renderer, mode):
    full = outputs(renderer, boxes_scene())
    minimal = outputs(renderer, boxes_scene(mode))
    assert_geometry_matches(minimal, full)


def test_no_rendering_is_black(renderer):
    color = outputs(renderer, boxes_scene(tokens.NONE))["LdrColor"]
    assert (color[..., :3] == 0).all()
    assert (color[..., 3] == 255).all()


def test_constant_diffuse_lights_one_colour(renderer):
    out = outputs(renderer, boxes_scene(tokens.CONSTANT_DIFFUSE, color=(1.0, 0.5, 0.0)))
    color = out["LdrColor"].astype(np.int32)
    hit = out["PrimId"] != 0
    assert (color[hit][:, 2] == 0).all(), "no blue in an orange constant colour"
    assert (color[hit][:, 0] >= color[hit][:, 1]).all()
    assert (color[~hit][:, :3] == 0).all(), "the background is the clear colour"
    # Lit by the sun from above: the ground's top brighter than shadowed sides at grazing light.
    assert color[hit][:, 0].max() > 128


def test_textured_diffuse_shows_display_colours(renderer):
    out = outputs(renderer, boxes_scene(tokens.TEXTURED_DIFFUSE))
    color = out["LdrColor"].astype(np.int32)
    ids = out["PrimId"]
    boxes = [i for i in np.unique(ids) if i != 0]
    dominant = set()
    for prim in boxes:
        mean = color[ids == prim][:, :3].mean(axis=0)
        if mean.max() > 2 * np.sort(mean)[-2] + 1:
            dominant.add(int(np.argmax(mean)))
    assert dominant == {0, 1, 2}, "a red, a green and a blue box"


def test_a_batch_maps_to_cuda_without_copies(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    scene = boxes_scene(tokens.TEXTURED_DIFFUSE, views=5)
    on_host = outputs(renderer, scene)
    on_gpu = outputs(renderer, scene, device=ernie.Device.CUDA)
    for name in VARS:
        assert on_gpu[name].__dlpack_device__()[0] == 2, name
        np.testing.assert_array_equal(np.array(renderer.cuda_download(on_gpu[name])),
                                      on_host[name], err_msg=name)
    assert on_host["LdrColor"].shape == (5, 64, 96, 4)
    assert on_host["DistanceToCamera"].shape == (5, 64, 96)
    # Each view sees the boxes from its own side.
    assert not np.array_equal(on_host["PrimId"][0], on_host["PrimId"][4])


def test_refuses_what_it_does_not_render(renderer):
    scene = boxes_scene(tokens.TEXTURED_DIFFUSE)
    scene.define("/Render/Vars/Hdr", tokens.RENDER_VAR).set(tokens.SOURCE_NAME, "HdrColor")
    scene.set("/Render/Camera", tokens.ORDERED_VARS, [ernie.Path("/Render/Vars/Hdr")])
    with pytest.raises(ValueError, match="minimal shading does not render"):
        renderer.step(scene, render_products=["/Render/Camera"])
    scene = boxes_scene("glossy")
    with pytest.raises(ValueError, match="not off, none"):
        renderer.step(scene, render_products=["/Render/Camera"])
