# SPDX-License-Identifier: BSD-3-Clause
"""Cameras under prims posed on the GPU (a camera on a robot's link): each view posed on the GPU
under its link's pose, rendered against the same links posed on the CPU."""

import numpy as np
import pytest

import ernie
from ernie import tokens
from test_xform_gpu import assert_same_image, cuda_renderer, quats  # noqa: F401

# The link's box wherever the link goes, in the world's frame.
EXTENT = [-2.0, -2.0, -0.5, 2.0, 2.0, 0.5]


def world_scene(count, ops=(tokens.TRANSLATE, tokens.ORIENT)):
    """`count` worlds, each a link posed by `ops` with a box and a camera on it, looking along
    the link at a few boxes of the world's own; one product renders every world's camera."""
    scene = ernie.Scene()
    # No shadows: the sun's map is fitted to the scene's bounds, the extents on the GPU and the
    # posed boxes on the CPU, shading the ground a little differently.
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, -0.5, 0.8))
    ).set(tokens.SHADOW_ENABLE, False)
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    links, cameras = [], []
    for i in range(count):
        env = scene.define(f"/Envs/Env{i}", tokens.XFORM).set(tokens.WORLD, i)
        env.set_xform(translate=(100.0 * i, 0.0, 0.0))
        for k in range(3):
            scene.define(f"/Envs/Env{i}/Box{k}", tokens.CUBE).set_xform(
                translate=(4.0 + 2.0 * k, -2.0 + 2.0 * k, 0.5 + k)
            ).set(tokens.DISPLAY_COLOR, (0.2 + 0.3 * k, 0.5, 0.9 - 0.3 * k))
        link = scene.define(f"/Envs/Env{i}/Link", tokens.XFORM)
        link.set(tokens.XFORM_OP_ORDER, list(ops)).set(tokens.XFORM_EXTENT, EXTENT)
        scene.define(f"/Envs/Env{i}/Link/Box", tokens.CUBE).set_xform(scale=(0.5, 0.5, 0.5))
        # The camera: up the link and turned to look along +x (cameras look down their -z).
        scene.define(f"/Envs/Env{i}/Link/Mount", tokens.XFORM).set_xform(translate=(0.0, 0.0, 1.0))
        camera = scene.define(f"/Envs/Env{i}/Link/Mount/Camera", tokens.CAMERA)
        camera.set(tokens.TRANSFORM, ernie.look_at_transform((0, 0, 0), (1, 0, -0.2)))
        camera.set(tokens.FOCAL_LENGTH, 12.0)
        links.append(str(link.path))
        cameras.append(ernie.Path(f"/Envs/Env{i}/Link/Mount/Camera"))
    scene.define("/Render/Wrists", tokens.RENDER_PRODUCT).set(tokens.CAMERA_REL, cameras).set(
        tokens.RESOLUTION, (64, 48)
    )
    return scene, links


def link_arrays(count, seed):
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-0.4, 0.4, count)  # radians about z
    orientations = np.zeros((count, 4), np.float32)
    orientations[:, 2] = np.sin(yaw / 2)
    orientations[:, 3] = np.cos(yaw / 2)
    return {
        tokens.TRANSLATE: (rng.uniform(-1, 1, (count, 3)) * [1, 1, 0]).astype(np.float32),
        tokens.ORIENT: orientations,
    }


def views(renderer, scene, product="/Render/Wrists"):
    outputs = renderer.step(scene, render_products=[product])[product]
    return np.array(outputs.frames[0].render_vars[f"{product}/LdrColor"].map()).astype(np.int32)


def render_both(renderer, count, arrays, scene_of=world_scene):
    images = []
    for upload in (None, renderer.cuda_upload):
        scene, links = scene_of(count)
        for op, rows in arrays.items():
            scene.write(links, op, upload(rows) if upload else rows)
        images.append(views(renderer, scene))
    return images


def test_wrist_cameras_see_like_cpu_posed_ones(cuda_renderer):
    cpu, gpu = render_both(cuda_renderer, 4, link_arrays(4, seed=1))
    assert cpu.shape[0] == 4
    for view in range(4):
        assert_same_image(gpu[view], cpu[view])
    assert cuda_renderer.take_validation_messages() == []


def test_each_wrist_camera_sees_its_own_world(cuda_renderer):
    """Worlds move their links differently: views differ as their links do."""
    arrays = link_arrays(2, seed=2)
    arrays[tokens.ORIENT][:] = [0.0, 0.0, 0.0, 1.0]
    arrays[tokens.TRANSLATE][:] = 0.0
    _, same = render_both(cuda_renderer, 2, arrays)
    np.testing.assert_array_equal(same[0], same[1])
    arrays[tokens.ORIENT][1] = [0.0, 0.0, np.sin(0.25), np.cos(0.25)]  # world 1 turned
    _, turned = render_both(cuda_renderer, 2, arrays)
    np.testing.assert_array_equal(turned[0], same[0])
    assert np.abs(turned[1] - same[1]).max() > 50


def test_one_wrist_camera(cuda_renderer):
    arrays = link_arrays(3, seed=3)
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene, links = world_scene(3)
        scene.define("/Render/One", tokens.RENDER_PRODUCT).set(
            tokens.CAMERA_REL, ernie.Path("/Envs/Env1/Link/Mount/Camera")
        ).set(tokens.RESOLUTION, (64, 48))
        for op, rows in arrays.items():
            scene.write(links, op, upload(rows) if upload else rows)
        images.append(views(cuda_renderer, scene, "/Render/One"))
    assert_same_image(images[1], images[0])


def test_wrist_cameras_follow_their_links_step_by_step(cuda_renderer):
    scene, links = world_scene(3)
    prims = ernie.PrimSet(links)
    previous = None
    for step in range(3):
        arrays = link_arrays(3, seed=10 + step)
        for op, rows in arrays.items():
            scene.write(prims, op, cuda_renderer.cuda_upload(rows))
        gpu = views(cuda_renderer, scene)
        reference, ref_links = world_scene(3)
        for op, rows in arrays.items():
            reference.write(ref_links, op, rows)
        cpu = views(cuda_renderer, reference)
        for view in range(3):
            assert_same_image(gpu[view], cpu[view])
        if previous is not None:
            assert np.abs(gpu - previous).max() > 50, "the views moved with the links"
        previous = gpu


def test_wrist_cameras_under_composed_links(cuda_renderer):
    """Links posed by Euler angles, composed on the GPU before the views read them."""
    ops = (tokens.TRANSLATE, tokens.ROTATE_ZYX)
    rng = np.random.default_rng(4)
    arrays = {
        tokens.TRANSLATE: (rng.uniform(-1, 1, (3, 3)) * [1, 1, 0]).astype(np.float32),
        tokens.ROTATE_ZYX: np.stack([np.zeros(3), np.zeros(3), rng.uniform(-25, 25, 3)], 1).astype(np.float32),
    }
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene, links = world_scene(3, ops)
        for op, rows in arrays.items():
            scene.write(links, op, upload(rows) if upload else rows)
        images.append(views(cuda_renderer, scene))
    for view in range(3):
        assert_same_image(images[1][view], images[0][view])


def test_wrist_cameras_two_posed_links_deep(cuda_renderer):
    """A camera on a link under a base, both posed on the GPU: the views read the composed pose."""
    count = 3
    base_arrays = link_arrays(count, seed=5)
    arrays = link_arrays(count, seed=6)
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene, links = world_scene(count)
        bases = []
        for i in range(count):
            # The link moves under a new base between the world and it.
            base = scene.define(f"/Envs/Env{i}/Base", tokens.XFORM)
            base.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
            base.set(tokens.XFORM_EXTENT, [-4.0, -4.0, -0.5, 4.0, 4.0, 0.5])
            scene.define(f"/Envs/Env{i}/Base/Link", tokens.XFORM).set(
                tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT]
            ).set(tokens.XFORM_EXTENT, EXTENT)
            scene.define(f"/Envs/Env{i}/Base/Link/Box", tokens.CUBE).set_xform(scale=(0.5, 0.5, 0.5))
            scene.define(f"/Envs/Env{i}/Base/Link/Mount", tokens.XFORM).set_xform(translate=(0.0, 0.0, 1.0))
            scene.define(f"/Envs/Env{i}/Base/Link/Mount/Camera", tokens.CAMERA).set(
                tokens.TRANSFORM, ernie.look_at_transform((0, 0, 0), (1, 0, -0.2))
            ).set(tokens.FOCAL_LENGTH, 12.0)
            scene.remove(links[i])
            bases.append(str(base.path))
        cameras = [ernie.Path(f"/Envs/Env{i}/Base/Link/Mount/Camera") for i in range(count)]
        scene.set("/Render/Wrists", tokens.CAMERA_REL, cameras)
        for paths, rows_of in ((bases, base_arrays), ([f"{b}/Link" for b in bases], arrays)):
            for op, rows in rows_of.items():
                scene.write(paths, op, upload(rows) if upload else rows)
        images.append(views(cuda_renderer, scene))
    for view in range(count):
        assert_same_image(images[1][view], images[0][view])


def test_wrist_cameras_take_no_poses(cuda_renderer):
    scene, links = world_scene(2)
    scene.write(links, tokens.TRANSLATE, cuda_renderer.cuda_upload(np.zeros((2, 3), np.float32)))
    scene.write(links, tokens.ORIENT, cuda_renderer.cuda_upload(quats(2)))
    scene.write("/Envs/Env0/Link/Mount/Camera", tokens.POSES, np.eye(4, dtype=np.float32)[None])
    with pytest.raises(ValueError, match="takes no"):
        views(cuda_renderer, scene)
