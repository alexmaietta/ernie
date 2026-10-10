# SPDX-License-Identifier: BSD-3-Clause
"""Lights under prims posed on the GPU (a lamp on a robot's link): planned on the CPU from their
links' rows copied back, rendered against the same links posed on the CPU."""

import numpy as np

import ernie
from ernie import tokens
from test_xform_gpu import assert_same_image, cuda_renderer, quats  # noqa: F401

OFFSETS = [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]


def lamp_scene(light=ernie.SphereLight, depth=1):
    """A lamp (a light class) on a link per world over a ground, lit by nothing else, under
    `depth` GPU-posed Xforms."""
    scene = ernie.Scene()
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=35.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, -24, 20), (0, 0, 0))
    )
    scene.define("/Render/Settings", ernie.RenderSettings, camera=camera, resolution=(96, 64))
    scene.define("/World/Ground", ernie.GroundPlane)
    levels = [[] for _ in range(depth)]
    for i, offset in enumerate(OFFSETS):
        link = scene.define(f"/World/Env{i}", ernie.Xform, translate=offset)
        for level in range(depth):
            link = link.define("Link" if level == depth - 1 else "Base", ernie.Xform).set(
                tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT]
            ).set(tokens.XFORM_EXTENT, [-6.0, -6.0, -1.0, 6.0, 6.0, 6.0])
            levels[level].append(str(link.path))
        link.define("Box", ernie.Cube, scale=(0.5, 0.5, 0.5))
        lamp = link.define("Lamp", light, translate=(1.0, 0.0, 2.0), intensity=40.0)
        if light is ernie.SphereLight:
            lamp.radius = 0.2
    return scene, levels


def link_arrays(count, seed):
    rng = np.random.default_rng(seed)
    translate = rng.uniform(-1.5, 1.5, (count, 3))
    translate[:, 2] = rng.uniform(0.5, 2.0, count)
    return {tokens.TRANSLATE: translate.astype(np.float32), tokens.ORIENT: quats(count, seed)}


def render_both(renderer, arrays_per_level, **scene_args):
    images = []
    for upload in (None, renderer.cuda_upload):
        scene, levels = lamp_scene(**scene_args)
        for paths, arrays in zip(levels, arrays_per_level):
            for op, rows in arrays.items():
                scene.write(paths, op, upload(rows) if upload else rows)
        images.append(renderer.render(scene))
    return images


def test_lamps_on_posed_links_light_like_cpu_posed_ones(cuda_renderer):
    cpu, gpu = render_both(cuda_renderer, [link_arrays(3, seed=1)])
    assert_same_image(gpu, cpu)
    assert cuda_renderer.take_validation_messages() == []


def test_lamps_two_posed_links_deep(cuda_renderer):
    cpu, gpu = render_both(cuda_renderer, [link_arrays(3, seed=2), link_arrays(3, seed=3)], depth=2)
    assert_same_image(gpu, cpu)


def test_lamps_follow_their_links_step_by_step(cuda_renderer):
    scene, levels = lamp_scene()
    previous = None
    for step in range(3):
        arrays = link_arrays(3, seed=10 + step)
        for op, rows in arrays.items():
            scene.write(levels[0], op, cuda_renderer.cuda_upload(rows))
        gpu = cuda_renderer.render(scene).astype(np.int32)
        cpu, _ = render_both(cuda_renderer, [arrays])
        assert_same_image(gpu, cpu)
        if previous is not None:
            assert np.abs(gpu - previous).max() > 20, "the light moved with the links"
        previous = gpu
