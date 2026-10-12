# SPDX-License-Identifier: BSD-3-Clause
"""Lights under prims posed on the GPU (a lamp on a robot's link):
planned on the CPU from their links' rows copied back, rendered
against the same links posed on the CPU."""

import numpy as np
from test_xform_gpu import (  # noqa: F401
    assert_same_image,
    cuda_renderer,
    quats,
)

import ernie
from ernie import tokens

OFFSETS = [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]


def lamp_scene(light=ernie.SphereLight, depth=1, lamp_args=None):
    """A lamp (a light class, its attributes `lamp_args`) on a link
    per world over a ground, lit by nothing else, under `depth`
    GPU-posed Xforms."""
    scene = ernie.Scene()
    camera = scene.define(
        "/World/Camera", ernie.Camera, focal_length=35.0
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0, -24, 20), (0, 0, 0)),
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera=camera,
        resolution=(96, 64),
    )
    scene.define("/World/Ground", ernie.GroundPlane)
    levels = [[] for _ in range(depth)]
    for i, offset in enumerate(OFFSETS):
        link = scene.define(
            f"/World/Env{i}", ernie.Xform, translate=offset
        )
        for level in range(depth):
            link = (
                link.define(
                    "Link" if level == depth - 1 else "Base",
                    ernie.Xform,
                )
                .set(
                    tokens.XFORM_OP_ORDER,
                    [tokens.TRANSLATE, tokens.ORIENT],
                )
                .set(
                    tokens.XFORM_EXTENT,
                    [-6.0, -6.0, -1.0, 6.0, 6.0, 6.0],
                )
            )
            levels[level].append(str(link.path))
        link.define("Box", ernie.Cube, scale=(0.5, 0.5, 0.5))
        lamp = link.define(
            "Lamp",
            light,
            translate=(1.0, 0.0, 2.0),
            **(lamp_args or {"intensity": 40.0}),
        )
        if light is ernie.SphereLight:
            lamp.radius = 0.2
    return scene, levels


def link_arrays(count, seed):
    rng = np.random.default_rng(seed)
    translate = rng.uniform(-1.5, 1.5, (count, 3))
    translate[:, 2] = rng.uniform(0.5, 2.0, count)
    return {
        tokens.TRANSLATE: translate.astype(np.float32),
        tokens.ORIENT: quats(count, seed),
    }


def render_both(renderer, arrays_per_level, **scene_args):
    images = []
    for upload in (None, renderer.cuda_upload):
        scene, levels = lamp_scene(**scene_args)
        for paths, arrays in zip(levels, arrays_per_level):
            for op, rows in arrays.items():
                scene.write(
                    paths, op, upload(rows) if upload else rows
                )
        images.append(renderer.render(scene))
    return images


def test_lamps_on_posed_links_light_like_cpu_posed_ones(
    cuda_renderer,
):
    cpu, gpu = render_both(cuda_renderer, [link_arrays(3, seed=1)])
    assert_same_image(gpu, cpu)
    assert cuda_renderer.take_validation_messages() == []


def test_laser_spots_on_posed_links_shadow_like_cpu_posed_ones(
    cuda_renderer,
):
    # A beam a metre wide from beside the box, which shadows part of
    # it: the light pass fits its view along the posed beam.
    args = {
        "light": ernie.LaserLight,
        "lamp_args": {"power": 5.0, "waist_radius": (1.0, 1.0)},
    }
    cpu, gpu = render_both(
        cuda_renderer, [link_arrays(3, seed=4)], **args
    )
    assert_same_image(gpu, cpu)
    unshadowed, _ = render_both(
        cuda_renderer,
        [link_arrays(3, seed=4)],
        light=ernie.LaserLight,
        lamp_args={**args["lamp_args"], "shadow_enable": False},
    )
    difference = np.abs(cpu.astype(np.int32) - unshadowed)
    assert difference.max() > 20, "the box shadows the beam"


def test_lamps_two_posed_links_deep(cuda_renderer):
    cpu, gpu = render_both(
        cuda_renderer,
        [link_arrays(3, seed=2), link_arrays(3, seed=3)],
        depth=2,
    )
    assert_same_image(gpu, cpu)


def test_lamps_follow_their_links_step_by_step(cuda_renderer):
    scene, levels = lamp_scene()
    previous = None
    for step in range(3):
        arrays = link_arrays(3, seed=10 + step)
        for op, rows in arrays.items():
            scene.write(
                levels[0], op, cuda_renderer.cuda_upload(rows)
            )
        gpu = cuda_renderer.render(scene).astype(np.int32)
        cpu, _ = render_both(cuda_renderer, [arrays])
        assert_same_image(gpu, cpu)
        if previous is not None:
            assert np.abs(gpu - previous).max() > 20, (
                "the light moved with the links"
            )
        previous = gpu


def test_lamps_posed_on_the_gpu_move_across_the_shutter(
    cuda_renderer,
):
    # Lamps on links sliding 1.5 m along x over a frame: blurred, the
    # frame is the shutter's weighting of sharp frames along the way,
    # each sub-frame posed on the GPU from the last frame's poses.
    scene, levels = lamp_scene()
    camera = ernie.Camera.at(scene, "/World/Camera")
    hdr = scene.define("/Render/Out/HdrColor", ernie.RenderVar)
    hdr.source_name = "HdrColor"
    scene.define(
        "/Render/Out",
        ernie.RenderProduct,
        camera=camera,
        resolution=(96, 64),
        ordered_vars=[hdr.path],
    )
    rows = link_arrays(3, seed=20)

    def move(s):
        translate = rows[tokens.TRANSLATE].copy()
        translate[:, 0] += 1.5 * s
        scene.write(
            levels[0],
            tokens.TRANSLATE,
            cuda_renderer.cuda_upload(translate),
        )
        scene.write(
            levels[0],
            tokens.ORIENT,
            cuda_renderer.cuda_upload(rows[tokens.ORIENT]),
        )

    def step():
        outputs = cuda_renderer.step(
            scene,
            render_products=["/Render/Out"],
            delta_time=1.0 / 30.0,
        )
        return np.array(
            outputs["/Render/Out"]
            .frames[0]
            .render_vars["/Render/Out/HdrColor"]
            .map()
        )[..., :3].astype(np.float64)

    def shutter(opening, samples=5):
        camera.set(
            shutter_open=opening,
            shutter_close=0.0,
            shutter_samples=samples,
        )

    samples = 5
    weights = np.full(samples, 1.0 / (samples - 1))
    weights[[0, -1]] *= 0.5
    shutter(0.0)
    expected = 0.0
    for k in range(samples):
        move(k / (samples - 1))
        step()  # the poses' last frame, for the next
        expected = expected + weights[k] * step()
    sharp = step()
    shutter(-1.0)
    move(0.0)
    step()
    move(1.0)
    blurred = step()
    peak = expected.max()
    assert np.abs(blurred - expected).max() < 0.01 * peak
    assert np.abs(blurred - sharp).max() > 0.05 * peak, "blurred"
