# SPDX-License-Identifier: BSD-3-Clause
"""Points and PointInstancers under prims posed on the GPU (particles or parts on a robot's link):
posed on the GPU under their link's pose, rendered against the same links posed on the CPU."""

import numpy as np
import pytest

import ernie
from ernie import tokens
from test_xform_gpu import assert_same_image, cuda_renderer, quats, staged_scene  # noqa: F401

OFFSETS = [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
# The link's points and instances wherever the link goes, in the world's frame.
EXTENT = [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0]


def link_scene(points=True, instancer=True, extent=EXTENT, depth=1):
    """A link per world under `depth` GPU-posed Xforms, holding a cloud of points and a
    PointInstancer of boxes; the sun casts no shadows, whose map is fitted to the bounds (the
    extents on the GPU, the posed prims on the CPU)."""
    scene = staged_scene()
    scene.set("/World/Sun", tokens.SHADOW_ENABLE, False)
    levels = [[] for _ in range(depth)]
    for i, offset in enumerate(OFFSETS):
        scene.define(f"/World/Env{i}", tokens.XFORM).set_xform(translate=offset, scale=2.0)
        path = f"/World/Env{i}"
        for level in range(depth):
            path += "/Link" if level == depth - 1 else "/Base"
            link = scene.define(path, tokens.XFORM)
            link.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
            if extent is not None:
                link.set(tokens.XFORM_EXTENT, extent)
            levels[level].append(path)
        scene.define(f"{path}/Box", tokens.CUBE).set_xform(scale=(0.4, 0.4, 0.4))
        if points:
            cloud = scene.define(f"{path}/Cloud", tokens.GEOM_POINTS).set_xform(translate=(1.5, 0.0, 0.0))
            cloud.write(tokens.POINTS, np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0.5]], np.float32))
            cloud.write(tokens.WIDTHS, np.array([0.8, 0.6, 0.7], np.float32))
            cloud.set(tokens.DISPLAY_COLOR, (0.9, 0.2, 0.2))
        if instancer:
            parts = scene.define(f"{path}/Parts", tokens.POINT_INSTANCER).set_xform(
                translate=(0.0, 1.5, 0.0), rotate=(0.0, 0.0, 30.0)
            )
            parts.write(tokens.POSITIONS, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], np.float32))
            parts.write(tokens.ORIENTATIONS, quats(3, seed=7))
            parts.write(tokens.SCALES, np.array([[1, 1, 1], [0.5, 1, 1], [1, 0.5, 1.5]], np.float32))
            parts.set(tokens.PROTOTYPES, [ernie.Path(f"{path}/Parts/Part")])
            scene.define(f"{path}/Parts/Part", tokens.CUBE).set_xform(scale=(0.3, 0.3, 0.3)).set(
                tokens.DISPLAY_COLOR, (0.2, 0.8, 0.3)
            )
    return scene, levels


def link_arrays(count, seed):
    rng = np.random.default_rng(seed)
    return {
        tokens.TRANSLATE: rng.uniform(-1.5, 1.5, (count, 3)).astype(np.float32),
        tokens.ORIENT: quats(count, seed),
    }


def write_levels(scene, levels, arrays_per_level, upload=None):
    for paths, arrays in zip(levels, arrays_per_level):
        for op, rows in arrays.items():
            scene.write(paths, op, upload(rows) if upload else rows)


def render_both(renderer, arrays_per_level, **scene_args):
    images = []
    for upload in (None, renderer.cuda_upload):
        scene, levels = link_scene(**scene_args)
        write_levels(scene, levels, arrays_per_level, upload)
        images.append(renderer.render(scene))
    return images


@pytest.mark.parametrize(
    "what", [{"instancer": False}, {"points": False}, {}], ids=["points", "instancer", "both"]
)
def test_points_and_instances_on_posed_links(cuda_renderer, what):
    cpu, gpu = render_both(cuda_renderer, [link_arrays(3, seed=1)], **what)
    assert_same_image(gpu, cpu)
    assert cuda_renderer.take_validation_messages() == []


def test_two_posed_links_deep(cuda_renderer):
    cpu, gpu = render_both(cuda_renderer, [link_arrays(3, seed=2), link_arrays(3, seed=3)], depth=2)
    assert_same_image(gpu, cpu)


def test_points_and_instances_follow_their_links_step_by_step(cuda_renderer):
    scene, levels = link_scene()
    previous = None
    for step in range(3):
        arrays = link_arrays(3, seed=20 + step)
        write_levels(scene, levels, [arrays], cuda_renderer.cuda_upload)
        gpu = cuda_renderer.render(scene)
        reference, reference_levels = link_scene()
        write_levels(reference, reference_levels, [arrays])
        assert_same_image(gpu, cuda_renderer.render(reference))
        if previous is not None:
            assert np.abs(gpu.astype(np.int32) - previous).max() > 50, "they moved with the links"
        previous = gpu.astype(np.int32)


def test_gpu_points_and_instances_on_posed_links(cuda_renderer):
    """Clouds and instances whose own arrays are on the GPU too."""
    arrays = link_arrays(3, seed=4)
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene, levels = link_scene()
        for i in range(3):
            link = levels[0][i]
            for path, attribute in ((f"{link}/Cloud", tokens.POINTS), (f"{link}/Parts", tokens.POSITIONS)):
                rows = np.array(scene.read(path, attribute))
                scene.write(path, attribute, upload(rows) if upload else rows)
        write_levels(scene, levels, [arrays], upload)
        images.append(cuda_renderer.render(scene))
    assert_same_image(images[1], images[0])


def test_points_under_posed_links_need_their_extent(cuda_renderer):
    scene, levels = link_scene(instancer=False, extent=None)
    write_levels(scene, levels, [link_arrays(3, seed=5)], cuda_renderer.cuda_upload)
    with pytest.raises(ValueError, match="xform:extent"):
        cuda_renderer.render(scene)
    scene, levels = link_scene(points=False, extent=None)
    write_levels(scene, levels, [link_arrays(3, seed=5)], cuda_renderer.cuda_upload)
    with pytest.raises(ValueError, match="xform:extent"):
        cuda_renderer.render(scene)
