# SPDX-License-Identifier: BSD-3-Clause
"""Transform ops composed on the GPU: Euler rotations, rotations about one axis, matrices, other op
orders, and GPU-posed prims under GPU-posed prims, each rendered against the same ops composed on
the CPU."""

import numpy as np
import pytest

import ernie
from ernie import tokens
from test_xform_gpu import assert_same_image, cuda_renderer, quats, staged_scene  # noqa: F401

EXTENT = [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0]
EULER_OPS = [
    tokens.ROTATE_XYZ,
    tokens.ROTATE_XZY,
    tokens.ROTATE_YXZ,
    tokens.ROTATE_YZX,
    tokens.ROTATE_ZXY,
    tokens.ROTATE_ZYX,
]


def links(scene, count, ops, offsets=None, box_scale=(2.0, 1.0, 1.0)):
    """One link per world, its box moved off its origin, the link's ops in `ops`."""
    paths = []
    for i in range(count):
        world = scene.define(
            f"/World/Env{i}",
            ernie.Xform,
            translate=offsets[i] if offsets is not None else (-6.0 + 4.0 * i, 0.0, 0.0),
        )
        link = world.define("Link", ernie.Xform)
        link.set(tokens.XFORM_OP_ORDER, list(ops)).set(tokens.XFORM_EXTENT, EXTENT)
        link.define("Box", ernie.Cube, translate=(1.0, 0.0, 0.0), scale=box_scale)
        paths.append(str(link.path))
    return paths


def write(scene, paths, arrays, upload=None):
    for op, rows in arrays.items():
        scene.write(paths, op, upload(rows) if upload else rows)


def render_both(renderer, count, ops, arrays, configure=None):
    """The same links and op arrays, posed from CPU arrays and from CUDA arrays."""
    images = []
    for upload in (None, renderer.cuda_upload):
        scene = staged_scene()
        paths = links(scene, count, ops)
        if configure:
            configure(scene, paths)
        write(scene, paths, arrays, upload)
        images.append(renderer.render(scene))
    return images


def rng_arrays(count, ops, seed):
    rng = np.random.default_rng(seed)
    arrays = {}
    for op in ops:
        if op == tokens.TRANSLATE:
            arrays[op] = rng.uniform(-1.5, 1.5, (count, 3))
        elif op == tokens.ORIENT:
            arrays[op] = quats(count, seed)
        elif op == tokens.SCALE:
            arrays[op] = rng.uniform(0.6, 1.4, (count, 3))
        elif op in (tokens.ROTATE_X, tokens.ROTATE_Y, tokens.ROTATE_Z):
            arrays[op] = rng.uniform(-170, 170, (count, 1))
        elif op == tokens.TRANSFORM:
            m = np.zeros((count, 16))
            for i in range(count):
                q = quats(1, seed + i)[0]
                x, y, z, w = q
                r = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    ]
                )
                full = np.eye(4)
                full[:3, :3] = r * rng.uniform(0.7, 1.3)  # rotation and a uniform scale
                full[:3, 3] = rng.uniform(-1.5, 1.5, 3)
                m[i] = full.reshape(16)
            arrays[op] = m
        else:  # Euler angles
            arrays[op] = rng.uniform(-80, 80, (count, 3))
    return {op: np.asarray(rows, np.float32) for op, rows in arrays.items()}


@pytest.mark.parametrize("euler", EULER_OPS)
def test_every_euler_order_composes_like_the_cpu(cuda_renderer, euler):
    ops = (tokens.TRANSLATE, euler, tokens.SCALE)
    cpu, gpu = render_both(cuda_renderer, 4, ops, rng_arrays(4, ops, seed=1))
    assert_same_image(gpu, cpu)
    assert cuda_renderer.take_validation_messages() == []


@pytest.mark.parametrize(
    "ops",
    [
        (tokens.ROTATE_X,),
        (tokens.TRANSLATE, tokens.ROTATE_Y),
        (tokens.ROTATE_Z, tokens.TRANSLATE),
        (tokens.TRANSFORM,),
        (tokens.TRANSLATE, tokens.TRANSFORM, tokens.ROTATE_ZYX),
        (tokens.ORIENT, tokens.TRANSLATE, tokens.ROTATE_X),
        (tokens.SCALE, tokens.TRANSLATE),
    ],
)
def test_other_ops_and_orders_compose_like_the_cpu(cuda_renderer, ops):
    cpu, gpu = render_both(cuda_renderer, 4, ops, rng_arrays(4, ops, seed=2))
    assert_same_image(gpu, cpu)


def test_uniform_scales_of_one_number(cuda_renderer):
    ops = (tokens.TRANSLATE, tokens.SCALE)
    arrays = {tokens.TRANSLATE: np.zeros((4, 3), np.float32), tokens.SCALE: np.array([[0.5], [1.0], [1.5], [2.0]], np.float32)}
    cpu, gpu = render_both(cuda_renderer, 4, ops, arrays)
    assert_same_image(gpu, cpu)


def test_w_first_orientations_in_composed_poses(cuda_renderer):
    ops = (tokens.ROTATE_X, tokens.ORIENT)
    arrays = rng_arrays(4, ops, seed=3)
    arrays[tokens.ORIENT] = np.ascontiguousarray(np.roll(arrays[tokens.ORIENT], 1, axis=1))

    def w_first(scene, paths):
        for path in paths:
            scene.set(path, tokens.QUATERNION_ORDER, tokens.WXYZ)

    cpu, gpu = render_both(cuda_renderer, 4, ops, arrays, w_first)
    assert_same_image(gpu, cpu)


def test_composed_poses_step_on_the_fast_path(cuda_renderer):
    ops = (tokens.TRANSLATE, tokens.ROTATE_ZYX)
    gpu_scene = staged_scene()
    paths = ernie.PrimSet(links(gpu_scene, 4, ops))
    for step in range(3):
        arrays = rng_arrays(4, ops, seed=10 + step)
        write(gpu_scene, paths, arrays, cuda_renderer.cuda_upload)
        image = cuda_renderer.render(gpu_scene)
        cpu_scene = staged_scene()
        write(cpu_scene, links(cpu_scene, 4, ops), arrays)
        assert_same_image(image, cuda_renderer.render(cpu_scene))


# --- Nested GPU-posed prims -------------------------------------------------------------------


def arm(scene, count):
    """Two-link arms: link1 under a CPU-posed elbow Xform under link0, a box under each, both
    links posed by GPU rows (translate, then a rotation about Y)."""
    link0, link1 = [], []
    for i in range(count):
        scene.define(f"/World/Env{i}", ernie.Xform, translate=(-6.0 + 6.0 * i, 0.0, 0.0))
        for path, length in ((f"/World/Env{i}/Link0", 2.0), (f"/World/Env{i}/Link0/Elbow/Link1", 1.5)):
            link = scene.define(path, ernie.Xform).set(
                tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ROTATE_Y]
            ).set(tokens.XFORM_EXTENT, EXTENT)
            link.define(
                "Box", ernie.Cube, translate=(length / 2, 0.0, 0.0), scale=(length, 0.4, 0.4)
            )
        # The elbow: at the end of link0, turned a little, on the CPU.
        scene.get_prim(f"/World/Env{i}/Link0/Elbow").set_xform(
            translate=(2.0, 0.0, 0.0), rotate=(0.0, 0.0, 15.0)
        )
        link0.append(f"/World/Env{i}/Link0")
        link1.append(f"/World/Env{i}/Link0/Elbow/Link1")
    return link0, link1


def arm_arrays(count, seed):
    rng = np.random.default_rng(seed)
    return (
        {tokens.TRANSLATE: rng.uniform(-1, 1, (count, 3)).astype(np.float32),
         tokens.ROTATE_Y: rng.uniform(-60, 60, (count, 1)).astype(np.float32)},
        {tokens.TRANSLATE: np.zeros((count, 3), np.float32),
         tokens.ROTATE_Y: rng.uniform(-90, 90, (count, 1)).astype(np.float32)},
    )


def test_nested_prims_from_two_writes(cuda_renderer):
    first, second = arm_arrays(3, seed=20)
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene = staged_scene()
        link0, link1 = arm(scene, 3)
        write(scene, link0, first, upload)
        write(scene, link1, second, upload)
        images.append(cuda_renderer.render(scene))
    assert_same_image(images[1], images[0])
    assert cuda_renderer.take_validation_messages() == []


def test_nested_prims_in_one_array(cuda_renderer):
    """Parents and children as rows of the same arrays: composed level by level."""
    first, second = arm_arrays(3, seed=21)
    # Children first in the rows, so composing in row order alone would read unwritten parents.
    rows = {op: np.concatenate([second[op], first[op]]) for op in first}
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene = staged_scene()
        link0, link1 = arm(scene, 3)
        write(scene, link1 + link0, rows, upload)
        images.append(cuda_renderer.render(scene))
    assert_same_image(images[1], images[0])


def test_children_follow_a_parent_written_alone(cuda_renderer):
    """Only link0 rewritten: link1, posed under it, moves with it."""
    first, second = arm_arrays(3, seed=22)
    gpu_scene = staged_scene()
    link0, link1 = arm(gpu_scene, 3)
    write(gpu_scene, link1, second, cuda_renderer.cuda_upload)
    for step in range(3):
        moved = {op: (rows + 25.0 * step if op == tokens.ROTATE_Y else rows) for op, rows in first.items()}
        write(gpu_scene, link0, {op: np.asarray(rows, np.float32) for op, rows in moved.items()}, cuda_renderer.cuda_upload)
        image = cuda_renderer.render(gpu_scene)
        cpu_scene = staged_scene()
        c0, c1 = arm(cpu_scene, 3)
        write(cpu_scene, c0, {op: np.asarray(rows, np.float32) for op, rows in moved.items()})
        write(cpu_scene, c1, second)
        assert_same_image(image, cuda_renderer.render(cpu_scene))


def test_nested_under_plain_gpu_parents(cuda_renderer):
    """A composed child under a parent whose ops are plain instance arrays (translate, orient)."""
    rng = np.random.default_rng(23)
    parent = {tokens.TRANSLATE: rng.uniform(-1, 1, (3, 3)).astype(np.float32), tokens.ORIENT: quats(3, 23)}
    child = {tokens.TRANSLATE: np.zeros((3, 3), np.float32), tokens.ROTATE_Y: rng.uniform(-90, 90, (3, 1)).astype(np.float32)}
    images = []
    for upload in (None, cuda_renderer.cuda_upload):
        scene = staged_scene()
        link0, link1 = arm(scene, 3)
        for path in link0:
            scene.set(path, tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
        write(scene, link0, parent, upload)
        write(scene, link1, child, upload)
        images.append(cuda_renderer.render(scene))
    assert_same_image(images[1], images[0])


def test_matrices_as_4_by_4_rows(cuda_renderer):
    arrays = rng_arrays(4, (tokens.TRANSFORM,), seed=30)
    square = {tokens.TRANSFORM: np.ascontiguousarray(arrays[tokens.TRANSFORM].reshape(4, 4, 4))}
    cpu, flat = render_both(cuda_renderer, 4, (tokens.TRANSFORM,), arrays)
    _, gpu_square = render_both(cuda_renderer, 4, (tokens.TRANSFORM,), square)
    assert_same_image(flat, cpu)
    assert_same_image(gpu_square, cpu)
