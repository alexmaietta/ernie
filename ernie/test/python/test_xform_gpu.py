# SPDX-License-Identifier: BSD-3-Clause
"""Transform ops written as rows of CUDA arrays: prims posed on the GPU, rendered against the same
poses on the CPU."""

import numpy as np
import pytest

import ernie
from ernie import tokens


@pytest.fixture
def cuda_renderer(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    return renderer


def staged_scene():
    scene = ernie.Scene()
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=35.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, -30, 18), (0, 0, 0))
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera=camera,
        resolution=(96, 64),
        clear_color=(0.0, 0.0, 0.0),
    )
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, -0.5, 0.8))
    )
    return scene


def quats(count, seed=1):
    """Unit quaternions as rows of x, y, z, w."""
    q = np.random.default_rng(seed).normal(size=(count, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def robots(scene, count, offsets=None, ops=(tokens.TRANSLATE, tokens.ORIENT, tokens.SCALE)):
    """One link per world: an Xform posed by the ops, a box under it moved off its origin, under a
    world prim offset on the CPU."""
    links = []
    for i in range(count):
        world = scene.define(f"/World/Env{i}", ernie.Xform)
        if offsets is not None:
            world.set(translate=offsets[i])
        link = world.define("Link", ernie.Xform).set(tokens.XFORM_OP_ORDER, list(ops))
        link.define("Box", ernie.Cube, translate=(1.0, 0.0, 0.0), scale=(2.0, 1.0, 1.0))
        links.append(str(link.path))
    return links


def pose_arrays(count, seed=1):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-6, 6, (count, 3)).astype(np.float32)
    scales = rng.uniform(0.5, 1.5, (count, 3)).astype(np.float32)
    return positions, quats(count, seed), scales


def write_poses(scene, links, arrays, upload=None):
    positions, orientations, scales = arrays
    for op, rows in ((tokens.TRANSLATE, positions), (tokens.ORIENT, orientations), (tokens.SCALE, scales)):
        if rows is not None:
            scene.write(links, op, upload(rows) if upload else rows)


def assert_same_image(a, b, tolerance=2):
    a, b = np.asarray(a).astype(np.int32), np.asarray(b).astype(np.int32)
    assert (a[..., :3] > 0).any(-1).mean() > 0.03, "the prims fill part of the view"
    # The GPU composes in float32 what the CPU composes in double, and bounds fit the shadows: a
    # few edge pixels differ. A pose off by a step differs on a fifth of the image or more.
    differing = (np.abs(a - b).max(-1) > tolerance).mean()
    assert differing < 0.01, differing


def test_gpu_poses_render_like_cpu_ones(cuda_renderer):
    arrays = pose_arrays(6)
    on_cpu, on_gpu = staged_scene(), staged_scene()
    write_poses(on_cpu, robots(on_cpu, 6), arrays)
    write_poses(on_gpu, robots(on_gpu, 6), arrays, cuda_renderer.cuda_upload)
    with pytest.raises(ValueError, match="GPU array"):
        on_gpu.local_transform("/World/Env0/Link")
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))
    assert cuda_renderer.take_validation_messages() == []


def test_parents_and_children_compose_around_the_gpu_pose(cuda_renderer):
    offsets = [(-8.0 + 4.0 * i, 2.0 * (i % 2), 0.0) for i in range(5)]
    arrays = pose_arrays(5, seed=2)
    on_cpu, on_gpu = staged_scene(), staged_scene()
    write_poses(on_cpu, robots(on_cpu, 5, offsets), arrays)
    write_poses(on_gpu, robots(on_gpu, 5, offsets), arrays, cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))


def test_extents_spare_the_host_copy_and_render_the_same(cuda_renderer):
    arrays = pose_arrays(4, seed=3)
    unbounded, bounded = staged_scene(), staged_scene()
    write_poses(unbounded, robots(unbounded, 4), arrays, cuda_renderer.cuda_upload)
    links = robots(bounded, 4)
    for link in links:
        bounded.set(link, tokens.XFORM_EXTENT, [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0])
    write_poses(bounded, links, arrays, cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(bounded), cuda_renderer.render(unbounded))


@pytest.mark.parametrize(
    "ops",
    [
        (tokens.TRANSLATE,),
        (tokens.ORIENT,),
        (tokens.SCALE,),
        (tokens.TRANSLATE, tokens.ORIENT),
        (tokens.ORIENT, tokens.SCALE),
    ],
)
def test_any_subset_of_the_ops(cuda_renderer, ops):
    positions, orientations, scales = pose_arrays(3, seed=4)
    arrays = (
        positions if tokens.TRANSLATE in ops else None,
        orientations if tokens.ORIENT in ops else None,
        scales if tokens.SCALE in ops else None,
    )
    offsets = [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
    on_cpu, on_gpu = staged_scene(), staged_scene()
    write_poses(on_cpu, robots(on_cpu, 3, offsets, ops), arrays)
    write_poses(on_gpu, robots(on_gpu, 3, offsets, ops), arrays, cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))


def test_reset_xform_stack_ignores_the_cpu_parent(cuda_renderer):
    arrays = pose_arrays(3, seed=5)
    offsets = [(100.0, 0.0, 0.0)] * 3  # would put the boxes out of view
    ops = (tokens.RESET_XFORM_STACK, tokens.TRANSLATE, tokens.ORIENT, tokens.SCALE)
    on_cpu, on_gpu = staged_scene(), staged_scene()
    write_poses(on_cpu, robots(on_cpu, 3, offsets, ops), arrays)
    write_poses(on_gpu, robots(on_gpu, 3, offsets, ops), arrays, cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))


def test_stepping_moves_the_prims_with_new_or_reused_arrays(cuda_renderer):
    on_gpu = staged_scene()
    links = robots(on_gpu, 4)
    previous = None
    for step in range(3):
        arrays = pose_arrays(4, seed=10 + step)
        # Fresh device arrays each step: the group keeps its instancer.
        write_poses(on_gpu, links, arrays, cuda_renderer.cuda_upload)
        image = np.asarray(cuda_renderer.render(on_gpu)).astype(np.int32)
        on_cpu = staged_scene()
        write_poses(on_cpu, robots(on_cpu, 4), arrays)
        assert_same_image(image, cuda_renderer.render(on_cpu))
        if previous is not None:
            assert np.abs(image - previous).max() > 50, "the prims moved"
        previous = image
    assert cuda_renderer.take_validation_messages() == []


def test_two_groups_from_two_writes(cuda_renderer):
    on_cpu, on_gpu = staged_scene(), staged_scene()
    for scene, upload in ((on_cpu, None), (on_gpu, cuda_renderer.cuda_upload)):
        links = robots(scene, 6)
        first, second = pose_arrays(3, seed=20), pose_arrays(3, seed=21)
        write_poses(scene, links[:3], first, upload)
        write_poses(scene, links[3:], second, upload)
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))


def test_unsupported_gpu_poses_raise_naming_the_prim(cuda_renderer):
    def render_error(build):
        scene = staged_scene()
        build(scene)
        with pytest.raises(ValueError) as error:
            cuda_renderer.render(scene)
        return str(error.value)

    def euler_unbounded(scene):
        links = robots(scene, 2, ops=(tokens.TRANSLATE, tokens.ROTATE_XYZ))
        scene.write(links, tokens.TRANSLATE, cuda_renderer.cuda_upload(np.zeros((2, 3), np.float32)))
        scene.write(links, tokens.ROTATE_XYZ, cuda_renderer.cuda_upload(np.zeros((2, 3), np.float32)))

    # Composed on the GPU, its poses never reach the host: it needs its bounds.
    assert "ernie:xform:extent" in render_error(euler_unbounded)

    def pivot(scene):
        links = robots(scene, 2, ops=(tokens.TRANSLATE, "xformOp:translate:pivot"))
        scene.write(links, tokens.TRANSLATE, cuda_renderer.cuda_upload(np.zeros((2, 3), np.float32)))

    assert "suffixed and inverted ops" in render_error(pivot)

    def doubles(scene):
        links = robots(scene, 2, ops=(tokens.TRANSLATE,))
        scene.write(links, tokens.TRANSLATE, cuda_renderer.cuda_upload(np.zeros((2, 3))))

    assert "float32" in render_error(doubles)

    def one_by_one(scene):
        links = robots(scene, 1, ops=(tokens.TRANSLATE,))
        scene.write(links[0], tokens.TRANSLATE, cuda_renderer.cuda_upload(np.zeros(3, np.float32)))

    assert "written for many prims" in render_error(one_by_one)


def test_writing_cpu_values_again_moves_back_to_the_cpu(cuda_renderer):
    arrays = pose_arrays(2, seed=30)
    scene = staged_scene()
    links = robots(scene, 2)
    write_poses(scene, links, arrays, cuda_renderer.cuda_upload)
    cuda_renderer.render(scene)
    write_poses(scene, links, arrays)  # the same poses from host arrays
    expected = staged_scene()
    write_poses(expected, robots(expected, 2), arrays)
    assert_same_image(cuda_renderer.render(scene), cuda_renderer.render(expected), tolerance=0)


# --- Camera poses as positions and quaternions ------------------------------------------------


def camera_poses(count, seed=40):
    """Views looking at the origin from around it: as (views, 4, 4) matrices and as (views, 7)
    positions and x, y, z, w quaternions of the same poses."""
    rng = np.random.default_rng(seed)
    matrices, rows = [], []
    for _ in range(count):
        eye = rng.uniform(-1, 1, 3) * np.array([20.0, 20.0, 5.0]) + np.array([0.0, -25.0, 15.0])
        m = np.array(ernie.look_at_transform(tuple(eye), (0.0, 0.0, 0.0)))
        matrices.append(m)
        q = quat_from_matrix(m[:3, :3])
        rows.append([*m[:3, 3], *q])
    return np.array(matrices, np.float32), np.array(rows, np.float32)


def quat_from_matrix(r):
    """x, y, z, w of a rotation matrix (Shepperd's method)."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    return np.array([x, y, z, w])


def batch_images(renderer, poses):
    scene = staged_scene()
    write_poses(scene, robots(scene, 3, [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]), pose_arrays(3, 50))
    scene.write("/World/Camera", tokens.POSES, poses)
    scene.define("/Render/Batch", ernie.RenderProduct, camera="/World/Camera", resolution=(64, 48))
    outputs = renderer.step(scene, render_products=["/Render/Batch"])["/Render/Batch"]
    return np.array(outputs.frames[0].render_vars["/Render/Batch/LdrColor"].map()).astype(np.int32)


def test_camera_poses_as_quaternions_on_the_cpu(renderer):
    matrices, rows = camera_poses(4)
    by_matrix = batch_images(renderer, matrices)
    by_quaternion = batch_images(renderer, rows)
    assert by_matrix.shape[0] == 4
    for view in range(4):
        assert_same_image(by_quaternion[view], by_matrix[view])
    # float64 rows too.
    np.testing.assert_array_equal(batch_images(renderer, rows.astype(np.float64)), by_quaternion)


def test_camera_poses_as_quaternions_on_the_gpu(cuda_renderer):
    matrices, rows = camera_poses(4, seed=41)
    expected = batch_images(cuda_renderer, matrices)
    on_gpu = batch_images(cuda_renderer, cuda_renderer.cuda_upload(rows))
    for view in range(4):
        assert_same_image(on_gpu[view], expected[view])
    assert cuda_renderer.take_validation_messages() == []


def test_badly_shaped_camera_poses_raise(renderer):
    with pytest.raises(ValueError, match=r"\(views, 7\)"):
        batch_images(renderer, np.zeros((2, 6), np.float32))


def test_rewriting_part_of_a_group_from_new_arrays(cuda_renderer):
    """Steps that rewrite every prim of a group take the delegate's fast path; rewriting only some
    of them from new arrays splits the group, which a full sync must handle."""
    first, second = pose_arrays(4, seed=60), pose_arrays(2, seed=61)
    on_gpu = staged_scene()
    links = robots(on_gpu, 4)
    write_poses(on_gpu, links, first, cuda_renderer.cuda_upload)
    cuda_renderer.render(on_gpu)
    write_poses(on_gpu, links[:2], second, cuda_renderer.cuda_upload)
    on_cpu = staged_scene()
    cpu_links = robots(on_cpu, 4)
    write_poses(on_cpu, cpu_links[2:], tuple(a[2:] for a in first))
    write_poses(on_cpu, cpu_links[:2], second)
    assert_same_image(cuda_renderer.render(on_gpu), cuda_renderer.render(on_cpu))


# --- Quaternion order and PrimSets ------------------------------------------------------------


def test_w_first_orientations_render_like_w_last(cuda_renderer):
    positions, orientations, scales = pose_arrays(5, seed=70)
    expected, w_first = staged_scene(), staged_scene()
    write_poses(expected, robots(expected, 5), (positions, orientations, scales), cuda_renderer.cuda_upload)
    links = robots(w_first, 5)
    for link in links:
        w_first.set(link, tokens.QUATERNION_ORDER, tokens.WXYZ)
    wxyz = np.ascontiguousarray(np.roll(orientations, 1, axis=1))
    write_poses(w_first, ernie.PrimSet(links), (positions, wxyz, scales), cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(w_first), cuda_renderer.render(expected), tolerance=0)
    # The same on the CPU path.
    on_cpu = staged_scene()
    cpu_links = robots(on_cpu, 5)
    for link in cpu_links:
        on_cpu.set(link, tokens.QUATERNION_ORDER, tokens.WXYZ)
    write_poses(on_cpu, cpu_links, (positions, wxyz, scales))
    assert_same_image(cuda_renderer.render(on_cpu), cuda_renderer.render(expected))


def test_mixed_quaternion_orders_in_one_group_raise(cuda_renderer):
    scene = staged_scene()
    links = robots(scene, 2, ops=(tokens.ORIENT,))
    scene.set(links[0], tokens.QUATERNION_ORDER, tokens.WXYZ)
    scene.write(links, tokens.ORIENT, cuda_renderer.cuda_upload(quats(2)))
    with pytest.raises(ValueError, match="quaternionOrder"):
        cuda_renderer.render(scene)


def test_w_first_camera_poses(cuda_renderer):
    matrices, rows = camera_poses(4, seed=71)
    expected = batch_images(cuda_renderer, matrices)
    wxyz = rows.copy()
    wxyz[:, 3:] = np.roll(rows[:, 3:], 1, axis=1)

    def w_first_images(poses):
        scene = staged_scene()
        write_poses(scene, robots(scene, 3, [(-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]), pose_arrays(3, 50))
        scene.set("/World/Camera", tokens.QUATERNION_ORDER, tokens.WXYZ)
        scene.write("/World/Camera", tokens.POSES, poses)
        scene.define(
            "/Render/Batch", ernie.RenderProduct, camera="/World/Camera", resolution=(64, 48)
        )
        outputs = cuda_renderer.step(scene, render_products=["/Render/Batch"])["/Render/Batch"]
        return np.array(outputs.frames[0].render_vars["/Render/Batch/LdrColor"].map()).astype(np.int32)

    for images in (w_first_images(wxyz), w_first_images(cuda_renderer.cuda_upload(wxyz))):
        for view in range(4):
            assert_same_image(images[view], expected[view])


def test_prim_sets_write_like_paths(cuda_renderer):
    arrays = pose_arrays(4, seed=72)
    by_paths, by_set = staged_scene(), staged_scene()
    write_poses(by_paths, robots(by_paths, 4), arrays, cuda_renderer.cuda_upload)
    prims = ernie.PrimSet(robots(by_set, 4))
    assert len(prims) == 4 and "4 prims" in repr(prims)
    for step in range(2):  # the set's prims are looked up once and kept
        write_poses(by_set, prims, arrays, cuda_renderer.cuda_upload)
    assert_same_image(cuda_renderer.render(by_set), cuda_renderer.render(by_paths), tolerance=0)
