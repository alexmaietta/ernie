# SPDX-License-Identifier: BSD-3-Clause
"""USD transform ops (xformOp:translate, :orient, :rotateXYZ..., :scale, xformOpOrder): values,
composition, Quat, set_xform, set_local_transform, row writes, and rendering."""

import itertools
import math

import numpy as np
import pytest

import ernie
from ernie import tokens

ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
ROTATE_OPS = {
    "XYZ": tokens.ROTATE_XYZ,
    "XZY": tokens.ROTATE_XZY,
    "YXZ": tokens.ROTATE_YXZ,
    "YZX": tokens.ROTATE_YZX,
    "ZXY": tokens.ROTATE_ZXY,
    "ZYX": tokens.ROTATE_ZYX,
}


# Reference math in NumPy, written independently of Ernie.


def axis_rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    a, b = (axis + 1) % 3, (axis + 2) % 3
    r = np.eye(4)
    r[a, a], r[a, b], r[b, a], r[b, b] = c, -s, s, c
    return r


def euler(degrees, order):
    """Fixed-axis rotation applying order[0]'s axis first: R_last @ R_middle @ R_first."""
    m = np.eye(4)
    for letter in order:
        axis = "XYZ".index(letter)
        m = axis_rotation(axis, degrees[axis]) @ m
    return m


def translate(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scale(s):
    return np.diag([*np.broadcast_to(s, 3), 1.0])


def quat_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return m


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)  # w, x, y, z


def rock(scene=None):
    scene = scene or ernie.Scene()
    return scene, scene.define("/World/Rock", tokens.XFORM)


# --- Tokens and values ------------------------------------------------------------------------


def test_tokens_are_usd_names():
    assert tokens.XFORM_OP_ORDER == "xformOpOrder"
    assert tokens.RESET_XFORM_STACK == "!resetXformStack!"
    assert tokens.TRANSLATE == "xformOp:translate"
    assert tokens.SCALE == "xformOp:scale"
    assert tokens.ORIENT == "xformOp:orient"
    assert (tokens.ROTATE_X, tokens.ROTATE_Y, tokens.ROTATE_Z) == (
        "xformOp:rotateX",
        "xformOp:rotateY",
        "xformOp:rotateZ",
    )
    for order, op in ROTATE_OPS.items():
        assert op == "xformOp:rotate" + order


def test_quats_round_trip_and_name_their_layout():
    scene, prim = rock()
    q = ernie.Quat(0.5, 0.5, -0.5, 0.5)
    assert (q.w, q.x, q.y, q.z) == (0.5, 0.5, -0.5, 0.5)
    assert q.wxyz == (0.5, 0.5, -0.5, 0.5) and q.xyzw == (0.5, -0.5, 0.5, 0.5)
    assert ernie.Quat.from_xyzw(0.5, -0.5, 0.5, 0.5) == q
    prim.set(tokens.ORIENT, q)
    got = prim.get(tokens.ORIENT)
    assert isinstance(got, ernie.Quat) and got == q
    assert "w=0.5" in repr(q)
    q.w = 1.0
    assert q.w == 1.0


def test_token_lists_round_trip():
    scene, prim = rock()
    prim.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.SCALE])
    assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.TRANSLATE, tokens.SCALE]
    with pytest.raises(TypeError):
        prim.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, 3])


def test_quat_constructors_match_reference_rotations():
    q = ernie.Quat.from_axis_angle((0, 0, 2), 90.0)
    np.testing.assert_allclose(quat_matrix(*q.wxyz), axis_rotation(2, 90.0), atol=1e-15)
    rng = np.random.default_rng(3)
    for order in ORDERS:
        for _ in range(50):
            degrees = rng.uniform(-180, 180, 3)
            q = ernie.Quat.from_euler(degrees, order)
            np.testing.assert_allclose(quat_matrix(*q.wxyz), euler(degrees, order), atol=1e-12)


def test_quat_euler_round_trips_in_every_order():
    rng = np.random.default_rng(5)
    for order in ORDERS:
        middle = "XYZ".index(order[1])
        for _ in range(50):
            degrees = rng.uniform(-179, 179, 3)
            degrees[middle] = rng.uniform(-89, 89)
            np.testing.assert_allclose(
                ernie.Quat.from_euler(degrees, order).euler(order), degrees, atol=1e-9
            )
    with pytest.raises(ValueError):
        ernie.Quat.from_euler((0, 0, 0), "XYX")


# --- Composition ------------------------------------------------------------------------------


def test_without_ops_the_transform_op_is_the_local_transform():
    scene, prim = rock()
    np.testing.assert_array_equal(scene.local_transform("/World/Rock"), np.eye(4))
    prim.set(tokens.TRANSFORM, translate((1, 2, 3)))
    np.testing.assert_array_equal(scene.local_transform("/World/Rock"), translate((1, 2, 3)))


@pytest.mark.parametrize("order", ORDERS)
def test_every_euler_op_matches_the_reference(order):
    scene, prim = rock()
    degrees = (12.0, -34.0, 56.0)
    prim.set(ROTATE_OPS[order], degrees).set(tokens.XFORM_OP_ORDER, [ROTATE_OPS[order]])
    np.testing.assert_allclose(scene.local_transform("/World/Rock"), euler(degrees, order), atol=1e-12)


@pytest.mark.parametrize("axis", range(3))
def test_single_axis_rotations(axis):
    scene, prim = rock()
    op = (tokens.ROTATE_X, tokens.ROTATE_Y, tokens.ROTATE_Z)[axis]
    prim.set(op, 30.0).set(tokens.XFORM_OP_ORDER, [op])
    np.testing.assert_allclose(scene.local_transform("/World/Rock"), axis_rotation(axis, 30.0), atol=1e-15)


def test_ops_multiply_in_order_first_outermost():
    scene, prim = rock()
    q = (0.9, 0.1, -0.3, 0.2)
    prim.set(tokens.TRANSLATE, (10.0, -2.0, 0.5))
    prim.set(tokens.ORIENT, ernie.Quat(*q))
    prim.set(tokens.SCALE, (2.0, 3.0, 4.0))
    for order in itertools.permutations([tokens.TRANSLATE, tokens.ORIENT, tokens.SCALE]):
        prim.set(tokens.XFORM_OP_ORDER, list(order))
        parts = {
            tokens.TRANSLATE: translate((10.0, -2.0, 0.5)),
            tokens.ORIENT: quat_matrix(*q),
            tokens.SCALE: scale((2.0, 3.0, 4.0)),
        }
        expected = parts[order[0]] @ parts[order[1]] @ parts[order[2]]
        np.testing.assert_allclose(scene.local_transform("/World/Rock"), expected, atol=1e-12)


def test_pivots_and_inverted_ops():
    scene, prim = rock()
    prim.set("xformOp:translate:pivot", (5.0, 5.0, 0.0)).set(tokens.ROTATE_Z, 90.0)
    prim.set(
        tokens.XFORM_OP_ORDER,
        ["xformOp:translate:pivot", tokens.ROTATE_Z, "!invert!xformOp:translate:pivot"],
    )
    m = scene.local_transform("/World/Rock")
    np.testing.assert_allclose(m @ [5, 5, 0, 1], [5, 5, 0, 1], atol=1e-12)
    np.testing.assert_allclose(m @ [6, 5, 0, 1], [5, 6, 0, 1], atol=1e-12)


def test_empty_order_is_the_identity():
    scene, prim = rock()
    prim.set(tokens.TRANSFORM, translate((1, 1, 1))).set(tokens.XFORM_OP_ORDER, [])
    np.testing.assert_array_equal(scene.local_transform("/World/Rock"), np.eye(4))


def test_reset_xform_stack_ignores_ancestors():
    scene = ernie.Scene()
    scene.define("/World/Robot/Arm", tokens.XFORM)
    scene.get_prim("/World").set_xform(translate=(100, 0, 0))
    arm = scene.get_prim("/World/Robot/Arm")
    arm.set(tokens.TRANSLATE, (0, 0, 1)).set(
        tokens.XFORM_OP_ORDER, [tokens.RESET_XFORM_STACK, tokens.TRANSLATE]
    )
    np.testing.assert_array_equal(scene.world_transform("/World/Robot/Arm"), translate((0, 0, 1)))


def test_op_edits_mark_the_transform_dirty():
    for attribute in (tokens.XFORM_OP_ORDER, tokens.TRANSLATE, tokens.ORIENT, tokens.ROTATE_ZYX):
        assert ernie.dirty_of(attribute) == ernie.Dirty.TRANSFORM


def test_bad_ops_raise_naming_the_problem():
    scene, prim = rock()
    prim.set(tokens.ORIENT, (1.0, 0.0, 0.0, 0.0)).set(tokens.XFORM_OP_ORDER, [tokens.ORIENT])
    with pytest.raises(ValueError, match="quaternion"):
        scene.local_transform("/World/Rock")
    prim.set(tokens.XFORM_OP_ORDER, ["xformOp:shear"])
    with pytest.raises(ValueError, match="xformOp:shear"):
        scene.local_transform("/World/Rock")


# --- set_xform and set_local_transform --------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_set_xform_writes_translate_rotate_scale(order):
    scene, prim = rock()
    assert prim.set_xform(translate=(1, 2, 3), rotate=(10, 20, 30), rotation_order=order, scale=2) is prim
    assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.TRANSLATE, ROTATE_OPS[order], tokens.SCALE]
    expected = translate((1, 2, 3)) @ euler((10, 20, 30), order) @ scale(2.0)
    np.testing.assert_allclose(scene.local_transform("/World/Rock"), expected, atol=1e-12)


def test_set_xform_with_orient_and_partial_ops():
    scene, prim = rock()
    prim.set_xform(orient=ernie.Quat.from_xyzw(0, 0, 1, 1))
    assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.ORIENT]
    np.testing.assert_allclose(scene.local_transform("/World/Rock"), axis_rotation(2, 90), atol=1e-15)
    prim.set_xform(scale=(1, 2, 3))
    assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.SCALE]
    with pytest.raises(ValueError):
        prim.set_xform(rotate=(0, 0, 0), orient=ernie.Quat(1, 0, 0, 0))
    with pytest.raises(TypeError):
        prim.set_xform(orient=(1, 0, 0, 0))
    with pytest.raises(ValueError):
        prim.set_xform(rotate=(0, 0, 0), rotation_order="XXY")


def test_set_local_transform_keeps_the_prims_ops():
    rng = np.random.default_rng(7)
    for order in ORDERS:
        scene, prim = rock()
        prim.set_xform(translate=(0, 0, 0), rotate=(0, 0, 0), rotation_order=order, scale=1)
        target = translate(rng.uniform(-5, 5, 3)) @ quat_matrix(*random_quat(rng)) @ scale(
            rng.uniform(0.5, 2, 3)
        )
        scene.set_local_transform("/World/Rock", target)
        assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.TRANSLATE, ROTATE_OPS[order], tokens.SCALE]
        np.testing.assert_allclose(scene.local_transform("/World/Rock"), target, atol=1e-9)


def test_set_local_transform_falls_back_to_the_transform_op():
    scene, prim = rock()
    prim.set_xform(translate=(0, 0, 0))
    turned = axis_rotation(0, 30.0)
    scene.set_local_transform("/World/Rock", turned)
    assert prim.get(tokens.XFORM_OP_ORDER) == [tokens.TRANSFORM]
    np.testing.assert_allclose(scene.local_transform("/World/Rock"), turned, atol=1e-15)
    # Without ops it simply sets xformOp:transform.
    scene2, prim2 = rock()
    scene2.set_local_transform("/World/Rock", turned)
    assert prim2.get(tokens.XFORM_OP_ORDER) is None


# --- Row writes -------------------------------------------------------------------------------


def make_links(scene, n):
    paths = [f"/World_{i}/Link" for i in range(n)]
    for path in paths:
        scene.define(path, ernie.Xform).set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
    return paths


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_array_poses_many_prims(dtype):
    scene = ernie.Scene()
    rng = np.random.default_rng(11)
    paths = make_links(scene, 64)
    positions = rng.uniform(-100, 100, (64, 3)).astype(dtype)
    wxyz = np.array([random_quat(rng) for _ in paths])
    xyzw = np.roll(wxyz, -1, axis=1).astype(dtype)  # rows of x, y, z, w
    scene.write(paths, tokens.TRANSLATE, positions)
    scene.write(paths, tokens.ORIENT, xyzw)
    tolerance = 1e-5 if dtype == np.float32 else 1e-12
    for i, path in enumerate(paths):
        expected = translate(positions[i]) @ quat_matrix(*wxyz[i])
        np.testing.assert_allclose(
            scene.local_transform(path), expected, atol=tolerance * 100, rtol=tolerance
        )


def test_transform_rows_are_row_major_matrices():
    scene = ernie.Scene()
    paths = [f"/P{i}" for i in range(4)]
    for path in paths:
        scene.define(path, ernie.Xform)
    matrices = np.stack([translate((i, 2 * i, 3 * i)) for i in range(4)])
    scene.write(paths, tokens.TRANSFORM, matrices)
    for i, path in enumerate(paths):
        np.testing.assert_array_equal(scene.local_transform(path), matrices[i])


def test_rewriting_rows_moves_the_prims():
    scene = ernie.Scene()
    paths = make_links(scene, 3)
    cursor = ernie.ChangeCursor()
    scene.changes(cursor)
    scene.write(paths, tokens.TRANSLATE, np.ones((3, 3)))
    changed = dict(scene.changes(cursor))
    assert all(changed[ernie.Path(p)] & ernie.Dirty.TRANSFORM for p in paths)
    np.testing.assert_array_equal(scene.local_transform(paths[0])[:3, 3], [1, 1, 1])


# --- Rendering --------------------------------------------------------------------------------


def staged_scene():
    scene = ernie.Scene()
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=24.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, -20, 12), (0, 0, 0))
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera=camera,
        resolution=(64, 48),
        clear_color=(0.0, 0.0, 0.0),
    )
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, -0.5, 0.8))
    )
    return scene


POSES = [
    dict(translate=(-4.0, 0.0, 0.0), rotate=(20.0, 0.0, 45.0), rotation_order="ZYX", scale=(1.0, 2.0, 1.0)),
    dict(translate=(0.0, 2.0, 1.0), orient=ernie.Quat.from_axis_angle((1, 1, 0), 60.0), scale=1.5),
    dict(translate=(4.0, -1.0, 0.0), rotate=(0.0, 30.0, 0.0), rotation_order="XYZ"),
]


def reference_matrix(pose):
    m = translate(pose["translate"])
    if "rotate" in pose:
        m = m @ euler(pose["rotate"], pose["rotation_order"])
    if "orient" in pose:
        m = m @ quat_matrix(*pose["orient"].wxyz)
    if "scale" in pose:
        m = m @ scale(pose["scale"])
    return m


def test_ops_render_like_the_equivalent_matrices(renderer):
    by_ops, by_matrix = staged_scene(), staged_scene()
    for i, pose in enumerate(POSES):
        by_ops.define(f"/World/Box{i}", ernie.Cube, **pose)
        by_matrix.define(f"/World/Box{i}", ernie.Cube).set(tokens.TRANSFORM, reference_matrix(pose))
    ops_image = renderer.render(by_ops).astype(np.int32)
    matrix_image = renderer.render(by_matrix).astype(np.int32)
    assert (ops_image[..., :3] > 0).any(), "the boxes are in view"
    assert np.abs(ops_image - matrix_image).max() <= 1
    assert renderer.take_validation_messages() == []


def test_editing_ops_moves_what_renders(renderer):
    scene = staged_scene()
    box = scene.define("/World/Box", ernie.Cube, translate=(-4.0, 0.0, 0.0))
    left = renderer.render(scene).astype(np.int32)
    box.set(tokens.TRANSLATE, (4.0, 0.0, 0.0))
    right = renderer.render(scene).astype(np.int32)
    expected = staged_scene()
    expected.define("/World/Box", ernie.Cube).set(tokens.TRANSFORM, translate((4.0, 0.0, 0.0)))
    assert np.abs(right - left).max() > 50, "the box moved"
    assert np.abs(right - renderer.render(expected).astype(np.int32)).max() <= 1


def test_row_written_poses_render(renderer):
    scene, expected = staged_scene(), staged_scene()
    paths = [f"/World/Box{i}" for i in range(3)]
    for path in paths:
        scene.define(path, ernie.Cube).set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
    positions = np.array([[-4.0, 0.0, 0.0], [0.0, 0.0, 2.0], [4.0, 0.0, 0.0]])
    wxyz = [ernie.Quat.from_euler((0.0, 0.0, 30.0 * i)).wxyz for i in range(3)]
    scene.write(paths, tokens.TRANSLATE, positions)
    scene.write(paths, tokens.ORIENT, np.roll(np.array(wxyz), -1, axis=1))
    for i, path in enumerate(paths):
        expected.define(path, ernie.Cube).set(
            tokens.TRANSFORM, translate(positions[i]) @ quat_matrix(*wxyz[i])
        )
    assert np.abs(
        renderer.render(scene).astype(np.int32) - renderer.render(expected).astype(np.int32)
    ).max() <= 1
