# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from ernie_bench import reference
from ernie_bench.datasets import build_scene, shapes
from ernie_bench.scene import Cameras, DistantLight, Instances, LissajousMotion, Material, Scene, compose, decompose, quaternions
from ernie_bench.config import SceneSpec


def look(position, target, up=(0, 0, 1)):
    forward = np.subtract(target, position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, np.cross(right, forward), -forward, position
    return matrix


def floor_scene(camera):
    instances = Instances(
        world=np.array([0, 0]),
        mesh=np.array([0, 1]),
        material=np.array([0, 1]),
        semantic=np.array([0, 1]),
        transform=np.stack([compose(np.zeros((1, 3)), np.zeros(1), np.array([[100.0, 100.0, 1.0]]))[0], compose(np.array([[0.0, 3.0, 0.0]]), np.zeros(1), np.ones((1, 3)))[0]]),
    )
    motion = LissajousMotion(np.array([1]), np.array([[0.0, 3.0]]), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1), np.ones((1, 3)))
    cameras = Cameras(np.array([0]), np.array([-1]), camera[None].astype(np.float32), np.array([90.0]), np.array([0.01]), np.array([100.0]))
    materials = [Material((0.5, 0.25, 0.125)), Material((0.8, 0.8, 0.8))]
    sun = DistantLight((0.0, -0.6, -0.8), np.pi)
    return Scene("floor", 1, [shapes.plane(), shapes.box()], materials, ("floor", "box"), instances, cameras, [sun], motion, ambient=0.25)


def test_depth_to_a_floor_matches_geometry():
    scene = floor_scene(look((0, 0, 2), (0, 0, 0), up=(0, 1, 0)))
    depth, classes, _ = reference.render(scene, scene.state(0.0), 33, 33, [0])
    assert depth[0, 16, 16] == pytest.approx(2.0, rel=1e-6)
    corner_angle = np.arctan(np.hypot(16 / 16.5, 16 / 16.5))
    assert depth[0, 0, 0] == pytest.approx(2.0 / np.cos(corner_angle), rel=1e-6)
    assert classes[0, 16, 16] == 1


def test_image_rows_run_top_down_and_columns_left_to_right():
    scene = floor_scene(look((0, -3, 0.5), (0, 3, 0.5)))
    depth, classes, _ = reference.render(scene, scene.state(0.0), 41, 41, [0])
    assert classes[0, 20, 20] == 2
    assert depth[0, 20, 20] == pytest.approx(5.5, rel=1e-6)
    assert classes[0, -1, 20] == 1 and classes[0, 0, 20] == 0
    assert np.isinf(depth[0, 0, 20])


def test_near_and_far_planes_clip():
    scene = floor_scene(look((0, 0, 2), (0, 0, 0), up=(0, 1, 0)))
    scene.cameras.near[:] = 2.5
    depth, classes, _ = reference.render(scene, scene.state(0.0), 9, 9, [0])
    assert np.isinf(depth).all() and (classes == 0).all()


def test_quaternions_round_trip_through_matrices():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(500, 4))
    raw = np.concatenate([raw, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]]])
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    x, y, z, w = raw.T
    matrices = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )
    recovered = quaternions(matrices)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", recovered, raw)), 1, atol=1e-9)


def test_decompose_recovers_translation_rotation_and_scale():
    yaw = np.array([0.0, np.pi / 2, np.pi, -2.0])
    matrices = compose(np.arange(12, dtype=float).reshape(4, 3), yaw, np.array([[1, 2, 3], [4, 5, 6], [0.5, 0.5, 2], [1, 1, 1]], float))
    translation, rotation, scale = decompose(matrices)
    assert np.allclose(translation, np.arange(12).reshape(4, 3))
    assert np.allclose(scale, [[1, 2, 3], [4, 5, 6], [0.5, 0.5, 2], [1, 1, 1]])
    assert np.allclose(np.abs(rotation[:, 2]), np.abs(np.sin(yaw / 2)), atol=1e-6)
    assert np.allclose(rotation[:, :2], 0, atol=1e-6)


def test_state_carries_decomposed_poses():
    scene = build_scene(SceneSpec("h", "hide_and_seek"), 3, 2)
    state = scene.state(0.7)
    assert np.allclose(state.translation, state.transform[:, :3, 3])
    assert np.allclose(state.camera_translation, state.camera[:, :3, 3])
    assert np.allclose(np.linalg.norm(state.rotation, axis=1), 1)


def test_agreement_scores_identical_and_shifted_frames():
    depth = np.array([[[1.0, 2.0], [np.inf, 4.0]]])
    classes = np.array([[[1, 2], [0, 3]]])
    perfect = reference.agreement(depth, classes, {"depth": np.where(np.isinf(depth), 0, depth)[..., None], "segmentation": classes[..., None]})
    assert perfect == {"hit_agreement": 1.0, "depth_agreement": 1.0, "depth_median_error": 0.0, "segmentation_agreement": 1.0}
    off = reference.agreement(depth, classes, {"depth": (depth * 1.1)[..., None], "segmentation": np.zeros_like(classes)[..., None]})
    assert off["depth_agreement"] == 0.0 and off["segmentation_agreement"] == 0.25


def test_unlit_color_is_albedo():
    scene = floor_scene(look((0, 0, 2), (0, 0, 0), up=(0, 1, 0)))
    _, classes, rgb = reference.render(scene, scene.state(0.0), 33, 33, [0], "unlit")
    assert np.allclose(rgb[0][classes[0] == 1], [0.5, 0.25, 0.125])
    assert np.allclose(rgb[0][classes[0] == 2], [0.8, 0.8, 0.8])


def test_direct_light_is_albedo_times_ambient_plus_cosine():
    scene = floor_scene(look((0, -1.5, 2), (0, -1.5, 0), up=(0, 1, 0)))
    _, classes, rgb = reference.render(scene, scene.state(0.0), 33, 33, [0], "direct")
    assert classes[0, 16, 16] == 1
    assert np.allclose(rgb[0, 16, 16], np.array([0.5, 0.25, 0.125]) * (0.25 + 0.8))


def test_shadowed_floor_keeps_only_ambient():
    scene = floor_scene(look((0, 2.1, 2.5), (0, 2.1, 0), up=(0, 1, 0)))
    _, classes, rgb = reference.render(scene, scene.state(0.0), 65, 65, [0], "direct")
    floor = rgb[0][classes[0] == 1]
    shadowed = np.isclose(floor, np.array([0.5, 0.25, 0.125]) * 0.25).all(axis=1)
    sunlit = np.isclose(floor, np.array([0.5, 0.25, 0.125]) * 1.05).all(axis=1)
    assert shadowed.sum() > 50 and sunlit.sum() > 50 and (shadowed | sunlit).all()
