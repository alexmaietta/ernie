# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from ernie_bench.datasets import build_scene, shapes
from ernie_bench.datasets.hide_and_seek import CLASSES
from ernie_bench.config import SceneSpec

SPEC = SceneSpec("hide_and_seek", "hide_and_seek", seed=3)


@pytest.mark.parametrize("mesh", [shapes.plane(), shapes.box(), shapes.ramp(), shapes.cylinder()], ids=["plane", "box", "ramp", "cylinder"])
def test_meshes_wind_counter_clockwise_with_unit_normals(mesh):
    a, b, c = (mesh.positions[mesh.indices[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    assert np.all(np.einsum("ij,ij->i", face, mesh.normals[mesh.indices[:, 0]]) > 0)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1)


def test_scene_is_deterministic_per_seed():
    a = build_scene(SPEC, 8, 2)
    b = build_scene(SPEC, 8, 2)
    c = build_scene(SceneSpec("hide_and_seek", "hide_and_seek", seed=4), 8, 2)
    assert np.array_equal(a.instances.transform, b.instances.transform)
    assert not np.array_equal(a.instances.transform, c.instances.transform)


def test_layout_counts_per_world():
    scene = build_scene(SPEC, 5, 3)
    per_world = 1 + 4 + 2 + 2 + 4 + 2 + 2 + 2
    assert len(scene.instances) == 5 * per_world
    assert np.array_equal(np.bincount(scene.instances.world), [per_world] * 5)
    assert len(scene.motion.instance) == 5 * (4 + 2 + 2 + 2)
    assert len(scene.cameras) == 15
    assert set(scene.instances.semantic) == set(range(len(CLASSES)))


def test_state_matches_initial_transforms_and_moves_over_time():
    scene = build_scene(SPEC, 4, 2)
    start = scene.state(0.0)
    assert np.allclose(start.transform, scene.instances.transform[start.instance])
    later = scene.state(2.0)
    assert not np.allclose(start.transform, later.transform)
    assert not np.allclose(start.camera, later.camera)


def test_moving_instances_stay_inside_the_arena():
    scene = build_scene(SPEC, 64, 1)
    for time in np.linspace(0, 60, 31):
        position = scene.state(time).transform[:, :2, 3]
        assert np.all(np.abs(position) <= 10)


def test_cameras_are_rigid_and_ride_their_agents():
    scene = build_scene(SPEC, 4, 6)
    state = scene.state(1.5)
    rotation = state.camera[:, :3, :3]
    assert np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5)
    assert np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
    assert np.allclose(rotation[:, :, 1], [0, 0, 1], atol=1e-5)

    agent = state.transform[np.searchsorted(state.instance, scene.cameras.parent)]
    forward = -rotation[:, :, 2]
    assert np.allclose(state.camera[:, :2, 3], agent[:, :2, 3] + 0.3 * forward[:, :2], atol=1e-5)
    assert np.allclose(state.camera[:, 2, 3], 0.8)
    radius = 0.5 * np.linalg.norm(agent[:, :3, 0], axis=1)
    assert (np.linalg.norm(state.camera[:, :2, 3] - agent[:, :2, 3], axis=1) > radius).all()
    heading = agent[:, :3, 0] / np.linalg.norm(agent[:, :3, 0], axis=1, keepdims=True)
    assert np.allclose(np.einsum("ij,ij->i", forward[0::6], heading[0::6]), 1, atol=1e-5)


def test_unknown_param_is_rejected():
    with pytest.raises(ValueError, match="params"):
        build_scene(SceneSpec("x", "hide_and_seek", params={"lava": True}), 1, 1)
