# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

import ernie
from ernie import tokens


def test_sampled_attributes_follow_the_time_code():
    scene = ernie.Scene()
    arm = scene.define("/Arm", ernie.Xform)
    arm.set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        time=0.0,
    )
    moved = np.array(
        ernie.look_at_transform((0, 0, 0), (1, 0, 0), (0, 0, 1))
    ).reshape(4, 4)
    moved[3, :3] = (0.0, 4.0, 0.0)
    arm.set(tokens.TRANSFORM, moved, time=8.0)
    assert scene.sample_times("/Arm", tokens.TRANSFORM) == [0.0, 8.0]
    assert scene.animated_prims() == [ernie.Path("/Arm")]
    ernie.Stage.at(scene, "/").time_code = 2.0
    world = np.array(scene.world_transform("/Arm")).reshape(4, 4)
    assert abs(world[3, 1] - 1.0) < 1e-9

    cloth = scene.define("/Cloth", ernie.Mesh)
    cloth.write(tokens.POINTS, np.zeros((1, 3), np.float32), time=0.0)
    cloth.write(
        tokens.POINTS, np.full((1, 3), 8.0, np.float32), time=8.0
    )
    assert np.allclose(
        np.asarray(scene.read("/Cloth", tokens.POINTS)), 2.0
    )
    assert scene.get_at("/Arm", tokens.TRANSFORM, 4.0) is not None
    assert scene.clear_samples("/Cloth", tokens.POINTS)
    assert scene.sample_times("/Cloth", tokens.POINTS) == []
