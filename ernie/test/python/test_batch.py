# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens
from ernie.batch import BatchCamera


def noiseless(scene):
    """The terrain scene without sensor noise or sun shadows, so a
    batch's views render as single views of the same poses do (a batch
    fits its sun shadows to the scene, a single view to its
    frustum)."""
    ernie.Camera.at(scene, "/World/Camera").sensor_noise = False
    ernie.DistantLight.at(scene, "/World/Sun").shadow_enable = False
    return scene


def poses_around(count):
    return np.stack(
        [
            np.array(
                ernie.look_at_transform(
                    (
                        60.0 * np.cos(0.7 * i),
                        60.0 * np.sin(0.7 * i),
                        60.0,
                    ),
                    (0, 0, 0),
                    up=(0, 0, 1),
                ),
                np.float32,
            )
            for i in range(count)
        ]
    )


def test_views_render_as_single_cameras_at_their_poses(
    renderer, terrain_scene
):
    scene = noiseless(terrain_scene)
    poses = poses_around(3)
    camera = BatchCamera(
        renderer,
        scene,
        "/World/Camera",
        3,
        resolution=(48, 48),
        render_vars=("LdrColor", "DistanceToCamera"),
    )
    observations = camera.step(poses, device="cpu")
    colour = np.array(observations["LdrColor"])
    distance = np.array(observations["DistanceToCamera"])
    assert colour.shape == (3, 48, 48, 4) and colour.dtype == np.uint8
    assert (
        distance.shape == (3, 48, 48) and distance.dtype == np.float32
    )

    # The same camera without poses, rendered alone at each view's
    # pose.
    single = scene.define(
        "/World/Single",
        ernie.Camera,
        focal_length=18.0,
        sensor_noise=False,
    )
    scene.define(
        "/Render/Single",
        ernie.RenderSettings,
        camera=single,
        resolution=(48, 48),
        clear_color=(0.0, 0.0, 1.0),
    )
    for view in range(3):
        single.set(tokens.TRANSFORM, poses[view].astype(np.float64))
        expected = np.array(
            renderer.render(scene, ernie.Path("/Render/Single"))
        )
        np.testing.assert_array_equal(
            colour[view], expected, err_msg=f"view {view}"
        )
    assert renderer.take_validation_messages() == []


@pytest.mark.parametrize("on_gpu", [False, True], ids=["cpu", "cuda"])
def test_each_view_sees_its_world(renderer, terrain_scene, on_gpu):
    if on_gpu and not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    worlds = renderer.cuda_upload if on_gpu else (lambda array: array)
    scene = noiseless(terrain_scene)
    env = scene.define("/Envs/Env0", ernie.Xform, world=0)
    env.define(
        "Box", ernie.Cube, size=40.0, display_color=(1.0, 0.0, 0.0)
    )
    scene.clone("/Envs/Env0", "/Envs/Env1")
    ernie.Xform.at(scene, "/Envs/Env1").world = 1
    ernie.Cube.at(scene, "/Envs/Env1/Box").display_color = (
        0.0,
        1.0,
        0.0,
    )
    pose = np.array(
        scene.get("/World/Camera", tokens.TRANSFORM), np.float32
    )
    camera = BatchCamera(
        renderer, scene, "/World/Camera", 2, resolution=(48, 48)
    )
    for order in (
        [0, 1],
        [1, 0],
    ):  # written again, the views see the new worlds
        colour = np.array(
            camera.step(
                np.stack([pose, pose]),
                worlds(np.array(order, np.int32)),
                device="cpu",
            )["LdrColor"]
        )
        centre = colour[:, 24, 24, :3].astype(int)
        red, green = (0, 1) if order == [0, 1] else (1, 0)
        assert centre[red, 0] > centre[red, 1], (
            "a view of world 0 sees its red box"
        )
        assert centre[green, 1] > centre[green, 0], (
            "a view of world 1 sees its green box"
        )


def test_rejects_arrays_of_another_view_count(
    renderer, terrain_scene
):
    camera = BatchCamera(renderer, terrain_scene, "/World/Camera", 4)
    with pytest.raises(ValueError):
        camera.set_poses(poses_around(3))
    with pytest.raises(ValueError):
        camera.set_worlds(np.zeros(5, np.int32))
    with pytest.raises(ValueError):
        BatchCamera(renderer, terrain_scene, "/World/Camera", 0)
    with pytest.raises(ValueError):
        camera.step(device="tpu")


def test_cuda_observations_match_cpu_ones(renderer, terrain_scene):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    scene = noiseless(terrain_scene)
    poses = poses_around(4)
    camera = BatchCamera(
        renderer,
        scene,
        "/World/Camera",
        4,
        resolution=(48, 48),
        render_vars=("LdrColor", "DistanceToCamera"),
    )
    expected = {
        name: np.array(array)
        for name, array in camera.step(poses, device="cpu").items()
    }
    observations = camera.step(renderer.cuda_upload(poses))
    for name, array in observations.items():
        assert array.__dlpack_device__()[0] == 2, name  # kDLCUDA
        on_host = np.array(renderer.cuda_download(array))
        # Poses on the GPU compute the views' matrices in float32.
        got, want = (
            on_host.astype(np.float64),
            expected[name].astype(np.float64),
        )
        both_infinite = (
            np.isinf(got)
            & np.isinf(want)
            & (np.sign(got) == np.sign(want))
        )
        gap = np.abs(
            np.where(both_infinite, 0.0, got)
            - np.where(both_infinite, 0.0, want)
        )
        differing = np.mean(~(gap <= 8))
        assert differing < 0.005, name
    assert renderer.take_validation_messages() == []


LINEAR_VARS = (
    "SpectralRadiance",
    "Electrons",
    "RawDN",
    "HdrColor",
    "LdrColor",
    "DistanceToImagePlane",
    "DistanceToCamera",
    "Normal",
    "PrimId",
    "InstanceId",
)


def test_cuda_observations_written_in_place_match_cpu_ones(
    renderer, terrain_scene
):
    # The first map to CUDA copies; the frames after it hold the vars
    # as linear buffers the sensor writes beside the images, which
    # later maps view in place. Every way gives the same observations.
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    scene = noiseless(terrain_scene)
    poses = poses_around(3)
    camera = BatchCamera(
        renderer,
        scene,
        "/World/Camera",
        3,
        resolution=(32, 24),
        render_vars=LINEAR_VARS,
    )
    expected = {
        name: np.array(array)
        for name, array in camera.step(poses, device="cpu").items()
    }
    for step in range(4):
        observations = camera.step(poses)
        for name in LINEAR_VARS:
            got = np.array(renderer.cuda_download(observations[name]))
            assert got.shape == expected[name].shape, (name, step)
            np.testing.assert_array_equal(
                got, expected[name], err_msg=f"{name}, step {step}"
            )
        del observations
    assert renderer.take_validation_messages() == []


def test_minimal_batch_with_poses_mapped_in_place(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    from test_minimal import boxes_scene

    scene = boxes_scene()
    camera = BatchCamera(
        renderer,
        scene,
        "/World/Camera",
        3,
        resolution=(48, 32),
        render_vars=("LdrColor", "PrimId"),
        minimal="texturedDiffuse",
    )
    eyes = [
        (-20.0, -16.0, 12.0),
        (0.0, -24.0, 14.0),
        (20.0, -16.0, 12.0),
    ]
    poses = np.stack(
        [
            np.array(
                ernie.look_at_transform(e, (0, 0, 1), up=(0, 0, 1)),
                np.float32,
            )
            for e in eyes
        ]
    )
    first = camera.step(renderer.cuda_upload(poses), device="cpu")
    assert first["LdrColor"].shape == (3, 32, 48, 4)
    assert (first["PrimId"] != 0).mean() > 0.2
    # The next poses written in place into the renderer's own pose
    # buffer.
    moved = poses[::-1].copy()
    renderer.cuda_copy(
        renderer.cuda_upload(moved), camera.map_poses()
    )
    mapped = camera.step(device="cpu")
    expected = camera.step(renderer.cuda_upload(moved), device="cpu")
    for name in ("LdrColor", "PrimId"):
        np.testing.assert_array_equal(
            mapped[name], expected[name], err_msg=name
        )
    np.testing.assert_array_equal(
        mapped["PrimId"][0], first["PrimId"][2]
    )
