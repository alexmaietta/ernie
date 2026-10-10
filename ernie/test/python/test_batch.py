import numpy as np
import pytest

import ernie
from ernie import tokens
from ernie.batch import BatchCamera


def noiseless(scene):
    """The terrain scene without sensor noise or sun shadows, so a batch's views render as single
    views of the same poses do (a batch fits its sun shadows to the scene, a single view to its
    frustum)."""
    scene.set("/World/Camera", tokens.SENSOR_NOISE, False)
    scene.set("/World/Sun", tokens.SHADOW_ENABLE, False)
    return scene


def poses_around(count):
    return np.stack(
        [
            np.array(
                ernie.look_at_transform(
                    (60.0 * np.cos(0.7 * i), 60.0 * np.sin(0.7 * i), 60.0), (0, 0, 0), up=(0, 0, 1)
                ),
                np.float32,
            )
            for i in range(count)
        ]
    )


def test_views_render_as_single_cameras_at_their_poses(renderer, terrain_scene):
    scene = noiseless(terrain_scene)
    poses = poses_around(3)
    camera = BatchCamera(
        renderer, scene, "/World/Camera", 3, resolution=(48, 48),
        render_vars=("LdrColor", "DistanceToCamera"),
    )
    observations = camera.step(poses, device="cpu")
    colour = np.array(observations["LdrColor"])
    distance = np.array(observations["DistanceToCamera"])
    assert colour.shape == (3, 48, 48, 4) and colour.dtype == np.uint8
    assert distance.shape == (3, 48, 48) and distance.dtype == np.float32

    # The same camera without poses, rendered alone at each view's pose.
    scene.define("/World/Single", tokens.CAMERA).set(tokens.FOCAL_LENGTH, 18.0).set(
        tokens.SENSOR_NOISE, False
    )
    scene.define("/Render/Single", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Single")
    ).set(tokens.RESOLUTION, (48, 48)).set(tokens.CLEAR_COLOR, (0.0, 0.0, 1.0))
    for view in range(3):
        scene.set("/World/Single", tokens.TRANSFORM, poses[view].astype(np.float64))
        expected = np.array(renderer.render(scene, ernie.Path("/Render/Single")))
        np.testing.assert_array_equal(colour[view], expected, err_msg=f"view {view}")
    assert renderer.take_validation_messages() == []


@pytest.mark.parametrize("on_gpu", [False, True], ids=["cpu", "cuda"])
def test_each_view_sees_its_world(renderer, terrain_scene, on_gpu):
    if on_gpu and not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    worlds = renderer.cuda_upload if on_gpu else (lambda array: array)
    scene = noiseless(terrain_scene)
    scene.define("/Envs/Env0", tokens.XFORM).set(tokens.WORLD, 0)
    scene.define("/Envs/Env0/Box", tokens.CUBE).set(tokens.SIZE, 40.0).set(
        tokens.DISPLAY_COLOR, (1.0, 0.0, 0.0)
    )
    scene.clone("/Envs/Env0", "/Envs/Env1")
    scene.set("/Envs/Env1", tokens.WORLD, 1)
    scene.set("/Envs/Env1/Box", tokens.DISPLAY_COLOR, (0.0, 1.0, 0.0))
    pose = np.array(scene.get("/World/Camera", tokens.TRANSFORM), np.float32)
    camera = BatchCamera(renderer, scene, "/World/Camera", 2, resolution=(48, 48))
    for order in ([0, 1], [1, 0]):  # written again, the views see the new worlds
        colour = np.array(
            camera.step(
                np.stack([pose, pose]), worlds(np.array(order, np.int32)), device="cpu"
            )["LdrColor"]
        )
        centre = colour[:, 24, 24, :3].astype(int)
        red, green = (0, 1) if order == [0, 1] else (1, 0)
        assert centre[red, 0] > centre[red, 1], "a view of world 0 sees its red box"
        assert centre[green, 1] > centre[green, 0], "a view of world 1 sees its green box"


def test_rejects_arrays_of_another_view_count(renderer, terrain_scene):
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
        renderer, scene, "/World/Camera", 4, resolution=(48, 48),
        render_vars=("LdrColor", "DistanceToCamera"),
    )
    expected = {name: np.array(array) for name, array in camera.step(poses, device="cpu").items()}
    observations = camera.step(renderer.cuda_upload(poses))
    for name, array in observations.items():
        assert array.__dlpack_device__()[0] == 2, name  # kDLCUDA
        on_host = np.array(renderer.cuda_download(array))
        # Poses on the GPU compute the views' matrices in float32.
        got, want = on_host.astype(np.float64), expected[name].astype(np.float64)
        both_infinite = np.isinf(got) & np.isinf(want) & (np.sign(got) == np.sign(want))
        gap = np.abs(np.where(both_infinite, 0.0, got) - np.where(both_infinite, 0.0, want))
        differing = np.mean(~(gap <= 8))
        assert differing < 0.005, name
    assert renderer.take_validation_messages() == []
