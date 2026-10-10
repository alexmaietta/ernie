# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import Device, tokens
from test_cuda import instancer_scene


def product_scene():
    """Instances seen from above by two products: the top camera and a side camera."""
    scene = instancer_scene(np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], np.float32))
    side = scene.define("/World/Side", ernie.Camera, focal_length=18.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((60, 0, 20), (0, 0, 0))
    )
    scene.define(
        "/Render/Camera", ernie.RenderProduct, camera="/World/Camera", resolution=(32, 32)
    ).set(tokens.ORDERED_VARS, [ernie.Path("/Render/Camera/LdrColor")])
    scene.define("/Render/Camera/LdrColor", ernie.RenderVar, source_name="LdrColor")
    scene.define("/Render/Side", ernie.RenderProduct, camera=side, resolution=(24, 16))
    return scene


def test_step_returns_outputs_by_product(renderer):
    scene = product_scene()
    products = renderer.step(scene, render_products={"/Render/Camera", "/Render/Side"})
    assert set(products) == {"/Render/Camera", "/Render/Side"}
    for _name, product in products.items():
        for frame in product.frames:
            for _path, var in frame.render_vars.items():
                assert var.name == "LdrColor"
    camera = products["/Render/Camera"].frames[0].render_vars["/Render/Camera/LdrColor"]
    pixels = camera.map(device=Device.CPU)
    assert pixels.shape == (32, 32, 4) and pixels.dtype == np.uint8
    side = products["/Render/Side"].frames[0].render_vars["/Render/Side/LdrColor"]
    assert side.shape == (16, 24, 4)
    assert side.map().shape == (16, 24, 4)
    assert renderer.take_validation_messages() == []


def test_step_async_waits_polls_and_fetches(renderer):
    scene = product_scene()
    expected = np.array(
        renderer.step(scene, render_products=["/Render/Camera"])["/Render/Camera"]
        .frames[0]
        .render_vars["/Render/Camera/LdrColor"]
        .map()
    )
    op = renderer.step_async(scene, render_products=["/Render/Camera"])
    assert isinstance(op, ernie.Operation)
    while op.wait(timeout_ns=0) is None:
        pass
    assert op.done
    products = op.wait().fetch()
    pixels = products["/Render/Camera"].frames[0].render_vars["/Render/Camera/LdrColor"].map()
    np.testing.assert_array_equal(pixels, expected)


def test_settings_products_and_bad_paths(renderer):
    scene = product_scene()
    scene.set("/Render/Settings", tokens.PRODUCTS, [ernie.Path("/Render/Side")])
    assert set(renderer.step(scene)) == {"/Render/Side"}
    with pytest.raises(ValueError, match="RenderProduct"):
        renderer.step(scene, render_products=["/World/Camera"])


def test_maps_to_cuda_and_cuda_arrays(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    scene = product_scene()
    op = renderer.step_async(scene, render_products=["/Render/Camera"])
    var = op.outputs()["/Render/Camera"].frames[0].render_vars["/Render/Camera/LdrColor"]
    expected = np.array(var.map(device=Device.CPU))

    linear = var.map(device=Device.CUDA, sync_stream=1)
    assert linear.__dlpack_device__() == (2, 0)
    np.testing.assert_array_equal(np.array(renderer.cuda_download(linear)), expected)

    mapping = var.map(device=Device.CUDA_ARRAY)
    assert isinstance(mapping, ernie.CudaArrayMapping)
    assert mapping.cuda_array != 0 and mapping.shape == (32, 32, 4)
    assert var.mapped
    var.unmap(stream=1)
    assert not var.mapped
    assert renderer.take_validation_messages() == []


def test_maps_every_render_var_with_its_shape_and_type(renderer):
    scene = product_scene()
    names = ["SpectralRadiance", "Electrons", "RawDN", "HdrColor", "DistanceToCamera", "PrimId"]
    for name in names:
        scene.define(f"/Render/Vars/{name}", ernie.RenderVar, source_name=name)
    scene.set(
        "/Render/Camera", tokens.ORDERED_VARS, [ernie.Path(f"/Render/Vars/{n}") for n in names]
    )
    frame = renderer.step(scene, render_products={"/Render/Camera"})["/Render/Camera"].frames[0]
    var = frame.render_vars
    expected = {
        "SpectralRadiance": ((16, 32, 32), "float32"),
        "Electrons": ((32, 32, 4), "float32"),
        "RawDN": ((32, 32, 4), "uint16"),
        "HdrColor": ((32, 32, 4), "float32"),
        "DistanceToCamera": ((32, 32), "float32"),
        "PrimId": ((32, 32), "uint32"),
    }
    for name, (shape, dtype) in expected.items():
        render_var = var[f"/Render/Vars/{name}"]
        assert render_var.shape == shape and render_var.dtype == dtype, name
        array = render_var.map()
        assert array.shape == shape and array.dtype == np.dtype(dtype), name

    ids = var["/Render/Vars/PrimId"].map()
    hit = ids[ids != 0]
    assert hit.size, "the instances are in view"
    paths = {str(renderer.prim_path(int(i))) for i in np.unique(hit)}
    assert paths == {"/World/Rocks"}, paths
    assert str(renderer.prim_path(0)) == ""
    distance = var["/Render/Vars/DistanceToCamera"].map()
    assert np.all(np.isinf(distance[ids == 0])) and np.all(np.isfinite(distance[ids != 0]))
    assert renderer.take_validation_messages() == []


def test_products_choose_their_sensor_filters(renderer):
    scene = product_scene()
    scene.define("/Render/Filters/Cie", ernie.SensorFilter, filter_type="cie")
    scene.set("/Render/Camera", tokens.FILTERS, [ernie.Path("/Render/Filters/Cie")])
    scene.define("/Render/Vars/Hdr", ernie.RenderVar, source_name="HdrColor")
    scene.set("/Render/Camera", tokens.ORDERED_VARS, [ernie.Path("/Render/Vars/Hdr")])
    products = renderer.step(scene, render_products={"/Render/Camera"})
    hdr = products["/Render/Camera"].frames[0].render_vars["/Render/Vars/Hdr"].map()
    assert hdr.dtype == np.float32 and hdr[..., :3].max() > 0.0

    scene.set("/Render/Camera", tokens.ORDERED_VARS, [ernie.Path("/Render/Camera/LdrColor")])
    with pytest.raises(ValueError, match="sensor filters do not write"):
        renderer.step(scene, render_products={"/Render/Camera"})


def test_a_camera_with_poses_renders_a_batch_of_views(renderer, terrain_scene):
    scene = terrain_scene
    poses = np.stack(
        [
            ernie.look_at_transform((150.0 * np.cos(a), 150.0 * np.sin(a), 200.0), (0, 0, 0))
            for a in np.linspace(0.0, np.pi, 5)
        ]
    )
    camera = ernie.Camera(scene.get_prim("/World/Camera"))
    camera.poses = poses
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(24, 16))
    batch = renderer.step(scene, render_products=["/Render/Batch"])["/Render/Batch"]
    var = batch.frames[0].render_vars["/Render/Batch/LdrColor"]
    assert var.shape == (5, 16, 24, 4)
    images = var.map()
    assert images.shape == (5, 16, 24, 4) and images.dtype == np.uint8
    assert not np.array_equal(images[0], images[4]), "every view has its own pose"
    assert renderer.take_validation_messages() == []


def test_cloned_worlds_render_one_per_view(renderer):
    scene = ernie.Scene()
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.0, 0.0, 1.0))
    )
    env = scene.define("/Envs/Env0", ernie.Xform, world=0)
    env.define("Box", ernie.Cube, size=4.0)
    for i in range(1, 3):
        scene.clone("/Envs/Env0", f"/Envs/Env{i}")
        ernie.Xform(scene.get_prim(f"/Envs/Env{i}")).world = i
        ernie.Cube(scene.get_prim(f"/Envs/Env{i}/Box")).size = 4.0 + 4.0 * i
    above = ernie.look_at_transform((0, 0, 60), (0, 0, 0), up=(0, 1, 0))
    camera = scene.define(
        "/World/Camera",
        ernie.Camera,
        sensor_noise=False,
        poses=np.stack([above] * 3),
        worlds=np.arange(3, dtype=np.int32),
    )
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(32, 32))
    batch = renderer.step(scene, render_products=["/Render/Batch"])["/Render/Batch"]
    images = batch.frames[0].render_vars["/Render/Batch/LdrColor"].map()
    covered = (images[..., :3].sum(axis=-1) > 0).sum(axis=(1, 2))
    assert covered[0] < covered[1] < covered[2], "each view sees only its own world's box"
    assert renderer.take_validation_messages() == []


def test_cloned_worlds_render_apart_in_one_batch(renderer, terrain_scene):
    scene = terrain_scene
    env = scene.define("/Envs/Env0", ernie.Xform, world=0)
    env.define("Box", ernie.Cube, size=40.0, display_color=(1.0, 0.0, 0.0))
    scene.clone("/Envs/Env0", "/Envs/Env1")
    ernie.Xform(scene.get_prim("/Envs/Env1")).world = 1
    ernie.Cube(scene.get_prim("/Envs/Env1/Box")).display_color = (0.0, 1.0, 0.0)
    camera = ernie.Camera(scene.get_prim("/World/Camera"))
    pose = camera.get(tokens.TRANSFORM)
    camera.set(poses=np.stack([pose, pose]), worlds=np.array([0, 1], dtype=np.int32))
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(48, 48))
    images = (
        renderer.step(scene, render_products=["/Render/Batch"])["/Render/Batch"]
        .frames[0]
        .render_vars["/Render/Batch/LdrColor"]
        .map()
    )
    centre = images[:, 24, 24, :3].astype(int)
    assert centre[0, 0] > centre[0, 1], "view 0 sees world 0's red box"
    assert centre[1, 1] > centre[1, 0], "view 1 sees world 1's green box"
    assert renderer.take_validation_messages() == []
