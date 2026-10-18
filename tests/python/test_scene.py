# SPDX-License-Identifier: BSD-3-Clause
"""The scene from Python: prims, attributes as Python values, arrays shared in place, JSON scene
files and transforms."""

import json
import math
import os
import pathlib

import pytest

import ernie

np = pytest.importorskip("numpy")


def test_defines_finds_and_removes_prims():
    scene = ernie.Scene()
    ball = scene.define("/World/Ball", "Sphere")
    assert isinstance(ball, ernie.Prim)
    assert scene.find("/World/Ball") == ball
    assert scene.find("/World/Missing") is None
    assert "/World" in scene, "ancestors are defined too"
    assert scene.type_name(ball) == "Sphere"
    assert scene.type_name("/World") == ""
    assert scene.path(ball) == "/World/Ball"
    assert scene.children("/World") == ["/World/Ball"]
    assert scene.prims("Sphere") == ["/World/Ball"]
    with pytest.raises(ValueError):
        scene.define("/World/Rock", "NoSuchType")
    assert scene.remove("/World")
    assert not scene.remove("/World")
    assert scene.find("/World/Ball") is None
    with pytest.raises(ValueError, match="removed"):
        scene.get(ball, "radius")


def test_sets_and_gets_python_values():
    scene = ernie.Scene()
    scene.define("/World/Rover", "Xform")
    cam = scene.define("/World/Rover/Mast", "Camera")
    scene.set(cam, "focalLength", 0.035)
    scene.set(cam, "clippingRange", (0.05, 5000))
    assert scene.get(cam, "focalLength") == pytest.approx(0.035)
    assert scene.get(cam, "clippingRange") == pytest.approx([0.05, 5000])
    assert scene.get(cam, "horizontalAperture") is None, "not authored"

    cylinder = scene.define("/World/Rover/Wheel", "Cylinder")
    scene.set(cylinder, "axis", "Y")
    scene.set("/World/Rover/Wheel", "radius", 1)
    assert scene.get(cylinder, "axis") == "Y"
    assert scene.get(cylinder, "radius") == 1.0
    with pytest.raises(ValueError, match="'/World/Rover/Wheel' radius"):
        scene.set(cylinder, "radius", "wide")

    scene.set("/World/Rover", "ernie:count", 3)
    scene.set("/World/Rover", "ernie:flag", True)
    scene.set("/World/Rover", "ernie:tags", ["a", "b"])
    assert scene.get("/World/Rover", "ernie:count") == 3
    assert scene.get("/World/Rover", "ernie:flag") is True
    assert scene.get("/World/Rover", "ernie:tags") == ["a", "b"]

    rows = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]]
    scene.set(cam, "xformOp:transform", rows)
    scene.set(cam, "xformOpOrder", ["xformOp:transform"])
    assert scene.get(cam, "xformOp:transform") == rows
    assert "focalLength" in scene.attributes(cam)
    assert scene.clear(cam, "focalLength")
    assert scene.get(cam, "focalLength") is None


def test_relationships_take_paths():
    scene = ernie.Scene()
    scene.define("/Looks/Paint", "Material")
    rover = scene.define("/World/Rover", "Xform")
    scene.apply_api(rover, "MaterialBindingAPI")
    scene.set(rover, "material:binding", "/Looks/Paint")
    assert scene.get(rover, "material:binding") == "/Looks/Paint"


def test_writes_numpy_arrays_in_place():
    scene = ernie.Scene()
    mesh = scene.define("/World/Body", "Mesh")
    points = np.array([[-1, -0.5, 0.3], [1, -0.5, 0.3], [1, 0.5, 0.3]], dtype=np.float32)
    scene.write(mesh, "points", points)
    scene.set(mesh, "faceVertexCounts", np.array([3], dtype=np.int32))
    scene.set(mesh, "faceVertexIndices", [0, 1, 2])

    shared = np.from_dlpack(scene.read(mesh, "points"))
    assert np.shares_memory(shared, points), "no copy either way"
    points[0, 0] = -2
    assert shared[0, 0] == -2
    assert scene.read(mesh, "faceVertexIndices").shape == [3]
    assert scene.read(mesh, "normals") is None
    with pytest.raises(ValueError):
        scene.write(mesh, "points", np.zeros((3, 2), dtype=np.float32))


def test_transforms():
    scene = ernie.Scene()
    scene.define("/World", "Xform")
    rover = scene.define("/World/Rover", "Xform")
    scene.set_xform("/World", translate=(1, 2, 3))
    scene.set_xform(rover, rotate=(0, 0, 90), rotation_order="ZYX", scale=2)
    world = scene.world_transform(rover)
    assert len(world) == 4 and all(len(row) == 4 for row in world)
    assert [row[3] for row in world] == pytest.approx([1, 2, 3, 1]), "translation last"
    # X turns to Y, doubled.
    assert [world[0][0], world[1][0], world[2][0]] == pytest.approx([0, 2, 0], abs=1e-12)
    assert scene.local_transform("/World")[0][3] == pytest.approx(1)
    with pytest.raises(ValueError):
        scene.set_xform(rover, rotation_order="XXY")
    half = math.sqrt(0.5)
    scene.set_xform(rover, orient=(half, 0, 0, half))
    assert scene.world_transform(rover)[1][0] == pytest.approx(1)


def test_loads_and_saves_json(tmp_path):
    fixtures = pathlib.Path(os.environ["ERNIE_TEST_FIXTURES"])
    scene = ernie.Scene()
    scene.load(fixtures / "scenes" / "rover.json")
    assert scene.type_name("/World/Rover/Body") == "Mesh"
    text = scene.save_json()
    assert json.loads(text)["prims"]["/World/Rover/Wheel"]["axis"] == "Y"

    path = tmp_path / "rover.json"
    scene.save(path)
    again = ernie.Scene()
    again.load(path)
    assert again.save_json() == text

    points = np.zeros((2, 3), dtype=np.float32)
    shared = ernie.Scene()
    shared.load_json(
        json.dumps({"prims": {"/Cloud": {"typeName": "Points", "points": {"$array": 0}}}}),
        [points],
    )
    assert np.shares_memory(np.from_dlpack(shared.read("/Cloud", "points")), points)
    with pytest.raises(ValueError):
        shared.load_json("{not json")
