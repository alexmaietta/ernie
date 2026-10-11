# SPDX-License-Identifier: BSD-3-Clause
import json
import pathlib

import numpy as np
import pytest

import ernie
from ernie import tokens


def test_registry_describes_types_and_attributes():
    names = [t.name for t in ernie.prim_types()]
    assert (
        names.index("Light")
        < names.index("BoundableLightBase")
        < names.index("SphereLight")
    )
    sphere = ernie.find_prim_type("SphereLight")
    assert (
        sphere.base.name == "BoundableLightBase"
        and sphere.base.abstract
    )
    assert sphere.is_a("Light") and not sphere.is_a(
        "NonboundableLightBase"
    )
    assert ernie.find_prim_type("Nope") is None
    radius = sphere.find("radius")
    assert radius is sphere.find(tokens.RADIUS)
    assert (radius.name, radius.py_name, radius.default) == (
        "inputs:radius",
        "radius",
        0.5,
    )
    assert (
        radius.kind == ernie.AttributeKind.DOUBLE
        and radius.group == "Shape"
    )
    assert sphere.find("physical_illuminant").options == [
        "white",
        "blackbody",
        "illuminant D",
        "custom",
        "sun",
    ]
    assert [a.py_name for a in sphere.attributes] == [
        "radius",
        "treat_as_point",
        "shaping_cone_angle",
        "shaping_cone_softness",
    ]
    assert [a.py_name for a in sphere.all_attributes()][:4] == [
        "visibility",
        "world",
        "quaternion_order",
        "intensity",
    ]
    assert (
        sphere.is_a("Imageable")
        and ernie.find_prim_type("Xform").base.name == "Imageable"
    )


def test_typed_prims_read_defaults_and_convert_on_set():
    scene = ernie.Scene()
    lamp = scene.define(
        "/World/Lamp", ernie.SphereLight, radius=0.2, intensity=40
    )
    assert isinstance(lamp, ernie.SphereLight) and isinstance(
        lamp, ernie.Light
    )
    assert repr(lamp) == "SphereLight('/World/Lamp')"
    assert (
        lamp.path == ernie.Path("/World/Lamp")
        and lamp.type == "SphereLight"
    )
    assert lamp.radius == 0.2
    assert lamp.intensity == 40.0 and isinstance(
        lamp.intensity, float
    )
    assert scene.get("/World/Lamp", tokens.INTENSITY) == 40.0
    assert lamp.color_temperature == 6500.0 and lamp.color == (
        1.0,
        1.0,
        1.0,
    )
    assert lamp.spectrum == []
    lamp.color = (1.0, 0.5, 0.25)
    lamp.treat_as_point = True
    assert (
        lamp.color == (1.0, 0.5, 0.25) and lamp.treat_as_point is True
    )
    lamp.spectrum = [400.0, 1.0, 700.0, 1.0]
    assert lamp.spectrum == [400.0, 1.0, 700.0, 1.0]
    with pytest.raises(ValueError):
        lamp.radius = "big"
    # Class attributes describe the attribute.
    assert ernie.SphereLight.radius.doc == "Metres."
    assert [a.py_name for a in ernie.DiskLight.attributes()][
        -1
    ] == "radius"


def test_unknown_attributes_raise_with_suggestions():
    scene = ernie.Scene()
    lamp = ernie.SphereLight.define(scene, "/Lamp")
    with pytest.raises(AttributeError, match="did you mean 'radius'"):
        lamp.radus = 1.0
    with pytest.raises(AttributeError, match="did you mean 'radius'"):
        lamp.radus
    with pytest.raises(TypeError, match="did you mean 'radius'"):
        scene.define("/Other", ernie.SphereLight, radus=1.0)
    with pytest.raises(TypeError, match="not a registered prim type"):
        scene.define("/Rock", "Boulder", radius=1.0)


def test_define_takes_names_classes_and_relative_paths():
    scene = ernie.Scene()
    plain = scene.define("/World", tokens.XFORM)
    assert type(plain) is ernie.Prim
    sun = plain.define(
        "Sun", ernie.DistantLight, angle=1.0, rotate=(30, 0, 0)
    )
    assert sun.path == ernie.Path("/World/Sun") and sun.angle == 1.0
    assert sun.photometric_illuminance == 10000.0
    # On a typed prim, define makes a child; on its class, a prim of
    # that type.
    world = ernie.Xform.define(scene, "/Typed")
    lamp = world.define("Lamp", ernie.SphereLight, radius=0.25)
    assert (
        lamp.path == ernie.Path("/Typed/Lamp") and lamp.radius == 0.25
    )
    assert scene.get("/World/Sun", "xformOp:rotateXYZ") == (
        30.0,
        0.0,
        0.0,
    )
    # A registered type by name still returns the Prim, with keyword
    # attributes set.
    disk = scene.define("/World/Disk", tokens.DISK_LIGHT, radius=0.25)
    assert (
        type(disk) is ernie.Prim
        and ernie.DiskLight(disk).radius == 0.25
    )
    # Typed prims from existing prims check the type.
    assert (
        ernie.Light.at(scene, "/World/Sun").color_temperature
        == 6500.0
    )
    with pytest.raises(TypeError, match="not a SphereLight"):
        ernie.SphereLight.at(scene, "/World/Sun")
    with pytest.raises(TypeError, match="abstract"):
        ernie.Light.define(scene, "/Abstract")
    # The raw USD-name form still works on typed prims.
    sun.set(tokens.INTENSITY, 2.0)
    assert sun.intensity == 2.0


def test_existing_prims_are_typed_by_lookup():
    scene = ernie.Scene()
    world = scene.define("/World", ernie.Xform)
    lamp = world.define("Rig/Lamp", ernie.SphereLight, radius=0.25)
    # Class.at: in a scene by path, or below a prim or typed prim by
    # relative path.
    assert (
        type(ernie.SphereLight.at(scene, "/World/Rig/Lamp"))
        is ernie.SphereLight
    )
    assert (
        ernie.SphereLight.at(scene, ernie.Path("/World/Rig/Lamp"))
        == lamp
    )
    assert ernie.SphereLight.at(world, "Rig/Lamp") == lamp
    assert (
        ernie.SphereLight.at(
            scene.get_prim("/World/Rig"), "Lamp"
        ).radius
        == 0.25
    )
    assert (
        type(ernie.Light.at(scene, "/World/Rig/Lamp")) is ernie.Light
    )
    with pytest.raises(LookupError, match="No prim at '/World/Nope'"):
        ernie.SphereLight.at(scene, "/World/Nope")
    with pytest.raises(LookupError, match="'Nope' below '/World'"):
        ernie.SphereLight.at(world, "Nope")
    with pytest.raises(
        TypeError, match="is a SphereLight, not a DistantLight"
    ):
        ernie.DistantLight.at(scene, "/World/Rig/Lamp")
    with pytest.raises(TypeError, match="Scene or below a prim"):
        ernie.SphereLight.at("/World", "Rig/Lamp")
    # Prim.get_prim looks below a prim, as Scene.get_prim does in a
    # scene.
    assert world.get_prim("Rig/Lamp").path == lamp.path
    assert world.get_prim("Nope") is None
    # typed: the prim's own class, the most derived, or the Prim when
    # it has none.
    assert (
        type(ernie.typed(scene.get_prim("/World/Rig/Lamp")))
        is ernie.SphereLight
    )
    assert (
        type(ernie.typed(ernie.Light.at(scene, "/World/Rig/Lamp")))
        is ernie.SphereLight
    )
    untyped = ernie.typed(scene.get_prim("/World/Rig"))
    assert type(untyped) is ernie.Prim and untyped.path == ernie.Path(
        "/World/Rig"
    )
    with pytest.raises(TypeError, match="typed needs"):
        ernie.typed(scene.get_prim("/Missing"))


def test_tensor_attributes_write_and_read_arrays():
    scene = ernie.Scene()
    dome = scene.define("/Sky", ernie.DomeLight)
    assert dome.texture is None
    texture = np.ones((4, 8, 3), np.float32)
    dome.texture = texture
    assert dome.texture.shape == (4, 8, 3)


def test_class_attributes_name_their_attribute_for_bulk_writes():
    scene = ernie.Scene()
    paths = [f"/Swarm{i}" for i in range(3)]
    for path in paths:
        ernie.PointInstancer.define(scene, path)
    prims = ernie.PrimSet(paths)
    # The class attribute is its USD name, and describes it.
    positions = ernie.PointInstancer.positions
    assert (
        isinstance(positions, ernie.AttributeName)
        and positions == "positions"
    )
    assert (
        positions.kind == ernie.AttributeKind.TENSOR
        and positions.py_name == "positions"
    )
    rows = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
    scene.write(prims, positions, rows)
    assert np.array_equal(scene.read("/Swarm1", positions), rows[1])
    # write_rows: arrays by Python name, and the transform ops.
    orientations = np.tile(np.float32([1, 0, 0, 0]), (3, 2, 1))
    translate = np.float32([[0, 0, i] for i in range(3)])
    ernie.PointInstancer.write_rows(
        scene,
        prims,
        positions=rows + 1,
        orientations=orientations,
        translate=translate,
    )
    assert np.array_equal(
        ernie.PointInstancer.at(scene, "/Swarm2").positions,
        rows[2] + 1,
    )
    assert np.array_equal(
        scene.read("/Swarm2", "xformOp:translate"), translate[2]
    )
    with pytest.raises(TypeError, match="did you mean 'positions'"):
        ernie.PointInstancer.write_rows(scene, prims, postions=rows)
    with pytest.raises(
        TypeError, match="no array attribute 'prototypes'"
    ):
        ernie.PointInstancer.write_rows(scene, prims, prototypes=rows)
    # A PrimSet write is checked against each prim's type before any
    # row is written.
    mixed = ernie.Scene()
    ernie.PointInstancer.define(mixed, "/Swarm0")
    ernie.Mesh.define(mixed, "/Swarm1")
    with pytest.raises(ValueError, match="positions"):
        mixed.write(
            ernie.PrimSet(["/Swarm0", "/Swarm1"]), positions, rows[:2]
        )
    assert mixed.read("/Swarm0", positions) is None


def test_scene_functions_take_prims_for_paths():
    scene = ernie.Scene()
    lamp = scene.define("/Lamp", ernie.SphereLight, radius=0.3)
    assert scene.get(lamp, tokens.RADIUS) == 0.3
    assert scene.get(lamp.prim, tokens.RADIUS) == 0.3
    assert scene.type_of(lamp) == "SphereLight"
    assert ernie.Path(lamp) == ernie.Path("/Lamp")
    with pytest.raises(TypeError):
        scene.get(3, tokens.RADIUS)


def test_scenes_load_from_descriptions(tmp_path):
    description = {
        "/World": {
            "type": "Xform",
            "Rock": {
                "type": "Sphere",
                "radius": 0.5,
                "translate": [0, 0, 1],
            },
            "Sun": {
                "type": "DistantLight",
                "angle": 0.53,
                "inputs:intensity": 2.0,
            },
            "Ground": {
                "type": "Terrain",
                "heights": [[0, 1], [2, 3.5]],
                "cell_size": 2.0,
            },
            "Quad": {
                "type": "Mesh",
                "points": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                "face_vertex_counts": [3],
                "face_vertex_indices": [0, 1, 2],
            },
            "Group": {"Child": {"type": "Cube", "size": 2.0}},
        },
        "Render/Settings": {
            "type": "RenderSettings",
            "camera": "/World/Camera",
            "resolution": [640, 480],
        },
    }
    scene = ernie.load_scene(description)
    assert ernie.Sphere.at(scene, "/World/Rock").radius == 0.5
    assert scene.get("/World/Rock", tokens.TRANSLATE) == (
        0.0,
        0.0,
        1.0,
    )
    assert ernie.DistantLight.at(scene, "/World/Sun").intensity == 2.0
    assert (
        ernie.Terrain.at(scene, "/World/Ground").heights.dtype
        == np.float32
    )
    assert (
        ernie.Mesh.at(scene, "/World/Quad").face_vertex_indices.dtype
        == np.int32
    )
    # Float attributes stay float32 however their numbers are written.
    assert (
        ernie.Mesh.at(scene, "/World/Quad").points.dtype == np.float32
    )
    assert scene.type_of("/World/Group") == "" and ernie.Cube.at(
        scene, "/World/Group/Child"
    )
    assert ernie.RenderSettings.at(
        scene, "/Render/Settings"
    ).camera == ernie.Path("/World/Camera")

    # JSON text and files load the same way, into an existing scene;
    # no type leaves a prim's alone.
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"/World/Rock": {"radius": 0.75}}))
    assert ernie.load_scene(path, scene) is scene
    assert ernie.Sphere.at(scene, "/World/Rock").radius == 0.75
    ernie.load_scene('{"/World/Sun": {"angle": 1.0}}', scene)
    assert ernie.DistantLight.at(scene, "/World/Sun").angle == 1.0

    with pytest.raises(ValueError, match="did you mean 'radius'"):
        ernie.load_scene({"/X": {"type": "Sphere", "radus": 1.0}})
    with pytest.raises(TypeError, match="dict of prim paths"):
        ernie.load_scene([1, 2])


def test_scene_files_load_as_cpp_loads_them():
    # The fixture C++'s LoadSceneFile test loads
    # (scene_json_test.cpp), with the same results.
    fixture = (
        pathlib.Path(__file__).resolve().parents[1]
        / "fixtures"
        / "scene_basic.json"
    )
    scene = ernie.load_scene(fixture)
    assert ernie.GroundPlane.at(scene, "/World/Ground").checker
    assert (
        ernie.Camera.at(scene, "/World/Camera").focal_length == 35.0
    )
    expected = ernie.look_at_transform(
        (0, -10, 5), (0, 0, 0), (0, 0, 1)
    )
    assert np.allclose(
        scene.local_transform("/World/Camera"), expected
    )
    assert ernie.RenderSettings.at(
        scene, "/Render/Settings"
    ).resolution == (640.0, 480.0)

    # A quaternion in a description is [w, x, y, z], as Quat(w, x, y,
    # z) takes it.
    scene = ernie.load_scene(
        {"/A": {"orient": [0.5**0.5, 0, 0, 0.5**0.5]}}
    )
    scene.define("/B", "").set_xform(
        orient=ernie.Quat.from_axis_angle((0, 0, 1), 90.0)
    )
    assert np.allclose(
        scene.local_transform("/A"), scene.local_transform("/B")
    )


def test_values_are_read_by_their_declared_kind():
    # One value, or an array in its place where one is declared to
    # take it.
    assert (
        ernie.Mesh.display_color.takes_array
        and not ernie.Mesh.material_binding.takes_array
    )
    scene = ernie.Scene()
    mesh = ernie.Mesh.define(scene, "/Mesh", display_color=(1, 0, 0))
    assert mesh.display_color == (1.0, 0.0, 0.0)
    mesh.display_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert (
        mesh.display_color.dtype == np.float32
        and mesh.display_color.shape == (3, 3)
    )
    with pytest.raises(ValueError, match="not an array"):
        scene.write(
            "/Mesh",
            tokens.MATERIAL_BINDING,
            np.zeros((2, 3), np.float32),
        )
    # Transform ops by their kind, untyped prims' too: four numbers
    # are not a quaternion.
    scene.define("/Group", "")
    with pytest.raises(ValueError, match="quaternion"):
        scene.set("/Group", tokens.ORIENT, (0.0, 0.0, 0.0, 1.0))
    # A description dict may hold arrays, quaternions and paths; C++
    # reads it. Arrays go over beside the JSON, written where an array
    # goes without a copy, and read as numbers elsewhere.
    points = np.eye(3, dtype=np.float32)
    colors = np.eye(3, dtype=np.float64)
    loaded = ernie.load_scene(
        {
            "/Mesh": {
                "type": "Mesh",
                "points": points,
                "display_color": colors,
                "material_binding": ernie.Path("/Looks/A"),
                "translate": np.array([0, 0, 2.0]),
                "extent": [0, 0, 0, 1, 1, 1],
                "xformOp:orient:pivot": ernie.Quat(1, 0, 0, 0),
                "user:size": np.float32(2),
            },
        }
    )
    loaded_mesh = ernie.Mesh.at(loaded, "/Mesh")
    assert np.shares_memory(loaded_mesh.points, points)
    assert (
        loaded_mesh.display_color.dtype == np.float64
    )  # as written, not converted
    assert loaded.get("/Mesh", tokens.TRANSLATE) == (0.0, 0.0, 2.0)
    assert loaded.get("/Mesh", "user:size") == 2.0
    assert loaded_mesh.display_color.shape == (3, 3)
    assert loaded_mesh.material_binding == ernie.Path("/Looks/A")
    assert isinstance(
        loaded.get("/Mesh", "xformOp:orient:pivot"), ernie.Quat
    )
    with pytest.raises(ValueError, match="'/Mesh'.*xformOp:orient"):
        ernie.load_scene({"/Mesh": {"xformOp:orient": [1, 0, 0]}})


def test_saved_scenes_load_back_equal(tmp_path):
    scene = ernie.Scene()
    world = scene.define("/World", ernie.Xform)
    world.set_xform(
        translate=(1, 2, 3),
        orient=ernie.Quat.from_axis_angle((0, 0, 1), 30.0),
        scale=2.0,
    )
    world.define("Rock", ernie.Sphere, radius=0.25)
    world.define(
        "Sun",
        ernie.DistantLight,
        angle=0.53,
        intensity=2.0,
        color=(1.0, 0.9, 0.8),
    )
    world.define("Camera", ernie.Camera, focal_length=35.0)
    quad = ernie.Mesh.define(scene, "/World/Quad")
    quad.points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0.5]], dtype=np.float32
    )
    quad.face_vertex_counts = np.array([3], dtype=np.int32)
    quad.face_vertex_indices = np.array([0, 1, 2], dtype=np.int32)
    scene.write(
        "/World/Quad",
        tokens.DISPLAY_COLOR,
        np.eye(3, dtype=np.float32),
    )
    scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera="/World/Camera",
        resolution=(640, 480),
    )
    scene.define("/Custom", "")
    scene.set("/Custom", "user:tags", ["a", "b"])
    scene.set("/Custom", "user:count", 3)
    scene.define("/__session/Perspective", ernie.Camera)

    path = tmp_path / "saved.json"
    text = ernie.save_scene(scene, path)
    assert (
        path.read_text() == text
        and json.loads(text)["/World"]["type"] == "Xform"
    )
    assert "/__session" not in json.loads(text)

    loaded = ernie.load_scene(path)
    assert loaded.prims() == [
        p
        for p in scene.prims()
        if not str(p).startswith("/__session")
    ]
    for prim in loaded.prims():
        assert loaded.type_of(prim) == scene.type_of(prim)
        assert loaded.attributes(prim) == scene.attributes(prim), prim
        for name in scene.attributes(prim):
            array = scene.read(prim, name)
            if array is not None:
                assert np.array_equal(
                    np.asarray(loaded.read(prim, name)),
                    np.asarray(array),
                ), name
                assert (
                    np.asarray(loaded.read(prim, name)).dtype
                    == np.asarray(array).dtype
                ), name
            else:
                assert loaded.get(prim, name) == scene.get(
                    prim, name
                ), (prim, name)
    assert np.allclose(
        loaded.local_transform("/World"),
        scene.local_transform("/World"),
    )
    assert ernie.save_scene(loaded) == text


def test_stage_metadata_is_on_the_root():
    scene = ernie.Scene()
    stage = ernie.Stage(scene)
    assert stage == ernie.Stage(scene.get_prim("/"))
    assert (stage.up_axis, stage.meters_per_unit) == ("Z", 1.0)
    assert stage.time_codes_per_second == 24.0
    stage.up_axis = "Y"
    stage.time_codes_per_second = 60.0
    stage.time = 2.5
    assert stage.time_code == 150.0 and stage.time == 2.5
    assert scene.get("/", tokens.UP_AXIS) == "Y"
    with pytest.raises(AttributeError):
        stage.metres_per_unit = 1.0
    with pytest.raises(TypeError):
        ernie.Stage(scene.define("/World", "Xform"))
    loaded = ernie.load_scene(ernie.save_scene(scene))
    assert ernie.Stage(loaded).up_axis == "Y"
    assert ernie.Stage(loaded).time_code == 150.0
    loaded = ernie.load_scene({"/": {"meters_per_unit": 0.01}})
    assert ernie.Stage(loaded).meters_per_unit == 0.01
