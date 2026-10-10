# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


def test_registry_describes_types_and_attributes():
    names = [t.name for t in ernie.prim_types()]
    assert names.index("Light") < names.index("BoundableLightBase") < names.index("SphereLight")
    sphere = ernie.find_prim_type("SphereLight")
    assert sphere.base.name == "BoundableLightBase" and sphere.base.abstract
    assert sphere.is_a("Light") and not sphere.is_a("NonboundableLightBase")
    assert ernie.find_prim_type("Nope") is None
    radius = sphere.find("radius")
    assert radius is sphere.find(tokens.RADIUS)
    assert (radius.name, radius.py_name, radius.default) == ("inputs:radius", "radius", 0.5)
    assert radius.kind == ernie.AttributeKind.DOUBLE and radius.group == "Shape"
    assert sphere.find("physical_illuminant").options == ["white", "blackbody", "illuminant D", "custom"]
    assert [a.py_name for a in sphere.attributes] == [
        "radius", "treat_as_point", "shaping_cone_angle", "shaping_cone_softness"]
    assert [a.py_name for a in sphere.all_attributes()][:4] == [
        "visibility", "world", "quaternion_order", "intensity"]
    assert sphere.is_a("Imageable") and ernie.find_prim_type("Xform").base.name == "Imageable"


def test_typed_prims_read_defaults_and_convert_on_set():
    scene = ernie.Scene()
    lamp = scene.define("/World/Lamp", ernie.SphereLight, radius=0.2, intensity=40)
    assert isinstance(lamp, ernie.SphereLight) and isinstance(lamp, ernie.Light)
    assert repr(lamp) == "SphereLight('/World/Lamp')"
    assert lamp.path == ernie.Path("/World/Lamp") and lamp.type == "SphereLight"
    assert lamp.radius == 0.2
    assert lamp.intensity == 40.0 and isinstance(lamp.intensity, float)
    assert scene.get("/World/Lamp", tokens.INTENSITY) == 40.0
    assert lamp.color_temperature == 6500.0 and lamp.color == (1.0, 1.0, 1.0)
    assert lamp.spectrum == []
    lamp.color = (1.0, 0.5, 0.25)
    lamp.treat_as_point = True
    assert lamp.color == (1.0, 0.5, 0.25) and lamp.treat_as_point is True
    lamp.spectrum = [400.0, 1.0, 700.0, 1.0]
    assert lamp.spectrum == [400.0, 1.0, 700.0, 1.0]
    with pytest.raises(ValueError):
        lamp.radius = "big"
    # Class attributes describe the attribute.
    assert ernie.SphereLight.radius.doc == "Metres."
    assert [a.py_name for a in ernie.DiskLight.attributes()][-1] == "radius"


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
    sun = plain.define("Sun", ernie.DistantLight, angle=1.0, rotate=(30, 0, 0))
    assert sun.path == ernie.Path("/World/Sun") and sun.angle == 1.0
    assert sun.photometric_illuminance == 10000.0
    # On a typed prim, define makes a child; on its class, a prim of that type.
    world = ernie.Xform.define(scene, "/Typed")
    lamp = world.define("Lamp", ernie.SphereLight, radius=0.25)
    assert lamp.path == ernie.Path("/Typed/Lamp") and lamp.radius == 0.25
    assert scene.get("/World/Sun", "xformOp:rotateXYZ") == (30.0, 0.0, 0.0)
    # A registered type by name still returns the Prim, with keyword attributes set.
    disk = scene.define("/World/Disk", tokens.DISK_LIGHT, radius=0.25)
    assert type(disk) is ernie.Prim and ernie.DiskLight(disk).radius == 0.25
    # Typed prims from existing prims check the type.
    assert ernie.Light.at(scene, "/World/Sun").color_temperature == 6500.0
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
    # Class.at: in a scene by path, or below a prim or typed prim by relative path.
    assert type(ernie.SphereLight.at(scene, "/World/Rig/Lamp")) is ernie.SphereLight
    assert ernie.SphereLight.at(scene, ernie.Path("/World/Rig/Lamp")) == lamp
    assert ernie.SphereLight.at(world, "Rig/Lamp") == lamp
    assert ernie.SphereLight.at(scene.get_prim("/World/Rig"), "Lamp").radius == 0.25
    assert type(ernie.Light.at(scene, "/World/Rig/Lamp")) is ernie.Light
    with pytest.raises(LookupError, match="No prim at '/World/Nope'"):
        ernie.SphereLight.at(scene, "/World/Nope")
    with pytest.raises(LookupError, match="'Nope' below '/World'"):
        ernie.SphereLight.at(world, "Nope")
    with pytest.raises(TypeError, match="is a SphereLight, not a DistantLight"):
        ernie.DistantLight.at(scene, "/World/Rig/Lamp")
    with pytest.raises(TypeError, match="Scene or below a prim"):
        ernie.SphereLight.at("/World", "Rig/Lamp")
    # Prim.get_prim looks below a prim, as Scene.get_prim does in a scene.
    assert world.get_prim("Rig/Lamp").path == lamp.path
    assert world.get_prim("Nope") is None
    # typed: the prim's own class, the most derived, or the Prim when it has none.
    assert type(ernie.typed(scene.get_prim("/World/Rig/Lamp"))) is ernie.SphereLight
    assert type(ernie.typed(ernie.Light.at(scene, "/World/Rig/Lamp"))) is ernie.SphereLight
    untyped = ernie.typed(scene.get_prim("/World/Rig"))
    assert type(untyped) is ernie.Prim and untyped.path == ernie.Path("/World/Rig")
    with pytest.raises(TypeError, match="typed needs"):
        ernie.typed(scene.get_prim("/Missing"))


def test_tensor_attributes_write_and_read_arrays():
    scene = ernie.Scene()
    dome = scene.define("/Sky", ernie.DomeLight)
    assert dome.texture is None
    texture = np.ones((4, 8, 3), np.float32)
    dome.texture = texture
    assert dome.texture.shape == (4, 8, 3)
