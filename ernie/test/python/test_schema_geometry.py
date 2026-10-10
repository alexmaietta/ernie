# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


def test_geometry_classes_follow_usd_geom():
    assert issubclass(ernie.Cylinder, ernie.AxisShape) and issubclass(ernie.AxisShape, ernie.Gprim)
    assert issubclass(ernie.Mesh, ernie.PointBased) and issubclass(ernie.Points, ernie.PointBased)
    assert not issubclass(ernie.PointInstancer, ernie.Gprim)
    assert ernie.GroundPlane.prim_type.base.name == "Imageable"
    assert ernie.Points.prim_type.name == tokens.GEOM_POINTS
    assert ernie.Capsule.radius.default == 0.5 and ernie.Cylinder.radius.default == 1.0
    assert ernie.Plane.axis.options == ["X", "Y", "Z"]
    assert ernie.Mesh.points.kind == ernie.AttributeKind.TENSOR
    with pytest.raises(TypeError, match="abstract"):
        ernie.AxisShape.define(ernie.Scene(), "/Shape")


def test_define_with_keyword_attributes():
    scene = ernie.Scene()
    ball = scene.define("/World/Ball", ernie.Sphere, radius=0.25, display_color=(1.0, 0.0, 0.0))
    assert ball.radius == 0.25 and ball.display_color == (1.0, 0.0, 0.0)
    assert scene.get("/World/Ball", tokens.GEOM_RADIUS) == 0.25
    capsule = scene.define("/World/Capsule", ernie.Capsule, axis="X", translate=(0, 0, 1))
    assert (capsule.radius, capsule.height, capsule.axis) == (0.5, 1.0, "X")
    assert ernie.Cube.define(scene, "/World/Box").size == 2.0

    ground = scene.define("/World/Ground", ernie.GroundPlane, height=-1.0, checker=True)
    ground.checker_color1 = (0.1, 0.2, 0.3)
    ground.material_binding = scene.define("/Looks/Grass", tokens.MATERIAL)
    assert ground.height == -1.0 and ground.checker and ground.checker_size == 1.0
    assert scene.get("/World/Ground", tokens.CHECKER_COLOR1) == pytest.approx((0.1, 0.2, 0.3))
    assert ground.material_binding == ernie.Path("/Looks/Grass")

    cloud = scene.define("/World/Cloud", ernie.Points, widths=0.1)
    assert cloud.widths == pytest.approx([0.1])


def test_bulk_arrays_are_tensors():
    scene = ernie.Scene()
    quad = ernie.Mesh.define(scene, "/Quad")
    assert quad.points is None
    quad.write(tokens.POINTS, np.zeros((4, 3), np.float32))
    assert quad.points.shape == (4, 3)
    instancer = ernie.PointInstancer.define(scene, "/Rocks")
    instancer.write(tokens.POSITIONS, np.zeros((5, 3), np.float32))
    assert instancer.positions.shape == (5, 3) and instancer.proto_indices is None


def test_boundables_instancers_and_point_widths():
    assert issubclass(ernie.Gprim, ernie.Boundable) and issubclass(ernie.PointInstancer, ernie.Boundable)
    assert ernie.Boundable.prim_type.abstract
    scene = ernie.Scene()
    mesh = scene.define("/Mesh", ernie.Mesh, extent=[-1, -1, -1, 1, 1, 1])
    assert mesh.extent == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    rock = scene.define("/Protos/Rock", ernie.Mesh)
    instancer = scene.define("/Rocks", ernie.PointInstancer, prototypes=[rock, "/Protos/Tree"])
    assert instancer.prototypes == [ernie.Path("/Protos/Rock"), ernie.Path("/Protos/Tree")]
    assert ernie.PointInstancer.prototypes.kind == ernie.AttributeKind.RELATIONSHIPS
    assert instancer.worlds is None
    instancer.worlds = np.array([0, 1], np.int32)
    assert instancer.worlds.tolist() == [0, 1]
    cloud = scene.define("/Cloud", ernie.Points, widths=[0.5, 0.6, 0.7])
    assert cloud.widths == pytest.approx([0.5, 0.6, 0.7])
    assert ernie.Points.widths.default == [1.0]
