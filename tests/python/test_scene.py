# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest

import ernie


def test_defined_prims_read_back_every_kind_of_value(scene):
    half = math.sqrt(0.5)
    ball = ernie.Sphere.define(
        scene,
        "/Ball",
        radius=2.5,
        translate=(1, 2, 3),
        orient=(0, 0, half, half),
        display_color=(0.1, 0.2, 0.3),
        visibility="invisible",
        semantics_labels={"class": ["ball", "toy"]},
    )
    assert ball.path == "/Ball"
    assert ball.name == "Ball"
    assert ball.radius == 2.5
    assert ball.translate == (1, 2, 3)
    np.testing.assert_allclose(ball.orient, (0, 0, half, half))
    np.testing.assert_allclose(ball.display_color, (0.1, 0.2, 0.3), rtol=1e-6)
    assert ball.visibility == "invisible"

    ball.radius = 4
    ball.visibility = "inherited"
    assert ball.radius == 4
    assert ball.visibility == "inherited"

    product = ernie.RenderProduct.define(
        scene, "/Product", resolution=(64, 48), render_mode="RealTime", camera=["/Cam"]
    )
    assert product.resolution == (64, 48)
    assert product.render_mode == "RealTime"


def test_bad_values_raise(scene):
    ball = ernie.Sphere.define(scene, "/Ball")
    with pytest.raises((TypeError, ValueError)):
        ball.radius = "wide"
    with pytest.raises(RuntimeError, match="translate"):
        ball.translate = (1, 2)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        ball.visibility = "sometimes"
    with pytest.raises(AttributeError):
        ball.no_such_field = 1
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        ernie.Sphere.define(scene, "/Other", radius="huge")


def test_selections_read_and_write_arrays(scene):
    for i in range(3):
        ernie.Sphere.define(scene, f"/Ball{i}", radius=i + 1)
    ernie.Cube.define(scene, "/Box")
    balls = ernie.Sphere.select(scene, "/Ball*")
    assert len(balls) == 3
    assert balls.paths == ["/Ball0", "/Ball1", "/Ball2"]
    np.testing.assert_array_equal(balls.radius, [1, 2, 3])

    balls.radius = 9
    np.testing.assert_array_equal(balls.radius, [9, 9, 9])
    balls.translate = np.arange(9, dtype=np.float64).reshape(3, 3)
    np.testing.assert_array_equal(balls.translate[2], [6, 7, 8])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        balls.radius = [1, 2]


def test_structure_edits_keep_paths_consistent(scene):
    ernie.Xform.define(scene, "/Group")
    ernie.Sphere.define(scene, "/Group/Ball", radius=3)
    ernie.Xform.define(scene, "/Other")
    assert scene.find("/Group/Ball") is not None
    assert scene.find("/Nothing") is None

    scene.rename(scene.find("/Group"), "Renamed")
    assert scene.find("/Renamed/Ball") is not None
    scene.move(scene.find("/Renamed"), "/Other/Renamed")
    assert scene.find("/Other/Renamed/Ball") is not None

    copied = scene.copy(scene.find("/Other/Renamed"))
    scene.paste(copied, "/Pasted")
    assert ernie.Sphere.select(scene, "/Pasted/Ball").radius[0] == 3

    count = scene.prim_count
    scene.remove(scene.find("/Pasted"))
    assert scene.prim_count == count - 2
    with pytest.raises(RuntimeError):
        ernie.Sphere.define(scene, "relative")


def test_worlds_are_counted_and_selected(scene):
    for world in range(3):
        ernie.World.define(scene, f"/env_{world}")
        ernie.Sphere.define(scene, f"/env_{world}/Ball")
    assert scene.world_count == 3
    balls = ernie.Sphere.select(scene, "/env_*/Ball")
    assert len(balls.world(1)) == 1
    assert balls.world(1).paths == ["/env_1/Ball"]
