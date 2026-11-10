# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie

xr = pytest.importorskip("ernie.xr")


def simulated(script=None):
    desc = xr.SessionDesc()
    desc.backend = xr.Backend.SIMULATED
    profile = xr.SimulatedProfile.quest3()
    profile.resolution = (16, 12)
    desc.simulated.profile = profile
    desc.simulated.place_controllers = True
    if script is not None:
        desc.simulated.script = script
    return xr.Session(desc)


def ball_ahead(scene):
    ernie.Sphere.define(scene, "/Ball", translate=(0.2, 5, 0.96), radius=1)


def test_a_simulated_session_presents_both_eyes(scene):
    ball_ahead(scene)
    session = simulated()
    rig = xr.Rig(scene, session)
    renderer = ernie.Renderer(scene)
    for _ in range(2):
        frame = session.begin_frame()
        rig.update(frame)
        rig.draw(renderer.overlays, frame)
        session.end_frame(frame, renderer.step([rig.product], dt=frame.delta_time))
    color = session.presented.numpy()
    depth = session.presented_depth.numpy()
    assert color.shape == (2, 12, 16, 4)
    assert depth.shape == (2, 12, 16)
    assert depth.min() < 1.0
    assert session.status.frames_presented == 2
    assert renderer.overlays.names() == ["xr"]


def test_scripts_drive_controllers_and_bindings(scene):
    def script(index, time, state):
        if index == 1:
            state.press(xr.Side.RIGHT, "trigger/value", 0.9)

    session = simulated(script)
    binding = xr.ActionBinding("select", input=xr.Input.TRIGGER)
    session.push_bindings("base", [binding])
    assert not session.begin_frame().held("select")
    frame = session.begin_frame()
    assert frame.pressed("select")
    assert frame.controller(xr.Side.RIGHT).trigger == pytest.approx(0.9)
    assert session.binding_layers == ["base"]


def test_pointers_and_locomotion_move_the_rig(scene):
    ball_ahead(scene)
    session = simulated()
    rig = xr.Rig(scene, session)
    frame = session.begin_frame()
    rig.update(frame)
    laser = rig.laser(xr.Side.RIGHT)
    assert laser.valid
    assert laser.local_position[1] == pytest.approx(4, abs=1e-3)

    rig.move(frame, (0, 1))
    assert rig.root_pose.position[1] == pytest.approx(1.5 * frame.delta_time)
    rig.snap_turn(frame, 1)
    rig.user_scale = 2
    assert rig.user_scale == 2
    assert np.allclose(ernie.Xform(rig.root).scale, (2, 2, 2))


def test_anchors_put_the_stage_on_their_target(scene):
    ernie.Xform.define(scene, "/Car", translate=(10, 0, 3))
    session = simulated()
    rig = xr.Rig(scene, session)
    rig.anchor = xr.AnchorDesc(xr.Anchor.PRIM, "/Car")
    frame = session.begin_frame()
    rig.update(frame)
    origin = rig.world_pose(xr.Pose((0, 0, 0)))
    assert origin.position == pytest.approx((10, 0, 3))
    assert rig.anchor.mode == xr.Anchor.PRIM
    with pytest.raises(RuntimeError):
        rig.anchor = xr.AnchorDesc(xr.Anchor.PRIM, "/Nowhere")
