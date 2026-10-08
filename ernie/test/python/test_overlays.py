import numpy as np
import pytest

import ernie
from ernie import tokens

from conftest import make_terrain_scene


def test_overlays_build_from_python_values():
    scene = ernie.Scene()
    overlay = scene.overlay("debug")
    assert overlay is not None and overlay.empty
    endpoints = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]]], dtype=np.float32)
    returned = overlay.lines(endpoints, colors=[(1, 0, 0, 1), (0, 1, 0, 0.5)], width=3)
    overlay.lines(endpoints.reshape(-1, 3), colors=[(1, 0, 0), (0, 1, 0)])
    assert returned is overlay
    overlay.line((0, 0, 0), (0, 0, 1), color=(1, 1, 1), depth="xray", world=2).box(
        (-1, -1, -1), (1, 1, 1), transform=np.eye(4)
    ).arrow((0, 0, 0), (0, 0, 5)).sphere((0, 0, 0), 2.0).axes(np.eye(4), 0.5)
    overlay.polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], closed=True)
    overlay.wire_mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [[0, 1, 2]])
    overlay.triangles(np.zeros((3, 3)), np.array([0, 1, 2]))
    overlay.marker((0, 0, 0), shape="diamond", size=10).markers(np.zeros((4, 3)), shape="plus")
    overlay.label((0, 0, 0), "contact", size=14, background=(0, 0, 0, 0))
    assert "segments" in repr(overlay)
    assert scene.overlay_names() == ["debug"]

    version = overlay.version
    overlay.visible = False
    assert not overlay.visible and overlay.version != version
    with pytest.raises(TypeError, match="unknown overlay style"):
        overlay.line((0, 0, 0), (1, 0, 0), colour=(1, 0, 0))
    with pytest.raises(ValueError, match="depth"):
        overlay.line((0, 0, 0), (1, 0, 0), depth="behind")
    with pytest.raises(ValueError, match="shape"):
        overlay.marker((0, 0, 0), shape="star")
    with pytest.raises(ValueError):
        overlay.lines(np.zeros((3, 3)))
    assert scene.remove_overlay("debug")
    assert scene.overlay_names() == []


def test_overlays_draw_over_products_that_ask_for_them(renderer):
    scene = make_terrain_scene()
    plain = np.array(renderer.render(scene))
    scene.overlay("debug").line((-60, 0, 0), (60, 0, 0), color=(1, 0, 1), width=4, depth="always")
    np.testing.assert_array_equal(np.array(renderer.render(scene)), plain)
    scene.set("/Render/Settings", tokens.OVERLAYS, "*")
    drawn = np.array(renderer.render(scene))
    magenta = (drawn[..., 0] == 255) & (drawn[..., 1] == 0) & (drawn[..., 2] == 255)
    assert magenta.sum() > 20
    assert renderer.take_validation_messages() == []
