# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from conftest import make_terrain_scene

import ernie
import ernie.batch


def test_overlays_build_from_python_values():
    scene = ernie.Scene()
    overlay = scene.overlay("debug")
    assert overlay is not None and overlay.empty
    endpoints = np.array(
        [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]]],
        dtype=np.float32,
    )
    returned = overlay.lines(
        endpoints, colors=[(1, 0, 0, 1), (0, 1, 0, 0.5)], width=3
    )
    overlay.lines(
        endpoints.reshape(-1, 3), colors=[(1, 0, 0), (0, 1, 0)]
    )
    assert returned is overlay
    overlay.line(
        (0, 0, 0), (0, 0, 1), color=(1, 1, 1), depth="xray", world=2
    ).box((-1, -1, -1), (1, 1, 1), transform=np.eye(4)).arrow(
        (0, 0, 0), (0, 0, 5)
    ).sphere((0, 0, 0), 2.0).axes(np.eye(4), 0.5)
    overlay.polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], closed=True)
    overlay.wire_mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [[0, 1, 2]])
    overlay.triangles(np.zeros((3, 3)), np.array([0, 1, 2]))
    overlay.marker((0, 0, 0), shape="diamond", size=10).markers(
        np.zeros((4, 3)), shape="plus"
    )
    overlay.label(
        (0, 0, 0), "contact", size=14, background=(0, 0, 0, 0)
    )
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
    scene.overlay("debug").line(
        (-60, 0, 0),
        (60, 0, 0),
        color=(1, 0, 1),
        width=4,
        depth="always",
    )
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), plain
    )
    ernie.RenderSettings.at(scene, "/Render/Settings").overlays = "*"
    drawn = np.array(renderer.render(scene))
    magenta = (
        (drawn[..., 0] == 255)
        & (drawn[..., 1] == 0)
        & (drawn[..., 2] == 255)
    )
    assert magenta.sum() > 20
    assert renderer.take_validation_messages() == []


def test_overlay_batches_check_their_arrays():
    overlay = ernie.Scene().overlay("debug")
    starts = np.zeros((4, 3))
    returned = overlay.segments(
        starts, starts + 1, colors=np.ones((4, 4)), widths=[2] * 4
    )
    assert returned is overlay and not overlay.empty
    overlay.arrows(starts, starts + 1, head=0.2, worlds=[0, 1, 2, 3])
    overlay.points(starts, shape="cross", sizes=np.full(4, 8.0))
    overlay.boxes(
        starts, np.ones((4, 3)), orientations=[(0, 0, 0, 1)] * 4
    )
    overlay.spheres(starts, np.ones(4)).frames(starts, length=0.5)
    with pytest.raises(ValueError, match="one row per item"):
        overlay.segments(starts, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="colours"):
        overlay.segments(starts, starts, colors=np.ones((4, 2)))
    with pytest.raises(ValueError, match="radii"):
        overlay.spheres(starts, np.ones(3))
    overlay.clear()
    assert overlay.empty


def batch_scene():
    """make_terrain_scene drawing every overlay over its image."""
    scene = make_terrain_scene()
    ernie.RenderSettings.at(scene, "/Render/Settings").overlays = "*"
    return scene


# Items each overlay method draws one of, with their batch's arrays.
STARTS = np.float32([[-30, -20, 0], [10, 5, 0], [25, -25, 0]])
ENDS = STARTS + np.float32([[20, 10, 0], [0, 20, 0], [-15, 15, 0]])
COLORS = np.float32([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
TURNS = np.float32(
    [[0, 0, 0, 1], [0, 0, 0.3826834, 0.9238795], [0, 0, 1, 0]]
)


def rotation(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def placed(q, at):
    transform = np.eye(4)
    transform[:3, :3] = rotation(q)
    transform[:3, 3] = at
    return transform


KINDS = ("arrows", "points", "boxes", "spheres", "frames")


def per_item(overlay, kind):
    style = {"width": 3, "depth": "always"}
    for start, end, colour, turn in zip(STARTS, ENDS, COLORS, TURNS):
        colour = tuple(float(c) for c in colour)
        if kind == "arrows":
            overlay.arrow(start, end, color=colour, **style)
        elif kind == "points":
            overlay.marker(
                end,
                size=10,
                shape="ring",
                color=colour,
                depth="always",
            )
        elif kind == "boxes":
            overlay.box(
                (-4, -2, -1),
                (4, 2, 1),
                color=colour,
                transform=placed(turn, start),
                **style,
            )
        elif kind == "spheres":
            overlay.sphere(end, 5.0, color=colour, **style)
        else:
            overlay.axes(placed(turn, start), 6.0, **style)


def batched(overlay, kind, array):
    style = {"width": 3, "depth": "always"}
    colors = array(COLORS)
    if kind == "arrows":
        overlay.arrows(
            array(STARTS), array(ENDS), colors=colors, **style
        )
    elif kind == "points":
        overlay.points(
            array(ENDS),
            shape="ring",
            colors=colors,
            size=10,
            depth="always",
        )
    elif kind == "boxes":
        overlay.boxes(
            array(STARTS),
            array(np.float32([[4, 2, 1]] * 3)),
            orientations=array(TURNS),
            colors=colors,
            **style,
        )
    elif kind == "spheres":
        overlay.spheres(
            array(ENDS),
            array(np.full(3, 5.0, np.float32)),
            colors=colors,
            **style,
        )
    else:
        overlay.frames(
            array(STARTS), array(TURNS), length=6.0, **style
        )


def differing(a, b):
    return int((np.abs(np.int32(a) - np.int32(b)).max(-1) > 8).sum())


@pytest.mark.parametrize("kind", KINDS)
def test_overlay_batches_draw_as_their_items(renderer, kind):
    scene = batch_scene()
    plain = np.array(renderer.render(scene))
    per_item(scene.overlay("debug"), kind)
    expected = np.array(renderer.render(scene))
    assert differing(expected, plain) > 10
    scene.overlay("debug").clear()
    batched(scene.overlay("debug"), kind, lambda a: a)
    # Within a level or so: points from float32 sines and cosines on
    # the GPU rather than the host's doubles.
    assert differing(np.array(renderer.render(scene)), expected) == 0
    assert renderer.take_validation_messages() == []


@pytest.mark.parametrize("kind", KINDS)
def test_overlay_batches_draw_from_cuda_arrays(renderer, kind):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    scene = batch_scene()
    batched(scene.overlay("debug"), kind, lambda a: a)
    expected = np.array(renderer.render(scene))
    scene.overlay("debug").clear()
    batched(scene.overlay("debug"), kind, renderer.cuda_upload)
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), expected
    )


def test_overlay_batches_show_in_their_worlds(renderer):
    # Three views of three worlds, from the same pose: each shows its
    # own world's marker, and the batch without worlds in all.
    scene = make_terrain_scene()
    camera = ernie.batch.BatchCamera(
        renderer, scene, "/World/Camera", 3, resolution=(48, 48)
    )
    scene.get_prim(camera.product).set("ernie:overlays", "*")
    overlay = scene.overlay("debug")
    centre = np.float32([[0, 0, 0]] * 3)
    overlay.points(
        centre,
        shape="square",
        size=12,
        depth="always",
        colors=np.eye(3, dtype=np.float32),
        worlds=np.arange(3, dtype=np.int32),
    )
    overlay.segments(
        np.float32([[-60, 30, 0]]),
        np.float32([[60, 30, 0]]),
        color=(1, 1, 1),
        width=4,
        depth="always",
    )
    images = camera.step(
        worlds=np.arange(3, dtype=np.int32), device="cpu"
    )["LdrColor"]
    images = np.asarray(images)
    assert images.shape[0] == 3
    for view in range(3):
        texel = images[view, 24, 24, :3]
        assert texel[view] > 200 and np.delete(texel, view).max() < 60
        white = (images[view, ..., :3] > 240).all(-1).sum()
        assert white > 20, (
            "the batch without worlds shows in every view"
        )
