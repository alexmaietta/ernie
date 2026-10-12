# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from conftest import make_terrain_scene

import ernie
import ernie.batch
from ernie import tokens


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


def test_overlay_batches_redraw_arrays_written_in_place(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    wp = pytest.importorskip("warp")
    scene = batch_scene()
    points = wp.array(
        np.float32([[-20, 0, 0]]), dtype=wp.vec3, device="cuda:0"
    )
    overlay = scene.overlay("debug").points(
        points,
        shape="square",
        size=8,
        color=(1, 0, 1),
        depth="always",
    )
    before = np.array(renderer.render(scene))
    points.assign(np.float32([[20, 0, 0]]))
    overlay.touch()
    after = np.array(renderer.render(scene))
    magenta = lambda image: (  # noqa: E731
        (image[..., 0] > 200)
        & (image[..., 1] < 60)
        & (image[..., 2] > 200)
    )
    columns = lambda image: np.nonzero(magenta(image).any(0))[0]  # noqa: E731
    assert columns(before).size and columns(after).size
    assert columns(after).mean() > columns(before).mean() + 5


def test_overlay_gpu_arrays_redraw_without_a_touch(renderer):
    # GPU arrays are read every frame: written in place, they draw
    # anew with no touch.
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    wp = pytest.importorskip("warp")
    scene = batch_scene()
    points = wp.array(
        np.float32([[-20, 0, 0]]), dtype=wp.vec3, device="cuda:0"
    )
    scene.overlay("debug").points(
        points,
        shape="square",
        size=8,
        color=(1, 0, 1),
        depth="always",
    )
    before = np.array(renderer.render(scene))
    points.assign(np.float32([[20, 0, 0]]))
    after = np.array(renderer.render(scene))
    assert differing(before, after) > 20


def test_overlay_arrays_map_into_the_renderers_memory(renderer):
    # A batch's points mapped in place: written there, they draw as
    # the same points given as a host array.
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    wp = pytest.importorskip("warp")
    style = dict(
        shape="square", size=8, color=(0, 1, 1), depth="always"
    )
    moved = np.float32([[15, -10, 0], [-15, 10, 0]])
    reference = batch_scene()
    reference.overlay("debug").points(moved, **style)
    expected = np.array(renderer.render(reference))

    scene = batch_scene()
    overlay = scene.overlay("debug").points(
        renderer.cuda_upload(np.zeros((2, 3), np.float32)), **style
    )
    renderer.render(scene)
    with pytest.raises(ValueError, match="ends"):
        renderer.map_overlay(scene, "debug", 0, "ends")
    mapped = wp.from_dlpack(
        renderer.map_overlay(
            scene, "debug", overlay.batch_count - 1, "points"
        )
    )
    mapped.assign(moved)
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), expected
    )


def test_overlays_follow_prims_posed_on_the_gpu(renderer):
    # Markers and labels posed under links whose translations are CUDA
    # rows draw where the same items in world coordinates do.
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    places = np.float32([[-20, -10, 0], [5, 15, 0], [25, -20, 0]])
    style = dict(
        shape="square", size=8, color=(1, 0.5, 0), depth="always"
    )

    reference = batch_scene()
    debug = reference.overlay("debug").points(places, **style)
    debug.label(places[0].tolist(), "link 0", depth="always")
    expected = np.array(renderer.render(reference))

    scene = batch_scene()
    links = []
    for i in range(len(places)):
        link = scene.define(f"/World/Env{i}/Link", ernie.Xform).set(
            tokens.XFORM_OP_ORDER, [tokens.TRANSLATE]
        )
        links.append(str(link.path))
    scene.write(links, tokens.TRANSLATE, renderer.cuda_upload(places))
    debug = scene.overlay("debug").points(
        np.zeros((3, 3), np.float32), anchors=links, **style
    )
    debug.label((0, 0, 0), "link 0", depth="always", anchor=links[0])
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), expected
    )
    # Moved on the GPU alone, the overlay unchanged.
    version = debug.version
    scene.write(
        links, tokens.TRANSLATE, renderer.cuda_upload(places + 5)
    )
    assert differing(np.array(renderer.render(scene)), expected) > 20
    assert debug.version == version


def test_hud_numbers_print_cuda_values_as_host_text(renderer):
    if not renderer.cuda_available:
        pytest.skip("No CUDA interop on this machine")
    values = np.float32([3.14159, -42.0])
    style = dict(size=12, color=(1, 1, 1), background=(0, 0, 0, 0.6))
    host = batch_scene()
    for i, text in enumerate(["sim 3.142 s", "sim -42.000 s"]):
        host.overlay("hud").text(
            text, at=(1, 0), offset=(-4, 4 + 18 * i), **style
        )
    expected = np.array(renderer.render(host))

    scene = batch_scene()
    scene.overlay("hud").numbers(
        renderer.cuda_upload(values),
        prefix="sim ",
        suffix=" s",
        decimals=3,
        screen=True,
        at=(1, 0),
        offset=(-4, 4),
        **style,
    )
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), expected
    )


POINT_STYLE = dict(
    shape="square", size=8, color=(1, 0, 1), depth="always"
)


def render_overlay(renderer, draw):
    scene = batch_scene()
    draw(scene.overlay("debug"))
    return np.array(renderer.render(scene))


def test_overlay_counts_draw_only_rows_in_use(renderer):
    # A count of 2 draws the first two rows; written on the GPU, a new
    # count draws anew with no touch.
    expected = render_overlay(
        renderer, lambda o: o.points(STARTS[:2], **POINT_STYLE)
    )
    counted = render_overlay(
        renderer,
        lambda o: o.points(
            STARTS, count=np.int32([2]), **POINT_STYLE
        ),
    )
    np.testing.assert_array_equal(counted, expected)
    # Sorted by world, as a batched camera's views draw them.
    worlds = render_overlay(
        renderer,
        lambda o: o.points(
            STARTS,
            count=np.int32([2]),
            worlds=np.full(3, -1, np.int32),
            **POINT_STYLE,
        ),
    )
    np.testing.assert_array_equal(worlds, expected)
    if not renderer.cuda_available:
        return
    wp = pytest.importorskip("warp")
    count = wp.array(np.int32([0]), dtype=wp.int32, device="cuda:0")
    scene = batch_scene()
    scene.overlay("debug").points(
        renderer.cuda_upload(STARTS), count=count, **POINT_STYLE
    )
    empty = np.array(renderer.render(scene))
    count.assign(np.int32([2]))
    np.testing.assert_array_equal(
        np.array(renderer.render(scene)), expected
    )
    assert differing(empty, expected) > 20


def test_overlay_items_follow_pose_arrays(renderer):
    # Boxes at their poses' origins, turned by them, from rows of
    # position and quaternion x, y, z, w, picked by pose indices.
    expected = render_overlay(
        renderer,
        lambda o: o.boxes(
            STARTS, np.full((3, 3), 3.0), orientations=TURNS, width=3
        ),
    )
    poses = np.concatenate([STARTS, TURNS], axis=1).astype(np.float32)
    shuffled = poses[[2, 0, 1]]
    posed = render_overlay(
        renderer,
        lambda o: o.boxes(
            np.zeros((3, 3), np.float32),
            np.full((3, 3), 3.0),
            poses=shuffled,
            pose_indices=np.int32([1, 2, 0]),
            width=3,
        ),
    )
    assert differing(posed, expected) == 0
    # A frame moving every pose.
    frame = np.eye(4)
    frame[:3, 3] = (5, 0, 0)
    moved = render_overlay(
        renderer,
        lambda o: o.boxes(
            np.zeros((3, 3), np.float32),
            np.full((3, 3), 3.0),
            poses=poses,
            pose_frame=frame,
            width=3,
        ),
    )
    shifted = render_overlay(
        renderer,
        lambda o: o.boxes(
            STARTS + np.float32([5, 0, 0]),
            np.full((3, 3), 3.0),
            orientations=TURNS,
            width=3,
        ),
    )
    assert differing(moved, shifted) == 0


def test_overlay_directions_scale_vectors_from_the_starts(renderer):
    vectors = (ENDS - STARTS) * 0.5
    expected = render_overlay(
        renderer, lambda o: o.arrows(STARTS, ENDS, width=3)
    )
    directed = render_overlay(
        renderer,
        lambda o: o.arrows(
            STARTS, vectors, directions=True, scale=2.0, width=3
        ),
    )
    assert differing(directed, expected) == 0


def test_overlay_trails_draw_their_rings_from_the_head(renderer):
    # A ring of 5 points whose newest is at index 1: segments 1-0,
    # 0-4, 4-3, 3-2; a NaN point leaves out the segments meeting it.
    ring = np.float32(
        [[[0, 0, 0], [-5, 0, 0], [20, 10, 0], [15, 5, 0], [10, 0, 0]]]
    )
    order = [1, 0, 4, 3, 2]
    path = ring[0][order]
    expected = render_overlay(
        renderer, lambda o: o.segments(path[:-1], path[1:], width=3)
    )
    trailed = render_overlay(
        renderer,
        lambda o: o.trails(
            ring, heads=np.int32([1]), fade=False, width=3
        ),
    )
    assert differing(trailed, expected) == 0
    gapped = ring.copy()
    gapped[0, 4] = np.nan
    expected = render_overlay(
        renderer,
        lambda o: o.segments(path[:1], path[1:2], width=3).segments(
            path[3:4], path[4:5], width=3
        ),
    )
    trailed = render_overlay(
        renderer,
        lambda o: o.trails(
            gapped, heads=np.int32([1]), fade=False, width=3
        ),
    )
    assert differing(trailed, expected) == 0


def test_overlay_capsules_draw_around_their_axes(renderer):
    plain = render_overlay(renderer, lambda o: None)
    capsule = dict(
        radii=np.float32([12.0]),
        half_heights=np.float32([8.0]),
        width=2,
        depth="always",
    )
    upright = render_overlay(
        renderer,
        lambda o: o.capsules(np.float32([[0, 0, 0]]), **capsule),
    )
    assert differing(upright, plain) > 50
    # Half a turn about x: the same capsule.
    flipped = render_overlay(
        renderer,
        lambda o: o.capsules(
            np.float32([[0, 0, 0]]),
            orientations=np.float32([[1, 0, 0, 0]]),
            **capsule,
        ),
    )
    assert differing(flipped, upright) < 20
    with pytest.raises(ValueError, match="half heights"):
        ernie.Scene().overlay("o").capsules(
            np.zeros((1, 3)), np.ones(1), np.ones(2)
        )


def test_overlay_trails_colour_each_segment_by_its_newer_point(renderer):
    ring = np.float32([[[-20, 0, 0], [0, 10, 0], [20, 0, 0]]])
    colors = np.float32([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    # Newest last: segments 2-1 in blue, 1-0 in green.
    expected = render_overlay(
        renderer,
        lambda o: o.segments(
            ring[0][[2, 1]],
            ring[0][[1, 0]],
            colors=colors[0][[2, 1]],
            width=3,
        ),
    )
    trailed = render_overlay(
        renderer,
        lambda o: o.trails(ring, colors=colors, fade=False, width=3),
    )
    assert differing(trailed, expected) == 0
