"""Transform ops of prims posed on the GPU mapped over the renderer's own memory: rows written in
place render as rows written from the simulator's arrays, with nothing copied."""

import numpy as np
import pytest

import ernie
from ernie import tokens
from test_xform_gpu import assert_same_image, cuda_renderer, staged_scene  # noqa: F401
from test_xform_gpu_compose import links, rng_arrays

PLAIN = (tokens.TRANSLATE, tokens.ORIENT, tokens.SCALE)
COMPOSED = (tokens.TRANSLATE, tokens.ROTATE_XYZ, tokens.SCALE)


def posed(renderer, ops, arrays, upload=True):
    scene = staged_scene()
    paths = links(scene, 4, ops)
    for op, rows in arrays.items():
        scene.write(paths, op, renderer.cuda_upload(rows) if upload else rows)
    return scene, ernie.PrimSet(paths)


def write_mapped(renderer, scene, prims, arrays):
    """Maps each op, checks it holds the rows the scene had, and writes `arrays` into it."""
    for op, rows in arrays.items():
        before = np.array(renderer.cuda_download(scene.read(prims.paths[0], op)))
        mapped = renderer.map(scene, prims, op)
        held = np.array(renderer.cuda_download(mapped))
        assert held.shape == rows.shape
        np.testing.assert_array_equal(held[0], before)
        renderer.cuda_copy(renderer.cuda_upload(rows), mapped)
        scene.unmap(prims, op)


@pytest.mark.parametrize("ops", [PLAIN, COMPOSED], ids=["plain", "composed"])
def test_mapped_rows_render_as_written_ones(cuda_renderer, ops):
    first, moved = rng_arrays(4, ops, seed=3), rng_arrays(4, ops, seed=4)
    scene, prims = posed(cuda_renderer, ops, first)
    cuda_renderer.render(scene)
    for _ in range(2):  # mapped again each step
        write_mapped(cuda_renderer, scene, prims, moved)
        expected, _ = posed(cuda_renderer, ops, moved, upload=False)
        assert_same_image(cuda_renderer.render(scene), cuda_renderer.render(expected))
        moved = rng_arrays(4, ops, seed=5)
    assert cuda_renderer.take_validation_messages() == []


def test_a_view_kept_past_its_step_still_renders(cuda_renderer):
    first, moved = rng_arrays(4, PLAIN, seed=3), rng_arrays(4, PLAIN, seed=6)
    scene, prims = posed(cuda_renderer, PLAIN, first)
    cuda_renderer.render(scene)
    stale = cuda_renderer.map(scene, prims, tokens.TRANSLATE)
    scene.unmap(prims, tokens.TRANSLATE)
    cuda_renderer.render(scene)
    # The view is of the half the last frame drew; handed back, it is copied from.
    cuda_renderer.cuda_copy(cuda_renderer.cuda_upload(moved[tokens.TRANSLATE]), stale)
    scene.unmap(prims, tokens.TRANSLATE)
    expected, _ = posed(cuda_renderer, PLAIN, {**first, tokens.TRANSLATE: moved[tokens.TRANSLATE]},
                        upload=False)
    assert_same_image(cuda_renderer.render(scene), cuda_renderer.render(expected))


def test_only_a_whole_group_in_row_order_maps(cuda_renderer):
    scene, prims = posed(cuda_renderer, PLAIN, rng_arrays(4, PLAIN, seed=3))
    cuda_renderer.render(scene)
    paths = prims.paths
    with pytest.raises(ValueError, match="every prim"):
        cuda_renderer.map(scene, paths[:2], tokens.TRANSLATE)
    with pytest.raises(ValueError, match="not row 0"):
        cuda_renderer.map(scene, list(reversed(paths)), tokens.TRANSLATE)
    with pytest.raises(ValueError, match="no such transform op"):
        cuda_renderer.map(scene, paths, tokens.ROTATE_X)
    # A single prim of a larger group is refused too, while lists of paths map.
    with pytest.raises(ValueError, match="every prim"):
        cuda_renderer.map(scene, paths[0], tokens.TRANSLATE)
    mapped = cuda_renderer.map(scene, paths, tokens.TRANSLATE)
    assert np.array(cuda_renderer.cuda_download(mapped)).shape == (4, 3)
    scene.unmap(paths, tokens.TRANSLATE)
