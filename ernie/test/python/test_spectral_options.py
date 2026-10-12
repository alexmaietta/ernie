# SPDX-License-Identifier: BSD-3-Clause
"""The camera's spectral mode (ernie:spectral:mode: spectral bands or
RGB mode's three) and the raster radiance's precision
(ernie:spectral:precision: 32- or 16-bit floats)."""

import numpy as np
import pytest

import ernie
from ernie import tokens


def cube_scene(mode=None, precision=None):
    """A cyan cube and an orange ball under the sun."""
    scene = ernie.Scene()
    scene.define(
        "/World/Cube",
        ernie.Cube,
        size=2.0,
        display_color=(0.0, 0.8, 0.8),
    )
    scene.define(
        "/World/Ball",
        ernie.Sphere,
        radius=0.7,
        translate=(1.6, 0.0, 0.7),
        display_color=(0.9, 0.3, 0.2),
    )
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM,
        ernie.distant_light_transform((0.2, -0.3, 1.0)),
    )
    camera = scene.define(
        "/World/Camera", ernie.Camera, sensor_noise=False
    ).set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0.5, -0.5, 6), (0, 0, 0)),
    )
    if mode is not None:
        camera.set(tokens.SPECTRAL_MODE, mode)
    scene.define(
        "/Render/Vars/Hdr", ernie.RenderVar, source_name="HdrColor"
    )
    scene.define(
        "/Render/Out",
        ernie.RenderProduct,
        camera=camera,
        resolution=(64, 64),
    ).set(tokens.ORDERED_VARS, [ernie.Path("/Render/Vars/Hdr")])
    settings = scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        products=[ernie.Path("/Render/Out")],
    )
    if precision is not None:
        settings.set(tokens.SPECTRAL_PRECISION, precision)
    return scene


def hdr(scene):
    frames = ernie.Renderer().step(scene)
    var = (
        frames[next(iter(frames))]
        .frames[0]
        .render_vars["/Render/Vars/Hdr"]
    )
    return np.asarray(var.map())[..., :3].astype(np.float64)


def test_rgb_mode_keeps_colours_pure_and_matches_the_spectral_image():
    spectral = hdr(cube_scene())
    rgb = hdr(cube_scene(tokens.RGB))
    centre = rgb[32, 32]
    assert abs(centre[0]) < 1e-6 * centre[1], "cyan has no red"
    assert centre[1] == pytest.approx(centre[2], rel=1e-3)
    # Spectral bands mix saturated colours a little (the cyan's red).
    assert np.abs(rgb - spectral).mean() < 0.05 * spectral.mean()


@pytest.mark.parametrize("mode", [tokens.SPECTRAL, tokens.RGB])
def test_half_precision_radiance_keeps_the_image(mode):
    # Radiance in this scene's units (about 1e-5) lies below 16-bit
    # floats' normal range: stored scaled to the exposure, it keeps
    # their precision.
    full = hdr(cube_scene(mode, tokens.FLOAT32))
    half = hdr(cube_scene(mode, tokens.FLOAT16))
    relative = np.abs(half - full) / np.maximum(np.abs(full), 1e-3)
    assert relative.max() < 0.01


def test_unknown_modes_and_precisions_are_refused():
    with pytest.raises(ValueError, match="float32 or float16"):
        hdr(cube_scene(precision="float8"))
    with pytest.raises(ValueError, match="spectral or rgb"):
        hdr(cube_scene(mode="hsv"))
