# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens

GAIN = """
import sensor;
struct Gain : IPixelFilter {
    struct Params { float scale; }
    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.rgb *= params.scale;
    }
}
"""


def hdr_through(renderer, scene, types, inputs=None):
    """HdrColor of the scene's camera through SensorFilter prims of
    `types`."""
    paths = []
    for i, kind in enumerate(types):
        path = f"/Render/Filters/F{i}"
        prim = scene.define(
            path, ernie.SensorFilter, filter_type=kind
        )
        for name, value in (inputs or {}).items():
            if kind != "cie":
                prim.set(f"inputs:{name}", value)
        paths.append(ernie.Path(path))
    scene.define(
        "/Render/Vars/Hdr", ernie.RenderVar, source_name="HdrColor"
    )
    scene.define(
        "/Render/Hdr",
        ernie.RenderProduct,
        camera="/World/Camera",
        resolution=(48, 48),
    ).set(tokens.ORDERED_VARS, [ernie.Path("/Render/Vars/Hdr")]).set(
        tokens.FILTERS, paths
    )
    frame = renderer.step(scene, render_products=["/Render/Hdr"])[
        "/Render/Hdr"
    ].frames[0]
    return np.array(frame.render_vars["/Render/Vars/Hdr"].map())


def test_custom_filters_registered_from_python_run_in_chains(
    renderer, terrain_scene
):
    if "gain" not in renderer.sensor_filters:
        renderer.add_sensor_filter(
            "gain",
            "Gain",
            GAIN,
            kind="pixel",
            domain="rgb",
            params={"scale": 2.0},
        )
    assert renderer.sensor_filters[:6] == [
        "lens",
        "qe",
        "noise",
        "adc",
        "cie",
        "exposure",
    ]
    assert "gain" in renderer.sensor_filters
    base = hdr_through(renderer, terrain_scene, ["cie"])
    assert base[..., :3].max() > 0
    expected = base.copy()
    expected[..., :3] *= np.float32(3.0)
    np.testing.assert_array_equal(
        hdr_through(
            renderer, terrain_scene, ["cie", "gain"], {"scale": 3.0}
        ),
        expected,
    )
    assert renderer.take_validation_messages() == []


SHIFT = """
import sensor;
struct Shift : ISpatialPixelFilter {
    struct Params { float dx; }
    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        value.rgb = input.values(pixel, int2(pixel.id.xy) - int2(int(params.dx), 0)).xyz;
    }
}
"""


def test_spatial_filters_read_neighbouring_pixels(
    renderer, terrain_scene
):
    if "shift" not in renderer.sensor_filters:
        renderer.add_sensor_filter(
            "shift",
            "Shift",
            SHIFT,
            kind="spatial",
            domain="rgb",
            params={"dx": 1.0},
        )
    base = hdr_through(renderer, terrain_scene, ["cie"])
    shifted = hdr_through(
        renderer, terrain_scene, ["cie", "shift"], {"dx": 2.0}
    )
    expected = base.copy()
    expected[:, 2:] = base[:, :-2]
    expected[:, :2] = base[
        :, :1
    ]  # the left edge repeats its first column
    np.testing.assert_array_equal(shifted, expected)
    assert renderer.take_validation_messages() == []


def test_add_sensor_filter_rejects_bad_filters(renderer):
    with pytest.raises(ValueError):
        renderer.add_sensor_filter(
            "bad1", "Gain", GAIN, kind="spatial"
        )
    with pytest.raises(ValueError):
        renderer.add_sensor_filter(
            "bad2", "Gain", GAIN, kind="pixel", domain="spectrum"
        )
    with pytest.raises(ValueError, match="failed to compile"):
        renderer.add_sensor_filter(
            "bad3",
            "Gain",
            GAIN.replace("params.scale", "nothing"),
            domain="rgb",
            params={"scale": 1.0},
        )
    assert "bad3" not in renderer.sensor_filters
