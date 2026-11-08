# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie

WIDTH, HEIGHT = 6, 4
PRODUCT = "/Render/View"


def bench(scene):
    scene.define("/Sky", "DomeLight")
    camera = scene.define("/Cam", "Camera", {"exposureTime": 1e-6})
    pipe = scene.define_pipeline("/Raw")
    signal = pipe.add_node("Signal", "CameraSignal")
    integrate = pipe.add_node("Integrate", "Integrate", input=signal)
    adc = pipe.add_node("Adc", "Adc", input=integrate)
    pipe.connect_output("raw", adc)
    camera.bind_pipeline(pipe)
    scene.define("/Render", "Scope")
    scene.define("/Render/Rate", "RenderVar", {"sourceName": "ElectronRate"})
    scene.define("/Render/Raw", "RenderVar", {"sourceName": "raw"})
    scene.define(
        PRODUCT,
        "RenderProduct",
        {
            "camera": [camera],
            "resolution": (WIDTH, HEIGHT),
            "orderedVars": ["/Render/Rate", "/Render/Raw"],
            "renderMode": "RealTime",
        },
    )
    return pipe, integrate, adc


def render(renderer):
    output = renderer.step([PRODUCT], 0)
    output.wait()
    [frame] = output[PRODUCT].frames
    return frame.source("ElectronRate").tensor.numpy(), frame.source("raw").tensor.numpy()


def test_nodes_describe_their_ports():
    adc = ernie.pipeline_node("Adc")
    assert adc["kind"] == "point"
    assert adc["inputs"] == {"electrons": ("Electrons", False)}
    assert adc["outputs"] == {"digital": ("Digital", False)}
    assert ernie.pipeline_node("NoSuchNode") is None


def test_graphs_build_render_and_edit_in_place(scene):
    pipe, integrate, adc = bench(scene)
    renderer = ernie.Renderer(scene)
    rates, raw = render(renderer)
    assert raw.shape == (1, HEIGHT, WIDTH, 3)
    expected = np.clip(np.round(rates * 1e-6 * 0.25), 0, 4095)
    np.testing.assert_allclose(raw, expected, atol=1)
    assert raw.max() > 0

    gain = pipe.insert_after(adc, "Gain", "Gain", gain=2.0)
    assert ("raw", "/Raw/Gain", "digital") in pipe.connections
    assert ("digital", "/Raw/Adc", "digital") in gain.connections
    np.testing.assert_allclose(render(renderer)[1], raw * 2)

    gain["gain"] = 3.0
    np.testing.assert_allclose(render(renderer)[1], raw * 3)

    noise = pipe.insert_after(integrate, "Noise", "Emva1288")
    assert ("electrons", "/Raw/Noise", "electrons") in adc.connections
    pipe.remove_node(noise)
    pipe.remove_node(gain)
    assert adc.connections == [("electrons", "/Raw/Integrate", "electrons")]
    assert ("raw", "/Raw/Adc", "digital") in pipe.connections
    np.testing.assert_allclose(render(renderer)[1], raw)


def test_standalone_runs_match_the_traced_pipeline(scene):
    bench(scene)
    rates, raw = render(ernie.Renderer(scene))
    pipeline = ernie.Pipeline(scene, "/Raw")
    assert pipeline.sources == ["/Raw/Signal"]
    assert pipeline.outputs == ["raw"]
    detector = ernie.DetectorDesc(pattern="rgb")
    out = pipeline.run({"Signal": rates}, detector=detector, exposure_time=1e-6)
    np.testing.assert_array_equal(out["raw"].numpy(), raw)

    session = pipeline.session()
    session.run({"Signal": rates}, detector=detector, exposure_time=1e-6)
    session.run({"Signal": rates}, detector=detector, exposure_time=1e-6)
    assert session.captures == 2

    with pytest.raises(ValueError, match="detector pattern"):
        ernie.DetectorDesc(pattern="cmyk")
    with pytest.raises(RuntimeError, match="no input"):
        pipeline.run({}, detector=detector)
