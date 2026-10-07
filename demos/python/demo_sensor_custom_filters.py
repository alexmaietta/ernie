"""Adds two custom sensor filters written in Slang and renders through them: an IR-cut filter on
the spectrum (a band filter, before the sensor integrates the bands) and hot pixels on the
electrons (a pixel filter, between QE and the ADC). Saves the RawDN and LdrColor images.

    PYTHONPATH=build/python python demos/python/demo_sensor_custom_filters.py [out_prefix]

A custom filter is a Slang struct implementing IBandFilter or IPixelFilter from module `sensor`,
with a nested Params struct of one float per parameter; SensorFilter prims name it in
ernie:filter:type and set its parameters as inputs:<name>. The renderer fuses it with the
built-in filters into one kernel, so it costs about as much as its own arithmetic.
"""

import sys

import numpy as np

import ernie
from ernie import tokens

IR_CUT = """
import sensor;

// Passes wavelengths below `cutoff` nm, rolling off over `width` nm above it.
struct IrCut : IBandFilter {
    struct Params { float cutoff; float width; }
    static float apply(float radiance, uint band, PixelContext pixel, Params params) {
        let t = saturate((pixel.wavelength(band) - params.cutoff) / max(params.width, 1e-3));
        return radiance * (1.0 - t);
    }
}
"""

HOT_PIXELS = """
import sensor;

// A fixed fraction of pixels collects `electrons` extra electrons in every channel.
struct HotPixels : IPixelFilter {
    struct Params { float fraction; float electrons; }
    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        if (pixel.fixedRandom(0x407u) < params.fraction) {
            value.electrons += params.electrons;
        }
    }
}
"""


def build_scene():
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    box = np.eye(4)
    box[2, 3] = 1.0
    scene.define("/World/Box", tokens.CUBE).set(tokens.SIZE, 2.0).set(tokens.TRANSFORM, box).set(
        tokens.DISPLAY_COLOR, (0.8, 0.3, 0.2)
    )
    # One pose, two exposures: the photographic default, and a short one leaving the sensor
    # unsaturated for the raw numbers.
    pose = ernie.look_at_transform((6.0, -6.0, 4.0), (0.0, 0.0, 1.0))
    scene.define("/World/Photo", tokens.CAMERA).set(tokens.TRANSFORM, pose).set(
        tokens.FOCAL_LENGTH, 18.0
    )
    scene.define("/World/Raw", tokens.CAMERA).set(tokens.TRANSFORM, pose).set(
        tokens.FOCAL_LENGTH, 18.0
    ).set(tokens.EXPOSURE_TIME, 0.0002)

    # The chain: lens (implied), IR cut, QE, hot pixels, noise, ADC, CIE, exposure.
    chain = [
        ("IrCut", "irCut", {"cutoff": 650.0, "width": 40.0}),
        ("Qe", "qe", {}),
        ("HotPixels", "hotPixels", {"fraction": 0.002, "electrons": 5000.0}),
        ("Noise", "noise", {}),
        ("Adc", "adc", {}),
        ("Cie", "cie", {}),
        ("Exposure", "exposure", {}),
    ]
    paths = []
    for name, kind, inputs in chain:
        prim = scene.define(f"/Render/Filters/{name}", tokens.SENSOR_FILTER).set(
            tokens.FILTER_TYPE, kind
        )
        for input_name, value in inputs.items():
            prim.set(f"inputs:{input_name}", value)
        paths.append(ernie.Path(f"/Render/Filters/{name}"))
    for product, var in (("Raw", "RawDN"), ("Photo", "LdrColor")):
        scene.define(f"/Render/Vars/{var}", tokens.RENDER_VAR).set(tokens.SOURCE_NAME, var)
        scene.define(f"/Render/{product}", tokens.RENDER_PRODUCT).set(
            tokens.CAMERA_REL, ernie.Path(f"/World/{product}")
        ).set(tokens.RESOLUTION, (320, 240)).set(
            tokens.ORDERED_VARS, [ernie.Path(f"/Render/Vars/{var}")]
        ).set(tokens.FILTERS, paths)
    return scene


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "demo_sensor_custom_filters"
    renderer = ernie.Renderer()
    renderer.add_sensor_filter(
        "irCut", "IrCut", IR_CUT, kind="band", params={"cutoff": 700.0, "width": 50.0}
    )
    renderer.add_sensor_filter(
        "hotPixels", "HotPixels", HOT_PIXELS, kind="pixel", domain="electrons",
        params={"fraction": 1e-4, "electrons": 2000.0},
    )
    outputs = renderer.step(build_scene(), render_products=["/Render/Raw", "/Render/Photo"])
    raw = np.array(outputs["/Render/Raw"].frames[0].render_vars["/Render/Vars/RawDN"].map())
    ldr = np.array(outputs["/Render/Photo"].frames[0].render_vars["/Render/Vars/LdrColor"].map())
    print(f"RawDN {raw.shape} {raw.dtype}, max {raw.max()}; LdrColor {ldr.shape}")
    try:
        from PIL import Image

        # Scaled to the 99th percentile: the scene in range, the hot pixels saturated.
        scale = 255.0 / max(float(np.percentile(raw[..., :3], 99)), 1.0)
        raw_image = np.clip(raw[..., :3] * scale, 0, 255).astype(np.uint8)
        Image.fromarray(raw_image).save(f"{prefix}_raw.png")
        Image.fromarray(ldr).save(f"{prefix}_ldr.png")
        print(f"Saved {prefix}_raw.png and {prefix}_ldr.png")
    except ImportError:
        np.save(f"{prefix}_raw.npy", raw)
        np.save(f"{prefix}_ldr.npy", ldr)
        print("Saved .npy files (install Pillow for PNGs)")


if __name__ == "__main__":
    main()
