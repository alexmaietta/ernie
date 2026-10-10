# SPDX-License-Identifier: BSD-3-Clause
"""Adds three custom sensor filters written in Slang and renders through them: an IR-cut filter on
the spectrum (a band filter, before the sensor integrates the bands), hot pixels on the
electrons (a pixel filter, between QE and the ADC) and a glow on linear colour (a spatial
filter, reading the pixels around its own). Saves the RawDN and LdrColor images.

    PYTHONPATH=build/python python demos/python/demo_sensor_custom_filters.py [out_prefix]

A custom filter is a Slang struct implementing IBandFilter, IPixelFilter, ISpatialBandFilter or
ISpatialPixelFilter from module `sensor`, with a nested Params struct of one float per
parameter; SensorFilter prims name it in ernie:filter:type and set its parameters as
inputs:<name>. The renderer fuses band and pixel filters with the built-in ones into one kernel,
so they cost about as much as their own arithmetic; a spatial filter splits the kernel, the part
before writing its domain to an image it reads.
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

GLOW = """
import sensor;

// Adds `strength` times the mean linear colour within `radius` pixels: a soft bloom, light
// spreading from every region into its surroundings.
struct Glow : ISpatialPixelFilter {
    struct Params { float radius; float strength; }
    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let r = int(params.radius);
        let centre = int2(pixel.id.xy);
        var sum = float3(0.0);
        for (int y = -r; y <= r; ++y) {
            for (int x = -r; x <= r; ++x) {
                sum += input.values(pixel, centre + int2(x, y)).xyz;
            }
        }
        let count = float((2 * r + 1) * (2 * r + 1));
        value.rgb = input.values(pixel, centre).xyz + params.strength * sum / count;
    }
}
"""


def build_scene():
    scene = ernie.Scene()
    world = scene.define("/World", ernie.Xform)
    world.define("Ground", ernie.GroundPlane)
    world.define("Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    world.define("Box", ernie.Cube, size=2.0, translate=(0.0, 0.0, 1.0),
                 display_color=(0.8, 0.3, 0.2))
    # One pose, two exposures: the photographic default, and a short one leaving the sensor
    # unsaturated for the raw numbers.
    pose = ernie.look_at_transform((6.0, -6.0, 4.0), (0.0, 0.0, 1.0))
    photo = world.define("Photo", ernie.Camera, focal_length=18.0).set(tokens.TRANSFORM, pose)
    raw = world.define("Raw", ernie.Camera, focal_length=18.0, exposure_time=0.0002).set(
        tokens.TRANSFORM, pose
    )

    # The chain: lens (implied), IR cut, QE, hot pixels, noise, ADC, CIE, glow, exposure. The glow
    # splits the colour branch into a second kernel; the raw branch stays in the first.
    chain = [
        ("IrCut", "irCut", {"cutoff": 650.0, "width": 40.0}),
        ("Qe", "qe", {}),
        ("HotPixels", "hotPixels", {"fraction": 0.002, "electrons": 5000.0}),
        ("Noise", "noise", {}),
        ("Adc", "adc", {}),
        ("Cie", "cie", {}),
        ("Glow", "glow", {"radius": 6.0, "strength": 0.2}),
        ("Exposure", "exposure", {}),
    ]
    render = scene.define("/Render", ernie.Scope)
    paths = []
    for name, kind, inputs in chain:
        prim = render.define(f"Filters/{name}", ernie.SensorFilter, filter_type=kind)
        # A filter's parameters are its own, so they have no Python names: set them by USD name.
        for input_name, value in inputs.items():
            prim.set(f"inputs:{input_name}", value)
        paths.append(prim.path)
    for product, camera, var in (("Raw", raw, "RawDN"), ("Photo", photo, "LdrColor")):
        render_var = render.define(f"Vars/{var}", ernie.RenderVar, source_name=var)
        render.define(product, ernie.RenderProduct, camera=camera, resolution=(320, 240)).set(
            tokens.ORDERED_VARS, [render_var.path]
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
    renderer.add_sensor_filter(
        "glow", "Glow", GLOW, kind="spatial", domain="rgb",
        params={"radius": 4.0, "strength": 0.3},
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
