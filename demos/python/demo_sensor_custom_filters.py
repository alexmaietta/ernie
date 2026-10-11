# SPDX-License-Identifier: BSD-3-Clause
"""Adds three custom sensor filters written in Slang and renders through them: an IR-cut filter on
the spectrum (a band filter, before the sensor integrates the bands), hot pixels on the
electrons (a pixel filter, between QE and the ADC) and a glow on linear colour (a spatial
filter, reading the pixels around its own). `--output` writes the RawDN and LdrColor images side
by side.

    PYTHONPATH=build/python python demos/python/demo_sensor_custom_filters.py \
        [--viewer null --output filters.png]

A custom filter is a Slang struct implementing IBandFilter, IPixelFilter, ISpatialBandFilter or
ISpatialPixelFilter from module `sensor`, with a nested Params struct of one float per
parameter; SensorFilter prims name it in ernie:filter:type and set its parameters as
inputs:<name>. The renderer fuses band and pixel filters with the built-in ones into one kernel,
so they cost about as much as their own arithmetic; a spatial filter splits the kernel, the part
before writing its domain to an image it reads.

The filters are added to the demo's own Renderer, which renders the two products; in a viewer the
viewport shows the photographic camera without them.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

from demo_support import side_by_side

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


def build_scene(scene):
    """Builds the box, sun, both cameras, the filter chain and the two products into `scene`;
    returns the photographic camera."""
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
    return photo


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        photo = build_scene(self.scene)
        renderer = self.renderer
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
        self.raw = None
        self.ldr = None
        if viewer is not None:
            viewer.active_camera = photo.path

    def step(self):
        if self.raw is not None:
            return  # a still scene: the products rendered once
        outputs = self.renderer.step(self.scene, render_products=["/Render/Raw", "/Render/Photo"])
        frames = {path: outputs[path].frames[0] for path in ("/Render/Raw", "/Render/Photo")}
        self.raw = np.array(frames["/Render/Raw"].render_vars["/Render/Vars/RawDN"].map())
        self.ldr = np.array(frames["/Render/Photo"].render_vars["/Render/Vars/LdrColor"].map())
        print(f"RawDN {self.raw.shape} {self.raw.dtype}, max {self.raw.max()}; "
              f"LdrColor {self.ldr.shape}")

    def render(self):
        # Without a viewer the frame is the photographic product, through the custom filters.
        if self.viewer is not None:
            super().render()
        else:
            self.image = self.ldr

    def save_output(self, path):
        # RawDN scaled to its 99th percentile (the scene in range, the hot pixels saturated)
        # beside LdrColor.
        scale = 255.0 / max(float(np.percentile(self.raw[..., :3], 99)), 1.0)
        raw_image = np.clip(self.raw[..., :3] * scale, 0, 255).astype(np.uint8)
        ernie.demo.write_png(path, side_by_side([raw_image, self.ldr[..., :3]]))

    def test_final(self):
        raw = self.raw[..., :3].astype(np.float64)
        ldr = self.ldr[..., :3]
        assert raw.max() > 0 and ldr.std() > 1.0, "both products should show the scene"
        # Hot pixels: a fraction of 0.002 with 5000 extra electrons stands far above the scene.
        hot = np.mean(raw.max(axis=-1) > 4.0 * np.percentile(raw, 99))
        assert 1e-4 < hot < 0.01, f"about 0.2% of pixels should be hot, not {hot:.4%}"


if __name__ == "__main__":
    ernie.demo.main(Demo)
