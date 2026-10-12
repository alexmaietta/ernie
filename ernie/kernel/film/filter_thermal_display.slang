// SPDX-License-Identifier: BSD-3-Clause
// A thermal camera's picture: the brightness temperature a radiometric integrator earlier in the
// chain found (`bolometer`) from `low` to `high` kelvin, or with `automatic` 1 over the range of
// the product's last frame (its automatic gain control, linear between the least and most its
// prims showed, held at least `span` kelvin wide; the fixed range until it has one), in `palette`
// 0 white-hot, 1
// black-hot, 2 ironbow or 3 rainbow. Shown as it is: the palette's colour over the exposure's
// scale, which `exposure` applies. An integrator into colour, in place of `cie`.
//
// Reference: M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging", 2nd ed., Wiley-VCH, 2018
// (display ranges, automatic gain control and colour palettes).
import sensor;

// The palette's colour at `t` in [0, 1], sRGB-encoded.
float3 paletteAt(uint palette, float t) {
    let x = saturate(t);
    if (palette == 1) {
        return float3(1.0 - x);
    }
    if (palette == 2) {
        static const float3 iron[6] = { float3(0.0, 0.0, 0.0),  float3(0.15, 0.0, 0.5),
                                        float3(0.6, 0.0, 0.6),  float3(0.9, 0.25, 0.1),
                                        float3(1.0, 0.75, 0.0), float3(1.0, 1.0, 0.9) };
        let p = x * 5.0;
        let i = min(uint(p), 4u);
        return lerp(iron[i], iron[i + 1], p - float(i));
    }
    if (palette == 3) {
        static const float3 rainbow[5] = { float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 1.0),
                                           float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0),
                                           float3(1.0, 0.0, 0.0) };
        let p = x * 4.0;
        let i = min(uint(p), 3u);
        return lerp(rainbow[i], rainbow[i + 1], p - float(i));
    }
    return float3(x);
}

float decodeSrgb(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

struct ThermalDisplay : ICustomIntegrator {
    struct Params {
        float automatic;  // 1: the last frame's range
        float low;        // K
        float high;       // K
        float span;       // K, the automatic range's least
        float palette;    // 0 white-hot, 1 black-hot, 2 ironbow, 3 rainbow
    }

    static void accumulate(inout SensorPixel value, uint band, float radiance, PixelContext pixel,
                           Params params) {}

    static void finish(inout SensorPixel value, PixelContext pixel, Params params) {
        // The gain control ranges over what prims fill, not the empty background.
        let kelvin = value.kelvin;
        if (kelvin > 0.0 && !value.background) {
            pixel.agcRecord(kelvin);
        }
        var range = float2(params.low, params.high);
        if (params.automatic > 0.5) {
            let last = pixel.agcLast();
            if (last.y >= last.x && last.x > 0.0) {
                let middle = 0.5 * (last.x + last.y);
                let half = max(0.5 * (last.y - last.x), 0.5 * params.span);
                range = float2(middle - half, middle + half);
            }
        }
        let t = (kelvin - range.x) / max(range.y - range.x, 1e-3);
        let shown = paletteAt(uint(params.palette + 0.5), t);
        let scale = max(pixel.uniforms().lens.z, 1e-30);
        value.rgb = float3(decodeSrgb(shown.r), decodeSrgb(shown.g), decodeSrgb(shown.b)) / scale;
    }
}
