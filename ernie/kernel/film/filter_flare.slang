// SPDX-License-Identifier: BSD-3-Clause
// Lens flare ghosts: light reflected twice between lens surfaces (off one surface, back off an
// earlier one, then on to the sensor) reaches the sensor out of focus, imaging every bright
// source as a ghost on the line through the optical centre, mirrored through it or not, at a
// scale and with a blur of its own fixed by the pair of surfaces and the lens's prescription. A
// ghost of a point source is a defocused image of the aperture stop: a disc, or the polygon of an
// iris of `blades` blades. Each pair carries about the product of the two surfaces'
// reflectances, wavelength dependent through their anti-reflection coatings (a quarter-wave layer
// reflects least at its design wavelength), so ghosts are tinted, often green or magenta.
//
// Ghost i of each pixel's light lands at c + s_i (p - c), c the principal point and s_i its
// scale (negative mirrors it through the centre), spread over a disc of radius r_i (a fraction of
// the image's half diagonal), with a share `strength` w_i t_ic of channel c's light, the source
// being the light above `threshold`. As a gather, a pixel q gathers each ghost from the source at
// c + (q - u - c) / s_i for taps u over its aperture shape, over s_i^2 for the magnification, from
// the image's pyramid at the level of the taps' spacing, so a few taps make a filled shape. The
// light the ghosts take leaves the direct image, so light is kept but for ghosts falling off the
// sensor. This is the screen-space approximation: ghost scales and sizes are parameters, not
// traced through the lens (Hullin et al. trace them per surface pair). On electrons, after `qe`
// (before `noise`, as the ghosts reach the sensor during the exposure), reading its pyramid; the
// tints apply to an RGB sensor's channels (the NIR channel takes red's, a mono sensor's channel
// their mean). At zero strength the image is untouched.
//
// References:
// - M. Hullin, E. Eisemann, H.-P. Seidel and S. Lee, "Physically-Based Real-Time Lens Flare
//   Rendering", ACM Transactions on Graphics 30(4) (SIGGRAPH 2011), 108.
//   https://doi.org/10.1145/2010324.1965003
// - J. Chapman, "Pseudo Lens Flare", blog post, 2013 (screen-space ghosts: the image mirrored
//   through the centre at several scales, from a downsampled copy).
// - W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill, 2008 (ghost images; anti-
//   reflection coatings).
// - H. Vogel, "A better way to construct the sunflower head", Mathematical Biosciences 44, 1979
//   (the taps' spiral).
import sensor;

struct Flare : ISpatialPixelFilter {
    struct Params {
        float strength;      // the share of the light each ghost takes, times its weight and tint
        float ghosts;        // ghosts drawn, 0 to 8
        float threshold;     // electrons above which a pixel makes ghosts
        float blades;        // the iris's blades: the ghosts' polygon (below 3 a disc)
        float rotation;      // the polygon's rotation, degrees
        float scales[8];     // each ghost's scale about the principal point, negative mirrored
        float radii[8];      // each ghost's blur radius, a fraction of the image's half diagonal
        float weights[8];    // each ghost's relative share
        float tintRed[8];    // each ghost's tint in red
        float tintGreen[8];  // in green
        float tintBlue[8];   // in blue
    }

    static const uint kTaps = 32;

    // The light above the threshold: what makes ghosts.
    static float4 sourceOf(float4 value, float threshold) {
        return max(value - threshold, float4(0.0));
    }

    // Ghost `i`'s tint per channel of `pixel`'s sensor.
    static float4 tintOf(PixelContext pixel, Params params, uint i) {
        let rgb = float3(params.tintRed[i], params.tintGreen[i], params.tintBlue[i]);
        if (pixel.channels() == 1) {
            return float4((rgb.x + rgb.y + rgb.z) / 3.0, 0.0, 0.0, 0.0);
        }
        return float4(rgb, rgb.x);
    }

    // Tap `k` of `kTaps` over the unit aperture: a Vogel spiral filling the disc evenly, pushed
    // out onto the polygon of `blades` blades (inscribed in the unit circle) when there are any.
    static float2 apertureTap(uint k, float blades, float rotation) {
        let golden = 2.39996323;
        let r = sqrt((float(k) + 0.5) / float(kTaps));
        let phi = float(k) * golden;
        var radius = r;
        if (blades >= 3.0) {
            let n = floor(blades);
            let sector = 6.28318531 / n;
            let local = phi - rotation - sector * floor((phi - rotation) / sector) - 0.5 * sector;
            radius *= cos(0.5 * sector) / cos(local);
        }
        return radius * float2(cos(phi), sin(phi));
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let own = input.values(pixel, int2(pixel.id.xy));
        let count = min(uint(max(params.ghosts, 0.0)), 8u);
        let size = float2(input.size(pixel));
        let centre = pixel.uniforms().intrinsics.yz;
        let q = float2(pixel.id.xy) + 0.5;
        let halfDiagonal = 0.5 * length(size);
        let rotation = radians(params.rotation);
        let top = float(input.levels() - 1);
        var taken = float4(0.0);
        var ghosts = float4(0.0);
        for (uint i = 0; i < count; ++i) {
            let share = params.strength * max(params.weights[i], 0.0) * tintOf(pixel, params, i);
            taken += share;
            let s = params.scales[i];
            let scale = sign(s) * max(abs(s), 0.05);
            let radius = max(params.radii[i], 0.0) * halfDiagonal;
            // Each tap stands for an equal share of the shape's area: the source read over the
            // span of that share, through the ghost's magnification.
            let spacing = radius * sqrt(3.14159265 / float(kTaps)) / abs(scale);
            let level = clamp(log2(max(spacing, 1.0)), 0.0, top);
            var sum = float4(0.0);
            for (uint k = 0; k < kTaps; ++k) {
                let u = radius * apertureTap(k, params.blades, rotation);
                let p = centre + (q - u - centre) / scale;
                if (any(p < float2(0.0)) || any(p > size)) {
                    continue;  // no light from beyond the sensor
                }
                sum += sourceOf(input.valuesAt(pixel, p / size, level), params.threshold);
            }
            ghosts += share * sum / (float(kTaps) * scale * scale);
        }
        value.electrons = own - taken * sourceOf(own, params.threshold) + ghosts;
    }
}
