// SPDX-License-Identifier: BSD-3-Clause
// Radiation events: energetic particles (cosmic rays, trapped or solar protons and electrons)
// crossing the sensor during the exposure, each leaving a track of charge. Hits arrive at random,
// `flux` per square centimetre per second, new every frame; each crosses the sensitive layer
// (`thickness` micrometres) at an angle theta off its normal, cos(theta) distributed as sqrt(u) for
// an isotropic flux through a plane, so its track runs `thickness` tan(theta) across the focal
// plane in a random direction (at most `maxLength` pixels) and its path `thickness` / cos(theta)
// deposits `electronsPerMicron` per micrometre (a minimum-ionising particle's most probable loss in
// silicon, about 80 at 3.6 eV per pair), times a log-normal straggling of `straggling` (the Landau
// fluctuations of the loss). The charge spreads across the track as a Gaussian of `spread` pixels
// (diffusion before collection), integrated over each pixel. Added to every channel, after `qe`
// and before `noise`. AOV `deposited`: the electrons the hits left, a ground-truth mask.
//
// Hits are generated per square cell of the image, wider than a track reaches, from a hash of the
// cell and the frame seed, so each pixel finds those that reach it in its own cell and the eight
// around it.
//
// References:
// - J. R. Janesick, "Scientific Charge-Coupled Devices", SPIE Press, 2001 (cosmic ray and proton
//   events in CCDs: their rates, tracks and charge).
// - Particle Data Group, "Review of Particle Physics", section "Passage of particles through
//   matter" (most probable energy loss and Landau straggling in silicon).
// - STAR-Dundee, "PANGU v7 Datasheet", 2024 (radiation events in its camera model).
// - M. Abramowitz and I. A. Stegun, "Handbook of Mathematical Functions", 1964, 7.1.26 (erf).
import sensor;

uint3 radiationHash(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Uniform in (0, 1].
float radiationUniform(uint bits) {
    return (float(bits >> 8) + 1.0) / 16777216.0;
}

float radiationErf(float x) {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
                   0.254829592) * t * exp(-x * x);
    return x < 0.0 ? -y : y;
}

struct Radiation : IPixelFilter {
    struct Params {
        float flux;                // particles per square centimetre per second
        float thickness;           // the sensitive layer, micrometres
        float electronsPerMicron;  // charge deposited per micrometre of path
        float straggling;          // log-normal sigma of each hit's charge
        float spread;              // Gaussian sigma of the charge across the track, pixels
        float maxLength;           // the longest track across the focal plane, pixels
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let sigma = sqrt(params.spread * params.spread + 1.0 / 12.0);
        let longest = max(params.maxLength, 0.0);
        let cell = max(ceil(0.5 * longest + 3.0 * sigma + 1.0), 8.0);
        let pitchCm = pixel.pixelPitchMicrons() * 1e-4;
        let mean = max(params.flux, 0.0) * (cell * pitchCm) * (cell * pitchCm) *
                   pixel.exposureTime();
        let frameSeed = pixel.uniforms().noise.z;
        let at = float2(pixel.id.xy) + 0.5;
        let home = int2(floor(at / cell));
        var deposited = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                let c = home + int2(dx, dy);
                var state = radiationHash(uint3(uint(c.x) * 7919u, uint(c.y) * 104729u,
                                                frameSeed ^ 0x52414431u));
                // Poisson hits in the cell: Knuth's product of uniforms, a normal above 30.
                var hits = 0;
                if (mean < 30.0) {
                    let limit = exp(-mean);
                    var product = radiationUniform(state.x);
                    while (product > limit && hits < 64) {
                        state = radiationHash(state);
                        product *= radiationUniform(state.x);
                        hits += 1;
                    }
                } else {
                    state = radiationHash(state);
                    let n = sqrt(-2.0 * log(radiationUniform(state.x))) *
                            cos(6.28318530718 * radiationUniform(state.y));
                    hits = int(clamp(floor(mean + sqrt(mean) * n + 0.5), 0.0, 64.0));
                }
                for (int k = 0; k < hits; ++k) {
                    let a = radiationHash(state + uint3(uint(k) * 2654435761u, 17u, 31u));
                    let b = radiationHash(a);
                    let centre = (float2(c) + float2(radiationUniform(a.x),
                                                     radiationUniform(a.y))) * cell;
                    let cosTheta = sqrt(radiationUniform(a.z));
                    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
                    let phi = 6.28318530718 * radiationUniform(b.x);
                    var across = params.thickness * sinTheta / max(cosTheta, 1e-6) /
                                 max(pixel.pixelPitchMicrons(), 1e-6);
                    var path = params.thickness / max(cosTheta, 1e-6);
                    if (across > longest) {
                        // A grazing track, cut to the longest: its path through the layer too.
                        let kept = longest * pixel.pixelPitchMicrons();
                        path = sqrt(params.thickness * params.thickness + kept * kept);
                        across = longest;
                    }
                    let n = sqrt(-2.0 * log(radiationUniform(b.y))) *
                            cos(6.28318530718 * radiationUniform(b.z));
                    let s = max(params.straggling, 0.0);
                    let charge = params.electronsPerMicron * path * exp(s * n - 0.5 * s * s);
                    let dir = float2(cos(phi), sin(phi));
                    let start = centre - 0.5 * across * dir;
                    let t = dot(at - start, dir);
                    let off = at - start - t * dir;
                    let perp = exp(-dot(off, off) / (2.0 * sigma * sigma)) /
                               (2.50662827463 * sigma);
                    if (across < 1e-4) {
                        deposited += charge * perp * exp(-t * t / (2.0 * sigma * sigma)) /
                                     (2.50662827463 * sigma);
                    } else {
                        let along = 0.5 * (radiationErf((across - t) / (1.41421356 * sigma)) -
                                           radiationErf(-t / (1.41421356 * sigma)));
                        deposited += charge / across * perp * along;
                    }
                }
            }
        }
        for (uint ch = 0; ch < pixel.channels(); ++ch) {
            value.electrons[ch] += deposited;
        }
        writeOutput(pixel, 0, float4(deposited));
    }
}
