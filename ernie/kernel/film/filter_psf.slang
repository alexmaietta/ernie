// SPDX-License-Identifier: BSD-3-Clause
// Point-spread function: each band blurred by the lens's diffraction, redder bands spreading
// wider, combined with a Gaussian of `blur` pixels for aberrations and the sensor's optical
// crosstalk. The weights are normalised, so the filter moves light and keeps it all: a point
// source's total stays its irradiance, spread into a disc that grows with the f-number. Bright
// stars and highlights grow into discs as through a real lens.
//
// Two profiles. `profile` 0: the Gaussian fit to the Airy pattern, sigma = 0.42 lambda N in the
// image plane, cheap and smooth. `profile` 1: the Airy pattern itself, the Fraunhofer
// diffraction of a circular aperture of f-number N with a central obstruction (a secondary
// mirror) of `obstruction` times its diameter, eps:
//   I(r) = [2 J1(x) / x - eps^2 2 J1(eps x) / (eps x)]^2 / (1 - eps^2)^2,  x = pi r / (lambda N),
// with its dark rings (the first at 1.22 lambda N without obstruction, holding 84% of the light)
// and wings falling as r^-3. Each pixel integrates the pattern over its area, `samples` per axis
// (fewer where the rings span pixels), convolved with the `blur` Gaussian; where the rings are
// finer than a pixel and beyond the core, a pixel takes their average, 4 / (pi x^3) for the
// clear aperture. The kernel reaches `maxRadius` pixels; the wings beyond it fold back into it
// (into the centre pixel when the core is finer than the samples). An obstruction narrows the
// core and moves light into the rings. Dark neighbourhoods (a star field's sky) skip the kernel.
//
// References:
// - G. B. Airy, "On the Diffraction of an Object-glass with Circular Aperture", Transactions of
//   the Cambridge Philosophical Society 5:283-291, 1835.
// - M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University Press, 1999,
//   section 8.5.2 (the Airy pattern, its rings and encircled energy).
// - J. W. Goodman, "Introduction to Fourier Optics", 3rd ed., Roberts & Company, 2005, section
//   4.4 (Fraunhofer patterns of circular and annular apertures).
// - V. N. Mahajan, "Uniform versus Gaussian beams: a comparison of the effects of diffraction,
//   obscuration, and aberrations", JOSA A 3(4):470-485, 1986 (the obscured aperture).
// - B. Zhang, J. Zerubia and J.-C. Olivo-Marin, "Gaussian approximations of fluorescence
//   microscope point-spread function models", Applied Optics 46(10):1819-1829, 2007 (the
//   Gaussian fit to the Airy disc, sigma = 0.42 lambda N).
// - W. H. Press et al., "Numerical Recipes in C", 2nd ed., 1992, section 6.5 (bessj1's rational
//   approximations of J1).
// - M. Abramowitz and I. A. Stegun, "Handbook of Mathematical Functions", 1964, 7.1.26 (erf).
import sensor;

struct Psf : ISpatialBandFilter {
    struct Params {
        float blur;         // Gaussian sigma in pixels added to diffraction's.
        float maxRadius;    // the widest kernel in pixels, bounding the cost
        float profile;      // 0 the Gaussian fit to the Airy disc, 1 the Airy pattern
        float obstruction;  // the central obstruction's diameter over the aperture's, 0 to 0.95
        float samples;      // sub-pixel samples per axis integrating the Airy pattern, 1 to 16
    }

    // 2 J1(x) / x, the clear aperture's amplitude, 1 at the centre.
    static float airyAmplitude(float x) {
        let ax = abs(x);
        if (ax < 8.0) {
            let y = x * x;
            let p =
                72362614232.0 +
                y * (-7895059235.0 +
                     y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))));
            let q =
                144725228442.0 +
                y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
            return 2.0 * p / q;
        }
        let z = 8.0 / ax;
        let y = z * z;
        let phase = ax - 2.356194491;
        let p = 1.0 + y * (0.183105e-2 +
                           y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
        let q =
            0.04687499995 + y * (-0.2002690873e-3 +
                                 y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        return 2.0 * sqrt(0.636619772 / ax) * (cos(phase) * p - z * sin(phase) * q) / ax;
    }

    // The obstructed aperture's intensity at x, 1 at the centre.
    static float airy(float x, float eps) {
        let a = airyAmplitude(x) - eps * eps * airyAmplitude(eps * x);
        return a * a / ((1.0 - eps * eps) * (1.0 - eps * eps));
    }

    // Its average over rings finer than a pixel: the clear aperture's 4 / (pi x^3), plus the
    // obstruction's own pattern; their cross term averages out.
    static float airyEnvelope(float x, float eps) {
        let ex = eps * x;
        let own =
            ex > 12.0 ? 4.0 / (3.14159265 * ex * ex * ex) : airyAmplitude(ex) * airyAmplitude(ex);
        let clear = 4.0 / (3.14159265 * x * x * x);
        return (clear + eps * eps * eps * eps * own) / ((1.0 - eps * eps) * (1.0 - eps * eps));
    }

    static float erf(float x) {
        let t = 1.0 / (1.0 + 0.3275911 * abs(x));
        let y = 1.0 -
                t *
                    (0.254829592 + t * (-0.284496736 +
                                        t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
                    exp(-x * x);
        return x < 0.0 ? -y : y;
    }

    // The pixel's box, 1 wide, convolved with the blur's Gaussian, at offset u along an axis.
    static float blurredBox(float u, float sigma) {
        if (sigma < 1e-3) {
            return abs(u) <= 0.5 ? 1.0 : 0.0;
        }
        let k = 0.70710678 / sigma;
        return 0.5 * (erf((u + 0.5) * k) - erf((u - 0.5) * k));
    }

    // The light a point at a pixel's centre lands in the pixel `d` pixels away over the pattern's
    // peak intensity: the pattern integrated over the pixel convolved with the blur, on an m x m
    // grid `h` apart, leaving out the core within `core` of the point; or the rings' average when
    // they are finer than a pixel and far from the core (`envelope`).
    static float airyWeight(int2 d, float scale, float eps, float sigma, uint m, float h,
                            float core, bool envelope) {
        if (envelope) {
            return airyEnvelope(length(float2(d)) * scale, eps);
        }
        let extent = 0.5 * h * float(m);
        var w = 0.0;
        for (uint j = 0; j < m; ++j) {
            let v = -extent + (float(j) + 0.5) * h;
            let bv = blurredBox(v, sigma);
            for (uint i = 0; i < m; ++i) {
                let u = -extent + (float(i) + 0.5) * h;
                let r = length(float2(d) + float2(u, v));
                if (r >= core) {
                    w += airy(r * scale, eps) * blurredBox(u, sigma) * bv;
                }
            }
        }
        return w * h * h;
    }

    static float band(uint band, PixelContext pixel, SpatialImage input, Params params) {
        let wavelengthUm = pixel.wavelength(band) * 1e-3;
        let pitch = max(pixel.pixelPitchMicrons(), 1e-3);
        let centre = int2(pixel.id.xy);
        // The Airy pattern's scale, lambda N, in pixels: its rings' spacing.
        let lambdaN = wavelengthUm * pixel.diffractionFNumber() / pitch;
        if (params.profile >= 0.5 && lambdaN > 1e-3) {
            let radius = int(max(params.maxRadius, 1.0));
            // Dark (a star field's sky): nothing to spread, and the kernel's weights skipped.
            var any = false;
            for (int y = -radius; y <= radius && !any; ++y) {
                for (int x = -radius; x <= radius; ++x) {
                    any = any || input.band(pixel, centre + int2(x, y), band) != 0.0;
                }
            }
            if (!any) {
                return 0.0;
            }
            let eps = clamp(params.obstruction, 0.0, 0.95);
            let sigma = max(params.blur, 0.0);
            let scale = 3.14159265 / lambdaN;  // x per pixel
            // The pattern's intensity per pixel area over its peak's: it integrates to 1.
            let norm = 3.14159265 * (1.0 - eps * eps) / (4.0 * lambdaN * lambdaN);
            // Six samples across each ring (lambda N apart), at most `samples` per pixel and twice
            // that over a pixel and its blur.
            let most = uint(clamp(params.samples, 1.0, 16.0));
            let n = clamp(uint(ceil(6.0 / lambdaN)), 1, most);
            let extent = 0.5 + 3.0 * sigma;
            let m = min(uint(ceil(2.0 * extent * float(n))), 2 * most);
            let h = 2.0 * extent / float(m);
            // A core the samples resolve keeps its shape, and the wings beyond the kernel spread
            // over it in proportion. A core finer than the samples, within a sample spacing of
            // the point, is the light the samples leave, spread by the pixel and the blur alone.
            let resolved = lambdaN >= 1.5 * h;
            let core = resolved ? 0.0 : h;
            // Rings (and their beat with the obstruction's) finer than a pixel average out beyond
            // the fourth ring.
            let fine = lambdaN < 1.0 - eps;
            var sum = 0.0;
            var weights = 0.0;
            var coreSum = 0.0;
            var coreWeights = 0.0;
            for (int y = -radius; y <= radius; ++y) {
                for (int x = -radius; x <= radius; ++x) {
                    let d = int2(x, y);
                    let value = input.band(pixel, centre + d, band);
                    let near = (length(float2(d)) - 0.75 - 3.0 * sigma) * scale * (1.0 - eps);
                    let w =
                        norm * airyWeight(d, scale, eps, sigma, m, h, core, fine && near > 12.0);
                    sum += w * value;
                    weights += w;
                    if (!resolved) {
                        let g = blurredBox(float(x), sigma) * blurredBox(float(y), sigma);
                        coreSum += g * value;
                        coreWeights += g;
                    }
                }
            }
            if (resolved) {
                return sum / weights;
            }
            let rest = max(1.0 - weights, 0.0);
            return (sum + rest * coreSum / coreWeights) / (weights + rest);
        }
        let diffraction = 0.42 * lambdaN;
        let sigma = sqrt(diffraction * diffraction + params.blur * params.blur);
        if (sigma < 0.2) {
            return input.band(pixel, centre, band);
        }
        let radius = int(min(ceil(3.0 * sigma), max(params.maxRadius, 1.0)));
        let k = -0.5 / (sigma * sigma);
        var sum = 0.0;
        var weights = 0.0;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                let w = exp(k * float(x * x + y * y));
                sum += w * input.band(pixel, centre + int2(x, y), band);
                weights += w;
            }
        }
        return sum / weights;
    }
}
