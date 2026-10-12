// SPDX-License-Identifier: BSD-3-Clause
// An uncooled microbolometer, as thermal cameras' focal planes. Its signal is the in-band
// radiance S = sum over bands of R(lambda) L dlambda (a flat response from `cutOn` to `cutOff`
// nm, edges 250 nm soft), not photons, read as `countsPerRadiance` counts per W m^-2 sr^-1 into
// the electrons domain, which `adc` then digitises (its electrons per DN 1 for counts as they
// are). Each pixel follows the scene with its thermal time constant `timeConstant` over the time
// between frames (temporal: moving hot spots smear; at once on a first frame or a still). It
// reads out with noise of `netd` kelvin rms and, after non-uniformity correction, a fixed
// pattern of `fpn` kelvin rms and an offset `drift` kelvin as its housing warms, each through the
// response's slope at 300 K. The camera's radiometry inverts the signal to a brightness
// temperature, for the BrightnessTemperature var and `thermalDisplay`, for an object of
// emissivity `emissivity` reflecting surroundings at `reflected` kelvin, as thermal cameras'
// object parameters do: S = e S_bb(T) + (1 - e) S_bb(T_reflected), S_bb a black body's. An
// integrator into electrons, in place of `qe`.
//
// References:
// - M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging", 2nd ed., Wiley-VCH, 2018
//   (microbolometers, NETD, the radiometric chain and its object parameters).
// - P. W. Kruse, "Uncooled Thermal Imaging: Arrays, Systems, and Applications", SPIE Press, 2001
//   (the bolometer's thermal time constant).
import sensor;

// A black body's spectral radiance at `wavelength` (nm) and `kelvin`, W m^-2 sr^-1 nm^-1.
float planckRadiance(float wavelength, float kelvin) {
    let r = 1.0 / wavelength;
    return 1.1910430e20 * r * r * r * r * r / (exp(1.4387769e7 * r / max(kelvin, 1.0)) - 1.0);
}

struct Bolometer : ICustomIntegrator {
    struct Params {
        float cutOn;              // nm
        float cutOff;             // nm
        float countsPerRadiance;  // counts per W m^-2 sr^-1 in band
        float netd;               // K rms, temporal
        float fpn;                // K rms, fixed after non-uniformity correction
        float drift;              // K, an offset common to every pixel
        float timeConstant;       // s
        float emissivity;         // the object's, for brightness temperature
        float reflected;          // K, what it reflects
    }

    // The response at `wavelength`: 1 between the cut-on and cut-off, falling over 250 nm.
    static float response(float wavelength, Params params) {
        return smoothstep(params.cutOn - 125.0, params.cutOn + 125.0, wavelength) *
               (1.0 - smoothstep(params.cutOff - 125.0, params.cutOff + 125.0, wavelength));
    }

    // A black body's in-band radiance at `kelvin`, W m^-2 sr^-1.
    static float blackBody(float kelvin, PixelContext pixel, Params params) {
        var sum = 0.0;
        for (uint b = 0; b < pixel.bands(); ++b) {
            let lambda = pixel.wavelength(b);
            sum += response(lambda, params) * planckRadiance(lambda, kelvin);
        }
        return sum * pixel.bandWidth();
    }

    static void accumulate(inout SensorPixel value, uint band, float radiance, PixelContext pixel,
                           Params params) {
        // The in-band radiance gathers in electrons.x until `finish`.
        value.electrons.x +=
            response(pixel.wavelength(band), params) * radiance * pixel.bandWidth();
    }

    static void finish(inout SensorPixel value, PixelContext pixel, Params params) {
        // The bolometer's own lag behind the scene: first order over the time between frames.
        let scene = value.electrons.x;
        let last = readState(pixel, 0);
        let interval = pixel.frameInterval();
        let alpha = params.timeConstant > 0.0 && interval > 0.0 && last.y > 0.5
                        ? 1.0 - exp(-interval / params.timeConstant)
                        : 1.0;
        let followed = last.x + alpha * (scene - last.x);
        writeState(pixel, 0, float4(followed, 1.0, 0.0, 0.0));
        // Readout noise and the corrected offsets, in kelvin through the slope at 300 K.
        let slope = blackBody(300.5, pixel, params) - blackBody(299.5, pixel, params);
        let temporal =
            sqrt(-2.0 * log(pixel.random(0x5b1))) * cos(6.28318530718 * pixel.random(0x5b2));
        let fixed = sqrt(-2.0 * log(pixel.fixedRandom(0x5b3))) *
                    cos(6.28318530718 * pixel.fixedRandom(0x5b4));
        let signal =
            followed + slope * (params.netd * temporal + params.fpn * fixed + params.drift);
        value.electrons = float4(max(signal, 0.0) * params.countsPerRadiance, 0.0, 0.0, 0.0);
        // Brightness temperature: the black body's signal the object's emission makes, inverted
        // by Newton's method from a guess at 300 K.
        let e = clamp(params.emissivity, 0.01, 1.0);
        let target = (signal - (1.0 - e) * blackBody(params.reflected, pixel, params)) / e;
        var kelvin = 300.0;
        for (uint i = 0; i < 8; ++i) {
            let s = blackBody(kelvin, pixel, params);
            let ds = (blackBody(kelvin + 0.5, pixel, params) - s) / 0.5;
            kelvin = clamp(kelvin - (s - target) / max(ds, 1e-12), 20.0, 5000.0);
        }
        value.kelvin = kelvin;
    }
}
