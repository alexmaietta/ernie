// SPDX-License-Identifier: BSD-3-Clause
/// @file spectrum.slang
/// @brief Light and reflectance as spectra, so materials, lights and media are written once for
/// every spectral mode: against `ISpectrum`, with each kernel's `Spectrum` fixing its mode and
/// band count at compile time, so a 3-band kernel pays nothing for the generality. RGB mode is 3
/// bands holding linear sRGB. Sampled wavelengths, spectral uplift and CIE conversions come with
/// light transport, as further `ISpectrum` forms behind these same functions.
///
///     Spectrum albedo = Uplift(colour);
///     float3 shown = ToLinearSrgb(albedo.Scaled(lit));
///
/// The band count is shared with C++ (rt/camera_pass.h).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#endif

/// The bands this kernel carries light in: RGB mode's three, as Minimal uses by default.
static const uint kBands = 3;

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

/// What every spectrum offers the code that carries light.
interface ISpectrum {
    /// The same value in every band or at every wavelength.
    __init(float value);
    This Scaled(float factor);
    This Times(This other);
    This Plus(This other);
}

/// A value per band, each the band's average.
struct Banded<let N : int> : ISpectrum {
    float values[N];

    __init(float value) {
        for (int i = 0; i < N; ++i) {
            values[i] = value;
        }
    }

    Banded<N> Scaled(float factor) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] *= factor;
        }
        return result;
    }

    Banded<N> Times(Banded<N> other) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] *= other.values[i];
        }
        return result;
    }

    Banded<N> Plus(Banded<N> other) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] += other.values[i];
        }
        return result;
    }
}

/// RGB mode: three bands of linear sRGB.
typealias Rgb = Banded<kBands>;
/// The spectrum this kernel carries light in.
typealias Spectrum = Rgb;

/// @return The spectrum of an authored linear sRGB colour; in RGB mode, the colour itself.
Spectrum Uplift(float3 colour) {
    Spectrum spectrum;
    spectrum.values = { colour.r, colour.g, colour.b };
    return spectrum;
}

/// @return A spectrum seen as linear sRGB; in RGB mode, its bands.
float3 ToLinearSrgb(Spectrum spectrum) {
    return float3(spectrum.values[0], spectrum.values[1], spectrum.values[2]);
}

#endif
