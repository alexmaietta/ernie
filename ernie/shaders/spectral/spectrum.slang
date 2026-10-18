// SPDX-License-Identifier: BSD-3-Clause
/// @file spectrum.slang
/// @brief Light and reflectance as spectra, so materials, lights and media are written once for
/// every spectral mode. Bands are fixed per kernel, so a 3-band kernel pays nothing for the
/// generality; RGB mode is 3 bands holding linear sRGB. Sampled wavelengths, spectral uplift and
/// CIE conversions come with light transport, behind these same functions.
///
///     Spectrum albedo = Uplift(colour);
///     float3 shown = ToLinearSrgb(albedo.Scaled(lit));

/// A value per band, each the band's average.
public struct Banded<let N : int> {
    public float values[N];

    public __init(float value) {
        for (int i = 0; i < N; ++i) {
            values[i] = value;
        }
    }

    public Banded<N> Scaled(float factor) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] *= factor;
        }
        return result;
    }

    public Banded<N> Times(Banded<N> other) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] *= other.values[i];
        }
        return result;
    }

    public Banded<N> Plus(Banded<N> other) {
        Banded<N> result = this;
        for (int i = 0; i < N; ++i) {
            result.values[i] += other.values[i];
        }
        return result;
    }
}

/// The bands this kernel carries light in: RGB mode's three, as Minimal uses by default.
public static const int kBands = 3;
/// RGB mode: three bands of linear sRGB.
public typealias Rgb = Banded<kBands>;
public typealias Spectrum = Rgb;

/// @return The spectrum of an authored linear sRGB colour; in RGB mode, the colour itself.
public Spectrum Uplift(float3 colour) {
    Spectrum spectrum;
    spectrum.values = { colour.r, colour.g, colour.b };
    return spectrum;
}

/// @return A spectrum seen as linear sRGB; in RGB mode, its bands.
public float3 ToLinearSrgb(Spectrum spectrum) {
    return float3(spectrum.values[0], spectrum.values[1], spectrum.values[2]);
}
