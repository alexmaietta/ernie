// SPDX-License-Identifier: BSD-3-Clause
/// @file spectrum.h
/// @brief Spectra, spectral bands, black bodies and colour matching.
/// Wavelengths are in nm, irradiance in W m^-2 nm^-1 and radiance in
/// W m^-2 sr^-1 nm^-1.
#pragma once

#include "ernie/physics/constants.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// @brief A camera's spectral sampling: a range split into equal
/// bands.
/// Band i covers [Lower(i), Lower(i) + Width()).
struct SpectralBands {
    /// Lower end of the range, in nm.
    double minNm = 380.0;
    /// Upper end of the range, in nm.
    double maxNm = 780.0;
    /// Number of bands.
    int count = 16;
    /// RGB mode: three bands carry linear sRGB blue, green and red,
    /// so colours match a 3-channel renderer's instead of shifting.
    bool rgb = false;

    /// @return RGB mode's bands: 400-700 nm in three.
    static SpectralBands Rgb() {
        return {400.0, 700.0, 3, true};
    }

    /// @return A band's width, in nm.
    double Width() const {
        return (maxNm - minNm) / count;
    }

    /// @return Band `band`'s lower edge, in nm.
    double Lower(int band) const {
        return minNm + band * Width();
    }

    /// @return Band `band`'s centre, in nm.
    double Centre(int band) const {
        return Lower(band) + 0.5 * Width();
    }

    /// @brief Validates the bands.
    /// @throws std::invalid_argument Unless 0 < minNm < maxNm,
    ///         1 <= count <= kMaxBands, and count is 3 in RGB mode.
    void Check() const;

    /// The most bands a camera may have.
    static constexpr int kMaxBands = 64;

    bool operator==(const SpectralBands&) const = default;
};

/// @brief A spectrum's value in one band.
/// @param bands    The bands.
/// @param band     The band index.
/// @param spectrum The spectrum, by wavelength in nm.
/// @return Its band average; in RGB mode, the band's linear sRGB
///         channel relative to equal-energy white, at least 0.
double BandValue(const SpectralBands& bands, int band,
                 const std::function<double(double)>& spectrum);

/// @brief A tabulated spectrum, linear between samples.
/// Constant beyond the ends; an empty spectrum is zero.
class Spectrum {
  public:
    Spectrum() = default;

    /// @brief Tabulates a spectrum.
    /// @param wavelengthsNm Strictly increasing wavelengths, in nm.
    /// @param values        The value at each.
    /// @throws std::invalid_argument If the wavelengths are not
    ///         strictly increasing or the sizes differ.
    Spectrum(std::vector<double> wavelengthsNm, std::vector<double> values);

    /// @param value The value.
    /// @return A spectrum equal to `value` everywhere.
    static Spectrum Constant(double value);

    /// @brief Builds from interleaved (wavelength, value) pairs, the
    /// form of `ernie:` spectrum attributes.
    /// @param pairs The pairs.
    /// @return The spectrum.
    /// @throws std::invalid_argument For an odd count, or as the
    ///         constructor.
    static Spectrum FromPairs(std::span<const float> pairs);

    /// @param wavelengthNm The wavelength, in nm.
    /// @return The interpolated value.
    double operator()(double wavelengthNm) const;

    /// @brief Exact mean of the interpolant over a range.
    /// @param lowNm  The range's start, in nm.
    /// @param highNm The range's end, in nm.
    /// @return The mean.
    double Average(double lowNm, double highNm) const;

    /// @param bands The bands.
    /// @return The average over each band.
    std::vector<float> Bin(const SpectralBands& bands) const;

    /// @return Whether it has no samples.
    bool Empty() const {
        return wavelengths_.empty();
    }

    /// @return The sample wavelengths, in nm.
    const std::vector<double>& Wavelengths() const {
        return wavelengths_;
    }

    /// @return The sample values.
    const std::vector<double>& Values() const {
        return values_;
    }

  private:
    double Integral(double lowNm, double highNm) const;

    std::vector<double> wavelengths_;
    std::vector<double> values_;
};

/// @brief Planck's law.
/// @param wavelengthNm The wavelength, in nm.
/// @param kelvin       The temperature, in kelvin.
/// @return Black-body spectral radiance, in W m^-2 sr^-1 nm^-1.
double PlanckRadiance(double wavelengthNm, double kelvin);

/// @brief CIE standard illuminant D (daylight).
/// Interpolated from the CIE's 10 nm basis over 300-830 nm; 0
/// outside.
/// @param wavelengthNm The wavelength, in nm.
/// @param kelvin       Correlated colour temperature, 4000 to 25000 K
///                     (D65 is 6504 K).
/// @return Relative spectral power, 100 at 560 nm.
/// @throws std::invalid_argument For a temperature out of range.
///
/// References:
/// - CIE 015:2018, "Colorimetry", 4th edition, section 4.1.2 and
///   table 6 (the S0, S1 and S2 components of daylight).
double CieDaylight(double wavelengthNm, double kelvin);

/// @brief A black body's irradiance averaged over each band.
/// @param bands           The bands.
/// @param kelvin          The temperature, in kelvin.
/// @param totalIrradiance Its integral over all wavelengths, in
///                        W/m^2.
/// @return Spectral irradiance per band, in W m^-2 nm^-1.
std::vector<float> BlackbodyIrradiance(const SpectralBands& bands, double kelvin,
                                       double totalIrradiance);

/// @brief Thermal emission of a surface, by Kirchhoff's law.
/// @param bands      The bands.
/// @param kelvin     The temperature; at or below 0 K gives zeros.
/// @param emissivity Emissivity by wavelength in nm.
/// @return Emissivity times Planck radiance, averaged per band, in
///         W m^-2 sr^-1 nm^-1.
///
/// References:
/// - M. Planck, "The Theory of Heat Radiation", 1914 (Kirchhoff's
///   law and the black-body spectrum).
std::vector<float> ThermalEmission(const SpectralBands& bands, double kelvin,
                                   const std::function<double(double)>& emissivity);

/// @brief CIE 1931 2-degree colour matching functions, by Wyman et
/// al.'s multi-lobe Gaussian fit.
/// @param wavelengthNm The wavelength, in nm.
/// @return x-bar, y-bar and z-bar; y-bar peaks at 1 near 555 nm.
///
/// References:
/// - C. Wyman, P.-P. Sloan and P. Shirley, "Simple Analytic
///   Approximations to the CIE XYZ Color Matching Functions", Journal
///   of Computer Graphics Techniques 2(2):1-11, 2013.
glm::dvec3 CieXyz(double wavelengthNm);

/// @brief Colour matching functions integrated over each band.
/// XYZ = sum_i value_i * weights_i integrates a spectrum.
/// @param bands The bands.
/// @return One XYZ weight per band.
std::vector<glm::dvec3> CieWeights(const SpectralBands& bands);

/// @brief The weights a renderer turns band values into XYZ with.
/// `CieWeights` for 16 bands or more. For fewer, a least-squares fit
/// over sigmoid reflectances sampled at band centres, greys exact;
/// plain averages would shift colours (green reads yellow with 3).
/// @param bands The bands.
/// @return One XYZ weight per band.
///
/// References:
/// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for
///   Efficient Spectral Upsampling", Computer Graphics Forum
///   38(2):147-155, 2019 (the reflectances fitted over).
std::vector<glm::dvec3> ColourWeights(const SpectralBands& bands);

/// @brief CIE XYZ of band values, without the luminous efficacy.
/// @param bands  The bands.
/// @param values One value per band.
/// @return The sum of value times `CieWeights`.
glm::dvec3 ToXyz(const SpectralBands& bands, std::span<const float> values);

/// @brief Converts CIE XYZ to linear sRGB (D65, IEC 61966-2-1).
/// @param xyz The colour.
/// @return Linear sRGB.
glm::dvec3 XyzToLinearSrgb(glm::dvec3 xyz);

/// @brief Converts XYZ relative to illuminant E to linear sRGB,
/// Bradford-adapted to D65 so flat spectra map to greys.
/// @param xyz The colour, white at a flat spectrum's XYZ.
/// @return Linear sRGB.
///
/// References:
/// - K. M. Lam, "Metamerism and Colour Constancy", PhD thesis,
///   University of Bradford, 1985; matrices as given by B. Lindbloom,
///   http://www.brucelindbloom.com/Eqn_ChromAdapt.html
glm::dvec3 EqualEnergyXyzToLinearSrgb(glm::dvec3 xyz);

/// @brief A smooth reflectance in Jakob and Hanika's
/// sigmoid-polynomial form.
/// s = 1/2 + x / (2 sqrt(1 + x^2)), x = c0 t^2 + c1 t + c2, with t
/// the wavelength mapped from 360-830 nm to 0-1. Always in [0, 1].
///
/// References:
/// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for
///   Efficient Spectral Upsampling", Computer Graphics Forum (Proc.
///   Eurographics) 38(2):147-155, 2019.
///   https://doi.org/10.1111/cgf.13626
struct SigmoidSpectrum {
    /// c0, c1 and c2; flat when c0 = c1 = 0.
    glm::dvec3 coefficients{0.0};

    /// @param wavelengthNm The wavelength, in nm.
    /// @return The reflectance.
    double operator()(double wavelengthNm) const;

    /// @return The spectrum every 5 nm over 360-830 nm.
    Spectrum Tabulate() const;
};

/// @brief Upsamples a linear sRGB colour to a sigmoid reflectance.
/// Gauss-Newton as Jakob and Hanika, but minimising in linear sRGB
/// rather than CIELAB. Greys give flat spectra exactly; unreachable
/// colours end at the nearest spectrum found.
/// @param rgb The colour, each component clamped to [0, 1].
/// @return The spectrum whose colour under E, adapted to D65, is
///         `rgb`.
SigmoidSpectrum RgbToSigmoid(glm::dvec3 rgb);

/// @brief A reflectance's colour under illuminant E, adapted to D65.
/// @param reflectance The spectrum.
/// @return Linear sRGB.
glm::dvec3 ReflectanceToLinearSrgb(const Spectrum& reflectance);

/// @brief `RgbToSigmoid` tabulated as a `Spectrum`.
/// @param rgb The colour.
/// @return The spectrum.
Spectrum RgbToSpectrum(glm::dvec3 rgb);

}  // namespace ernie
