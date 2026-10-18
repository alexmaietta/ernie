// SPDX-License-Identifier: BSD-3-Clause
/// @file spectrum.cpp
/// @brief Spectra, black bodies, colour matching and RGB upsampling.
#include "ernie/physics/spectrum.h"

#include "ernie/core/linear_algebra.h"
#include "ernie/core/math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ernie {

namespace {

/// The visible range RGB mode's channels are integrated over, in nm.
constexpr double kVisibleMinNm = 380.0;
constexpr double kVisibleMaxNm = 780.0;

/// The range sigmoid spectra are defined and fitted over, in nm.
constexpr double kSigmoidMinNm = 360.0;
constexpr double kSigmoidMaxNm = 830.0;

/// Sub-samples per band when averaging analytic functions, enough for 1e-4 relative accuracy with
/// bands up to 100 nm.
constexpr int kSubSamples = 32;

/// Below this many bands, colour weights are fitted rather than integrated.
constexpr int kFitBelow = 16;

// ----------------------------------------------------------------------------------------------
// Colour matching

/// Piecewise Gaussian of Wyman et al.: width `below` left of `mean` and `above` right of it.
double Lobe(double wavelength, double mean, double below, double above) {
    const double t = (wavelength - mean) / (wavelength < mean ? below : above);
    return std::exp(-0.5 * t * t);
}

/// A spectrum's CIE XYZ, integrated in 1 nm steps at their midpoints over [lowNm, highNm).
template <typename Function>
glm::dvec3 IntegrateXyz(Function spectrum, double lowNm, double highNm) {
    glm::dvec3 xyz(0.0);
    for (double w = lowNm + 0.5; w < highNm; w += 1.0) {
        xyz += spectrum(w) * CieXyz(w);
    }
    return xyz;
}

/// The colour matching functions every 1 nm over the sigmoid range, computed once.
const std::vector<glm::dvec3>& SampledCmfs() {
    static const std::vector<glm::dvec3> cmfs = [] {
        std::vector<glm::dvec3> values;
        for (double w = kSigmoidMinNm; w <= kSigmoidMaxNm; w += 1.0) {
            values.push_back(CieXyz(w));
        }
        return values;
    }();
    return cmfs;
}

/// A reflectance's XYZ under the equal-energy illuminant, normalised by the integral of y.
template <typename Function> glm::dvec3 EqualEnergyXyz(Function reflectance) {
    const std::vector<glm::dvec3>& cmfs = SampledCmfs();
    glm::dvec3 xyz(0.0);
    double y = 0.0;
    for (std::size_t i = 0; i < cmfs.size(); ++i) {
        xyz += reflectance(kSigmoidMinNm + static_cast<double>(i)) * cmfs[i];
        y += cmfs[i].y;
    }
    return xyz / y;
}

/// The Bradford adaptation from the equal-energy white to D65, as one matrix.
glm::dmat3 EqualEnergyToD65() {
    // The cone response matrix, rows written out and transposed into GLM's columns.
    const glm::dmat3 bradford = glm::transpose(glm::dmat3(0.8951, 0.2664, -0.1614,  //
                                                          -0.7502, 1.7135, 0.0367,  //
                                                          0.0389, -0.0685, 1.0296));
    const glm::dvec3 source = bradford * EqualEnergyXyz([](double) { return 1.0; });
    const glm::dvec3 target = bradford * glm::dvec3(0.95047, 1.0, 1.08883);
    const glm::dmat3 scale(glm::dvec3(target.x / source.x, 0.0, 0.0),
                           glm::dvec3(0.0, target.y / source.y, 0.0),
                           glm::dvec3(0.0, 0.0, target.z / source.z));
    return glm::inverse(bradford) * scale * bradford;
}

// ----------------------------------------------------------------------------------------------
// Bands and RGB mode

/// The equal-energy white's linear sRGB, which RGB mode's channels are relative to.
glm::dvec3 RgbWhite() {
    static const glm::dvec3 white = EqualEnergyXyzToLinearSrgb(
        IntegrateXyz([](double) { return 1.0; }, kVisibleMinNm, kVisibleMaxNm));
    return white;
}

/// The linear sRGB channel RGB-mode band `band` carries of spectrum `f` (blue, green then red, by
/// rising wavelength), relative to the equal-energy white's, so a flat spectrum has its value in
/// every channel. At least 0, for lights more saturated than sRGB holds.
template <typename Function> double RgbChannel(int band, Function f) {
    const glm::dvec3 xyz = IntegrateXyz(f, kVisibleMinNm, kVisibleMaxNm);
    const glm::dvec3 rgb = EqualEnergyXyzToLinearSrgb(xyz) / RgbWhite();
    return std::max(rgb[2 - band], 0.0);
}

/// The mean of `f` over a band, from midpoint sub-samples. In RGB mode a scalar spectrum's band
/// value is its RGB channel instead.
template <typename Function> auto BandAverage(const SpectralBands& bands, int band, Function f) {
    using Result = decltype(f(0.0));
    if constexpr (std::is_same_v<Result, double>) {
        if (bands.rgb) {
            return RgbChannel(band, f);
        }
    }
    Result sum{};
    for (int k = 0; k < kSubSamples; ++k) {
        sum += f(bands.Lower(band) + (k + 0.5) * bands.Width() / kSubSamples);
    }
    return sum / static_cast<double>(kSubSamples);
}

// ----------------------------------------------------------------------------------------------
// Colour weights

/// RGB mode's weights: each channel's XYZ, inverting `RgbChannel`'s projection, so band values of
/// 1 give the equal-energy white's XYZ.
std::vector<glm::dvec3> RgbModeColourWeights() {
    const glm::dmat3 toRgb(EqualEnergyXyzToLinearSrgb({1.0, 0.0, 0.0}),
                           EqualEnergyXyzToLinearSrgb({0.0, 1.0, 0.0}),
                           EqualEnergyXyzToLinearSrgb({0.0, 0.0, 1.0}));
    const glm::dmat3 toXyz = glm::inverse(toRgb);
    const glm::dvec3 white = RgbWhite();
    std::vector<glm::dvec3> weights(3);
    for (int band = 0; band < 3; ++band) {
        glm::dvec3 channel(0.0);
        channel[2 - band] = white[2 - band];
        weights[static_cast<std::size_t>(band)] = toXyz * channel;
    }
    return weights;
}

/// Colours of every hue and saturation at full brightness; the fit is linear, so darker ones
/// follow. Greys are left out: white is added separately with a large weight.
std::vector<glm::dvec3> SaturatedColours() {
    constexpr int kSteps = 6;
    std::vector<glm::dvec3> colours;
    for (int r = 0; r <= kSteps; ++r) {
        for (int g = 0; g <= kSteps; ++g) {
            for (int b = 0; b <= kSteps; ++b) {
                const glm::dvec3 rgb = glm::dvec3(r, g, b) / static_cast<double>(kSteps);
                const double top = std::max({rgb.r, rgb.g, rgb.b});
                if (top == 1.0 && top != std::min({rgb.r, rgb.g, rgb.b})) {
                    colours.push_back(rgb);
                }
            }
        }
    }
    return colours;
}

/// Weights w minimising the error of sum_i w_i a_i over many reflectances, a being a reflectance
/// sampled at the band centres as shaders sample it, against its exact XYZ.
std::vector<glm::dvec3> FitColourWeights(const SpectralBands& bands) {
    const auto n = static_cast<std::size_t>(bands.count);
    std::vector<double> normal(n * n, 0.0);
    std::vector<glm::dvec3> right(n, glm::dvec3(0.0));
    const auto add = [&](const SigmoidSpectrum& spectrum, double weight) {
        std::vector<double> sampled(n);
        for (std::size_t i = 0; i < n; ++i) {
            sampled[i] = spectrum(bands.Centre(static_cast<int>(i)));
        }
        const glm::dvec3 xyz = IntegrateXyz(spectrum, bands.minNm, bands.maxNm);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                normal[i * n + j] += weight * sampled[i] * sampled[j];
            }
            right[i] += weight * sampled[i] * xyz;
        }
    };
    for (const glm::dvec3& rgb : SaturatedColours()) {
        add(RgbToSigmoid(rgb), 1.0);
    }
    add(SigmoidSpectrum{}, 1e3);  // White, weighted to keep greys exact.
    return SolveLinearSystem(std::move(normal), std::move(right));
}

/// `FitColourWeights`, computed once per band layout since renderers ask every frame.
const std::vector<glm::dvec3>& CachedColourFit(const SpectralBands& bands) {
    static std::mutex mutex;
    static std::map<std::tuple<double, double, int>, std::vector<glm::dvec3>> fitted;
    const std::lock_guard lock(mutex);
    const auto key = std::tuple(bands.minNm, bands.maxNm, bands.count);
    if (const auto it = fitted.find(key); it != fitted.end()) {
        return it->second;
    }
    return fitted.emplace(key, FitColourWeights(bands)).first->second;
}

// ----------------------------------------------------------------------------------------------
// CIE daylight

/// CIE daylight components S0, S1 and S2 from 300 to 830 nm every 10 nm (CIE 015:2018, table 6).
constexpr double kDaylightFirstNm = 300.0;
constexpr double kDaylightStepNm = 10.0;
constexpr std::array<double, 54> kDaylightS0{
    0.04,  6.0,   29.6,  55.3,  57.3,  61.8,  61.5,  68.8,  63.4,  65.8,  94.8,  104.8, 105.9, 96.8,
    113.9, 125.6, 125.5, 121.3, 121.3, 113.5, 113.1, 110.8, 106.5, 108.8, 105.3, 104.4, 100.0, 96.0,
    95.1,  89.1,  90.5,  90.3,  88.4,  84.0,  85.1,  81.9,  82.6,  84.9,  81.3,  71.9,  74.3,  76.4,
    63.3,  71.7,  77.0,  65.2,  47.7,  68.6,  65.0,  66.0,  61.0,  53.3,  58.9,  61.9};
constexpr std::array<double, 54> kDaylightS1{
    0.02,  4.5,   22.4,  42.0,  40.6, 41.6,  38.0,  42.4,  38.5,  35.0,  43.4,  46.3,  43.9,  37.1,
    36.7,  35.9,  32.6,  27.9,  24.3, 20.1,  16.2,  13.2,  8.6,   6.1,   4.2,   1.9,   0.0,   -1.6,
    -3.5,  -3.5,  -5.8,  -7.2,  -8.6, -9.5,  -10.9, -10.7, -12.0, -14.0, -13.6, -12.0, -13.3, -12.9,
    -10.6, -11.6, -12.2, -10.2, -7.8, -11.2, -10.4, -10.6, -9.7,  -8.3,  -9.3,  -9.8};
constexpr std::array<double, 54> kDaylightS2{
    0.0,  2.0,  4.0,  8.5,  7.8,  6.7,  5.3,  6.1,  3.0,  1.2,  -1.1, -0.5, -0.7, -1.2,
    -2.6, -2.9, -2.8, -2.6, -2.6, -1.8, -1.5, -1.3, -1.2, -1.0, -0.5, -0.3, 0.0,  0.2,
    0.5,  2.1,  3.2,  4.1,  4.7,  5.1,  6.7,  7.3,  8.6,  9.8,  10.2, 8.3,  9.6,  8.5,
    7.0,  7.6,  8.0,  6.7,  5.2,  7.4,  6.8,  7.0,  6.4,  5.5,  6.1,  6.5};

/// The daylight locus: the chromaticity (x, y) of daylight at a correlated colour temperature.
glm::dvec2 DaylightChromaticity(double kelvin) {
    const double t = kelvin;
    const double x = t <= 7000.0
                         ? -4.6070e9 / (t * t * t) + 2.9678e6 / (t * t) + 0.09911e3 / t + 0.244063
                         : -2.0064e9 / (t * t * t) + 1.9018e6 / (t * t) + 0.24748e3 / t + 0.237040;
    return {x, -3.000 * x * x + 2.870 * x - 0.275};
}

/// The weights M1 and M2 of components S1 and S2 for daylight of chromaticity `xy`.
glm::dvec2 DaylightComponentWeights(glm::dvec2 xy) {
    const double m = 0.0241 + 0.2562 * xy.x - 0.7341 * xy.y;
    return {(-1.3515 - 1.7703 * xy.x + 5.9114 * xy.y) / m,
            (0.0300 - 31.4424 * xy.x + 30.0717 * xy.y) / m};
}

// ----------------------------------------------------------------------------------------------
// Sigmoid spectra

double Sigmoid(double x) {
    return 0.5 + x / (2.0 * std::sqrt(1.0 + x * x));
}

/// Inverse of `Sigmoid` for values strictly inside (0, 1).
double InverseSigmoid(double y) {
    return (2.0 * y - 1.0) / (2.0 * std::sqrt(y * (1.0 - y)));
}

/// A sigmoid spectrum's colour under illuminant E, adapted to D65.
glm::dvec3 SigmoidColour(const SigmoidSpectrum& spectrum) {
    return EqualEnergyXyzToLinearSrgb(EqualEnergyXyz(spectrum));
}

/// The colour's derivatives with respect to the three coefficients, by forward differences.
glm::dmat3 SigmoidColourJacobian(const SigmoidSpectrum& spectrum) {
    glm::dmat3 jacobian;
    for (int k = 0; k < 3; ++k) {
        SigmoidSpectrum nudged = spectrum;
        const double h = 1e-5 * std::max(1.0, std::abs(spectrum.coefficients[k]));
        nudged.coefficients[k] += h;
        jacobian[k] = (SigmoidColour(nudged) - SigmoidColour(spectrum)) / h;
    }
    return jacobian;
}

/// The Gauss-Newton step `step` from `spectrum`, halved up to ten times until it brings the colour
/// closer to `rgb` than `error`; nothing if no length of it does.
std::optional<SigmoidSpectrum> DampedStep(const SigmoidSpectrum& spectrum, const glm::dvec3& step,
                                          const glm::dvec3& rgb, double error) {
    double damping = 1.0;
    for (int halving = 0; halving < 10; ++halving) {
        SigmoidSpectrum next = spectrum;
        next.coefficients = spectrum.coefficients - damping * step;
        if (glm::length(SigmoidColour(next) - rgb) < error) {
            return next;
        }
        damping *= 0.5;
    }
    return std::nullopt;
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// Spectral bands

void SpectralBands::Check() const {
    if (!(minNm > 0.0) || !(maxNm > minNm) || count < 1 || count > kMaxBands) {
        throw std::invalid_argument("A spectral range needs 0 < min < max and 1 to " +
                                    std::to_string(kMaxBands) + " bands");
    }
    if (rgb && count != 3) {
        throw std::invalid_argument("RGB mode has 3 bands");
    }
}

double BandValue(const SpectralBands& bands, int band,
                 const std::function<double(double)>& spectrum) {
    return BandAverage(bands, band, spectrum);
}

// ----------------------------------------------------------------------------------------------
// Tabulated spectra

Spectrum::Spectrum(std::vector<double> wavelengthsNm, std::vector<double> values)
    : wavelengths_(std::move(wavelengthsNm)), values_(std::move(values)) {
    if (wavelengths_.size() != values_.size()) {
        throw std::invalid_argument("A spectrum needs one value per wavelength");
    }
    for (std::size_t i = 1; i < wavelengths_.size(); ++i) {
        if (!(wavelengths_[i] > wavelengths_[i - 1])) {
            throw std::invalid_argument("Spectrum wavelengths must be strictly increasing");
        }
    }
}

Spectrum Spectrum::Constant(double value) {
    return Spectrum({1.0}, {value});
}

Spectrum Spectrum::FromPairs(std::span<const float> pairs) {
    if (pairs.size() % 2 != 0) {
        throw std::invalid_argument("A spectrum needs (wavelength, value) pairs");
    }
    std::vector<double> wavelengths;
    std::vector<double> values;
    for (std::size_t i = 0; i < pairs.size(); i += 2) {
        wavelengths.push_back(pairs[i]);
        values.push_back(pairs[i + 1]);
    }
    return Spectrum(std::move(wavelengths), std::move(values));
}

double Spectrum::operator()(double wavelengthNm) const {
    return Interpolate<double>(wavelengths_, values_, wavelengthNm);
}

double Spectrum::Integral(double lowNm, double highNm) const {
    // The interpolant is linear between samples, so the trapezoid rule over the range's ends and
    // every sample inside it is exact.
    std::vector<double> knots{lowNm, highNm};
    for (const double w : wavelengths_) {
        if (w > lowNm && w < highNm) {
            knots.push_back(w);
        }
    }
    std::sort(knots.begin(), knots.end());
    double sum = 0.0;
    for (std::size_t i = 1; i < knots.size(); ++i) {
        sum += 0.5 * ((*this)(knots[i - 1]) + (*this)(knots[i])) * (knots[i] - knots[i - 1]);
    }
    return sum;
}

double Spectrum::Average(double lowNm, double highNm) const {
    if (!(highNm > lowNm)) {
        return (*this)(lowNm);
    }
    return Integral(lowNm, highNm) / (highNm - lowNm);
}

std::vector<float> Spectrum::Bin(const SpectralBands& bands) const {
    bands.Check();
    std::vector<float> binned(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        binned[static_cast<std::size_t>(i)] =
            static_cast<float>(bands.rgb ? RgbChannel(i, [this](double w) { return (*this)(w); })
                                         : Average(bands.Lower(i), bands.Lower(i) + bands.Width()));
    }
    return binned;
}

// ----------------------------------------------------------------------------------------------
// Black bodies and daylight

double PlanckRadiance(double wavelengthNm, double kelvin) {
    const double lambda = wavelengthNm * 1e-9;
    const double exponent = kPlanck * kSpeedOfLight / (lambda * kBoltzmann * kelvin);
    const double perMetre = 2.0 * kPlanck * kSpeedOfLight * kSpeedOfLight /
                            (std::pow(lambda, 5.0) * std::expm1(exponent));
    return perMetre * 1e-9;
}

double CieDaylight(double wavelengthNm, double kelvin) {
    if (!(kelvin >= 4000.0 && kelvin <= 25000.0)) {
        throw std::invalid_argument("CIE illuminant D needs a temperature from 4000 to 25000 K");
    }
    const glm::dvec2 m = DaylightComponentWeights(DaylightChromaticity(kelvin));
    const double position = (wavelengthNm - kDaylightFirstNm) / kDaylightStepNm;
    if (position < 0.0 || position > static_cast<double>(kDaylightS0.size() - 1)) {
        return 0.0;
    }
    const auto i = std::min(static_cast<std::size_t>(position), kDaylightS0.size() - 2);
    const double f = position - static_cast<double>(i);
    const auto daylight = [&](std::size_t k) {
        return kDaylightS0[k] + m.x * kDaylightS1[k] + m.y * kDaylightS2[k];
    };
    return (1.0 - f) * daylight(i) + f * daylight(i + 1);
}

std::vector<float> BlackbodyIrradiance(const SpectralBands& bands, double kelvin,
                                       double totalIrradiance) {
    bands.Check();
    if (!(kelvin > 0.0)) {
        throw std::invalid_argument("A black body needs a positive temperature");
    }
    const double totalRadiance = kStefanBoltzmann * std::pow(kelvin, 4.0) / std::numbers::pi;
    std::vector<float> irradiance(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        const double radiance =
            BandAverage(bands, i, [kelvin](double w) { return PlanckRadiance(w, kelvin); });
        irradiance[static_cast<std::size_t>(i)] =
            static_cast<float>(totalIrradiance * radiance / totalRadiance);
    }
    return irradiance;
}

std::vector<float> ThermalEmission(const SpectralBands& bands, double kelvin,
                                   const std::function<double(double)>& emissivity) {
    bands.Check();
    std::vector<float> radiance(static_cast<std::size_t>(bands.count), 0.0f);
    if (!(kelvin > 0.0)) {
        return radiance;
    }
    for (int i = 0; i < bands.count; ++i) {
        radiance[static_cast<std::size_t>(i)] = static_cast<float>(BandAverage(
            bands, i, [&](double w) { return emissivity(w) * PlanckRadiance(w, kelvin); }));
    }
    return radiance;
}

// ----------------------------------------------------------------------------------------------
// Colour

glm::dvec3 CieXyz(double w) {
    const double x = 1.056 * Lobe(w, 599.8, 37.9, 31.0) + 0.362 * Lobe(w, 442.0, 16.0, 26.7) -
                     0.065 * Lobe(w, 501.1, 20.4, 26.2);
    const double y = 0.821 * Lobe(w, 568.8, 46.9, 40.5) + 0.286 * Lobe(w, 530.9, 16.3, 31.1);
    const double z = 1.217 * Lobe(w, 437.0, 11.8, 36.0) + 0.681 * Lobe(w, 459.0, 26.0, 13.8);
    return {x, y, z};
}

std::vector<glm::dvec3> CieWeights(const SpectralBands& bands) {
    bands.Check();
    std::vector<glm::dvec3> weights(static_cast<std::size_t>(bands.count));
    for (int i = 0; i < bands.count; ++i) {
        weights[static_cast<std::size_t>(i)] = BandAverage(bands, i, CieXyz) * bands.Width();
    }
    return weights;
}

std::vector<glm::dvec3> ColourWeights(const SpectralBands& bands) {
    if (bands.rgb) {
        return RgbModeColourWeights();
    }
    if (bands.count >= kFitBelow) {
        return CieWeights(bands);
    }
    return CachedColourFit(bands);
}

glm::dvec3 ToXyz(const SpectralBands& bands, std::span<const float> values) {
    const std::vector<glm::dvec3> weights = CieWeights(bands);
    if (values.size() != weights.size()) {
        throw std::invalid_argument("ToXyz needs one value per band");
    }
    glm::dvec3 xyz(0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        xyz += static_cast<double>(values[i]) * weights[i];
    }
    return xyz;
}

glm::dvec3 XyzToLinearSrgb(glm::dvec3 xyz) {
    return {3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
            -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
            0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z};
}

glm::dvec3 EqualEnergyXyzToLinearSrgb(glm::dvec3 xyz) {
    static const glm::dmat3 adaptation = EqualEnergyToD65();
    return XyzToLinearSrgb(adaptation * xyz);
}

// ----------------------------------------------------------------------------------------------
// RGB upsampling

double SigmoidSpectrum::operator()(double wavelengthNm) const {
    const double t = (wavelengthNm - kSigmoidMinNm) / (kSigmoidMaxNm - kSigmoidMinNm);
    return Sigmoid((coefficients.x * t + coefficients.y) * t + coefficients.z);
}

Spectrum SigmoidSpectrum::Tabulate() const {
    std::vector<double> wavelengths;
    std::vector<double> values;
    for (double w = kSigmoidMinNm; w <= kSigmoidMaxNm; w += 5.0) {
        wavelengths.push_back(w);
        values.push_back((*this)(w));
    }
    return Spectrum(std::move(wavelengths), std::move(values));
}

glm::dvec3 ReflectanceToLinearSrgb(const Spectrum& reflectance) {
    return EqualEnergyXyzToLinearSrgb(EqualEnergyXyz([&](double w) { return reflectance(w); }));
}

SigmoidSpectrum RgbToSigmoid(glm::dvec3 rgb) {
    rgb = glm::clamp(rgb, glm::dvec3(0.0), glm::dvec3(1.0));
    // Start from the flat spectrum of the colour's mean, which is exact for greys.
    const double mean = (rgb.r + rgb.g + rgb.b) / 3.0;
    SigmoidSpectrum spectrum{{0.0, 0.0, InverseSigmoid(std::clamp(mean, 1e-6, 1.0 - 1e-6))}};
    if (rgb.r == rgb.g && rgb.g == rgb.b) {
        return spectrum;
    }
    for (int iteration = 0; iteration < 40; ++iteration) {
        const glm::dvec3 residual = SigmoidColour(spectrum) - rgb;
        const double error = glm::length(residual);
        if (error < 1e-7) {
            break;
        }
        const glm::dvec3 step = Solve(SigmoidColourJacobian(spectrum), residual);
        const std::optional<SigmoidSpectrum> next = DampedStep(spectrum, step, rgb, error);
        if (!next) {
            break;
        }
        spectrum = *next;
    }
    return spectrum;
}

Spectrum RgbToSpectrum(glm::dvec3 rgb) {
    return RgbToSigmoid(rgb).Tabulate();
}

}  // namespace ernie
