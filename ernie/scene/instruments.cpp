#include "ernie/scene/instruments.h"

#include "ernie/scene/tokens.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ernie {

void AuthorLrocNac(Scene& scene, const Path& camera, const Path& product, LrocNac which) {
    using namespace tokens;
    if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
        throw std::invalid_argument("'" + camera.String() + "' is not a Camera");
    }
    if (!scene.Has(product) || scene.TypeOf(product) != kRenderProduct) {
        throw std::invalid_argument("'" + product.String() + "' is not a RenderProduct");
    }
    const bool left = which == LrocNac::Left;
    // Robinson et al. 2010, Table 2 (NAC-L, NAC-R).
    const double fNumber = left ? 3.577 : 3.590;
    const double electronsPerDn = left ? 90.5 : 92.5;
    const double detectorNoise = left ? 101.0 : 97.0;  // electrons
    const double fullWell = left ? 334000.0 : 352000.0;
    // Section 3.1.2: the KLI-5001G's average QE through the 400-760 nm response is > 60%.
    const std::vector<float> qe{390.0f, 0.0f, 400.0f, 0.6f, 760.0f, 0.6f, 770.0f, 0.0f};
    scene.GetPrim(camera)
        .Set(kSensorType, std::string("mono"))
        .Set(std::string(kSensorQePrefix) + "mono", qe)
        .Set(kExposureFStop, fNumber)
        .Set(kExposureIso, 100.0)  // the gain is the ADC's, at ISO 100
        .Set(kSensorGain, electronsPerDn)
        .Set(kSensorReadNoise, detectorNoise)
        .Set(kSensorFullWell, fullWell)
        .Set(kSensorAdcBits, 12)  // 12-bit, encoded to 8-bit
        .Set(kSensorBlackLevel, 0.0)
        .Set(kSensorNoise, true);

    // The readout: electrons, their noise, the 12-bit ADC, the even and odd channels (sections
    // 3.1.2 and 4.1.1) and the nominal 12- to 8-bit table (Appendix A, code 0); and the
    // photographic outputs, for products that show it (they run only when held).
    const std::pair<const char*, const char*> steps[] = {
        {"Qe", "qe"},           {"Noise", "noise"},   {"Adc", "adc"},
        {"EvenOdd", "readoutTaps"}, {"Compand", "compandTable"}, {"Cie", "cie"},
        {"Exposure", "exposure"}};
    std::vector<Path> filters;
    for (const auto& [name, type] : steps) {
        const Path path = product.Child("LrocNac").Child(name);
        if (scene.Has(path)) {
            scene.Remove(path);
        }
        Prim filter = scene.Define(path, kSensorFilter);
        filter.Set(kFilterType, std::string(type));
        if (std::string(type) == "readoutTaps") {
            filter.Set("inputs:taps", 2.0);
        }
        filters.push_back(path);
    }
    scene.GetPrim(product).Set(kFilters, filters);
}

namespace {

// Robinson et al. 2010, Table 3: the KAI-1001's gain, detector noise and full well.
constexpr double kWacElectronsPerDn = 25.9;
constexpr double kWacReadNoise = 66.0;
constexpr double kWacFullWell = 46100.0;

}  // namespace

double LrocWacBandSpec::EffectiveQe() const {
    constexpr double kPlanck = 6.62607015e-34;    // J s
    constexpr double kSpeedOfLight = 299792458.0;  // m / s
    // A flat spectral radiance L (W/m^3/sr) gives L / RC DN per second; Ernie's camera equation
    // gives L FWHM pi / (4 N^2) A (lambda / h c) QE / gain over the band.
    const double area = pixelPitchUm * 1e-6 * pixelPitchUm * 1e-6;
    const double optics = M_PI / (4.0 * fNumber * fNumber);
    return kWacElectronsPerDn * kPlanck * kSpeedOfLight /
           (responsivity * fwhmNm * 1e-9 * optics * area * effectiveNm * 1e-9);
}

LrocWacBandSpec LrocWacBandOf(LrocWacBand band) {
    // Robinson et al. 2010, Table 4 (band name, effective wavelength, FWHM, responsivity
    // coefficient) and Table 3 (f-numbers; the UV's 4 x 4 summed 9 um pixels).
    constexpr double kVisible = 5.052;
    constexpr double kUv = 5.65;
    switch (band) {
        case LrocWacBand::Uv320: return {320.0, 321.0, 32.0, 865.0, kUv, 36.0};
        case LrocWacBand::Uv360: return {360.0, 360.0, 15.0, 918.0, kUv, 36.0};
        case LrocWacBand::Vis415: return {415.0, 415.0, 36.0, 3727.0, kVisible, 9.0};
        case LrocWacBand::Vis565: return {565.0, 566.0, 20.0, 3310.0, kVisible, 9.0};
        case LrocWacBand::Vis605: return {605.0, 604.0, 20.0, 3377.0, kVisible, 9.0};
        case LrocWacBand::Vis645: return {645.0, 643.0, 23.0, 3273.0, kVisible, 9.0};
        case LrocWacBand::Vis690: return {690.0, 689.0, 39.0, 2612.0, kVisible, 9.0};
    }
    throw std::invalid_argument("Unknown WAC band");
}

LrocWacBand LrocWacBandNamed(int nanometres) {
    for (const LrocWacBand band :
         {LrocWacBand::Uv320, LrocWacBand::Uv360, LrocWacBand::Vis415, LrocWacBand::Vis565,
          LrocWacBand::Vis605, LrocWacBand::Vis645, LrocWacBand::Vis690}) {
        const LrocWacBandSpec spec = LrocWacBandOf(band);
        if (nanometres == static_cast<int>(spec.nameNm) ||
            nanometres == static_cast<int>(spec.effectiveNm)) {
            return band;
        }
    }
    throw std::invalid_argument("The WAC has no " + std::to_string(nanometres) +
                                " nm band; its bands are 320, 360, 415, 565, 605, 645 and 690");
}

void AuthorLrocWac(Scene& scene, const Path& camera, const Path& product, LrocWacBand band) {
    using namespace tokens;
    if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
        throw std::invalid_argument("'" + camera.String() + "' is not a Camera");
    }
    if (!scene.Has(product) || scene.TypeOf(product) != kRenderProduct) {
        throw std::invalid_argument("'" + product.String() + "' is not a RenderProduct");
    }
    const LrocWacBandSpec spec = LrocWacBandOf(band);
    // The passband: FWHM wide about the effective wavelength, its edges ramped over 2 nm about
    // the half maxima so its area is the FWHM's.
    const double qe = spec.EffectiveQe();
    const double low = spec.effectiveNm - 0.5 * spec.fwhmNm;
    const double high = spec.effectiveNm + 0.5 * spec.fwhmNm;
    const std::vector<float> curve{
        static_cast<float>(low - 1.0),  0.0f, static_cast<float>(low + 1.0),  float(qe),
        static_cast<float>(high - 1.0), float(qe), static_cast<float>(high + 1.0), 0.0f};
    scene.GetPrim(camera)
        .Set(kSensorType, std::string("mono"))
        .Set(std::string(kSensorQePrefix) + "mono", curve)
        .Set(kSpectralRange, glm::dvec2(low - 1.0, high + 1.0))
        .Set(kSpectralBands, 4)
        .Set(kLensTransmission, 1.0)  // in the effective QE
        .Set(kExposureFStop, spec.fNumber)
        .Set(kExposureIso, 100.0)  // the gain is the ADC's, at ISO 100
        .Set(kSensorGain, kWacElectronsPerDn)
        .Set(kSensorReadNoise, kWacReadNoise)
        .Set(kSensorFullWell, kWacFullWell)
        .Set(kSensorAdcBits, 11)  // 11-bit, encoded to 8-bit
        .Set(kSensorBlackLevel, 0.0)
        .Set(kSensorNoise, true);

    const std::pair<const char*, const char*> steps[] = {
        {"Qe", "qe"}, {"Noise", "noise"}, {"Adc", "adc"}};
    std::vector<Path> filters;
    for (const auto& [name, type] : steps) {
        const Path path = product.Child("LrocWac").Child(name);
        if (scene.Has(path)) {
            scene.Remove(path);
        }
        scene.Define(path, kSensorFilter).Set(kFilterType, std::string(type));
        filters.push_back(path);
    }
    scene.GetPrim(product).Set(kFilters, filters);
}

}  // namespace ernie
