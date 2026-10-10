// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/instruments.h"

#include "ernie/scene/camera.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/tokens.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
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
    Camera(scene, camera)
        .SetSensorType("mono")
        .SetSensorQeMono(qe)
        .SetExposureFStop(fNumber)
        .SetExposureIso(100.0)  // the gain is the ADC's, at ISO 100
        .SetSensorElectronsPerDn(electronsPerDn)
        .SetSensorReadNoise(detectorNoise)
        .SetSensorFullWell(fullWell)
        .SetSensorAdcBits(12)  // 12-bit, encoded to 8-bit
        .SetSensorBlackLevel(0.0)
        .SetSensorNoise(true);

    // The readout: electrons, their noise, the 12-bit ADC, the even and odd channels (sections
    // 3.1.2 and 4.1.1) and the nominal 12- to 8-bit table (Appendix A, code 0); and the
    // photographic outputs, for products that show it (they run only when held).
    const std::pair<const char*, const char*> steps[] = {{"Qe", "qe"},
                                                         {"Noise", "noise"},
                                                         {"Adc", "adc"},
                                                         {"EvenOdd", "readoutTaps"},
                                                         {"Compand", "compandTable"},
                                                         {"Cie", "cie"},
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
    constexpr double kPlanck = 6.62607015e-34;     // J s
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
    case LrocWacBand::Uv320:
        return {320.0, 321.0, 32.0, 865.0, kUv, 36.0};
    case LrocWacBand::Uv360:
        return {360.0, 360.0, 15.0, 918.0, kUv, 36.0};
    case LrocWacBand::Vis415:
        return {415.0, 415.0, 36.0, 3727.0, kVisible, 9.0};
    case LrocWacBand::Vis565:
        return {565.0, 566.0, 20.0, 3310.0, kVisible, 9.0};
    case LrocWacBand::Vis605:
        return {605.0, 604.0, 20.0, 3377.0, kVisible, 9.0};
    case LrocWacBand::Vis645:
        return {645.0, 643.0, 23.0, 3273.0, kVisible, 9.0};
    case LrocWacBand::Vis690:
        return {690.0, 689.0, 39.0, 2612.0, kVisible, 9.0};
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
        static_cast<float>(low - 1.0),  0.0f,      static_cast<float>(low + 1.0),  float(qe),
        static_cast<float>(high - 1.0), float(qe), static_cast<float>(high + 1.0), 0.0f};
    Camera(scene, camera)
        .SetSensorType("mono")
        .SetSensorQeMono(curve)
        .SetSpectralRange(glm::dvec2(low - 1.0, high + 1.0))
        .SetSpectralBands(4)
        .SetLensTransmission(1.0)  // in the effective QE
        .SetExposureFStop(spec.fNumber)
        .SetExposureIso(100.0)  // the gain is the ADC's, at ISO 100
        .SetSensorElectronsPerDn(kWacElectronsPerDn)
        .SetSensorReadNoise(kWacReadNoise)
        .SetSensorFullWell(kWacFullWell)
        .SetSensorAdcBits(11)  // 11-bit, encoded to 8-bit
        .SetSensorBlackLevel(0.0)
        .SetSensorNoise(true);

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

namespace {

// The f-theta lens's Kannala-Brandt k1 for an engineering camera of focal length `focal` (mm) on
// the CMV-20000 whose published fields of view (horizontal, vertical, diagonal; degrees) are
// `fields`: each field's half angle theta reaches the frame's half width over the focal length,
// r, and k1 is the least squares of theta (1 + k1 theta^2) = r over the three.
double FThetaK1(double focal, const double (&fields)[3]) {
    using N = Mars2020Navcam;
    const double halfWidth = 0.5e-3 * N::kPixelPitchUm * N::kWidth;  // mm
    const double halfHeight = 0.5e-3 * N::kPixelPitchUm * N::kHeight;
    const double reaches[3] = {halfWidth, halfHeight, std::hypot(halfWidth, halfHeight)};
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double theta = 0.5 * fields[i] * M_PI / 180.0;
        const double r = reaches[i] / focal;
        const double cube = theta * theta * theta;
        numerator += cube * (r - theta);
        denominator += cube * cube;
    }
    return numerator / denominator;
}

// A Mars 2020 engineering camera (Maki et al. 2020, Table 2): the CMV-20000 camera body with its
// readout chain under `product`/`name`, behind an f-theta lens of focal length `focal` (mm),
// f-number `fNumber`, focused at `focus` (m), its Kannala-Brandt k1 `k1`.
void AuthorEngineeringCamera(Scene& scene, const Path& camera, const Path& product,
                             const char* name, double focal, double fNumber, double focus,
                             double k1, BayerPattern pattern, double jpegQuality) {
    using namespace tokens;
    if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
        throw std::invalid_argument("'" + camera.String() + "' is not a Camera");
    }
    if (!scene.Has(product) || scene.TypeOf(product) != kRenderProduct) {
        throw std::invalid_argument("'" + product.String() + "' is not a RenderProduct");
    }
    using N = Mars2020Navcam;
    // Maki et al. 2020, Table 2: the optics and the CMV-20000.
    Camera(scene, camera)
        .SetProjection(kPerspective)
        .SetLensProjection(kFisheye)
        .SetLensFisheye(std::vector<float>{static_cast<float>(k1), 0.0f, 0.0f, 0.0f})
        .SetLensDistortion(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f})
        .SetFocalLength(focal)
        .SetVerticalAperture(1e-3 * N::kPixelPitchUm * N::kHeight)
        .SetHorizontalAperture(1e-3 * N::kPixelPitchUm * N::kWidth)
        .SetFStop(fNumber)
        .SetFocusDistance(focus)
        .SetExposureFStop(fNumber)
        .SetSensorType("rgb")
        .SetExposureIso(100.0)  // the gain is the ADC's, at ISO 100
        .SetSensorElectronsPerDn(N::kElectronsPerDn)
        .SetSensorReadNoise(N::kDarkNoiseElectrons)
        .SetSensorFullWell(N::kFullWellElectrons)
        .SetSensorAdcBits(N::kAdcBits)
        .SetSensorBlackLevel(0.0)
        .SetSensorNoise(true);

    struct Step {
        const char* name;
        const char* type;
    };
    const Step steps[] = {{"Qe", "qe"},
                          {"Bayer", "bayer"},
                          {"Noise", "noise"},
                          {"Adc", "adc"},
                          {"Demosaic", "demosaic"},
                          {"Cie", "cie"},
                          {"Exposure", "exposure"},
                          {"Jpeg", "jpeg"}};
    std::vector<Path> filters;
    for (const auto& [step, type] : steps) {
        const Path path = product.Child(name).Child(step);
        if (scene.Has(path)) {
            scene.Remove(path);
        }
        Prim filter = scene.Define(path, kSensorFilter);
        filter.Set(kFilterType, std::string(type));
        const std::string kind(type);
        if (kind == "bayer" || kind == "demosaic") {
            filter.Set("inputs:pattern", static_cast<double>(pattern));
        } else if (kind == "jpeg") {
            filter.Set("inputs:quality", jpegQuality);
        }
        filters.push_back(path);
    }
    scene.GetPrim(product).Set(kFilters, filters);
}

// Cameras `rig`/Left and `rig`/Right `baseline` apart along the rig's x, each authored by
// `author` with its product.
template <typename Author>
void AuthorStereoPair(Scene& scene, const Path& rig, const Path& leftProduct,
                      const Path& rightProduct, double baseline, const Author& author) {
    using namespace tokens;
    if (!scene.Has(rig)) {
        scene.Define(rig, kXform);
    }
    const std::pair<const char*, std::pair<double, const Path*>> eyes[] = {
        {"Left", {-0.5 * baseline, &leftProduct}}, {"Right", {0.5 * baseline, &rightProduct}}};
    for (const auto& [name, eye] : eyes) {
        const Path camera = rig.Child(name);
        if (!scene.Has(camera)) {
            scene.Define(camera, kCamera);
        }
        glm::dmat4 offset(1.0);
        offset[3] = glm::dvec4(eye.first, 0.0, 0.0, 1.0);
        scene.GetPrim(camera).Set(kTransform, offset);
        author(camera, *eye.second);
    }
}

}  // namespace

double Mars2020Navcam::FisheyeK1() {
    return FThetaK1(kFocalLengthMm, kFieldOfViewDeg);
}

double Mars2020Hazcam::FisheyeK1() {
    return FThetaK1(kFocalLengthMm, kFieldOfViewDeg);
}

void AuthorMars2020Navcam(Scene& scene, const Path& camera, const Path& product,
                          BayerPattern pattern, double jpegQuality) {
    using N = Mars2020Navcam;
    AuthorEngineeringCamera(scene, camera, product, "Mars2020Navcam", N::kFocalLengthMm,
                            N::kFNumber, N::kBestFocusM, N::FisheyeK1(), pattern, jpegQuality);
}

void AuthorMars2020Hazcam(Scene& scene, const Path& camera, const Path& product,
                          BayerPattern pattern, double jpegQuality) {
    using H = Mars2020Hazcam;
    AuthorEngineeringCamera(scene, camera, product, "Mars2020Hazcam", H::kFocalLengthMm,
                            H::kFNumber, H::kBestFocusM, H::FisheyeK1(), pattern, jpegQuality);
}

void AuthorMars2020HazcamPair(Scene& scene, const Path& rig, const Path& leftProduct,
                              const Path& rightProduct, Mars2020Hazcams which, BayerPattern pattern,
                              double jpegQuality) {
    const double baseline = which == Mars2020Hazcams::Front ? Mars2020Hazcam::kFrontBaselineM
                                                            : Mars2020Hazcam::kRearBaselineM;
    AuthorStereoPair(scene, rig, leftProduct, rightProduct, baseline,
                     [&](const Path& camera, const Path& product) {
                         AuthorMars2020Hazcam(scene, camera, product, pattern, jpegQuality);
                     });
}

MastcamZ::Filter MastcamZ::FilterOf(MastcamZEye eye, int filter) {
    // Bell et al. 2021, Table 3: effective wavelength and HWHM, nm (filter 0 by its green
    // channel; filter 7 with its neutral density).
    static constexpr Filter kLeft[8] = {{544.0, 41.0, 0.0}, {800.0, 9.0, 0.0}, {754.0, 10.0, 0.0},
                                        {677.0, 11.0, 0.0}, {605.0, 9.0, 0.0}, {528.0, 11.0, 0.0},
                                        {442.0, 12.0, 0.0}, {590.0, 88.0, 6.0}};
    static constexpr Filter kRight[8] = {
        {544.0, 42.0, 0.0}, {800.0, 9.0, 0.0},  {866.0, 10.0, 0.0},  {910.0, 12.0, 0.0},
        {939.0, 12.0, 0.0}, {978.0, 10.0, 0.0}, {1022.0, 19.0, 0.0}, {880.0, 10.0, 5.0}};
    if (filter < 0 || filter > 7) {
        throw std::invalid_argument("Mastcam-Z's filter wheels hold filters 0 to 7");
    }
    return (eye == MastcamZEye::Left ? kLeft : kRight)[filter];
}

std::array<MastcamZ::Filter, 3> MastcamZ::BayerOf(MastcamZEye eye) {
    // Table 3, L0/R0: red, green, blue.
    if (eye == MastcamZEye::Left) {
        return {{{630.0, 43.0, 0.0}, {544.0, 41.0, 0.0}, {480.0, 46.0, 0.0}}};
    }
    return {{{631.0, 43.0, 0.0}, {544.0, 42.0, 0.0}, {480.0, 46.0, 0.0}}};
}

double MastcamZ::FNumberAt(double focalLengthMm) {
    const double t = (focalLengthMm - kShortestFocalMm) / (kLongestFocalMm - kShortestFocalMm);
    return kWidestFNumber + t * (kNarrowestFNumber - kWidestFNumber);
}

double MastcamZ::FullWellElectrons(MastcamZEye eye) {
    return eye == MastcamZEye::Left ? 21826.0 : 21818.0;  // Table 5
}

double MastcamZ::ReadNoiseElectrons(MastcamZEye eye) {
    return eye == MastcamZEye::Left ? 22.0 : 21.0;  // Table 5
}

void AuthorMastcamZ(Scene& scene, const Path& camera, const Path& product, MastcamZEye eye,
                    int filter, double focalLengthMm, BayerPattern pattern) {
    using namespace tokens;
    if (!scene.Has(camera) || scene.TypeOf(camera) != kCamera) {
        throw std::invalid_argument("'" + camera.String() + "' is not a Camera");
    }
    if (!scene.Has(product) || scene.TypeOf(product) != kRenderProduct) {
        throw std::invalid_argument("'" + product.String() + "' is not a RenderProduct");
    }
    if (focalLengthMm < MastcamZ::kShortestFocalMm || focalLengthMm > MastcamZ::kLongestFocalMm) {
        throw std::invalid_argument("Mastcam-Z zooms from 26 to 110 mm");
    }
    const MastcamZ::Filter passband = MastcamZ::FilterOf(eye, filter);
    using Z = MastcamZ;
    // Bell et al. 2021, Table 5.
    Camera prim(scene, camera);
    prim.SetProjection(kPerspective)
        .SetLensProjection(kPerspective)
        .SetLensDistortion(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f})
        .SetFocalLength(focalLengthMm)
        .SetVerticalAperture(1e-3 * Z::kPixelPitchUm * Z::kHeight)
        .SetHorizontalAperture(1e-3 * Z::kPixelPitchUm * Z::kWidth)
        .SetExposureFStop(Z::FNumberAt(focalLengthMm))
        .SetExposureIso(100.0)  // the gain is the ADC's, at ISO 100
        .SetSensorElectronsPerDn(Z::kElectronsPerDn)
        .SetSensorReadNoise(Z::ReadNoiseElectrons(eye))
        .SetSensorFullWell(Z::FullWellElectrons(eye))
        .SetSensorAdcBits(Z::kAdcBits)
        .SetSensorBlackLevel(0.0)
        .SetSensorNoise(true);

    std::vector<std::pair<const char*, const char*>> steps;
    if (filter == 0) {
        // The Bayer channels as Gaussians of Table 3's half widths at the ~40% peak, every
        // 5 nm over the visible.
        const std::array<MastcamZ::Filter, 3> bayer = MastcamZ::BayerOf(eye);
        std::array<std::vector<float>, 3> curves;
        for (int c = 0; c < 3; ++c) {
            const double sigma = bayer[c].hwhmNm / std::sqrt(2.0 * std::log(2.0));
            for (double w = 380.0; w <= 780.0; w += 5.0) {
                const double x = (w - bayer[c].effectiveNm) / sigma;
                curves[c].push_back(static_cast<float>(w));
                curves[c].push_back(static_cast<float>(Z::kPeakQe * std::exp(-0.5 * x * x)));
            }
        }
        prim.SetSensorQeR(curves[0])
            .SetSensorQeG(curves[1])
            .SetSensorQeB(curves[2])
            .SetSensorType("rgb")
            .SetSpectralRange(glm::dvec2(380.0, 780.0))
            .SetSpectralBands(16);
        steps = {{"Qe", "qe"},
                 {"Bayer", "bayer"},
                 {"Noise", "noise"},
                 {"Adc", "adc"},
                 {"Demosaic", "demosaic"}};
    } else {
        // The passband on the detector's response (Ernie's generic silicon curve scaled to the
        // ~40% peak), its edges ramped over 2 nm about the half maxima so its area is the FWHM's.
        const Spectrum silicon = SensorModel::Mono().channels.front().quantumEfficiency;
        double siliconPeak = 0.0;
        for (double w = 350.0; w <= 1100.0; w += 1.0) {
            siliconPeak = std::max(siliconPeak, silicon(w));
        }
        const double transmission = std::pow(10.0, -passband.neutralDensity);
        const double low = passband.effectiveNm - passband.hwhmNm;
        const double high = passband.effectiveNm + passband.hwhmNm;
        std::vector<float> curve;
        const auto add = [&](double w, double inBand) {
            curve.push_back(static_cast<float>(w));
            curve.push_back(
                static_cast<float>(inBand * transmission * Z::kPeakQe * silicon(w) / siliconPeak));
        };
        add(low - 1.0, 0.0);
        for (double w = low + 1.0; w <= high - 1.0 + 1e-9;
             w += std::max(1.0, (high - low - 2.0) / 8.0)) {
            add(w, 1.0);
        }
        add(high + 1.0, 0.0);
        prim.SetSensorType("mono")
            .SetSensorQeMono(curve)
            .SetSpectralRange(glm::dvec2(low - 1.0, high + 1.0))
            .SetSpectralBands(4);
        steps = {{"Qe", "qe"}, {"Noise", "noise"}, {"Adc", "adc"}};
    }

    std::vector<Path> filters;
    for (const auto& [name, type] : steps) {
        const Path path = product.Child("MastcamZ").Child(name);
        if (scene.Has(path)) {
            scene.Remove(path);
        }
        Prim step = scene.Define(path, kSensorFilter);
        step.Set(kFilterType, std::string(type));
        if (std::string(type) == "bayer" || std::string(type) == "demosaic") {
            step.Set("inputs:pattern", static_cast<double>(pattern));
        }
        filters.push_back(path);
    }
    scene.GetPrim(product).Set(kFilters, filters);
}

void AuthorMastcamZPair(Scene& scene, const Path& rig, const Path& leftProduct,
                        const Path& rightProduct, int leftFilter, int rightFilter,
                        double focalLengthMm, BayerPattern pattern) {
    using namespace tokens;
    if (!scene.Has(rig)) {
        scene.Define(rig, kXform);
    }
    const double half = 0.5 * MastcamZ::kBaselineM;
    const double toeIn = MastcamZ::kToeInDeg * M_PI / 180.0;
    struct Eye {
        const char* name;
        MastcamZEye eye;
        double x;
        double yaw;  // about the rig's +y: the left camera turns towards +x, the right to -x
        const Path* product;
        int filter;
    };
    const Eye eyes[] = {{"Left", MastcamZEye::Left, -half, -toeIn, &leftProduct, leftFilter},
                        {"Right", MastcamZEye::Right, half, toeIn, &rightProduct, rightFilter}};
    for (const Eye& eye : eyes) {
        const Path camera = rig.Child(eye.name);
        if (!scene.Has(camera)) {
            scene.Define(camera, kCamera);
        }
        glm::dmat4 pose = glm::rotate(glm::dmat4(1.0), eye.yaw, glm::dvec3(0.0, 1.0, 0.0));
        pose[3] = glm::dvec4(eye.x, 0.0, 0.0, 1.0);
        scene.GetPrim(camera).Set(kTransform, pose);
        AuthorMastcamZ(scene, camera, *eye.product, eye.eye, eye.filter, focalLengthMm, pattern);
    }
}

void AuthorMars2020NavcamPair(Scene& scene, const Path& rig, const Path& leftProduct,
                              const Path& rightProduct, BayerPattern pattern, double jpegQuality) {
    AuthorStereoPair(scene, rig, leftProduct, rightProduct, Mars2020Navcam::kBaselineM,
                     [&](const Path& camera, const Path& product) {
                         AuthorMars2020Navcam(scene, camera, product, pattern, jpegQuality);
                     });
}

}  // namespace ernie
