#include "ernie/scene/instruments.h"

#include "ernie/scene/tokens.h"

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

}  // namespace ernie
