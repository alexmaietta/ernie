// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/solar.h"

#include "ernie/util/solar_table.h"

#include <utility>
#include <vector>

namespace ernie {

const Spectrum& SolarSpectralIrradiance() {
    static const Spectrum spectrum = [] {
        std::vector<double> wavelengths;
        std::vector<double> values;
        wavelengths.reserve(solar_table::kCount);
        values.reserve(solar_table::kCount);
        for (int i = 0; i < solar_table::kCount; ++i) {
            wavelengths.push_back(solar_table::kPairs[2 * i]);
            values.push_back(solar_table::kPairs[2 * i + 1]);
        }
        return Spectrum(std::move(wavelengths), std::move(values));
    }();
    return spectrum;
}

double TotalSolarIrradiance() {
    static const double total = [] {
        const Spectrum& sun = SolarSpectralIrradiance();
        const double low = sun.Wavelengths().front();
        const double high = sun.Wavelengths().back();
        return sun.Average(low, high) * (high - low);
    }();
    return total;
}

}  // namespace ernie
