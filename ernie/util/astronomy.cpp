// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/astronomy.h"

#include "ernie/util/spectrum.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace ernie {

namespace {

constexpr double kMinKelvin = 1000.0;
constexpr double kMaxKelvin = 50000.0;
constexpr std::size_t kTableSize = 256;

/// The illuminance in lux of a black body at `kelvin` whose spectral
/// irradiance at 555 nm is 1 W m^-2 nm^-1, by the midpoint rule at 1
/// nm over 360 to 830 nm.
double LuxPerUnitAt555(double kelvin) {
    const double at555 = PlanckRadiance(555.0, kelvin);
    double sum = 0.0;
    for (double nm = 360.5; nm < 830.0; nm += 1.0) {
        sum += PlanckRadiance(nm, kelvin) / at555 * CieXyz(nm).y;
    }
    return kLuminousEfficacy * sum;
}

/// `LuxPerUnitAt555` at temperatures spaced evenly in log temperature
/// over 1000 to 50000 K.
const std::array<double, kTableSize>& LuxTable() {
    static const std::array<double, kTableSize> table = [] {
        std::array<double, kTableSize> values{};
        for (std::size_t i = 0; i < kTableSize; ++i) {
            const double t =
                static_cast<double>(i) / (kTableSize - 1);
            values[i] = LuxPerUnitAt555(
                kMinKelvin * std::pow(kMaxKelvin / kMinKelvin, t));
        }
        return values;
    }();
    return table;
}

}  // namespace

StarCatalogue SyntheticStars(std::size_t count, std::uint64_t seed,
                             double faintest) {
    // SplitMix64 (Steele, Lea and Flood 2014), for numbers that
    // repeat on every platform.
    std::uint64_t state = seed;
    const auto uniform = [&state] {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return (static_cast<double>(z >> 11) + 0.5) /
               9007199254740992.0;
    };
    // Counts brighter than m grow as 10^(0.48 m), so magnitudes
    // follow that by inversion between the brightest star's (about
    // -1.5) and the faintest.
    constexpr double kGrowth = 0.48;
    constexpr double kBrightest = -1.5;
    const double low = std::pow(10.0, kGrowth * kBrightest);
    const double high = std::pow(10.0, kGrowth * faintest);
    StarCatalogue stars;
    for (std::size_t i = 0; i < count; ++i) {
        stars.rightAscension.push_back(
            static_cast<float>(360.0 * uniform()));
        stars.declination.push_back(static_cast<float>(
            glm::degrees(std::asin(2.0 * uniform() - 1.0))));
        stars.magnitude.push_back(static_cast<float>(
            std::log10(low + (high - low) * uniform()) / kGrowth));
        // A normal spread of B-V about 0.6 (Box-Muller), within the
        // stars' range.
        const double normal =
            std::sqrt(-2.0 * std::log(uniform())) *
            std::cos(2.0 * glm::pi<double>() * uniform());
        stars.colorIndex.push_back(static_cast<float>(
            std::clamp(0.6 + 0.45 * normal, -0.3, 2.0)));
    }
    return stars;
}

double StarIlluminance(double magnitude, double zeroPoint) {
    return zeroPoint * std::pow(10.0, -0.4 * magnitude);
}

double ColorIndexToTemperature(double colorIndex) {
    return 4600.0 * (1.0 / (0.92 * colorIndex + 1.7) +
                     1.0 / (0.92 * colorIndex + 0.62));
}

double BlackbodySpectralIrradiance(double wavelengthNm, double kelvin,
                                   double lux) {
    kelvin = std::clamp(kelvin, kMinKelvin, kMaxKelvin);
    const double t = std::log(kelvin / kMinKelvin) /
                     std::log(kMaxKelvin / kMinKelvin) *
                     static_cast<double>(kTableSize - 1);
    const std::size_t i =
        std::min(static_cast<std::size_t>(t), kTableSize - 2);
    const double f = t - static_cast<double>(i);
    const std::array<double, kTableSize>& table = LuxTable();
    // Interpolated in log space, where the table is smooth.
    const double luxPerUnit = std::exp(
        std::log(table[i]) * (1.0 - f) + std::log(table[i + 1]) * f);
    return lux / luxPerUnit * PlanckRadiance(wavelengthNm, kelvin) /
           PlanckRadiance(555.0, kelvin);
}

double JulianDate(int year, int month, int day, double hours) {
    if (month <= 2) {
        year -= 1;
        month += 12;
    }
    const double a = std::floor(year / 100.0);
    const double b = 2.0 - a + std::floor(a / 4.0);
    return std::floor(365.25 * (year + 4716)) +
           std::floor(30.6001 * (month + 1)) + day + b - 1524.5 +
           hours / 24.0;
}

double GreenwichMeanSiderealTime(double julianDate) {
    const double days = julianDate - 2451545.0;
    const double centuries = days / 36525.0;
    const double degrees =
        280.46061837 + 360.98564736629 * days +
        0.000387933 * centuries * centuries -
        centuries * centuries * centuries / 38710000.0;
    const double wrapped = std::fmod(degrees, 360.0);
    return wrapped < 0.0 ? wrapped + 360.0 : wrapped;
}

glm::dmat4 CelestialToLocal(double latitude, double longitude,
                            double julianDate) {
    const double localSidereal = glm::radians(
        GreenwichMeanSiderealTime(julianDate) + longitude);
    const double phi = glm::radians(latitude);
    // To hour-angle axes: x the equator's point on the meridian, y
    // east, z the pole.
    const double c = std::cos(localSidereal);
    const double s = std::sin(localSidereal);
    const glm::dmat3 hourAngle(glm::dvec3(c, -s, 0.0),
                               glm::dvec3(s, c, 0.0),
                               glm::dvec3(0.0, 0.0, 1.0));
    // Those axes in east, north, up: the meridian's equator point due
    // south at altitude 90 - latitude, and the pole due north at
    // altitude latitude.
    const glm::dmat3 site(
        glm::dvec3(0.0, -std::sin(phi), std::cos(phi)),
        glm::dvec3(1.0, 0.0, 0.0),
        glm::dvec3(0.0, std::cos(phi), std::sin(phi)));
    return glm::dmat4(site * hourAngle);
}

}  // namespace ernie
