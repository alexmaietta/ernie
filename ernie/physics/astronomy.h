// SPDX-License-Identifier: BSD-3-Clause
/// @file astronomy.h
/// @brief Star catalogues, stellar photometry and sky orientation.
#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ernie {

/// @brief A star catalogue in the arrays a StarField takes.
struct StarCatalogue {
    /// Right ascension per star, in degrees.
    std::vector<float> rightAscension;
    /// Declination per star, in degrees.
    std::vector<float> declination;
    /// Apparent visual magnitude per star.
    std::vector<float> magnitude;
    /// B-V colour index per star.
    std::vector<float> colorIndex;
};

/// @brief Makes up stars for placeholders and tests.
/// Spread evenly over the sky; counts grow about threefold per
/// magnitude as near the naked-eye limit; colours spread around the
/// Sun's.
/// @param count    The number of stars.
/// @param seed     The seed; the same seed gives the same stars.
/// @param faintest The faintest magnitude.
/// @return The catalogue.
///
/// References:
/// - A. N. Cox (ed.), "Allen's Astrophysical Quantities", 4th
///   edition, Springer, 2000, section 19.3 (star counts by
///   magnitude).
StarCatalogue SyntheticStars(std::size_t count, std::uint64_t seed = 1, double faintest = 6.5);

/// Illuminance of a magnitude 0 star, in lux; magnitude m gives
/// 10^(-0.4 m) of it.
///
/// References:
/// - A. N. Cox (ed.), "Allen's Astrophysical Quantities", 4th
///   edition, Springer, 2000.
inline constexpr double kMagnitudeZeroLux = 2.54e-6;

/// @brief A star's illuminance by Pogson's ratio.
/// @param magnitude Apparent visual magnitude.
/// @param zeroPoint Illuminance of magnitude 0, in lux.
/// @return The illuminance, in lux.
double StarIlluminance(double magnitude, double zeroPoint = kMagnitudeZeroLux);

/// @brief A star's effective temperature from its B-V index.
/// Ballesteros' black-body formula; within a few percent for main
/// sequence stars of 3000-30000 K.
/// @param colorIndex The B-V colour index.
/// @return The temperature, in kelvin.
///
/// References:
/// - F. J. Ballesteros, "New insights into black bodies", EPL
///   97(3):34008, 2012. https://doi.org/10.1209/0295-5075/97/34008
double ColorIndexToTemperature(double colorIndex);

/// @brief A black body's spectral irradiance for a given illuminance.
/// Uses the CIE photopic curve (683 lm/W times y-bar); within a
/// fraction of a percent for 1000-50000 K.
/// @param wavelengthNm The wavelength, in nm.
/// @param kelvin       The temperature, in kelvin.
/// @param lux          The illuminance, in lux.
/// @return The spectral irradiance, in W m^-2 nm^-1.
double BlackbodySpectralIrradiance(double wavelengthNm, double kelvin, double lux);

/// @brief The Julian date of a Gregorian UT moment.
/// @param year  The year.
/// @param month The month, 1 to 12.
/// @param day   The day of the month.
/// @param hours Hours past midnight UT.
/// @return The Julian date.
///
/// References:
/// - J. Meeus, "Astronomical Algorithms", 2nd edition, Willmann-Bell,
///   1998, chapter 7.
double JulianDate(int year, int month, int day, double hours = 0.0);

/// @brief Greenwich mean sidereal time.
/// @param julianDate The Julian date (UT).
/// @return The time in degrees, [0, 360).
///
/// References:
/// - J. Meeus, "Astronomical Algorithms", 2nd edition, 1998,
///   equation 12.4.
double GreenwichMeanSiderealTime(double julianDate);

/// @brief Rotation from J2000 equatorial to a site's local frame.
/// The local frame is x east, y north, z up: a StarField's transform
/// for the sky over the site. Ignores precession, nutation,
/// aberration and refraction (under half a degree near 2000).
/// @param latitude   Site latitude, in degrees.
/// @param longitude  Site longitude, in degrees, east positive.
/// @param julianDate The Julian date (UT).
/// @return The rotation.
///
/// References:
/// - J. Meeus, "Astronomical Algorithms", 2nd edition, 1998,
///   chapters 12 and 13.
glm::dmat4 CelestialToLocal(double latitude, double longitude, double julianDate);

}  // namespace ernie
