// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ernie {

/// A star catalogue as a StarField's arrays take it: right ascension
/// and declination in degrees, visual magnitude and B-V colour index
/// per star.
struct StarCatalogue {
    std::vector<float> rightAscension;
    std::vector<float> declination;
    std::vector<float> magnitude;
    std::vector<float> colorIndex;
};

/// `count` made-up stars, the same for the same `seed`: spread evenly
/// over the sky, brighter than `faintest` with the number brighter
/// than a magnitude growing about threefold per magnitude, as the
/// sky's do near the naked-eye limit, and colour indices spread
/// around the Sun's. For placeholders and tests; real skies come from
/// catalogues such as the Yale Bright Star Catalogue.
///
/// Reference: A. N. Cox (ed.), "Allen's Astrophysical Quantities",
/// 4th edition, Springer, 2000, section 19.3 (star counts by
/// magnitude).
StarCatalogue SyntheticStars(std::size_t count,
                             std::uint64_t seed = 1,
                             double faintest = 6.5);

/// The illuminance of a magnitude 0 star in lux, from Allen's
/// Astrophysical Quantities: a star of visual magnitude m gives
/// 10^(-0.4 m) of it.
///
/// Reference: A. N. Cox (ed.), "Allen's Astrophysical Quantities",
/// 4th edition, Springer, 2000.
inline constexpr double kMagnitudeZeroLux = 2.54e-6;

/// The illuminance in lux of a star of apparent visual magnitude
/// `magnitude`, with `zeroPoint` the illuminance of magnitude 0
/// (Pogson's ratio: five magnitudes are a factor of 100).
double StarIlluminance(double magnitude,
                       double zeroPoint = kMagnitudeZeroLux);

/// A star's effective temperature in kelvin from its B-V colour
/// index, by Ballesteros' formula for a black body seen through the
/// Johnson B and V filters (good to a few percent for main sequence
/// stars from about 3000 to 30000 K).
///
/// Reference: F. J. Ballesteros, "New insights into black bodies",
/// EPL 97(3):34008, 2012. https://doi.org/10.1209/0295-5075/97/34008
double ColorIndexToTemperature(double colorIndex);

/// The spectral irradiance at `wavelengthNm` (W m^-2 nm^-1) of a
/// black body at `kelvin` whose illuminance is `lux`, through the CIE
/// photopic luminous efficiency (683 lm/W times y-bar). Accurate to a
/// fraction of a percent from 1000 to 50000 K (interpolated in
/// temperature).
double BlackbodySpectralIrradiance(double wavelengthNm, double kelvin,
                                   double lux);

/// The Julian date of a moment in Universal Time: `hours` past
/// midnight of the Gregorian calendar's `day` of `month` (1 to 12) of
/// `year`.
///
/// Reference: J. Meeus, "Astronomical Algorithms", 2nd edition,
/// Willmann-Bell, 1998, chapter 7.
double JulianDate(int year, int month, int day, double hours = 0.0);

/// The Greenwich mean sidereal time in degrees [0, 360) at Julian
/// date `julianDate` (UT).
///
/// Reference: J. Meeus, "Astronomical Algorithms", 2nd edition, 1998,
/// equation 12.4.
double GreenwichMeanSiderealTime(double julianDate);

/// The rotation from the J2000 equatorial frame (x towards right
/// ascension 0, z the north celestial pole) to a site's local frame
/// with x east, y north and z up (Ernie's terrain convention), at
/// `latitude` and `longitude` (degrees, east positive) and Julian
/// date `julianDate` (UT): a StarField's transform for the sky over
/// the site. Stars on the meridian at declination `latitude` are
/// overhead. Precession, nutation, aberration and refraction are left
/// out (at most about half a degree for dates within a few decades of
/// 2000).
///
/// Reference: J. Meeus, "Astronomical Algorithms", 2nd edition, 1998,
/// chapters 12 and 13.
glm::dmat4 CelestialToLocal(double latitude, double longitude,
                            double julianDate);

}  // namespace ernie
