// SPDX-License-Identifier: BSD-3-Clause
/// @file reflectance_library.h
/// @brief Measured reflectance spectra embedded from ECOSTRESS.
#pragma once

#include "ernie/physics/spectrum.h"

#include <span>
#include <string_view>

namespace ernie {

/// @brief A measured reflectance spectrum Ernie embeds.
struct MeasuredReflectance {
    /// The id a material names in `ernie:reflectance:library`.
    std::string_view id;
    /// What was measured.
    std::string_view name;
    /// The ECOSTRESS file it comes from.
    std::string_view source;
    /// Reflectance as a fraction, from 350 nm (or where measurement
    /// starts) to 2.5 um, for most on to 12.5-15 um.
    Spectrum spectrum;
};

/// @brief The ECOSTRESS reflectances (tools/reflectance_library.py).
/// Vegetation, soils, building materials, paints, metals, snow, ice
/// and water. Unlike colours, they hold beyond the visible: the red
/// edge of leaves, snow dark past 1.4 um, metals bright in the
/// thermal infrared.
/// @return Every embedded spectrum.
///
/// References:
/// - S. K. Meerdink, S. J. Hook, D. A. Roberts and E. A. Abbott, "The
///   ECOSTRESS spectral library version 1.0", Remote Sensing of
///   Environment 230:111196, 2019.
/// - A. M. Baldridge, S. J. Hook, C. I. Grove and G. Rivera, "The
///   ASTER spectral library version 2.0", Remote Sensing of
///   Environment 113(4):711-715, 2009.
std::span<const MeasuredReflectance> ReflectanceLibrary();

/// @brief Looks up a measured reflectance.
/// @param id The id, such as "grass" or "fine_snow".
/// @return The reflectance, or null if there is none.
const MeasuredReflectance* FindReflectance(std::string_view id);

}  // namespace ernie
