// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/spectrum.h"

#include <span>
#include <string_view>

namespace ernie {

/// A measured reflectance spectrum Ernie embeds: its id (as a
/// material names it in `ernie:reflectance:library`), what was
/// measured, the ECOSTRESS file it comes from, and the spectrum, a
/// fraction from 350 nm (or where its measurement starts) to 2.5 um
/// and for most on into the thermal infrared, 12.5 to 15 um.
struct MeasuredReflectance {
    std::string_view id;
    std::string_view name;
    std::string_view source;
    Spectrum spectrum;
};

/// The measured reflectances, from the ECOSTRESS Spectral Library
/// (tools/reflectance_library.py): green rye grass ("grass"), chamise
/// leaves ("chaparral"), wild oat litter ("dry_grass"), a sandy loam
/// and a sand ("loam", "sand"), weathered concrete, asphalt, red
/// brick and pine ("concrete", "asphalt", "brick", "pine_wood"),
/// black and olive alkyd paints ("black_paint", "olive_paint"), a
/// white roof coating ("white_coating"), weathered aluminium,
/// galvanized steel and copper ("aluminium", "galvanized_steel",
/// "copper"), fine and granular snow, ice and tap water
/// ("fine_snow", "granular_snow", "ice", "water"). Beyond the
/// visible they reflect as the materials do, which colours cannot
/// say: green leaves bright in the near infrared (the red edge),
/// snow dark past 1.4 um, metals bright in the thermal infrared.
///
/// References:
/// - S. K. Meerdink, S. J. Hook, D. A. Roberts and E. A. Abbott, "The
///   ECOSTRESS spectral library version 1.0", Remote Sensing of
///   Environment 230:111196, 2019.
/// - A. M. Baldridge, S. J. Hook, C. I. Grove and G. Rivera, "The
///   ASTER spectral library version 2.0", Remote Sensing of
///   Environment 113(4):711-715, 2009.
std::span<const MeasuredReflectance> ReflectanceLibrary();

/// The measured reflectance with id `id`, or null.
const MeasuredReflectance* FindReflectance(std::string_view id);

}  // namespace ernie
