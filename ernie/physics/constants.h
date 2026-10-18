// SPDX-License-Identifier: BSD-3-Clause
/// @file constants.h
/// @brief Physical constants, in SI units.
///
/// References:
/// - E. Tiesinga et al., "CODATA Recommended Values of the Fundamental Physical Constants: 2018",
///   Reviews of Modern Physics 93, 025010, 2021.
/// - CIE 015:2018, "Colorimetry" (the maximum luminous efficacy).
#pragma once

namespace ernie {

/// Speed of light in vacuum, in m/s (exact).
inline constexpr double kSpeedOfLight = 299792458.0;
/// Planck's constant, in J s (exact).
inline constexpr double kPlanck = 6.62607015e-34;
/// Boltzmann's constant, in J/K (exact).
inline constexpr double kBoltzmann = 1.380649e-23;
/// The Stefan-Boltzmann constant, in W m^-2 K^-4.
inline constexpr double kStefanBoltzmann = 5.670374419e-8;
/// The molar gas constant, in J mol^-1 K^-1 (exact).
inline constexpr double kGasConstant = 8.314462618;
/// Maximum luminous efficacy, in lm/W.
inline constexpr double kLuminousEfficacy = 683.002;

}  // namespace ernie
