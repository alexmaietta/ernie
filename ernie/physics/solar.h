// SPDX-License-Identifier: BSD-3-Clause
/// @file solar.h
/// @brief The Sun's light above any atmosphere.
#pragma once

#include "ernie/physics/spectrum.h"

namespace ernie {

/// @brief The Sun's spectral irradiance at 1 au, 115 nm to 200 um.
/// @return The TSIS-1 HSRS v2 full spectrum (W m^-2 nm^-1).
/// @note Solar minimum of cycles 24-25; 1 nm steps to 2730 nm, 1%
///       bins beyond (tools/solar_spectrum.py). Uncertainty 0.3% in
///       460-2365 nm, 1.3% elsewhere in 202-2730 nm. CC BY 4.0, LASP.
///
/// References:
/// - O. M. Coddington et al., "The TSIS-1 Hybrid Solar Reference
///   Spectrum", Geophysical Research Letters 48(12):e2020GL091709,
///   2021. https://doi.org/10.1029/2020GL091709
/// - O. M. Coddington et al., "Version 2 of the TSIS-1 Hybrid Solar
///   Reference Spectrum and Extension to the Full Spectrum", Earth
///   and Space Science 10(3):e2022EA002637, 2023.
///   https://doi.org/10.1029/2022EA002637
const Spectrum& SolarSpectralIrradiance();

/// @brief Integrates `SolarSpectralIrradiance` over all wavelengths.
/// @return The total solar irradiance at 1 au, about 1361.5 W/m^2.
double TotalSolarIrradiance();

}  // namespace ernie
