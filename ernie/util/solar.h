// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/spectrum.h"

/// The Sun's light above any atmosphere.
namespace ernie {

/// The Sun's spectral irradiance at 1 au (W m^-2 nm^-1), from 115 nm
/// to 200 um: the TSIS-1 Hybrid Solar Reference Spectrum version 2
/// with its full spectrum extension, representative of the solar
/// minimum between cycles 24 and 25, kept at 1 nm to 2730 nm and in
/// bins 1% wide beyond (tools/solar_spectrum.py). Its uncertainty is
/// 0.3% between 460 and 2365 nm and 1.3% elsewhere in 202-2730 nm.
/// Licensed CC BY 4.0 by LASP, University of Colorado.
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

/// The integral of `SolarSpectralIrradiance` over all its
/// wavelengths, the total solar irradiance at 1 au (W/m^2), about
/// 1361.5.
double TotalSolarIrradiance();

}  // namespace ernie
