// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <vector>

/// Airless bodies' diurnal surface temperatures from a 1-D model of
/// their regolith.
namespace ernie {

/// A regolith's thermophysical properties and its body's sunlight,
/// the Moon's by default: density rising from its surface value to
/// its deep one over an e-folding depth, contact conductivity
/// following the density, radiative conductivity growing with the
/// cube of temperature, and heat capacity a quartic in temperature.
///
/// References:
/// - P. O. Hayne et al., "Global regolith thermophysical properties
///   of the Moon from the Diviner Lunar Radiometer Experiment",
///   Journal of Geophysical Research: Planets 122(12):2371-2400,
///   2017, table A1 and appendix A. https://doi.org/10.1002/2017JE005387
/// - A. R. Vasavada et al., "Lunar equatorial surface temperatures
///   and regolith properties from the Diviner Lunar Radiometer
///   Experiment", Journal of Geophysical Research: Planets
///   117(E12), 2012. https://doi.org/10.1029/2011JE003987
struct RegolithProperties {
    /// Sunlight at the body, W/m^2.
    double solarIrradiance = 1361.0;
    /// Its solar day, s: the Moon's synodic month.
    double dayLength = 29.530589 * 86400.0;
    /// Bond albedo at normal incidence, rising with incidence i as
    /// A0 + a (i / (pi/4))^3 + b (i / (pi/2))^8.
    double albedo = 0.12;
    double albedoA = 0.06;
    double albedoB = 0.25;
    /// Infrared emissivity.
    double emissivity = 0.95;
    /// Density at the surface and at depth, kg/m^3, and the depth over
    /// which it rises, m.
    double surfaceDensity = 1100.0;
    double deepDensity = 1800.0;
    double densityScale = 0.07;
    /// Contact conductivity at the surface and at depth, W/(m K).
    double surfaceConductivity = 7.4e-4;
    double deepConductivity = 3.4e-3;
    /// Radiative conductivity's share at 350 K: k = kc (1 + chi
    /// (T / 350)^3).
    double radiativeChi = 2.7;
    /// Heat from below, W/m^2.
    double geothermalFlux = 0.018;
    /// Heat capacity, J/(kg K), as c0 + c1 T + ... + c4 T^4.
    std::array<double, 5> heatCapacity{-3.6125, 2.7431, 2.3616e-3,
                                       -1.2340e-5, 8.9093e-9};
};

/// The surface temperature of level ground at `latitude` (radians)
/// over a solar day, `samples` evenly spaced from local midnight, in
/// kelvin, the sun in the equator's plane: the regolith's heat
/// conduction solved implicitly on a grid finer near the surface,
/// run to its periodic state, absorbed sunlight and the geothermal
/// flux in, thermal emission out.
/// @throws std::invalid_argument for fewer than 2 samples.
std::vector<double>
DiurnalSurfaceTemperatures(const RegolithProperties& regolith,
                           double latitude, int samples);

/// Diurnal surface temperatures over latitude and local time.
struct DiurnalTable {
    int latitudes = 0;  ///< Rows, 0 to 90 degrees evenly.
    int times = 0;      ///< Columns, local midnight on, evenly over
                        ///< the day, the last before midnight again.
    std::vector<float> kelvin;  ///< Row by row.
};

/// `DiurnalSurfaceTemperatures` at `latitudes` latitudes from 0 to 90
/// degrees and `times` local times.
/// @throws std::invalid_argument for fewer than 2 of either.
DiurnalTable DiurnalTemperatureTable(const RegolithProperties& regolith,
                                     int latitudes, int times);

/// The Moon's table (default `RegolithProperties`), every 5 degrees of
/// latitude and quarter hour of its 24 local hours, computed once.
/// Its equator reaches about 390 K at noon and cools to about 95 K
/// before dawn, as Diviner measures.
const DiurnalTable& LunarDiurnalTable();

}  // namespace ernie
