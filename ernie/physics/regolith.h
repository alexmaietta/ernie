// SPDX-License-Identifier: BSD-3-Clause
/// @file regolith.h
/// @brief Airless bodies' diurnal surface temperatures from a 1-D
/// regolith model.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace ernie {

/// @brief A regolith's thermophysical properties and sunlight.
/// Defaults to the Moon. Density rises exponentially with depth;
/// conductivity follows density plus a T^3 radiative term; heat
/// capacity is a quartic in temperature.
///
/// References:
/// - P. O. Hayne et al., "Global regolith thermophysical properties
///   of the Moon from the Diviner Lunar Radiometer Experiment",
///   Journal of Geophysical Research: Planets 122(12):2371-2400,
///   2017, table A1 and appendix A.
///   https://doi.org/10.1002/2017JE005387
/// - A. R. Vasavada et al., "Lunar equatorial surface temperatures
///   and regolith properties from the Diviner Lunar Radiometer
///   Experiment", Journal of Geophysical Research: Planets
///   117(E12), 2012. https://doi.org/10.1029/2011JE003987
struct RegolithProperties {
    /// Sunlight at the body, in W/m^2.
    double solarIrradiance = 1361.0;
    /// The solar day, in seconds: the Moon's synodic month.
    double dayLength = 29.530589 * 86400.0;
    /// Bond albedo A0 at normal incidence; at incidence i it is
    /// A0 + a (i / (pi/4))^3 + b (i / (pi/2))^8.
    double albedo = 0.12;
    /// The albedo's cubic coefficient a.
    double albedoA = 0.06;
    /// The albedo's eighth-power coefficient b.
    double albedoB = 0.25;
    /// Infrared emissivity.
    double emissivity = 0.95;
    /// Density at the surface, in kg/m^3.
    double surfaceDensity = 1100.0;
    /// Density at depth, in kg/m^3.
    double deepDensity = 1800.0;
    /// E-folding depth of the density, in metres.
    double densityScale = 0.07;
    /// Contact conductivity at the surface, in W/(m K).
    double surfaceConductivity = 7.4e-4;
    /// Contact conductivity at depth, in W/(m K).
    double deepConductivity = 3.4e-3;
    /// Radiative share at 350 K: k = kc (1 + chi (T / 350)^3).
    double radiativeChi = 2.7;
    /// Heat from below, in W/m^2.
    double geothermalFlux = 0.018;
    /// Heat capacity in J/(kg K), as c0 + c1 T + ... + c4 T^4.
    std::array<double, 5> heatCapacity{-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9};
};

/// @brief Surface temperature of level ground over a solar day.
/// Implicit 1-D heat conduction, finer near the surface, run to its
/// periodic state; the sun stays in the equator's plane.
/// @param regolith The regolith.
/// @param latitude The latitude, in radians.
/// @param samples  Samples evenly spaced from local midnight.
/// @return The temperatures, in kelvin.
/// @throws std::invalid_argument For fewer than 2 samples.
std::vector<double> DiurnalSurfaceTemperatures(const RegolithProperties& regolith, double latitude,
                                               int samples);

/// @brief The deep layer's thermal inertia at 250 K, contact
/// conductivity only.
/// @param regolith The regolith.
/// @return sqrt(k rho c) in J m^-2 K^-1 s^-1/2; about 64 for the
///         Moon (Hayne et al. give 55 at 273 K near the surface).
double ThermalInertia(const RegolithProperties& regolith);

/// @brief Rescales contact conductivity to reach a thermal inertia.
/// Densities and heat capacity are kept. Typical values run from 50
/// for fine regolith to 2000 for bare rock.
/// @param regolith The regolith.
/// @param inertia  The target, as `ThermalInertia`.
/// @return The adjusted regolith.
/// @throws std::invalid_argument Unless `inertia` is above 0.
///
/// References:
/// - J. L. Bandfield et al., "Lunar surface rock abundance and
///   regolith fines temperatures derived from LRO Diviner Radiometer
///   data", Journal of Geophysical Research: Planets 116(E12), 2011.
///   https://doi.org/10.1029/2011JE003866
RegolithProperties WithThermalInertia(RegolithProperties regolith, double inertia);

/// @brief Diurnal surface temperatures over latitude, local time,
/// thermal inertia and albedo, as PANGU's look-up tables.
struct DiurnalTable {
    /// Latitude rows, evenly from 0 to 90 degrees.
    int latitudes = 0;
    /// Local times, evenly from midnight; the last precedes midnight.
    int times = 0;
    /// Thermal inertias, ascending; one for a single regolith.
    std::vector<float> inertias;
    /// Bond albedos at normal incidence, ascending; one for a single
    /// albedo.
    std::vector<float> albedos;
    /// Kelvin by latitude, then time, inertia and albedo (`At`).
    std::vector<float> kelvin;

    /// @brief Indexes `kelvin`.
    /// @param latitude The latitude row.
    /// @param time     The time column.
    /// @param inertia  The inertia index.
    /// @param albedo   The albedo index.
    /// @return The index.
    std::size_t At(int latitude, int time, int inertia = 0, int albedo = 0) const;

    /// @brief Packs the table as `ernie:thermal:model:data` for the
    /// "diurnal" model.
    /// @return `kModelDataTag`, the four axis sizes, the inertias,
    ///         the albedos, then `kelvin`.
    std::vector<float> ModelData() const;

    /// @brief Unpacks `ModelData`.
    /// @param data The packed table.
    /// @return The table.
    /// @throws std::invalid_argument For another layout, the wrong
    ///         size, or a table `Check` rejects.
    static DiurnalTable FromModelData(const std::vector<float>& data);

    /// @brief Validates the table.
    /// @throws std::invalid_argument For fewer than 2 latitudes or
    ///         times, empty or unsorted axes, inertias not above 0,
    ///         or a `kelvin` size not matching the axes.
    void Check() const;

    /// `ModelData`'s first value, which no older layout starts with.
    static constexpr float kModelDataTag = -2.0f;
};

/// @brief Bakes `DiurnalSurfaceTemperatures` into a table, rows in
/// parallel.
/// @param regolith  The regolith.
/// @param latitudes Latitude rows from 0 to 90 degrees.
/// @param times     Local times per day.
/// @param inertias  Thermal inertias; empty for the regolith's own.
/// @param albedos   Albedos; empty for the regolith's own.
/// @return The table.
/// @throws std::invalid_argument For fewer than 2 latitudes or
///         times, unsorted axes, inertias not above 0, or albedos
///         outside [0, 1).
DiurnalTable DiurnalTemperatureTable(const RegolithProperties& regolith, int latitudes, int times,
                                     std::vector<float> inertias = {},
                                     std::vector<float> albedos = {});

/// @brief The Moon's table, computed once.
/// Every 5 degrees of latitude and quarter hour; the equator reaches
/// about 390 K at noon and 95 K before dawn, as Diviner measures.
/// @return The table.
const DiurnalTable& LunarDiurnalTable();

}  // namespace ernie
