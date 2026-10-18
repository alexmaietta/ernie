// SPDX-License-Identifier: BSD-3-Clause
/// @file atmosphere.h
/// @brief Planetary atmospheres as physical constituents, and their
/// reduction to a camera's spectral bands.
/// Altitudes and radii are in m, number densities m^-3, cross
/// sections m^2 and coefficients m^-1.
#pragma once

#include "ernie/physics/scattering.h"
#include "ernie/physics/spectrum.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ernie {

/// @brief One term of a vertical density profile, relative to its
/// reference density.
struct DensityLayer {
    /// @brief The layer's shape over altitude h.
    enum class Shape {
        /// exp(-h / height): an isothermal gas.
        Exponential,
        /// exp(1 - exp(h / height)), 1 at the ground: dust falling
        /// off faster than exponentially (Conrath; Schneegans et
        /// al.).
        DoubleExponential,
        /// 1 - |h - height| / width, clamped at 0: peaks at `height`.
        Tent,
    };

    /// The shape.
    Shape shape = Shape::Exponential;
    /// Scale height or peak altitude, in metres.
    double height = 8000.0;
    /// A tent's half width, in metres.
    double width = 0.0;
    /// The layer's weight in the profile's mean.
    double weight = 1.0;
};

/// @brief A vertical density profile: the weighted mean of its
/// layers, or a table.
/// Exponential layers give 1 at the ground.
///
/// References:
/// - B. J. Conrath, "Thermal structure of the Martian atmosphere
///   during the dissipation of the dust storm of 1971", Icarus
///   24(1):36-46, 1975. https://doi.org/10.1016/0019-1035(75)90156-6
/// - S. Schneegans, T. Meyran, I. Ginkel, G. Zachmann and A. Gerndt,
///   "Physically Based Real-Time Rendering of Atmospheres using Mie
///   Theory", Computer Graphics Forum 43(2), 2024, eq. (7).
///   https://doi.org/10.1111/cgf.15010
struct DensityProfile {
    /// The layers, when `table` is empty.
    std::vector<DensityLayer> layers;
    /// Relative densities every `tableStep` m from the ground, used
    /// in place of the layers when not empty; linear between, last
    /// value beyond.
    std::vector<double> table;
    /// The table's altitude step, in metres.
    double tableStep = 0.0;

    /// @param scaleHeight The scale height, in metres.
    /// @return One exponential layer.
    static DensityProfile Exponential(double scaleHeight);
    /// @param peakAltitude The peak's altitude, in metres.
    /// @param halfWidth    The half width, in metres.
    /// @return One tent layer.
    static DensityProfile Tent(double peakAltitude, double halfWidth);

    /// @brief A hydrostatic gas under the U.S. Standard Atmosphere
    /// 1976's lapse rates, shifted to a surface temperature.
    /// Tabulated every 100 m to 120 km; heights are treated as
    /// geometric (the standard's are geopotential, 1.6% apart at 100
    /// km).
    /// @param surfaceTemperature Ground temperature, in kelvin.
    /// @param molarMass          Molar mass, in kg/mol.
    /// @param gravity            Surface gravity, in m/s^2.
    /// @return Density relative to the ground's.
    ///
    /// References:
    /// - NOAA, NASA and USAF, "U.S. Standard Atmosphere, 1976",
    ///   NOAA-S/T 76-1562, 1976, table 4.
    ///   https://ntrs.nasa.gov/citations/19770009539
    static DensityProfile StandardAtmosphere(double surfaceTemperature, double molarMass,
                                             double gravity);

    /// @param altitude The altitude, in metres.
    /// @return The relative density; 0 without layers or a table.
    double operator()(double altitude) const;

    /// @param top The upper altitude, in metres.
    /// @return The relative density integrated from the ground.
    double Column(double top) const;
};

/// Optical properties of one particle or molecule at a wavelength
/// in nm, with the phase function at the given cosines.
using OpticsModel = std::function<ParticleOptics(double wavelengthNm, std::span<const double>)>;

/// @brief Dry air molecules, Rayleigh after Bodhaine et al. (1999).
/// @param co2Fraction The CO2 volume fraction.
/// @return The optics model.
OpticsModel AirMolecules(double co2Fraction = 360e-6);

/// @brief Molecules of a CO2 atmosphere such as Mars's.
/// CO2 scatters per Owens and Bates; the remainder (mostly N2 and Ar
/// on Mars, within 7% of air) scatters as air.
/// @param co2Fraction The CO2 volume fraction.
/// @return The optics model.
OpticsModel CarbonDioxideMolecules(double co2Fraction);

/// @return Ozone molecules, absorbing only (`OzoneCrossSection`).
OpticsModel OzoneMolecules();

/// @brief Mie spheres with a log-normal size distribution
/// (`MieParticles`).
/// @param modes           The size distribution.
/// @param refractiveIndex The index, possibly varying by wavelength.
/// @param radiusSamples   Integration samples per mode.
/// @return The optics model.
OpticsModel MieAerosol(std::vector<LogNormalMode> modes, RefractiveIndex refractiveIndex,
                       int radiusSamples = 96);

/// @brief One constituent of an atmosphere.
struct Constituent {
    /// Its name.
    std::string name;
    /// Particles per m^3 where the profile is 1.
    double numberDensity = 0.0;
    /// Its vertical profile.
    DensityProfile density;
    /// Its optics per particle.
    OpticsModel optics;
    /// Refractivity n - 1 by wavelength in nm where the profile is 1,
    /// scaling with density (Gladstone-Dale); empty for constituents
    /// too sparse to bend light, such as aerosols and ozone.
    std::function<double(double)> refractivity;
};

/// @brief A spherical shell of constituents around a planet.
struct AtmosphereModel {
    /// The planet's radius, in metres.
    double bottomRadius = 6371e3;
    /// The shell's thickness, in metres.
    double topAltitude = 100e3;
    /// The constituents.
    std::vector<Constituent> constituents;

    /// @param wavelengthNm The wavelength, in nm.
    /// @return Extinction optical depth from the ground straight up.
    double VerticalOpticalDepth(double wavelengthNm) const;
};

/// @brief Earth's atmosphere: an isothermal dry-air column, ozone in
/// a tent profile, and haze.
struct EarthAtmosphereOptions {
    /// Sea-level pressure, in Pa; with the temperature, gives the
    /// molecules' number density p / (k T).
    double surfacePressure = 101325.0;
    /// Sea-level temperature, in kelvin.
    double surfaceTemperature = 288.15;
    /// Surface gravity, in m/s^2.
    double gravity = 9.80665;
    /// The CO2 volume fraction.
    double co2Fraction = 360e-6;
    /// Total column ozone in Dobson units, spread over a 10-40 km
    /// tent peaking at 25 km as in Bruneton's model.
    double ozoneColumn = 300.0;
    /// Vertical aerosol optical depth at 550 nm.
    double aerosolOpticalDepth = 0.1;
    /// The aerosol scale height, in metres.
    double aerosolScaleHeight = 1200.0;
};

/// @brief Builds Earth's atmosphere.
/// The haze is Schneegans et al.'s bimodal fit to AERONET, index
/// 1.4 + 0.015i.
/// @param radius  The planet's radius, in metres.
/// @param options The atmosphere's settings.
/// @return The model.
///
/// References:
/// - NOAA, NASA and USAF, "U.S. Standard Atmosphere, 1976", NOAA-S/T
///   76-1562, 1976. https://ntrs.nasa.gov/citations/19770009539
/// - E. Bruneton, "Precomputed Atmospheric Scattering: a New
///   Implementation", 2017 (ozone profile and column).
///   https://ebruneton.github.io/precomputed_atmospheric_scattering/
/// - Schneegans et al. 2024 (see `DensityProfile`), section 3 (haze).
/// - B. N. Holben et al., "AERONET - A Federated Instrument Network
///   and Data Archive for Aerosol Characterization", Remote Sensing
///   of Environment 66(1):1-16, 1998.
///   https://doi.org/10.1016/S0034-4257(98)00031-5
AtmosphereModel EarthAtmosphere(double radius, const EarthAtmosphereOptions& options = {});

/// @brief Mars's atmosphere: an isothermal CO2 column and dust.
struct MarsAtmosphereOptions {
    /// Mean surface pressure, in Pa.
    double surfacePressure = 610.0;
    /// Mean surface temperature, in kelvin.
    double surfaceTemperature = 210.0;
    /// Surface gravity, in m/s^2.
    double gravity = 3.71;
    /// The CO2 volume fraction, after Curiosity's SAM.
    double co2Fraction = 0.951;
    /// Mean molar mass, in kg/mol.
    double molarMass = 43.42e-3;
    /// Vertical dust optical depth at 880 nm, the rovers' reference
    /// (clear seasons 0.3-0.9, dust storms above 1).
    double dustOpticalDepth = 0.5;
};

/// @brief Builds Mars's atmosphere.
/// Dust is feldspar (n = 1.5) with 3% hematite by Maxwell Garnett
/// (Ehlers et al.), Schneegans et al.'s bimodal sizes (0.04 and 1.4
/// um, 1000:1) in two double-exponential layers.
/// @param radius  The planet's radius, in metres.
/// @param options The atmosphere's settings.
/// @return The model.
///
/// References:
/// - H. B. Franz et al., "Initial SAM calibration gas experiments on
///   Mars: Quadrupole mass spectrometer results and implications",
///   Planetary and Space Science 138:44-54, 2017.
///   https://doi.org/10.1016/j.pss.2017.01.014
/// - M. T. Lemmon et al., "Dust aerosol, clouds, and the atmospheric
///   optical depth record over 5 Mars years of the Mars Exploration
///   Rover mission", Icarus 251:96-111, 2015.
///   https://doi.org/10.1016/j.icarus.2014.03.029
/// - K. Ehlers, R. Chakrabarty and H. Moosmueller, "Blue moons and
///   Martian sunsets", Applied Optics 53(9):1808-1819, 2014.
///   https://doi.org/10.1364/AO.53.001808
/// - Schneegans et al. 2024 (see `DensityProfile`), section 3 and
///   table 1.
AtmosphereModel MarsAtmosphere(double radius, const MarsAtmosphereOptions& options = {});

/// @brief Scattering angles for tabulated phase functions.
/// theta_i = pi (i / (count - 1))^2, dense near forward where
/// diffraction peaks are narrow.
/// @param count The number of angles.
/// @return The angles' cosines.
std::vector<double> PhaseCosines(int count);

/// @brief A constituent averaged over a camera's bands.
struct BandedConstituent {
    /// Its name.
    std::string name;
    /// Its vertical profile.
    DensityProfile density;
    /// Scattering coefficient per band where the profile is 1, in
    /// m^-1.
    std::vector<float> scattering;
    /// Absorption coefficient per band where the profile is 1, in
    /// m^-1.
    std::vector<float> absorption;
    /// Phase function per band at `PhaseCosines`, band-major:
    /// phase[band * angles + i].
    std::vector<float> phase;
    /// Refractivity n - 1 per band where the profile is 1; zeros
    /// without one.
    std::vector<float> refractivity;
};

/// @brief An atmosphere averaged over a camera's bands, ready to
/// render.
struct BandedAtmosphere {
    /// The planet's radius, in metres.
    double bottomRadius = 0.0;
    /// The shell's thickness, in metres.
    double topAltitude = 0.0;
    /// The bands averaged over.
    SpectralBands bands;
    /// Phase function samples (`PhaseCosines`).
    int phaseAngles = 0;
    /// The constituents.
    std::vector<BandedConstituent> constituents;

    /// @param altitude The altitude, in metres.
    /// @param band     The band.
    /// @return The extinction coefficient, in m^-1.
    double Extinction(double altitude, int band) const;

    /// @brief Transmittance to the top along a ray above the
    /// horizon, by the midpoint rule.
    /// The reference the renderer's transmittance table is tested
    /// against.
    /// @param r     The start radius, in metres.
    /// @param mu    The ray's cosine from the zenith.
    /// @param band  The band.
    /// @param steps Integration steps.
    /// @return The transmittance.
    double TransmittanceToTop(double r, double mu, int band, int steps = 4000) const;

    /// @param altitude The altitude, in metres.
    /// @param band     The band.
    /// @return 1 plus each constituent's refractivity times density.
    double RefractiveIndex(double altitude, int band) const;

    /// @brief Where a refracted ray ends (`Refract`).
    struct Refraction {
        /// Angle between the first and last directions, in radians:
        /// its bending towards the ground.
        double deviation = 0.0;
        /// Whether it reached the ground rather than the top.
        bool ground = false;
    };
    /// @brief Traces a refracted ray out of the atmosphere by the ray
    /// equation d(n t)/ds = grad n.
    /// @param r    The start radius, in metres.
    /// @param mu   The apparent zenith cosine.
    /// @param band The band.
    /// @param step The largest step, in metres.
    /// @return The ray's deviation and where it ended.
    Refraction Refract(double r, double mu, int band, double step = 100.0) const;
};

/// @brief Averages an atmosphere over a camera's bands.
/// Samples each band at 10 nm or finer; phase functions are weighted
/// by scattering to preserve scattered radiance.
/// @param atmosphere  The model.
/// @param bands       The bands.
/// @param phaseAngles Phase function samples (`PhaseCosines`).
/// @return The banded atmosphere.
BandedAtmosphere BandAverage(const AtmosphereModel& atmosphere, const SpectralBands& bands,
                             int phaseAngles = 256);

}  // namespace ernie
