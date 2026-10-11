// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/scattering.h"
#include "ernie/util/spectrum.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

/// A planet's atmosphere as constituents (gases, aerosols, absorbers)
/// with physical optical properties and vertical density profiles,
/// and its reduction to a camera's spectral bands. Altitudes and
/// radii are in metres, number densities in m^-3, cross sections in
/// m^2 and coefficients in m^-1.
namespace ernie {

/// One term of a vertical density profile, relative to the profile's
/// reference density.
struct DensityLayer {
    enum class Shape {
        /// exp(-h / height): a gas in hydrostatic equilibrium at
        /// constant temperature.
        Exponential,
        /// exp(1 - exp(h / height)), 1 at the ground: dust that falls
        /// off faster than exponentially, after Conrath's mixing
        /// ratio model as used for Mars by Schneegans et al. 2024.
        DoubleExponential,
        /// 1 - |h - height| / width, clamped at 0: a layer peaking at
        /// `height`.
        Tent,
    };

    Shape shape = Shape::Exponential;
    double height = 8000.0;
    double width = 0.0;
    double weight = 1.0;
};

/// A vertical density profile: the weighted mean of its layers, so 1
/// at the ground for exponential and double exponential layers.
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
    std::vector<DensityLayer> layers;

    static DensityProfile Exponential(double scaleHeight);
    static DensityProfile Tent(double peakAltitude, double halfWidth);

    /// The relative density at `altitude`; 0 for a profile without
    /// layers.
    double operator()(double altitude) const;

    /// The integral of the relative density from the ground to `top`
    /// (m).
    double Column(double top) const;
};

/// Optical properties of one particle or molecule at a wavelength
/// (nm), with the phase function at the given cosines.
using OpticsModel = std::function<ParticleOptics(
    double wavelengthNm, std::span<const double>)>;

/// Molecules of dry air: Rayleigh scattering after Bodhaine et al.
/// (1999) with a CO2 volume fraction `co2Fraction`.
OpticsModel AirMolecules(double co2Fraction = 360e-6);

/// Molecules of a CO2 atmosphere such as Mars's: Rayleigh scattering
/// by CO2 at volume fraction `co2Fraction` (Owens 1967, Bates 1984),
/// with the remainder (on Mars mostly N2 and Ar, whose refractivities
/// are within 7% of air's) scattering as air.
OpticsModel CarbonDioxideMolecules(double co2Fraction);

/// Ozone molecules: absorption only (`OzoneCrossSection`).
OpticsModel OzoneMolecules();

/// Spherical particles with a log-normal size distribution and a
/// refractive index that may vary with wavelength (`MieParticles`).
OpticsModel MieAerosol(std::vector<LogNormalMode> modes,
                       RefractiveIndex refractiveIndex,
                       int radiusSamples = 96);

/// One constituent of an atmosphere: `numberDensity` particles per
/// m^3 where its profile is 1.
struct Constituent {
    std::string name;
    double numberDensity = 0.0;
    DensityProfile density;
    OpticsModel optics;
};

/// A spherical shell of constituents around a planet of radius
/// `bottomRadius`, up to `bottomRadius + topAltitude`.
struct Atmosphere {
    double bottomRadius = 6371e3;
    double topAltitude = 100e3;
    std::vector<Constituent> constituents;

    /// The extinction optical depth from the ground straight up to
    /// the top at `wavelengthNm`.
    double VerticalOpticalDepth(double wavelengthNm) const;
};

/// Earth's atmosphere: an isothermal dry-air column, ozone in a tent
/// profile and a haze aerosol.
struct EarthAtmosphereOptions {
    /// Sea-level state and gravity of the U.S. Standard Atmosphere
    /// 1976, which give the molecules' number density p / (k T) and
    /// scale height R T / (M g).
    double surfacePressure = 101325.0;
    double surfaceTemperature = 288.15;
    double gravity = 9.80665;
    double co2Fraction = 360e-6;
    /// Total column ozone in Dobson units (2.6867e20 molecules m^-2),
    /// spread over a tent from 10 to 40 km peaking at 25 km as in
    /// Bruneton's model.
    double ozoneColumn = 300.0;
    /// Vertical aerosol optical depth at 550 nm, and the aerosol
    /// scale height.
    double aerosolOpticalDepth = 0.1;
    double aerosolScaleHeight = 1200.0;
};

/// Earth's atmosphere over a planet of radius `radius` (m).
///
/// The haze is Schneegans et al.'s bimodal log-normal fit to AERONET
/// retrievals, with refractive index 1.4 + 0.015i.
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
Atmosphere
EarthAtmosphere(double radius,
                const EarthAtmosphereOptions& options = {});

/// Mars's atmosphere: an isothermal CO2 column and suspended dust.
struct MarsAtmosphereOptions {
    /// Mean surface state; composition after Curiosity's SAM (95.1%
    /// CO2, molar mass 43.4 g/mol).
    double surfacePressure = 610.0;
    double surfaceTemperature = 210.0;
    double gravity = 3.71;
    double co2Fraction = 0.951;
    double molarMass = 43.42e-3;
    /// Vertical dust optical depth at 880 nm, the reference
    /// wavelength of the Mars Exploration Rovers' record (clear
    /// seasons about 0.3 to 0.9, dust storms above 1).
    double dustOpticalDepth = 0.5;
};

/// Mars's atmosphere over a planet of radius `radius` (m).
///
/// The dust is feldspar (n = 1.5) with 3% hematite by Maxwell Garnett
/// mixing, as proposed by Ehlers et al., with Schneegans et al.'s
/// bimodal log-normal sizes (effective radii 0.04 and 1.4 um, number
/// ratio 1000:1, after Shaposhnikov et al.) and two
/// double-exponential layers.
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
Atmosphere MarsAtmosphere(double radius,
                          const MarsAtmosphereOptions& options = {});

/// The scattering angles of tabulated phase functions: theta_i = pi
/// (i / (count - 1))^2, dense near the forward direction where
/// aerosols' diffraction peaks are narrow. Returns the cosines.
std::vector<double> PhaseCosines(int count);

/// A constituent averaged over a camera's spectral bands.
struct BandedConstituent {
    std::string name;
    DensityProfile density;
    /// Scattering and absorption coefficients (m^-1) where the
    /// density profile is 1, per band.
    std::vector<float> scattering;
    std::vector<float> absorption;
    /// The phase function per band at `PhaseCosines`, band-major:
    /// phase[band * angles + i].
    std::vector<float> phase;
};

/// An atmosphere averaged over a camera's spectral bands, ready for
/// rendering.
struct BandedAtmosphere {
    double bottomRadius = 0.0;
    double topAltitude = 0.0;
    SpectralBands bands;
    int phaseAngles = 0;
    std::vector<BandedConstituent> constituents;
};

/// Averages `atmosphere` over `bands`, sampling each band every 10 nm
/// or less. Phase functions are averaged weighted by scattering, so
/// that the band's scattered radiance is preserved.
BandedAtmosphere BandAverage(const Atmosphere& atmosphere,
                             const SpectralBands& bands,
                             int phaseAngles = 256);

}  // namespace ernie
