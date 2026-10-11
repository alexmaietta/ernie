// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <complex>
#include <span>
#include <utility>
#include <vector>

/// Light scattering by molecules and particles, for participating
/// media such as atmospheres: Rayleigh scattering by gases, Mie
/// scattering by spheres and particle size distributions, and
/// tabulated absorption cross sections. Wavelengths are in nanometres
/// in vacuum, radii in metres and cross sections in m^2. Phase
/// functions are in sr^-1, normalised to integrate to 1 over the
/// sphere, and take the cosine of the scattering angle (1 is
/// forward).
namespace ernie {

/// Number density of an ideal gas at 288.15 K and 101325 Pa (m^-3),
/// the reference state of the refractive indices below: Avogadro's
/// number over the molar volume at 273.15 K, scaled to 288.15 K.
///
/// Reference: B. A. Bodhaine, N. B. Wood, E. G. Dutton and J. R.
/// Slusser, "On Rayleigh Optical Depth Calculations", Journal of
/// Atmospheric and Oceanic Technology 16(11):1854-1861, 1999, eq.
/// (24).
/// https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
inline constexpr double kStandardNumberDensity = 2.546899e25;

/// The refractivity n - 1 of dry air at 288.15 K and 101325 Pa with a
/// CO2 volume fraction `co2Fraction` (e.g. 360e-6), from Peck and
/// Reeder's dispersion formula for air with 300 ppm CO2 and Bodhaine
/// et al.'s CO2 correction.
///
/// References:
/// - E. R. Peck and K. Reeder, "Dispersion of Air", Journal of the
///   Optical Society of America 62(8):958-962, 1972.
///   https://doi.org/10.1364/JOSA.62.000958
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eqs. (18)
///   and (19).
double AirRefractivity(double wavelengthNm,
                       double co2Fraction = 360e-6);

/// The refractivity n - 1 of CO2 at 288.15 K and 101325 Pa.
///
/// References:
/// - J. C. Owens, "Optical Refractive Index of Air: Dependence on
///   Pressure, Temperature and Composition", Applied Optics
///   6(1):51-59, 1967. https://doi.org/10.1364/AO.6.000051
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eq. (27).
double CarbonDioxideRefractivity(double wavelengthNm);

/// The King correction factor (6 + 3 rho) / (6 - 7 rho) of dry air
/// for molecular anisotropy, from Bates's factors for N2 and O2, 1
/// for Ar and 1.15 for CO2, weighted by volume fraction.
///
/// References:
/// - D. R. Bates, "Rayleigh scattering by air", Planetary and Space
///   Science 32(6):785-790, 1984.
///   https://doi.org/10.1016/0032-0633(84)90102-8
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eqs. (5),
///   (6) and (23).
double AirKingFactor(double wavelengthNm,
                     double co2Fraction = 360e-6);

/// The King factor of CO2 (Bates 1984, as used by Bodhaine et al.
/// 1999).
inline constexpr double kCarbonDioxideKingFactor = 1.15;

/// The Rayleigh scattering cross section (m^2) of one molecule of a
/// gas with refractivity `refractivity` at `kStandardNumberDensity`
/// and King factor `kingFactor`: 24 pi^3 (n^2 - 1)^2 / (lambda^4 Ns^2
/// (n^2 + 2)^2) F. It does not depend on the gas's density, as n - 1
/// is proportional to it.
///
/// Reference: Bodhaine et al. 1999 (see `kStandardNumberDensity`),
/// eq. (22).
double RayleighCrossSection(double wavelengthNm, double refractivity,
                            double kingFactor);

/// The Rayleigh phase function with depolarisation, for the King
/// factor `kingFactor` (1 for isotropic molecules): 3 / (16 pi) ((1 +
/// 3 gamma) + (1 - gamma) mu^2) / (1 + 2 gamma) with gamma = rho / (2
/// - rho) and rho the depolarisation ratio.
///
/// References:
/// - S. Chandrasekhar, "Radiative Transfer", Oxford University Press,
///   1950, chapter I.
/// - A. Bucholtz, "Rayleigh-scattering calculations for the
///   terrestrial atmosphere", Applied Optics 34(15):2765-2773, 1995,
///   eqs. (12)-(13). https://doi.org/10.1364/AO.34.002765
double RayleighPhase(double cosTheta, double kingFactor);

/// Mie efficiencies of a homogeneous sphere: cross sections over the
/// geometric cross section pi r^2, and the asymmetry parameter g (the
/// mean cosine of the scattering angle).
struct MieEfficiencies {
    double extinction = 0.0;
    double scattering = 0.0;
    double asymmetry = 0.0;

    double Absorption() const {
        // Rounding can leave a non-absorbing sphere slightly
        // negative.
        return extinction > scattering ? extinction - scattering
                                       : 0.0;
    }
};

/// Mie scattering by a homogeneous sphere of size parameter
/// `sizeParameter` (2 pi r / lambda) and relative refractive index
/// `refractiveIndex` (n + ik with k >= 0 for absorption), with its
/// unpolarised phase function at `cosines` written to `phase` (same
/// size, or empty to skip), summing the series to Wiscombe's number
/// of terms.
///
/// References:
/// - G. Mie, "Beitraege zur Optik trueber Medien, speziell
///   kolloidaler Metalloesungen", Annalen der Physik 330(3):377-445,
///   1908. https://doi.org/10.1002/andp.19083300302
/// - C. F. Bohren and D. R. Huffman, "Absorption and Scattering of
///   Light by Small Particles", Wiley, 1983, chapter 4 and appendix A
///   (BHMIE). https://doi.org/10.1002/9783527618156
/// - W. J. Lentz, "Generating Bessel functions in Mie scattering
///   calculations using continued fractions", Applied Optics
///   15(3):668-671, 1976. https://doi.org/10.1364/AO.15.000668
/// - W. J. Wiscombe, "Improved Mie scattering algorithms", Applied
///   Optics 19(9):1505-1509, 1980.
///   https://doi.org/10.1364/AO.19.001505
MieEfficiencies MieSphere(double sizeParameter,
                          std::complex<double> refractiveIndex,
                          std::span<const double> cosines,
                          std::span<double> phase);

/// One mode of a log-normal particle size distribution: the number of
/// particles per unit ln(r) is Gaussian with median radius
/// `medianRadius` (m) and standard deviation `sigma`, and the mode
/// holds `relativeNumber` particles relative to the other modes.
///
/// Reference: J. E. Hansen and L. D. Travis, "Light scattering in
/// planetary atmospheres", Space Science Reviews 16(4):527-610, 1974,
/// section 2.4 (effective radius and variance).
/// https://doi.org/10.1007/BF00168069
struct LogNormalMode {
    double medianRadius = 1e-6;
    double sigma = 0.5;
    double relativeNumber = 1.0;

    /// The mode with effective (area-weighted mean) radius
    /// `effectiveRadius` (m) and effective variance
    /// `effectiveVariance`: sigma^2 = ln(1 + v_eff), r_median = r_eff
    /// exp(-2.5 sigma^2).
    static LogNormalMode FromEffective(double effectiveRadius,
                                       double effectiveVariance,
                                       double relativeNumber = 1.0);

    double EffectiveRadius() const;
    double EffectiveVariance() const;
};

/// A complex refractive index tabulated at increasing wavelengths
/// (nm), linearly interpolated and constant beyond its ends.
class RefractiveIndex {
  public:
    RefractiveIndex() = default;

    /// A refractive index equal to `value` at every wavelength.
    explicit RefractiveIndex(std::complex<double> value);

    /// @throws std::invalid_argument if the wavelengths are not
    /// strictly increasing or the sizes differ.
    RefractiveIndex(std::vector<double> wavelengthsNm,
                    std::vector<std::complex<double>> values);

    std::complex<double> operator()(double wavelengthNm) const;

  private:
    std::vector<double> wavelengths_;
    std::vector<std::complex<double>> values_;
};

/// The effective refractive index of a matrix with spherical
/// inclusions taking the volume fraction `fraction`, by the Maxwell
/// Garnett mixing rule on the permittivities m^2.
///
/// References:
/// - J. C. Maxwell Garnett, "Colours in Metal Glasses and in Metallic
///   Films", Philosophical Transactions of the Royal Society A
///   203:385-420, 1904. https://doi.org/10.1098/rsta.1904.0024
/// - Bohren and Huffman 1983 (see `MieSphere`), section 8.5.
std::complex<double> MaxwellGarnett(std::complex<double> matrix,
                                    std::complex<double> inclusion,
                                    double fraction);

/// Mean optical properties per particle of a particle population at
/// one wavelength: scattering and absorption cross sections (m^2) and
/// the phase function at the requested cosines.
struct ParticleOptics {
    double scattering = 0.0;
    double absorption = 0.0;
    std::vector<double> phase;
};

/// The mean Mie optics per particle of spheres of refractive index
/// `refractiveIndex` with a multi-mode log-normal size distribution,
/// at `wavelengthNm`: each mode integrated over ln(r) with
/// `radiusSamples` trapezoid samples, the phase function weighted by
/// scattering cross section. Integrating deterministically, unlike
/// Monte Carlo radius sampling, gives tables that are smooth across
/// wavelengths.
///
/// Reference: Hansen and Travis 1974 (see `LogNormalMode`), section
/// 2.
/// @throws std::invalid_argument if there are no modes or their
/// numbers do not sum above zero.
ParticleOptics MieParticles(std::span<const LogNormalMode> modes,
                            std::complex<double> refractiveIndex,
                            double wavelengthNm,
                            std::span<const double> cosines,
                            int radiusSamples = 96);

/// The absorption cross section of ozone (m^2) at 233 K, the 10 nm
/// averages of the IUP Bremen measurements over 360-840 nm (including
/// the Chappuis band) as tabulated by CosmoScout VR's
/// scattering-table generator, placed at the bin centres. Zero
/// outside 360-840 nm.
///
/// References:
/// - V. Gorshelev, A. Serdyuchenko, M. Weber, W. Chehade and J. P.
///   Burrows, "High spectral resolution ozone absorption
///   cross-sections - Part 1: Measurements, data analysis and
///   comparison with previous measurements around 293 K", Atmospheric
///   Measurement Techniques 7(2):609-624, 2014.
///   https://doi.org/10.5194/amt-7-609-2014
/// - A. Serdyuchenko, V. Gorshelev, M. Weber, W. Chehade and J. P.
///   Burrows, "High spectral resolution ozone absorption
///   cross-sections - Part 2: Temperature dependence", Atmospheric
///   Measurement Techniques 7(2):625-636, 2014.
///   https://doi.org/10.5194/amt-7-625-2014
/// - CosmoScout VR,
///   plugins/csp-atmospheres/scattering-table-generator/ozoneMode.cpp
///   (MIT), German Aerospace Center (DLR).
///   https://github.com/cosmoscout/cosmoscout-vr
double OzoneCrossSection(double wavelengthNm);

/// The refractive index of hematite (alpha-Fe2O3) over 220-1000 nm,
/// as tabulated by CosmoScout VR (CC0) after Ehlers et al., used with
/// feldspar for Martian dust.
///
/// References:
/// - K. Ehlers, R. Chakrabarty and H. Moosmueller, "Blue moons and
///   Martian sunsets", Applied Optics 53(9):1808-1819, 2014.
///   https://doi.org/10.1364/AO.53.001808
/// - S. Schneegans, T. Meyran, I. Ginkel, G. Zachmann and A. Gerndt,
///   "Physically Based Real-Time Rendering of Atmospheres using Mie
///   Theory", Computer Graphics Forum 43(2), 2024.
///   https://doi.org/10.1111/cgf.15010
const RefractiveIndex& HematiteRefractiveIndex();

}  // namespace ernie
