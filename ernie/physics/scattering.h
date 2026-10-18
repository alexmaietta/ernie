// SPDX-License-Identifier: BSD-3-Clause
/// @file scattering.h
/// @brief Rayleigh and Mie scattering and absorption cross sections.
/// Wavelengths are vacuum nm, radii m, cross sections m^2. Phase
/// functions are in sr^-1, integrate to 1 and take the scattering
/// angle's cosine (1 is forward).
#pragma once

#include <complex>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// The isotropic phase function, 1 / (4 pi) sr^-1: light scattered equally in every direction.
inline constexpr double kIsotropicPhase = 1.0 / (4.0 * std::numbers::pi);

/// Ideal gas number density at 288.15 K and 101325 Pa, in m^-3: the
/// reference state of the refractivities below.
///
/// References:
/// - B. A. Bodhaine, N. B. Wood, E. G. Dutton and J. R. Slusser, "On
///   Rayleigh Optical Depth Calculations", Journal of Atmospheric and
///   Oceanic Technology 16(11):1854-1861, 1999, eq. (24).
///   https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
inline constexpr double kStandardNumberDensity = 2.546899e25;

/// @brief Dry air's refractivity at 288.15 K and 101325 Pa.
/// Peck and Reeder's formula with Bodhaine et al.'s CO2 correction.
/// @param wavelengthNm The wavelength, in nm.
/// @param co2Fraction  The CO2 volume fraction.
/// @return n - 1.
///
/// References:
/// - E. R. Peck and K. Reeder, "Dispersion of Air", Journal of the
///   Optical Society of America 62(8):958-962, 1972.
///   https://doi.org/10.1364/JOSA.62.000958
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eqs. (18)
///   and (19).
double AirRefractivity(double wavelengthNm, double co2Fraction = 360e-6);

/// @brief CO2's refractivity at 288.15 K and 101325 Pa.
/// @param wavelengthNm The wavelength, in nm.
/// @return n - 1.
///
/// References:
/// - J. C. Owens, "Optical Refractive Index of Air: Dependence on
///   Pressure, Temperature and Composition", Applied Optics
///   6(1):51-59, 1967. https://doi.org/10.1364/AO.6.000051
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eq. (27).
double CarbonDioxideRefractivity(double wavelengthNm);

/// @brief Dry air's King factor for molecular anisotropy.
/// Bates's N2 and O2 factors, 1 for Ar and 1.15 for CO2, weighted
/// by volume fraction.
/// @param wavelengthNm The wavelength, in nm.
/// @param co2Fraction  The CO2 volume fraction.
/// @return (6 + 3 rho) / (6 - 7 rho).
///
/// References:
/// - D. R. Bates, "Rayleigh scattering by air", Planetary and Space
///   Science 32(6):785-790, 1984.
///   https://doi.org/10.1016/0032-0633(84)90102-8
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eqs. (5),
///   (6) and (23).
double AirKingFactor(double wavelengthNm, double co2Fraction = 360e-6);

/// CO2's King factor (Bates 1984, as used by Bodhaine et al. 1999).
inline constexpr double kCarbonDioxideKingFactor = 1.15;

/// @brief Rayleigh cross section of one molecule.
/// 24 pi^3 (n^2 - 1)^2 / (lambda^4 Ns^2 (n^2 + 2)^2) F; independent
/// of density since n - 1 scales with it.
/// @param wavelengthNm The wavelength, in nm.
/// @param refractivity n - 1 at `kStandardNumberDensity`.
/// @param kingFactor   The King factor F.
/// @return The cross section, in m^2.
///
/// References:
/// - Bodhaine et al. 1999 (see `kStandardNumberDensity`), eq. (22).
double RayleighCrossSection(double wavelengthNm, double refractivity, double kingFactor);

/// @brief The Rayleigh phase function with depolarisation.
/// 3 / (16 pi) ((1 + 3 g) + (1 - g) mu^2) / (1 + 2 g), with
/// g = rho / (2 - rho) and rho the depolarisation ratio.
/// @param cosTheta   The scattering angle's cosine.
/// @param kingFactor The King factor; 1 for isotropic molecules.
/// @return The phase function, in sr^-1.
///
/// References:
/// - S. Chandrasekhar, "Radiative Transfer", Oxford University Press,
///   1950, chapter I.
/// - A. Bucholtz, "Rayleigh-scattering calculations for the
///   terrestrial atmosphere", Applied Optics 34(15):2765-2773, 1995,
///   eqs. (12)-(13). https://doi.org/10.1364/AO.34.002765
double RayleighPhase(double cosTheta, double kingFactor);

/// @brief Mie efficiencies of a homogeneous sphere.
/// Efficiencies are cross sections over pi r^2.
struct MieEfficiencies {
    /// Extinction efficiency.
    double extinction = 0.0;
    /// Scattering efficiency.
    double scattering = 0.0;
    /// Asymmetry parameter g, the mean scattering cosine.
    double asymmetry = 0.0;

    /// @return The absorption efficiency, at least 0.
    double Absorption() const {
        // Rounding can leave a non-absorbing sphere slightly
        // negative.
        return extinction > scattering ? extinction - scattering : 0.0;
    }
};

/// @brief Mie scattering by a homogeneous sphere.
/// Sums the series to Wiscombe's number of terms.
/// @param sizeParameter   2 pi r / lambda.
/// @param refractiveIndex Relative index n + ik, k >= 0 absorbing.
/// @param cosines         Cosines to evaluate the phase function at.
/// @param phase           Receives the unpolarised phase function at
///                        `cosines`; empty to skip.
/// @return The efficiencies.
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
MieEfficiencies MieSphere(double sizeParameter, std::complex<double> refractiveIndex,
                          std::span<const double> cosines, std::span<double> phase);

/// @brief One mode of a log-normal particle size distribution.
/// Particles per unit ln(r) are Gaussian.
///
/// References:
/// - J. E. Hansen and L. D. Travis, "Light scattering in planetary
///   atmospheres", Space Science Reviews 16(4):527-610, 1974,
///   section 2.4 (effective radius and variance).
///   https://doi.org/10.1007/BF00168069
struct LogNormalMode {
    /// Median radius, in metres.
    double medianRadius = 1e-6;
    /// Standard deviation of ln(r).
    double sigma = 0.5;
    /// Particle count relative to the other modes.
    double relativeNumber = 1.0;

    /// @brief A mode from its effective radius and variance.
    /// sigma^2 = ln(1 + v_eff); r_median = r_eff exp(-2.5 sigma^2).
    /// @param effectiveRadius   Area-weighted mean radius, in metres.
    /// @param effectiveVariance The effective variance.
    /// @param relativeNumber    Count relative to other modes.
    /// @return The mode.
    static LogNormalMode FromEffective(double effectiveRadius, double effectiveVariance,
                                       double relativeNumber = 1.0);

    /// @return The effective radius, in metres.
    double EffectiveRadius() const;
    /// @return The effective variance.
    double EffectiveVariance() const;
};

/// @brief A complex refractive index tabulated by wavelength.
/// Linear between samples, constant beyond the ends.
class RefractiveIndex {
  public:
    RefractiveIndex() = default;

    /// @brief A constant index.
    /// @param value The index at every wavelength.
    explicit RefractiveIndex(std::complex<double> value);

    /// @brief Tabulates an index.
    /// @param wavelengthsNm Strictly increasing wavelengths, in nm.
    /// @param values        The index at each.
    /// @throws std::invalid_argument If the wavelengths are not
    ///         strictly increasing or the sizes differ.
    RefractiveIndex(std::vector<double> wavelengthsNm, std::vector<std::complex<double>> values);

    /// @param wavelengthNm The wavelength, in nm.
    /// @return The interpolated index.
    std::complex<double> operator()(double wavelengthNm) const;

  private:
    std::vector<double> wavelengths_;
    std::vector<std::complex<double>> values_;
};

/// @brief Effective index of a matrix with spherical inclusions, by
/// the Maxwell Garnett rule on permittivities m^2.
/// @param matrix    The matrix's index.
/// @param inclusion The inclusions' index.
/// @param fraction  The inclusions' volume fraction.
/// @return The effective index.
///
/// References:
/// - J. C. Maxwell Garnett, "Colours in Metal Glasses and in Metallic
///   Films", Philosophical Transactions of the Royal Society A
///   203:385-420, 1904. https://doi.org/10.1098/rsta.1904.0024
/// - Bohren and Huffman 1983 (see `MieSphere`), section 8.5.
std::complex<double> MaxwellGarnett(std::complex<double> matrix, std::complex<double> inclusion,
                                    double fraction);

/// @brief A particle population's mean optics per particle at one
/// wavelength.
struct ParticleOptics {
    /// Scattering cross section, in m^2.
    double scattering = 0.0;
    /// Absorption cross section, in m^2.
    double absorption = 0.0;
    /// Phase function at the requested cosines, in sr^-1.
    std::vector<double> phase;
};

/// @brief Mean Mie optics of a multi-mode log-normal population.
/// Each mode is integrated over ln(r) by trapezoids, which unlike
/// Monte Carlo keeps tables smooth across wavelengths. The phase
/// function is weighted by scattering cross section.
/// @param modes           The size distribution's modes.
/// @param refractiveIndex The particles' index.
/// @param wavelengthNm    The wavelength, in nm.
/// @param cosines         Cosines to evaluate the phase function at.
/// @param radiusSamples   Trapezoid samples per mode.
/// @return The mean optics per particle.
/// @throws std::invalid_argument If there are no modes or their
///         numbers do not sum above zero.
///
/// References:
/// - Hansen and Travis 1974 (see `LogNormalMode`), section 2.
ParticleOptics MieParticles(std::span<const LogNormalMode> modes,
                            std::complex<double> refractiveIndex, double wavelengthNm,
                            std::span<const double> cosines, int radiusSamples = 96);

/// @brief Ozone's absorption cross section at 233 K.
/// 10 nm averages of the IUP Bremen data over 360-840 nm (Chappuis
/// band included), as tabulated by CosmoScout VR.
/// @param wavelengthNm The wavelength, in nm.
/// @return The cross section in m^2; 0 outside 360-840 nm.
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

/// @brief Hematite's (alpha-Fe2O3) refractive index, 220-1000 nm.
/// As tabulated by CosmoScout VR (CC0) after Ehlers et al.; mixed
/// with feldspar for Martian dust.
/// @return The tabulated index.
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
