// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/atmosphere.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kIsotropic = 1.0 / (4.0 * kPi);

/// The molar gas constant (J mol^-1 K^-1, SI 2019 exact) and one
/// Dobson unit (molecules m^-2).
constexpr double kGasConstant = 8.314462618;
constexpr double kDobsonUnit = 2.6867e20;

/// The molar mass of dry air (kg/mol) in the U.S. Standard Atmosphere
/// 1976.
constexpr double kAirMolarMass = 28.9644e-3;

/// Intervals of the Simpson rule integrating density profiles.
constexpr int kColumnIntervals = 4000;

/// The longest spacing between samples when averaging over a band
/// (nm).
constexpr double kBandSampleNm = 10.0;

double LayerDensity(const DensityLayer& layer, double altitude) {
    switch (layer.shape) {
    case DensityLayer::Shape::Exponential:
        return std::exp(-altitude / layer.height);
    case DensityLayer::Shape::DoubleExponential:
        return std::exp(1.0 - std::exp(altitude / layer.height));
    case DensityLayer::Shape::Tent:
        return std::max(0.0, 1.0 - std::abs(altitude - layer.height) /
                                       layer.width);
    }
    return 0.0;
}

/// An isothermal gas column: number density p / (k T) at the ground
/// and scale height R T / (M g).
std::pair<double, DensityProfile> IsothermalGas(double pressure,
                                                double temperature,
                                                double molarMass,
                                                double gravity) {
    const double numberDensity =
        pressure / (kBoltzmann * temperature);
    const double scaleHeight =
        kGasConstant * temperature / (molarMass * gravity);
    return {numberDensity, DensityProfile::Exponential(scaleHeight)};
}

/// The number density that gives `profile` the vertical optical depth
/// `opticalDepth` at `wavelengthNm` up to `top`.
double NumberDensityForDepth(const OpticsModel& optics,
                             const DensityProfile& profile,
                             double top, double wavelengthNm,
                             double opticalDepth) {
    const ParticleOptics sample = optics(wavelengthNm, {});
    const double extinction = sample.scattering + sample.absorption;
    const double column = profile.Column(top);
    if (!(extinction > 0.0) || !(column > 0.0)) {
        throw std::invalid_argument("An aerosol needs extinction and "
                                    "a column to set its depth");
    }
    return opticalDepth / (extinction * column);
}

}  // namespace

DensityProfile DensityProfile::Exponential(double scaleHeight) {
    return {
        {{DensityLayer::Shape::Exponential, scaleHeight, 0.0, 1.0}}};
}

DensityProfile DensityProfile::Tent(double peakAltitude,
                                    double halfWidth) {
    return {
        {{DensityLayer::Shape::Tent, peakAltitude, halfWidth, 1.0}}};
}

double DensityProfile::operator()(double altitude) const {
    double sum = 0.0;
    double weights = 0.0;
    for (const DensityLayer& layer : layers) {
        sum += layer.weight * LayerDensity(layer, altitude);
        weights += layer.weight;
    }
    return weights > 0.0 ? sum / weights : 0.0;
}

double DensityProfile::Column(double top) const {
    const double step = top / kColumnIntervals;
    double sum = (*this)(0.0) + (*this)(top);
    for (int i = 1; i < kColumnIntervals; ++i) {
        sum += (i % 2 == 1 ? 4.0 : 2.0) * (*this)(i * step);
    }
    return sum * step / 3.0;
}

OpticsModel AirMolecules(double co2Fraction) {
    return [co2Fraction](double wavelengthNm,
                         std::span<const double> cosines) {
        const double king = AirKingFactor(wavelengthNm, co2Fraction);
        ParticleOptics optics;
        optics.scattering = RayleighCrossSection(
            wavelengthNm, AirRefractivity(wavelengthNm, co2Fraction),
            king);
        for (const double mu : cosines) {
            optics.phase.push_back(RayleighPhase(mu, king));
        }
        return optics;
    };
}

OpticsModel CarbonDioxideMolecules(double co2Fraction) {
    return [co2Fraction](double wavelengthNm,
                         std::span<const double> cosines) {
        const double co2 =
            co2Fraction * RayleighCrossSection(
                              wavelengthNm,
                              CarbonDioxideRefractivity(wavelengthNm),
                              kCarbonDioxideKingFactor);
        const double airKing = AirKingFactor(wavelengthNm);
        const double rest =
            (1.0 - co2Fraction) *
            RayleighCrossSection(
                wavelengthNm, AirRefractivity(wavelengthNm), airKing);
        ParticleOptics optics;
        optics.scattering = co2 + rest;
        // Each gas scatters with its own depolarisation; the
        // mixture's phase function is their mean weighted by
        // scattering.
        for (const double mu : cosines) {
            optics.phase.push_back(
                (co2 * RayleighPhase(mu, kCarbonDioxideKingFactor) +
                 rest * RayleighPhase(mu, airKing)) /
                optics.scattering);
        }
        return optics;
    };
}

OpticsModel OzoneMolecules() {
    return [](double wavelengthNm, std::span<const double> cosines) {
        ParticleOptics optics;
        optics.absorption = OzoneCrossSection(wavelengthNm);
        optics.phase.assign(cosines.size(), kIsotropic);
        return optics;
    };
}

OpticsModel MieAerosol(std::vector<LogNormalMode> modes,
                       RefractiveIndex refractiveIndex,
                       int radiusSamples) {
    return [modes = std::move(modes),
            refractiveIndex = std::move(refractiveIndex),
            radiusSamples](double wavelengthNm,
                           std::span<const double> cosines) {
        return MieParticles(modes, refractiveIndex(wavelengthNm),
                            wavelengthNm, cosines, radiusSamples);
    };
}

double
AtmosphereModel::VerticalOpticalDepth(double wavelengthNm) const {
    double depth = 0.0;
    for (const Constituent& constituent : constituents) {
        const ParticleOptics optics =
            constituent.optics(wavelengthNm, {});
        depth += constituent.numberDensity *
                 (optics.scattering + optics.absorption) *
                 constituent.density.Column(topAltitude);
    }
    return depth;
}

AtmosphereModel
EarthAtmosphere(double radius,
                const EarthAtmosphereOptions& options) {
    AtmosphereModel atmosphere;
    atmosphere.bottomRadius = radius;
    atmosphere.topAltitude = 100e3;

    const auto [airDensity, airProfile] = IsothermalGas(
        options.surfacePressure, options.surfaceTemperature,
        kAirMolarMass, options.gravity);
    atmosphere.constituents.push_back(
        {"molecules", airDensity, airProfile,
         AirMolecules(options.co2Fraction)});

    // A tent from 10 to 40 km holds the column in a triangle of base
    // 30 km: peak * 15 km.
    const DensityProfile ozoneProfile =
        DensityProfile::Tent(25e3, 15e3);
    const double ozonePeak = options.ozoneColumn * kDobsonUnit / 15e3;
    atmosphere.constituents.push_back(
        {"ozone", ozonePeak, ozoneProfile, OzoneMolecules()});

    if (options.aerosolOpticalDepth > 0.0) {
        // Schneegans et al. 2024, CosmoScout VR
        // mie-settings/earth_haze.json (CC0): median radii
        // exp(-16.316) and exp(-13.616) m, sigma 0.6, 10000 small
        // particles per large one.
        OpticsModel haze =
            MieAerosol({{std::exp(-16.316), 0.6, 10000.0},
                        {std::exp(-13.616), 0.6, 1.0}},
                       RefractiveIndex({1.4, 0.015}));
        const DensityProfile hazeProfile =
            DensityProfile::Exponential(options.aerosolScaleHeight);
        const double hazeDensity = NumberDensityForDepth(
            haze, hazeProfile, atmosphere.topAltitude, 550.0,
            options.aerosolOpticalDepth);
        atmosphere.constituents.push_back(
            {"aerosols", hazeDensity, hazeProfile, std::move(haze)});
    }
    return atmosphere;
}

AtmosphereModel MarsAtmosphere(double radius,
                               const MarsAtmosphereOptions& options) {
    AtmosphereModel atmosphere;
    atmosphere.bottomRadius = radius;
    atmosphere.topAltitude = 100e3;

    const auto [gasDensity, gasProfile] = IsothermalGas(
        options.surfacePressure, options.surfaceTemperature,
        options.molarMass, options.gravity);
    atmosphere.constituents.push_back(
        {"molecules", gasDensity, gasProfile,
         CarbonDioxideMolecules(options.co2Fraction)});

    if (options.dustOpticalDepth > 0.0) {
        // Feldspar with 3% hematite (Ehlers et al. 2014), indexed
        // where the hematite table is.
        const RefractiveIndex& hematite = HematiteRefractiveIndex();
        std::vector<double> wavelengths;
        std::vector<std::complex<double>> values;
        for (double nm = 220.0; nm <= 1000.0; nm += 10.0) {
            wavelengths.push_back(nm);
            values.push_back(
                MaxwellGarnett({1.5, 0.0}, hematite(nm), 0.03));
        }
        OpticsModel dust = MieAerosol(
            {LogNormalMode::FromEffective(0.04e-6, 0.1, 1000.0),
             LogNormalMode::FromEffective(1.4e-6, 0.35, 1.0)},
            RefractiveIndex(std::move(wavelengths),
                            std::move(values)));
        // Dense low dust and thin high dust (Schneegans et al. 2024,
        // table 1).
        DensityProfile dustProfile;
        dustProfile.layers = {{DensityLayer::Shape::DoubleExponential,
                               4000.0, 0.0, 3.0},
                              {DensityLayer::Shape::DoubleExponential,
                               20000.0, 0.0, 1.0}};
        const double dustDensity = NumberDensityForDepth(
            dust, dustProfile, atmosphere.topAltitude, 880.0,
            options.dustOpticalDepth);
        atmosphere.constituents.push_back(
            {"dust", dustDensity, dustProfile, std::move(dust)});
    }
    return atmosphere;
}

std::vector<double> PhaseCosines(int count) {
    if (count < 2) {
        throw std::invalid_argument(
            "A phase table needs at least two angles");
    }
    std::vector<double> cosines(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double u = static_cast<double>(i) / (count - 1);
        cosines[static_cast<std::size_t>(i)] = std::cos(kPi * u * u);
    }
    return cosines;
}

double BandedAtmosphere::Extinction(double altitude, int band) const {
    const auto b = static_cast<std::size_t>(band);
    double extinction = 0.0;
    for (const BandedConstituent& constituent : constituents) {
        extinction +=
            (static_cast<double>(constituent.scattering[b]) +
             static_cast<double>(constituent.absorption[b])) *
            constituent.density(altitude);
    }
    return extinction;
}

double BandedAtmosphere::TransmittanceToTop(double r, double mu,
                                            int band,
                                            int steps) const {
    const double top = bottomRadius + topAltitude;
    // Where the ray leaves the sphere of radius `top`.
    const double discriminant = r * r * (mu * mu - 1.0) + top * top;
    const double length = std::max(
        0.0, -r * mu + std::sqrt(std::max(discriminant, 0.0)));
    const double step = length / steps;
    double depth = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double t = (i + 0.5) * step;
        const double radius =
            std::sqrt(r * r + t * t + 2.0 * r * mu * t);
        depth += Extinction(radius - bottomRadius, band) * step;
    }
    return std::exp(-depth);
}

BandedAtmosphere BandAverage(const AtmosphereModel& atmosphere,
                             const SpectralBands& bands,
                             int phaseAngles) {
    bands.Check();
    const std::vector<double> cosines = PhaseCosines(phaseAngles);
    const int samples = std::max(
        1,
        static_cast<int>(std::ceil(bands.Width() / kBandSampleNm)));

    BandedAtmosphere result;
    result.bottomRadius = atmosphere.bottomRadius;
    result.topAltitude = atmosphere.topAltitude;
    result.bands = bands;
    result.phaseAngles = phaseAngles;
    for (const Constituent& constituent : atmosphere.constituents) {
        BandedConstituent banded;
        banded.name = constituent.name;
        banded.density = constituent.density;
        banded.phase.resize(static_cast<std::size_t>(bands.count) *
                            cosines.size());
        for (int band = 0; band < bands.count; ++band) {
            double scattering = 0.0;
            double absorption = 0.0;
            std::vector<double> phase(cosines.size(), 0.0);
            for (int s = 0; s < samples; ++s) {
                const double nm = bands.Lower(band) +
                                  (s + 0.5) * bands.Width() / samples;
                const ParticleOptics optics =
                    constituent.optics(nm, cosines);
                scattering += optics.scattering;
                absorption += optics.absorption;
                for (std::size_t i = 0; i < cosines.size(); ++i) {
                    phase[i] += optics.scattering * optics.phase[i];
                }
            }
            banded.scattering.push_back(static_cast<float>(
                constituent.numberDensity * scattering / samples));
            banded.absorption.push_back(static_cast<float>(
                constituent.numberDensity * absorption / samples));
            float* out =
                banded.phase.data() +
                static_cast<std::size_t>(band) * cosines.size();
            for (std::size_t i = 0; i < cosines.size(); ++i) {
                out[i] = static_cast<float>(
                    scattering > 0.0 ? phase[i] / scattering
                                     : kIsotropic);
            }
        }
        result.constituents.push_back(std::move(banded));
    }
    return result;
}

}  // namespace ernie
