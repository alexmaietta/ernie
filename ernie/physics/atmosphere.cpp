// SPDX-License-Identifier: BSD-3-Clause
/// @file atmosphere.cpp
/// @brief Atmosphere models, density profiles and band averaging.
#include "ernie/physics/atmosphere.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace {

/// One Dobson unit, in molecules m^-2.
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
        return std::max(0.0, 1.0 - std::abs(altitude - layer.height) / layer.width);
    }
    return 0.0;
}

/// An isothermal gas column: number density p / (k T) at the ground
/// and scale height R T / (M g).
std::pair<double, DensityProfile> IsothermalGas(double pressure, double temperature,
                                                double molarMass, double gravity) {
    const double numberDensity = pressure / (kBoltzmann * temperature);
    const double scaleHeight = kGasConstant * temperature / (molarMass * gravity);
    return {numberDensity, DensityProfile::Exponential(scaleHeight)};
}

/// The number density that gives `profile` the vertical optical depth
/// `opticalDepth` at `wavelengthNm` up to `top`.
double NumberDensityForDepth(const OpticsModel& optics, const DensityProfile& profile, double top,
                             double wavelengthNm, double opticalDepth) {
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
    DensityProfile profile;
    profile.layers = {{DensityLayer::Shape::Exponential, scaleHeight, 0.0, 1.0}};
    return profile;
}

DensityProfile DensityProfile::Tent(double peakAltitude, double halfWidth) {
    DensityProfile profile;
    profile.layers = {{DensityLayer::Shape::Tent, peakAltitude, halfWidth, 1.0}};
    return profile;
}

DensityProfile DensityProfile::StandardAtmosphere(double surfaceTemperature, double molarMass,
                                                  double gravity) {
    // U.S. Standard Atmosphere 1976, table 4: the layers' base
    // heights (m) and lapse rates (K/m) to 84.852 km.
    struct Layer {
        double base;
        double lapse;
    };
    static constexpr Layer kLayers[] = {{0.0, -6.5e-3},     {11000.0, 0.0}, {20000.0, 1.0e-3},
                                        {32000.0, 2.8e-3},  {47000.0, 0.0}, {51000.0, -2.8e-3},
                                        {71000.0, -2.0e-3}, {84852.0, 0.0}};
    const auto temperature = [&](double h) {
        double t = surfaceTemperature;
        for (std::size_t i = 0; i < std::size(kLayers); ++i) {
            const double end = i + 1 < std::size(kLayers) ? kLayers[i + 1].base : 1e30;
            t += kLayers[i].lapse * (std::clamp(h, kLayers[i].base, end) - kLayers[i].base);
        }
        return t;
    };
    constexpr double kStep = 100.0;
    constexpr int kSamples = 1201;  // to 120 km
    constexpr int kSubSteps = 10;
    const double scale = molarMass * gravity / kGasConstant;
    DensityProfile profile;
    profile.tableStep = kStep;
    profile.table.reserve(kSamples);
    double logPressure = 0.0;  // relative to the ground's
    profile.table.push_back(1.0);
    for (int i = 1; i < kSamples; ++i) {
        // d ln p / dh = -M g / (R T), by the midpoint rule.
        for (int s = 0; s < kSubSteps; ++s) {
            const double h = (i - 1) * kStep + (s + 0.5) * kStep / kSubSteps;
            logPressure -= scale / temperature(h) * kStep / kSubSteps;
        }
        const double h = i * kStep;
        profile.table.push_back(std::exp(logPressure) * surfaceTemperature / temperature(h));
    }
    return profile;
}

double DensityProfile::operator()(double altitude) const {
    if (!table.empty()) {
        const double x =
            std::clamp(altitude / tableStep, 0.0, static_cast<double>(table.size() - 1));
        const auto i = std::min(static_cast<std::size_t>(x), table.size() - 2);
        const double t = x - static_cast<double>(i);
        return table[i] + (table[i + 1] - table[i]) * t;
    }
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
    return [co2Fraction](double wavelengthNm, std::span<const double> cosines) {
        const double king = AirKingFactor(wavelengthNm, co2Fraction);
        ParticleOptics optics;
        optics.scattering =
            RayleighCrossSection(wavelengthNm, AirRefractivity(wavelengthNm, co2Fraction), king);
        for (const double mu : cosines) {
            optics.phase.push_back(RayleighPhase(mu, king));
        }
        return optics;
    };
}

OpticsModel CarbonDioxideMolecules(double co2Fraction) {
    return [co2Fraction](double wavelengthNm, std::span<const double> cosines) {
        const double co2 = co2Fraction * RayleighCrossSection(
                                             wavelengthNm, CarbonDioxideRefractivity(wavelengthNm),
                                             kCarbonDioxideKingFactor);
        const double airKing = AirKingFactor(wavelengthNm);
        const double rest =
            (1.0 - co2Fraction) *
            RayleighCrossSection(wavelengthNm, AirRefractivity(wavelengthNm), airKing);
        ParticleOptics optics;
        optics.scattering = co2 + rest;
        // Each gas scatters with its own depolarisation; the
        // mixture's phase function is their mean weighted by
        // scattering.
        for (const double mu : cosines) {
            optics.phase.push_back((co2 * RayleighPhase(mu, kCarbonDioxideKingFactor) +
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
        optics.phase.assign(cosines.size(), kIsotropicPhase);
        return optics;
    };
}

OpticsModel MieAerosol(std::vector<LogNormalMode> modes, RefractiveIndex refractiveIndex,
                       int radiusSamples) {
    return [modes = std::move(modes), refractiveIndex = std::move(refractiveIndex),
            radiusSamples](double wavelengthNm, std::span<const double> cosines) {
        return MieParticles(modes, refractiveIndex(wavelengthNm), wavelengthNm, cosines,
                            radiusSamples);
    };
}

double AtmosphereModel::VerticalOpticalDepth(double wavelengthNm) const {
    double depth = 0.0;
    for (const Constituent& constituent : constituents) {
        const ParticleOptics optics = constituent.optics(wavelengthNm, {});
        depth += constituent.numberDensity * (optics.scattering + optics.absorption) *
                 constituent.density.Column(topAltitude);
    }
    return depth;
}

AtmosphereModel EarthAtmosphere(double radius, const EarthAtmosphereOptions& options) {
    AtmosphereModel atmosphere;
    atmosphere.bottomRadius = radius;
    atmosphere.topAltitude = 100e3;

    // Air with the U.S. Standard Atmosphere's temperatures: their
    // fall through the troposphere shapes both the scattering and,
    // through the density's gradient at the ground, the refraction
    // near the horizon (some 3 arcminutes less than an isothermal
    // column's).
    const double airDensity = options.surfacePressure / (kBoltzmann * options.surfaceTemperature);
    const DensityProfile airProfile = DensityProfile::StandardAtmosphere(
        options.surfaceTemperature, kAirMolarMass, options.gravity);
    // Air's refractivity at the reference state scaled to its density
    // (refractivity is proportional to it, Gladstone-Dale).
    const double airRatio = airDensity / kStandardNumberDensity;
    atmosphere.constituents.push_back({"molecules", airDensity, airProfile,
                                       AirMolecules(options.co2Fraction),
                                       [airRatio, co2 = options.co2Fraction](double nm) {
                                           return AirRefractivity(nm, co2) * airRatio;
                                       }});

    // A tent from 10 to 40 km holds the column in a triangle of base
    // 30 km: peak * 15 km.
    const DensityProfile ozoneProfile = DensityProfile::Tent(25e3, 15e3);
    const double ozonePeak = options.ozoneColumn * kDobsonUnit / 15e3;
    atmosphere.constituents.push_back({"ozone", ozonePeak, ozoneProfile, OzoneMolecules(), {}});

    if (options.aerosolOpticalDepth > 0.0) {
        // Schneegans et al. 2024, CosmoScout VR
        // mie-settings/earth_haze.json (CC0): median radii
        // exp(-16.316) and exp(-13.616) m, sigma 0.6, 10000 small
        // particles per large one.
        OpticsModel haze =
            MieAerosol({{std::exp(-16.316), 0.6, 10000.0}, {std::exp(-13.616), 0.6, 1.0}},
                       RefractiveIndex({1.4, 0.015}));
        const DensityProfile hazeProfile = DensityProfile::Exponential(options.aerosolScaleHeight);
        const double hazeDensity = NumberDensityForDepth(haze, hazeProfile, atmosphere.topAltitude,
                                                         550.0, options.aerosolOpticalDepth);
        atmosphere.constituents.push_back(
            {"aerosols", hazeDensity, hazeProfile, std::move(haze), {}});
    }
    return atmosphere;
}

AtmosphereModel MarsAtmosphere(double radius, const MarsAtmosphereOptions& options) {
    AtmosphereModel atmosphere;
    atmosphere.bottomRadius = radius;
    atmosphere.topAltitude = 100e3;

    const auto [gasDensity, gasProfile] = IsothermalGas(
        options.surfacePressure, options.surfaceTemperature, options.molarMass, options.gravity);
    // CO2's refractivity and the remainder's (as air's) at the
    // reference state, scaled to the gas's density (Gladstone-Dale).
    const double gasRatio = gasDensity / kStandardNumberDensity;
    atmosphere.constituents.push_back(
        {"molecules", gasDensity, gasProfile, CarbonDioxideMolecules(options.co2Fraction),
         [gasRatio, co2 = options.co2Fraction](double nm) {
             return (co2 * CarbonDioxideRefractivity(nm) + (1.0 - co2) * AirRefractivity(nm)) *
                    gasRatio;
         }});

    if (options.dustOpticalDepth > 0.0) {
        // Feldspar with 3% hematite (Ehlers et al. 2014), indexed
        // where the hematite table is.
        const RefractiveIndex& hematite = HematiteRefractiveIndex();
        std::vector<double> wavelengths;
        std::vector<std::complex<double>> values;
        for (double nm = 220.0; nm <= 1000.0; nm += 10.0) {
            wavelengths.push_back(nm);
            values.push_back(MaxwellGarnett({1.5, 0.0}, hematite(nm), 0.03));
        }
        OpticsModel dust = MieAerosol({LogNormalMode::FromEffective(0.04e-6, 0.1, 1000.0),
                                       LogNormalMode::FromEffective(1.4e-6, 0.35, 1.0)},
                                      RefractiveIndex(std::move(wavelengths), std::move(values)));
        // Dense low dust and thin high dust (Schneegans et al. 2024,
        // table 1).
        DensityProfile dustProfile;
        dustProfile.layers = {{DensityLayer::Shape::DoubleExponential, 4000.0, 0.0, 3.0},
                              {DensityLayer::Shape::DoubleExponential, 20000.0, 0.0, 1.0}};
        const double dustDensity = NumberDensityForDepth(dust, dustProfile, atmosphere.topAltitude,
                                                         880.0, options.dustOpticalDepth);
        atmosphere.constituents.push_back({"dust", dustDensity, dustProfile, std::move(dust), {}});
    }
    return atmosphere;
}

std::vector<double> PhaseCosines(int count) {
    if (count < 2) {
        throw std::invalid_argument("A phase table needs at least two angles");
    }
    std::vector<double> cosines(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double u = static_cast<double>(i) / (count - 1);
        cosines[static_cast<std::size_t>(i)] = std::cos(std::numbers::pi * u * u);
    }
    return cosines;
}

double BandedAtmosphere::Extinction(double altitude, int band) const {
    const auto b = static_cast<std::size_t>(band);
    double extinction = 0.0;
    for (const BandedConstituent& constituent : constituents) {
        extinction += (static_cast<double>(constituent.scattering[b]) +
                       static_cast<double>(constituent.absorption[b])) *
                      constituent.density(altitude);
    }
    return extinction;
}

double BandedAtmosphere::TransmittanceToTop(double r, double mu, int band, int steps) const {
    const double top = bottomRadius + topAltitude;
    // Where the ray leaves the sphere of radius `top`.
    const double discriminant = r * r * (mu * mu - 1.0) + top * top;
    const double length = std::max(0.0, -r * mu + std::sqrt(std::max(discriminant, 0.0)));
    const double step = length / steps;
    double depth = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double t = (i + 0.5) * step;
        const double radius = std::sqrt(r * r + t * t + 2.0 * r * mu * t);
        depth += Extinction(radius - bottomRadius, band) * step;
    }
    return std::exp(-depth);
}

double BandedAtmosphere::RefractiveIndex(double altitude, int band) const {
    const auto b = static_cast<std::size_t>(band);
    double n = 1.0;
    for (const BandedConstituent& constituent : constituents) {
        n += static_cast<double>(constituent.refractivity[b]) * constituent.density(altitude);
    }
    return n;
}

BandedAtmosphere::Refraction BandedAtmosphere::Refract(double r, double mu, int band,
                                                       double step) const {
    // In the plane of the ray and the planet's centre: x across, z
    // up through the start.
    const double top = bottomRadius + topAltitude;
    const auto index = [&](double radius) { return RefractiveIndex(radius - bottomRadius, band); };
    // The gradient of n, radial: dn/dr times the unit position.
    const auto bend = [&](double px, double pz, double tx, double tz, double& ax, double& az) {
        const double radius = std::sqrt(px * px + pz * pz);
        const double h = 1.0;
        const double dndr = (index(radius + h) - index(radius - h)) / (2.0 * h);
        const double gx = dndr * px / radius;
        const double gz = dndr * pz / radius;
        const double along = gx * tx + gz * tz;
        const double n = index(radius);
        ax = (gx - along * tx) / n;
        az = (gz - along * tz) / n;
    };
    double px = 0.0;
    double pz = r;
    const double sine = std::sqrt(std::max(1.0 - mu * mu, 0.0));
    double tx = sine;
    double tz = mu;
    Refraction result;
    for (int i = 0; i < 1000000; ++i) {
        const double radius = std::sqrt(px * px + pz * pz);
        const bool outwards = px * tx + pz * tz > 0.0;
        if (radius > top && outwards) {
            break;
        }
        if (radius < bottomRadius) {
            result.ground = true;
            break;
        }
        // The midpoint rule on the ray equation: t' = (grad n -
        // (grad n . t) t) / n.
        double ax = 0.0;
        double az = 0.0;
        bend(px, pz, tx, tz, ax, az);
        const double mx = px + 0.5 * step * tx;
        const double mz = pz + 0.5 * step * tz;
        double mtx = tx + 0.5 * step * ax;
        double mtz = tz + 0.5 * step * az;
        bend(mx, mz, mtx, mtz, ax, az);
        px += step * mtx;
        pz += step * mtz;
        tx += step * ax;
        tz += step * az;
        const double length = std::sqrt(tx * tx + tz * tz);
        tx /= length;
        tz /= length;
    }
    // atan2 of the cross and dot products keeps small angles (Mars's
    // microradians) that acos rounds away.
    result.deviation = std::atan2(std::abs(sine * tz - mu * tx), tx * sine + tz * mu);
    return result;
}

BandedAtmosphere BandAverage(const AtmosphereModel& atmosphere, const SpectralBands& bands,
                             int phaseAngles) {
    bands.Check();
    const std::vector<double> cosines = PhaseCosines(phaseAngles);
    const int samples = std::max(1, static_cast<int>(std::ceil(bands.Width() / kBandSampleNm)));

    BandedAtmosphere result;
    result.bottomRadius = atmosphere.bottomRadius;
    result.topAltitude = atmosphere.topAltitude;
    result.bands = bands;
    result.phaseAngles = phaseAngles;
    for (const Constituent& constituent : atmosphere.constituents) {
        BandedConstituent banded;
        banded.name = constituent.name;
        banded.density = constituent.density;
        banded.phase.resize(static_cast<std::size_t>(bands.count) * cosines.size());
        for (int band = 0; band < bands.count; ++band) {
            double scattering = 0.0;
            double absorption = 0.0;
            double refractivity = 0.0;
            std::vector<double> phase(cosines.size(), 0.0);
            for (int s = 0; s < samples; ++s) {
                const double nm = bands.Lower(band) + (s + 0.5) * bands.Width() / samples;
                const ParticleOptics optics = constituent.optics(nm, cosines);
                if (constituent.refractivity) {
                    refractivity += constituent.refractivity(nm);
                }
                scattering += optics.scattering;
                absorption += optics.absorption;
                for (std::size_t i = 0; i < cosines.size(); ++i) {
                    phase[i] += optics.scattering * optics.phase[i];
                }
            }
            banded.scattering.push_back(
                static_cast<float>(constituent.numberDensity * scattering / samples));
            banded.absorption.push_back(
                static_cast<float>(constituent.numberDensity * absorption / samples));
            banded.refractivity.push_back(static_cast<float>(refractivity / samples));
            float* out = banded.phase.data() + static_cast<std::size_t>(band) * cosines.size();
            for (std::size_t i = 0; i < cosines.size(); ++i) {
                out[i] =
                    static_cast<float>(scattering > 0.0 ? phase[i] / scattering : kIsotropicPhase);
            }
        }
        result.constituents.push_back(std::move(banded));
    }
    return result;
}

}  // namespace ernie
