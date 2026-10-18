// SPDX-License-Identifier: BSD-3-Clause
/// @file cloud.cpp
/// @brief Cloud droplet phase tables, Monte Carlo slab transport, and tileable noise textures.
//
// The noise textures' recipes follow three-geospatial's clouds (MIT, Copyright (c) 2024 Shota
// Matsuda) and Hillaire's TileableVolumeNoise (MIT, Copyright (c) 2017 Sebastien Hillaire); see
// cloud.h.
#include "ernie/physics/cloud.h"

#include "ernie/physics/scattering.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/math.h"
#include "ernie/util/noise.h"
#include "ernie/util/parallel.h"
#include "ernie/util/random.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <future>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace ernie {

namespace {

using std::numbers::pi;

/// The integral of f(theta) sin(theta) over [theta0, theta1] by the trapezoid rule; times 2 pi it
/// is the integral over that band of the sphere.
double SphereTrapezoid(double f0, double theta0, double f1, double theta1) {
    return 0.5 * (f0 * std::sin(theta0) + f1 * std::sin(theta1)) * (theta1 - theta0);
}

// ----------------------------------------------------------------------------------------------
// Droplet phase functions

/// Liquid water's refractive index across the visible, its absorption negligible there.
constexpr std::complex<double> kWater(1.333, 0.0);

/// One band's phase function with its forward peak's share of the light, which it is divided by
/// 1 minus.
struct BandPhase {
    std::vector<double> phase;
    double peak = 0.0;
};

/// The share of light within `cut` of forward, by the trapezoid rule over the sphere.
double ForwardPeak(const std::vector<double>& phase, const std::vector<double>& thetas,
                   double cut) {
    double peak = 0.0;
    for (std::size_t i = 0; i + 1 < thetas.size() && thetas[i] < cut; ++i) {
        const double to = std::min(thetas[i + 1], cut);
        const double along = (to - thetas[i]) / (thetas[i + 1] - thetas[i]);
        const double end = phase[i] + (phase[i + 1] - phase[i]) * along;
        peak += SphereTrapezoid(phase[i], thetas[i], end, to) * 2.0 * pi;
    }
    return std::clamp(peak, 0.0, 0.95);
}

/// One band's Mie phase function against the extinction less its forward peak: the peak kept for
/// single scattering (its silver lining about the sun), folded away from the rest.
BandPhase DropletBandPhase(const LogNormalMode& mode, double wavelengthNm,
                           const std::vector<double>& thetas, const std::vector<double>& cosines,
                           double cut) {
    ParticleOptics optics = MieParticles(std::span(&mode, 1), kWater, wavelengthNm, cosines, 48);
    BandPhase band{std::move(optics.phase), 0.0};
    band.peak = ForwardPeak(band.phase, thetas, cut);
    for (double& value : band.phase) {
        value /= 1.0 - band.peak;
    }
    return band;
}

/// The bands' mean beyond the peak, held at the peak's edge value within it so light scattered
/// many times shows no hole about the sun, and normalised to integrate to 1 again.
std::vector<double> FoldedMean(std::vector<double> mean, const std::vector<double>& thetas,
                               double cut) {
    const std::size_t n = thetas.size();
    std::size_t edge = 0;
    while (edge + 1 < n && thetas[edge] < cut) {
        ++edge;
    }
    std::fill(mean.begin(), mean.begin() + static_cast<std::ptrdiff_t>(edge), mean[edge]);
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        total += SphereTrapezoid(mean[i], thetas[i], mean[i + 1], thetas[i + 1]) * 2.0 * pi;
    }
    for (double& value : mean) {
        value /= total;
    }
    return mean;
}

// ----------------------------------------------------------------------------------------------
// Transport in slabs

/// Scattering angles drawn from a phase function's table, by inverting its cumulative distribution
/// over the sphere.
class PhaseSampler {
  public:
    explicit PhaseSampler(const DropletPhaseTable& phase) {
        const std::size_t n = phase.angles;
        const float* mean = phase.values.data() + phase.values.size() - n;
        thetas_.resize(n);
        cdf_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            thetas_[i] = phase.Angle(i);
            if (i > 0) {
                cdf_[i] = cdf_[i - 1] + SphereTrapezoid(double(mean[i - 1]), thetas_[i - 1],
                                                        double(mean[i]), thetas_[i]);
            }
        }
        for (double& value : cdf_) {
            value /= cdf_.back();
        }
    }

    /// The scattering angle at cumulative probability `xi`.
    double Theta(double xi) const {
        return Interpolate<double>(cdf_, thetas_, xi);
    }

  private:
    std::vector<double> thetas_;
    std::vector<double> cdf_;
};

/// `direction` turned by scattering angle `theta` about azimuth `phi`.
glm::dvec3 Scatter(const glm::dvec3& direction, double theta, double phi) {
    const double cosT = std::cos(theta);
    const double sinT = std::sin(theta);
    const double cosP = std::cos(phi);
    const double sinP = std::sin(phi);
    const double s = std::sqrt(std::max(0.0, 1.0 - direction.z * direction.z));
    if (s < 1e-6) {
        return {sinT * cosP, sinT * sinP, direction.z < 0.0 ? -cosT : cosT};
    }
    return {sinT * (direction.x * direction.z * cosP - direction.y * sinP) / s + direction.x * cosT,
            sinT * (direction.y * direction.z * cosP + direction.x * sinP) / s + direction.y * cosT,
            -sinT * cosP * s + direction.z * cosT};
}

/// One slab's tallies: per tally set (light scattered once, then the order sets), per depth, view
/// cosine and azimuth, the crossing weights and their entry offsets' sums.
class SlabTallies {
  public:
    SlabTallies(const CloudTransportShape& shape, double thickness)
        : shape_(shape), perDepth_(shape.viewCosines * shape.azimuths),
          perSet_(shape.depths * perDepth_), values_((kCloudOrderSets + 1) * perSet_ * 4, 0.0) {
        for (std::size_t k = 0; k < shape.depths; ++k) {
            planes_.push_back(shape.DepthFraction(k) * thickness);
        }
    }

    /// Adds a photon's crossings of the depth planes between `from` and `to`, travelling along
    /// `direction` after scattering `order` times; its entry point was the origin.
    void AddSegment(std::size_t order, const glm::dvec3& from, const glm::dvec3& to,
                    const glm::dvec3& direction) {
        if (order == 0 || direction.z == 0.0) {
            return;
        }
        const std::size_t set = order == 1 ? 0 : 1 + CloudOrderSet(order);
        const double lo = std::min(from.z, to.z);
        const double hi = std::max(from.z, to.z);
        const double weight = 1.0 / std::abs(direction.z);
        const std::size_t cosine =
            std::min(static_cast<std::size_t>(
                         std::max(0.0, (direction.z + 1.0) * 0.5 * double(shape_.viewCosines))),
                     shape_.viewCosines - 1);
        const std::size_t azimuth =
            std::min(static_cast<std::size_t>(std::atan2(std::abs(direction.y), direction.x) / pi *
                                              double(shape_.azimuths)),
                     shape_.azimuths - 1);
        const double side = direction.y < 0.0 ? -1.0 : 1.0;
        for (std::size_t k = 0; k < shape_.depths; ++k) {
            if (!(planes_[k] > lo && planes_[k] <= hi)) {
                continue;
            }
            const double along = (planes_[k] - from.z) / direction.z;
            const double u = -(from.x + direction.x * along);
            const double v = -(from.y + direction.y * along) * side;
            double* tally =
                values_.data() +
                (set * perSet_ + k * perDepth_ + cosine * shape_.azimuths + azimuth) * 4;
            tally[0] += weight;
            tally[1] += weight * u;
            tally[2] += weight * v;
            tally[3] += weight * (u * u + v * v);
        }
    }

    /// The four sums of tally set `set`, cell `i` within the slab.
    const double* At(std::size_t set, std::size_t i) const {
        return values_.data() + (set * perSet_ + i) * 4;
    }

  private:
    const CloudTransportShape& shape_;
    std::size_t perDepth_;
    std::size_t perSet_;
    std::vector<double> planes_;
    std::vector<double> values_;
};

/// Traces a slab's photons, each entering the lit face at the origin and scattering without
/// absorption until it leaves.
SlabTallies SimulateSlab(const CloudTransportShape& shape, const PhaseSampler& sampler,
                         std::size_t light, std::size_t thickness, std::size_t photons,
                         std::uint64_t seed) {
    const double muL = shape.LightCosine(light);
    const double tau = shape.Thickness(thickness);
    SlabTallies tallies(shape, tau);
    SplitMix64 random(seed);
    for (std::size_t p = 0; p < photons; ++p) {
        glm::dvec3 position(0.0);
        glm::dvec3 direction(std::sqrt(1.0 - muL * muL), 0.0, muL);
        for (std::size_t order = 0;; ++order) {
            const double step = -std::log(1.0 - random.NextUniform());
            const glm::dvec3 next = position + direction * step;
            tallies.AddSegment(order, position, next, direction);
            if (next.z < 0.0 || next.z > tau) {
                break;
            }
            position = next;
            direction = Scatter(direction, sampler.Theta(random.NextUniform()),
                                2.0 * pi * random.NextUniform());
        }
    }
    return tallies;
}

/// Writes a slab's tallies into the table as radiance and collectors. Each photon carries the
/// light's flux through the lit face (its cosine) over the photons, over each bin's solid angle
/// (both sides of the light's plane).
void StoreSlab(const SlabTallies& tallies, std::size_t light, std::size_t thickness,
               std::size_t photons, CloudTransportTable& table) {
    const CloudTransportShape& shape = table.shape;
    const std::size_t perSlab = shape.depths * shape.viewCosines * shape.azimuths;
    const double solidAngle = 2.0 / double(shape.viewCosines) * 2.0 * pi / double(shape.azimuths);
    const double scale = shape.LightCosine(light) / (static_cast<double>(photons) * solidAngle);
    const std::size_t first = shape.Cell(light, thickness, 0, 0, 0);
    for (std::size_t set = 0; set <= kCloudOrderSets; ++set) {
        for (std::size_t i = 0; i < perSlab; ++i) {
            const double* tally = tallies.At(set, i);
            const double weight = tally[0];
            const auto radiance = static_cast<float>(weight * scale);
            if (set == 0) {
                table.single[first + i] = radiance;
                continue;
            }
            float* cell = table.values.data() + ((set - 1) * shape.Cells() + first + i) * 4;
            cell[0] = radiance;
            if (weight > 0.0) {
                const double u = tally[1] / weight;
                const double v = tally[2] / weight;
                cell[1] = static_cast<float>(u);
                cell[2] = static_cast<float>(v);
                cell[3] =
                    static_cast<float>(std::sqrt(std::max(0.0, tally[3] / weight - u * u - v * v)));
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Noise textures

/// The weather map's Worley sum: four octaves, each doubling in frequency and losing 5%.
double WeatherWorley(const glm::dvec3& p, double frequency) {
    double noise = 0.0;
    double amplitude = 0.4;
    for (int i = 0; i < 4; ++i) {
        noise += amplitude * (1.0 - TileableWorley(p, frequency));
        frequency *= 2.0;
        amplitude *= 0.95;
    }
    return noise;
}

/// The centre of texel (i, j, k) of an n-texel texture, in [0, 1].
glm::dvec3 TexelCentre(std::size_t i, std::size_t j, std::size_t k, std::size_t n) {
    return (glm::dvec3(i, j, k) + 0.5) / double(n);
}

/// The shape: Perlin-Worley noise, remapped by a Worley fractal sum.
std::vector<float> MakeShapeNoise(std::size_t n) {
    std::vector<float> shape(n * n * n, 0.0f);
    ParallelFor(n, [&](std::size_t k) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                const glm::dvec3 p = TexelCentre(i, j, k, n);
                const double perlin = std::clamp(TileablePerlin(p, glm::dvec3(8.0), 3), 0.0, 1.0);
                const double worley = 0.625 * (1.0 - TileableWorley(p, 8.0)) +
                                      0.25 * (1.0 - TileableWorley(p, 32.0)) +
                                      0.125 * (1.0 - TileableWorley(p, 56.0));
                const double perlinWorley = worley + perlin * (1.0 - worley);
                const double fbm = WorleyFbm(p, 4.0);
                shape[(k * n + j) * n + i] =
                    static_cast<float>((perlinWorley - (fbm - 1.0)) / (2.0 - fbm));
            }
        }
    });
    return shape;
}

/// The detail: a Worley fractal sum.
std::vector<float> MakeDetailNoise(std::size_t n) {
    std::vector<float> detail(n * n * n, 0.0f);
    ParallelFor(n, [&](std::size_t k) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                detail[(k * n + j) * n + i] =
                    static_cast<float>(WorleyFbm(TexelCentre(i, j, k, n), 1.0));
            }
        }
    });
    return detail;
}

/// The local weather: the low and mid layers' Worley cells, the high layer's streaky Perlin.
std::vector<float> MakeWeatherMap(std::size_t n) {
    std::vector<float> weather(n * n * 4, 0.0f);
    ParallelFor(n, [&](std::size_t j) {
        for (std::size_t i = 0; i < n; ++i) {
            const glm::dvec3 p = TexelCentre(i, j, 0, n) * glm::dvec3(1.0, 1.0, 0.0);
            const double mid = SmoothStep(1.0, 1.4, WeatherWorley(p + glm::dvec3(0.5), 8.0));
            const double low =
                std::clamp(SmoothStep(0.8, 1.4, WeatherWorley(p, 16.0)) - mid, 0.0, 1.0);
            const double high =
                SmoothStep(-0.5, 0.5, TileablePerlin(p, glm::dvec3(6.0, 12.0, 1.0), 8));
            float* texel = weather.data() + (j * n + i) * 4;
            texel[0] = static_cast<float>(low);
            texel[1] = static_cast<float>(mid);
            texel[2] = static_cast<float>(high);
            texel[3] = 1.0f;
        }
    });
    return weather;
}

/// The turbulence: the direction of Perlin noise's curl, mapped to [0, 1].
std::vector<float> MakeTurbulenceMap(std::size_t n) {
    std::vector<float> turbulence(n * n * 3, 0.0f);
    ParallelFor(n, [&](std::size_t j) {
        constexpr double kDelta = 0.1;
        for (std::size_t i = 0; i < n; ++i) {
            const glm::dvec3 p = TexelCentre(i, j, 0, n) * glm::dvec3(1.0, 1.0, 0.0);
            const glm::dvec3 x0 = TileablePerlin3(p - glm::dvec3(kDelta, 0.0, 0.0));
            const glm::dvec3 x1 = TileablePerlin3(p + glm::dvec3(kDelta, 0.0, 0.0));
            const glm::dvec3 y0 = TileablePerlin3(p - glm::dvec3(0.0, kDelta, 0.0));
            const glm::dvec3 y1 = TileablePerlin3(p + glm::dvec3(0.0, kDelta, 0.0));
            const glm::dvec3 z0 = TileablePerlin3(p - glm::dvec3(0.0, 0.0, kDelta));
            const glm::dvec3 z1 = TileablePerlin3(p + glm::dvec3(0.0, 0.0, kDelta));
            glm::dvec3 curl(y1.z - y0.z - z1.y + z0.y, z1.x - z0.x - x1.z + x0.z,
                            x1.y - x0.y - y1.x + y0.x);
            const double length = glm::length(curl);
            curl = length > 0.0 ? curl / length : glm::dvec3(0.0);
            float* texel = turbulence.data() + (j * n + i) * 3;
            for (int c = 0; c < 3; ++c) {
                texel[c] = static_cast<float>(0.5 * curl[c] + 0.5);
            }
        }
    });
    return turbulence;
}

/// The path of a cloud cache file for a key, or nothing when caching is off.
std::optional<std::filesystem::path> CloudCachePath(const CacheKey& key) {
    const std::optional<std::filesystem::path> directory =
        CacheDirectory("ERNIE_CLOUD_CACHE", "clouds");
    if (!directory) {
        return std::nullopt;
    }
    return *directory / (key.Hex() + ".bin");
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// Droplet phase functions

DropletPhaseTable DropletPhase(double effectiveRadiusUm, double spread, const SpectralBands& bands,
                               std::size_t angles) {
    DropletPhaseTable table;
    table.angles = std::max<std::size_t>(angles, 2);
    const std::size_t n = table.angles;
    const auto count = static_cast<std::size_t>(std::max(bands.count, 0));
    std::vector<double> thetas(n);
    std::vector<double> cosines(n);
    for (std::size_t i = 0; i < n; ++i) {
        thetas[i] = table.Angle(i);
        cosines[i] = std::cos(thetas[i]);
    }
    const LogNormalMode mode = LogNormalMode::FromEffective(std::max(effectiveRadiusUm, 0.5) * 1e-6,
                                                            1.0 / (std::max(spread, 0.1) + 2.0));
    const double cut = kDropletForwardDegrees * pi / 180.0;
    std::vector<std::future<BandPhase>> jobs;
    for (int b = 0; b < bands.count; ++b) {
        jobs.push_back(std::async(std::launch::async, [&, b] {
            return DropletBandPhase(mode, bands.Centre(b), thetas, cosines, cut);
        }));
    }
    // Each band's row, then the bands' mean beyond the peak, summed in doubles.
    table.values.assign((count + 1) * n, 0.0f);
    std::vector<double> mean(n, 0.0);
    for (std::size_t b = 0; b < count; ++b) {
        const BandPhase band = jobs[b].get();
        table.forwardFraction += band.peak / static_cast<double>(count);
        for (std::size_t i = 0; i < n; ++i) {
            table.values[b * n + i] = static_cast<float>(band.phase[i]);
            if (thetas[i] >= cut) {
                mean[i] += band.phase[i] / static_cast<double>(count);
            }
        }
    }
    const std::vector<double> folded = FoldedMean(std::move(mean), thetas, cut);
    for (std::size_t i = 0; i < n; ++i) {
        table.values[count * n + i] = static_cast<float>(folded[i]);
    }
    return table;
}

// ----------------------------------------------------------------------------------------------
// Transport in slabs

std::size_t CloudOrderSet(std::size_t order) {
    if (order <= 2) {
        return 0;
    }
    if (order <= 8) {
        return (order - 1) / 2;
    }
    if (order <= 12) {
        return 4;
    }
    if (order <= 18) {
        return 5;
    }
    return order <= 30 ? 6 : 7;
}

double CloudTransportShape::LightCosine(std::size_t i) const {
    return static_cast<double>(i + 1) / double(lightCosines);
}

double CloudTransportShape::Thickness(std::size_t i) const {
    return 0.25 * std::exp2(static_cast<double>(i));
}

double CloudTransportShape::DepthFraction(std::size_t i) const {
    return CosineSpacing(i, depths);
}

double CloudTransportShape::ViewCosine(std::size_t i) const {
    return -1.0 + (static_cast<double>(i) + 0.5) * 2.0 / double(viewCosines);
}

double CloudTransportShape::Azimuth(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * pi / double(azimuths);
}

std::size_t CloudTransportShape::Cells() const {
    return lightCosines * thicknesses * depths * viewCosines * azimuths;
}

std::size_t CloudTransportShape::Cell(std::size_t light, std::size_t thickness, std::size_t depth,
                                      std::size_t viewCosine, std::size_t azimuth) const {
    return (((light * thicknesses + thickness) * depths + depth) * viewCosines + viewCosine) *
               azimuths +
           azimuth;
}

CloudTransportTable SimulateCloudTransport(const DropletPhaseTable& phase,
                                           const CloudTransportOptions& options) {
    const CloudTransportShape& shape = options.shape;
    const PhaseSampler sampler(phase);
    CloudTransportTable table;
    table.shape = shape;
    table.values.assign(kCloudOrderSets * shape.Cells() * 4, 0.0f);
    table.single.assign(shape.Cells(), 0.0f);
    // Each light and thickness's slab on its own; the thickest first, as they take longest.
    ParallelFor(shape.lightCosines * shape.thicknesses, [&](std::size_t job) {
        const std::size_t light = job % shape.lightCosines;
        const std::size_t thickness = shape.thicknesses - 1 - job / shape.lightCosines;
        const std::uint64_t seed =
            options.seed * 0x100000001b3ull + static_cast<std::uint64_t>(job) * 0x9e3779b9ull;
        const SlabTallies tallies =
            SimulateSlab(shape, sampler, light, thickness, options.photons, seed);
        StoreSlab(tallies, light, thickness, options.photons, table);
    });
    return table;
}

CloudTransportTable CachedCloudTransport(const CloudTransportOptions& options) {
    const CloudTransportShape& shape = options.shape;
    CacheKey key;
    key.Add("ernie-cloud-transport-1");
    for (const std::uint64_t value :
         {std::uint64_t{shape.lightCosines}, std::uint64_t{shape.thicknesses},
          std::uint64_t{shape.depths}, std::uint64_t{shape.viewCosines},
          std::uint64_t{shape.azimuths}, std::uint64_t{options.photons}, options.seed}) {
        key.Add(std::to_string(value));
    }
    const std::optional<std::filesystem::path> path = CloudCachePath(key);
    if (path) {
        if (std::optional<std::vector<float>> values =
                ReadCachedFloats(*path, kCloudOrderSets * shape.Cells() * 4)) {
            return {shape, std::move(*values), {}};
        }
    }
    CloudTransportTable table =
        SimulateCloudTransport(DropletPhase(7.0, 2.0, {400.0, 700.0, 8}, 1024), options);
    table.single.clear();
    if (path) {
        WriteCachedFloats(*path, table.values);
    }
    return table;
}

// ----------------------------------------------------------------------------------------------
// Noise textures

CloudNoise MakeCloudNoise(std::size_t shapeSize, std::size_t detailSize, std::size_t weatherSize,
                          std::size_t turbulenceSize) {
    CloudNoise noise{shapeSize, {}, detailSize, {}, weatherSize, {}, turbulenceSize, {}};
    noise.shape = MakeShapeNoise(noise.shapeSize);
    noise.detail = MakeDetailNoise(noise.detailSize);
    noise.weather = MakeWeatherMap(noise.weatherSize);
    noise.turbulence = MakeTurbulenceMap(noise.turbulenceSize);
    return noise;
}

CloudNoise CachedCloudNoise() {
    CloudNoise noise;
    std::vector<float>* parts[] = {&noise.shape, &noise.detail, &noise.weather, &noise.turbulence};
    const std::size_t sizes[] = {noise.shapeSize * noise.shapeSize * noise.shapeSize,
                                 noise.detailSize * noise.detailSize * noise.detailSize,
                                 noise.weatherSize * noise.weatherSize * 4,
                                 noise.turbulenceSize * noise.turbulenceSize * 3};
    std::size_t total = 0;
    for (const std::size_t size : sizes) {
        total += size;
    }
    CacheKey key;
    key.Add("ernie-cloud-noise-1");
    const std::optional<std::filesystem::path> path = CloudCachePath(key);
    if (path) {
        // The four textures one after another in one file.
        if (const std::optional<std::vector<float>> values = ReadCachedFloats(*path, total)) {
            std::size_t offset = 0;
            for (std::size_t p = 0; p < 4; ++p) {
                parts[p]->assign(values->begin() + static_cast<std::ptrdiff_t>(offset),
                                 values->begin() + static_cast<std::ptrdiff_t>(offset + sizes[p]));
                offset += sizes[p];
            }
            return noise;
        }
    }
    noise = MakeCloudNoise();
    if (path) {
        std::vector<float> values;
        values.reserve(total);
        for (const std::vector<float>* part : parts) {
            values.insert(values.end(), part->begin(), part->end());
        }
        WriteCachedFloats(*path, values);
    }
    return noise;
}

}  // namespace ernie
