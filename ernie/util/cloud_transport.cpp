// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_transport.h"

#include "ernie/util/disk_cache.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <random>
#include <string>
#include <thread>

namespace ernie {

namespace {

constexpr double kPi = std::numbers::pi;

// SplitMix64's uniform doubles in [0, 1).
struct Random {
    std::uint64_t state;

    double Next() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * 0x1.0p-53;
    }
};

// Scattering angles drawn from a phase function's table, by inverting
// its cumulative distribution over the sphere.
class PhaseSampler {
  public:
    explicit PhaseSampler(const DropletPhaseTable& phase) {
        const std::size_t n = phase.angles;
        const float* mean =
            phase.values.data() + phase.values.size() - n;
        thetas_.resize(n);
        cdf_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double u =
                static_cast<double>(i) / static_cast<double>(n - 1);
            thetas_[i] = 0.5 * kPi * (1.0 - std::cos(kPi * u));
            if (i > 0) {
                cdf_[i] =
                    cdf_[i - 1] +
                    0.5 *
                        (double(mean[i - 1]) *
                             std::sin(thetas_[i - 1]) +
                         double(mean[i]) * std::sin(thetas_[i])) *
                        (thetas_[i] - thetas_[i - 1]);
            }
        }
        for (double& value : cdf_) {
            value /= cdf_.back();
        }
    }

    double Theta(double xi) const {
        const auto it =
            std::upper_bound(cdf_.begin(), cdf_.end(), xi);
        const std::size_t i = std::min(
            static_cast<std::size_t>(
                std::max<std::ptrdiff_t>(it - cdf_.begin() - 1, 0)),
            cdf_.size() - 2);
        const double span = cdf_[i + 1] - cdf_[i];
        const double along = span > 0.0 ? (xi - cdf_[i]) / span : 0.0;
        return thetas_[i] + (thetas_[i + 1] - thetas_[i]) * along;
    }

  private:
    std::vector<double> thetas_;
    std::vector<double> cdf_;
};

struct Vec {
    double x, y, z;
};

// `direction` turned by scattering angle `theta` about azimuth `phi`.
Vec Scatter(const Vec& direction, double theta, double phi) {
    const double cosT = std::cos(theta);
    const double sinT = std::sin(theta);
    const double cosP = std::cos(phi);
    const double sinP = std::sin(phi);
    const double s =
        std::sqrt(std::max(0.0, 1.0 - direction.z * direction.z));
    if (s < 1e-6) {
        return {sinT * cosP, sinT * sinP,
                direction.z < 0.0 ? -cosT : cosT};
    }
    return {sinT *
                    (direction.x * direction.z * cosP -
                     direction.y * sinP) /
                    s +
                direction.x * cosT,
            sinT *
                    (direction.y * direction.z * cosP +
                     direction.x * sinP) /
                    s +
                direction.y * cosT,
            -sinT * cosP * s + direction.z * cosT};
}

// One slab's photons, tallied into `tallies`: per tally set (light
// scattered once, then the order sets), per depth, view cosine and
// azimuth, the weights and their entry offsets' sums.
void SimulateSlab(const CloudTransportShape& shape,
                  const PhaseSampler& sampler, std::size_t light,
                  std::size_t thickness, std::size_t photons,
                  std::uint64_t seed, std::vector<double>& tallies) {
    const double muL = shape.LightCosine(light);
    const double tau = shape.Thickness(thickness);
    std::vector<double> planes(shape.depths);
    for (std::size_t k = 0; k < shape.depths; ++k) {
        planes[k] = shape.DepthFraction(k) * tau;
    }
    const std::size_t perDepth = shape.viewCosines * shape.azimuths;
    const std::size_t perSet = shape.depths * perDepth;
    Random random{seed};
    for (std::size_t p = 0; p < photons; ++p) {
        Vec position{0.0, 0.0, 0.0};
        Vec direction{std::sqrt(1.0 - muL * muL), 0.0, muL};
        std::size_t order = 0;
        while (true) {
            const double step = -std::log(1.0 - random.Next());
            const Vec next{position.x + direction.x * step,
                           position.y + direction.y * step,
                           position.z + direction.z * step};
            if (order >= 1 && direction.z != 0.0) {
                const std::size_t set =
                    order == 1 ? 0 : 1 + CloudOrderSet(order);
                const double lo = std::min(position.z, next.z);
                const double hi = std::max(position.z, next.z);
                const double weight = 1.0 / std::abs(direction.z);
                const std::size_t cosine =
                    std::min(static_cast<std::size_t>(std::max(
                                 0.0, (direction.z + 1.0) * 0.5 *
                                          double(shape.viewCosines))),
                             shape.viewCosines - 1);
                const std::size_t azimuth =
                    std::min(static_cast<std::size_t>(
                                 std::atan2(std::abs(direction.y),
                                            direction.x) /
                                 kPi * double(shape.azimuths)),
                             shape.azimuths - 1);
                const double side = direction.y < 0.0 ? -1.0 : 1.0;
                for (std::size_t k = 0; k < shape.depths; ++k) {
                    if (!(planes[k] > lo && planes[k] <= hi)) {
                        continue;
                    }
                    const double along =
                        (planes[k] - position.z) / direction.z;
                    const double u =
                        -(position.x + direction.x * along);
                    const double v =
                        -(position.y + direction.y * along) * side;
                    double* tally =
                        tallies.data() +
                        (set * perSet + k * perDepth +
                         cosine * shape.azimuths + azimuth) *
                            4;
                    tally[0] += weight;
                    tally[1] += weight * u;
                    tally[2] += weight * v;
                    tally[3] += weight * (u * u + v * v);
                }
            }
            if (next.z < 0.0 || next.z > tau) {
                break;
            }
            position = next;
            ++order;
            direction =
                Scatter(direction, sampler.Theta(random.Next()),
                        2.0 * kPi * random.Next());
        }
    }
}

}  // namespace

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
    const double u =
        static_cast<double>(i) / static_cast<double>(depths - 1);
    return 0.5 * (1.0 - std::cos(kPi * u));
}

double CloudTransportShape::ViewCosine(std::size_t i) const {
    return -1.0 +
           (static_cast<double>(i) + 0.5) * 2.0 / double(viewCosines);
}

double CloudTransportShape::Azimuth(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * kPi / double(azimuths);
}

std::size_t CloudTransportShape::Cells() const {
    return lightCosines * thicknesses * depths * viewCosines *
           azimuths;
}

std::size_t CloudTransportShape::Cell(std::size_t light,
                                      std::size_t thickness,
                                      std::size_t depth,
                                      std::size_t viewCosine,
                                      std::size_t azimuth) const {
    return (((light * thicknesses + thickness) * depths + depth) *
                viewCosines +
            viewCosine) *
               azimuths +
           azimuth;
}

CloudTransportTable
SimulateCloudTransport(const DropletPhaseTable& phase,
                       const CloudTransportOptions& options) {
    const CloudTransportShape& shape = options.shape;
    const PhaseSampler sampler(phase);
    const std::size_t perSlab =
        shape.depths * shape.viewCosines * shape.azimuths;
    const std::size_t slabs = shape.lightCosines * shape.thicknesses;
    CloudTransportTable table;
    table.shape = shape;
    table.values.assign(kCloudOrderSets * shape.Cells() * 4, 0.0f);
    table.single.assign(shape.Cells(), 0.0f);
    // Each light and thickness's slab on its own; the thickest first,
    // they take longest.
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        std::vector<double> tallies;
        for (std::size_t job; (job = next.fetch_add(1)) < slabs;) {
            const std::size_t light = job % shape.lightCosines;
            const std::size_t thickness =
                shape.thicknesses - 1 - job / shape.lightCosines;
            tallies.assign((kCloudOrderSets + 1) * perSlab * 4, 0.0);
            SimulateSlab(
                shape, sampler, light, thickness, options.photons,
                options.seed * 0x100000001b3ull +
                    static_cast<std::uint64_t>(job) * 0x9e3779b9ull,
                tallies);
            // Radiance from the crossings' weights: each photon
            // carries the light's flux through the lit face, its
            // cosine, over the photons; over each bin's solid angle
            // (both sides of the light's plane).
            const double solidAngle =
                2.0 / double(shape.viewCosines) * 2.0 * kPi /
                double(shape.azimuths);
            const double scale =
                shape.LightCosine(light) /
                (static_cast<double>(options.photons) * solidAngle);
            const std::size_t first =
                shape.Cell(light, thickness, 0, 0, 0);
            for (std::size_t set = 0; set <= kCloudOrderSets; ++set) {
                for (std::size_t i = 0; i < perSlab; ++i) {
                    const double* tally =
                        tallies.data() + (set * perSlab + i) * 4;
                    const double weight = tally[0];
                    const float radiance =
                        static_cast<float>(weight * scale);
                    if (set == 0) {
                        table.single[first + i] = radiance;
                        continue;
                    }
                    float* cell =
                        table.values.data() +
                        ((set - 1) * shape.Cells() + first + i) * 4;
                    cell[0] = radiance;
                    if (weight > 0.0) {
                        const double u = tally[1] / weight;
                        const double v = tally[2] / weight;
                        cell[1] = static_cast<float>(u);
                        cell[2] = static_cast<float>(v);
                        cell[3] = static_cast<float>(std::sqrt(
                            std::max(0.0, tally[3] / weight - u * u -
                                              v * v)));
                    }
                }
            }
        }
    };
    const std::size_t threads = std::clamp<std::size_t>(
        std::thread::hardware_concurrency() / 2, 1, slabs);
    std::vector<std::thread> pool;
    for (std::size_t i = 1; i < threads; ++i) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
    return table;
}

CloudTransportTable
CachedCloudTransport(const CloudTransportOptions& options) {
    namespace fs = std::filesystem;
    const CloudTransportShape& shape = options.shape;
    CloudTransportTable table;
    table.shape = shape;
    const std::size_t count = kCloudOrderSets * shape.Cells() * 4;
    const std::optional<fs::path> directory =
        CacheDirectory("ERNIE_CLOUD_CACHE", "clouds");
    fs::path path;
    if (directory) {
        CacheKey key;
        key.Add("ernie-cloud-transport-1");
        for (const std::uint64_t value :
             {std::uint64_t{shape.lightCosines},
              std::uint64_t{shape.thicknesses},
              std::uint64_t{shape.depths},
              std::uint64_t{shape.viewCosines},
              std::uint64_t{shape.azimuths},
              std::uint64_t{options.photons}, options.seed}) {
            key.Add(std::to_string(value));
        }
        path = *directory / (key.Hex() + ".bin");
        std::ifstream in(path, std::ios::binary);
        table.values.resize(count);
        if (in.read(reinterpret_cast<char*>(table.values.data()),
                    static_cast<std::streamsize>(count *
                                                 sizeof(float))) &&
            in.peek() == std::ifstream::traits_type::eof()) {
            return table;
        }
    }
    table = SimulateCloudTransport(
        DropletPhase(7.0, 2.0, {400.0, 700.0, 8}, 1024), options);
    table.single.clear();
    if (directory) {
        // Written beside it, then renamed over it, so that no reader
        // sees a part of it.
        fs::path partial = path;
        partial += ".part" + std::to_string(std::random_device{}());
        std::error_code error;
        fs::create_directories(*directory, error);
        {
            std::ofstream out(partial, std::ios::binary);
            out.write(
                reinterpret_cast<const char*>(table.values.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
            if (!out) {
                error = std::make_error_code(std::errc::io_error);
            }
        }
        if (!error) {
            fs::rename(partial, path, error);
        }
        if (error) {
            fs::remove(partial, error);
        }
    }
    return table;
}

}  // namespace ernie
