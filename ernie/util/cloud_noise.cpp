// SPDX-License-Identifier: BSD-3-Clause
//
// The textures' recipes follow three-geospatial's clouds (MIT,
// Copyright (c) 2024 Shota Matsuda) and Hillaire's
// TileableVolumeNoise (MIT, Copyright (c) 2017 Sebastien Hillaire);
// see cloud_noise.h.
#include "ernie/util/cloud_noise.h"

#include "ernie/util/disk_cache.h"

#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <thread>

namespace ernie {

namespace {

double Fract(double x) {
    return x - std::floor(x);
}

double Hash(double n) {
    return Fract(std::sin(n + 1.951) * 43758.5453);
}

// Smoothed value noise on the integer lattice.
double ValueNoise(double x, double y, double z) {
    const double px = std::floor(x);
    const double py = std::floor(y);
    const double pz = std::floor(z);
    double fx = x - px;
    double fy = y - py;
    double fz = z - pz;
    fx = fx * fx * (3.0 - 2.0 * fx);
    fy = fy * fy * (3.0 - 2.0 * fy);
    fz = fz * fz * (3.0 - 2.0 * fz);
    const double n = px + py * 57.0 + 113.0 * pz;
    const auto mix = [](double a, double b, double t) {
        return a + (b - a) * t;
    };
    return mix(mix(mix(Hash(n), Hash(n + 1.0), fx),
                   mix(Hash(n + 57.0), Hash(n + 58.0), fx), fy),
               mix(mix(Hash(n + 113.0), Hash(n + 114.0), fx),
                   mix(Hash(n + 170.0), Hash(n + 171.0), fx), fy),
               fz);
}

double Mod(double x, double m) {
    return x - m * std::floor(x / m);
}

// Runs `body(i)` for i in [0, count) across half the machine's
// threads.
void ParallelFor(std::size_t count,
                 const std::function<void(std::size_t)>& body) {
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < count;) {
            body(i);
        }
    };
    const std::size_t threads = std::clamp<std::size_t>(
        std::thread::hardware_concurrency() / 2, 1, count);
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
}

double WorleyFbm(double x, double y, double z, double cells) {
    const double a = 1.0 - TileableWorley(x, y, z, cells * 2.0);
    const double b = 1.0 - TileableWorley(x, y, z, cells * 4.0);
    const double c = 1.0 - TileableWorley(x, y, z, cells * 8.0);
    const double d = 1.0 - TileableWorley(x, y, z, cells * 16.0);
    const double f0 = 0.625 * a + 0.25 * b + 0.125 * c;
    const double f1 = 0.625 * b + 0.25 * c + 0.125 * d;
    const double f2 = 0.75 * c + 0.25 * d;
    return 0.625 * f0 + 0.25 * f1 + 0.125 * f2;
}

double SmoothStep(double edge0, double edge1, double x) {
    const double t =
        std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

double WeatherWorley(double x, double y, double z, double frequency) {
    double noise = 0.0;
    double amplitude = 0.4;
    for (int i = 0; i < 4; ++i) {
        noise +=
            amplitude * (1.0 - TileableWorley(x, y, z, frequency));
        frequency *= 2.0;
        amplitude *= 0.95;
    }
    return noise;
}

glm::dvec3 Perlin3(double x, double y, double z) {
    return {TileablePerlin(x, y, z, 12.0, 12.0, 12.0, 3),
            TileablePerlin(y - 19.1, z + 33.4, x + 47.2, 12.0, 12.0,
                           12.0, 3),
            TileablePerlin(z + 74.2, x - 124.5, y + 99.4, 12.0, 12.0,
                           12.0, 3)};
}

}  // namespace

double TileableWorley(double x, double y, double z, double cells) {
    const double cx = x * cells;
    const double cy = y * cells;
    const double cz = z * cells;
    double nearest = 1e10;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                const double tx = std::floor(cx) + i;
                const double ty = std::floor(cy) + j;
                const double tz = std::floor(cz) + k;
                // The cell's point, offset alike in all axes (as
                // Hillaire's), its lattice wrapped to tile.
                const double offset = ValueNoise(
                    Mod(tx, cells), Mod(ty, cells), Mod(tz, cells));
                const double dx = cx - tx - offset;
                const double dy = cy - ty - offset;
                const double dz = cz - tz - offset;
                nearest =
                    std::min(nearest, dx * dx + dy * dy + dz * dz);
            }
        }
    }
    return std::clamp(nearest, 0.0, 1.0);
}

double TileablePerlin(double x, double y, double z, double fx,
                      double fy, double fz, int octaves) {
    double sum = 0.0;
    double weightSum = 0.0;
    double weight = 1.0;
    for (int i = 0; i < octaves; ++i) {
        const glm::vec4 p(static_cast<float>(x * fx),
                          static_cast<float>(y * fy),
                          static_cast<float>(z * fz), 0.0f);
        const glm::vec4 rep(static_cast<float>(fx),
                            static_cast<float>(fy),
                            static_cast<float>(fz), 1.0f);
        sum += double(glm::perlin(p, rep)) * weight;
        weightSum += weight;
        weight *= 0.5;
        fx *= 2.0;
        fy *= 2.0;
        fz *= 2.0;
    }
    return sum / weightSum;
}

CloudNoise MakeCloudNoise() {
    CloudNoise noise;
    // The shape: Perlin-Worley, remapped by a Worley fractal sum.
    {
        const std::size_t n = noise.shapeSize;
        noise.shape.assign(n * n * n, 0.0f);
        ParallelFor(n, [&](std::size_t k) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                    const double x = (double(i) + 0.5) / double(n);
                    const double y = (double(j) + 0.5) / double(n);
                    const double z = (double(k) + 0.5) / double(n);
                    const double perlin = std::clamp(
                        TileablePerlin(x, y, z, 8.0, 8.0, 8.0, 3),
                        0.0, 1.0);
                    const double worley =
                        0.625 * (1.0 - TileableWorley(x, y, z, 8.0)) +
                        0.25 * (1.0 - TileableWorley(x, y, z, 32.0)) +
                        0.125 * (1.0 - TileableWorley(x, y, z, 56.0));
                    const double perlinWorley =
                        worley + perlin * (1.0 - worley);
                    const double fbm = WorleyFbm(x, y, z, 4.0);
                    noise.shape[(k * n + j) * n + i] =
                        static_cast<float>(
                            (perlinWorley - (fbm - 1.0)) /
                            (2.0 - fbm));
                }
            }
        });
    }
    // The detail: a Worley fractal sum.
    {
        const std::size_t n = noise.detailSize;
        noise.detail.assign(n * n * n, 0.0f);
        ParallelFor(n, [&](std::size_t k) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                    noise.detail[(k * n + j) * n + i] =
                        static_cast<float>(WorleyFbm(
                            (double(i) + 0.5) / double(n),
                            (double(j) + 0.5) / double(n),
                            (double(k) + 0.5) / double(n), 1.0));
                }
            }
        });
    }
    // The local weather: low and mid layers' Worley cells, the high
    // layer's streaky Perlin.
    {
        const std::size_t n = noise.weatherSize;
        noise.weather.assign(n * n * 4, 0.0f);
        ParallelFor(n, [&](std::size_t j) {
            for (std::size_t i = 0; i < n; ++i) {
                const double x = (double(i) + 0.5) / double(n);
                const double y = (double(j) + 0.5) / double(n);
                const double mid = SmoothStep(
                    1.0, 1.4,
                    WeatherWorley(x + 0.5, y + 0.5, 0.5, 8.0));
                const double low = std::clamp(
                    SmoothStep(0.8, 1.4,
                               WeatherWorley(x, y, 0.0, 16.0)) -
                        mid,
                    0.0, 1.0);
                const double high = SmoothStep(
                    -0.5, 0.5,
                    TileablePerlin(x, y, 0.0, 6.0, 12.0, 1.0, 8));
                float* texel = noise.weather.data() + (j * n + i) * 4;
                texel[0] = static_cast<float>(low);
                texel[1] = static_cast<float>(mid);
                texel[2] = static_cast<float>(high);
                texel[3] = 1.0f;
            }
        });
    }
    // The turbulence: Perlin noise's curl, its direction.
    {
        const std::size_t n = noise.turbulenceSize;
        noise.turbulence.assign(n * n * 3, 0.0f);
        ParallelFor(n, [&](std::size_t j) {
            constexpr double kDelta = 0.1;
            for (std::size_t i = 0; i < n; ++i) {
                const double x = (double(i) + 0.5) / double(n);
                const double y = (double(j) + 0.5) / double(n);
                const glm::dvec3 x0 = Perlin3(x - kDelta, y, 0.0);
                const glm::dvec3 x1 = Perlin3(x + kDelta, y, 0.0);
                const glm::dvec3 y0 = Perlin3(x, y - kDelta, 0.0);
                const glm::dvec3 y1 = Perlin3(x, y + kDelta, 0.0);
                const glm::dvec3 z0 = Perlin3(x, y, -kDelta);
                const glm::dvec3 z1 = Perlin3(x, y, kDelta);
                glm::dvec3 curl(y1.z - y0.z - z1.y + z0.y,
                                z1.x - z0.x - x1.z + x0.z,
                                x1.y - x0.y - y1.x + y0.x);
                const double length = glm::length(curl);
                curl = length > 0.0 ? curl / length : glm::dvec3(0.0);
                float* texel =
                    noise.turbulence.data() + (j * n + i) * 3;
                for (int c = 0; c < 3; ++c) {
                    texel[c] =
                        static_cast<float>(0.5 * curl[c] + 0.5);
                }
            }
        });
    }
    return noise;
}

CloudNoise CachedCloudNoise() {
    namespace fs = std::filesystem;
    CloudNoise noise;
    std::vector<float>* parts[] = {&noise.shape, &noise.detail,
                                   &noise.weather, &noise.turbulence};
    const std::size_t sizes[] = {
        noise.shapeSize * noise.shapeSize * noise.shapeSize,
        noise.detailSize * noise.detailSize * noise.detailSize,
        noise.weatherSize * noise.weatherSize * 4,
        noise.turbulenceSize * noise.turbulenceSize * 3};
    const std::optional<fs::path> directory =
        CacheDirectory("ERNIE_CLOUD_CACHE", "clouds");
    fs::path path;
    if (directory) {
        CacheKey key;
        key.Add("ernie-cloud-noise-1");
        path = *directory / (key.Hex() + ".bin");
        std::ifstream in(path, std::ios::binary);
        bool whole = static_cast<bool>(in);
        for (std::size_t p = 0; p < 4 && whole; ++p) {
            parts[p]->resize(sizes[p]);
            whole = static_cast<bool>(
                in.read(reinterpret_cast<char*>(parts[p]->data()),
                        static_cast<std::streamsize>(sizes[p] *
                                                     sizeof(float))));
        }
        if (whole && in.peek() == std::ifstream::traits_type::eof()) {
            return noise;
        }
    }
    noise = MakeCloudNoise();
    if (directory) {
        // Written beside it, then renamed over it, so that no reader
        // sees a part of it.
        fs::path partial = path;
        partial += ".part" + std::to_string(std::random_device{}());
        std::error_code error;
        fs::create_directories(*directory, error);
        {
            std::ofstream out(partial, std::ios::binary);
            for (const std::vector<float>* part : parts) {
                out.write(reinterpret_cast<const char*>(part->data()),
                          static_cast<std::streamsize>(
                              part->size() * sizeof(float)));
            }
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
    return noise;
}

}  // namespace ernie
