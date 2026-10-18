// SPDX-License-Identifier: BSD-3-Clause
/// @file noise.cpp
/// @brief Seeded Perlin gradient noise and its fBm, and tileable value, Worley and Perlin noise.
#include "ernie/util/noise.h"

#include "ernie/util/math.h"

#include <glm/gtc/noise.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace ernie {

namespace {

// Perlin's quintic fade, 6t^5 - 15t^4 + 10t^3.
double Fade(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// The dot of offset (x, y, z) with one of Perlin's 12 edge gradients
// (16 with 4 repeated) that `hash` picks.
double Gradient(int hash, double x, double y, double z) {
    const int h = hash & 15;
    const double u = h < 8 ? x : y;
    const double v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// The classic shader sine hash, matching Hillaire's noise.
double SineHash(double n) {
    return Fract(std::sin(n + 1.951) * 43758.5453);
}

}  // namespace

GradientNoise::GradientNoise(std::uint32_t seed) {
    std::array<std::uint8_t, 256> order{};
    std::iota(order.begin(), order.end(), std::uint8_t{0});
    // A Fisher-Yates shuffle driven by a 32-bit LCG (Numerical
    // Recipes' constants), the same on every platform.
    std::uint32_t state = seed * 2654435761u + 1013904223u;
    for (std::size_t i = 255; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        const std::size_t j = (state >> 8) % (i + 1);
        std::swap(order[i], order[j]);
    }
    for (std::size_t i = 0; i < 512; ++i) {
        permutation_[i] = order[i & 255];
    }
}

double GradientNoise::operator()(const glm::dvec3& p) const {
    const glm::dvec3 floor = glm::floor(p);
    const auto x = static_cast<std::size_t>(static_cast<int>(floor.x) & 255);
    const auto y = static_cast<std::size_t>(static_cast<int>(floor.y) & 255);
    const auto z = static_cast<std::size_t>(static_cast<int>(floor.z) & 255);
    const glm::dvec3 f = p - floor;
    const auto& P = permutation_;
    const std::size_t a = P[x] + y;
    const std::size_t aa = P[a] + z;
    const std::size_t ab = P[a + 1] + z;
    const std::size_t b = P[x + 1] + y;
    const std::size_t ba = P[b] + z;
    const std::size_t bb = P[b + 1] + z;
    const std::array<double, 8> corners{Gradient(P[aa], f.x, f.y, f.z),
                                        Gradient(P[ba], f.x - 1, f.y, f.z),
                                        Gradient(P[ab], f.x, f.y - 1, f.z),
                                        Gradient(P[bb], f.x - 1, f.y - 1, f.z),
                                        Gradient(P[aa + 1], f.x, f.y, f.z - 1),
                                        Gradient(P[ba + 1], f.x - 1, f.y, f.z - 1),
                                        Gradient(P[ab + 1], f.x, f.y - 1, f.z - 1),
                                        Gradient(P[bb + 1], f.x - 1, f.y - 1, f.z - 1)};
    return Trilinear(corners, glm::dvec3(Fade(f.x), Fade(f.y), Fade(f.z)));
}

double GradientNoise::Fbm(const glm::dvec3& p, int octaves, double lacunarity, double gain) const {
    double sum = 0.0;
    double total = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    for (int octave = 0; octave < octaves; ++octave) {
        sum += amplitude * (*this)(p * frequency);
        total += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return total > 0.0 ? sum / total : 0.0;
}

double ValueNoise(const glm::dvec3& p) {
    const glm::dvec3 cell = glm::floor(p);
    glm::dvec3 f = p - cell;
    f = f * f * (3.0 - 2.0 * f);
    // Lattice points hash as n = x + 57 y + 113 z.
    const double n = cell.x + cell.y * 57.0 + 113.0 * cell.z;
    const std::array<double, 8> corners{
        SineHash(n),         SineHash(n + 1.0),   SineHash(n + 57.0),  SineHash(n + 58.0),
        SineHash(n + 113.0), SineHash(n + 114.0), SineHash(n + 170.0), SineHash(n + 171.0)};
    return Trilinear(corners, f);
}

double TileableWorley(const glm::dvec3& p, double cells) {
    const glm::dvec3 c = p * cells;
    const glm::dvec3 base = glm::floor(c);
    double nearest = 1e10;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                const glm::dvec3 t = base + glm::dvec3(i, j, k);
                // The cell's point, offset alike in all axes (as Hillaire's), its lattice wrapped
                // so the noise tiles.
                const double offset =
                    ValueNoise(glm::dvec3(Mod(t.x, cells), Mod(t.y, cells), Mod(t.z, cells)));
                const glm::dvec3 d = c - t - offset;
                nearest = std::min(nearest, glm::dot(d, d));
            }
        }
    }
    return std::clamp(nearest, 0.0, 1.0);
}

double WorleyFbm(const glm::dvec3& p, double cells) {
    const double a = 1.0 - TileableWorley(p, cells * 2.0);
    const double b = 1.0 - TileableWorley(p, cells * 4.0);
    const double c = 1.0 - TileableWorley(p, cells * 8.0);
    const double d = 1.0 - TileableWorley(p, cells * 16.0);
    const double f0 = 0.625 * a + 0.25 * b + 0.125 * c;
    const double f1 = 0.625 * b + 0.25 * c + 0.125 * d;
    const double f2 = 0.75 * c + 0.25 * d;
    return 0.625 * f0 + 0.25 * f1 + 0.125 * f2;
}

double TileablePerlin(const glm::dvec3& p, const glm::dvec3& frequency, int octaves) {
    double sum = 0.0;
    double weightSum = 0.0;
    double weight = 1.0;
    glm::dvec3 f = frequency;
    for (int i = 0; i < octaves; ++i) {
        const glm::vec4 point(glm::vec3(p * f), 0.0f);
        const glm::vec4 period(glm::vec3(f), 1.0f);
        sum += double(glm::perlin(point, period)) * weight;
        weightSum += weight;
        weight *= 0.5;
        f *= 2.0;
    }
    return sum / weightSum;
}

glm::dvec3 TileablePerlin3(const glm::dvec3& p) {
    const glm::dvec3 frequency(12.0);
    return {TileablePerlin(p, frequency, 3),
            TileablePerlin(glm::dvec3(p.y - 19.1, p.z + 33.4, p.x + 47.2), frequency, 3),
            TileablePerlin(glm::dvec3(p.z + 74.2, p.x - 124.5, p.y + 99.4), frequency, 3)};
}

}  // namespace ernie
