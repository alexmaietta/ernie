// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/noise.h"

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

}  // namespace

GradientNoise::GradientNoise(std::uint32_t seed) {
    std::array<std::uint8_t, 256> order{};
    std::iota(order.begin(), order.end(), std::uint8_t{0});
    // A Fisher-Yates shuffle driven by a 32-bit LCG (Numerical
    // Recipes' constants), the same on every platform.
    std::uint32_t state = seed * 2654435761u + 1013904223u;
    for (int i = 255; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        const auto j = static_cast<int>((state >> 8) % (i + 1));
        std::swap(order[i], order[j]);
    }
    for (int i = 0; i < 512; ++i) {
        permutation_[i] = order[i & 255];
    }
}

double GradientNoise::operator()(const glm::dvec3& p) const {
    const glm::dvec3 floor = glm::floor(p);
    const int x = static_cast<int>(floor.x) & 255;
    const int y = static_cast<int>(floor.y) & 255;
    const int z = static_cast<int>(floor.z) & 255;
    const glm::dvec3 f = p - floor;
    const double u = Fade(f.x);
    const double v = Fade(f.y);
    const double w = Fade(f.z);
    const auto& P = permutation_;
    const int a = P[x] + y;
    const int aa = P[a] + z;
    const int ab = P[a + 1] + z;
    const int b = P[x + 1] + y;
    const int ba = P[b] + z;
    const int bb = P[b + 1] + z;
    const auto lerp = [](double t, double a0, double a1) {
        return a0 + t * (a1 - a0);
    };
    return lerp(
        w,
        lerp(v,
             lerp(u, Gradient(P[aa], f.x, f.y, f.z),
                  Gradient(P[ba], f.x - 1, f.y, f.z)),
             lerp(u, Gradient(P[ab], f.x, f.y - 1, f.z),
                  Gradient(P[bb], f.x - 1, f.y - 1, f.z))),
        lerp(v,
             lerp(u, Gradient(P[aa + 1], f.x, f.y, f.z - 1),
                  Gradient(P[ba + 1], f.x - 1, f.y, f.z - 1)),
             lerp(u, Gradient(P[ab + 1], f.x, f.y - 1, f.z - 1),
                  Gradient(P[bb + 1], f.x - 1, f.y - 1, f.z - 1))));
}

double GradientNoise::Fbm(const glm::dvec3& p, int octaves,
                          double lacunarity, double gain) const {
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

}  // namespace ernie
