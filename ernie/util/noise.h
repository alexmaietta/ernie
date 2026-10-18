// SPDX-License-Identifier: BSD-3-Clause
/// @file noise.h
/// @brief Procedural noise: seeded gradient noise, and value, Worley and Perlin noise that tile.
#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstdint>

namespace ernie {

/// @brief Perlin's improved 3D gradient noise with a seeded lattice.
/// Smooth, zero at integer points and within about [-1, 1].
///
/// References:
/// - K. Perlin, "Improving Noise", ACM Transactions on Graphics 21(3)
///   (SIGGRAPH 2002), 681-682. https://doi.org/10.1145/566654.566636
/// - F. K. Musgrave, "Fractal Solid Textures", in D. S. Ebert et al.,
///   Texturing and Modeling: A Procedural Approach, 3rd ed., Morgan
///   Kaufmann, 2003 (fractional Brownian motion).
class GradientNoise {
  public:
    /// @brief Shuffles the lattice for a seed.
    /// @param seed The seed.
    explicit GradientNoise(std::uint32_t seed = 0);

    /// @brief Evaluates the noise.
    /// @param p The point.
    /// @return The noise at `p`, within about [-1, 1].
    double operator()(const glm::dvec3& p) const;

    /// @brief Fractional Brownian motion over the noise.
    /// @param p          The point; the first octave is sampled here.
    /// @param octaves    The number of layers.
    /// @param lacunarity Each layer's frequency over the previous.
    /// @param gain       Each layer's amplitude over the previous.
    /// @return The sum normalized by total amplitude, within about
    ///         [-1, 1].
    double Fbm(const glm::dvec3& p, int octaves = 5, double lacunarity = 2.0,
               double gain = 0.5) const;

    /// @return The lattice permutation, 256 entries twice, for
    ///         shaders to evaluate the same noise.
    const std::array<std::uint8_t, 512>& Permutation() const {
        return permutation_;
    }

  private:
    std::array<std::uint8_t, 512> permutation_{};
};

/// @brief Smoothed value noise on the integer lattice, from a sine hash.
/// @param p The point.
/// @return The noise at `p`, in [0, 1].
double ValueNoise(const glm::dvec3& p);

/// @brief Hillaire's tileable Worley noise, repeating over [0, 1] on every axis.
/// @param p     The point.
/// @param cells Cells across each axis.
/// @return The squared distance to the nearest cell point, clamped to 1.
///
/// References:
/// - S. Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996.
/// - S. Hillaire, TileableVolumeNoise, 2017, MIT. https://github.com/sebh/TileableVolumeNoise
double TileableWorley(const glm::dvec3& p, double cells);

/// @brief A fractal sum of inverted tileable Worley noise over four octaves (Hillaire's weights).
/// @param p     The point.
/// @param cells The first octave's cells across each axis, doubling each octave.
/// @return The sum, in [0, 1].
double WorleyFbm(const glm::dvec3& p, double cells);

/// @brief A fractal sum of periodic Perlin noise; each octave doubles the frequency and halves
/// the weight.
/// @param p         The point.
/// @param frequency The first octave's frequency on each axis, which is also its period.
/// @param octaves   The number of octaves.
/// @return The sum over the weights' sum, in about [-1, 1], unclamped.
double TileablePerlin(const glm::dvec3& p, const glm::dvec3& frequency, int octaves);

/// @brief Three decorrelated tileable Perlin sums, a vector field for curl noise.
/// @param p The point.
/// @return Three-octave sums at frequency 12, each at an offset of the point.
glm::dvec3 TileablePerlin3(const glm::dvec3& p);

}  // namespace ernie
