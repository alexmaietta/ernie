// SPDX-License-Identifier: BSD-3-Clause
/// @file noise.h
/// @brief Seeded 3D gradient noise for procedural fields.
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

}  // namespace ernie
