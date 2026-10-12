// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstdint>

namespace ernie {

/// Perlin's improved gradient noise in three dimensions, its lattice
/// shuffled by a seed, for procedural fields such as volumes'
/// density: smooth, zero at every integer point, and within about
/// [-1, 1]. The same seed gives the same noise everywhere, from C++
/// or Python (`ernie.fbm_noise`), so demo twins build the same
/// fields.
///
/// References:
/// - K. Perlin, "Improving Noise", ACM Transactions on Graphics 21(3)
///   (SIGGRAPH 2002), 681-682. https://doi.org/10.1145/566654.566636
/// - F. K. Musgrave, "Fractal Solid Textures", in D. S. Ebert et al.,
///   Texturing and Modeling: A Procedural Approach, 3rd ed., Morgan
///   Kaufmann, 2003 (fractional Brownian motion).
class GradientNoise {
  public:
    /// The noise of `seed`: its permutation of the lattice.
    explicit GradientNoise(std::uint32_t seed = 0);

    /// The noise at `p`.
    double operator()(const glm::dvec3& p) const;

    /// Fractional Brownian motion: `octaves` layers of the noise,
    /// each `lacunarity` times the frequency and `gain` times the
    /// amplitude of the one before, the first at `p` itself; divided
    /// by the sum of the amplitudes, so within about [-1, 1] too.
    double Fbm(const glm::dvec3& p, int octaves = 5,
               double lacunarity = 2.0, double gain = 0.5) const;

    /// Its lattice's permutation, 256 entries twice over, for shaders
    /// to evaluate the same noise on the GPU.
    const std::array<std::uint8_t, 512>& Permutation() const {
        return permutation_;
    }

  private:
    std::array<std::uint8_t, 512> permutation_{};
};

}  // namespace ernie
